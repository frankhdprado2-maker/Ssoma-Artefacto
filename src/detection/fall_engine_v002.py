"""Pose-backed temporal fall candidates, without confirmation or compliance."""
import json
import math
import os
from collections import Counter,defaultdict
from pathlib import Path

from .inference_v001 import digest,path_for_host,write_json
from .tracking_ppe_v001 import iou
from .event_engine_v001 import group_events


def load_pose(weight,metadata):
    from ultralytics import YOLO
    if digest(weight)!=metadata['sha256']:raise ValueError('Pose weight signature mismatch')
    model=YOLO(str(weight),task='pose')
    if model.task!='pose' or list(model.model.kpt_shape)!=[17,3]:raise ValueError('Not COCO17 pose')
    return model


def associate(tracks,poses,config):
    """Global 1:1 IoU assignment; reject ambiguous alternatives on either axis."""
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    observed=[t for t in tracks if t['tracking_state']=='OBSERVED']
    result={t['track_id']:(None,'UNMATCHED' if t['tracking_state']=='OBSERVED' else 'UNOBSERVED') for t in tracks}
    if not observed or not poses:return result
    matrix=np.array([[iou(t['person_bbox'],p['pose_bbox']) for p in poses] for t in observed])
    for t in observed:result[t['track_id']]=(None,'UNMATCHED')
    cost=np.where(matrix>=config['association_iou'],1-matrix,1e6)
    rr,cc=linear_sum_assignment(cost)
    for r,c in zip(rr,cc):
        value=float(matrix[r,c]);tid=observed[r]['track_id']
        if value<config['association_iou']:continue
        rivals=[matrix[r,j] for j in range(len(poses)) if j!=c]+[matrix[i,c] for i in range(len(observed)) if i!=r]
        if any(v>=config['association_iou'] and value-v<=config['ambiguity_margin'] for v in rivals):
            result[tid]=(None,'AMBIGUOUS');continue
        result[tid]=(dict(poses[c],association_iou=value,pose_index=int(c)),'MATCHED')
    return result


def pose_features(row,config):
    points=row['keypoints'];conf=row['keypoint_confidences']
    if points is None or conf is None:return None
    if len(points)!=17 or len(conf)!=17:return None
    valid=[i for i,(p,c) in enumerate(zip(points,conf)) if math.isfinite(c) and c>=config['keypoint_conf'] and len(p)==2 and all(math.isfinite(v) for v in p)]
    if len(valid)<config['minimum_keypoints'] or not {5,6,11,12}.issubset(valid):return None
    mid=lambda a,b:[(points[a][j]+points[b][j])/2 for j in (0,1)]
    shoulder,hip=mid(5,6),mid(11,12)
    vector=[hip[j]-shoulder[j] for j in (0,1)]
    if math.hypot(*vector)<5:return None
    line=lambda a,b:math.degrees(math.atan2(points[b][1]-points[a][1],points[b][0]-points[a][0]))%180
    xs=[points[i][0] for i in valid];ys=[points[i][1] for i in valid]
    return dict(shoulder_orientation_degrees=line(5,6),hip_orientation_degrees=line(11,12),
        torso_angle_from_vertical=math.degrees(math.atan2(abs(vector[0]),abs(vector[1]))),
        shoulder_hip_vector=vector,body_horizontal_vertical_ratio=(max(xs)-min(xs))/max(max(ys)-min(ys),1),
        center_of_mass_proxy=[(shoulder[j]+hip[j])/2 for j in (0,1)],
        valid_keypoints=len(valid),pose_quality=sum(conf[i] for i in (5,6,11,12))/4)


class FallEngine:
    def __init__(self,config,fps):self.c,self.fps,self.history=config,fps,{}

    def update(self,row):
        tid,f=row['track_id'],row['frame_id'];c=self.c
        feature=pose_features(row,c) if row['association_state']=='MATCHED' else None
        s=self.history.setdefault(tid,dict(last=-2,upright=0,horizontal=0,anchor=None,previous=None))
        if f!=s['last']+1 or feature is None:s.update(upright=0,horizontal=0,anchor=None,previous=None)
        s['last']=f
        state='UNKNOWN';descent=velocity=reduction=change=None
        scores=dict(pose_score=0.,motion_score=0.,geometry_score=0.,persistence_score=0.,combined_fall_score=0.)
        conditions=dict(A_motion=False,B_posture_change=False,C_horizontal=False,D_persistence=False)
        if feature:
            b=row['person_bbox'];height=max(b[3]-b[1],1)
            y=feature['center_of_mass_proxy'][1];angle=feature['torso_angle_from_vertical']
            if s['previous'] is not None:velocity=(y-s['previous']['y'])*self.fps/s['previous']['height']
            s['previous']=dict(y=y,height=height)
            if angle<=c['upright_angle_max']:
                s['upright']+=1;s['horizontal']=0
                if s['upright']>=math.ceil(c['upright_seconds']*self.fps):
                    state='UPRIGHT';s['anchor']=dict(frame=f,y=y,height=height,angle=angle)
            else:
                s['upright']=0
                horizontal=angle>=c['horizontal_angle_min'] and feature['body_horizontal_vertical_ratio']>=c['horizontal_body_ratio']
                s['horizontal']=s['horizontal']+1 if horizontal else 0
                persistence=min(1.,s['horizontal']/math.ceil(c['persistence_seconds']*self.fps))
                state='HORIZONTAL' if horizontal and persistence>=1 else 'TRANSITION'
                anchor=s['anchor']
                if anchor and f-anchor['frame']>c['transition_seconds']*self.fps:s['anchor']=None;anchor=None
                if anchor:
                    descent=(y-anchor['y'])/anchor['height'];reduction=1-height/anchor['height'];change=angle-anchor['angle']
                    # Persistent displacement is required; instantaneous speed alone cannot confirm motion.
                    motion=descent>=c['descent_heights'] or (reduction>=c['height_reduction_min'] and descent>=c['descent_heights']/2)
                    conditions=dict(A_motion=motion,B_posture_change=change>=c['posture_change_min'],C_horizontal=horizontal,D_persistence=persistence>=1)
                    scores['motion_score']=min(1.,max(0.,descent)/c['descent_heights'])
                    scores['geometry_score']=min(1.,max(0.,reduction)/c['height_reduction_min'])
                    if all(conditions.values()):state='POSSIBLE_FALL'
                scores['persistence_score']=persistence
            scores['pose_score']=feature['pose_quality']*min(1.,angle/90)
            scores['combined_fall_score']=.4*scores['pose_score']+.25*scores['motion_score']+.1*scores['geometry_score']+.25*scores['persistence_score']
        return dict(analysis_id=row['analysis_id'],track_id=tid,frame_id=f,timestamp_seconds=row['timestamp'],
            state=state,event_type='FALL_CANDIDATE' if state=='POSSIBLE_FALL' else None,
            features=feature,vertical_displacement_heights=descent,vertical_velocity_heights_per_second=velocity,
            bbox_height_reduction=reduction,torso_angle_change=change,horizontal_frames=s['horizontal'],
            conditions=conditions,**scores,evidence_score=scores['combined_fall_score'],source_module='FallEngine_v002_pose')


def run(analysis,root,preview_path=None):
    import cv2
    import torch
    import ultralytics
    analysis=path_for_host(analysis).resolve();root=Path(root).resolve()
    if root/'results/analysis' not in analysis.parents:raise ValueError('Invalid output directory')
    config_path=root/'config/fall_engine_v002.json';c=json.loads(config_path.read_text())
    weight=root/'models/pose_v002/yolo11n-pose.pt';metadata_path=weight.parent/'source_metadata.json'
    metadata=json.loads(metadata_path.read_text())
    meta=json.loads((analysis/'metadata.json').read_text());fps=meta['fps'];aid=meta['analysis_id']
    n=json.loads((analysis/'summary.json').read_text())['frames_processed']
    load=lambda p:[json.loads(line) for line in p.read_text().splitlines()]
    tracks=load(analysis/'tracks.jsonl');byframe=defaultdict(list)
    for t in tracks:
        if t['analysis_id']!=aid or abs(t['timestamp_seconds']-t['frame_id']/fps)>1e-8:raise ValueError('Track timeline mismatch')
        byframe[t['frame_id']].append(t)
    immutable=[analysis/p for p in ('tracks.jsonl','detections.jsonl','ppe_associations.jsonl','metadata.json','event_summary.json','annotated_events.mp4')]+[weight,metadata_path,config_path]
    before={str(p):digest(p) for p in immutable}
    old_events=(analysis/'events.jsonl').read_bytes()
    source=path_for_host(meta['input_path'])
    if digest(source)!=meta['input_sha256']:raise ValueError('Source video changed')
    cache=analysis/'pose_keypoints.jsonl';cache_meta=analysis/'pose_keypoints_metadata.json'
    identity=dict(tracks_sha256=digest(analysis/'tracks.jsonl'),source_sha256=meta['input_sha256'],weight_sha256=metadata['sha256'],config_sha256=digest(config_path))
    inference_frames=0
    if cache.exists():
        cm=json.loads(cache_meta.read_text())
        if cm['identity']!=identity or cm['sha256']!=digest(cache):raise ValueError('Existing pose cache mismatch')
        rows=load(cache);inference_frames=cm['inference_frames']
    else:
        model=load_pose(weight,metadata)
        cap=cv2.VideoCapture(str(source));rows=[]
        try:
            for f in range(n):
                ok,image=cap.read()
                if not ok:raise RuntimeError('Source decode failure')
                ts=byframe[f];poses=[]
                if any(t['tracking_state']=='OBSERVED' for t in ts):
                    r=model.predict(image,imgsz=c['imgsz'],conf=c['pose_conf'],iou=c['nms_iou'],device=0,verbose=False,save=False)[0]
                    inference_frames+=1
                    if r.keypoints is not None:
                        for b,k in zip(r.boxes.xyxy.cpu().tolist(),r.keypoints.data.cpu().tolist()):
                            poses.append(dict(pose_bbox=b,keypoints=[p[:2] for p in k],keypoint_confidences=[p[2] for p in k]))
                matches=associate(ts,poses,c)
                for t in ts:
                    pose,state=matches[t['track_id']]
                    row=dict(analysis_id=aid,frame_id=f,timestamp=f/fps,timestamp_seconds=f/fps,track_id=t['track_id'],
                        person_bbox=t['person_bbox'],association_state=state,keypoints=None,keypoint_confidences=None,pose_bbox=None,pose_quality=0.)
                    if pose:
                        row.update(pose);features=pose_features(row,c)
                        row['pose_quality']=features['pose_quality'] if features else 0.
                    rows.append(row)
        finally:cap.release()
        temporary=cache.with_suffix('.tmp')
        temporary.write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in rows));temporary.replace(cache)
        write_json(cache_meta,dict(identity=identity,sha256=digest(cache),inference_frames=inference_frames,ultralytics=ultralytics.__version__,torch=torch.__version__,device=torch.cuda.get_device_name(0)))
    if len(rows)!=len(tracks):raise ValueError('Pose/track row loss')
    engine=FallEngine(c,fps);fall=[engine.update(r) for r in rows]
    signals=[dict(r,subject_class='person') for r in fall if r['event_type']=='FALL_CANDIDATE']
    events=group_events(signals,aid,fps,2)
    evidence=analysis/'fall_evidence_v002';evidence.mkdir(exist_ok=True)
    cap=cv2.VideoCapture(str(source))
    lookup={(r['track_id'],r['frame_id']):r for r in rows}
    def render(row):
        cap.set(cv2.CAP_PROP_POS_FRAMES,row['frame_id']);ok,frame=cap.read()
        if not ok:raise RuntimeError('Evidence decode failure')
        if row['keypoints']:
            for p,confidence in zip(row['keypoints'],row['keypoint_confidences']):
                if confidence>=c['keypoint_conf']:cv2.circle(frame,tuple(map(lambda x:int(round(x)),p)),3,(0,255,0),-1)
            for a,b in ((5,6),(11,12),(5,11),(6,12)):
                if min(row['keypoint_confidences'][a],row['keypoint_confidences'][b])>=c['keypoint_conf']:
                    cv2.line(frame,tuple(map(int,row['keypoints'][a])),tuple(map(int,row['keypoints'][b])),(0,220,255),2)
        cv2.putText(frame,f"POSE v002 track #{row['track_id']} frame {row['frame_id']}",(8,20),0,.5,(255,255,255),1)
        return frame
    try:
        for i,e in enumerate(events):
            e['event_id']=f'{aid}_fall_v002_{i+1:05d}'
            e['evidence_frame']=f"fall_evidence_v002/{e['event_id']}.jpg"
            if not cv2.imwrite(str(analysis/e['evidence_frame']),render(lookup[e['track_id'],e['peak_frame']])):raise RuntimeError('Evidence write failure')
        if rows:
            best=max(rows,key=lambda r:r['pose_quality'])
            cv2.imwrite(str(preview_path or root/'reports/integration/fall_pose_v002_preview.jpg'),render(best))
    finally:cap.release()
    nonfall=[line for line in old_events.splitlines(keepends=True) if json.loads(line)['event_type']!='FALL_CANDIDATE']
    updated=b''.join(nonfall)+''.join(json.dumps(e,allow_nan=False)+'\n' for e in events).encode()
    snapshot=analysis/'events.pre_fall_v002.jsonl'
    if snapshot.exists() and snapshot.read_bytes()!=old_events:raise ValueError('Already migrated; refusing overwrite')
    if before!={str(p):digest(p) for p in immutable} or digest(source)!=meta['input_sha256']:raise ValueError('Immutable input changed')
    if (analysis/'events.jsonl').read_bytes()!=old_events:raise ValueError('Events changed concurrently')
    if not snapshot.exists():snapshot.write_bytes(old_events)
    (analysis/'fall_candidates_v002.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in fall))
    (analysis/'events.v002.jsonl').write_bytes(updated)
    temp=analysis/'events.jsonl.tmp';temp.write_bytes(updated);temp.replace(analysis/'events.jsonl')
    summary=dict(status='FALL_ENGINE_V002_READY',pose_model=metadata,config=c,frames_in_smoke=n,
        pose_frames_processed=inference_frames,pose_track_rows=len(rows),matched_rows=sum(r['association_state']=='MATCHED' for r in rows),
        quality_rows=sum(r['pose_quality']>0 for r in rows),association_states=dict(Counter(r['association_state'] for r in rows)),
        states=dict(Counter(r['state'] for r in fall)),fall_candidates=len(events),nonfall_events_preserved=len(nonfall),
        total_events=len(nonfall)+len(events),input_signatures=before,events_before_sha256=digest(snapshot),
        events_after_sha256=digest(analysis/'events.jsonl'),unchanged_nonfall_bytes=True,
        runtime=json.loads(cache_meta.read_text()),limitations=['2D pose and center proxy, not biomechanical center of mass.',
        'Camera motion, occlusion, ID switches and deliberate lying down can confound the heuristic.',
        'No accuracy claim without formal ground truth. v001 annotated_events.mp4 remains historical.'],
        outputs={name:digest(analysis/name) for name in ('pose_keypoints.jsonl','fall_candidates_v002.jsonl','events.v002.jsonl')})
    write_json(analysis/'fall_engine_v002_summary.json',summary)
    print(json.dumps({k:summary[k] for k in ('status','pose_frames_processed','matched_rows','quality_rows','fall_candidates','total_events')}))
    return summary
