"""Temporal postprocessing only. Scores are heuristic evidence, not probability."""
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from .inference_v001 import SUPPORT, digest, path_for_host, write_json
from .tracking_ppe_v001 import PPE, box, center, regions

CONTEXT = ('smoke', 'fire', 'vehicle', 'machinery')


class Features:
    def __init__(self):
        self.history = {}

    def update(self, t, fps):
        tid,f = t['track_id'],t['frame_id']
        b = t['person_bbox']
        w,h = b[2]-b[0],b[3]-b[1]
        if w<=0 or h<=0: raise ValueError('Degenerate track bbox')
        current = center(b)
        s = self.history.setdefault(tid,dict(first=f,observed=0,last=None,last_observation=None,max_gap=0))
        observed = t['tracking_state']=='OBSERVED'
        previous = s['last_observation']
        gap = f-previous['frame_id']-1 if previous else 0
        contiguous = observed and previous is not None and gap==0
        velocity = [(a-c)*fps for a,c in zip(current,previous['center'])] if contiguous else None
        s['observed'] += int(observed)
        s['max_gap'] = max(s['max_gap'],gap if observed else gap+1)
        result = dict(analysis_id=t['analysis_id'],track_id=tid,frame_id=f,timestamp_seconds=f/fps,
            center=list(current),bbox_width=w,bbox_height=h,aspect_ratio=w/h,
            velocity_pixels_per_second=velocity,speed_pixels_per_second=math.hypot(*velocity) if velocity else None,
            normalized_speed_per_second=math.hypot(*velocity)/h if velocity else None,
            duration_seconds=(f-s['first']+1)/fps,observed_frames=s['observed'],
            visibility_fraction=s['observed']/(f-s['first']+1),observed=observed,
            gap_frames=gap if observed else gap+1,max_gap_frames=s['max_gap'],
            motion_valid=contiguous,torso_orientation_degrees=None,
            orientation_source='BBOX_PROXY_ONLY_NO_POSE')
        if observed: s['last_observation']=result
        s['last']=f
        return result


class PPEEvidence:
    def __init__(self, config, geometry, width, height, fps):
        self.config,self.geometry,self.width,self.height,self.fps=config,geometry,width,height,fps
        self.history={}

    def update(self, t, name, rows):
        f=t['frame_id']
        key=(t['track_id'],name)
        s=self.history.setdefault(key,dict(last=-2,missing=0,ever_present=False))
        if f!=s['last']+1:
            s['missing']=0
            s['ever_present']=False
        s['last']=f
        b=t['person_bbox']
        visible=t['tracking_state']=='OBSERVED' and (t['person_confidence'] or 0)>=self.config['minimum_confidence'] and b[3]-b[1]>=self.config['minimum_person_height']
        visible=visible and all(0<=a<c<=self.width and 0<=y<z<=self.height for a,y,c,z in regions(b,name,self.geometry))
        raw={r['association_state'] for r in rows}
        current=any(r.get('ppe_bbox') is not None for r in rows)
        state='UNKNOWN'
        reason='INSUFFICIENT_EVIDENCE'
        if 'AMBIGUOUS' in raw:
            state,reason='AMBIGUOUS','COMPETING_PERSON_OWNERS'
            s['missing']=0
        elif 'PRESENT' in raw:
            state,reason='PRESENT','REPEATED_ASSOCIATION'
            s['ever_present']=True
            s['missing']=0
        elif name not in SUPPORT and visible and s['ever_present'] and not current:
            s['missing']+=1
            if s['missing']/self.fps>=self.config['missing_seconds']:
                state,reason='POSSIBLE_MISSING','PRIOR_PRESENT_THEN_PERSISTENT_GEOMETRICALLY_OBSERVABLE_LOSS'
        else: s['missing']=0
        scores=[r['association_score'] for r in rows if r.get('association_score') is not None]
        value=max(scores,default=0.) if state=='PRESENT' else min(1.,s['missing']/max(self.fps*self.config['missing_seconds'],1)) if state=='POSSIBLE_MISSING' else 0.
        return dict(analysis_id=t['analysis_id'],track_id=t['track_id'],frame_id=f,timestamp_seconds=f/self.fps,
            ppe_class=name,state=state,evidence_mode='SUPPORT_ONLY' if name in SUPPORT else 'DETECTION_ONLY',
            evidence_score=value,reason=reason,geometric_visibility_proxy=bool(visible),
            missing_streak_frames=s['missing'],prior_present=s['ever_present'],
            source_detection_ids=[r['detection_id'] for r in rows if r.get('detection_id') is not None])


def rectangle_distance(a,b):
    return math.hypot(max(a[0]-b[2],b[0]-a[2],0),max(a[1]-b[3],b[1]-a[3],0))


class ContextEvidence:
    def __init__(self,config): self.config,self.history=config,{}

    def update(self,t,name,detections):
        b=t['person_bbox']; f=t['frame_id']; key=(t['track_id'],name)
        candidates=[d for d in detections if d['canonical_class']==name]
        raw='UNKNOWN'; distance=None; selected=None
        if t['tracking_state']=='OBSERVED' and candidates:
            selected=min(candidates,key=lambda d:rectangle_distance(b,box(d)))
            distance=rectangle_distance(b,box(selected))/max(b[3]-b[1],1)
            raw='NEAR' if distance<=self.config['near_distance_person_heights'] else 'FAR'
        old=self.history.get(key,(-2,None,0))
        streak=old[2]+1 if old[0]==f-1 and old[1]==raw else 1
        self.history[key]=(f,raw,streak)
        state=raw if raw!='UNKNOWN' and streak>=self.config['minimum_frames'] else 'UNKNOWN'
        return dict(analysis_id=t['analysis_id'],track_id=t['track_id'],frame_id=f,timestamp_seconds=t['timestamp_seconds'],
            context_class=name,state=state,raw_state=raw,streak_frames=streak,normalized_distance=distance,
            context_bbox=box(selected) if selected else None,
            source_detection_id=selected['detection_id'] if selected else None,
            evidence_score=(1-distance/(self.config['near_distance_person_heights']+1))*selected['confidence'] if state=='NEAR' else 0.,
            interpretation='IMAGE_PLANE_GEOMETRY_NOT_RISK')


class FallEngine:
    def __init__(self,config,fps): self.config,self.fps,self.history=config,fps,{}

    def update(self,feature):
        f,tid=feature['frame_id'],feature['track_id']; c=self.config
        s=self.history.setdefault(tid,dict(last=-2,upright=0,horizontal=0,anchor=None,transition=None))
        if f!=s['last']+1 or not feature['observed']:
            s.update(upright=0,horizontal=0,anchor=None,transition=None)
        s['last']=f
        state='UNKNOWN'; descent=None
        if feature['observed']:
            aspect=feature['aspect_ratio']
            if aspect<=c['upright_aspect_max']:
                s['upright']+=1; s['horizontal']=0; s['transition']=None
                if s['upright']/self.fps>=c['upright_seconds']:
                    state='UPRIGHT'
                    s['anchor']=(f,feature['center'][1],feature['bbox_height'])
            else:
                s['upright']=0
                anchor=s['anchor']
                if anchor:
                    descent=(feature['center'][1]-anchor[1])/anchor[2]
                    if f-anchor[0]<=self.fps*c['transition_seconds'] and descent>=c['minimum_descent_heights']:
                        if s['transition'] is None:s['transition']=f
                horizontal=aspect>=c['horizontal_aspect_min']
                s['horizontal']=s['horizontal']+1 if horizontal else 0
                state='HORIZONTAL' if horizontal and s['horizontal']/self.fps>=c['horizontal_seconds'] else 'TRANSITION' if anchor or horizontal else 'UNKNOWN'
                if s['transition'] is not None and f-s['transition']<=self.fps*c['transition_seconds'] and s['horizontal']/self.fps>=c['horizontal_seconds']:
                    state='POSSIBLE_FALL'
                elif s['transition'] is not None and f-s['transition']>self.fps*c['transition_seconds']:
                    s['transition']=None; s['anchor']=None
        return dict(analysis_id=feature['analysis_id'],track_id=tid,frame_id=f,timestamp_seconds=feature['timestamp_seconds'],
            state=state,event_type='FALL_CANDIDATE' if state=='POSSIBLE_FALL' else None,
            aspect_ratio=feature['aspect_ratio'],descent_person_heights=descent,
            horizontal_frames=s['horizontal'],torso_orientation_degrees=None,pose_available=False,
            evidence_score=min(1.,s['horizontal']/max(self.fps*c['horizontal_seconds'],1))*.6 if state=='POSSIBLE_FALL' else 0.,
            method='GEOMETRIC_FALL_PROXY',limitation='No torso pose; camera motion, bending and ID switches may mimic a fall')


def group_events(signals,analysis_id,fps,minimum_frames):
    groups=defaultdict(list)
    for s in signals: groups[s['track_id'],s['event_type'],s['subject_class']].append(s)
    events=[]
    def finish(run,key):
        if len(run)<minimum_frames:return
        peak=max(run,key=lambda s:s['evidence_score'])
        events.append(dict(analysis_id=analysis_id,track_id=key[0],event_type=key[1],subject_class=key[2],
            start_timestamp=run[0]['frame_id']/fps,end_timestamp=(run[-1]['frame_id']+1)/fps,
            peak_timestamp=peak['frame_id']/fps,duration=len(run)/fps,
            evidence_score=peak['evidence_score'],supporting_frames=[r['frame_id'] for r in run],
            source_modules=sorted({r['source_module'] for r in run}),peak_frame=peak['frame_id'],
            evidence_mode='SUPPORT_ONLY' if key[2] in SUPPORT else 'DEVELOPMENT_HEURISTIC'))
    for key,items in sorted(groups.items()):
        run=[]
        for s in sorted(items,key=lambda s:s['frame_id']):
            if run and s['frame_id']==run[-1]['frame_id']:raise ValueError('Duplicate event signal')
            if run and s['frame_id']!=run[-1]['frame_id']+1:finish(run,key);run=[]
            run.append(s)
        finish(run,key)
    events.sort(key=lambda e:(e['start_timestamp'],e['track_id'],e['event_type'],e['subject_class']))
    for i,e in enumerate(events):e['event_id']=f'{analysis_id}_event_{i+1:05d}'
    return events


def process(analysis,config_path):
    import cv2
    analysis=path_for_host(analysis).resolve(); config_path=path_for_host(config_path).resolve()
    root=Path(__file__).resolve().parents[2]
    if root/'results'/'analysis' not in analysis.parents:raise ValueError('Output must be an analysis directory')
    config=json.loads(config_path.read_text())
    geometry_path=root/'config/tracking_ppe_v001.json'
    geometry=json.loads(geometry_path.read_text())
    sources=[analysis/p for p in ('detections.jsonl','tracks.jsonl','ppe_associations.jsonl','metadata.json','summary.json','association_report.json','annotated_tracking.mp4')]+[config_path,geometry_path,Path(__file__)]
    signatures={str(p):digest(p) for p in sources}
    meta=json.loads((analysis/'metadata.json').read_text()); previous=json.loads((analysis/'association_report.json').read_text())
    if previous['status']!='TRACKING_ASSOCIATION_READY':raise ValueError('Tracking not ready')
    for name in ('tracks.jsonl','ppe_associations.jsonl','annotated_tracking.mp4'):
        if digest(analysis/name)!=previous['output_signatures'][name]:raise ValueError('Tracking output signature mismatch')
    fps=meta['fps']; n=previous['frames_processed']; aid=meta['analysis_id']
    def load(name):
        result=[]
        for line in (analysis/name).read_text().splitlines():
            d=json.loads(line); f=d['frame_id']
            if d['analysis_id']!=aid or not isinstance(f,int) or not 0<=f<n or abs(d['timestamp_seconds']-f/fps)>1e-8:raise ValueError('Invalid input timeline')
            result.append(d)
        return result
    tracks=load('tracks.jsonl'); detections=load('detections.jsonl'); associations=load('ppe_associations.jsonl')
    if len({(t['frame_id'],t['track_id']) for t in tracks})!=len(tracks):raise ValueError('Duplicate track frame')
    by_frame=defaultdict(list); by_key=defaultdict(list)
    for i,d in enumerate(detections):d['detection_id']=i;by_frame[d['frame_id']].append(d)
    for a in associations:by_key[a['frame_id'],a['track_id'],a['ppe_class']].append(a)
    features=Features(); ppe=PPEEvidence(config['ppe'],geometry,meta['width'],meta['height'],fps)
    context=ContextEvidence(config['context']); fall=FallEngine(config['fall'],fps)
    outputs={p:[] for p in ('track_temporal_features','ppe_temporal_evidence','context_associations','fall_candidates')}
    signals=[]
    def signal(row,event,name,module):
        signals.append(dict(track_id=row['track_id'],frame_id=row['frame_id'],event_type=event,
            subject_class=name,evidence_score=row['evidence_score'],source_module=module))
    for t in sorted(tracks,key=lambda t:(t['frame_id'],t['track_id'])):
        feature=features.update(t,fps);outputs['track_temporal_features'].append(feature)
        for name in PPE:
            row=ppe.update(t,name,by_key[t['frame_id'],t['track_id'],name]);outputs['ppe_temporal_evidence'].append(row)
            if row['state'] in {'PRESENT','POSSIBLE_MISSING'}:signal(row,'PPE_'+row['state'],name,'ppe_temporal')
        for name in CONTEXT:
            row=context.update(t,name,by_frame[t['frame_id']]);outputs['context_associations'].append(row)
            if row['state']=='NEAR':signal(row,{'smoke':'HAZARD_PROXIMITY','fire':'HAZARD_PROXIMITY','vehicle':'VEHICLE_PROXIMITY','machinery':'MACHINERY_PROXIMITY'}[name],name,'context_geometry')
        row=fall.update(feature);outputs['fall_candidates'].append(row)
        if row['state']=='POSSIBLE_FALL':signal(row,'FALL_CANDIDATE','person','fall_geometry')
    events=group_events(signals,aid,fps,config['events']['minimum_frames'])
    evidence=analysis/'event_evidence';evidence.mkdir(exist_ok=True)
    peaks=defaultdict(list); active=defaultdict(list)
    for e in events:
        e['evidence_frame']=f"event_evidence/{e['event_id']}_peak.jpg"
        peaks[e['peak_frame']].append(e)
        for f in e['supporting_frames']:active[f].append(e)
    cap=cv2.VideoCapture(str(analysis/'annotated_tracking.mp4'))
    # A separate panel preserves all existing person/PPE/context annotations.
    panel_width=520
    panel_height=max(meta['height'],30+18*max((len(v) for v in active.values()),default=0))
    panel_height+=panel_height%2
    video=analysis/'annotated_events.mp4'
    writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),fps,(meta['width']+panel_width,panel_height))
    if not cap.isOpened() or not writer.isOpened():raise RuntimeError('Video unavailable')
    import numpy as np
    try:
        for f in range(n):
            ok,frame=cap.read()
            if not ok:raise RuntimeError('Incomplete tracking video')
            canvas=np.zeros((panel_height,meta['width']+panel_width,3),dtype=np.uint8)
            canvas[:meta['height'],:meta['width']]=frame
            cv2.putText(canvas,f'EVENT_ENGINE v001 | t={f/fps:.2f}s',(meta['width']+8,18),0,.45,(255,255,255),1)
            for j,e in enumerate(active[f]):
                label=f"#{e['track_id']} {e['event_type']} {e['subject_class']}"
                cv2.putText(canvas,label,(meta['width']+8,38+j*18),0,.38,(0,210,255),1)
            writer.write(canvas)
            for e in peaks[f]:
                if not cv2.imwrite(str(analysis/e['evidence_frame']),canvas):raise RuntimeError('Evidence frame save failed')
    finally:cap.release();writer.release()
    cap=cv2.VideoCapture(str(video)); decoded=0
    while cap.read()[0]:decoded+=1
    video_fps=cap.get(cv2.CAP_PROP_FPS);cap.release()
    if decoded!=n or abs(video_fps-fps)>.01:raise ValueError('Event video timeline mismatch')
    outputs['events']=events
    for name,rows in outputs.items():
        with (analysis/(name+'.jsonl')).open('w') as stream:
            for row in rows:stream.write(json.dumps(row,allow_nan=False)+'\n')
    if signatures!={str(p):digest(p) for p in sources}:raise ValueError('Input changed')
    summary=dict(analysis_id=aid,status='EVENT_ENGINE_READY',config=config,frames_processed=n,
        total_events=len(events),events_by_type=dict(Counter(e['event_type'] for e in events)),
        fall_engine='GEOMETRIC_PROXY_NO_POSE',pose_model=None,
        fall_candidates=sum(e['event_type']=='FALL_CANDIDATE' for e in events),
        state_rows={k:dict(Counter(r.get('state','FEATURE') for r in v)) for k,v in outputs.items() if k!='events'},
        input_signatures=signatures,inputs_unchanged=True,video_decoded_frames=decoded,
        detector_runs=0,tracking_runs=0,association_reruns=0,training_runs=0,test_accessed=False,
        event_interval_policy='Half-open [start_timestamp,end_timestamp), nominal fps; duration = supporting frame count / fps.',
        fall_file_policy='All per-track state rows, including UNKNOWN; candidates are rows with event_type FALL_CANDIDATE; event count is grouped.',
        limitations=['No pose checkpoint found in project or runtime search; torso orientation is null.',
            'Fall engine uses bounding-box proxies, not torso measurements. No ground truth accuracy.',
            'PPE visibility is geometric, not proof of occlusion-free visibility. Possible missing is a review hint only.',
            'Context is 2D class-level proximity, not physical distance, object identity or risk.',
            'Inherited track duplicates, fragmentation, cuts and camera motion may produce false events.',
            'Scores are heuristic evidence, not calibrated probabilities. No confirmed accident or compliance decision.'])
    summary['output_signatures']={name+'.jsonl':digest(analysis/(name+'.jsonl')) for name in outputs}
    summary['output_signatures']['annotated_events.mp4']=digest(video)
    summary['evidence_signatures']={e['evidence_frame']:digest(analysis/e['evidence_frame']) for e in events}
    write_json(analysis/'event_summary.json',summary)
    return summary
