"""Selected-frame detector activation views, temporal evidence and fuzzy traces."""
import json
import math
from collections import Counter,defaultdict
from pathlib import Path

from .inference_v001 import OWNERS,SUPPORT,digest,path_for_host,write_json,load_models
from .tracking_ppe_v001 import iou,regions


def eigen_cam(activation):
    import numpy as np
    a=np.asarray(activation,dtype=np.float64)
    if a.ndim!=3 or min(a.shape)<1 or not np.isfinite(a).all():raise ValueError('Invalid/NaN activation')
    matrix=a.reshape(a.shape[0],-1).T
    matrix-=matrix.mean(axis=0)
    _,s,v=np.linalg.svd(matrix,full_matrices=False)
    if not len(s) or s[0]<=1e-12:raise ValueError('Empty activation variance')
    vector=v[0]
    if vector[np.argmax(np.abs(vector))]<0:vector=-vector
    heat=(matrix@vector).reshape(a.shape[1:])
    heat-=heat.min()
    if heat.max()<=1e-12:raise ValueError('Empty heatmap')
    heat/=heat.max()
    validate_heatmap(heat)
    return heat.astype(np.float32)


def validate_heatmap(heat):
    import numpy as np
    a=np.asarray(heat)
    if a.ndim!=2 or not a.size or not np.isfinite(a).all() or a.min()<0 or a.max()>1 or float(a.max()-a.min())<=1e-8:raise ValueError('Invalid/empty heatmap')


def select_xai(risk,context,associations,events):
    ids=set();reasons=[]
    for item in risk['source_evidence']:
        if item['variable'] not in risk['dominant_factors'] or item['value']<=0:continue
        event=events[item['event_id']]
        if event['event_type']=='PPE_POSSIBLE_MISSING':reasons.append('NOT_APPLICABLE_TO_NON_DETECTION')
        elif event['event_type']=='FALL_CANDIDATE':reasons.append('POSE_TEMPORAL_EXPLANATION_NOT_CAM')
        elif item['variable'] in {'hazard_proximity','machinery_proximity'}:
            key=(risk['track_id'],risk['peak_frame'],item['subject_class'])
            row=context.get(key)
            if row and row['source_detection_id'] is not None:ids.add(row['source_detection_id'])
    # PRESENT is not a dominant risk factor; no optional CAMs are fabricated for LOW.
    if not ids and not reasons:reasons.append('NO_RELEVANT_POSITIVE_DETECTION_FOR_DOMINANT_FACTORS')
    return sorted(ids),reasons


def verify_detection(d,entry,frame):
    if d['frame_id']!=frame or OWNERS[d['canonical_class']]!=entry['model_id'] or d['model_id']!=entry['model_id']:raise ValueError('Responsible model/frame mismatch')
    if d['canonical_class'] not in entry['canonical_mapping'].values():raise ValueError('Class mapping mismatch')
    if digest(path_for_host(entry['checkpoint']))!=entry['sha256']:raise ValueError('Checkpoint signature mismatch')


def make_cam(model,entry,frame,d,directory,policy):
    import cv2
    import numpy as np
    capture={}
    def net_hook(module,args):capture['input_shape']=list(args[0].shape[-2:])
    def head_hook(module,args):capture['activation']=args[0][-1].detach().float().cpu().numpy()[0].copy()
    net=model.model
    # Fuse before installing hooks; otherwise the predictor could replace internals.
    net.fuse(verbose=False)
    head=net.model[-1]
    hooks=[net.register_forward_pre_hook(net_hook),head.register_forward_pre_hook(head_hook)]
    try:
        arguments={k:policy[k] for k in ('imgsz','conf','iou','max_det','half','agnostic_nms')}
        result=model.predict(frame,device='0',verbose=False,save=False,**arguments)[0]
    finally:
        for h in hooks:h.remove()
    matches=[]
    target=[d[k] for k in ('x1','y1','x2','y2')]
    for b,confidence,cls in zip(result.boxes.xyxy.cpu().tolist(),result.boxes.conf.cpu().tolist(),result.boxes.cls.cpu().tolist()):
        if entry['canonical_mapping'].get(str(int(cls)))==d['canonical_class']:matches.append((iou(target,b),confidence,b))
    if not matches:raise ValueError('Selected saved detection not reproduced by responsible model')
    best=max(matches,key=lambda x:x[0])
    if best[0]<.9 or abs(best[1]-d['confidence'])>.05:raise ValueError('Detection reproduction mismatch')
    heat=eigen_cam(capture['activation']);ih,iw=capture['input_shape'];h,w=frame.shape[:2]
    scale=min(ih/h,iw/w);rw,rh=round(w*scale),round(h*scale)
    left=round((iw-rw)/2-.1);top=round((ih-rh)/2-.1)
    heat=cv2.resize(heat,(iw,ih))[top:top+rh,left:left+rw]
    heat=cv2.resize(heat,(w,h));validate_heatmap(heat)
    gray=np.round(heat*255).astype(np.uint8)
    colored=cv2.applyColorMap(gray,cv2.COLORMAP_JET)
    overlay=cv2.addWeighted(frame,.6,colored,.4,0)
    x,y,r,b=map(int,target);cv2.rectangle(overlay,(x,y),(r,b),(255,255,255),1)
    heat_path=directory/'heatmap.png';overlay_path=directory/'overlay.png';raw_path=directory/'heatmap.npy'
    directory.mkdir(parents=True,exist_ok=True)
    np.save(raw_path,heat)
    if not cv2.imwrite(str(heat_path),gray) or not cv2.imwrite(str(overlay_path),overlay):raise RuntimeError('CAM image write failed')
    return dict(xai_method='EigenCAM_v001',target_layer=f'model.model[{len(net.model)-1}] pre-Detect input[-1]',
        input_tensor_shape=capture['input_shape'],activation_shape=list(capture['activation'].shape),
        heatmap_path=str(heat_path),overlay_path=str(overlay_path),raw_heatmap_path=str(raw_path),
        reproduction_iou=best[0],reproduction_confidence=best[1],shape=list(heat.shape),
        scope='CLASS_AGNOSTIC_FEATURE_PROJECTION_ASSOCIATED_WITH_DETECTION_NOT_CLASS_SPECIFIC_ATTRIBUTION',
        sign_policy='Largest-absolute PCA loading positive; min-max normalization',
        preprocessing='Ultralytics prediction letterbox; removed padding and resized to original frame')


def process(analysis,root):
    import cv2
    root=Path(root).resolve();analysis=path_for_host(analysis).resolve()
    if root/'results/analysis' not in analysis.parents:raise ValueError('Invalid evidence destination')
    read=lambda name:json.loads((analysis/name).read_text())
    load=lambda name:[json.loads(l) for l in (analysis/name).read_text().splitlines()]
    registry_path=root/'config/model_registry_v001.json';registry=json.loads(registry_path.read_text());entries={m['model_id']:m for m in registry['models']}
    meta=read('metadata.json');fps=meta['fps'];n=read('summary.json')['frames_processed'];source=path_for_host(meta['input_path'])
    if digest(source)!=meta['input_sha256']:raise ValueError('Video signature mismatch')
    names=['risk_events.jsonl','events.jsonl','detections.jsonl','tracks.jsonl','ppe_associations.jsonl','ppe_temporal_evidence.jsonl','context_associations.jsonl','fall_candidates_v002.jsonl','pose_keypoints.jsonl','risk_summary.json','metadata.json']
    paths=[analysis/p for p in names]+[registry_path,root/'config/fuzzy_risk_v001.json',root/'config/tracking_ppe_v001.json']
    before={str(p):digest(p) for p in paths}
    if digest(analysis/'risk_events.jsonl')!=read('risk_summary.json')['outputs']['risk_events.jsonl']:raise ValueError('Risk signature mismatch')
    risks=load('risk_events.jsonl');events={e['event_id']:e for e in load('events.jsonl')};detections=load('detections.jsonl')
    tracks=load('tracks.jsonl');associations=load('ppe_associations.jsonl');ppe=load('ppe_temporal_evidence.jsonl');contexts=load('context_associations.jsonl');falls=load('fall_candidates_v002.jsonl');poses=load('pose_keypoints.jsonl')
    track={(t['track_id'],t['frame_id']):t for t in tracks};context={(t['track_id'],t['frame_id'],t['context_class']):t for t in contexts}
    geometry=json.loads((root/'config/tracking_ppe_v001.json').read_text());config=json.loads((root/'config/fuzzy_risk_v001.json').read_text());rulemap={r['id']:r for r in config['rules']}
    cache={};models={};cap=cv2.VideoCapture(str(source));bundles=[];index=[];forward_count=0
    def frame(f):
        if f not in cache:
            cap.set(cv2.CAP_PROP_POS_FRAMES,f);ok,img=cap.read()
            if not ok:raise ValueError('Evidence frame unavailable')
            cache[f]=img
        return cache[f].copy()
    def relative(path):return str(Path(path).relative_to(analysis)).replace('\\','/')
    try:
        for risk in risks:
            tid,f=risk['track_id'],risk['peak_frame'];event=events[risk['event_id']]
            if risk['analysis_id']!=meta['analysis_id'] or abs(risk['peak_timestamp']-f/fps)>1e-8 or tid!=event['track_id']:raise ValueError('Risk/event/timestamp mismatch')
            directory=analysis/'evidence'/risk['risk_id'];temporal=directory/'temporal';temporal.mkdir(parents=True,exist_ok=True)
            original=frame(f);original_path=directory/'frame.png';cv2.imwrite(str(original_path),original)
            annotated=original.copy();t=track.get((tid,f))
            if t:
                x,y,r,b=map(int,t['person_bbox']);cv2.rectangle(annotated,(x,y),(r,b),(40,230,40),2)
                cv2.putText(annotated,f'Person #{tid}',(max(x,0),max(y,14)),0,.45,(40,230,40),1)
            selected,reasons=select_xai(risk,context,associations,events)
            for item in risk['source_evidence']:
                if item['variable']=='ppe_evidence' and item['source_state']=='POSSIBLE_MISSING' and t:
                    for region in regions(t['person_bbox'],item['subject_class'],geometry):
                        a,b,c,d=map(int,region);cv2.rectangle(annotated,(a,b),(c,d),(0,220,255),1)
                    cv2.putText(annotated,'Possible missing: temporal evidence only',(8,meta['height']-12),0,.45,(0,220,255),1)
            xais=[]
            for did in selected:
                d=detections[did];entry=entries[d['model_id']]
                item=dict(event_id=next(p['event_id'] for p in risk['source_evidence'] if p['subject_class']==d['canonical_class']),
                    frame_id=f,timestamp=f/fps,detection_id=did,model_id=d['model_id'],checkpoint_sha256=entry['sha256'],
                    architecture=entry['architecture'],canonical_class=d['canonical_class'],confidence=d['confidence'],
                    bbox=[d[k] for k in ('x1','y1','x2','y2')],evidence_mode=d['evidence_mode'])
                try:
                    verify_detection(d,entry,f)
                    if entry['model_id'] not in models:models[entry['model_id']]=load_models({'models':[entry]},'0')[0][1]
                    forward_count+=1
                    cam=make_cam(models[entry['model_id']],entry,original,d,directory/'xai'/str(did),registry['inference'])
                    for key in ('heatmap_path','overlay_path','raw_heatmap_path'):cam[key]=relative(cam[key])
                    item.update(cam,status='XAI_AVAILABLE',source_frame_sha256=digest(original_path))
                except Exception as error:item.update(status='XAI_FAILED',error=f'{type(error).__name__}: {error}')
                xais.append(item)
                a,b,c,dv=map(int,item['bbox']);cv2.rectangle(annotated,(a,b),(c,dv),(255,180,70),2)
            annotated_path=directory/'annotated.png';cv2.imwrite(str(annotated_path),annotated)
            frames=sorted(set([max(0,f-round(fps/2)),f,min(n-1,f+round(fps/2))]))
            start=max(0,risk['supporting_frames'][0]-math.ceil(2*fps));end=min(n-1,risk['supporting_frames'][-1]+round(fps/2))
            missing={p['subject_class'] for p in risk['source_evidence'] if p['source_state']=='POSSIBLE_MISSING'}
            for name in missing:
                prior=[r['frame_id'] for r in ppe if r['track_id']==tid and r['ppe_class']==name and r['frame_id']<f and r['state']=='PRESENT']
                if prior:start=min(start,max(prior));frames=sorted(set(frames+[max(prior)]))
            sample=[]
            for sf in frames:
                path=temporal/f'frame_{sf:06d}.png';cv2.imwrite(str(path),frame(sf))
                sample.append(dict(frame_id=sf,timestamp=sf/fps,path=relative(path),sha256=digest(path)))
            subset=lambda rows:[r for r in rows if r.get('track_id')==tid and start<=r['frame_id']<=end]
            history=dict(start_frame=start,end_frame=end,tracks=subset(tracks),ppe_associations=subset(associations),ppe_temporal=subset(ppe),context=subset(contexts),fall_states=subset(falls),pose=subset(poses))
            history_path=temporal/'history.json';write_json(history_path,history)
            # Trace every source event, not just those selected for CAM.
            traces=[]
            for eid in risk['source_event_ids']:
                e=events[eid];fs=set(e['supporting_frames']);ids=set()
                for row in associations:
                    if row.get('track_id')==tid and row['frame_id'] in fs and row['ppe_class']==e['subject_class'] and row.get('detection_id') is not None:ids.add(row['detection_id'])
                for row in contexts:
                    if row['track_id']==tid and row['frame_id'] in fs and row['context_class']==e['subject_class'] and row['source_detection_id'] is not None:ids.add(row['source_detection_id'])
                sources=[]
                for did in sorted(ids):
                    d=detections[did];entry=entries[d['model_id']]
                    if OWNERS[d['canonical_class']]!=d['model_id']:raise ValueError('Trace owner mismatch')
                    sources.append(dict(detection_id=did,**d,checkpoint_sha256=entry['sha256'],architecture=entry['architecture']))
                traces.append(dict(event=e,detections=sources,non_detection=e['event_type']=='PPE_POSSIBLE_MISSING'))
            status='XAI_FAILED' if any(x['status']=='XAI_FAILED' for x in xais) else 'XAI_AVAILABLE' if xais else 'XAI_NOT_APPLICABLE'
            fuzzy=dict(normalized_inputs=risk['inputs'],membership_activations=risk['memberships'],rules_fired=risk['rules_fired'],dominant_factors=risk['dominant_factors'],risk_score=risk['risk_score'],risk_level=risk['risk_level'],raw_mamdani_score=risk['raw_mamdani_score'],temporal_cap_applied=risk['temporal_cap_applied'],rule_definitions=[rulemap[r['rule_id']] for r in risk['rules_fired']],config_sha256=digest(root/'config/fuzzy_risk_v001.json'),EXPERT_VALIDATION_REQUIRED=True)
            bundle=dict(risk_id=risk['risk_id'],event_id=risk['event_id'],track_id=tid,timestamp=f/fps,frame_id=f,risk_score=risk['risk_score'],risk_level=risk['risk_level'],
                visual_evidence=[dict(original_frame=relative(original_path),annotated_frame=relative(annotated_path),original_frame_sha256=digest(original_path),frame_id=f,timestamp=f/fps)],
                xai_status=status,xai_visual_reasons=reasons,xai_evidence=xais,temporal_evidence=dict(samples=sample,history_path=relative(history_path),history_sha256=digest(history_path)),
                fuzzy=fuzzy,trace=traces,explanation_short=risk['explanation_short'],
                fall_explanation='POSE_TEMPORAL_HISTORY' if any(e['event']['event_type']=='FALL_CANDIDATE' for e in traces) else 'NOT_APPLICABLE_NO_FALL_CANDIDATE',
                limitations=['EigenCAM is class-agnostic feature projection, not causal or detection-specific attribution.','Detector activation does not explain Mamdani risk; fuzzy rules are explained separately.','Non-detection is not visual evidence of absence; SUPPORT_ONLY remains unchanged.','Temporal/pose history is observational, not proof of accident.','Fuzzy parameters require expert validation; LOW does not certify safety.'])
            write_json(directory/'explanation.json',bundle);bundles.append(bundle)
            index.append(dict(risk_id=risk['risk_id'],event_id=risk['event_id'],track_id=tid,peak_timestamp=f/fps,thumbnail=relative(original_path),annotated_frame=relative(annotated_path),xai_overlay=next((x['overlay_path'] for x in xais if x['status']=='XAI_AVAILABLE'),None),risk_level=risk['risk_level'],risk_score=risk['risk_score'],short_explanation=risk['explanation_short'],explanation_path=relative(directory/'explanation.json'),xai_status=status))
    finally:cap.release()
    # Verify every responsible model referenced in the bundle once, read-only.
    verified={}
    for entry in registry['models']:
        actual=digest(path_for_host(entry['checkpoint']))
        if actual!=entry['sha256']:raise ValueError('Registry checkpoint changed')
        verified[entry['model_id']]=actual
    if before!={str(p):digest(p) for p in paths} or digest(source)!=meta['input_sha256']:raise ValueError('Inputs changed')
    (analysis/'explanation_bundle.jsonl').write_text(''.join(json.dumps(b,ensure_ascii=False,allow_nan=False)+'\n' for b in bundles),encoding='utf-8')
    write_json(analysis/'dashboard_evidence_index.json',dict(analysis_id=meta['analysis_id'],path_base='analysis_directory',events=index))
    stats=Counter(b['xai_status'] for b in bundles)
    summary=dict(status='XAI_EVIDENCE_READY' if not stats['XAI_FAILED'] else 'ACTION_REQUIRED',risk_events=len(bundles),
        **{k:stats[k] for k in ('XAI_AVAILABLE','XAI_NOT_APPLICABLE','XAI_FAILED')},
        cam_counts=dict(Counter(x['model_id'] for b in bundles for x in b['xai_evidence'] if x['status']=='XAI_AVAILABLE')),
        selected_detector_forwards=forward_count,selected_frames_decoded=len(cache),input_signatures=before,
        verified_checkpoint_signatures=verified,source_video_sha256=meta['input_sha256'],
        outputs={p:digest(analysis/p) for p in ('explanation_bundle.jsonl','dashboard_evidence_index.json')})
    write_json(analysis/'xai_evidence_summary.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('input_signatures','outputs','verified_checkpoint_signatures')}))
    return summary
