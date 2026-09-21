import threading,json
from urllib.parse import quote
from collections import Counter
from . import store
from src.detection.inference_v001 import path_for_host,digest

locks={};guard=threading.Lock()
def lock_for(key):
    with guard:return locks.setdefault(key,threading.Lock())
def url(aid,p):return f'/api/analyses/{aid}/assets/'+quote(str(p).replace('\\','/'),safe='/')

def video(aid):
    p=store.folder(aid)
    if (p/'dashboard_preview_v001.webm').is_file():return p/'dashboard_preview_v001.webm'
    cache=store.CACHE/aid;cache.mkdir(parents=True,exist_ok=True);out=cache/'annotated_risk.webm'
    with lock_for((aid,'video')):
        if out.is_file():return out
        import cv2
        src=p/'annotated_risk.mp4'
        if not src.is_file():raise FileNotFoundError('Processed video not ready')
        before=digest(src);cap=cv2.VideoCapture(str(src));fps=cap.get(5);size=(int(cap.get(3)),int(cap.get(4)))
        temp=cache/'encoding.webm';writer=cv2.VideoWriter(str(temp),cv2.VideoWriter_fourcc(*'VP80'),fps,size)
        if not writer.isOpened():cap.release();raise RuntimeError('Web codec unavailable')
        n=0
        try:
            while True:
                ok,im=cap.read()
                if not ok:break
                writer.write(im);n+=1
        finally:cap.release();writer.release()
        cap=cv2.VideoCapture(str(temp));decoded=0
        while cap.read()[0]:decoded+=1
        cap.release()
        if decoded!=n or n==0 or digest(src)!=before:raise RuntimeError('Presentation transcode integrity failure')
        temp.replace(out);store.atomic(cache/'video.json',dict(source='annotated_risk.mp4',sha256=before,codec='VP8/WebM',frames=n,fps=fps))
        return out

def texts(event,risk):
    kind=event['event_type'];name=event['subject_class'];level=risk['risk_level']
    interpretation=f"El evento {kind} se registró en track #{event['track_id']}, frame {event['peak_frame']}, {event['peak_timestamp']:.3f} s. "+risk['explanation_short']
    finding='una posible pérdida de evidencia de '+name if kind=='PPE_POSSIBLE_MISSING' else 'presencia temporal de '+name if kind=='PPE_PRESENT' else 'un candidato de caída' if kind=='FALL_CANDIDATE' else 'proximidad geométrica a '+name
    diagnosis=f'La evidencia es compatible con {finding}. El intervalo asociado tiene prioridad {level} ({risk["risk_score"]:.1f}/100); requiere revisión y no prueba por sí solo una condición peligrosa. Resultado generado como apoyo a la decisión y sujeto a revisión por personal SSOMA.'
    action='Revisar el segmento y verificar el uso de EPP por personal SSOMA.' if kind=='PPE_POSSIBLE_MISSING' else 'Revisar el segmento y evaluar la interacción observada con maquinaria o vehículo.' if kind in {'MACHINERY_PROXIMITY','VEHICLE_PROXIMITY'} else 'Revisar el segmento de video asociado al evento.'
    return dict(interpretation=interpretation,diagnosis=diagnosis,suggested_action=action,limitations=['Evidencia de desarrollo; riesgo provisional sujeto a juicio SSOMA.','No se afirma causalidad; ausencia de detección no implica incumplimiento.'],norm_reference=None)

def dataset(aid):
    p=store.folder(aid);meta=store.read(p/'metadata.json',{});sm=store.read(p/'summary.json',{});status=store.status(aid)
    if status['status']!='COMPLETED':
        return dict(analysis_id=aid,filename=status['filename'],fps=meta.get('fps',30),frames=status.get('total_frames',0),events=[],risks=[],summary={k:0 for k in ('LOW','MEDIUM','HIGH','CRITICAL')},tracks=0,ppe_counts={},processing_fps=0,status=status)
    events=store.rows(p/'events.jsonl');risks=store.rows(p/'risk_events.jsonl');bundles={b['risk_id']:b for b in store.rows(p/'explanation_bundle.jsonl')}
    riskmap={eid:r for r in risks for eid in r['source_event_ids']};detections=store.rows(p/'detections.jsonl')
    ppe=store.rows(p/'ppe_temporal_evidence.jsonl');ctx=store.rows(p/'context_associations.jsonl');falls=store.rows(p/'fall_candidates_v002.jsonl');poses=store.rows(p/'pose_keypoints.jsonl');assocs=store.rows(p/'ppe_associations.jsonl');tracks=store.rows(p/'tracks.jsonl')
    subset=lambda rows,t,f:[r for r in rows if r.get('track_id')==t and r['frame_id']==f]
    result=[]
    for e in events:
        t,f=e['track_id'],e['peak_frame'];r=riskmap.get(e['event_id'])
        if not r:continue
        b=bundles.get(r['risk_id']);aa=subset(assocs,t,f);cc=subset(ctx,t,f);tr=next(iter(subset(tracks,t,f)),None)
        ids={a['detection_id'] for a in aa if a.get('detection_id') is not None}|{a['source_detection_id'] for a in cc if a.get('source_detection_id') is not None}
        if tr:
            ids.update(i for i,d in enumerate(detections) if d['frame_id']==f and d['canonical_class']=='person' and all(abs(d[k]-v)<.001 for k,v in zip(('x1','y1','x2','y2'),tr['person_bbox'])))
        related=[]
        for i in sorted(ids):
            d=detections[i];a=next((a for a in aa if a.get('detection_id')==i),None);c=next((c for c in cc if c.get('source_detection_id')==i),None)
            related.append(dict(d,detection_id=i,association_state=a['association_state'] if a else c['state'] if c else tr['tracking_state']))
        fuzzy=b['fuzzy'] if b else dict(normalized_inputs=r['inputs'],membership_activations=r['memberships'],rules_fired=r['rules_fired'],rule_definitions=[],dominant_factors=r['dominant_factors'])
        xai=[]
        for x in b['xai_evidence'] if b else []:
            x=dict(x)
            for key in ('heatmap_path','overlay_path','raw_heatmap_path'):
                if x.get(key):x[key]=url(aid,x[key])
            xai.append(x)
        result.append(dict(e,original=f'/api/analyses/{aid}/events/{e["event_id"]}/frame/original',annotated=f'/api/analyses/{aid}/events/{e["event_id"]}/frame/annotated',
            detections=related,ppe=subset(ppe,t,f),context=cc,fall=next(iter(subset(falls,t,f)),None),pose=next(iter(subset(poses,t,f)),None),risk=r,fuzzy=fuzzy,xai=xai,xai_status=b['xai_status'] if b else 'XAI_NOT_APPLICABLE',temporal_evidence=b.get('temporal_evidence') if b else None,**texts(e,r)))
    return dict(analysis_id=aid,filename=status['filename'],fps=meta.get('fps',30),frames=sm.get('frames_processed',status.get('total_frames',0)),events=result,risks=risks,
        summary=store.read(p/'risk_summary.json',{k:0 for k in ('LOW','MEDIUM','HIGH','CRITICAL')}),tracks=len({t['track_id'] for t in tracks}),
        ppe_counts=dict(Counter(d['canonical_class'] for d in detections if d['canonical_class'] in {'helmet','safety_vest','gloves','goggles','safety_boots','safety_harness'})),processing_fps=sm.get('processing_fps',0),status=status)

def detail(aid,eid):
    store.safe_id(eid)
    e=next((e for e in dataset(aid)['events'] if e['event_id']==eid),None)
    if not e:raise FileNotFoundError('Event not found')
    return e

def frame(aid,eid,kind):
    if kind not in {'original','annotated'}:raise ValueError('Invalid frame kind')
    p=store.folder(aid);store.safe_id(eid)
    existing=p/'dashboard_ui_v002'/f'{eid}_{kind}.png'
    if existing.is_file():return existing
    cache=store.CACHE/aid/'frames';cache.mkdir(parents=True,exist_ok=True);out=cache/f'{eid}_{kind}.png'
    with lock_for((aid,eid,kind)):
        if out.is_file():return out
        import cv2
        e=detail(aid,eid);meta=store.read(p/'metadata.json');source=path_for_host(meta['input_path'])
        # Source comes only from trusted local pipeline metadata, never an HTTP path.
        cap=cv2.VideoCapture(str(source));cap.set(cv2.CAP_PROP_POS_FRAMES,e['peak_frame']);ok,image=cap.read();cap.release()
        if not ok:raise RuntimeError('Cannot decode event peak')
        if kind=='annotated':
            from src.detection.tracking_ppe_v001 import draw
            tracks=[r for r in store.rows(p/'tracks.jsonl') if r['track_id']==e['track_id'] and r['frame_id']==e['peak_frame']]
            rows=[r for r in store.rows(p/'ppe_associations.jsonl') if r.get('track_id')==e['track_id'] and r['frame_id']==e['peak_frame']]
            image=draw(image,tracks,rows,e['detections'])
            pose=e['pose']
            if pose and pose.get('keypoints'):
                for a,b in ((5,6),(5,11),(6,12),(11,12)):
                    if min(pose['keypoint_confidences'][a],pose['keypoint_confidences'][b])>=.5:cv2.line(image,tuple(map(int,pose['keypoints'][a])),tuple(map(int,pose['keypoints'][b])),(255,120,60),1)
        if not cv2.imwrite(str(out),image):raise RuntimeError('Cannot cache event frame')
        return out
