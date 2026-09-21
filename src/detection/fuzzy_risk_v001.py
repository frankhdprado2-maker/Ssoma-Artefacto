"""Configured Mamdani decision support over existing temporal evidence only."""
import json
import math
from collections import Counter,defaultdict
from pathlib import Path

from .inference_v001 import digest,path_for_host,write_json


def membership(x,spec):
    p=spec['points']
    if spec['type']=='triangular':a,b,d=p;c=b
    elif spec['type']=='trapezoidal':a,b,c,d=p
    else:raise ValueError('Unknown membership type')
    if not a<=b<=c<=d:raise ValueError('Unordered membership bounds')
    if x<a or x>d:return 0.
    if b<=x<=c:return 1.
    return (x-a)/(b-a) if x<b else (d-x)/(d-c)


def level(score,config):
    return next(r['level'] for r in reversed(config['risk_levels']) if score>=r['lower_inclusive'])


class Mamdani:
    def __init__(self,config):
        self.c=config
        if config['operators']!={'and':'min','implication':'min','aggregation':'max','defuzzification':'sampled_centroid'}:raise ValueError('Unsupported operators')
        o=config['output_universe']
        self.grid=[o['min']+i*o['step'] for i in range(round((o['max']-o['min'])/o['step'])+1)]
        self.outputs={term:[membership(x,spec) for x in self.grid] for term,spec in config['output_memberships'].items()}

    def evaluate(self,inputs,support_frames,support_seconds):
        c=self.c;lo,hi=c['input_universe']
        if set(inputs)!=set(c['input_variables']) or any(not math.isfinite(v) or not lo<=v<=hi for v in inputs.values()):raise ValueError('Inputs outside normalized universe')
        memberships={v:{t:membership(x,spec) for t,spec in c[c['membership_assignment'][v]].items()} for v,x in inputs.items()}
        fired=[];aggregate=[0.]*len(self.grid)
        for rule in c['rules']:
            strength=min(memberships[v][term] for v,term in rule['if_all'].items())
            if strength>0:
                fired.append(dict(rule_id=rule['id'],strength=strength,consequent=rule['then']))
                aggregate=[max(a,min(strength,b)) for a,b in zip(aggregate,self.outputs[rule['then']])]
        total=sum(aggregate)
        raw=sum(x*m for x,m in zip(self.grid,aggregate))/total if total else c['safety']['no_rule_score']
        short=support_frames<c['safety']['minimum_support_frames'] or support_seconds<c['safety']['minimum_support_seconds_for_high']
        score=min(raw,c['safety']['short_evidence_score_cap']) if short else raw
        return dict(risk_score=score,risk_level=level(score,c),raw_mamdani_score=raw,
            temporal_cap_applied=short and score<raw,rules_fired=fired,memberships=memberships,no_rule_coverage=not fired)


def ppe_strength(row,config):
    n=config['normalization']
    if row is None or row['ppe_class'] in n['ppe_support_only_excluded'] or row.get('evidence_mode')=='SUPPORT_ONLY':return 0.
    if row['state'] not in n['ppe_allowed_states'] or not row.get('prior_present') or not row.get('geometric_visibility_proxy'):return 0.
    return max(0.,min(1.,row['evidence_score']))


def context_strength(row,detection,config,fps):
    n=config['normalization']
    if not row or row['state']!=n['context_required_state'] or detection is None:return 0.
    distance=row['normalized_distance']
    if distance is None or not math.isfinite(distance) or distance<0:return 0.
    persistence=min(1.,row['streak_frames']/fps/n['persistence_full_seconds'])
    proximity=1/(1+distance/n['proximity_distance_scale'])
    return max(0.,min(1.,detection['confidence']*proximity*persistence*n['source_severity'][row['context_class']]))


def fall_strength(row,candidate_active,config):
    if row is None or (config['normalization']['fall_requires_candidate_event'] and not candidate_active):return 0.
    return max(0.,min(1.,row['combined_fall_score']*config['normalization']['fall_state_weights'][row['state']]))


def components(events):
    """Disjoint connected intervals per track; no merging of merely adjacent events."""
    bytrack=defaultdict(list)
    for e in events:bytrack[e['track_id']].append(e)
    out=[]
    for tid,items in sorted(bytrack.items()):
        group=[];end=-1
        for e in sorted(items,key=lambda e:(e['supporting_frames'][0],e['event_id'])):
            start=e['supporting_frames'][0]
            if group and start>=end:out.append(group);group=[]
            group.append(e);end=max(end,e['supporting_frames'][-1]+1) if len(group)>1 else e['supporting_frames'][-1]+1
        if group:out.append(group)
    return out


def frame_inputs(frame,tid,active,ppe,context,fall,detections,config,fps):
    values={v:0. for v in config['input_variables']};provenance=[]
    eligible=[]
    mapping={'PPE_POSSIBLE_MISSING':'ppe_evidence','HAZARD_PROXIMITY':'hazard_proximity','MACHINERY_PROXIMITY':'machinery_proximity','VEHICLE_PROXIMITY':'machinery_proximity','FALL_CANDIDATE':'fall_evidence'}
    for e in active:
        kind=e['event_type'];name=e['subject_class'];key=(tid,frame,name);value=0.;source=None
        if kind=='PPE_POSSIBLE_MISSING':
            source=ppe.get(key);value=ppe_strength(source,config)
        elif kind in {'HAZARD_PROXIMITY','MACHINERY_PROXIMITY','VEHICLE_PROXIMITY'}:
            source=context.get(key);index=source.get('source_detection_id') if source else None
            det=detections[index] if index is not None else None
            value=context_strength(source,det,config,fps)
        elif kind=='FALL_CANDIDATE':
            source=fall.get((tid,frame));value=fall_strength(source,True,config)
        if kind in mapping:
            v=mapping[kind];values[v]=max(values[v],value)
            age=(frame-e['supporting_frames'][0]+1)/fps
            # Source persistence (already accumulated upstream) is causal, not future event duration.
            if source and kind=='PPE_POSSIBLE_MISSING':age=max(age,source['missing_streak_frames']/fps)
            if source and kind in {'HAZARD_PROXIMITY','MACHINERY_PROXIMITY','VEHICLE_PROXIMITY'}:age=max(age,source['streak_frames']/fps)
            if source and kind=='FALL_CANDIDATE':age=max(age,source['horizontal_frames']/fps)
            if value>0:eligible.append(age)
            provenance.append(dict(event_id=e['event_id'],variable=v,value=value,subject_class=name,source_frame=frame,source_state=source['state'] if source else None,source_evidence_seconds=age))
    duration=max(eligible,default=0.)
    values['temporal_persistence']=min(1.,duration/config['normalization']['persistence_full_seconds'])
    return values,provenance,duration


def risk_records(events,ppe_rows,context_rows,fall_rows,detections,config,fps,aid):
    engine=Mamdani(config)
    ppe={(r['track_id'],r['frame_id'],r['ppe_class']):r for r in ppe_rows}
    context={(r['track_id'],r['frame_id'],r['context_class']):r for r in context_rows}
    fall={(r['track_id'],r['frame_id']):r for r in fall_rows}
    records=[]
    descriptions={'ppe_evidence':'posible ausencia temporal de PPE','hazard_proximity':'proximidad geométrica a humo/fuego','machinery_proximity':'proximidad geométrica a maquinaria/vehículo','fall_evidence':'candidato de caída con pose'}
    for group in components(events):
        tid=group[0]['track_id'];frames=sorted({f for e in group for f in e['supporting_frames']})
        peak=None
        for f in frames:
            active=[e for e in group if e['supporting_frames'][0]<=f<=e['supporting_frames'][-1]]
            inputs,provenance,seconds=frame_inputs(f,tid,active,ppe,context,fall,detections,config,fps)
            event_frames=max((f-e['supporting_frames'][0]+1 for e in active if e['event_type']!='PPE_PRESENT'),default=0)
            result=engine.evaluate(inputs,support_frames=min(round(seconds*fps),event_frames),support_seconds=seconds)
            if peak is None or result['risk_score']>peak['result']['risk_score']:
                peak=dict(frame=f,active=active,inputs=inputs,provenance=provenance,result=result)
        inputs=peak['inputs'];result=peak['result']
        dominant=[v for v in descriptions if inputs[v]>=config['dominant_factor_minimum']]
        dominant.sort(key=lambda v:(-inputs[v],v))
        primary=max(peak['active'],key=lambda e:(next((p['value'] for p in peak['provenance'] if p['event_id']==e['event_id']),0),config['event_priority'][e['event_type']],e['event_id']))
        explanation=('Prioridad '+result['risk_level']+' por '+', '.join(descriptions[v] for v in dominant)+'. Requiere revisión humana.') if dominant else 'Evidencia insuficiente para elevar prioridad; LOW no certifica seguridad ni cumplimiento.'
        source_states={name:ppe.get((tid,peak['frame'],name),{}).get('state','UNAVAILABLE') for name in ('helmet','safety_vest','gloves','goggles','safety_boots','safety_harness')}
        quality=dict(interpretation='Evidence strength and availability; not calibrated probability',
            strongest_contributing_evidence=max((inputs[v] for v in descriptions),default=0),
            ppe_states=source_states,fall_state=fall.get((tid,peak['frame']),{}).get('state','UNAVAILABLE'),
            context_states={name:context.get((tid,peak['frame'],name),{}).get('state','UNAVAILABLE') for name in ('smoke','fire','vehicle','machinery')})
        records.append(dict(risk_id=f'{aid}_risk_{len(records)+1:05d}',analysis_id=aid,event_id=primary['event_id'],
            source_event_ids=sorted(e['event_id'] for e in group),peak_source_event_ids=sorted(e['event_id'] for e in peak['active']),
            source_event_types=sorted({e['event_type'] for e in group}),track_id=tid,
            start_timestamp=frames[0]/fps,end_timestamp=(frames[-1]+1)/fps,peak_timestamp=peak['frame']/fps,
            peak_frame=peak['frame'],supporting_frames=frames,inputs=inputs,**result,dominant_factors=dominant,
            evidence_quality=quality,source_evidence=peak['provenance'],explanation_short=explanation,
            EXPERT_VALIDATION_REQUIRED=True,score_interval_policy='Maximum simultaneous evidence score within grouped interval; not a per-frame current score'))
    return records


def process(analysis,root,preview_path=None):
    import cv2
    import numpy as np
    root=Path(root).resolve();analysis=path_for_host(analysis).resolve()
    if root/'results/analysis' not in analysis.parents:raise ValueError('Invalid output directory')
    config_path=root/'config/fuzzy_risk_v001.json';c=json.loads(config_path.read_text())
    names=['detections.jsonl','tracks.jsonl','ppe_associations.jsonl','ppe_temporal_evidence.jsonl','context_associations.jsonl','fall_candidates_v002.jsonl','events.jsonl','metadata.json','annotated_tracking.mp4','fall_engine_v002_summary.json']
    paths=[analysis/n for n in names]+[config_path,Path(__file__)]
    before={str(p):digest(p) for p in paths}
    summary=json.loads((analysis/'fall_engine_v002_summary.json').read_text())
    if summary['status']!='FALL_ENGINE_V002_READY' or digest(analysis/'events.jsonl')!=summary['events_after_sha256']:raise ValueError('Fall event provenance mismatch')
    if digest(analysis/'fall_candidates_v002.jsonl')!=summary['outputs']['fall_candidates_v002.jsonl']:raise ValueError('Fall evidence signature mismatch')
    meta=json.loads((analysis/'metadata.json').read_text());fps=meta['fps'];aid=meta['analysis_id']
    n=summary['frames_in_smoke']
    def load(name):return [json.loads(line) for line in (analysis/name).read_text().splitlines()]
    events=load('events.jsonl');ppe=load('ppe_temporal_evidence.jsonl');context=load('context_associations.jsonl');fall=load('fall_candidates_v002.jsonl');detections=load('detections.jsonl')
    for rows in (ppe,context,fall,detections):
        for r in rows:
            if r['analysis_id']!=aid or not 0<=r['frame_id']<n or abs(r['timestamp_seconds']-r['frame_id']/fps)>1e-8:raise ValueError('Evidence timeline mismatch')
    for e in events:
        if e['analysis_id']!=aid or e['supporting_frames']!=list(range(e['supporting_frames'][0],e['supporting_frames'][-1]+1)):raise ValueError('Noncontiguous event')
    risks=risk_records(events,ppe,context,fall,detections,c,fps,aid)
    output=analysis/'risk_events.jsonl';output.write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in risks))
    active=defaultdict(list)
    for r in risks:
        for f in r['supporting_frames']:active[f].append(r)
    cap=cv2.VideoCapture(str(analysis/'annotated_tracking.mp4'))
    vc=c['video'];height=max(meta['height'],vc['heading_height']+vc['row_height']*max((len(v) for v in active.values()),default=0));height+=height%2
    video=analysis/'annotated_risk.mp4'
    writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),fps,(meta['width']+vc['panel_width'],height))
    if not cap.isOpened() or not writer.isOpened():raise RuntimeError('Video unavailable')
    preview=max(risks,key=lambda r:r['risk_score'])['peak_frame'] if risks else 0
    try:
        for f in range(n):
            ok,frame=cap.read()
            if not ok:raise RuntimeError('Truncated tracking video')
            canvas=np.zeros((height,meta['width']+vc['panel_width'],3),dtype=np.uint8);canvas[:meta['height'],:meta['width']]=frame
            cv2.putText(canvas,'PROVISIONAL | interval peak, review required',(meta['width']+8,20),0,vc['detail_font_scale'],(255,255,255),1)
            for i,r in enumerate(active[f]):
                x=meta['width']+8;y=vc['heading_height']+i*vc['row_height']
                cv2.putText(canvas,f"Track #{r['track_id']} {r['risk_level']} {r['risk_score']:.1f}/100",(x,y),0,vc['font_scale'],(0,215,255),1)
                relevant=[e['event_type'] for e in events if e['event_id'] in r['peak_source_event_ids'] and e['event_type']!='PPE_PRESENT']
                cv2.putText(canvas,', '.join(sorted(set(relevant))) or 'PPE evidence; no elevated signal',(x,y+18),0,vc['detail_font_scale'],(220,220,220),1)
            writer.write(canvas)
            if f==preview:cv2.imwrite(str(preview_path or root/'reports/integration/fuzzy_risk_preview.jpg'),canvas)
    finally:cap.release();writer.release()
    cap=cv2.VideoCapture(str(video));decoded=0
    while cap.read()[0]:decoded+=1
    vf=cap.get(cv2.CAP_PROP_FPS);cap.release()
    if decoded!=n or abs(vf-fps)>.01:raise ValueError('Risk video timeline mismatch')
    if before!={str(p):digest(p) for p in paths}:raise ValueError('Input mutation')
    counts=Counter(r['risk_level'] for r in risks)
    result=dict(status='FUZZY_RISK_ENGINE_READY',analysis_id=aid,engine=c['version'],EXPERT_VALIDATION_REQUIRED=True,
        total_risk_events=len(risks),**{v:counts[v] for v in c['output_memberships']},
        tracks_with_risk=sorted({r['track_id'] for r in risks}),tracks_with_elevated_risk=sorted({r['track_id'] for r in risks if r['risk_level']!='LOW'}),
        risk_by_event_type={t:dict(Counter(r['risk_level'] for r in risks if t in r['source_event_types'])) for t in sorted({t for r in risks for t in r['source_event_types']})},
        dominant_factors_counts=dict(Counter(v for r in risks for v in r['dominant_factors'])),
        highest_risk_events=[dict(risk_id=r['risk_id'],track_id=r['track_id'],risk_score=r['risk_score'],risk_level=r['risk_level']) for r in sorted(risks,key=lambda r:-r['risk_score'])[:5]],
        no_rule_coverage_records=sum(r['no_rule_coverage'] for r in risks),input_signatures=before,
        outputs={'risk_events.jsonl':digest(output),'annotated_risk.mp4':digest(video)},video_decoded_frames=decoded,
        vision_runs=0,training_runs=0,test_accessed=False,config_sha256=digest(config_path),
        limitations=['Engineering parameters require SSOMA expert validation; no accuracy claim.',
        'LOW can mean missing evidence; it does not establish safety.',
        'Evidence scores are not calibrated probabilities; HIGH/CRITICAL do not confirm an accident.',
        'Composed groups use peak simultaneous frame evidence, never independent maxima across different moments.',
        'Video shows offline interval peak score; not a causal real-time alert.',
        'Event-type counts may overlap; tracks_with_risk includes LOW assessment records.'])
    write_json(analysis/'risk_summary.json',result)
    return result
