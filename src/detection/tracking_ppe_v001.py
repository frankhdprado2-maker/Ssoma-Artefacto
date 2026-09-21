"""Postprocess saved detections. Geometric evidence, not pose or compliance."""
from __future__ import annotations

import json
import math
import time
from collections import Counter, defaultdict, deque
from pathlib import Path

from .inference_v001 import OWNERS, SUPPORT, bbox, digest, path_for_host, timestamp, write_json

PPE = ('helmet', 'safety_vest', 'gloves', 'goggles', 'safety_boots', 'safety_harness')
STATES = {'PRESENT', 'UNKNOWN', 'AMBIGUOUS'}


def box(d):
    return [d[k] for k in ('x1', 'y1', 'x2', 'y2')]


def area(b):
    return max(0., b[2]-b[0]) * max(0., b[3]-b[1])


def intersection(a, b):
    return max(0., min(a[2], b[2])-max(a[0], b[0])) * max(0., min(a[3], b[3])-max(a[1], b[1]))


def iou(a, b):
    overlap = intersection(a, b)
    return overlap / max(area(a)+area(b)-overlap, 1e-9)


def center(b):
    return ((b[0]+b[2])/2, (b[1]+b[3])/2)


def regions(person, name, config):
    x, y, r, b = person
    return [[x+a*(r-x), y+c*(b-y), x+d*(r-x), y+e*(b-y)]
            for a, c, d, e in config['regions'][name]]


def score(person, detection, config):
    """Hard anatomy/size gates, then weighted five-factor score in [0,1]."""
    item = box(detection)
    name = detection['canonical_class']
    cx, cy = center(item)
    ratio = area(item)/max(area(person), 1e-9)
    limit = config['maximum_area_fraction'][name]
    anatomical = regions(person, name, config)
    if not (0 < ratio <= limit) or not any(a <= cx <= c and b <= cy <= d for a,b,c,d in anatomical):
        return None
    w, h = person[2]-person[0], person[3]-person[1]
    dx = max(person[0]-cx, 0, cx-person[2])/w
    dy = max(person[1]-cy, 0, cy-person[3])/h
    proximity = max(0., 1.-math.hypot(dx, dy)/.30)
    anatomy = max(intersection(item, r)/area(item) for r in anatomical)
    size = 1.-.5*ratio/limit
    overlap = intersection(person, item)/area(item)
    factors = dict(proximity=proximity, anatomy=anatomy, size=size,
                   overlap=overlap, confidence=detection['confidence'])
    value = sum(v*w for v,w in zip(factors.values(), config['association']['weights']))
    return value, factors


class PersonTracker:
    """Constant-velocity geometry + Hungarian assignment; no appearance/ReID."""
    def __init__(self, config):
        self.config = config['tracker']
        self.tracks = {}
        self.next_id = 1
        self.last_frame = -1

    def update(self, frame, detections):
        import numpy as np
        from scipy.optimize import linear_sum_assignment
        if frame != self.last_frame + 1:
            raise ValueError('Tracker requires contiguous zero-based frames')
        self.last_frame = frame
        people = [d for d in detections if d['canonical_class']=='person' and d['model_id']=='PERSON_B']
        self.tracks = {k:v for k,v in self.tracks.items() if frame-v['frame'] <= self.config['max_gap_frames']}
        ids = sorted(self.tracks)
        predictions = {k:[a+v*(frame-self.tracks[k]['frame']) for a,v in zip(self.tracks[k]['box'], self.tracks[k]['velocity'])] for k in ids}
        costs = np.full((len(ids), len(people)), 1e6)
        for i,k in enumerate(ids):
            p = predictions[k]
            for j,d in enumerate(people):
                b = box(d)
                scale = max(math.hypot(p[2]-p[0], p[3]-p[1]), 1.)
                distance = math.dist(center(p), center(b))/scale
                overlap = iou(p,b)
                ratio = max(area(p),area(b))/max(min(area(p),area(b)),1e-9)
                if ratio <= self.config['max_area_ratio'] and (overlap >= self.config['min_iou'] or distance <= self.config['max_center_distance']):
                    costs[i,j] = .7*(1-overlap)+.3*min(distance,1.)
        matched = {}
        if costs.size:
            rows, cols = linear_sum_assignment(costs)
            matched = {int(j):ids[int(i)] for i,j in zip(rows,cols) if costs[i,j] < 1e6}
        observed = set()
        for j,d in enumerate(people):
            k = matched.get(j)
            b = box(d)
            velocity = [0.]*4
            if k is None:
                k = self.next_id
                self.next_id += 1
            else:
                old = self.tracks[k]
                dt = frame-old['frame']
                velocity = [.5*v+.5*(a-c)/dt for a,c,v in zip(b,old['box'],old['velocity'])]
            self.tracks[k] = dict(box=b, velocity=velocity, frame=frame, confidence=d['confidence'])
            observed.add(k)
        result = []
        for k,t in sorted(self.tracks.items()):
            if frame-t['frame'] <= self.config['visible_prediction_frames']:
                result.append(dict(track_id=int(k), person_bbox=t['box'] if k in observed else predictions[k],
                    person_confidence=t['confidence'] if k in observed else None,
                    tracking_state='OBSERVED' if k in observed else 'PREDICTED', last_observed_frame=t['frame']))
        return result


def assign(tracks, detections, config):
    """Each PPE gets one owner, or explicit ambiguity across tied candidates."""
    result = []
    for d in detections:
        if d['canonical_class'] not in PPE:
            continue
        candidates = []
        for t in tracks:
            if t['tracking_state'] != 'OBSERVED':
                continue
            scored = score(t['person_bbox'], d, config)
            if scored and scored[0] >= config['association']['minimum_score']:
                candidates.append((scored[0], t['track_id'], scored[1]))
        candidates.sort(key=lambda x:(-x[0],x[1]))
        tied = [c for c in candidates if candidates[0][0]-c[0] <= config['association']['ambiguity_margin']]
        state = 'AMBIGUOUS' if len(tied)>1 else 'PRESENT' if tied else 'UNKNOWN'
        chosen = tied if state=='AMBIGUOUS' else candidates[:1]
        result.append(dict(detection=d, raw_state=state, candidates=chosen))
    return result


class TemporalEvidence:
    def __init__(self, config):
        self.config = config['temporal']
        self.history = defaultdict(deque)

    def update(self, frame, track_id, name, raw_state):
        history = self.history[track_id,name]
        while history and history[0] <= frame-self.config['window_frames']:
            history.popleft()
        if raw_state == 'PRESENT' and (not history or history[-1] != frame):
            history.append(frame)
        if raw_state == 'AMBIGUOUS':
            return 'AMBIGUOUS', len(history)
        present = len(history) >= self.config['minimum_positive_frames'] and frame-history[-1] <= self.config['hold_frames']
        return ('PRESENT' if present else 'UNKNOWN'), len(history)


def association_rows(base, tracks, assignments, temporal):
    grouped = defaultdict(list)
    rows = []
    for a in assignments:
        d = a['detection']
        for value, tid, factors in a['candidates']:
            grouped[tid,d['canonical_class']].append((a,value,factors))
        if not a['candidates']:
            rows.append(dict(base, track_id=None, ppe_class=d['canonical_class'], ppe_bbox=box(d),
                ppe_confidence=d['confidence'], association_score=None, association_state='UNKNOWN',
                raw_state='UNKNOWN', evidence_mode=d['evidence_mode'], model_id=d['model_id'],
                detection_id=d['detection_id'], evidence_origin='UNASSIGNED', candidate_track_ids=[]))
    for t in tracks:
        for name in PPE:
            matches = grouped[t['track_id'],name]
            raw = 'AMBIGUOUS' if any(a['raw_state']=='AMBIGUOUS' for a,_,_ in matches) else 'PRESENT' if matches else 'UNKNOWN'
            state, count = temporal.update(base['frame_id'], t['track_id'], name, raw)
            for match in matches or [None]:
                a,value,factors = match if match else (None,None,None)
                d = a['detection'] if a else None
                rows.append(dict(base, track_id=t['track_id'], ppe_class=name,
                    ppe_bbox=box(d) if d else None, ppe_confidence=d['confidence'] if d else None,
                    association_score=value, association_state=state, raw_state=raw,
                    evidence_mode='SUPPORT_ONLY' if name in SUPPORT else 'DETECTION_ONLY', model_id=OWNERS[name],
                    detection_id=d['detection_id'] if d else None, score_factors=factors,
                    evidence_origin='CURRENT' if d else 'TEMPORAL_HOLD' if state=='PRESENT' else 'NO_CURRENT_EVIDENCE',
                    positive_frames_in_window=count,
                    candidate_track_ids=[c[1] for c in a['candidates']] if a else []))
    return rows


def draw(frame, tracks, rows, detections):
    import cv2
    height,width = frame.shape[:2]
    labels = []
    def mark(b, label, color):
        x,y,r,s = [int(round(v)) for v in b]
        x,r = max(0,min(width-1,x)),max(0,min(width-1,r))
        y,s = max(0,min(height-1,y)),max(0,min(height-1,s))
        cv2.rectangle(frame,(x,y),(r,s),color,1)
        tw = cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,.35,1)[0][0]
        tx,ty = max(0,min(x,width-tw-2)),max(11,y-3)
        for offset in [0]+[sign*step for step in range(14,height,14) for sign in (1,-1)]:
            candidate = max(11,min(height-3,ty+offset))
            rect = [tx,candidate-10,tx+tw+2,candidate+2]
            if not any(intersection(rect,old)>0 for old in labels):
                ty = candidate
                break
        labels.append([tx,ty-10,tx+tw+2,ty+2])
        if abs(ty-y)>16:
            cv2.line(frame,(x,y),(tx,ty),color,1)
        cv2.rectangle(frame,(tx,ty-10),(tx+tw+2,ty+2),(20,20,20),-1)
        cv2.putText(frame,label,(tx,ty),cv2.FONT_HERSHEY_SIMPLEX,.35,color,1,cv2.LINE_AA)
    for d in detections:
        if d['domain'] in {'HAZARD','CONTEXT'}:
            mark(box(d),f"{d['canonical_class']} {d['confidence']:.2f}",(255,180,100))
    for t in tracks:
        mark(t['person_bbox'],f"Person #{t['track_id']} {t['tracking_state']}",(60,230,60))
    seen = set()
    for row in rows:
        if row['ppe_bbox'] is None or row['detection_id'] in seen:
            continue
        seen.add(row['detection_id'])
        owner = '/'.join(f'#{t}' for t in row['candidate_track_ids']) if row['association_state']=='AMBIGUOUS' else f"#{row['track_id']}" if row['track_id'] else 'unassigned'
        support = ' SUPPORT' if row['evidence_mode']=='SUPPORT_ONLY' else ''
        mark(row['ppe_bbox'],f"{owner} {row['ppe_class']} {row['ppe_confidence']:.2f} {row['association_state']}{support}",(0,210,255))
    return frame


def process(analysis, config_path):
    import cv2
    analysis,config_path = path_for_host(analysis),path_for_host(config_path)
    config = json.loads(config_path.read_text())
    inputs = [analysis/p for p in ('metadata.json','summary.json','detections.jsonl','processing_report.json')]+[config_path]
    before = {str(p):digest(p) for p in inputs}
    meta = json.loads((analysis/'metadata.json').read_text())
    summary = json.loads((analysis/'summary.json').read_text())
    if summary['status'] != 'COMPLETED' or summary['analysis_id'] != meta['analysis_id']:
        raise ValueError('Incomplete or mismatched analysis')
    source = path_for_host(meta['input_path'])
    if digest(source) != meta['input_sha256']:
        raise ValueError('Source video integrity mismatch')
    fps,n = meta['fps'],summary['frames_processed']
    if n <= 0 or n > meta['frames']:
        raise ValueError('Invalid processed frame count')
    frames = defaultdict(list)
    with (analysis/'detections.jsonl').open() as stream:
        for i,line in enumerate(stream):
            d = json.loads(line)
            f = d['frame_id']
            if d['analysis_id'] != meta['analysis_id'] or not 0 <= f < n or abs(d['timestamp_seconds']-timestamp(f,fps)) > 1e-8:
                raise ValueError('Invalid detection timeline')
            if d['model_id'] != OWNERS.get(d['canonical_class']):
                raise ValueError('Detection model ownership mismatch')
            bbox(box(d),meta['width'],meta['height'])
            if not math.isfinite(d['confidence']) or not 0 <= d['confidence'] <= 1:
                raise ValueError('Invalid confidence')
            expected = 'SUPPORT_ONLY' if d['canonical_class'] in SUPPORT else 'DETECTION_ONLY'
            if d['evidence_mode'] != expected:
                raise ValueError('Evidence policy mismatch')
            d['detection_id'] = i
            frames[f].append(d)
    tracker,temporal = PersonTracker(config),TemporalEvidence(config)
    counts,states = Counter(),Counter()
    track_stats = {}
    ambiguous = unassigned = 0
    tracking_seconds = association_seconds = 0.
    cap = cv2.VideoCapture(str(source))
    output = analysis/'annotated_tracking.mp4'
    writer = cv2.VideoWriter(str(output),cv2.VideoWriter_fourcc(*'mp4v'),fps,(meta['width'],meta['height']))
    if not cap.isOpened() or not writer.isOpened():
        cap.release();writer.release()
        raise RuntimeError('Video reader/writer unavailable')
    started = time.perf_counter()
    try:
        with (analysis/'tracks.jsonl').open('w') as tf, (analysis/'ppe_associations.jsonl').open('w') as af:
            for f in range(n):
                ok,image = cap.read()
                if not ok:
                    raise RuntimeError(f'Source decode failed at frame {f}')
                base = dict(analysis_id=meta['analysis_id'], frame_id=f, timestamp_seconds=timestamp(f,fps))
                start = time.perf_counter()
                tracks = tracker.update(f,frames[f])
                tracking_seconds += time.perf_counter()-start
                start = time.perf_counter()
                assignments = assign(tracks,frames[f],config)
                rows = association_rows(base,tracks,assignments,temporal)
                association_seconds += time.perf_counter()-start
                for t in tracks:
                    tf.write(json.dumps(dict(base,**t),allow_nan=False)+'\n')
                    stat = track_stats.setdefault(t['track_id'],dict(track_id=t['track_id'],first_frame=f,last_frame=f,observed_frames=0,predicted_frames=0,states_per_class={c:Counter() for c in PPE}))
                    stat['last_frame'] = f
                    stat['observed_frames' if t['tracking_state']=='OBSERVED' else 'predicted_frames'] += 1
                for a in assignments:
                    if a['raw_state']=='PRESENT':
                        counts[a['detection']['canonical_class']] += 1
                    else:
                        unassigned += 1
                        ambiguous += a['raw_state']=='AMBIGUOUS'
                frame_states = set()
                for row in rows:
                    if row['association_state'] not in STATES:
                        raise AssertionError('Invalid association state')
                    af.write(json.dumps(row,allow_nan=False)+'\n')
                    states[row['association_state']] += 1
                    if row['track_id'] is not None:
                        frame_states.add((row['track_id'],row['ppe_class'],row['association_state']))
                for tid,name,state in frame_states:
                    track_stats[tid]['states_per_class'][name][state] += 1
                writer.write(draw(image,tracks,rows,frames[f]))
    finally:
        cap.release();writer.release()
    elapsed = time.perf_counter()-started
    check = cv2.VideoCapture(str(output))
    decoded = 0
    while True:
        ok,_ = check.read()
        if not ok: break
        decoded += 1
    output_fps = check.get(cv2.CAP_PROP_FPS)
    check.release()
    if decoded != n or abs(output_fps-fps) > .01:
        raise AssertionError('Output video timeline mismatch')
    if before != {str(p):digest(p) for p in inputs} or digest(source) != meta['input_sha256']:
        raise AssertionError('Read-only input changed')
    report = dict(analysis_id=meta['analysis_id'],status='TRACKING_ASSOCIATION_READY',
        tracker='ConstantVelocity_Hungarian_v001', tracker_choice='ByteTrack unavailable: lap missing; existing scipy used. Saved detections only (conf >= 0.25).',
        frames_processed=n,unique_person_tracks=len(track_stats),
        average_track_length_frames=sum(s['observed_frames'] for s in track_stats.values())/max(len(track_stats),1),
        ppe_associations_per_class={c:counts[c] for c in PPE}, ambiguous_associations=ambiguous,
        unassigned_ppe=unassigned, association_states_rows=dict(states),
        count_policy='Associations count uniquely owned current PPE detections before temporal confirmation. Ambiguity counts unique PPE detections; unassigned includes ambiguity. Track length counts observed frames only.',
        tracking_processing_fps=n/max(tracking_seconds,1e-9),tracking_seconds=tracking_seconds,
        association_processing_seconds=association_seconds,postprocessing_seconds=elapsed,
        postprocessing_fps=n/elapsed,video_decoded_frames=decoded,
        input_signatures=before,source_sha256=meta['input_sha256'],inputs_unchanged=True,
        config=config,detector_inference_runs=0,training_runs=0,test_accessed=False,
        limitations=['Geometric regions are not pose estimates. No ground truth association/identity accuracy.',
          'No appearance ReID; crossings, occlusion and camera cuts can fragment or swap IDs.',
          'PRESENT requires repeated frames; UNKNOWN does not establish absence.',
          'Boots and goggles SUPPORT_ONLY. HAZARD/CONTEXT are visible but not associated with risk.',
          'Nominal constant-rate timestamps; audio not preserved. Development demonstration only.'])
    write_json(analysis/'track_summary.json',dict(analysis_id=meta['analysis_id'],tracks=list(track_stats.values()),
        unique_person_tracks=len(track_stats),average_track_length_frames=report['average_track_length_frames']))
    report['output_signatures'] = {p:digest(analysis/p) for p in ('tracks.jsonl','ppe_associations.jsonl','track_summary.json','annotated_tracking.mp4')}
    write_json(analysis/'association_report.json',report)
    return report
