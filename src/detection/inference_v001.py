"""Six-model development inference; no training, tracking or compliance rules."""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

CLASSES = ('person', 'helmet', 'safety_vest', 'gloves', 'goggles', 'safety_boots',
           'safety_harness', 'smoke', 'fire', 'vehicle', 'machinery')
SUPPORT = {'safety_boots', 'goggles'}
OWNERS = dict(zip(CLASSES, ('PERSON_B', 'PPE_BASELINE', 'VEST_B', 'GLOVES_B',
    'PPE_BASELINE', 'PPE_BASELINE', 'PPE_BASELINE', 'HAZARD', 'HAZARD', 'CONTEXT', 'CONTEXT')))


def path_for_host(value):
    value = str(value).replace('\\', '/')
    if os.name != 'nt' and len(value) > 1 and value[1] == ':':
        value = '/mnt/' + value[0].lower() + '/' + value[2:].lstrip('/')
    return Path(value)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def timestamp(frame_id, fps):
    if isinstance(frame_id, bool) or not isinstance(frame_id, int) or frame_id < 0:
        raise ValueError('frame_id must be a nonnegative integer')
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError('Invalid FPS')
    return frame_id / fps


def bbox(values, width, height):
    if len(values) != 4 or not all(math.isfinite(float(x)) for x in values):
        raise ValueError('Invalid bbox')
    x1, y1, x2, y2 = map(float, values)
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError('BBox outside frame or degenerate')
    return [x1, y1, x2, y2]


def validate_registry(registry, verify_files=True):
    if registry.get('canonical_classes') != list(CLASSES):
        raise ValueError('Canonical taxonomy mismatch')
    owners = {}
    ids = set()
    for model in registry['models']:
        mid = model['model_id']
        if mid in ids:
            raise ValueError('Duplicate model ID')
        ids.add(mid)
        expected_domain = 'PPE' if mid in {'PERSON_B','VEST_B','GLOVES_B','PPE_BASELINE'} else mid
        if model['domain'] != expected_domain:
            raise ValueError('Domain mismatch')
        for local, canonical in model['canonical_mapping'].items():
            if local not in model['local_classes'] or canonical not in CLASSES:
                raise ValueError('Invalid local-to-canonical mapping')
            if canonical in owners or OWNERS[canonical] != mid:
                raise ValueError('Class responsibility conflict')
            owners[canonical] = mid
        if verify_files:
            for path_key, sha_key in [('checkpoint','sha256'), ('architecture_checkpoint','architecture_sha256')]:
                if path_key in model:
                    p = path_for_host(model[path_key])
                    if not p.is_file() or digest(p) != model[sha_key]:
                        raise ValueError(f'Checkpoint integrity failure: {mid}/{path_key}')
    if owners != OWNERS:
        raise ValueError('Missing responsible class/model')
    policy = registry['inference']
    if (policy['imgsz'], policy['conf'], policy['iou']) != (640, .25, .70):
        raise ValueError('Frozen confidence policy changed')
    return owners


def normalize(model, raw, analysis_id, frame_id, fps, width, height):
    result = []
    for item in raw:
        if len(item) != 6:
            raise ValueError('Expected xyxy/confidence/local class')
        x1,y1,x2,y2,confidence,local = map(float, item)
        if not math.isfinite(local) or not local.is_integer():
            raise ValueError('Invalid local class ID')
        canonical = model['canonical_mapping'].get(str(int(local)))
        if canonical is None:  # Baseline person/vest/gloves must never enter composition.
            continue
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError('Invalid confidence')
        box = bbox([x1,y1,x2,y2],width,height)
        result.append(dict(analysis_id=analysis_id,frame_id=frame_id,
            timestamp_seconds=timestamp(frame_id,fps),model_id=model['model_id'],domain=model['domain'],
            canonical_class=canonical,canonical_class_id=CLASSES.index(canonical),confidence=confidence,
            x1=box[0],y1=box[1],x2=box[2],y2=box[3],
            evidence_mode='SUPPORT_ONLY' if canonical in SUPPORT else 'DETECTION_ONLY'))
    return result


def video_metadata(path):
    import cv2
    path = Path(path)
    if path.suffix.lower() not in {'.mp4','.avi','.mov'}:
        raise ValueError('Unsupported input; use mp4/avi/mov')
    if not path.is_file():
        raise FileNotFoundError(path)
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError('Video cannot be opened')
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width,height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        timestamp(0,fps)
        if frames <= 0 or width <= 0 or height <= 0:
            raise ValueError('Invalid video metadata')
        ok, first = cap.read()
        if not ok or first.shape[:2] != (height,width):
            raise ValueError('First video frame cannot be decoded')
        return dict(input_filename=path.name,input_path=str(path.resolve()),fps=fps,frames=frames,
            width=width,height=height,resolution=[width,height],duration=frames/fps,
            timestamp_policy='zero-based frame_id / nominal FPS; constant-rate timeline, no source PTS claim',audio_preserved=False)
    finally:
        cap.release()


def load_models(registry, device):
    os.environ.update(YOLO_AUTOINSTALL='false',ULTRALYTICS_AUTOINSTALL='false')
    import torch
    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel
    torch.set_num_threads(4)
    loaded = []
    for entry in registry['models']:
        p = path_for_host(entry['checkpoint'])
        if entry.get('format') == 'ema_state_dict':
            initial = torch.load(path_for_host(entry['architecture_checkpoint']),map_location='cpu',weights_only=False)
            saved = torch.load(p,map_location='cpu',weights_only=False)
            if saved['update'] != 700 or saved['initial_state_sha256'] != initial['state_sha256']:
                raise ValueError('GLOVES checkpoint identity mismatch')
            net = DetectionModel(initial['yaml'],nc=1,verbose=False)
            net.load_state_dict(saved['ema'],strict=True)
            net.names = {int(k):v for k,v in entry['local_classes'].items()}
            # Construct wrapper from an existing local architecture file, never a model download.
            import ultralytics
            wrapper = YOLO(str(Path(ultralytics.__file__).parent/'cfg/models/11/yolo11.yaml'),task='detect',verbose=False)
            wrapper.model = net.eval()
            wrapper.overrides = {'task':'detect','model':str(p)}
        else:
            wrapper = YOLO(str(p),task='detect')
        actual = {str(k):v for k,v in wrapper.names.items()}
        if actual != entry['local_classes']:
            raise ValueError(f'Actual checkpoint classes mismatch: {entry["model_id"]}: {actual}')
        wrapper.to('cuda:'+str(device) if str(device).isdigit() else device)
        loaded.append((entry,wrapper))
    return loaded


def process_video(input_path, registry_path, output_root, device='0', max_frames=None, *, output_dir=None, progress_callback=None):
    import cv2
    input_path,registry_path = path_for_host(input_path),path_for_host(registry_path)
    metadata = video_metadata(input_path)
    if max_frames is not None and max_frames <= 0:
        raise ValueError('max_frames must be positive')
    registry = json.loads(registry_path.read_text(encoding='utf-8'))
    validate_registry(registry)
    if output_dir is None:
        analysis_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]
        output = path_for_host(output_root)/analysis_id
        output.mkdir(parents=True,exist_ok=False)
    else:
        output=path_for_host(output_dir).resolve();analysis_id=output.name
        if not (output/'application_job.json').is_file() or (output/'metadata.json').exists():
            raise ValueError('Application output must be a new reserved job, never an existing analysis')
    start = time.perf_counter()
    metadata.update(analysis_id=analysis_id,input_sha256=digest(input_path),
        registry_fingerprint=digest(registry_path),start_timestamp=datetime.now(timezone.utc).isoformat(),
        end_timestamp=None,confidence_policy=registry['inference'],evidence_policy=registry['evidence_policy'],
        max_frames=max_frames,device=device)
    write_json(output/'metadata.json',metadata)
    count=0; totals=Counter(); errors=[]; cap=writer=None;loaded=[];complete=False
    try:
        loaded = load_models(registry,device)
        cap = cv2.VideoCapture(str(input_path))
        writer = cv2.VideoWriter(str(output/'annotated.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),
            metadata['fps'],(metadata['width'],metadata['height']))
        if not cap.isOpened() or not writer.isOpened():
            raise RuntimeError('Video reader/writer failed')
        target = min(metadata['frames'],max_frames) if max_frames else metadata['frames']
        with (output/'detections.jsonl').open('w',encoding='utf-8') as stream:
            while count < target:
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError(f'Unexpected decode failure at frame {count}/{target}')
                annotated = frame.copy();detections=[]
                for entry,model in loaded:
                    result = model.predict(frame,imgsz=640,conf=.25,iou=.7,classes=[int(k) for k in entry['canonical_mapping']],
                        agnostic_nms=False,max_det=300,half=False,device=device,augment=False,save=False,verbose=False)[0]
                    detections.extend(normalize(entry,result.boxes.data.cpu().tolist(),analysis_id,count,
                        metadata['fps'],metadata['width'],metadata['height']))
                # No composition NMS: boxes from different objects/classes stay intact.
                for d in detections:
                    stream.write(json.dumps(d,allow_nan=False)+'\n');totals[d['canonical_class']]+=1
                    color=(40,210,255) if d['evidence_mode']=='SUPPORT_ONLY' else (60,220,60)
                    p1=(int(d['x1']),int(d['y1']));p2=(int(d['x2']),int(d['y2']))
                    cv2.rectangle(annotated,p1,p2,color,2)
                    cv2.putText(annotated,f"{d['canonical_class']} {d['confidence']:.2f}",(p1[0],max(14,p1[1]-4)),cv2.FONT_HERSHEY_SIMPLEX,.45,color,1,cv2.LINE_AA)
                cv2.putText(annotated,'Development detections only - boots/goggles: support only',(8,metadata['height']-12),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1,cv2.LINE_AA)
                writer.write(annotated);count+=1
                if progress_callback is not None and (count % 5 == 0 or count == target):progress_callback(count,target)
        writer.release();writer=None
        verification = video_metadata(output/'annotated.mp4')
        if verification['frames'] != count:
            raise RuntimeError('Annotated frame count mismatch')
        if digest(input_path) != metadata['input_sha256']:
            raise RuntimeError('Input changed during analysis')
        validate_registry(registry)
        complete=True
    except Exception as error:
        errors.append(f'{type(error).__name__}: {error}')
        raise
    finally:
        if cap is not None:cap.release()
        if writer is not None:writer.release()
        elapsed=time.perf_counter()-start
        metadata['end_timestamp']=datetime.now(timezone.utc).isoformat()
        write_json(output/'metadata.json',metadata)
        summary=dict(analysis_id=analysis_id,status='COMPLETED' if complete else 'FAILED',
            detections_per_class={k:totals[k] for k in CLASSES},frames_processed=count,
            processing_seconds=elapsed,video_seconds=count/metadata['fps'],processing_fps=count/elapsed,errors=errors)
        write_json(output/'summary.json',summary)
        write_json(output/'processing_report.json',dict(**summary,models_loaded=[e['model_id'] for e,m in loaded],
            registry_fingerprint=metadata['registry_fingerprint'],scientific_accuracy_evaluated=False,
            training=False,compliance_rules=False,audio_preserved=False,output=str(output)))
    return output,summary
