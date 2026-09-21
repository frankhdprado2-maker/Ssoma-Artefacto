"""Positive CPU EigenCAM check without changing saved risk decisions."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
def main():
    from deployment.adapter_v001.settings import Settings
    from deployment.adapter_v001.runtime import prepare,bound_module,write
    from src.detection.inference_v001 import load_models
    import cv2,time
    s=Settings.env();runtime=prepare(s);xai,_=bound_module('xai_evidence_v001',s,'cpu',runtime)
    p=Path(sys.argv[1]);out=Path(sys.argv[2]);out.mkdir(parents=True,exist_ok=True)
    d=next(json.loads(x) for x in (p/'detections.jsonl').read_text().splitlines() if json.loads(x)['canonical_class']=='person')
    reg=json.loads((runtime/'config/model_registry_v001.json').read_text());entry=next(m for m in reg['models'] if m['model_id']==d['model_id'])
    cap=cv2.VideoCapture(json.loads((p/'metadata.json').read_text())['input_path']);cap.set(1,d['frame_id']);ok,frame=cap.read();cap.release();assert ok
    t=time.perf_counter();model=load_models({'models':[entry]},'cpu')[0][1]
    evidence=xai.make_cam(model,entry,frame,d,out,reg['inference'])
    write(out/'result.json',dict(status='PASS',seconds=time.perf_counter()-t,evidence=evidence))
if __name__=='__main__':main()
