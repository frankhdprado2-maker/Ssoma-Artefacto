"""Native ARM64 prerequisite check. Never claims full E2E from imports alone."""
import argparse,json,os,platform,subprocess,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))

def check(models=False):
    result={'architecture':platform.machine(),'ARM64_RUNTIME_VERIFIED':False,'status':'PENDING'}
    if platform.machine() not in {'aarch64','arm64'}:raise RuntimeError('ARM64 execution required; host architecture is '+platform.machine())
    if sys.version_info[:2]!=(3,12):raise RuntimeError('Python 3.12 required')
    import cv2,numpy as np,torch,torchvision,scipy,fastapi,uvicorn,sqlite3,PIL,ultralytics
    assert torch.__version__=='2.9.1+cpu' and torch.version.cuda is None and torch.version.hip is None
    assert torchvision.__version__.split('+')[0]=='0.24.1'
    assert ultralytics.__version__=='8.3.203'
    assert torchvision.ops.nms(torch.tensor([[0.,0.,5.,5.],[0.,0.,5.,5.]]),torch.tensor([.8,.7]),.5).tolist()==[0]
    result['torch']=torch.__version__;result['torchvision']=torchvision.__version__
    result['imports']='PASS';result['cpu_nms']='PASS'
    ffmpeg=os.getenv('IMAGEIO_FFMPEG_EXE','/usr/bin/ffmpeg')
    with tempfile.TemporaryDirectory(prefix='ssoma_codec_') as temp:
        p=Path(temp)/'probe.mp4'
        subprocess.run([ffmpeg,'-nostdin','-v','error','-y','-f','lavfi','-i','color=c=gray:s=64x64:r=10:d=1','-c:v','libx264','-pix_fmt','yuv420p',str(p)],check=True,capture_output=True)
        cap=cv2.VideoCapture(str(p));count=0
        while True:
            ok,im=cap.read()
            if not ok:break
            count+=1
            if count==5:assert cv2.imwrite(str(Path(temp)/'frame.png'),im)
        cap.release();assert count==10
    result['h264_encode_decode_extract']='PASS'
    from src.detection.xai_evidence_v001 import eigen_cam,validate_heatmap
    heat=eigen_cam(np.random.default_rng(42).normal(size=(8,8,8)).astype('float32'));validate_heatmap(heat)
    result['EigenCAM_math']='PASS';result['EigenCAM_network']='PENDING_BENCHMARK'
    if models:
        from deployment.adapter_v001.settings import Settings
        from deployment.adapter_v001.runtime import prepare
        from src.detection.inference_v001 import load_models
        from src.detection.fall_engine_v002 import load_pose
        s=Settings.env();s.prepare();runtime=prepare(s)
        loaded=load_models(json.loads((runtime/'config/model_registry_v001.json').read_text()),'cpu')
        assert len(loaded)==6
        pose=load_pose(s.models/'POSE.pt',json.loads((ROOT/'models/pose_v002/source_metadata.json').read_text()))
        result['models_loaded']=[entry['model_id'] for entry,model in loaded]+['POSE']
        del loaded,pose
    result['status']='PASS';result['ARM64_IMPORTS_VERIFIED']=True
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--models',action='store_true');parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    try:result=check(args.models);code=0
    except Exception as e:result={'status':'BLOCKED','architecture':platform.machine(),'ARM64_RUNTIME_VERIFIED':False,'error':str(e)};code=2
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'architecture':result['architecture'],'ARM64_RUNTIME_VERIFIED':False}));sys.exit(code)
