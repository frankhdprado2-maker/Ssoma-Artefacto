"""Distribution checks using isolated temporary storage; never runs inference."""
import json,os,secrets,socket,sqlite3,subprocess,sys,tempfile,urllib.request,urllib.error
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from run_runtime import configure,verify

def main():
    s=configure();verify(s);print('CONFIG       PASS');print('MODELS       PASS')
    from src.detection import inference_v001,tracking_ppe_v001,event_engine_v001,fall_engine_v002,fuzzy_risk_v001,xai_evidence_v001
    from deployment.adapter_v001 import app,queue,worker,db
    for name,mod in list(sys.modules.items()):
        if name.startswith(('src.','web.','deployment.')) and getattr(mod,'__file__',None):
            assert ROOT in Path(mod.__file__).resolve().parents,'Imported module outside distribution'
    print('IMPORTS      PASS')
    for aid in ('../x','x/y','x\\y'):
        try:db.safe_id(aid)
        except ValueError:pass
        else:raise AssertionError('Unsafe ID accepted')
    assert all(not (s.models/x['filename']).is_symlink() for x in json.loads((ROOT/'models/model_manifest.json').read_text()))
    print('PATHS        PASS')
    from dataclasses import replace
    with tempfile.TemporaryDirectory(prefix='ssoma_check_') as temp:
        base=Path(temp);isolated=replace(s,data=base/'state',uploads=base/'uploads',results=base/'results',history=None)
        store=db.Store(isolated)
        for _ in range(2):store.update(store.reserve('self_check.mp4'),status='QUEUED')
        assert store.claim(os.getpid()) and store.claim(os.getpid()) is None
        assert db.Store(isolated).recover()==1
        assert store.cleanup()['dry_run'] is True
        print('STORAGE      PASS')
        token=secrets.token_urlsafe(24)
        with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        env=dict(os.environ,SSOMA_DATA_DIR=str(base/'server'),SSOMA_UPLOAD_DIR=str(base/'server_uploads'),SSOMA_RESULT_DIR=str(base/'server_results'),SSOMA_MODEL_DIR=str(s.models),SSOMA_DEVICE='cpu',SSOMA_ACCESS_TOKEN=token,PORT=str(port),SSOMA_BIND='127.0.0.1',PYTHONDONTWRITEBYTECODE='1',YOLO_CONFIG_DIR=str(base/'ultralytics'))
        env.pop('PYTHONPATH',None);env.pop('SSOMA_HISTORY_DIR',None)
        proc=subprocess.Popen([sys.executable,str(ROOT/'scripts/run_runtime.py')],cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        try:
            for line in proc.stdout:
                if 'Uvicorn running on' in line:break
            else:raise RuntimeError('Server startup failed')
            def request(path,auth=True):
                req=urllib.request.Request(f'http://127.0.0.1:{port}'+path,headers={'Authorization':'Bearer '+token} if auth else {})
                try:
                    with urllib.request.urlopen(req,timeout=10) as r:return r.status,r.read()
                except urllib.error.HTTPError as e:return e.code,e.read()
            code,body=request('/health',False);assert code==200 and json.loads(body)['models_configured']
            assert request('/api/analyses',False)[0]==401
            for route in ('/','/monitor','/api/analyses','/branding/logo','/favicon.ico','/static/app.js','/static/app.css'):assert request(route)[0]==200,route
            assert json.loads(request('/api/analyses')[1])==[]
            print('SERVER       PASS')
        finally:
            proc.terminate()
            try:proc.wait(timeout=15)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()

if __name__=='__main__':main()
