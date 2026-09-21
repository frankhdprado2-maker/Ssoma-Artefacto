"""Portable local launcher. Only paths/configuration, never scientific parameters."""
import argparse,os,sys
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def configure():
    # Environment wins over the optional local .env; never execute its contents.
    file=ROOT/'.env'
    if file.is_file():
        for line in file.read_text(encoding='utf-8').splitlines():
            line=line.strip()
            if not line or line.startswith('#'):continue
            key,sep,value=line.partition('=')
            if not sep or not key.replace('_','').isalnum():raise ValueError('Invalid .env entry')
            os.environ.setdefault(key,value.strip().strip('"').strip("'"))
    from deployment.adapter_v001.settings import Settings
    s=Settings.env()
    for key,value in [('SSOMA_DATA_DIR',s.data),('SSOMA_MODEL_DIR',s.models),('SSOMA_UPLOAD_DIR',s.uploads),('SSOMA_RESULT_DIR',s.results)]:os.environ[key]=str(value)
    os.environ.setdefault('YOLO_CONFIG_DIR',str(s.data/'ultralytics'))
    os.environ.setdefault('YOLO_AUTOINSTALL','false');os.environ.setdefault('ULTRALYTICS_AUTOINSTALL','false')
    os.environ['PYTHONDONTWRITEBYTECODE']='1'
    return s

def verify(s):
    from deployment.adapter_v001.runtime import manifest,sha
    spec=manifest()
    for name,expected in spec['scientific_inputs'].items():
        if sha(ROOT/name)!=expected:raise ValueError('Scientific input signature mismatch: '+name)
    for item in spec['files']:
        if sha(s.models/item['relative_path'])!=item['sha256']:raise ValueError('Model signature mismatch: '+item['model_id'])

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');args=parser.parse_args()
    s=configure();verify(s)
    if args.check:print('CONFIG/MODELS PASS')
    else:
        import uvicorn
        print('http://127.0.0.1:'+os.getenv('PORT','8000'),flush=True)
        uvicorn.run('deployment.adapter_v001.app:factory',factory=True,host=os.getenv('SSOMA_BIND','127.0.0.1'),port=int(os.getenv('PORT','8000')),workers=1,proxy_headers=False)
