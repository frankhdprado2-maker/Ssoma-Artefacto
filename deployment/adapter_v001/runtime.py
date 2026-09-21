"""Explicit path/device bindings; original RC1 files remain byte-for-byte intact.

Each replacement is exact and fails closed if RC1 changes. No thresholds, model
parameters, numerical expressions, or scientific control flow are patched.
"""
import copy,hashlib,json,types
from pathlib import Path
from .settings import ROOT

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def write(path,data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def manifest():return json.loads((ROOT/'config/deploy_model_registry_v001.json').read_text())

def configured(s):
    return all((s.models/m['relative_path']).is_file() for m in manifest()['files'])

def prepare(s):
    spec=manifest();registry=copy.deepcopy(spec['registry'])
    for relative,expected in spec['scientific_inputs'].items():
        if sha(ROOT/relative)!=expected:raise ValueError('Frozen scientific input changed: '+relative)
    for f in spec['files']:
        rel=Path(f['relative_path'])
        if rel.is_absolute() or '..' in rel.parts:raise ValueError('Invalid model relative path')
        p=s.models/rel
        if not p.is_file() or sha(p)!=f['sha256']:raise ValueError('Model signature failure: '+f['model_id'])
    for entry in registry['models']:
        for key in ('checkpoint','architecture_checkpoint'):
            if key in entry:entry[key]=str(s.models/entry[key])
    runtime=s.data/'runtime';(runtime/'config').mkdir(parents=True,exist_ok=True)
    for name in ('tracking_ppe_v001','event_engine_v001','fall_engine_v002','fuzzy_risk_v001'):
        src=ROOT/'config'/f'{name}.json';(runtime/'config'/src.name).write_bytes(src.read_bytes())
    write(runtime/'config/model_registry_v001.json',registry)
    return runtime

def bound_module(name,s,device,runtime):
    file=ROOT/'src/detection'/f'{name}.py';source=file.read_text(encoding='utf-8')
    bindings={
      'event_engine_v001':[
        ("root=Path(__file__).resolve().parents[2]","root=DEPLOY_ROOT"),
        ("root/'results'/'analysis' not in analysis.parents","analysis.parent != DEPLOY_RESULTS")],
      'fall_engine_v002':[
        ("root/'results/analysis' not in analysis.parents","analysis.parent != DEPLOY_RESULTS"),
        ("weight=root/'models/pose_v002/yolo11n-pose.pt';metadata_path=weight.parent/'source_metadata.json'","weight=DEPLOY_MODELS/'POSE.pt';metadata_path=DEPLOY_POSE_METADATA"),
        ('device=0,verbose=False','device=DEPLOY_DEVICE,verbose=False'),
        ('torch.cuda.get_device_name(0)','DEPLOY_DEVICE_LABEL')],
      'fuzzy_risk_v001':[("root/'results/analysis' not in analysis.parents","analysis.parent != DEPLOY_RESULTS")],
      'xai_evidence_v001':[
        ("root/'results/analysis' not in analysis.parents","analysis.parent != DEPLOY_RESULTS"),
        ("device='0',verbose=False","device=DEPLOY_DEVICE,verbose=False")]
    }
    changes=[]
    for old,new in bindings[name]:
        if source.count(old)!=1:raise RuntimeError('RC1 operational binding no longer matches: '+old)
        source=source.replace(old,new);changes.append({'before':old,'after':new})
    module=types.ModuleType('src.detection._deploy_'+name)
    module.__file__=str(file);module.__package__='src.detection'
    module.__dict__.update(DEPLOY_ROOT=runtime,DEPLOY_RESULTS=s.results,
        DEPLOY_MODELS=s.models,DEPLOY_POSE_METADATA=ROOT/'models/pose_v002/source_metadata.json',
        DEPLOY_DEVICE=device,DEPLOY_DEVICE_LABEL=device)
    exec(compile(source,str(file),'exec'),module.__dict__)
    if name=='xai_evidence_v001':
        original=module.load_models
        module.load_models=lambda registry,ignored_device:original(registry,device)
        changes.append({'before':'load_models device argument','after':'central device (wrapper)'})
    return module,{'module':name,'source_sha256':sha(file),'operational_bindings':changes}
