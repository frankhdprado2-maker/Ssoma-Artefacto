import json,os,re,time,uuid
from pathlib import Path
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[2]
ANALYSES=ROOT/'results/analysis'
CACHE=ROOT/'web/runtime/presentation'
TERMINAL={'COMPLETED','FAILED'}

def safe_id(value):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,95}',value):raise ValueError('Invalid analysis or event ID')
    return value

def folder(aid):
    p=(ANALYSES/safe_id(aid)).resolve()
    if p.parent!=ANALYSES.resolve():raise ValueError('Unsafe analysis directory')
    if not p.is_dir():raise FileNotFoundError(aid)
    return p

def read(path,default=None):return json.loads(path.read_text()) if path.is_file() else default
def rows(path):return [json.loads(l) for l in path.read_text().splitlines()] if path.is_file() else []
def atomic(path,data):
    tmp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,allow_nan=False));tmp.replace(path)

def alive(pid):
    if not pid:return False
    try:os.kill(int(pid),0);return True
    except (ProcessLookupError,ValueError):return False

def update(aid,stage,**fields):
    p=folder(aid)/'application_job.json';s=read(p)
    s.update(status=stage,current_stage=stage,updated_at=datetime.now(timezone.utc).isoformat(),**fields)
    s['history']=s.get('history',[])
    if not s['history'] or s['history'][-1]['stage']!=stage:s['history'].append(dict(stage=stage,at=s['updated_at']))
    s['elapsed_seconds']=time.time()-s['created_epoch']
    if stage in TERMINAL:s['process_running']=False
    atomic(p,s);return s

def status(aid):
    p=folder(aid);s=read(p/'application_job.json')
    if s:
        pid=s.get('pid') or s.get('server_pid')
        if s['status'] not in TERMINAL and not alive(pid):s=update(aid,'FAILED',error='Worker/server interrupted; original upload retained. No automatic rerun.',process_running=False)
        if s['status'] not in TERMINAL:s['elapsed_seconds']=time.time()-s['created_epoch']
        return s
    meta=read(p/'metadata.json',{});sm=read(p/'summary.json',{});risk=read(p/'risk_summary.json',{})
    complete=(p/'events.jsonl').is_file() and bool(risk)
    return dict(analysis_id=aid,filename=meta.get('input_filename',aid),created_at=meta.get('start_timestamp'),status='COMPLETED' if complete else 'FAILED',current_stage='COMPLETED' if complete else 'FAILED',progress=100 if complete else 0,frames_processed=sm.get('frames_processed',0),total_frames=sm.get('frames_processed',meta.get('frames',0)),elapsed_seconds=sm.get('processing_seconds',0),error=None if complete else 'Not a completed application-compatible analysis',process_running=False)

def create(filename):
    aid=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:12]
    p=ANALYSES/aid;p.mkdir(parents=True,exist_ok=False)
    s=dict(analysis_id=aid,filename=filename,created_at=datetime.now(timezone.utc).isoformat(),created_epoch=time.time(),status='QUEUED',current_stage='QUEUED',progress=0,frames_processed=0,total_frames=0,elapsed_seconds=0,error=None,pid=None,server_pid=os.getpid(),process_running=False,history=[])
    atomic(p/'application_job.json',s);return aid,p

def safe_asset(aid,name):
    p=folder(aid);target=(p/name).resolve()
    allowed={'.png','.jpg','.jpeg','.webm','.mp4','.json','.jsonl','.npy'}
    if p not in target.parents or target.suffix.lower() not in allowed or any(part.startswith('.') for part in Path(name).parts):raise ValueError('Unsafe evidence path')
    if not target.is_file():raise FileNotFoundError(name)
    if target.name=='application_job.json':raise ValueError('Use status endpoint')
    return target
