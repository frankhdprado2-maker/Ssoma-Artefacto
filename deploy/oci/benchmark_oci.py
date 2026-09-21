"""Exact authorized fixture; one completion notification, no status polling.

Isolated benchmark admission permits 2.01s because the frozen fixture is
2.002002s. The public DEMO_TINY remains 2.0s. Scientific configuration is identical.
"""
import argparse,json,os,platform,subprocess,sys,time,urllib.request,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from deployment.adapter_v001.runtime import sha,write

def summarize(p):
    def rows(name):return [json.loads(x) for x in (p/name).read_text().splitlines() if x]
    fields=('track_id','event_type','subject_class','start_timestamp','end_timestamp','peak_frame','supporting_frames')
    return dict(detections=len(rows('detections.jsonl')),tracks=len({r['track_id'] for r in rows('tracks.jsonl')}),
      events=[{k:r[k] for k in fields} for r in rows('events.jsonl')],
      risks=[{k:r[k] for k in ('track_id','risk_score','risk_level','inputs')} for r in rows('risk_events.jsonl')],
      evidence_valid=all((p/r['evidence_frame']).is_file() for r in rows('events.jsonl')),
      xai_status=json.loads((p/'xai_evidence_summary.json').read_text())['status'])

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--clip',type=Path,required=True);parser.add_argument('--reference',type=Path,required=True);parser.add_argument('--ui',action='store_true');parser.add_argument('--port',type=int,default=8774);args=parser.parse_args()
    if platform.machine() not in {'aarch64','arm64'}:raise RuntimeError('Native ARM64 benchmark required')
    reference=json.loads(args.reference.read_text())
    if sha(args.clip)!=reference['fixture_sha256']:raise ValueError('Not the exact authorized 60-frame fixture')
    if not os.getenv('SSOMA_ACCESS_TOKEN'):raise ValueError('External access token required')
    base=Path(os.environ['SSOMA_DATA_DIR'])/'oci_benchmarks'/uuid.uuid4().hex;base.mkdir(parents=True)
    env=dict(os.environ,SSOMA_DATA_DIR=str(base/'state'),SSOMA_UPLOAD_DIR=str(base/'uploads'),SSOMA_RESULT_DIR=str(base/'results'),SSOMA_DEVICE='cpu',SSOMA_BIND='127.0.0.1',SSOMA_MAX_VIDEO_SECONDS='2.01',SSOMA_MAX_UPLOAD_MB='1',PORT=str(args.port))
    env.pop('SSOMA_HISTORY_DIR',None)
    verified=subprocess.run([sys.executable,str(ROOT/'deploy/oci/verify_arm64.py'),'--models','--output',str(base/'native_prerequisites.json')],env=env,cwd=ROOT,capture_output=True,text=True)
    if verified.returncode:raise RuntimeError('Native prerequisite validation failed; see '+str(base))
    def request(path,method='GET',data=None,headers=None):
        h={'Authorization':'Bearer '+env['SSOMA_ACCESS_TOKEN']};h.update(headers or {})
        with urllib.request.urlopen(urllib.request.Request(f'http://127.0.0.1:{args.port}'+path,data=data,headers=h,method=method),timeout=60) as r:return r.status,r.read()
    server=subprocess.Popen([sys.executable,'-m','deployment.adapter_v001.cli','serve'],cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    lines=[]
    try:
        for line in server.stdout:
            lines.append(line)
            if 'Uvicorn running on' in line:break
        else:raise RuntimeError('Server did not start')
        assert request('/health')[0]==200
        if args.ui:
            print(f'Open http://127.0.0.1:{args.port} through SSH forwarding; authenticate with the external token and upload the exact fixture. Waiting for one completed job.',flush=True)
            expected=None
        else:
            boundary='SSOMA'+uuid.uuid4().hex
            body=(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="authorized_clip.mp4"\r\nContent-Type: video/mp4\r\n\r\n'.encode()+args.clip.read_bytes()+f'\r\n--{boundary}--\r\n'.encode())
            code,body=request('/api/analyses','POST',body,{'Content-Type':'multipart/form-data; boundary='+boundary});assert code==202
            expected=json.loads(body)['analysis_id']
        for line in server.stdout:
            lines.append(line)
            if line.startswith('ADAPTER_JOB_END '):
                _,aid,status=line.split()
                if expected and aid!=expected:continue
                if status!='COMPLETED':raise RuntimeError('Pipeline failed; inspect private worker log')
                break
        else:raise RuntimeError('No completion received')
        p=base/'results'/aid
        assert json.loads((p/'metadata.json').read_text())['input_sha256']==reference['fixture_sha256']
        b=json.loads((p/'deployment_benchmark.json').read_text());assert b['frames']==60
        current=summarize(p);equivalent=current==reference['summary']
        # Exercise positive EigenCAM too: LOW risks legitimately omit network CAM.
        subprocess.run([sys.executable,str(ROOT/'deploy/oci/verify_xai.py'),str(p),str(base/'positive_xai')],cwd=ROOT,env=env,check=True)
        write(base/'oci_cpu_e2e_benchmark.json',dict(b,ARM64_RUNTIME_VERIFIED=True,architecture=platform.machine(),semantic_fixture_equivalence=equivalent,x86_reference_seconds=43.96819943,x86_reference_fps=1.3646226313,x86_reference_ratio=21.9621156153,performance_ratio_to_x86=b['seconds']/43.96819943,admission_exception_seconds=2.01,public_demo_seconds=2,ui_exercised=args.ui,summary=current))
        import csv
        with (base/'stage_timings.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=['stage','seconds','peak_ram_mb']);writer.writeheader();writer.writerows(b['timings'])
        print(json.dumps({'benchmark':str(base/'oci_cpu_e2e_benchmark.json'),'ARM64_RUNTIME_VERIFIED':True,'equivalence':equivalent}))
        if not equivalent:raise RuntimeError('Output differences require review before public demo; do not change thresholds')
        write(Path(os.environ['SSOMA_DATA_DIR'])/'oci_ready.json',dict(ARM64_RUNTIME_VERIFIED=True,semantic_fixture_equivalence=True,benchmark=str(base/'oci_cpu_e2e_benchmark.json'),registry_sha256=sha(ROOT/'config/deploy_model_registry_v001.json')))
    finally:
        server.terminate()
        try:server.wait(timeout=180)
        except subprocess.TimeoutExpired:server.kill();server.wait()
        (base/'server.log').write_text(''.join(lines))

if __name__=='__main__':main()
