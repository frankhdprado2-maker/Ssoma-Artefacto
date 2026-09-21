import gc,json,os,resource,sys,time
from .settings import Settings,select_device
from .db import Store,child
from .runtime import prepare,bound_module,write
from .video import validate,transcode

def execute(s,aid):
    store=Store(s);p=child(s.results,aid);row=store.get(aid);source=__import__('pathlib').Path(row['upload_path'])
    start=time.perf_counter();timings=[];bindings=[]
    # Required marker for the frozen inference writer's existing-directory check.
    write(p/'application_job.json',dict(row,adapter='SSOMA_DEPLOY_ADAPTER_v001'))
    try:
        device=select_device(s.device);runtime=prepare(s)
        from src.detection.inference_v001 import process_video
        from src.detection.tracking_ppe_v001 import process as tracking
        modules={}
        for name in ('event_engine_v001','fall_engine_v002','fuzzy_risk_v001','xai_evidence_v001'):
            modules[name],binding=bound_module(name,s,device,runtime);bindings.append(binding)
        write(p/'operational_bindings.json',bindings)
        def progress(frames,total):store.update(aid,frames_processed=frames,progress=5+40*frames/total)
        def association():
            if json.loads((p/'association_report.json').read_text())['status']!='TRACKING_ASSOCIATION_READY':raise ValueError('Association incomplete')
        def xai():
            if modules['xai_evidence_v001'].process(p,runtime)['status']!='XAI_EVIDENCE_READY':raise ValueError('XAI incomplete')
        operations=[
            ('VALIDATING',lambda:validate(source,s)),
            ('INFERENCE',lambda:process_video(source,runtime/'config/model_registry_v001.json',s.results,device,output_dir=p,progress_callback=progress)),
            ('TRACKING',lambda:tracking(p,runtime/'config/tracking_ppe_v001.json')),
            ('ASSOCIATION',association),
            ('EVENTS',lambda:modules['event_engine_v001'].process(p,runtime/'config/event_engine_v001.json')),
            ('POSE_FALL',lambda:modules['fall_engine_v002'].run(p,runtime,preview_path=p/'pose_preview.jpg')),
            ('FUZZY',lambda:modules['fuzzy_risk_v001'].process(p,runtime,preview_path=p/'fuzzy_preview.jpg')),
            ('XAI',xai),('FINALIZING',lambda:transcode(p/'annotated_risk.mp4',p/'web_video.mp4'))]
        for index,(name,operation) in enumerate(operations):
            store.update(aid,stage=name,progress=max(1,index*100/len(operations)))
            t=time.perf_counter();operation();gc.collect()
            timings.append(dict(stage=name,seconds=time.perf_counter()-t,peak_ram_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024/1e6))
        elapsed=time.perf_counter()-start
        write(p/'deployment_benchmark.json',dict(status='PASS',seconds=elapsed,frames=row['total_frames'],duration=row['duration'],fps=row['total_frames']/elapsed,ratio=elapsed/row['duration'],peak_ram_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024/1e6,device=device,timings=timings,modules=['PPE','HAZARD','CONTEXT','TRACKING','ASSOCIATION','EVENTS','POSE_FALL','FUZZY','XAI','FINALIZING']))
        store.complete(aid);write(p/'application_job.json',store.get(aid))
    except BaseException as error:
        store.update(aid,status='FAILED',error=f'{type(error).__name__}: {error}')
        write(p/'application_job.json',store.get(aid));raise

if __name__=='__main__':execute(Settings.env(),sys.argv[1])
