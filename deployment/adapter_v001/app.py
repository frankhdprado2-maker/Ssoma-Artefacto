import base64,hmac,json,tempfile,types
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlparse
from fastapi import FastAPI,File,HTTPException,UploadFile
from fastapi.responses import FileResponse,JSONResponse
from fastapi.staticfiles import StaticFiles
from .settings import ROOT,Settings,select_device
from .db import Store,child,safe_id
from .queue import Queue
from .runtime import configured
from .video import filename,validate

class Guard:
    def __init__(self,app,s):self.app=app;self.s=s
    async def __call__(self,scope,receive,send):
        if scope['type']!='http':return await self.app(scope,receive,send)
        headers={k.decode().lower():v.decode() for k,v in scope['headers']}
        async def reject(code,detail):
            response=JSONResponse({'detail':detail},code,headers={'WWW-Authenticate':'Basic realm="SSOMA", charset="UTF-8"'} if code==401 else {})
            await response(scope,receive,send)
        if self.s.token and scope['path']!='/health':
            auth=headers.get('authorization','');given=''
            if auth.startswith('Bearer '):given=auth[7:]
            elif auth.startswith('Basic '):
                try:given=base64.b64decode(auth[6:],validate=True).decode().split(':',1)[1]
                except (ValueError,UnicodeError,IndexError):pass
            if not hmac.compare_digest(given.encode(),self.s.token.encode()):return await reject(401,'Authentication required')
        elif not self.s.token and scope.get('client',('unknown',))[0] not in {'127.0.0.1','::1','testclient'}:
            return await reject(403,'A token is required for non-local access')
        modifying=scope['method'] not in {'GET','HEAD','OPTIONS'}
        if modifying and headers.get('origin') and urlparse(headers['origin']).netloc!=headers.get('host'):
            return await reject(403,'Cross-origin writes forbidden')
        async def secure_send(message):
            if message['type']=='http.response.start':
                message['headers']+= [(b'x-content-type-options',b'nosniff'),(b'referrer-policy',b'same-origin'),(b'cache-control',b'private, no-store')]
            await send(message)
        if not modifying:return await self.app(scope,receive,secure_send)
        limit=int(self.s.max_upload_mb*1024*1024)+1024*1024
        try:
            size=int(headers.get('content-length','0'))
            if size<0:raise ValueError()
        except ValueError:return await reject(400,'Invalid Content-Length')
        if size>limit:return await reject(413,'Upload size quota exceeded')
        # Bound actual streamed bytes even without Content-Length, before multipart parsing.
        with tempfile.SpooledTemporaryFile(max_size=1024*1024) as body:
            total=0
            while True:
                message=await receive()
                if message['type']=='http.disconnect':return
                chunk=message.get('body',b'');total+=len(chunk)
                if total>limit:return await reject(413,'Upload size quota exceeded')
                body.write(chunk)
                if not message.get('more_body'):break
            body.seek(0)
            async def replay():
                chunk=body.read(65536)
                return {'type':'http.request','body':chunk,'more_body':body.tell()<total}
            await self.app(scope,replay,secure_send)

def create_app(s=None,start_worker=True):
    s=s or Settings.env();db=Store(s);queue=Queue(db)
    # Separate module instance: no monkeypatch of frozen web.services.store.
    def read(path,default=None):return json.loads(path.read_text()) if path.is_file() else default
    def rows(path):return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.is_file() else []
    def folder(aid):
        try:db.get(aid);return child(s.results,aid)
        except FileNotFoundError:
            if s.history:
                p=child(s.history,aid)
                if p.is_dir():return p
            raise
    def status(aid):
        try:
            row=db.get(aid)
            # Approved RC1 UI knows FAILED, not the operational recovery state.
            if row['status']=='RECOVERY_REQUIRED':row.update(status='FAILED',operational_status='RECOVERY_REQUIRED')
            return row
        except FileNotFoundError:
            p=folder(aid);meta=read(p/'metadata.json',{});summary=read(p/'summary.json',{})
            return dict(analysis_id=aid,filename=Path(meta.get('input_path','historical')).name,status='COMPLETED',current_stage='COMPLETED',progress=100,frames_processed=summary.get('frames_processed',0),total_frames=summary.get('frames_processed',0),duration=meta.get('duration'),error=None,read_only=True)
    cache=s.data/'presentation';cache.mkdir(exist_ok=True)
    proxy=types.SimpleNamespace(folder=folder,status=status,read=read,rows=rows,safe_id=safe_id,CACHE=cache)
    from web.services import presentation as original
    presentation=types.ModuleType('deployment.adapter_v001.presentation')
    presentation.__dict__.update(original.__dict__)
    # Rebind only presentation functions to an isolated namespace and operational store.
    for name,value in vars(original).items():
        if isinstance(value,types.FunctionType) and value.__module__==original.__name__:
            presentation.__dict__[name]=types.FunctionType(value.__code__,presentation.__dict__,name,value.__defaults__,value.__closure__)
    presentation.store=proxy;presentation.locks={}
    @asynccontextmanager
    async def lifespan(app):
        if start_worker:queue.start()
        try:yield
        finally:
            if start_worker:queue.close()
    app=FastAPI(lifespan=lifespan,docs_url=None,redoc_url=None,openapi_url=None)
    app.add_middleware(Guard,s=s);app.state.store=db;app.state.queue=queue
    health={'status':'ok' if configured(s) else 'models_missing','release_id':'SSOMA_APPLICATION_RC1+SSOMA_DEPLOY_ADAPTER_v001','device':select_device(s.device) if s.device!='cpu' else 'cpu','models_configured':configured(s)}
    @app.exception_handler(ValueError)
    async def invalid(request,error):return JSONResponse({'detail':str(error)},400)
    @app.exception_handler(FileNotFoundError)
    async def missing(request,error):return JSONResponse({'detail':'Analysis or evidence not found'},404)
    @app.get('/health')
    def health_route():return JSONResponse(health,status_code=200 if health['models_configured'] else 503)
    @app.get('/api/analyses')
    def analyses():
        result=[status(r['analysis_id']) for r in db.list()]
        if s.history:
            known={r['analysis_id'] for r in result}
            for p in s.history.iterdir():
                if p.is_dir() and p.name not in known and (p/'metadata.json').is_file():result.append(status(p.name))
        for row in result:
            p=folder(row['analysis_id'])
            row['events']=len(rows(p/'events.jsonl')) if row['status']=='COMPLETED' else 0
            row['highest_risk']=max((r['risk_score'] for r in rows(p/'risk_events.jsonl')),default=0) if row['status']=='COMPLETED' else 0
        return result
    @app.post('/api/analyses',status_code=202)
    def upload(file:UploadFile=File(...)):
        aid=None
        try:
            name=filename(file.filename)
            try:aid=db.reserve(name)
            except ValueError as e:raise HTTPException(429,str(e)) from e
            dest=child(s.uploads,aid)/('original'+Path(name).suffix.lower());size=0
            with dest.open('xb') as out:
                while chunk:=file.file.read(1024*1024):
                    size+=len(chunk)
                    if size>s.max_upload_mb*1024*1024:raise HTTPException(413,'Upload size quota exceeded')
                    out.write(chunk)
            meta=validate(dest,s)
            db.update(aid,status='QUEUED',stage='QUEUED',upload_path=str(dest),duration=meta['duration'],total_frames=meta['frames'])
            queue.wake.set()
            return dict(analysis_id=aid,status='QUEUED',status_url=f'/api/analyses/{aid}/status')
        except Exception as e:
            if aid:
                db.update(aid,status='FAILED',error=str(e));db.delete(aid)
            if isinstance(e,HTTPException):raise
            raise HTTPException(422,str(e)) from e
        finally:file.file.close()
    @app.delete('/api/analyses/{aid}')
    def delete(aid:str):db.delete(aid);return {'deleted':aid}
    @app.get('/api/analyses/{aid}/status')
    def get_status(aid:str):return status(aid)
    @app.get('/api/analyses/{aid}/data')
    def data(aid:str):return presentation.dataset(aid)
    @app.get('/api/analyses/{aid}/events')
    def events(aid:str):return presentation.dataset(aid)['events']
    @app.get('/api/analyses/{aid}/events/{eid}')
    def event(aid:str,eid:str):return presentation.detail(aid,eid)
    @app.get('/api/analyses/{aid}/events/{eid}/frame/{kind}')
    def frame(aid:str,eid:str,kind:str):return FileResponse(presentation.frame(aid,eid,kind),media_type='image/png')
    @app.get('/api/analyses/{aid}/video')
    def video(aid:str):
        p=folder(aid)
        if (p/'web_video.mp4').is_file():return FileResponse(p/'web_video.mp4',media_type='video/mp4')
        # Historical reads never rewrite old output; reuse prebuilt web video only.
        if (p/'dashboard_preview_v001.webm').is_file():return FileResponse(p/'dashboard_preview_v001.webm',media_type='video/webm')
        raise FileNotFoundError('Web video not ready')
    @app.get('/api/analyses/{aid}/assets/{name:path}')
    def asset(aid:str,name:str):
        p=folder(aid);target=(p/name).resolve()
        if p.resolve() not in target.parents or any(x.startswith('.') for x in Path(name).parts) or target.suffix.lower() not in {'.png','.jpg','.jpeg','.mp4','.webm','.npy'}:raise ValueError('Unsafe evidence path')
        if not target.is_file():raise FileNotFoundError(name)
        return FileResponse(target)
    app.mount('/static/branding',StaticFiles(directory=ROOT/'branding'),name='branding')
    app.mount('/static',StaticFiles(directory=ROOT/'web/static'),name='static')
    @app.get('/')
    @app.get('/monitor')
    @app.get('/analysis/{aid}/events')
    @app.get('/analysis/{aid}/dashboard')
    def home(aid:str|None=None):
        if aid:folder(aid)
        return FileResponse(ROOT/'web/templates/index.html',media_type='text/html')
    @app.get('/favicon.ico')
    def favicon():return FileResponse(ROOT/'branding/ssoma_icon.png' if (ROOT/'branding/ssoma_icon.png').is_file() else ROOT/'branding/placeholder.svg')
    @app.get('/branding/logo')
    def logo():return FileResponse(ROOT/'branding/ssoma_logo.png' if (ROOT/'branding/ssoma_logo.png').is_file() else ROOT/'branding/placeholder.svg')
    return app

def factory():return create_app()
