import math,subprocess,time
from pathlib import Path
from .runtime import sha,write

def filename(value):
    import re
    name=Path((value or '').replace('\\','/')).name
    name=re.sub(r'[^\w. -]','_',name,flags=re.ASCII)[:160]
    if Path(name).suffix.lower() not in {'.mp4','.mov','.avi'}:raise ValueError('Use MP4, MOV or AVI')
    return name

def validate(path,s):
    import cv2
    if path.stat().st_size==0 or path.stat().st_size>s.max_upload_mb*1024*1024:raise ValueError('Upload size quota exceeded')
    with path.open('rb') as f:head=f.read(32)
    avi=head[:4]==b'RIFF' and head[8:12]==b'AVI '
    mp4=head[4:8]==b'ftyp'
    if not (avi or mp4):raise ValueError('Unsupported real video container')
    cap=cv2.VideoCapture(str(path))
    try:
        fps=cap.get(cv2.CAP_PROP_FPS);frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));w=int(cap.get(3));h=int(cap.get(4))
        codec=int(cap.get(cv2.CAP_PROP_FOURCC));codec=''.join(chr((codec>>(i*8))&255) for i in range(4))
        if not math.isfinite(fps) or fps<=0 or frames<=0 or w<=0 or h<=0:raise ValueError('Invalid video metadata')
        duration=frames/fps
        if duration>s.max_video_seconds or frames>18000 or w>4096 or h>2160:raise ValueError('Video duration, frame or resolution quota exceeded')
        if not codec.strip('\x00'):raise ValueError('Unknown video codec')
        # Decode the complete bounded upload before admitting it: catches truncation.
        decoded=0
        while cap.read()[0]:
            decoded+=1
            if decoded>18000 or decoded/fps>s.max_video_seconds:raise ValueError('Decoded video exceeds quota')
        if decoded!=frames:raise ValueError('Incomplete or inconsistent video frame count')
        return dict(frames=frames,fps=fps,width=w,height=h,duration=duration,codec=codec,container='AVI' if avi else 'MP4/MOV')
    finally:cap.release()

def transcode(source,destination):
    import imageio_ffmpeg,cv2,json
    signature=sha(source);meta=destination.with_suffix('.json')
    if destination.is_file() and meta.is_file() and json.loads(meta.read_text()).get('source_sha256')==signature:return destination
    start=time.perf_counter();tmp=destination.with_name('encoding.mp4')
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-nostdin','-y','-v','error','-i',str(source),'-an','-c:v','libx264','-pix_fmt','yuv420p','-movflags','+faststart',str(tmp)],check=True,capture_output=True,timeout=3600)
    cap=cv2.VideoCapture(str(tmp));n=0
    while cap.read()[0]:n+=1
    cap.release();cap=cv2.VideoCapture(str(source));expected=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));cap.release()
    if not n or n!=expected or sha(source)!=signature:raise RuntimeError('Transcode integrity failure')
    tmp.replace(destination);write(meta,dict(codec='H264',frames=n,source_sha256=signature,seconds=time.perf_counter()-start))
    return destination
