import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

@dataclass(frozen=True)
class Settings:
    data: Path
    models: Path
    uploads: Path
    results: Path
    device: str = 'cpu'
    max_upload_mb: float = 25
    max_video_seconds: float = 10
    retention_hours: float = 24
    max_analyses: int = 10
    max_queued: int = 2
    token: str = ''
    history: Path | None = None

    @classmethod
    def env(cls):
        def path(key, default):
            value = os.getenv(key, str(default))
            if '\\' in value or (len(value)>1 and value[1]==':'):
                raise ValueError(f'{key} must be a native absolute path')
            p = Path(value).expanduser()
            if not p.is_absolute(): p = ROOT / p
            return p.resolve()
        data = path('SSOMA_DATA_DIR', ROOT/'data')
        s = cls(data, path('SSOMA_MODEL_DIR', ROOT/'models'),
                path('SSOMA_UPLOAD_DIR', data/'uploads'),
                path('SSOMA_RESULT_DIR', data/'results'),
                os.getenv('SSOMA_DEVICE','auto'),
                float(os.getenv('SSOMA_MAX_UPLOAD_MB','25')),
                float(os.getenv('SSOMA_MAX_VIDEO_SECONDS','10')),
                float(os.getenv('SSOMA_RETENTION_HOURS','24')),
                int(os.getenv('SSOMA_MAX_ANALYSES','10')),
                int(os.getenv('SSOMA_MAX_QUEUED_ANALYSES','2')),
                os.getenv('SSOMA_ACCESS_TOKEN',''),
                path('SSOMA_HISTORY_DIR', '/') if os.getenv('SSOMA_HISTORY_DIR') else None)
        s.validate()
        return s

    def validate(self):
        import math
        if self.device not in {'auto','cpu','cuda','rocm','local','rocm/local'}: raise ValueError('Invalid SSOMA_DEVICE')
        if any(not math.isfinite(v) or v<=0 for v in (self.max_upload_mb,self.max_video_seconds,self.retention_hours,self.max_analyses,self.max_queued)):
            raise ValueError('Limits must be finite and positive')
        if os.getenv('SSOMA_MAX_ACTIVE_ANALYSES','1')!='1': raise ValueError('Only one active analysis is supported')
        # Writable directories can never contain frozen project files or models/history.
        protected=[ROOT/'src',ROOT/'config',ROOT/'models',ROOT/'scripts',ROOT/'deploy',ROOT/'branding',ROOT/'results',ROOT/'reports',ROOT/'web',self.models]
        if self.history: protected.append(self.history)
        for writable in (self.data,self.uploads,self.results):
            for p in protected:
                if writable==p or writable in p.parents or p in writable.parents:
                    raise ValueError('Writable deployment directory overlaps protected inputs')
        if self.uploads==self.results or self.uploads in self.results.parents or self.results in self.uploads.parents:
            raise ValueError('Uploads and results must be separate directories')

    def prepare(self):
        self.validate()
        for p in (self.data,self.uploads,self.results): p.mkdir(parents=True,exist_ok=True)

def select_device(requested):
    import torch
    available=torch.cuda.is_available()
    hip=bool(torch.version.hip)
    if requested=='cpu': return 'cpu'
    if requested=='auto': return '0' if available else 'cpu'
    if requested=='cuda' and available and not hip: return '0'
    if requested in {'rocm','local','rocm/local'} and available and hip: return '0'
    raise ValueError(f'Requested device {requested} is not supported by this runtime')
