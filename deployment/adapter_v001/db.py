import json,re,secrets,sqlite3,time
from contextlib import contextmanager
from pathlib import Path

TERMINAL={'COMPLETED','FAILED','RECOVERY_REQUIRED'}
def safe_id(value):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,95}',value): raise ValueError('Invalid analysis ID')
    return value

def child(root, aid):
    p=root/safe_id(aid)
    if p.is_symlink() or p.resolve().parent!=root.resolve(): raise ValueError('Unsafe analysis directory')
    return p

class Store:
    def __init__(self,s):
        self.s=s;s.prepare();self.path=s.data/'jobs.sqlite3'
        with self.connect() as db:
            db.executescript('''PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS analyses (
              analysis_id TEXT PRIMARY KEY, filename TEXT NOT NULL,created_at REAL NOT NULL,
              status TEXT NOT NULL,stage TEXT NOT NULL,progress REAL NOT NULL DEFAULT 0,
              result_path TEXT NOT NULL,upload_path TEXT,error TEXT,expires_at REAL NOT NULL,
              duration REAL,total_frames INTEGER DEFAULT 0,frames_processed INTEGER DEFAULT 0);
            CREATE TABLE IF NOT EXISTS jobs (analysis_id TEXT PRIMARY KEY REFERENCES analyses ON DELETE CASCADE,
              pid INTEGER,started_at REAL,finished_at REAL);
            CREATE TABLE IF NOT EXISTS artifacts (analysis_id TEXT REFERENCES analyses ON DELETE CASCADE,
              relative_path TEXT,bytes INTEGER,PRIMARY KEY(analysis_id,relative_path));''')

    @contextmanager
    def connect(self):
        db=sqlite3.connect(self.path,timeout=30);db.row_factory=sqlite3.Row
        db.execute('PRAGMA foreign_keys=ON')
        try:
            with db:yield db
        finally:db.close()

    @contextmanager
    def transaction(self):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE');yield db

    def reserve(self,filename):
        with self.transaction() as db:
            count=db.execute('SELECT count(*) FROM analyses').fetchone()[0]
            queued=db.execute("SELECT count(*) FROM analyses WHERE status IN ('UPLOADING','QUEUED')").fetchone()[0]
            if count>=self.s.max_analyses: raise ValueError('Analysis storage quota exceeded')
            if queued>=self.s.max_queued: raise ValueError('Queue quota exceeded')
            aid=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'_'+secrets.token_hex(6)
            p=child(self.s.results,aid);now=time.time()
            db.execute('INSERT INTO analyses(analysis_id,filename,created_at,status,stage,result_path,expires_at) VALUES (?,?,?,?,?,?,?)',
                       (aid,filename,now,'UPLOADING','VALIDATING',str(p),now+self.s.retention_hours*3600))
            db.execute('INSERT INTO jobs(analysis_id) VALUES (?)',(aid,))
        p.mkdir();child(self.s.uploads,aid).mkdir()
        return aid

    def update(self,aid,**values):
        allowed={'status','stage','progress','upload_path','error','duration','total_frames','frames_processed'}
        if not values or not set(values)<=allowed: raise ValueError('Invalid state fields')
        with self.transaction() as db:
            db.execute('UPDATE analyses SET '+','.join(k+'=?' for k in values)+' WHERE analysis_id=?',(*values.values(),safe_id(aid)))

    def get(self,aid):
        with self.connect() as db:r=db.execute('SELECT * FROM analyses WHERE analysis_id=?',(safe_id(aid),)).fetchone()
        if not r: raise FileNotFoundError(aid)
        r=dict(r);r['current_stage']=r['stage'];r['process_running']=r['status']=='RUNNING'
        return r

    def list(self):
        with self.connect() as db:ids=[r[0] for r in db.execute('SELECT analysis_id FROM analyses ORDER BY created_at DESC')]
        return [self.get(a) for a in ids]

    def recover(self):
        # Caller owns the process lock: no live worker can be mistaken for a crash.
        with self.transaction() as db:
            n=db.execute("UPDATE analyses SET status='RECOVERY_REQUIRED',stage='RECOVERY_REQUIRED',error='Interrupted operation; manual review required' WHERE status IN ('RUNNING','UPLOADING','DELETING')").rowcount
            return n

    def claim(self,pid):
        with self.transaction() as db:
            if db.execute("SELECT 1 FROM analyses WHERE status='RUNNING'").fetchone(): return None
            row=db.execute("SELECT analysis_id FROM analyses WHERE status='QUEUED' ORDER BY created_at LIMIT 1").fetchone()
            if not row:return None
            aid=row[0]
            db.execute("UPDATE analyses SET status='RUNNING',stage='VALIDATING' WHERE analysis_id=?",(aid,))
            db.execute('UPDATE jobs SET pid=?,started_at=? WHERE analysis_id=?',(pid,time.time(),aid))
            return aid

    def complete(self,aid):
        p=child(self.s.results,aid)
        with self.transaction() as db:
            db.execute("UPDATE analyses SET status='COMPLETED',stage='COMPLETED',progress=100,frames_processed=total_frames WHERE analysis_id=?",(aid,))
            db.execute('UPDATE jobs SET finished_at=? WHERE analysis_id=?',(time.time(),aid))
            db.executemany('INSERT OR REPLACE INTO artifacts VALUES (?,?,?)',[(aid,str(f.relative_to(p)),f.stat().st_size) for f in p.rglob('*') if f.is_file() and not f.is_symlink()])

    def cleanup(self,dry_run=True):
        ids=[r['analysis_id'] for r in self.list() if r['expires_at']<time.time() and r['status'] in TERMINAL]
        if not dry_run:
            for aid in ids:self.delete(aid)
        return {'dry_run':dry_run,'analysis_ids':ids}

    def delete(self,aid):
        import shutil
        with self.transaction() as db:
            r=db.execute('SELECT status FROM analyses WHERE analysis_id=?',(safe_id(aid),)).fetchone()
            if not r:raise FileNotFoundError(aid)
            if r[0] not in TERMINAL:raise ValueError('Cannot delete an active or queued analysis')
            paths=[child(root,aid) for root in (self.s.results,self.s.uploads,self.s.data/'presentation')]
            for p in paths:
                if any(f.is_symlink() for f in p.rglob('*')):raise ValueError('Refusing deletion of symlink-containing analysis')
            db.execute("UPDATE analyses SET status='DELETING' WHERE analysis_id=?",(aid,))
        # Only validated, direct, deployment-owned children; never an arbitrary supplied path.
        for p in paths:
            if p.exists():shutil.rmtree(p)
        with self.transaction() as db:db.execute('DELETE FROM analyses WHERE analysis_id=?',(aid,))
