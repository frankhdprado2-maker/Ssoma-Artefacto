"""Durable queue with a single process owner and notification-driven dispatch."""
import os,subprocess,sys,threading
from .db import child
from .settings import ROOT

class Queue:
    def __init__(self,store):
        self.store=store;self.wake=threading.Event();self.stop=threading.Event();self.thread=None;self.lock=None
    def start(self):
        import fcntl
        self.lock=(self.store.s.data/'worker.lock').open('a+')
        try:fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:self.lock.close();raise RuntimeError('Only one deployment server/worker may own this data directory')
        self.store.recover();self.wake.set();self.thread=threading.Thread(target=self.run,daemon=True);self.thread.start()
    def close(self):
        self.stop.set();self.wake.set()
        if self.thread:self.thread.join()
        if self.lock:self.lock.close()
    def run(self):
        while not self.stop.is_set():
            self.wake.wait();self.wake.clear()
            while not self.stop.is_set() and (aid:=self.store.claim(os.getpid())):
                p=child(self.store.s.results,aid)
                try:
                    with (p/'worker.log').open('ab') as log:
                        # Inherit lock into child so restart cannot overlap an orphan worker.
                        proc=subprocess.Popen([sys.executable,'-m','deployment.adapter_v001.worker',aid],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,pass_fds=(self.lock.fileno(),))
                        code=proc.wait()
                    if code or self.store.get(aid)['status']!='COMPLETED':self.store.update(aid,status='FAILED',error=f'Worker exited {code}; see private worker log')
                except Exception as e:self.store.update(aid,status='FAILED',error=str(e))
                print('ADAPTER_JOB_END '+aid+' '+self.store.get(aid)['status'],flush=True)
