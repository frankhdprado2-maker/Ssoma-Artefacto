"""Verify externally provisioned checkpoints, without copying or publishing."""
import hashlib,json,os,sys
from pathlib import Path
def verify(root):
    manifest=json.loads((Path(__file__).resolve().parents[1]/'model_manifest.json').read_text())
    for row in manifest['files']:
        p=root/row['filename']
        if not p.is_file() or p.stat().st_size!=row['size']:raise ValueError('Missing/wrong size: '+row['filename'])
        with p.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
        if digest!=row['sha256']:raise ValueError('Signature mismatch: '+row['filename'])
    return len(manifest['files'])
if __name__=='__main__':print('VERIFIED_MODEL_FILES='+str(verify(Path(sys.argv[1] if len(sys.argv)>1 else os.environ['SSOMA_MODEL_DIR']))))
