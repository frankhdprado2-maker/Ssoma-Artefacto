"""Validate the public source distribution without loading or requiring weights."""
import ast,csv,hashlib,json,sys
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    rows=list(csv.DictReader((ROOT/'distribution_manifest.csv').open(encoding='utf-8',newline='')))
    names={x['path'] for x in rows}
    assert len(names)==len(rows)
    for x in rows:
        p=ROOT/x['path'];assert p.is_file() and not p.is_symlink(),x['path']
        assert p.resolve().is_relative_to(ROOT)
        assert p.stat().st_size==int(x['size']),x['path']
        if x['path']=='distribution_manifest.csv':assert x['sha256']=='SELF_REFERENCE_EXCLUDED'
        else:assert sha(p)==x['sha256'],x['path']
        assert p.stat().st_size<100_000_000
        assert p.suffix not in ('.pt','.pth','.onnx','.pem','.key','.p12','.pfx','.db','.sqlite','.mp4','.log')
        if p.suffix=='.py':ast.parse(p.read_text(encoding='utf-8-sig'),filename=x['path'])
    for name in ('README.md','VERSION.json','.gitignore','THIRD_PARTY_NOTICES.md','scripts/self_check.py','models/model_manifest.json','data/uploads/.gitkeep','data/results/.gitkeep'):assert name in names,name
    v=json.loads((ROOT/'VERSION.json').read_text(encoding='utf-8'))
    for k in ('scientific_hashes','ui_hashes'):
        for name,digest in v[k].items():assert sha(ROOT/name)==digest,name
    assert sha(ROOT/'models/model_manifest.json')==v['model_manifest_sha256']
    assert sha(ROOT/'config/deploy_model_registry_v001.json')==v['registry_sha256']
    models=json.loads((ROOT/'models/model_manifest.json').read_text())
    assert len(models)==8
    for m in models:
        assert not m['public_file'] and m['distribution_status']=='DISTRIBUTION_REQUIRED_LOCALLY'
        assert 'models/'+m['filename'] not in names
        assert len(m['sha256'])==64 and m['size']>0
    for name in ('uploads','results'):
        assert {p.name for p in (ROOT/'data'/name).iterdir()}=={'.gitkeep'}
    print('PUBLIC_REPO_CHECK PASS; structure/syntax/manifests/frozen hashes/exclusions PASS; models not evaluated')
if __name__=='__main__':main()
