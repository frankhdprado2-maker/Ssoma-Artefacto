import argparse,json,os
from .settings import Settings
from .db import Store

def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['serve','cleanup']);parser.add_argument('--apply',action='store_true')
    args=parser.parse_args()
    if args.action=='cleanup':print(json.dumps(Store(Settings.env()).cleanup(dry_run=not args.apply)))
    else:
        import uvicorn
        uvicorn.run('deployment.adapter_v001.app:factory',factory=True,host=os.getenv('SSOMA_BIND','127.0.0.1'),port=int(os.getenv('PORT','8000')),workers=1,proxy_headers=False)

if __name__=='__main__':main()
