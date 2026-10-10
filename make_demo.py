"""Create fictional fixtures only; their results have no substantive interpretation."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd

def make(out,n=800):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);rng=np.random.default_rng(41)
    cfg=json.loads((Path(__file__).parent/'config.example.json').read_text())
    first=['AARON','ANNA','BOB','CARLOS','DAVID','ELENA','FATIMA','GEORGE','HAN','ISABEL','JOHN','KAI','LI','MARIA','NOOR','OLIVER']
    last=['ALI','BROWN','CHEN','DAVIS','EVANS','GARCIA','HALL','JONES','KHAN','LAM','MILLER','NG','ORTIZ','PATEL','SMITH','TORRES']
    groups=[('W','NL'),('B','NL'),('W','HL'),('A','NL'),('I','NL'),('M','NL'),('O','NL'),('','')]
    base=pd.DataFrame(dict(ncid=[f'NC{i:06}' for i in range(n)],first_name=rng.choice(first,n),last_name=rng.choice(last,n),
                           birth_year=rng.integers(1940,1991,n),gender_code=rng.choice(['M','F'],n),
                           race_code=[groups[i%8][0] for i in range(n)],ethnic_code=[groups[i%8][1] for i in range(n)],
                           voter_status_desc=rng.choice(['ACTIVE','INACTIVE','REMOVED'],n)))
    for spec in cfg['snapshots']:
        d=base.copy()
        if spec['date'][:4] in ['2012','2019']:
            for field in ['first_name','last_name']:
                change=rng.random(n)<.35
                d.loc[change,field]=d.loc[change,field].map(lambda s:s[:-1]+'Z' if s[-1]!='Z' else s+'A')
            d.loc[10,'birth_year']+=1;d.loc[11,'gender_code']='X'
        d=pd.concat([d,d.iloc[:2],pd.DataFrame([dict(base.iloc[0],ncid='')])],ignore_index=True)
        name=spec['date']+'.csv';d.to_csv(out/name,index=False);spec['path']=name
    cfg.update(replicates=2,model_max_records=300,evaluation_max_records=300,training_nonmatch_pairs=3000,
               minimum_profile_events=2,curve_points=21,mmr_targets=[.05,.1,.2],fmr_targets=[.001,.01,.05],
               evaluation_min_per_group=20,open_distractors=100,open_nonmatch_fraction=0.10)
    (out/'config.json').write_text(json.dumps(cfg,indent=2));return out/'config.json'
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='demo_data');p.add_argument('--n',type=int,default=800);a=p.parse_args();print(make(a.output,a.n))
