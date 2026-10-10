"""Explicit supervised Fellegi–Sunter comparators and held-out threshold evaluation.
Scores are ranking scores, NOT calibrated posterior match probabilities.
"""
from collections import Counter
import itertools
import numpy as np
import pandas as pd
from rapidfuzz.distance import JaroWinkler, Levenshtein
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from audit import FIELDS

MODELS=['jw','jw_tf','lev','lev_tf','feature_surname','hybrid']
FEATURE_NAMES=['length','letters','terms','vowel_fraction','unique_fraction',
               'hyphens','apostrophes','longest_vowel_run','longest_consonant_run',
               'ends_in_vowel','log_training_prefix_count','log_training_name_count']

def runs(s, vowels=True):
    return max([len(list(g)) for key,g in itertools.groupby(s,lambda c:c in 'AEIOUY') if key==vowels]+[0])

def features(names, prefix_counts, name_counts):
    values=[]
    for s in names:
        letters=''.join(c for c in s if c.isalpha()); n=max(1,len(letters))
        values.append([len(s),len(letters),len(s.split()),sum(c in 'AEIOUY' for c in letters)/n,
                       len(set(letters))/n,s.count('-'),s.count("'"),runs(letters),runs(letters,False),
                       int(s[-1:] in list('AEIOUY')),np.log1p(prefix_counts.get(s[:3],0)),np.log1p(name_counts.get(s,0))])
    return np.asarray(values,float)

def string_levels(a,b,kind):
    if kind=='jw':
        sim=np.fromiter((JaroWinkler.similarity(x,y) for x,y in zip(a,b)),float,len(a))
        return np.select([sim==1,sim>=.9,sim>=.7],[0,1,2],default=3).astype(int)
    dist=np.fromiter((Levenshtein.distance(x,y) for x,y in zip(a,b)),int,len(a))
    # Exact equality is its own level; otherwise term-frequency adjustment is invalid.
    return np.minimum(dist,4)

class Comparators:
    def __init__(self, cfg): self.cfg=cfg
    def fit(self, reference, pairs, freq_reference=None):
        self.freq={}; self.counts={}; self.prefix={}; self.pca={}; self.scalers={}; self.cutoffs={}
        self.n_train=len(reference)
        fr=reference if freq_reference is None else freq_reference
        self.n_freq=len(fr)
        for field in FIELDS:
            # v2: counts may come from the full eligible frame (frequency_reference=full_frame).
            self.counts[field]=Counter(fr[field]); self.prefix[field]=Counter(fr[field].str[:3])
            # Add-one smoothing includes an unseen-term category, frozen before 2022.
            denom=len(fr)+len(self.counts[field])+1
            self.freq[field]={n:(v+1)/denom for n,v in self.counts[field].items()}
            self.freq[field]['__unseen__']=1/denom
            X=features(reference[field],self.prefix[field],self.counts[field])
            self.scalers[field]=StandardScaler().fit(X)
            k=min(self.cfg['pca_components'],X.shape[1],len(X)-1)
            if k<1: raise ValueError('Too few records for PCA')
            self.pca[field]=PCA(n_components=k,svd_solver='full').fit(self.scalers[field].transform(X))
        # Synthetic positive pairs are generated from a disjoint 2022 training
        # sample; run.py enforces NCID separation from application evaluation.
        a=pairs[[f+'_a' for f in FIELDS]].rename(columns={f+'_a':f for f in FIELDS})
        b=pairs[[f+'_b' for f in FIELDS]].rename(columns={f+'_b':f for f in FIELDS})
        ta,tb=self.transform(a),self.transform(b)
        for field in FIELDS:
            changed=a[field].to_numpy()!=b[field].to_numpy()
            delta=np.abs(ta[field+'_pc']-tb[field+'_pc'])
            if not changed.any(): raise ValueError(f'No historical {field} changes for model bands')
            self.cutoffs[field]=[np.unique(np.quantile(delta[changed,j],[.05,.1,.25,.5])) for j in range(delta.shape[1])]
        rng=np.random.default_rng(self.cfg['seed'])
        n=min(self.cfg['training_nonmatch_pairs'],len(reference)*(len(reference)-1))
        i=rng.integers(len(reference),size=n); j=rng.integers(len(reference)-1,size=n); j+=j>=i
        ref_t=self.transform(reference)
        cm=self.compare(ta,tb,np.arange(len(a)),np.arange(len(b)))
        cu=self.compare(ref_t,ref_t,i,j)
        self.weights={}; self.u={}
        for key in cm['levels']:
            levels=max(int(cm['levels'][key].max()),int(cu['levels'][key].max()))+1
            # Also allocate possible but unobserved categories.
            levels=max(levels,4 if key.startswith('jw') else 5 if key.startswith('lev') else 6)
            m=np.bincount(cm['levels'][key],minlength=levels)+.5
            u=np.bincount(cu['levels'][key],minlength=levels)+.5
            m=m/m.sum();u=u/u.sum();self.u[key]=u;self.weights[key]=np.log2(m/u)
        # Hybrid uses a single regularised classifier, avoiding naive addition of
        # correlated spelling and structure evidence. No ethnicity/ID predictors.
        X=np.vstack([self.hybrid_x(cm),self.hybrid_x(cu)])
        y=np.r_[np.ones(len(a)),np.zeros(n)]
        self.hybrid=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=2000,random_state=self.cfg['seed']))
        self.hybrid.fit(X,y)
        return self
    def transform(self, d):
        result={}
        for f in FIELDS:
            result[f]=d[f].to_numpy(str)
            X=features(d[f],self.prefix[f],self.counts[f])
            result[f+'_pc']=self.pca[f].transform(self.scalers[f].transform(X))
        return result
    def compare(self,a,b,i,j):
        levels={};delta={};tf={}
        for f in FIELDS:
            x,y=a[f][i],b[f][j]
            levels['jw_'+f]=string_levels(x,y,'jw'); levels['lev_'+f]=string_levels(x,y,'lev')
            delta[f]=np.abs(a[f+'_pc'][i]-b[f+'_pc'][j])
            for k,cuts in enumerate(self.cutoffs[f]):
                # Explicit zero level, followed by nonzero empirical bands.
                levels[f+'_pc'+str(k)]=np.where(delta[f][:,k]<=1e-12,0,1+np.searchsorted(cuts,delta[f][:,k],side='right'))
            freq=self.freq[f];tf[f]=np.where(x==y,-np.log2([freq.get(v,freq['__unseen__']) for v in x]),0)
        return dict(levels=levels,delta=delta,tf=tf)
    def hybrid_x(self,c):
        parts=[]
        for f in FIELDS:
            parts += [np.eye(4)[c['levels']['jw_'+f]], c['tf'][f][:,None], c['delta'][f]]
        return np.column_stack(parts)
    def score(self,c):
        s={}
        for kind in ['jw','lev']:
            s[kind]=sum(self.weights[kind+'_'+f][c['levels'][kind+'_'+f]] for f in FIELDS)
            # For exact comparisons replace u_exact with the term probability.
            s[kind+'_tf']=s[kind].copy()
            for f in FIELDS:
                exact=c['levels'][kind+'_'+f]==0
                s[kind+'_tf']+=np.where(exact,np.log2(self.u[kind+'_'+f][0])+c['tf'][f],0)
        f='last_name'; key='jw_'+f; exact=c['levels'][key]==0
        s['feature_surname']=sum(self.weights[k][v] for k,v in c['levels'].items() if k.startswith('first_name_pc'))
        s['feature_surname']+=self.weights[key][c['levels'][key]]+np.where(exact,np.log2(self.u[key][0])+c['tf'][f],0)
        s['hybrid']=self.hybrid.decision_function(self.hybrid_x(c))
        return s

def candidates(reference,query,blocking,max_pairs,chunk_pairs):
    # Primary blocking does not use names. Sensitivity reproduces the old prefix rule.
    keys=['birth_year','gender']
    a=reference.copy();b=query.copy()
    if blocking=='legacy_prefix':
        a['prefix']=a.first_name.str[:3];b['prefix']=b.first_name.str[:3];keys+=['prefix']
    elif blocking!='demographic': raise ValueError(blocking)
    ag=a.groupby(keys,dropna=False,sort=True).indices;bg=b.groupby(keys,dropna=False,sort=True).indices
    total=sum(len(q)*len(ag.get(k,[])) for k,q in bg.items())
    if total>max_pairs:
        raise ValueError(f'{total:,} candidate pairs exceed max_candidate_pairs={max_pairs:,}. '
                         'Increase the explicit limit or reduce cohort size; no pairs were silently truncated.')
    for key,qidx in bg.items():
        ridx=ag.get(key,np.array([],dtype=int))
        if not len(ridx):continue
        batch=max(1,chunk_pairs//len(ridx))
        for start in range(0,len(qidx),batch):
            qs=qidx[start:start+batch]
            yield np.repeat(qs,len(ridx)),np.tile(ridx,len(qs))

def link(reference,query,model,blocking,cfg):
    n=len(query); rt=model.transform(reference); qt=model.transform(query)
    best={m:np.full(n,-np.inf) for m in MODELS}; chosen={m:np.full(n,-1,int) for m in MODELS}
    ntie={m:np.zeros(n,int) for m in MODELS};nc=np.zeros(n,int);truth=np.zeros(n,bool)
    for qi,ri in candidates(reference,query,blocking,cfg['max_candidate_pairs'],cfg['chunk_pairs']):
        nc+=np.bincount(qi,minlength=n)
        same=query.ncid.to_numpy()[qi]==reference.ncid.to_numpy()[ri]
        truth[np.unique(qi[same])]=True
        scores=model.score(model.compare(qt,rt,qi,ri))
        # Each query occurs in one block and one batch. All its candidates stay together.
        uq,start,count=np.unique(qi,return_index=True,return_counts=True)
        for m,s in scores.items():
            for q,st,ct in zip(uq,start,count):
                vals=s[st:st+ct]; top=vals.max(); tie=np.flatnonzero(np.isclose(vals,top,rtol=0,atol=1e-10))
                best[m][q]=top;ntie[m][q]=len(tie);chosen[m][q]=ri[st+tie[0]]
    result={}
    for m in MODELS:
        accepted=(ntie[m]==1)&(chosen[m]>=0)
        # At most one query per reference: highest score wins; equal maxima abstain.
        # No second-choice reassignment. This policy is fixed across every model.
        for rid,ix in pd.Series(np.flatnonzero(accepted)).groupby(chosen[m][accepted]):
            qs=ix.to_numpy();vals=best[m][qs]; win=qs[np.isclose(vals,vals.max(),rtol=0,atol=1e-10)]
            accepted[qs]=False
            if len(win)==1:accepted[win[0]]=True
        r=chosen[m]; correct=np.zeros(n,bool);good=r>=0
        correct[good]=query.ncid.to_numpy()[good]==reference.ncid.to_numpy()[r[good]]
        matchable=query.ncid.isin(set(reference.ncid)).to_numpy()
        result[m]=pd.DataFrame(dict(ncid=query.ncid,ethnic_group=query.ethnic_group,matchable=matchable,
                                   split=query.split,change_state=query.change_state,
                                   score=np.where(accepted,best[m],-np.inf),top_score=best[m],
                                   predicted_ncid=[reference.ncid.iloc[x] if x>=0 else '' for x in r],
                                   eligible_link=accepted,correct=correct & accepted,
                                   top_tie=ntie[m]>1,candidate_n=nc,true_pair_candidate=truth))
    return result

def _matchable(frame):
    return frame.matchable if 'matchable' in frame else pd.Series(True,index=frame.index)

def metric(frame,threshold):
    acc=frame.eligible_link & frame.score.ge(threshold); mt=_matchable(frame)
    tp=int((acc & frame.correct).sum());fp=int((acc & ~frame.correct).sum())
    n=int(mt.sum()); nq=len(frame); fp_nm=int((acc & ~mt).sum()); n_nm=nq-n
    # MMR denominator = matchable queries; FMR = wrong among all accepted links.
    return dict(n=n,n_queries=nq,accepted=tp+fp,true_matches=tp,false_matches=fp,
                missed_matches=n-tp,unlinked=nq-tp-fp,
                mmr=(n-tp)/n if n else np.nan,
                fmr=fp/(tp+fp) if tp+fp else np.nan,
                unmatched_rate=(nq-tp-fp)/nq if nq else np.nan,
                nonmatchable_n=n_nm,nonmatchable_false_links=fp_nm,
                nonmatchable_false_link_rate=fp_nm/n_nm if n_nm else np.nan)

def exact_curve(frame):
    q=frame[frame.eligible_link].sort_values('score',ascending=False)
    if q.empty:return pd.DataFrame([dict(threshold=np.inf,**metric(frame,np.inf))])
    g=q.groupby('score',sort=True).agg(accepted=('correct','size'),true_matches=('correct','sum')).iloc[::-1].cumsum()
    n=int(_matchable(frame).sum())
    g['false_matches']=g.accepted-g.true_matches;g['n']=n
    g['missed_matches']=n-g.true_matches;g['unlinked']=len(frame)-g.accepted
    g['mmr']=g.missed_matches/n;g['fmr']=g.false_matches/g.accepted;g['unmatched_rate']=g.unlinked/len(frame)
    g=g.reset_index().rename(columns={'score':'threshold'})
    return pd.concat([pd.DataFrame([dict(threshold=np.inf,**metric(frame,np.inf))]),g],ignore_index=True)

def choose_threshold(curve,kind,target,tolerance):
    c=curve[curve.accepted>0].copy()
    if c.empty:return np.inf,False
    if kind=='fmr':
        c=c[c.fmr<=target]
        if c.empty:return np.inf,False
        row=c.sort_values(['mmr','fmr','threshold'],ascending=[True,True,False]).iloc[0]
        return float(row.threshold),True # FMR target is an upper bound.
    c['distance']=(c.mmr-target).abs()
    row=c.sort_values(['distance','fmr','threshold'],ascending=[True,True,False]).iloc[0]
    return float(row.threshold),bool(abs(row.mmr-target)<=tolerance)

def summarise_predictions(predictions,cfg):
    ops=[];curves=[];blocking=[]
    for name,d in predictions.items():
        cal=d[d.split=='calibration'];test=d[d.split=='test']
        ec=exact_curve(cal);tc=exact_curve(test)
        # Sparse display grid only; target selection always uses ALL distinct scores.
        grid=tc.threshold.iloc[np.unique(np.linspace(0,len(tc)-1,min(len(tc),cfg['curve_points'])).astype(int))].tolist()
        for source,sourcecurve in [('calibration',ec),('test_descriptive',tc)]:
            for kind,targets in [('mmr',cfg['mmr_targets']),('fmr',cfg['fmr_targets'])]:
                for target in targets:
                    th,ok=choose_threshold(sourcecurve,kind,target,cfg['mmr_tolerance'])
                    grid.append(th)
                    for group,q in [('All',test)]+list(test.groupby('ethnic_group')):
                        ops.append(dict(model=name,selection=source,target_kind=kind,target=target,
                                        threshold=th,target_attained_on_selection=ok,
                                        ethnic_group=group,test_target_attained=(abs(metric(test,th)['mmr']-target)<=cfg['mmr_tolerance'] if kind=='mmr' else metric(test,th)['fmr']<=target),**metric(q,th)))
        for th in sorted(set(grid),reverse=True):
            for group,q in [('All',test)]+list(test.groupby('ethnic_group')):
                curves.append(dict(model=name,threshold=th,ethnic_group=group,**metric(q,th)))
        if name==MODELS[0]:
            for group,q in [('All',test)]+list(test.groupby('ethnic_group')):
                q=q[_matchable(q)]
                blocking.append(dict(ethnic_group=group,n=len(q),true_pairs_retained=int(q.true_pair_candidate.sum()),
                                     blocking_recall=q.true_pair_candidate.mean(),candidate_pairs=int(q.candidate_n.sum()),
                                     candidates_mean=q.candidate_n.mean(),candidates_median=q.candidate_n.median(),
                                     candidates_p95=q.candidate_n.quantile(.95),candidates_max=q.candidate_n.max()))
    return pd.DataFrame(ops),pd.DataFrame(curves),pd.DataFrame(blocking)
