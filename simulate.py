"""Empirical discrepancy templates and a fully crossed 2 x 2 experiment."""
import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein, JaroWinkler
from audit import FIELDS, save, stable_number

SCENARIOS = {'S1':('pooled','equal'), 'S2':('specific','equal'),
             'S3':('specific','unequal'), 'S4':('pooled','unequal')}

def age_band(year, birth):
    if pd.isna(birth) or birth == '':
        return 'Unknown'
    age = year-float(birth)
    return '18–39' if age<40 else '40–64' if age<65 else '65+'

def events_from_pairs(p, year, pair_name, out):
    stats=[]; events=[]
    p=p.copy()
    p['age_band_a']=p.birth_year_a.map(lambda b:age_band(year,b))
    birth_a=pd.to_numeric(p.birth_year_a,errors='coerce')
    birth_b=pd.to_numeric(p.birth_year_b,errors='coerce')
    both_births_known=birth_a.notna() & birth_b.notna()
    birth_concordant=~both_births_known | birth_a.eq(birth_b)
    names_present=p.first_name_a.ne('') & p.first_name_b.ne('') & p.last_name_a.ne('') & p.last_name_b.ne('')
    gender_known=p.gender_a.isin(['M','F']) & p.gender_b.isin(['M','F'])
    eligible=names_present & gender_known & p.gender_a.eq(p.gender_b) & birth_concordant
    p=p.loc[eligible].copy()
    for field in FIELDS:
        for (group,gender,band), q in p.groupby(['ethnic_group_a','gender_a','age_band_a'],dropna=False):
            stats.append(dict(pair=pair_name,field=field,ethnic_group=group,gender=gender,age_band=band,
                              linked_n=len(q), changed_n=int(q[field+'_a'].ne(q[field+'_b']).sum())))
        for row in p.loc[p[field+'_a'].ne(p[field+'_b'])].itertuples(index=False):
            a,b = getattr(row,field+'_a'),getattr(row,field+'_b')
            ops = Levenshtein.editops(a,b)
            counts={op:sum(o.tag==op for o in ops) for op in ['insert','delete','replace']}
            positions=[o.src_pos/max(1,len(a)) for o in ops]
            pos='start' if max(positions,default=1)<.25 else 'end' if min(positions,default=0)>=.75 else 'across'
            events.append(dict(ncid=row.ncid,pair=pair_name,field=field,old=a,new=b,
                               ethnic_group=row.ethnic_group_a,gender=row.gender_a,
                               age_band=age_band(year,row.birth_year_a),distance=len(ops),position=pos,**counts))
    save(pd.DataFrame(stats),out/'profiles'/f'{pair_name}_denominators.csv')
    return events

def transfer(old, donor_old, donor_new):
    """Transfer an observed edit sequence by relative source position.
    This is a NEW generative model, not a recreation of the old name lookup sampler.
    """
    if old == donor_old: return donor_new
    chars=list(old); edits=[]
    for op in Levenshtein.editops(donor_old,donor_new):
        pos=round(op.src_pos*len(old)/max(1,len(donor_old)))
        if op.tag!='insert': pos=min(pos,max(0,len(old)-1))
        edits.append((pos,op.tag,donor_new[op.dest_pos] if op.tag!='delete' else ''))
    # Reverse source order prevents offsets changing the positions of earlier edits.
    for pos,kind,char in reversed(edits):
        if kind=='insert': chars.insert(pos,char)
        elif kind=='delete' and pos<len(chars): chars.pop(pos)
        elif kind=='replace' and pos<len(chars): chars[pos]=char
    return ''.join(chars).strip()

# ---------------------------------------------------------------------------
# v2: type-aware generator (generator="typed"). See DESIGN_v2.md.
def classify(old,new):
    to,tn=old.split(),new.split()
    if old.replace('-','').replace(' ','')==new.replace('-','').replace(' ',''): return 'spacing_hyphen'
    if len(tn)<len(to) and set(tn)<set(to): return 'token_drop'
    if len(tn)>len(to) and set(to)<set(tn): return 'token_add'
    if len(to)>1 and sorted(to)==sorted(tn): return 'token_reorder'
    if old.startswith(new): return 'truncation'
    if new.startswith(old): return 'extension'
    if len(new)<=2 or len(old)<=2: return 'initial'
    if Levenshtein.distance(old,new)<=2: return 'typographic'
    if JaroWinkler.similarity(old,new)<0.7: return 'replacement'
    return 'other'

def apply_typed(target,donor_old,donor_new,kind):
    """Apply an observed change of a given type to a different target name; None if inapplicable."""
    tt=target.split()
    if kind in ('typographic','other'):
        return transfer(target,donor_old,donor_new)
    if kind=='truncation':
        k=max(1,round(len(target)*len(donor_new)/max(1,len(donor_old))))
        return target[:k].strip() if k<len(target) else None
    if kind=='extension':
        return (target+donor_new[len(donor_old):]).strip()
    if kind=='token_drop':
        if len(tt)<2: return None
        do=donor_old.split(); dropped=[i for i,t in enumerate(do) if t not in donor_new.split()]
        pos=round((dropped[0] if dropped else 0)*(len(tt)-1)/max(1,len(do)-1))
        return ' '.join(t for i,t in enumerate(tt) if i!=pos)
    if kind=='token_add':
        dn=donor_new.split(); added=[t for t in dn if t not in donor_old.split()]
        if not added: return None
        return ' '.join(added+tt) if dn[0] in added else ' '.join(tt+added)
    if kind=='spacing_hyphen':
        if ' ' in target: return target.replace(' ','-' if '-' in donor_new else '',1)
        if '-' in target: return target.replace('-',' ' if ' ' in donor_new else '',1)
        return None
    if kind=='token_reorder':
        return ' '.join(reversed(tt)) if len(tt)>1 else None
    if kind=='initial':
        return target[0] if len(donor_new)<len(donor_old) else None
    if kind=='replacement':
        return donor_new
    raise ValueError(kind)


def target_rates(groups, relative_rates, overall):
    n=groups.value_counts().sort_index()
    absent=set(n.index)-set(relative_rates)
    if absent: raise ValueError(f'Missing relative rates for {absent}')
    r=pd.Series({g:float(relative_rates[g]) for g in n.index})
    if (r<=0).any(): raise ValueError('Relative rates must be positive')
    p=overall*r/(n*r).sum()*n.sum()
    if (p>1).any(): raise ValueError('Requested relative rates imply p>1; choose a feasible design. No silent cap.')
    return p.to_dict()

def allocate_counts(groups, rates):
    # Largest-remainder allocation: fixed overall count, group rounding error < 1.
    ns=groups.value_counts().sort_index(); wanted=ns*pd.Series(rates)
    counts=np.floor(wanted).astype(int)
    extra=int(round(wanted.sum())-counts.sum())
    order=sorted(ns.index,key=lambda g:(-(wanted[g]-counts[g]),g))
    for g in order[:extra]: counts[g]+=1
    assert (counts<=ns).all()
    return counts.to_dict()

def simulate(base, events, cfg, scenario, replicate, sensitivity='all'):
    mechanism,exposure=SCENARIOS[scenario]
    bank=events if sensitivity in ('all','errorsonly') else events[events.distance<=cfg['small_edit_max']]
    if (cfg.get('generator','positional')=='typed' or sensitivity=='errorsonly') and 'kind' not in bank:
        bank=bank.assign(kind=[classify(o,n) for o,n in zip(bank.old,bank.new)])
    if sensitivity=='errorsonly':
        bank=bank[bank.kind!='replacement']
    if bank.empty: raise ValueError(f'No templates for {sensitivity}')
    result=base.copy(); reports=[]
    for fi,field in enumerate(FIELDS):
        target=cfg['alteration_rates'][field]
        rates=target_rates(base.ethnic_group,
            {g:1 for g in base.ethnic_group.unique()} if exposure=='equal' else cfg['relative_rates'],target)
        counts=allocate_counts(base.ethnic_group,rates)
        result[field+'_selected']=False; result[field+'_changed']=False
        result[field+'_fallback']=False; result[field+'_failed']=False
        result[field+'_distance']=0; result[field+'_how']=''
        field_bank=bank[bank.field==field]
        if field_bank.empty: raise ValueError(f'No {field} templates for {sensitivity}')
        for group, frame in base.groupby('ethnic_group',sort=True):
            # Common random numbers: same selected records across mechanism cells;
            # unequal-rate samples are nested within the same random order.
            seed=stable_number(f'{replicate}|{field}|{group}',str(cfg['seed']))
            select_rng=np.random.default_rng(seed)
            selected=select_rng.permutation(frame.index)[:counts[group]]
            pool=field_bank if mechanism=='pooled' else field_bank[field_bank.ethnic_group==group]
            fallback=mechanism=='specific' and len(pool)<cfg['minimum_profile_events']
            if fallback:
                if cfg['profile_fallback']=='error': raise ValueError(f'Too few {field} templates: {group}')
                pool=field_bank
            typed=cfg.get('generator','positional')=='typed'
            if typed and 'gender' not in pool: pool=pool.assign(gender=None)
            donors=list(pool[['old','new']+(['kind','gender'] if typed else [])].itertuples(index=False,name=None))
            variants=pool.groupby(['old','kind']).new.agg(list).to_dict() if typed else {}
            repl=pool[pool.kind=='replacement'].groupby('gender').new.agg(list).to_dict() if typed else {}
            # Same draw stream across all cells for each group/field.
            rng=np.random.default_rng(seed+1)
            result.loc[selected,field+'_selected']=True
            for idx in selected:
                old=base.at[idx,field]; changed=None; how=''
                for attempt in range(cfg['max_template_attempts']):
                    if typed:
                        # Change type follows the (pooled or group) profile; content is an observed
                        # variant of this exact name and type when one exists, else applied by rule.
                        da,db,kind,dg=donors[int(rng.integers(len(donors)))]
                        if (old,kind) in variants:
                            vs=variants[(old,kind)]; new=vs[int(rng.integers(len(vs)))]; how='observed:'+kind
                        else:
                            # A rule-based replacement substitutes a real name: keep it gender-consistent
                            # by drawing a same-gender replacement from the same pool (type share preserved).
                            tg=base.at[idx,'gender'] if 'gender' in base else None
                            if kind=='replacement' and tg is not None and dg!=tg:
                                same=repl.get(tg,[])
                                if not same: continue
                                db=same[int(rng.integers(len(same)))]
                            new=apply_typed(old,da,db,kind); how=kind
                        if new is None: continue
                    else:
                        da,db=donors[int(rng.integers(len(donors)))]; new=transfer(old,da,db)
                    distance=Levenshtein.distance(old,new)
                    if new and new!=old and (sensitivity!='small_edits' or distance<=cfg['small_edit_max']):
                        changed=new; break
                result.at[idx,field+'_how']=how if changed is not None else ''
                result.at[idx,field+'_fallback']=fallback
                if changed is None: result.at[idx,field+'_failed']=True
                else:
                    result.at[idx,field]=changed; result.at[idx,field+'_changed']=True
                    result.at[idx,field+'_distance']=distance
            q=result.loc[frame.index]
            subsets=[('all',q)]+list(q.groupby('split')) if 'split' in q else [('all',q)]
            for split,sub in subsets:
                reports.append(dict(scenario=scenario,replicate=replicate,sensitivity=sensitivity,field=field,
                                    ethnic_group=group,split=split,n=len(sub),target_rate=rates[group],
                                    selected_n=int(sub[field+'_selected'].sum()),changed_n=int(sub[field+'_changed'].sum()),
                                    realised_rate=float(sub[field+'_changed'].mean()),
                                    failed_n=int(sub[field+'_failed'].sum()),fallback_n=int(sub[field+'_fallback'].sum())))
    result['change_state']=np.select([
        result.first_name_changed & result.last_name_changed,result.first_name_changed,result.last_name_changed],
        ['both','forename_only','surname_only'],default='neither')
    assert result.ncid.is_unique
    return result,pd.DataFrame(reports)
