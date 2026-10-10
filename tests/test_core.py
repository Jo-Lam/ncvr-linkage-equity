import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from audit import normalize_name,link_pair
from simulate import allocate_counts,target_rates,transfer,simulate
from linkage import metric,exact_curve,choose_threshold,string_levels,candidates

def test_metrics_count_wrong_links_as_missed():
    d=pd.DataFrame(dict(eligible_link=[True,True,False],score=[9,8,-np.inf],correct=[True,False,False]))
    m=metric(d,0)
    assert m['mmr']==2/3 and m['fmr']==.5 and m['unmatched_rate']==1/3
    assert metric(d,np.inf)['accepted']==0 and np.isnan(metric(d,np.inf)['fmr'])

def test_threshold_ties_and_unattainable_target():
    d=pd.DataFrame(dict(eligible_link=[True]*4,score=[9,9,8,8],correct=[True,False,True,True]))
    c=exact_curve(d)
    assert c.accepted.tolist()==[0,2,4]
    _,ok=choose_threshold(c,'mmr',.1,.01);assert not ok
    th,ok=choose_threshold(c,'fmr',.01,.01);assert not ok and th==np.inf

def test_rates_do_not_allocate_equal_budget_shares():
    g=pd.Series(['White']*900+['Other']*100)
    p=target_rates(g,{'White':1,'Other':2},.1)
    assert np.isclose(p['Other']/p['White'],2)
    counts=allocate_counts(g,p);assert sum(counts.values())==100
    assert abs(counts['Other']/100-p['Other'])<.01

def test_name_normalization_and_distinct_comparators():
    assert normalize_name('  Aarón   de la Cruz  ')=='AARÓN DE LA CRUZ'
    a=np.array(['MARTHA','ABCD']);b=np.array(['MARHTA','AXCD'])
    assert not np.array_equal(string_levels(a,b,'jw'),string_levels(a,b,'lev'))
    assert transfer('AARON','AARON','AARÓN')=='AARÓN'

def test_blocking_cannot_hide_excluded_true_pairs():
    a=pd.DataFrame(dict(ncid=['1','2'],first_name=['ANNA','ANNE'],birth_year=[1980,1980],gender=['F','F']))
    b=a.copy();b.loc[0,'first_name']='ZNNA'
    main=list(candidates(a,b,'demographic',100,10));old=list(candidates(a,b,'legacy_prefix',100,10))
    assert sum(len(q) for q,r in main)==4
    assert all(0 not in q for q,r in old)

def test_join_rejects_duplicates():
    d=pd.DataFrame({'ncid':['01','01']})
    try:link_pair(d,d)
    except AssertionError:pass
    else:raise AssertionError('Duplicate IDs were silently accepted')

def test_scenario_pairing_and_unique_actual_changes():
    import json
    cfg=json.loads((Path(__file__).parents[1]/'config.example.json').read_text());cfg['minimum_profile_events']=1
    base=pd.DataFrame({'ncid':[str(i) for i in range(80)],'ethnic_group':['Non-Hispanic White']*40+['Other']*40,
                       'first_name':['ANNA']*80,'last_name':['BROWN']*80})
    events=pd.DataFrame([dict(field=f,ethnic_group=g,old='ANNA' if f=='first_name' else 'BROWN',new='ANNZ' if f=='first_name' else 'BROWZ',distance=1)
                         for f in ['first_name','last_name'] for g in base.ethnic_group.unique()])
    a,ra=simulate(base,events,cfg,'S1',0);b,rb=simulate(base,events,cfg,'S2',0)
    assert a.first_name_selected.equals(b.first_name_selected)
    assert ra.changed_n.sum()==16 and ra.failed_n.sum()==0
    again,_=simulate(base,events,cfg,'S1',0);pd.testing.assert_frame_equal(a,again)
    for f in ['first_name','last_name']:assert a[f+'_changed'].equals(a[f].ne(base[f]))

def test_one_to_one_conflicts_and_ties_abstain():
    from linkage import link,MODELS
    class Fixed:
        def transform(self,d):return d
        def compare(self,a,b,i,j):return (i,j)
        def score(self,c):
            m=np.array([[10.,1,0],[9,8,0],[1,1,1]])
            return {name:m[c] for name in MODELS}
    d=pd.DataFrame(dict(ncid=['1','2','3'],birth_year=[1980]*3,gender=['F']*3,
                        first_name=['A','B','C'],ethnic_group=['Other']*3,
                        split=['test']*3,change_state=['neither']*3))
    out=link(d,d,Fixed(),'demographic',{'max_candidate_pairs':100,'chunk_pairs':10})
    for p in out.values():
        assert p.eligible_link.tolist()==[True,False,False]
        assert p.top_tie.tolist()==[False,False,True]
        assert metric(p,0)['missed_matches']==2

def test_input_audit_excludes_all_duplicate_ids_and_retains_unknown(tmp_path):
    import json
    from audit import prepare_snapshot,load_clean
    cfg=json.loads((Path(__file__).parents[1]/'config.example.json').read_text())
    df=pd.DataFrame(dict(ncid=['001','002','002','','003'],first_name=['ANNA']*5,last_name=['LEE']*5,
                          birth_year=[1980]*5,gender_code=['F']*5,race_code=['W']*5,
                          ethnic_code=['NL','NL','NL','NL',''],voter_status_desc=['ACTIVE']*4+['REMOVED']))
    path=tmp_path/'input.csv';df.to_csv(path,index=False)
    report=prepare_snapshot({'date':'2022-10-02','path':str(path)},cfg,tmp_path)
    out=load_clean(tmp_path,'2022-10-02')
    assert out.ncid.tolist()==['001','003']
    assert report['duplicate_rows']==2 and report['missing_id_rows']==1
    assert out.loc[out.ncid=='003','ethnic_group'].iloc[0]=='Unknown'
    assert out.loc[out.ncid=='003','status'].iloc[0]=='REMOVED'
    assert set(out.role)=={'application'}

def test_roles_follow_time_not_person_hash(tmp_path):
    import json
    from audit import prepare_snapshot,load_clean
    cfg=json.loads((Path(__file__).parents[1]/'config.example.json').read_text())
    cfg['evaluation_date']='2022-10-02'
    source=pd.DataFrame(dict(ncid=['001'],first_name=['ANNA'],last_name=['LEE'],
        birth_year=[1980],gender_code=['F'],race_code=['W'],ethnic_code=['NL'],
        voter_status_desc=['ACTIVE']))
    path=tmp_path/'input.csv';source.to_csv(path,index=False)
    for date in ['2019-10-01','2022-10-02']:
        prepare_snapshot({'date':date,'path':str(path)},cfg,tmp_path)
    historical=load_clean(tmp_path,'2019-10-01')
    application=load_clean(tmp_path,'2022-10-02')
    assert historical.ncid.tolist()==application.ncid.tolist()==['001']
    assert historical.role.tolist()==['historical_development']
    assert application.role.tolist()==['application']

def test_prepared_source_requires_matching_snapshot_hashes(tmp_path):
    import json
    from audit import digest
    from run import validate_prepared_source,load_temporal_clean
    source=tmp_path/'snapshot.csv.gz'
    pd.DataFrame({'ncid':['001'],'role':['evaluation']}).to_csv(source,index=False,compression='gzip')
    prepared=tmp_path/'prepared'
    (prepared/'audit').mkdir(parents=True)
    (prepared/'clean').mkdir()
    (prepared/'COMPLETE.txt').write_text('complete')
    (prepared/'manifest.json').write_text(json.dumps({'snapshot_count':1,'audited_pairs':0}))
    pd.DataFrame([dict(date='2022-10-02',path=str(source.resolve()),sha256=digest(source))]).to_csv(
        prepared/'audit'/'snapshot_manifest.csv',index=False)
    for suffix in ['', '_unique']:
        pd.DataFrame({'ncid':['001'],'role':['evaluation']}).to_csv(
            prepared/'clean'/f'2022-10-02{suffix}.csv.gz',index=False,compression='gzip')
    checked,_=validate_prepared_source(
        {'snapshots':[{'date':'2022-10-02','path':str(source)}]},prepared)
    assert checked==prepared.resolve()
    loaded=load_temporal_clean(prepared,'2022-10-02','2022-10-02')
    assert loaded.role.tolist()==['application']
    source.write_text('changed')
    try:
        validate_prepared_source(
            {'snapshots':[{'date':'2022-10-02','path':str(source)}]},prepared)
    except ValueError as error:
        assert 'does not match' in str(error)
    else:
        raise AssertionError('Changed snapshots must invalidate a prepared source')

def test_2022_training_sample_is_disjoint_and_stably_selected():
    from run import sample_disjoint,make_training_pairs
    d=pd.DataFrame(dict(ncid=[f'{i:04}' for i in range(12)],
                        first_name=['ANNA']*12,last_name=['LEE']*12,
                        birth_year=[1980]*12,gender=['F']*12))
    evaluation={'0001','0004','0008'}
    a=sample_disjoint(d,evaluation,5,123)
    b=sample_disjoint(d,evaluation,5,123)
    pd.testing.assert_frame_equal(a,b)
    assert len(a)==5 and not set(a.ncid)&evaluation
    altered=a.copy()
    altered.loc[0,'first_name']='ANNE'
    pairs=make_training_pairs(a,altered)
    assert pairs.ncid.tolist()==a.ncid.tolist()
    assert pairs.loc[0,'first_name_a']=='ANNA'
    assert pairs.loc[0,'first_name_b']=='ANNE'

def test_model_date_must_equal_evaluation_date():
    import json
    from run import validate
    cfg=json.loads((Path(__file__).parents[1]/'config.example.json').read_text())
    validate(cfg)
    cfg['model_date']='2018-10-01'
    try:validate(cfg)
    except ValueError as error:
        assert 'must use the configured 2022 evaluation snapshot' in str(error)
    else:raise AssertionError('Model date must be the evaluation snapshot date')

def test_profile_denominators_include_age_and_unchanged(tmp_path):
    from simulate import events_from_pairs
    p=pd.DataFrame(dict(ncid=['1','2'],first_name_a=['ANNA','JO'],first_name_b=['ANNA','JON'],
                        last_name_a=['LEE','LAM'],last_name_b=['LEE','LAM'],birth_year_a=[1980,1980],
                        birth_year_b=[1980,1980],gender_a=['F','F'],gender_b=['F','F'],
                        ethnic_group_a=['Other','Other']))
    e=events_from_pairs(p,2018,'demo',tmp_path)
    den=pd.read_csv(tmp_path/'profiles/demo_denominators.csv')
    assert set(den.age_band)=={'18–39'}
    assert den.loc[den.field=='first_name','linked_n'].iloc[0]==2
    assert len(e)==1

def test_profiles_cover_pairs_with_missing_birth_year_and_concordance(tmp_path):
    from simulate import events_from_pairs
    p=pd.DataFrame(dict(
        ncid=['missing_birth','concordant','discordant_birth','discordant_gender'],
        first_name_a=['ANNA','JO','KIM','SAM'],
        first_name_b=['ANNE','JON','KIMI','SAMMY'],
        last_name_a=['LEE','LAM','LI','DOE'],
        last_name_b=['LEE','LAMY','LIU','DO'],
        birth_year_a=['','1980','1980','1975'],
        birth_year_b=['','1980','1981','1975'],
        gender_a=['F','M','F','F'],
        gender_b=['F','M','F','M'],
        ethnic_group_a=['Other']*4))
    events=pd.DataFrame(events_from_pairs(p,2018,'historical',tmp_path))
    denominators=pd.read_csv(tmp_path/'profiles/historical_denominators.csv')
    assert set(events.ncid)=={'missing_birth','concordant'}
    assert set(events.loc[events.ncid=='missing_birth','age_band'])=={'Unknown'}
    assert set(events.loc[events.ncid=='concordant','age_band'])=={'18–39'}
    assert denominators.linked_n.tolist()==[1,1,1,1]
    assert denominators.changed_n.tolist()==[1,1,0,1]
