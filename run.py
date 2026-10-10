"""Run the pipeline.

    python run.py --config config_ncvr.json --output results/run01
    python run.py --config config_ncvr.json --output results/run02 --reuse-clean results/run01

See README.md. Outputs contain NCIDs and names: keep them out of version control."""
import argparse
import importlib.metadata
import json
import logging
from pathlib import Path
import pickle
import platform
import shutil
import sys
import numpy as np
import pandas as pd
from audit import (FIELDS, prepare_snapshot, load_clean, audit_pair, link_pair, save, stable_number, digest)
from simulate import SCENARIOS, events_from_pairs, simulate
from linkage import Comparators, FEATURE_NAMES, features, link, summarise_predictions


def sample_fixed(d,n,seed):
    if n and len(d)>n:
        rank=d.ncid.map(lambda x:stable_number(x,str(seed)))
        d=d.loc[rank.nsmallest(n).index]
    return d.sort_values('ncid').reset_index(drop=True)

def top_up_groups(d,base,minimum,seed):
    """v2: add records (same stable order as sample_fixed) until every group has >= minimum."""
    if not minimum:return base.assign(oversampled=False)
    rank=d.ncid.map(lambda x:stable_number(x,str(seed)))
    rest=d[~d.ncid.isin(set(base.ncid))].assign(_r=rank)
    add=[]
    for g,n in base.ethnic_group.value_counts().items():
        if n<minimum:
            add.append(rest[rest.ethnic_group==g].nsmallest(minimum-n,'_r').drop(columns='_r'))
    out=pd.concat([base.assign(oversampled=False)]+[a.assign(oversampled=True) for a in add],ignore_index=True)
    return out.sort_values('ncid').reset_index(drop=True)

def sample_disjoint(d,excluded_ids,n,seed):
    eligible=~d.ncid.isin(excluded_ids)
    if int(eligible.sum())<n:raise ValueError('Not enough records remain for disjoint model training')
    rank=d.ncid.map(lambda x:stable_number(x,str(seed)))
    selected=rank[eligible].nsmallest(n).index
    result=d.loc[selected].sort_values('ncid').reset_index(drop=True)
    if result.ncid.isin(excluded_ids).any():raise AssertionError('Model and evaluation NCIDs overlap')
    return result

def make_training_pairs(reference,altered):
    if not reference.ncid.is_unique or not altered.ncid.is_unique:
        raise ValueError('Training records must have unique NCIDs')
    if reference.ncid.tolist()!=altered.ncid.tolist():
        raise ValueError('Synthetic training queries must align one-to-one with their clean references')
    pairs=pd.DataFrame({'ncid':reference.ncid.to_numpy()})
    for field in FIELDS:
        pairs[field+'_a']=reference[field].to_numpy()
        pairs[field+'_b']=altered[field].to_numpy()
    return pairs


def validate(cfg):
    if cfg.get('role_design','temporal_returning_people') != 'temporal_returning_people':
        raise ValueError('This pipeline copy requires role_design=temporal_returning_people')
    dates=[s['date'] for s in cfg['snapshots']]
    if len(set(dates))!=len(dates):raise ValueError('Snapshot dates must be unique')
    for d in dates:pd.Timestamp(d)
    ev=cfg['evaluation_date'];model_date=cfg.get('model_date');profiles=cfg['profile_pairs']
    if model_date!=ev:raise ValueError('Model development and application must use the configured 2022 evaluation snapshot')
    for pair in profiles:
        if len(pair)!=2 or pair[0]>=pair[1] or pair[1]>=ev:raise ValueError('Profile pairs must precede evaluation')
    if not set([ev]+[x for p in profiles for x in p])<=set(dates):raise ValueError('Undefined snapshot date')
    if not 0<cfg['calibration_fraction']<1:raise ValueError('Calibration fraction must be between 0 and 1')
    if cfg['replicates']<1:raise ValueError('replicates must be positive')
    if any(not 0<=p<=1 for p in cfg['alteration_rates'].values()):raise ValueError('Invalid alteration rate')
    if cfg['model_max_records']<10 or cfg['evaluation_max_records']<1:raise ValueError('Model/evaluation sample sizes are too small')
    if cfg['model_training_scenario'] not in SCENARIOS:raise ValueError('Unknown model-training scenario')

def load_temporal_clean(root,date,eval_date,unique=False):
    frame=load_clean(root,date,unique=unique)
    frame['role']='application' if date==eval_date else 'historical_development'
    return frame

def validate_prepared_source(cfg,root):
    root=Path(root).resolve()
    if not (root/'COMPLETE.txt').is_file():
        raise ValueError(f'Prepared source is not a completed run: {root}')
    prior=json.loads((root/'manifest.json').read_text(encoding='utf8'))
    dates=sorted(s['date'] for s in cfg['snapshots'])
    expected_pairs=len(dates)*(len(dates)-1)//2
    if prior.get('snapshot_count')!=len(dates) or prior.get('audited_pairs')!=expected_pairs:
        raise ValueError('Prepared source does not contain the required snapshot/pair audits')
    snapshot_manifest=pd.read_csv(root/'audit'/'snapshot_manifest.csv',dtype={'date':str})
    if set(snapshot_manifest.date)!=set(dates):
        raise ValueError('Prepared source snapshot dates do not match the current configuration')
    records=snapshot_manifest.set_index('date')
    for spec in cfg['snapshots']:
        path=Path(spec['path']).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        if str(path)!=str(records.at[spec['date'],'path']) or digest(path)!=records.at[spec['date'],'sha256']:
            raise ValueError(f'Prepared source input does not match configured snapshot {spec["date"]}')
        for suffix in ['', '_unique']:
            if not (root/'clean'/f'{spec["date"]}{suffix}.csv.gz').is_file():
                raise FileNotFoundError(root/'clean'/f'{spec["date"]}{suffix}.csv.gz')
    return root,prior


def add_metadata(d,meta):
    return d.assign(**meta)


def aggregate(all_ops,out):
    keys=['sensitivity','blocking','scenario','replicate','selection','target_kind','target']
    white=all_ops[all_ops.ethnic_group=='Non-Hispanic White'][keys+['model','mmr','fmr']]
    x=all_ops.merge(white,on=keys+['model'],how='left',suffixes=('','_white'),validate='many_to_one')
    x['mmr_disparity_pp']=100*(x.mmr-x.mmr_white);x['fmr_disparity_pp']=100*(x.fmr-x.fmr_white)
    x.loc[x.ethnic_group=='All',['mmr_disparity_pp','fmr_disparity_pp']]=np.nan
    save(x,out/'results'/'operating_points_with_disparities.csv.gz')
    metrics=['mmr','fmr','unmatched_rate','mmr_disparity_pp','fmr_disparity_pp']
    by=[k for k in keys if k!='replicate']+['model','ethnic_group']
    rows=[]
    for key,d in x.groupby(by,dropna=False):
        for metric in metrics:
            v=d[metric].dropna()
            rows.append(dict(zip(by,key))|dict(metric=metric,n_replicates=len(v),mean=v.mean(),
                         minimum=v.min(),maximum=v.max(),sd=v.std(ddof=1),
                         mcse=v.std(ddof=1)/np.sqrt(len(v)) if len(v)>1 else np.nan,
                         selection_target_attainment=d.target_attained_on_selection.mean()))
    save(pd.DataFrame(rows),out/'results'/'replicate_summary.csv')
    base=x[x.model=='jw_tf'][keys+['ethnic_group','target_attained_on_selection','test_target_attained']+metrics]
    paired=x.merge(base,on=keys+['ethnic_group'],suffixes=('','_jw_tf'),validate='many_to_one')
    paired['both_models_attain_target']=paired.target_attained_on_selection & paired.target_attained_on_selection_jw_tf & paired.test_target_attained & paired.test_target_attained_jw_tf
    for m in metrics:paired[m+'_difference_vs_jw_tf']=paired[m]-paired[m+'_jw_tf']
    save(paired,out/'results'/'paired_model_differences.csv.gz')
    # Paired 2x2 contrasts; report every replicate before any averaging.
    by2=[k for k in keys if k!='scenario']+['model','ethnic_group']
    contrast=[]
    for metric in ['mmr','fmr','mmr_disparity_pp','fmr_disparity_pp']:
        p=x.pivot(index=by2,columns='scenario',values=metric)
        for label,coef in {'specific_minus_pooled_equal':{'S2':1,'S1':-1},
                           'unequal_minus_equal_pooled':{'S4':1,'S1':-1},
                           'unequal_minus_equal_specific':{'S3':1,'S2':-1},
                           'interaction':{'S3':1,'S2':-1,'S4':-1,'S1':1}}.items():
            if set(coef)<=set(p):
                v=sum(p[k]*w for k,w in coef.items()).rename('difference').reset_index()
                contrast.append(v.assign(metric=metric,contrast=label))
    if contrast:save(pd.concat(contrast,ignore_index=True),out/'results'/'paired_scenario_contrasts.csv.gz')


def plot_curves(curves,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for (sensitivity,blocking,scenario,rep),d in curves.groupby(['sensitivity','blocking','scenario','replicate']):
        if rep!=0:continue # Display one illustrative replicate; CSV contains every replicate.
        groups=['All']+sorted(g for g in d.ethnic_group.unique() if g!='All')
        fig,axs=plt.subplots(3,3,figsize=(12,10),squeeze=False)
        for ax,group in zip(axs.flat,groups):
            for model,g in d[d.ethnic_group==group].groupby('model'):
                g=g.sort_values('mmr');ax.plot(100*g.mmr,100*g.fmr,label=model,lw=1)
            ax.set_title(group,fontsize=9);ax.set_xlabel('Missed true matches (%)');ax.set_ylabel('False matches / accepted (%)')
            ax.set_xlim(0,20);ax.grid(alpha=.2)
        for ax in list(axs.flat)[len(groups):]:ax.set_visible(False)
        handles,labels=axs.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3)
        fig.tight_layout(rect=[0,.06,1,.97]);fig.suptitle(f'{scenario} | {sensitivity} | {blocking} | replicate 0',fontsize=11)
        folder=out/'figures';folder.mkdir(exist_ok=True);fig.savefig(folder/f'{sensitivity}_{blocking}_{scenario}.png',dpi=180);plt.close(fig)


def reuse_clean_snapshots(prior):
    """Replace snapshot preparation by copying cleaned snapshots from an earlier run of this pipeline."""
    prior=Path(prior)
    def _reuse(spec,cfg,out_):
        (out_/'clean').mkdir(parents=True,exist_ok=True)
        for suffix in ['','_unique']:
            shutil.copy2(prior/'clean'/f"{spec['date']}{suffix}.csv.gz",out_/'clean')
        return dict(date=spec['date'],path=spec['path'],reused_clean_from=str(prior.resolve()))
    global prepare_snapshot
    prepare_snapshot=_reuse


def run(cfg,out,audit_only=False,prepared_root=None):
    validate(cfg)
    out=Path(out)
    if out.exists() and any(out.iterdir()):raise ValueError('Output folder must be new/empty; existing results are never overwritten')
    out.mkdir(parents=True,exist_ok=True)
    (out/'config.json').write_text(json.dumps(cfg,indent=2),encoding='utf8')
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.StreamHandler(),logging.FileHandler(out/'run.log')])
    dates=sorted(s['date'] for s in cfg['snapshots'])
    if prepared_root is None:
        data_root=out
        reports=[]
        for spec in sorted(cfg['snapshots'],key=lambda s:s['date']):
            logging.info('Audit and prepare %s',spec['date']);reports.append(prepare_snapshot(spec,cfg,out))
        save(pd.DataFrame(reports),out/'audit'/'snapshot_manifest.csv')
        # Optional: audit each snapshot against all later snapshots (diagnostics only; slow).
        for i,date in enumerate(dates[:-1] if cfg.get('audit_all_pairs',True) else []):
            a=load_temporal_clean(data_root,date,cfg['evaluation_date'],unique=True)
            for other in dates[i+1:]:
                audit_pair(a,load_temporal_clean(data_root,other,cfg['evaluation_date'],unique=True),
                           date+'_'+other,out)
        preparation_metadata={}
    else:
        data_root,prior=validate_prepared_source(cfg,prepared_root)
        shutil.copytree(data_root/'audit',out/'audit')
        reports=pd.read_csv(data_root/'audit'/'snapshot_manifest.csv').to_dict('records')
        preparation_metadata=dict(preparation_reused_from=str(data_root),
                                  preparation_manifest_sha256=digest(data_root/'manifest.json'))
        logging.info('Reusing hash-verified snapshot preparation and all-pairs audits from %s',data_root)
    version={p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn','rapidfuzz','matplotlib']}
    manifest=dict(python=sys.version,platform=platform.platform(),packages=version,
                  source_sha256={p.name:digest(p) for p in Path(__file__).parent.glob('*.py')},
                  snapshot_count=len(dates),audited_pairs=len(dates)*(len(dates)-1)//2,
                  profile_pair_count=len(cfg['profile_pairs']),
                  model_date=cfg['model_date'],
                  model_training_design='2022_training_ncids_disjoint_from_2022_evaluation',
                  role_design=cfg.get('role_design','temporal_returning_people'),
                  **preparation_metadata)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    if audit_only:return
    events=[]
    for a_date,b_date in cfg['profile_pairs']:
        # Profile eligibility is intentionally not limited to the model/evaluation
        # age-filtered cohort: early snapshots lack birth year. Historical profiles,
        # model development, and 2022 application may include the same people.
        a,b=load_temporal_clean(data_root,a_date,cfg['evaluation_date'],unique=True),\
            load_temporal_clean(data_root,b_date,cfg['evaluation_date'],unique=True)
        p=link_pair(a,b)
        events+=events_from_pairs(p,int(a_date[:4]),a_date+'_'+b_date,out)
    events=pd.DataFrame(events)
    if events.empty:raise ValueError('No observed discrepancies in profile records')
    save(events,out/'profiles'/'events.csv.gz')
    save(events.groupby(['field','ethnic_group','gender','age_band','distance','position']).agg(
        events=('ncid','size'),unique_people=('ncid','nunique')).reset_index(),out/'profiles'/'profile_counts.csv')
    # Explicit proportions per field and group, counting each changed record-pair once.
    prof=events.groupby(['field','ethnic_group','distance','position']).size().reset_index(name='n')
    prof['denominator']=prof.groupby(['field','ethnic_group']).n.transform('sum');prof['proportion']=prof.n/prof.denominator
    save(prof,out/'profiles'/'profile_distributions.csv')
    original=load_temporal_clean(data_root,cfg['evaluation_date'],cfg['evaluation_date'])
    base=sample_fixed(original,cfg['evaluation_max_records'],cfg['seed'])
    base=top_up_groups(original,base,cfg.get('evaluation_min_per_group',0),cfg['seed'])
    base['split']=base.ncid.map(lambda x:'calibration' if stable_number(x,'calibration')/2**64<cfg['calibration_fraction'] else 'test')
    if base.split.nunique()!=2:raise ValueError('Both calibration and test records are required')
    evaluation_ids=set(base.ncid)
    # v2 open cohort: clean distractor references with no query counterpart, and a fixed
    # share of evaluation people whose clean record is withheld from the reference.
    n_distractors=cfg.get('open_distractors',0)
    distractors=sample_disjoint(original,evaluation_ids,n_distractors,cfg['seed']+2) if n_distractors else original.iloc[:0]
    nonmatch_fraction=cfg.get('open_nonmatch_fraction',0)
    base['in_reference']=base.ncid.map(lambda x:stable_number(x,'open_nonmatch')/2**64>=nonmatch_fraction)
    link_reference=pd.concat([base[base.in_reference],distractors],ignore_index=True)
    if not link_reference.ncid.is_unique:raise AssertionError('Reference NCIDs must be unique')
    training_reference=sample_disjoint(
        original,evaluation_ids|set(distractors.ncid),cfg['model_max_records'],cfg['seed']+1)
    freq_reference=original[['first_name','last_name']] if cfg.get('frequency_reference')=='full_frame' else None
    del original
    training_altered,_=simulate(
        training_reference,events,cfg,cfg['model_training_scenario'],
        cfg['model_training_replicate'],'all')
    training_pairs=make_training_pairs(training_reference,training_altered)
    if training_pairs.ncid.isin(evaluation_ids).any():
        raise AssertionError('Model training NCIDs overlap the 2022 application cohort')
    training_counts=dict(
        model_date=cfg['model_date'],
        reference_n=len(training_reference),
        positive_pairs_n=len(training_pairs),
        evaluation_n=len(base),
        training_evaluation_ncid_overlap=0,
        training_scenario=cfg['model_training_scenario'],
        training_replicate=cfg['model_training_replicate'],
        forename_synthetic_changes=int(training_altered.first_name_changed.sum()),
        surname_synthetic_changes=int(training_altered.last_name_changed.sum()),
        forename_change_rate=float(training_altered.first_name_changed.mean()),
        surname_change_rate=float(training_altered.last_name_changed.mean()))
    save(pd.DataFrame([training_counts]),out/'model'/'training_counts.csv')
    logging.info('Fit models using %d 2022 reference records and %d synthetic positive pairs; training/evaluation NCID overlap is zero',
                 len(training_reference),len(training_pairs))
    model=Comparators(cfg).fit(training_reference,training_pairs,freq_reference=freq_reference)
    del freq_reference
    with open(out/'model'/'model.pkl','wb') as f:pickle.dump(model,f)
    for field in FIELDS:
        loadings=pd.DataFrame(model.pca[field].components_.T,index=FEATURE_NAMES).rename_axis('feature').reset_index()
        save(loadings,out/'model'/f'{field}_pca_loadings.csv')
        params=dict(cutoffs=[v.tolist() for v in model.cutoffs[field]],
                    explained_variance_ratio=model.pca[field].explained_variance_ratio_.tolist(),
                    scaler_mean=model.scalers[field].mean_.tolist(),scaler_scale=model.scalers[field].scale_.tolist())
        (out/'model'/f'{field}_pca.json').write_text(json.dumps(params,indent=2))
        raw_feats=pd.DataFrame(features(training_reference[field],model.prefix[field],model.counts[field]),columns=FEATURE_NAMES)
        raw_feats['ethnic_group']=training_reference.ethnic_group.to_numpy()
        save(raw_feats.groupby('ethnic_group').agg(['mean','std','count']).reset_index(),out/'model'/f'{field}_feature_group_summary.csv')
        feats=pd.DataFrame(model.transform(training_reference)[field+'_pc']);feats['ethnic_group']=training_reference.ethnic_group.to_numpy()
        save(feats.groupby('ethnic_group').agg(['mean','std','count']).reset_index(),out/'model'/f'{field}_pc_group_summary.csv')
    weights={k:dict(log2_bayes_factors=v.tolist(),u=model.u[k].tolist()) for k,v in model.weights.items()}
    (out/'model'/'weights.json').write_text(json.dumps(weights,indent=2))
    historical_ids=set()
    for historical_date in dates:
        if historical_date >= cfg['evaluation_date']:
            continue
        historical=load_temporal_clean(data_root,historical_date,cfg['evaluation_date'],unique=True)
        historical_ids.update(historical.loc[historical.ncid.isin(evaluation_ids),'ncid'])
    base['historically_observed']=base.ncid.isin(historical_ids)
    overlap=base.groupby(['split','ethnic_group']).historically_observed.agg(
        evaluation_n='size',historically_observed_n='sum').reset_index()
    overlap['historically_observed_rate']=overlap.historically_observed_n/overlap.evaluation_n
    save(overlap,out/'simulation'/'returning_person_overlap.csv')
    save(pd.DataFrame([dict(evaluation_n=len(base),
                            historically_observed_n=int(base.historically_observed.sum()),
                            historically_observed_rate=float(base.historically_observed.mean()))]),
         out/'simulation'/'returning_person_overlap_overall.csv')
    save(base,out/'simulation'/'reference.csv.gz')
    save(distractors[['ncid','ethnic_group']],out/'simulation'/'distractors.csv.gz')
    save(pd.DataFrame([dict(evaluation_n=len(base),oversampled_n=int(base.oversampled.sum()),
                            nonmatchable_n=int((~base.in_reference).sum()),distractors_n=len(distractors),
                            reference_n=len(link_reference))]),out/'simulation'/'open_cohort_counts.csv')
    save(base.groupby(['ethnic_group','split']).size().reset_index(name='n'),out/'simulation'/'evaluation_group_counts.csv')
    all_ops=[];all_curves=[];all_block=[];all_exposure=[];joint=[]
    for sensitivity in cfg['sensitivities']:
        if sensitivity not in ['all','small_edits','errorsonly']:raise ValueError(sensitivity)
        start=cfg.get('replicate_start',0)  # v2: lets replicate blocks run in parallel processes
        for rep in range(start,start+cfg['replicates']):
            for scenario in SCENARIOS:
                logging.info('Simulate and link %s %s replicate %s',sensitivity,scenario,rep)
                altered,exposure=simulate(base,events,cfg,scenario,rep,sensitivity)
                all_exposure.append(exposure)
                joint.append(altered.groupby(['ethnic_group','split','change_state']).size().reset_index(name='n').assign(
                    sensitivity=sensitivity,scenario=scenario,replicate=rep))
                if cfg['save_altered_records']:
                    save(altered,out/'simulation'/f'{sensitivity}_{scenario}_{rep}.csv.gz')
                for blocking in cfg['blocking']:
                    meta=dict(sensitivity=sensitivity,scenario=scenario,replicate=rep,blocking=blocking)
                    predictions=link(link_reference,altered,model,blocking,cfg)
                    ops,curves,block=summarise_predictions(predictions,cfg)
                    all_ops.append(add_metadata(ops,meta));all_curves.append(add_metadata(curves,meta));all_block.append(add_metadata(block,meta))
                    # Changed and unchanged records must be reported at each calibration threshold.
                    strat=[]
                    for row in ops[(ops.ethnic_group=='All')&(ops.selection=='calibration')].itertuples():
                        from linkage import metric
                        q=predictions[row.model];q=q[q.split=='test']
                        for (group,state),sub in q.groupby(['ethnic_group','change_state']):
                            strat.append(dict(model=row.model,target_kind=row.target_kind,target=row.target,
                                              threshold=row.threshold,ethnic_group=group,change_state=state,**metric(sub,row.threshold)))
                    save(add_metadata(pd.DataFrame(strat),meta),out/'results'/f'{sensitivity}_{scenario}_{rep}_{blocking}_by_change.csv.gz')
                    save(add_metadata(ops,meta),out/'results'/f'{sensitivity}_{scenario}_{rep}_{blocking}_operating.csv.gz')
                    if cfg['save_predictions']:
                        for name,frame in predictions.items():save(frame,out/'predictions'/f'{sensitivity}_{scenario}_{rep}_{blocking}_{name}.csv.gz')
    ops=pd.concat(all_ops,ignore_index=True);curves=pd.concat(all_curves,ignore_index=True)
    aggregate(ops,out)
    save(curves,out/'results'/'threshold_curves.csv.gz');save(pd.concat(all_block),out/'results'/'blocking.csv')
    save(pd.concat(all_exposure),out/'simulation'/'realised_exposure.csv')
    save(pd.concat(joint),out/'simulation'/'joint_exposure.csv')
    plot_curves(curves,out)
    (out/'COMPLETE.txt').write_text('Completed. Inspect audit, target attainment, realised exposure and model results before interpretation.\n')
    logging.info('Complete: %s',out)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--output',required=True)
    ap.add_argument('--audit-only',action='store_true');ap.add_argument('--prepared-root')
    ap.add_argument('--reuse-clean',help='copy cleaned snapshots from an earlier run instead of re-reading raw files')
    args=ap.parse_args()
    config_path=Path(args.config).resolve();cfg=json.loads(config_path.read_text(encoding='utf-8-sig'))
    # Relative inputs resolve against the configuration file, never against cwd.
    for spec in cfg['snapshots']:
        p=Path(spec['path'])
        if not p.is_absolute():spec['path']=str(config_path.parent/p)
    if args.reuse_clean:reuse_clean_snapshots(args.reuse_clean)
    run(cfg,Path(args.output),args.audit_only,args.prepared_root)
