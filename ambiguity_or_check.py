"""Why is the ambiguity odds ratio < 1 under TF-adjusted Jaro-Winkler (S3)? Fits the ambiguity term alone, with name
frequency, and in the full outcome model, using the same records and threshold as prelinkage.py.

Usage: python ambiguity_or_check.py RUN_DIR   (after prelinkage.py has written RUN_DIR/prelinkage/reference_measures.csv.gz)
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

run = Path(sys.argv[1])
ORDER = ['Non-Hispanic White', 'Non-Hispanic Black', 'Hispanic', 'Asian', 'Indigenous/Pacific Islander', 'Mixed', 'Other', 'Unknown']
measures = ['surname_freq', 'forename_freq', 'homonyms', 'surname_block_cardinality', 'surname_neighbours', 'forename_neighbours']
ref = pd.read_csv(run / 'prelinkage' / 'reference_measures.csv.gz', dtype={'ncid': str})
out = []
for sens in ['all', 'errorsonly']:
    rows = []
    for f in sorted((run / 'predictions').glob(f'{sens}_S3_*_demographic_jw_tf.csv.gz')):
        _, scen, rep, block, model = f.name[:-7].split('_', 4)
        ops = pd.read_csv(run / 'results' / f'{sens}_{scen}_{rep}_{block}_operating.csv.gz')
        th = ops.query("model==@model and selection=='calibration' and target_kind=='mmr' and target==0.10 and ethnic_group=='All'").threshold.iloc[0]
        d = pd.read_csv(f, dtype={'ncid': str}); d = d[(d.split == 'test') & d.matchable & d.change_state.isin(['forename_only', 'surname_only'])]
        acc = d.eligible_link & (d.score >= th)
        rows.append(pd.DataFrame(dict(ncid=d.ncid, ethnic_group=d.ethnic_group, change_state=d.change_state,
                                      missed=(~(acc & d.correct)).astype(int))))
    o = pd.concat(rows).merge(ref[['ncid'] + measures], on='ncid')
    for c in measures: o['l_' + c] = np.log1p(o[c])
    o['group'] = pd.Categorical(o.ethnic_group, ORDER)
    print(f'\n== {sens}: n={len(o)}  share with any homonym={100*(o.homonyms>0).mean():.1f}%')
    print('correlations of log ambiguity with:', o[['l_homonyms', 'l_forename_freq', 'l_surname_freq', 'l_surname_block_cardinality']].corr().l_homonyms.round(2).to_dict())
    b = 'missed ~ C(group, Treatment(reference="Non-Hispanic White")) + C(change_state)'
    specs = {'ambiguity only': b + ' + l_homonyms',
             '+ forename & surname frequency': b + ' + l_homonyms + l_forename_freq + l_surname_freq',
             'full model (Table)': b + ' + l_homonyms + l_forename_freq + l_surname_freq + l_surname_block_cardinality + l_surname_neighbours + l_forename_neighbours'}
    for lab, form in specs.items():
        for cs in ['all', 'forename_only', 'surname_only']:
            data = o if cs == 'all' else o[o.change_state == cs]
            ff = form if cs == 'all' else form.replace(' + C(change_state)', '')
            fit = smf.glm(ff, data, family=sm.families.Binomial()).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(data.ncid)[0]})
            ci = fit.conf_int().loc['l_homonyms']
            print(f'  {lab:32s} {cs:14s} ambiguity OR {np.exp(fit.params.l_homonyms):.2f} ({np.exp(ci[0]):.2f}-{np.exp(ci[1]):.2f})')
    # descriptive: missed share by any homonym, within alteration type
    print(o.assign(has_homonym=o.homonyms > 0).groupby(['change_state', 'has_homonym']).missed.agg(['mean', 'size']).round(3).to_string())
