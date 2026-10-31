"""Aggregate checks of exact-homonym ambiguity in the evaluation-date eligible frame (no record-level output).

Reports the share of registrants with an exact homonym (same forename, surname, birth year and gender) overall and
within each ethnic group, homonym group sizes, the share among active or inactive registrations only, how often
homonyms also share a middle name (possible duplicate registrations), and how much cross-group ambiguity involves
registrants of Unknown ethnicity.

Usage: python register_ambiguity.py RUN_DIR RAW_EVALUATION_SNAPSHOT
RAW_EVALUATION_SNAPSHOT is the raw NCVR file for the evaluation date (for middle names; e.g. data/ncvoter-20221002.csv.gz).
"""
import json
import sys
from pathlib import Path
import pandas as pd

run, raw_path = Path(sys.argv[1]), sys.argv[2]
eval_date = json.loads((run / 'config.json').read_text(encoding='utf8'))['evaluation_date']
d = pd.read_csv(run / 'clean' / f'{eval_date}.csv.gz',
                usecols=['ncid', 'status', 'first_name', 'last_name', 'birth_year', 'gender', 'ethnic_group', 'county_id'],
                dtype={'ncid': str})
raw = pd.read_csv(raw_path, sep=',', encoding='utf-8', encoding_errors='replace', dtype=str,
                  usecols=['ncid', 'middle_name'], keep_default_na=False).drop_duplicates('ncid')
d = d.merge(raw, on='ncid', how='left', validate='one_to_one')
d['middle_name'] = d.middle_name.str.strip().str.upper()
print('records', len(d))
SIG = ['first_name', 'last_name', 'birth_year', 'gender']
d['n_sig'] = d.groupby(SIG).ncid.transform('size')
d['amb'] = d.n_sig > 1
d['amb_within'] = d.groupby(SIG + ['ethnic_group']).ncid.transform('size') > 1

print('\n1. Exact-homonym share (all registrants): %.1f%%' % (100 * d.amb.mean()))
print(d.groupby('ethnic_group')[['amb', 'amb_within']].mean().mul(100).round(1))

print('\n2. Size of homonym groups, among ambiguous records')
a = d[d.amb]
print(pd.cut(a.n_sig, [1, 2, 5, 10, 50, 10**6], labels=['2', '3-5', '6-10', '11-50', '>50'])
      .value_counts(normalize=True).sort_index().mul(100).round(1))

print('\n3. Registration status')
print(d.status.value_counts(normalize=True).mul(100).round(1).head(8))
for keep, lab in [(['A'], 'active only'), (['A', 'I'], 'active or inactive')]:
    s = d[d.status.isin(keep)]
    print(f'homonym share, {lab}: {100 * (s.groupby(SIG).ncid.transform("size") > 1).mean():.1f}% (n={len(s):,})')

print('\n4. Possible duplicate registrations of the same person')
k = a.assign(mid=a.middle_name.replace('', pd.NA)).dropna(subset=['mid'])
print(f'ambiguous records with a middle name: {len(k):,}; sharing it with a homonym: '
      f'{100 * (k.groupby(SIG + ["mid"]).ncid.transform("size") > 1).mean():.1f}%')

print('\n5. Cross-group ambiguity (ambiguous overall but not within own group)')
x = d[d.amb & ~d.amb_within].join(d.groupby(SIG).ethnic_group.agg(set).rename('groups'), on=SIG)
for grp in ['Non-Hispanic Black', 'Indigenous/Pacific Islander', 'Non-Hispanic White']:
    y = x[x.ethnic_group == grp]
    only_unknown = y.groups.map(lambda s: s - {grp} == {'Unknown'}).mean() * 100
    print(f'{grp}: n={len(y):,}; all homonyms of Unknown ethnicity: {only_unknown:.0f}%')
