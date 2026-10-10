"""Input selection and reference-key diagnostics; no ID is silently deduplicated."""
from pathlib import Path
import hashlib
import json
import unicodedata
import numpy as np
import pandas as pd

GROUPS = ['Non-Hispanic White', 'Non-Hispanic Black', 'Hispanic', 'Asian',
          'Indigenous/Pacific Islander', 'Mixed', 'Other', 'Unknown']
FIELDS = ['first_name', 'last_name']

def stable_number(value, salt=''):
    return int.from_bytes(hashlib.sha256((salt+'|'+str(value)).encode()).digest()[:8], 'big')

def normalize_name(s):
    # Preserve accents, punctuation and internal word boundaries.
    return ' '.join(unicodedata.normalize('NFC', str(s)).upper().split())

def save(df, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, compression='infer')

def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(2**20), b''): h.update(chunk)
    return h.hexdigest()

def prepare_snapshot(spec, cfg, out):
    path = Path(spec['path'])
    mapping = cfg['columns'] | spec.get('columns', {})
    required = ['ncid', 'first_name', 'last_name', 'birth_year', 'gender', 'race', 'ethnic', 'status']
    missing = set(required) - set(mapping)
    if missing: raise ValueError(f'Missing column mappings: {missing}')
    read = spec.get('read_csv', cfg.get('read_csv', {}))
    if path.suffix.lower() == '.parquet':
        raw = pd.read_parquet(path, columns=list(dict.fromkeys(mapping.values()))).fillna('').astype(str)
    else:
        raw = pd.read_csv(path, usecols=list(dict.fromkeys(mapping.values())), dtype=str,
                          keep_default_na=False, **read).fillna('')
    d = raw.rename(columns={v:k for k,v in mapping.items()}).copy()
    # IDs are strings; preserve leading zeros. Only strip surrounding whitespace.
    raw_id = d.ncid.copy(); d['ncid'] = d.ncid.str.strip()
    invalid_id = d.ncid.str.casefold().isin(cfg.get('missing_id_tokens', ['', 'nan', 'null', 'none']))
    valid_id = d[~invalid_id]
    duplicate = valid_id.ncid.duplicated(keep=False)
    counts = valid_id.groupby('ncid').size()
    du = valid_id[duplicate]
    if len(du):
        detail = du.groupby('ncid').agg(rows=('ncid','size'))
        for col in required[1:]: detail[col+'_distinct'] = du.groupby('ncid')[col].nunique(dropna=False)
        save(detail.reset_index(), out/'audit'/f'{spec["date"]}_duplicate_ids.csv.gz')
    if 'voter_reg_num' in d and 'county_id' in d:
        v=d[d.voter_reg_num.str.strip().ne('')]
        alternatives=pd.DataFrame([dict(field='voter_reg_num',duplicate_rows=int(v.voter_reg_num.duplicated(keep=False).sum())),
                                   dict(field='county_id + voter_reg_num',duplicate_rows=int(v.duplicated(['county_id','voter_reg_num'],keep=False).sum()))])
        save(alternatives,out/'audit'/f'{spec["date"]}_alternative_keys.csv')
    report = dict(date=spec['date'], path=str(path.resolve()), sha256=digest(path),
                  loaded_rows=len(d), missing_id_rows=int(invalid_id.sum()),
                  id_whitespace_rows=int((raw_id != d.ncid).sum()),
                  duplicate_ids=int((counts>1).sum()), duplicate_rows=int(duplicate.sum()),
                  duplicate_excess_rows=int((counts-1).clip(lower=0).sum()),
                  repeated_rows_on_loaded_columns=int(valid_id.duplicated().sum()))
    for col in ['status','race','ethnic']:
        save(d[col].value_counts(dropna=False).rename_axis(col).reset_index(name='n'),
             out/'audit'/f'{spec["date"]}_{col}_raw.csv')
    # Conservative default: exclude every row belonging to a duplicated NCID.
    # Identical on selected fields is NOT enough evidence to pick a representative.
    unique = valid_id[~duplicate].copy()
    save(pd.DataFrame([report]), out/'audit'/f'{spec["date"]}_id_summary.csv')
    if cfg.get('duplicate_policy','exclude_all') == 'error' and len(du):
        raise ValueError(f'Duplicate NCIDs in {path}; audit written. Resolve source before rerunning.')
    if cfg.get('duplicate_policy','exclude_all') not in ['error','exclude_all']:
        raise ValueError('duplicate_policy must be error or exclude_all')
    for col in FIELDS: unique[col+'_raw'] = unique[col]; unique[col] = unique[col].map(normalize_name)
    for col in ['gender','race','ethnic','status']: unique[col] = unique[col].str.strip().str.upper()
    unique['birth_year'] = pd.to_numeric(unique.birth_year, errors='coerce')
    unique.loc[unique.birth_year.mod(1).ne(0), 'birth_year'] = np.nan
    ec = cfg['ethnicity']
    unique['ethnic_missing'] = unique.ethnic.isin(ec['missing_codes'])
    unique['race_missing'] = unique.race.isin(ec['missing_codes'])
    unique['ethnic_group'] = 'Unknown'
    # Hispanic takes precedence, regardless of race. Explicit non-Hispanic required
    # to call someone non-Hispanic White/Black. Never impute missing ethnicity.
    for race, group in ec['race_map'].items():
        unique.loc[(unique.race == race) & unique.ethnic.isin(ec['non_hispanic_codes']), 'ethnic_group'] = group
    unique.loc[unique.ethnic.isin(ec['hispanic_codes']), 'ethnic_group'] = 'Hispanic'
    unique['role'] = (
        'application'
        if spec['date'] == cfg['evaluation_date']
        else 'historical_development'
    )
    save(unique, out/'clean'/f'{spec["date"]}_unique.csv.gz')
    retained = unique.copy(); flow = [{'step':'unique_nonmissing_ncid','n':len(retained)}]
    filters = []
    if cfg.get('retained_statuses') is not None:
        filters.append(('retained_status', retained.status.isin(cfg['retained_statuses'])))
    filters += [('nonmissing_names', retained.first_name.ne('') & retained.last_name.ne('')),
                ('valid_birth_year', retained.birth_year.between(int(spec['date'][:4])-cfg['max_age'],
                                                                 int(spec['date'][:4])-cfg['min_age'])),
                ('known_gender', retained.gender.isin(cfg['gender_codes']))]
    # Each filter is aligned to the original unique index; record sequential attrition.
    for name, mask in filters:
        retained = retained.loc[mask.reindex(retained.index).fillna(False)]
        flow.append({'step':name,'n':len(retained)})
    save(pd.DataFrame(flow), out/'audit'/f'{spec["date"]}_selection.csv')
    for stage, frame in [('unique',unique),('retained',retained)]:
        save(frame.groupby(['ethnic_group','status'],dropna=False).agg(
            n=('ncid','size'), ethnic_missing=('ethnic_missing','sum'), race_missing=('race_missing','sum')).reset_index(),
            out/'audit'/f'{spec["date"]}_{stage}_groups_status.csv')
    assert retained.ncid.is_unique
    retained = retained.sort_values('ncid').reset_index(drop=True)
    save(retained, out/'clean'/f'{spec["date"]}.csv.gz')
    return report

def load_clean(out, date, unique=False):
    suffix = '_unique' if unique else ''
    return pd.read_csv(out/'clean'/f'{date}{suffix}.csv.gz', dtype={'ncid':str}, keep_default_na=False)

def link_pair(a, b):
    assert a.ncid.is_unique and b.ncid.is_unique
    return a.merge(b, on='ncid', suffixes=('_a','_b'), validate='one_to_one')

def audit_pair(a, b, tag, out):
    p = link_pair(a,b); records=[]
    for field in ['birth_year','gender','race','ethnic','status']:
        x,y = p[field+'_a'],p[field+'_b']
        known = x.notna() & y.notna() & x.ne('') & y.ne('')
        if field=='gender': known &= x.isin(['M','F']) & y.isin(['M','F'])
        if field in ['race','ethnic']: known &= ~x.isin(['U','UN','UNDESIGNATED']) & ~y.isin(['U','UN','UNDESIGNATED'])
        if field=='birth_year':
            x,y = pd.to_numeric(x,errors='coerce'),pd.to_numeric(y,errors='coerce'); known=x.notna()&y.notna()
        disagreement = known & x.ne(y)
        records.append(dict(pair=tag, field=field, linked_n=len(p), comparable_n=int(known.sum()),
                            discordant_n=int(disagreement.sum()),
                            discordant_rate=float(disagreement.sum()/known.sum()) if known.sum() else np.nan))
    save(pd.DataFrame(records),out/'audit'/f'{tag}_discordance.csv')
    return p
