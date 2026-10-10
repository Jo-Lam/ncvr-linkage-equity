"""Pre-linkage assessment of name dominance, cardinality and ambiguity by ethnic group,
and their association with linkage outcomes.

All pre-linkage measures use only the clean October 2022 eligible frame and the clean
evaluation reference file: no linkage labels, scores or alterations.

Definitions (per record; computed on the full 2022 eligible frame unless stated):
  Dominance    - population frequency of the record's surname and forename; at group
                 level, surname/forename concentration (top-10 share, Herfindahl index,
                 distinct names per 1,000 records).
  Cardinality  - number of OTHER registrants with the identical forename, surname,
                 birth year and gender (exact homonyms within the blocking key).
  Ambiguity    - (a) edit-neighbourhood: number of OTHER registrants whose surname
                 (forename) is exactly one Levenshtein edit from the record's;
                 (b) block ambiguity: number of other evaluation-reference records in the
                 same birth-year/gender block whose forename AND surname both have
                 Jaro-Winkler >= 0.90 with the record's (descriptive only: sparse in a
                 20,000-record cohort). Surname block cardinality: number of OTHER
                 registrants sharing surname, birth year and gender.

Usage: python prelinkage_assessment.py RUN_DIR
RUN_DIR must hold clean/<evaluation_date>.csv.gz, simulation/reference.csv.gz, predictions/
and results/*_operating.csv.gz (save_predictions=true).
"""
import json
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein
from rapidfuzz.process import cdist
from rapidfuzz.distance import JaroWinkler
import statsmodels.api as sm
import statsmodels.formula.api as smf

REF = 'Non-Hispanic White'
ORDER = [REF, 'Non-Hispanic Black', 'Hispanic', 'Asian', 'Indigenous/Pacific Islander', 'Mixed', 'Other', 'Unknown']


def one_edit_neighbours(counts):
    """For each distinct name, number of people whose name is exactly one edit away.
    Candidates come from a deletion index; each is verified with Levenshtein == 1."""
    names = counts.index.to_numpy(); cnt = counts.to_numpy()
    index = defaultdict(list)
    for i, s in enumerate(names):
        index[s].append(i)
        for k in range(len(s)):
            index[s[:k] + s[k + 1:]].append(i)
    out = np.zeros(len(names), np.int64)
    for i, s in enumerate(names):
        cand = set(index.get(s, ()))
        for k in range(len(s)):
            cand.update(index.get(s[:k] + s[k + 1:], ()))
        cand.discard(i)
        out[i] = sum(cnt[j] for j in cand if Levenshtein.distance(s, names[j], score_cutoff=1) == 1)
    return pd.Series(out, index=counts.index)


def concentration(names):
    p = names.value_counts(normalize=True).to_numpy()
    return dict(distinct_per_1000=1000 * len(p) / len(names), top10_share=p[:10].sum(), herfindahl=(p ** 2).sum())


def block_ambiguity(ref):
    out = np.zeros(len(ref), int)
    for _, idx in ref.groupby(['birth_year', 'gender']).indices.items():
        fn = ref.first_name.to_numpy()[idx]; ln = ref.last_name.to_numpy()[idx]
        f = cdist(fn, fn, scorer=JaroWinkler.similarity, workers=-1) >= .9
        l = cdist(ln, ln, scorer=JaroWinkler.similarity, workers=-1) >= .9
        both = f & l; np.fill_diagonal(both, False)
        out[idx] = both.sum(1)
    return out


def describe(frame, cols):
    rows = []
    for g in ORDER:
        d = frame[frame.ethnic_group == g]
        for c in cols:
            v = d[c]
            rows.append(dict(ethnic_group=g, measure=c, n=len(v), mean=v.mean(), median=v.median(),
                             p75=v.quantile(.75), share_positive=(v > 0).mean()))
    return pd.DataFrame(rows)


def main(run):
    run = Path(run); out = run / 'prelinkage'; out.mkdir(exist_ok=True)
    eval_date = json.loads((run / 'config.json').read_text(encoding='utf8'))['evaluation_date']
    pop = pd.read_csv(run / 'clean' / f'{eval_date}.csv.gz', dtype=str, keep_default_na=False,
                      usecols=['ncid', 'first_name', 'last_name', 'birth_year', 'gender', 'ethnic_group'])
    print('2022 frame', len(pop))
    ln_n = pop.last_name.value_counts(); fn_n = pop.first_name.value_counts()
    print('distinct surnames', len(ln_n), 'forenames', len(fn_n))
    ln_nb = one_edit_neighbours(ln_n); fn_nb = one_edit_neighbours(fn_n)
    key = ['first_name', 'last_name', 'birth_year', 'gender']
    pop['homonyms'] = pop.groupby(key).ncid.transform('size') - 1
    pop['surname_block_cardinality'] = pop.groupby(['last_name', 'birth_year', 'gender']).ncid.transform('size') - 1
    pop['surname_freq'] = pop.last_name.map(ln_n); pop['forename_freq'] = pop.first_name.map(fn_n)
    pop['surname_neighbours'] = pop.last_name.map(ln_nb); pop['forename_neighbours'] = pop.first_name.map(fn_nb)
    measures = ['surname_freq', 'forename_freq', 'homonyms', 'surname_block_cardinality', 'surname_neighbours', 'forename_neighbours']
    describe(pop, measures).to_csv(out / 'population_measures_by_group.csv', index=False)
    conc = pd.DataFrame([dict(ethnic_group=g, field=f, n=int((pop.ethnic_group == g).sum()),
                              **concentration(pop.loc[pop.ethnic_group == g, f]))
                         for g in ORDER for f in ['last_name', 'first_name']])
    conc.to_csv(out / 'population_name_concentration_by_group.csv', index=False)

    ref = pd.read_csv(run / 'simulation' / 'reference.csv.gz', dtype=str, keep_default_na=False)
    ref = ref.merge(pop[['ncid'] + measures], on='ncid', how='left', validate='one_to_one')
    ref['block_competitors'] = block_ambiguity(ref)
    ref[['ncid', 'ethnic_group', 'split'] + measures + ['block_competitors']].to_csv(out / 'reference_measures.csv.gz', index=False)
    describe(ref, measures + ['block_competitors']).to_csv(out / 'cohort_measures_by_group.csv', index=False)

    # Outcomes at the calibration-selected 5% MMR threshold (deployment-style), test queries only.
    rows = []
    for f in sorted((run / 'predictions').glob('*.csv.gz')):
        sens, scen, rep, block, model = f.name[:-7].split('_', 4)
        if model not in ('jw', 'jw_tf', 'feature_surname'):
            continue
        ops = pd.read_csv(run / 'results' / f'{sens}_{scen}_{rep}_{block}_operating.csv.gz')
        th = ops.query("model==@model and selection=='calibration' and target_kind=='mmr' and target==0.10 and ethnic_group=='All'").threshold.iloc[0]
        d = pd.read_csv(f, dtype={'ncid': str}); d = d[d.split == 'test']
        acc = d.eligible_link & (d.score >= th)
        if 'matchable' not in d: d = d.assign(matchable=True)
        rows.append(pd.DataFrame(dict(ncid=d.ncid, ethnic_group=d.ethnic_group, change_state=d.change_state, matchable=d.matchable,
                                      missed=(~(acc & d.correct)).astype(int), false=(acc & ~d.correct).astype(int),
                                      sensitivity=sens, scenario=scen, replicate=int(rep), model=model)))
    o = pd.concat(rows, ignore_index=True).merge(ref[['ncid'] + measures + ['block_competitors']], on='ncid', validate='many_to_one')
    for c in measures + ['block_competitors']:
        o['l_' + c] = np.log1p(o[c])
    o['group'] = pd.Categorical(o.ethnic_group, ORDER)
    o['altered'] = (o.change_state != 'neither').astype(int)

    # Descriptive: P(missed | altered) by ambiguity tertile.
    alt = o[(o.altered == 1) & o.matchable].copy()
    alt['surname_neighbour_tertile'] = pd.qcut(alt.surname_neighbours.rank(method='first'), 3, labels=['low', 'mid', 'high'])
    alt.groupby(['sensitivity', 'model', 'scenario', 'surname_neighbour_tertile'], observed=True).missed.agg(['mean', 'size']).to_csv(out / 'missed_by_neighbour_tertile.csv')

    base = 'missed ~ C(group, Treatment(reference="%s")) + C(change_state)' % REF
    full = base + ' + l_surname_freq + l_forename_freq + l_homonyms + l_surname_block_cardinality + l_surname_neighbours + l_forename_neighbours'
    fbase = 'false ~ C(group, Treatment(reference="%s")) + C(change_state)' % REF
    ffull = fbase + ' + l_surname_freq + l_forename_freq + l_homonyms + l_surname_block_cardinality + l_surname_neighbours + l_forename_neighbours'
    res = []
    for (sens, model, scen), d in o.groupby(['sensitivity', 'model', 'scenario']):
        for outcome, forms, data in [('missed', (base, full), d[(d.altered == 1) & d.matchable]), ('false', (fbase, ffull), d[d.altered == 1])]:
            if data[outcome].sum() < 50:  # too few events for stable odds ratios; report descriptively
                continue
            for label, form in zip(['groups_only', 'groups_plus_prelinkage'], forms):
                fit = smf.glm(form, data, family=sm.families.Binomial()).fit(
                    cov_type='cluster', cov_kwds={'groups': pd.factorize(data.ncid)[0]})
                ci = fit.conf_int()
                for term in fit.params.index:
                    if term == 'Intercept':
                        continue
                    res.append(dict(sensitivity=sens, model=model, scenario=scen, outcome=outcome, spec=label, term=term,
                                    odds_ratio=np.exp(fit.params[term]), lo=np.exp(ci.loc[term, 0]), hi=np.exp(ci.loc[term, 1]),
                                    p=fit.pvalues[term], n=len(data), events=int(data[outcome].sum())))
    pd.DataFrame(res).to_csv(out / 'outcome_models.csv', index=False)
    print('written', out)


if __name__ == '__main__':
    main(sys.argv[1])
