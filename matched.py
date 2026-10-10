"""Proof-of-concept check of the core claim: does the name-feature model narrow
ethnic MMR disparities relative to JW-TF, and does that hold at matched FMR?

Reads saved predictions from a run with save_predictions=true.
Usage: python matched.py RUN_DIR
Operating points here are chosen on TEST labels (descriptive/oracle), as in the
pipeline's selection=test_descriptive rows; they are not deployable thresholds.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

MODELS = ['jw', 'jw_tf', 'lev', 'lev_tf', 'feature_surname', 'hybrid']
REF = 'Non-Hispanic White'


def curve(d):
    """Overall test curve over distinct accepted-score thresholds."""
    q = d[d.eligible_link].sort_values('score', ascending=False)
    g = q.groupby('score', sort=True).agg(acc=('correct', 'size'), tp=('correct', 'sum')).iloc[::-1].cumsum()
    g['fp'] = g.acc - g.tp
    n = int(d.matchable.sum()) if 'matchable' in d else len(d)
    g['mmr'] = 1 - g.tp / n
    g['fmr'] = g.fp / g.acc
    return g.reset_index()


def by_group(d, th):
    acc = d.eligible_link & d.score.ge(th)
    rows = []
    for grp, q in [('All', d)] + list(d.groupby('ethnic_group')):
        a = acc[q.index]
        tp = int((a & q.correct).sum()); fp = int((a & ~q.correct).sum())
        nm = int(q.matchable.sum()) if 'matchable' in q else len(q)
        rows.append(dict(ethnic_group=grp, n=nm, mmr=1 - tp / nm,
                         fmr=fp / (tp + fp) if tp + fp else np.nan, accepted=tp + fp, false=fp))
    out = pd.DataFrame(rows)
    ref = out.set_index('ethnic_group').loc[REF]
    out['mmr_gap_pp'] = 100 * (out.mmr - ref.mmr)
    out['fmr_gap_pp'] = 100 * (out.fmr - ref.fmr)
    return out


def at_mmr(c, target):
    return float(c.loc[(c.mmr - target).abs().idxmin(), 'score'])


def at_fmr(c, bound):
    ok = c[c.fmr <= bound]
    return float(ok.loc[ok.mmr.idxmin(), 'score']) if len(ok) else np.inf


def main(run):
    run = Path(run); rows = []
    for f in sorted((run / 'predictions').glob('*.csv.gz')):
        stem = f.name[:-7]
        sens, scen, rep, block, model = stem.split('_', 4)
        d = pd.read_csv(f, dtype={'ncid': str})
        d = d[d.split == 'test'].reset_index(drop=True)
        d['score'] = d.score.replace(-np.inf, np.nan).fillna(-np.inf)
        c = curve(d)
        ops = [('MMR=10% (matched overall MMR)', at_mmr(c, .10))]
        ops += [(f'FMR<={b:.2%} (matched overall FMR)', at_fmr(c, b)) for b in (.0001, .0005, .001)]
        for label, th in ops:
            g = by_group(d, th).assign(sensitivity=sens, scenario=scen, replicate=int(rep), model=model, operating_point=label, threshold=th)
            rows.append(g)
    res = pd.concat(rows, ignore_index=True)
    out = run / 'matched'; out.mkdir(exist_ok=True)
    res.to_csv(out / 'group_rates_matched.csv', index=False)
    summ = res.groupby(['sensitivity', 'scenario', 'operating_point', 'model', 'ethnic_group']).agg(
        n=('n', 'first'), mmr=('mmr', 'mean'), mmr_sd=('mmr', 'std'), fmr=('fmr', 'mean'),
        mmr_gap_pp=('mmr_gap_pp', 'mean'), mmr_gap_sd=('mmr_gap_pp', 'std'), replicates=('mmr', 'size')).reset_index()
    summ.to_csv(out / 'group_rates_matched_summary.csv', index=False)
    print('written', out)


if __name__ == '__main__':
    main(sys.argv[1])
