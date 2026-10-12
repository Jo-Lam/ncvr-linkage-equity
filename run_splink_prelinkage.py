"""Group-level name dominance, cardinality and ambiguity of the evaluation-date eligible frame, computed with
splink_prelinkage (https://github.com/Jo-Lam/splink_prelinkage), with an independent pandas check.

Needs a separate environment (see requirements-prelinkage.txt):

    python -m venv .venv-prelinkage
    .venv-prelinkage/Scripts/python -m pip install -r requirements-prelinkage.txt
    .venv-prelinkage/Scripts/python run_splink_prelinkage.py RUN_DIR

RUN_DIR must hold config.json and clean/<evaluation_date>.csv.gz. Writes aggregate outputs only (no names or
NCIDs) to RUN_DIR/prelinkage/splink_prelinkage/. Ambiguity and cardinality are reported against the whole frame
and within each ethnic group; near-neighbour density is computed by prelinkage.py.
"""
import json
import sys
import time
from importlib.metadata import version
from pathlib import Path
import numpy as np
import pandas as pd
from splink import DuckDBAPI
import splink.exploratory as e

GROUPS = ['Non-Hispanic White', 'Non-Hispanic Black', 'Hispanic', 'Asian', 'Indigenous/Pacific Islander', 'Mixed',
          'Other', 'Unknown']
FULL = ['first_name', 'last_name', 'birth_year', 'gender']   # full signature
BLOCK = ['last_name', 'birth_year', 'gender']                 # surname block key


def main(run):
    run = Path(run); out = run / 'prelinkage' / 'splink_prelinkage'; out.mkdir(parents=True, exist_ok=True)
    t = time.time()
    eval_date = json.loads((run / 'config.json').read_text(encoding='utf8'))['evaluation_date']
    pop = pd.read_csv(run / 'clean' / f'{eval_date}.csv.gz', dtype=str, keep_default_na=False,
                      usecols=FULL + ['ethnic_group'])
    groups = [g for g in GROUPS if (pop.ethnic_group == g).any()]
    print('frame', len(pop), flush=True)
    db = DuckDBAPI(); frame = db.register(pop, table_name='frame')

    def save(name, records):
        pd.DataFrame(records).to_csv(out / f'{name}.csv', index=False)
        print(name, len(records), f'{time.time() - t:.0f}s', flush=True)

    save('combination_ambiguity_overall', e.combination_ambiguity_data(frame, column_combinations=[FULL, BLOCK]))
    save('subgroup_combination_ambiguity', e.subgroup_combination_ambiguity_data(
        frame, column_combinations=[FULL, BLOCK], auxiliary_columns=['ethnic_group']))
    save('subgroup_value_dominance_distribution', e.subgroup_value_dominance_distribution_data(
        frame, columns=['first_name', 'last_name'], auxiliary_columns=['ethnic_group']))
    save('subgroup_linkage_column_quality', e.subgroup_linkage_column_quality_data(
        frame, columns=['first_name', 'last_name'], auxiliary_columns=['ethnic_group']))
    per_group = [db.register(pop[pop.ethnic_group == g].drop(columns='ethnic_group').reset_index(drop=True),
                             table_name=f'group_{i}') for i, g in enumerate(groups)]
    save('combination_size_distribution_by_group', e.combination_size_distribution_data(
        per_group, column_combinations=[FULL, BLOCK], table_names_for_chart=groups))

    # Group summaries reconstructed from the package's frequency distributions (no name values involved).
    dom = pd.read_csv(out / 'subgroup_value_dominance_distribution.csv')
    rows = []
    for (g, col), d in dom.groupby(['auxiliary_value', 'column_name']):
        freqs = np.sort(np.repeat(d.frequency.to_numpy(), d.values_at_frequency.to_numpy()))[::-1]; n = freqs.sum()
        rec = d.sort_values('frequency'); cum = (rec.records_at_frequency.cumsum() / n).to_numpy()
        rows.append(dict(ethnic_group=g, field=col, n=n, distinct_per_1000=1000 * len(freqs) / n,
                         top10_share=freqs[:10].sum() / n, herfindahl=((freqs / n) ** 2).sum(),
                         median_within_group_frequency=rec.frequency.to_numpy()[np.searchsorted(cum, .5)]))
    pd.DataFrame(rows).to_csv(out / 'summary_concentration.csv', index=False)
    amb = pd.read_csv(out / 'subgroup_combination_ambiguity.csv')
    amb[['auxiliary_value', 'column_combination', 'group_records', 'ambiguous_record_share', 'largest_signature_group',
         'maximum_one_to_one_assignment_share', 'ambiguous_record_share_gap']].to_csv(out / 'summary_ambiguity.csv', index=False)
    pd.read_csv(out / 'combination_size_distribution_by_group.csv').to_csv(out / 'summary_size.csv', index=False)

    # Independent pandas check of the same within-group quantities.
    check = []
    for g in groups:
        d = pop[pop.ethnic_group == g]
        for key, label in [(FULL, 'full'), (BLOCK, 'block')]:
            size = d.groupby(key)[key[0]].transform('size')
            check.append(dict(ethnic_group=g, key=label, ambiguous_share=(size > 1).mean(), median_group_size=size.median()))
        for col in ['first_name', 'last_name']:
            p = d[col].value_counts(normalize=True).to_numpy()
            check.append(dict(ethnic_group=g, key=col, top10_share=p[:10].sum(), herfindahl=(p ** 2).sum()))
    check = pd.DataFrame(check); check.to_csv(out / 'pandas_check.csv', index=False)
    conc = pd.DataFrame(rows).set_index(['ethnic_group', 'field'])
    amb = amb.set_index(['auxiliary_value', 'column_combination'])
    full_key = ' + '.join(FULL)
    for _, r in check.iterrows():
        if r.key == 'full':
            assert np.isclose(r.ambiguous_share, amb.loc[(r.ethnic_group, full_key), 'ambiguous_record_share'], atol=1e-6)
        elif r.key in ('first_name', 'last_name'):
            assert np.isclose(r.top10_share, conc.loc[(r.ethnic_group, r.key), 'top10_share'], atol=1e-9)
    json.dump(dict(package='splink_prelinkage', repository='https://github.com/Jo-Lam/splink_prelinkage',
                   splink_version=version('splink'), frame_rows=len(pop), evaluation_date=eval_date,
                   pandas_check='passed', seconds=round(time.time() - t)),
              open(out / 'manifest.json', 'w'), indent=2)
    print('done; package and pandas check agree', f'{time.time() - t:.0f}s')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
