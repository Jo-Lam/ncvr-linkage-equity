"""Figures for the PLoS One manuscript (v2 results). Writes 300-dpi TIFF (LZW) and PNG previews.
Usage: python figures.py RUN_DIR OUT_DIR"""
import glob
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
from simulate import classify

RUN, OUT = Path(sys.argv[1]), Path(sys.argv[2])
PROF = RUN
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family': 'Arial', 'font.size': 8, 'axes.titlesize': 8.5, 'axes.labelsize': 8,
                     'legend.fontsize': 7, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'axes.spines.top': False,
                     'axes.spines.right': False})
GROUPS = ['Non-Hispanic White', 'Non-Hispanic Black', 'Hispanic', 'Asian', 'Indigenous/Pacific Islander', 'Mixed', 'Other']
SHORT = {'Non-Hispanic White': 'NH White', 'Non-Hispanic Black': 'NH Black', 'Hispanic': 'Hispanic', 'Asian': 'Asian',
         'Indigenous/Pacific Islander': 'Indigenous/PI', 'Mixed': 'Mixed', 'Other': 'Other', 'Unknown': 'Unknown'}
MODEL = {'jw': 'Jaro-Winkler', 'jw_tf': 'Jaro-Winkler + TF', 'lev': 'Levenshtein', 'lev_tf': 'Levenshtein + TF',
         'hybrid': 'Mixed model'}  # standalone name-feature comparator reported in S1/S2 Tables only
COL = {'jw': '#9ecae1', 'jw_tf': '#08519c', 'lev': '#fdae6b', 'lev_tf': '#d94801', 'feature_surname': '#6a3d9a', 'hybrid': '#33a02c'}
W = 7.5  # inches; PLoS maximum figure width


def save(fig, name):
    fig.savefig(OUT / f'{name}.tif', dpi=300, pil_kwargs={'compression': 'tiff_lzw'})
    fig.savefig(OUT / f'{name}.png', dpi=150)
    plt.close(fig)


# ---- Fig 1: study design ----
fig, ax = plt.subplots(figsize=(W, 3.9)); ax.set_xlim(0, 100); ax.set_ylim(0, 52); ax.axis('off')
def box(x, y, w, h, text, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.4,rounding_size=1.2', fc=fc, ec='#444', lw=0.6))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=6.4)
def arrow(x1, y1, x2, y2, rad=0):
    ax.annotate('', (x2, y2), (x1, y1), arrowprops=dict(arrowstyle='->', lw=0.7, color='#444', connectionstyle=f'arc3,rad={rad}'))
box(1, 38, 30, 11, 'NCVR snapshots 2011-2021\n10 annual pairs linked by NCID\n575,696 within-person\nname changes', '#e8f1fa')
box(35, 38, 30, 11, 'Empirical change profiles by group\nchange type, edit distance, position\n(pooled or group-specific)', '#e8f1fa')
box(69, 38, 30, 11, 'Pre-linkage assessment (2022 frame)\ndominance, cardinality, ambiguity,\nnear-neighbour density\nby ethnic group', '#fdf2e3')
box(1, 20, 30, 12, 'October 2022 eligible frame\n(n = 8,082,366)\nname-frequency reference', '#eef7ec')
box(35, 20, 30, 12, 'Model training sample (n = 50,000)\nsynthetic positives (pooled S1 profile)\n200,000 sampled non-matches', '#eef7ec')
box(69, 20, 30, 12, 'Evaluation cohort (n = 23,703)\n>=1,000 per group; 25% calibration\nand 75% held-out test records', '#eef7ec')
box(1, 2, 47, 12, '2x2 simulation, 20 replicates\nS1 pooled/equal | S2 group-specific/equal\nS3 group-specific/unequal | S4 pooled/unequal\n10% alteration per name field', '#f3eefa')
box(52, 2, 47, 12, 'Open-cohort linkage: altered queries vs reference\n(evaluation records without 10% of people + 30,000 distractors)\nfive models; birth-year and gender blocking;\nmissed and false matches by group', '#f3eefa')
arrow(31.5, 43.5, 34.5, 43.5); arrow(16, 38, 16, 32.5); arrow(31.5, 26, 34.5, 26); arrow(65.5, 26, 68.5, 26)
arrow(50, 38, 50, 32.5); arrow(36, 38.5, 33, 14.5, rad=0.35); arrow(84, 20, 75, 14.5); arrow(48.5, 8, 51.5, 8)
arrow(84, 38, 84, 32.5)
ax.text(86, 34.5, 'measured\nbefore linkage', fontsize=6, color='#7a4b00')
save(fig, 'Fig1')

# ---- Fig 2: observed discrepancy rates and change types ----
den = pd.concat([pd.read_csv(f) for f in glob.glob(str(PROF / 'profiles' / '*_denominators.csv'))])
rate = den.groupby(['ethnic_group', 'field'])[['linked_n', 'changed_n']].sum()
rate = (100 * rate.changed_n / rate.linked_n).unstack('field').reindex(GROUPS)
ev = pd.read_csv(PROF / 'profiles' / 'events.csv.gz', dtype=str, keep_default_na=False, usecols=['field', 'old', 'new', 'ethnic_group'])
ev['kind'] = [classify(o, n) for o, n in zip(ev.old, ev.new)]
lump = {'typographic': 'Typographic (<=2 edits)', 'truncation': 'Truncation/extension', 'extension': 'Truncation/extension',
        'token_drop': 'Token added/dropped', 'token_add': 'Token added/dropped', 'spacing_hyphen': 'Spacing/hyphen',
        'token_reorder': 'Other', 'initial': 'Other', 'other': 'Other', 'replacement': 'Different name'}
ev['kind'] = ev.kind.map(lump)
kinds = ['Typographic (<=2 edits)', 'Truncation/extension', 'Token added/dropped', 'Spacing/hyphen', 'Other', 'Different name']
kcol = ['#08519c', '#4292c6', '#9ecae1', '#c6dbef', '#d9d9d9', '#d94801']
fig, axs = plt.subplots(1, 3, figsize=(W, 3.0), gridspec_kw={'width_ratios': [1.1, 1, 1]})
y = np.arange(len(GROUPS))
axs[0].barh(y - 0.2, rate['first_name'], 0.4, label='Forename', color='#4292c6')
axs[0].barh(y + 0.2, rate['last_name'], 0.4, label='Surname', color='#d94801')
axs[0].set_yticks(y, [SHORT[g] for g in GROUPS]); axs[0].invert_yaxis(); axs[0].set_xlabel('Changed pair-observations (%)')
axs[0].set_title('A  Discrepancy rate', loc='left'); axs[0].legend(frameon=False, loc='center right', fontsize=6.3)
for ax, field, title in [(axs[1], 'first_name', 'B  Forename change types'), (axs[2], 'last_name', 'C  Surname change types')]:
    share = ev[ev.field == field].groupby('ethnic_group').kind.value_counts(normalize=True).unstack().reindex(index=GROUPS, columns=kinds).fillna(0) * 100
    left = np.zeros(len(GROUPS))
    for k, c in zip(kinds, kcol):
        ax.barh(y, share[k], left=left, color=c, label=k, height=0.7); left += share[k].to_numpy()
    ax.set_yticks(y, [] if field == 'last_name' else [SHORT[g] for g in GROUPS]); ax.invert_yaxis()
    ax.set_xlim(0, 100); ax.set_xlabel('Share of changes (%)'); ax.set_title(title, loc='left')
h, l = axs[2].get_legend_handles_labels()
fig.legend(h, l, frameon=False, loc='lower center', ncol=6, fontsize=6.3)
fig.tight_layout(rect=[0, 0.07, 1, 1]); save(fig, 'Fig2')

# ---- Fig 3: overall MMR at matched FMR bounds ----
r = pd.read_csv(RUN / 'matched' / 'group_rates_matched.csv')
minor = [g for g in GROUPS if g != 'Non-Hispanic White']
fig, axs = plt.subplots(1, 4, figsize=(W, 2.6), sharey=False)
bounds = ['FMR<=0.05% (matched overall FMR)', 'FMR<=0.10% (matched overall FMR)']
for i, sens in enumerate(['all', 'errorsonly']):
    d = r[(r.sensitivity == sens) & (r.scenario == 'S3') & r.operating_point.isin(bounds)]
    ov = d[d.ethnic_group == 'All'].groupby(['model', 'operating_point']).mmr.agg(['mean', 'std']).mul(100)
    m = d[d.ethnic_group.isin(minor)].assign(ex=lambda q: q.mmr_gap_pp.clip(lower=0))
    ex = m.groupby(['model', 'operating_point', 'replicate']).apply(lambda q: np.average(q.ex, weights=q.n), include_groups=False)
    ex = ex.groupby(['model', 'operating_point']).agg(['mean', 'std'])
    for j, (tab, lab) in enumerate([(ov, 'Overall missed matches (%)'), (ex, 'Excess missed matches,\nminoritised groups (pp)')]):
        ax = axs[2 * i + j]
        for k, mod in enumerate(MODEL):
            for b, bd in enumerate(bounds):
                v = tab.loc[(mod, bd)]
                ax.errorbar(b + (k - 2) * 0.12, v['mean'], yerr=v['std'], fmt='o', ms=3.5, color=COL[mod], lw=0.8,
                            label=MODEL[mod] if (b == 0 and i == 0 and j == 0) else None)
        ax.set_xticks([0, 1], ['0.05%', '0.1%']); ax.set_xlabel('False match rate bound'); ax.set_xlim(-0.6, 1.6); ax.set_ylim(bottom=0)
        ax.set_ylabel(lab); ax.set_title(('A  ' if i == 0 else 'C  ') if j == 0 else ('B  ' if i == 0 else 'D  '), loc='left')
        ax.text(0.5, 1.02, 'All name changes' if i == 0 else 'Recording errors only', transform=ax.transAxes, ha='center', fontsize=6.5, color='#555')
fig.legend(loc='lower center', ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=[0, 0.08, 1, 1]); save(fig, 'Fig3')

# ---- Fig 4: group differences from NH White at calibration-selected FMR<=0.1% ----
x = pd.read_csv(RUN / 'results' / 'operating_points_with_disparities.csv.gz')
c = x.query("selection=='calibration' and target_kind=='fmr' and target==0.001 and sensitivity=='all'")
g = c[c.ethnic_group.isin(minor)].groupby(['model', 'scenario', 'ethnic_group']).mmr_disparity_pp.agg(['mean', 'std'])
fig, axs = plt.subplots(1, 3, figsize=(W, 2.8), sharey=True, sharex=True)
yy = np.arange(len(minor))
for ax, mod in zip(axs, ['jw', 'jw_tf', 'hybrid']):
    for scen, off, col, lab in [('S1', -0.15, '#9e9e9e', 'S1: equal exposure'), ('S3', 0.15, COL[mod], 'S3: unequal exposure')]:
        v = g.loc[(mod, scen)].reindex(minor)
        ax.errorbar(v['mean'], yy + off, xerr=v['std'], fmt='o', ms=3.5, color=col, lw=0.8, label=lab)
    ax.axvline(0, color='k', lw=0.6); ax.set_yticks(yy, [SHORT[q] for q in minor]); ax.invert_yaxis()
    ax.set_title(MODEL[mod], loc='left'); ax.set_xlabel('Difference in missed matches\nvs NH White (pp)')
    ax.legend(frameon=False, fontsize=6, loc='lower right')
fig.tight_layout(); save(fig, 'Fig4')

# ---- Fig 5: odds ratios for missed match before/after pre-linkage adjustment (JW-TF) ----
if (RUN / 'prelinkage' / 'outcome_models.csv').stat().st_size < 10:  # e.g. demo data: too few events to model
    print('Fig 5 skipped: no outcome-model estimates'); print('figures written to', OUT); sys.exit()
mo = pd.read_csv(RUN / 'prelinkage' / 'outcome_models.csv')
mo = mo[(mo.sensitivity == 'all') & (mo.scenario == 'S3') & (mo.outcome == 'missed')]
mo['g'] = mo.term.str.extract(r'\[T\.(.*)\]')[0]
mo = mo[mo.g.isin(minor)]
if mo.empty:  # e.g. demo data: too few events for the outcome models
    print('Fig 5 skipped: no outcome-model estimates'); print('figures written to', OUT); sys.exit()
fig, axs = plt.subplots(1, 2, figsize=(W, 2.7), sharey=True)
for ax, mod in zip(axs, ['jw', 'jw_tf']):
    d = mo[mo.model == mod]
    for spec, off, col, lab in [('groups_only', -0.15, '#9e9e9e', 'Ethnic group only'),
                                ('groups_plus_prelinkage', 0.15, COL[mod], '+ pre-linkage name measures')]:
        v = d[d.spec == spec].set_index('g').reindex(minor)
        ax.errorbar(v.odds_ratio, yy + off, xerr=[v.odds_ratio - v.lo, v.hi - v.odds_ratio], fmt='o', ms=3.5, color=col, lw=0.8, label=lab)
    ax.axvline(1, color='k', lw=0.6); ax.set_xscale('log'); ax.set_xticks([0.1, 0.2, 0.5, 1], ['0.1', '0.2', '0.5', '1'])
    ax.set_yticks(yy, [SHORT[q] for q in minor]); ax.invert_yaxis(); ax.set_title(MODEL[mod], loc='left')
    ax.set_xlabel('Odds ratio for missed match vs NH White\n(altered queries, S3)'); ax.legend(frameon=False, fontsize=6, loc='lower left')
fig.tight_layout(); save(fig, 'Fig5')
print('figures written to', OUT)
