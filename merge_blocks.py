"""Merge parallel replicate blocks into one run folder.

Checks that every block used the same fitted model, evaluation cohort and open-cohort reference
(byte-identical files), then combines per-replicate outputs and re-runs the pipeline's aggregation.
Usage: python merge_blocks.py OUT BLOCK_DIR [BLOCK_DIR ...]
"""
import gzip
import hashlib
import shutil
import sys
from pathlib import Path
import pandas as pd
from run import aggregate
from audit import save

IDENTICAL = ['model/weights.json', 'model/first_name_pca.json', 'model/last_name_pca.json',
             'simulation/reference.csv.gz', 'simulation/distractors.csv.gz', 'simulation/open_cohort_counts.csv']


def sha(p):
    # Hash content, not container: gzip headers embed a timestamp.
    data = Path(p).read_bytes()
    if str(p).endswith('.gz'):
        data = gzip.decompress(data)
    return hashlib.sha256(data).hexdigest()


def main(out, blocks):
    out = Path(out); blocks = [Path(b) for b in blocks]
    for b in blocks:
        if not (b / 'COMPLETE.txt').is_file():
            raise SystemExit(f'Block not complete: {b}')
    for rel in IDENTICAL:
        hashes = {sha(b / rel) for b in blocks}
        if len(hashes) != 1:
            raise SystemExit(f'Blocks differ in {rel}; refusing to merge')
    out.mkdir(parents=True, exist_ok=False)
    for sub in ['predictions', 'results', 'simulation', 'model', 'profiles', 'clean']:
        (out / sub).mkdir(exist_ok=True)
    first = blocks[0]
    for f in (first / 'clean').glob('*.csv.gz'):
        shutil.copy2(f, out / 'clean' / f.name)
    for f in (first / 'profiles').glob('*'):
        shutil.copy2(f, out / 'profiles' / f.name)
    for rel in IDENTICAL + ['simulation/evaluation_group_counts.csv', 'model/training_counts.csv', 'config.json']:
        shutil.copy2(first / rel, out / rel)
    ops, curves, block_rows, exposure, joint = [], [], [], [], []
    for b in blocks:
        for f in (b / 'predictions').glob('*.csv.gz'):
            shutil.copy2(f, out / 'predictions' / f.name)
        for f in (b / 'results').glob('*_operating.csv.gz'):
            shutil.copy2(f, out / 'results' / f.name); ops.append(pd.read_csv(f))
        for f in (b / 'results').glob('*_by_change.csv.gz'):
            shutil.copy2(f, out / 'results' / f.name)
        curves.append(pd.read_csv(b / 'results' / 'threshold_curves.csv.gz'))
        block_rows.append(pd.read_csv(b / 'results' / 'blocking.csv'))
        exposure.append(pd.read_csv(b / 'simulation' / 'realised_exposure.csv'))
        joint.append(pd.read_csv(b / 'simulation' / 'joint_exposure.csv'))
    ops = pd.concat(ops, ignore_index=True)
    key = ['sensitivity', 'blocking', 'scenario', 'replicate', 'selection', 'target_kind', 'target', 'model', 'ethnic_group']
    if ops.duplicated(key).any():
        raise SystemExit('Duplicate operating-point rows across blocks')
    aggregate(ops, out)
    save(pd.concat(curves, ignore_index=True), out / 'results' / 'threshold_curves.csv.gz')
    save(pd.concat(block_rows, ignore_index=True), out / 'results' / 'blocking.csv')
    save(pd.concat(exposure, ignore_index=True), out / 'simulation' / 'realised_exposure.csv')
    save(pd.concat(joint, ignore_index=True), out / 'simulation' / 'joint_exposure.csv')
    (out / 'MERGED_FROM.txt').write_text('\n'.join(str(b.resolve()) for b in blocks) + '\n')
    print('merged', len(blocks), 'blocks;', ops.replicate.nunique(), 'replicates;', sorted(ops.sensitivity.unique()))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2:])
