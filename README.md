# Ethnic disparities in name-based record linkage: NCVR simulation pipeline

Code for the simulation study *"Unequal exposure to name changes drives ethnic disparities in record linkage, and
term-frequency weighting reduces them: a simulation study using North Carolina voter records"* (Lam J, Cortina-Borja M,
Aldridge RW, Blackburn R, Harron K; under review).

The pipeline:

1. **Profiles** within-person name changes by ethnic group from consecutive North Carolina Voter Registry (NCVR)
   snapshots linked by NCID (change type, edit distance, position).
2. **Assesses names before linkage**: dominance (name frequency, concentration), cardinality (homonyms and shared
   surnames within the blocking key) and ambiguity (one-edit neighbours), by ethnic group.
3. **Simulates name changes** in an evaluation cohort under a 2x2 design: pooled vs group-specific change profiles,
   equal vs unequal exposure. Changes follow the observed type mix; a recording-errors-only sensitivity excludes
   genuine name changes.
4. **Links** altered records to an **open** reference file (people without a true match plus extra non-matching
   records) with five models: Jaro-Winkler and Levenshtein, each with and without term-frequency (TF) adjustment, and a
   mixed logistic model adding name-structure features.
5. **Evaluates** missed and false match rates by ethnic group at thresholds chosen on calibration records, and relates
   errors to the pre-linkage name measures.

No data are included. The NCVR snapshots are public: <https://dl.ncsbe.gov/index.html?prefix=data/Snapshots/>.

## Files

| File | Purpose |
|---|---|
| `audit.py` | Read snapshots, NCID audit, eligibility, ethnicity coding, NCID-validated joins |
| `simulate.py` | Change profiles, change-type classification, the 2x2 simulation |
| `linkage.py` | Features, model fitting, blocking, one-to-one linking, metrics, threshold selection |
| `run.py` | Runs every stage and writes all outputs and a manifest |
| `matched.py` | Group results at matched false match rate bounds (descriptive benchmark) |
| `prelinkage.py` | Pre-linkage dominance, cardinality and ambiguity; outcome models |
| `figures.py` | Manuscript figures |
| `merge_blocks.py` | Merge replicate blocks run in parallel |
| `make_demo.py`, `tests/` | Fictional demo data and unit tests |
| `config_ncvr.json` | Configuration used for the paper (12 October snapshots, 2011-2022) |
| `config.example.json` | Template configuration |
| `docs/DESIGN.md` | Pre-specified design and pre-run amendments |

## Setup

Python 3.14 was used for the paper (3.12+ should work).

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt      # Windows; use .venv/bin/python on macOS/Linux
.venv/Scripts/python -m pytest tests -q
```

## Demo (fictional data, a few minutes)

```bash
python make_demo.py --output demo_data
python run.py --config demo_data/config.json --output demo_results/run
python matched.py demo_results/run
python prelinkage.py demo_results/run
python figures.py demo_results/run demo_results/figures
```

Demo results have no substantive meaning.

## Full NCVR analysis

1. Download the October snapshots for 2011-2022 into `data/` and check the file names, delimiter, encoding and column
   names in `config_ncvr.json` against your extracts (`read_csv`, `columns`). Relative paths are resolved from the
   configuration file's folder.
2. Run the pipeline (reading and cleaning 12 snapshots takes about 30 minutes; then about 4 minutes per replicate per
   sensitivity):

   ```bash
   python run.py --config config_ncvr.json --output results/main
   ```

   To run replicate blocks in parallel, copy the configuration with `"replicates": 10` and `"replicate_start": 0` or
   `10` (and one sensitivity each if you wish), run each block to its own output folder, then merge. The merge checks
   that all blocks used identical models, cohort and reference files:

   ```bash
   python run.py --config cfg_block0.json  --output results/block0
   python run.py --config cfg_block10.json --output results/block10 --reuse-clean results/block0
   python merge_blocks.py results/main results/block0 results/block10
   ```

   `--reuse-clean` copies the cleaned snapshots from an earlier run instead of re-reading the raw files.
3. Analyse and plot:

   ```bash
   python matched.py results/main
   python prelinkage.py results/main
   python figures.py results/main results/figures
   ```

## Key settings (`config_ncvr.json`)

| Setting | Value | Meaning |
|---|---|---|
| `profile_pairs` | 10 consecutive pairs, 2011-2021 | Source of observed name changes |
| `evaluation_max_records`, `evaluation_min_per_group` | 20,000; 1,000 | Evaluation cohort; small groups topped up |
| `open_distractors`, `open_nonmatch_fraction` | 30,000; 0.10 | Open cohort: extra non-matching references; share of people without a true match |
| `alteration_rates`, `relative_rates` | 10% per field; 1.0-2.0 | Overall target and relative exposure (scenarios S3, S4) |
| `generator` | `typed` | Change type from the profile; observed variant of the same name and type where available |
| `frequency_reference` | `full_frame` | TF weights from the full evaluation-year frame |
| `sensitivities` | `all`, `errorsonly` | All name changes; recording errors only (no replacement by a different name) |
| `replicates` | 20 | Independent simulation replicates |
| `audit_all_pairs` | `false` | Skip the optional all-pairs NCID audit (diagnostics only; slow) |

## Outputs

Each run folder contains `audit/`, `profiles/`, `model/`, `simulation/`, `results/` (operating points by group, threshold
curves, blocking recall, paired comparisons), `predictions/` and `manifest.json` (input hashes, package versions).
`results/operating_points_with_disparities.csv.gz` holds every calibration-selected and descriptive operating point
by sensitivity, scenario, replicate, model and ethnic group.

**Outputs contain NCIDs and names.** They are excluded by `.gitignore`; keep them in your approved local workspace and
share only aggregate results.

## Notes and limitations

NCID is used to align records and define true matches, not as a predictor or independent gold standard. Simulated
exposure rates are experimental choices, not estimates of real error rates. Positive training pairs are synthetic.
See the paper and `docs/DESIGN.md` for the full design and limitations.

## Citation

Please cite the paper (details to be added on publication) and this repository.

## Licence

MIT (see `LICENSE`).
