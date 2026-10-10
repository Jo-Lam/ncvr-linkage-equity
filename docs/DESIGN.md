# Design v2 — pre-specified before any v2 results were produced (9 October 2026)

An earlier closed-cohort version (v1) was the primary
analysis of an earlier manuscript draft. v2 changes four design elements that were identified as
limitations of v1. All v2 results will be reported, whichever direction they point.

## Changes

1. **Frequency reference = full October 2022 eligible frame** (n = 8,082,366), replacing the
   50,000-record training sample, for term-frequency weights and for the name/prefix-count PCA
   features. PCA loadings, scaling and m/u estimation still use the 50,000-record training sample.
   (`frequency_reference: "full_frame"`)

2. **Type-aware name-change generator** (`generator: "typed"`). Each observed profile event is
   classified (typographic <=2 edits; truncation/extension; token added/dropped; spacing/hyphen;
   token reorder; initial <-> full; replacement by a different name; other larger edit). When the
   target name itself has observed variants in the relevant template pool (pooled or group-specific),
   one is drawn directly. Otherwise a donor event is drawn and applied by type: typographic and
   other edits by relative position (as v1); truncation keeps the same proportional prefix;
   extension appends the donor's added suffix; token drop removes the token at the same relative
   position; token add inserts the donor's added token on the same side; spacing/hyphen swaps
   separators; reorder reverses tokens; initial keeps the first letter; replacement substitutes the
   donor's real new name. Inapplicable types are redrawn (max 20 attempts, as v1); the realised
   type mix is reported. Synthetic training positives use the same generator.

3. **Oversampling of small groups** (`evaluation_min_per_group: 1000`). The fixed 20,000-record
   simple random sample is topped up, by stable hash order, to at least 1,000 records in every
   ethnic group. Group-specific rates are the estimands; overall rates are reported unweighted and
   flagged as not population-representative.

4. **Open cohort.** (a) `open_distractors: 30000` additional clean 2022 reference records with no
   query counterpart (NCID-disjoint from evaluation and training); (b) `open_nonmatch_fraction: 0.10`
   of evaluation people have their clean record removed from the reference, so their queries have
   no true match. MMR = missed / matchable queries; FMR = wrong accepted links / all accepted links
   (now including any link made for a non-matchable query); non-matchable false-link rate reported
   separately. Blocking recall uses matchable queries only.

Everything else as v1: ten profile pairs; four scenarios; 10% per field; same relative exposure
rates; six models; demographic blocking; all-edits sensitivity; calibration/test split; MMR and
FMR targets; 20 replicates.

## Primary comparisons (fixed now)

- Exposure vs mechanism: JW-TF group MMR differences from Non-Hispanic White, S1/S2 vs S3/S4,
  calibration-selected 5% MMR target.
- TF: JW vs JW-TF overall MMR and excess MMR at matched held-out FMR bounds 0.01%, 0.05%, 0.1%,
  and at the calibration-selected 0.5% FMR target.
- Name-feature model vs JW-TF: the same quantities, plus group FMR.
- Pre-linkage measures: same definitions and models as v1 (`prelinkage_assessment.py`).

Each change's contribution is not separately identified in the main run; if needed, one-factor
ablations will be run afterwards and labelled as such.

## Amendments after the one-replicate test (9 October 2026), before the 20-replicate run

Made after inspecting a one-replicate test run. Each is motivated by fidelity or
feasibility, not by which model it favours.

1. **Generator: change type drawn from the profile first.** In the test, drawing an observed variant
   of the target name whenever one existed over-represented replacements for common forenames
   (Non-Hispanic White 34.6% simulated vs 23.8% observed). Now each attempt draws a donor event
   (and hence its type) from the pooled or group profile; an observed variant of the target name
   *of that type* is used when available, otherwise the type is applied by rule. Rule-based
   replacements draw a same-gender replacement name from the same pool. Realised replacement shares
   are within about 3 percentage points of the profile for most groups (Hispanic surnames 58% vs 47%,
   because token types inapplicable to single-token names are redrawn).
2. **MMR target for the exposure comparison: 10% instead of 5%.** In the open cohort the minimum
   achievable MMR was 5.4-8.1% (S3), so 5% is unattainable for most models. FMR-bound comparisons
   remain primary.
3. **Errors-only sensitivity** (`sensitivities: ["all", "errorsonly"]`): replacement-type events
   (different name, Jaro-Winkler < 0.70) are excluded from the template bank, separating recording
   errors from genuine name changes. Models are the same as in the main analysis.
4. **Execution:** replicates run in parallel blocks (`replicate_start`), merged afterwards. Seeds
   depend only on replicate index, so results are identical to a single sequential run.

## Post-results reporting decision (10 October 2026)

After the 20-replicate run, the authors chose to present name-structure features only through the mixed (hybrid)
model in the manuscript. The standalone name-feature comparator remains in the code (`feature_surname` in
`linkage.py`) and its results are reported in the supporting tables.
