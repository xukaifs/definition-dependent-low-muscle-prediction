# Reproduction guide

## 1. Install software

Python environment used for the archived external validation:
- Python 3.12.14
- numpy 2.5.2
- pandas 3.0.5
- scipy 1.18.1
- scikit-learn 1.9.0
- joblib 1.5.3

R cohort-preparation environment:
- R 4.2.1
- haven 2.5.4
- gamlss 5.4-22
- gamlss.dist 6.1-1

Install Python dependencies:

```bash
python -m pip install -r requirements.txt
```

## 2. Obtain data locally

See `docs/DATA_ACCESS.md`. Do not place participant-level KNHANES data in a public Git commit.

## 3. Configure paths

The sanitized R cohort-preparation script uses environment variables:

- `PAPER3_KNHANES_SOURCE_ROOT`
- `PAPER1_ALM_SCORER_ROOT`
- `PAPER3_PRIVATE_DATA_ROOT` (optional; defaults to `data_private/`)

The validation scripts use the repository-local frozen model objects and aggregate-output folders, but some complete subgroup/common-support sensitivity analyses require local U.S. development artifacts. If rerunning those analyses, configure:

- `PAPER3_US_RESULT_ROOT`
- `PAPER3_US_SOURCE_ROOT`

These local inputs are not redistributed here.

## 4. Run

Typical order:

```text
Rscript code/knhanes/00_prepare_knhanes_external_cohorts.R
python code/knhanes/01_run_frozen_knhanes_validation.py
python code/knhanes/02_add_calibration_ci.py
python code/knhanes/04_add_secondary_figures.py
```

The authoritative released results are already stored under `results/`.

## 5. Verify direction and integrity

Review:
- `qa/14_direction_and_label_QA.csv`
- `qa/13_final_QA.csv`
- `qa/technical_direction_calibration_audit.txt` (if present)
- `SHA256SUMS.txt`

Then run:

```bash
python tools_verify_release.py
```

## Scope of reproducibility

Because KNHANES participant-level analysis files are not redistributed, this public repository provides a transparent, auditable release rather than a push-button data bundle. Authorized users who obtain the source data can reproduce the external-validation workflow using the supplied scripts and frozen models.
