# Definition-dependent prediction of DXA-defined low muscle mass

Reproducibility resources for the manuscript:

**Definition-Dependent Low-Muscle Phenotypes: Temporal and Cross-Population Validation Using Simple Anthropometry**

This repository contains frozen model objects, locked thresholds, aggregate U.S. temporal-validation results, KNHANES external-validation code, non-participant-level output tables, figures, quality-assurance records, and software-environment information.

## Study structure

- **Development:** NHANES 1999–2006, age 20–59 years.
- **Temporal validation:** NHANES 2011–2018, age 20–59 years.
- **Primary external validation:** KNHANES 2008–2011, age 20–59 years.
- **Contemporary external validation:** KNHANES 2024, age 40–59 years.
- **Predictors:** age, sex, height, weight, waist circumference.
- **Targets:** conditional ALM P5, EWGSOP2 low ALMI, and FNIH low ALM/BMI.
- **Model classes:** linear logistic regression, spline logistic regression, and a locked shallow neural network.
- **External validation:** no Korean refitting, variable reselection, hyperparameter tuning, recalibration, or threshold optimization.

## Important data-access note

Participant-level KNHANES analytic data are **not included** in this repository and must not be redistributed. Researchers wishing to rerun the external-validation scripts must obtain KNHANES data through the Korea Disease Control and Prevention Agency and configure local paths as described in `docs/DATA_ACCESS.md`.

No participant-level NHANES or KNHANES records are included in this public release.

## Repository layout

- `code/knhanes/` — sanitized external-validation scripts.
- `models/` — frozen U.S. model objects and development-defined thresholds used in Korean validation.
- `results/us_temporal_validation/` — aggregate U.S. locked-analysis outputs.
- `results/knhanes/tables/` — aggregate Korean external-validation outputs and bootstrap summaries.
- `results/knhanes/figures/` — external-validation figures.
- `qa/` — analysis manifests, logs, label/direction QA, and integrity records.
- `config/` — configuration notes for non-public local inputs needed for full reruns.
- `docs/` — data-access, reproduction, and release instructions.
- `data_private/` — intentionally empty except for a README; participant-level data belong here locally and are ignored by Git.

## Relationship to the earlier U.S. screening repository

The earlier U.S. model-development and screening resource is available at:

https://github.com/xukaifs/dxa-low-muscle-screening

Its archived Zenodo release is:

https://doi.org/10.5281/zenodo.22228403

The present repository is a **new paper-specific public release** focused on definition-dependent prediction, U.S. temporal validation, and frozen Korean external validation.

The conditional-P5 scoring framework depends on the separately archived ALM reference scorer:

https://github.com/xukaifs/nhanes-alm-reference-scorer

## Reproduction

See `docs/REPRODUCTION.md`.

The public release intentionally separates:
1. public code/model objects/aggregate results, and
2. participant-level data that users must obtain from the original data custodians.

## Integrity

Run:

```bash
python tools_verify_release.py
```

from the repository root to verify all files listed in `SHA256SUMS.txt`.

## Citation

Use `CITATION.cff` for authorship metadata. After the first Zenodo archive is created, cite the **version-specific Zenodo DOI** for the exact released files.

## License

Repository-authored code and documentation are released under the MIT License. NHANES and KNHANES source data remain subject to their original data-provider terms and are not redistributed here.
