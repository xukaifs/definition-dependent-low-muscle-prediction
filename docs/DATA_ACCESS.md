# Data access

## NHANES
NHANES source data are publicly available from the U.S. National Center for Health Statistics. Participant-level NHANES data are not duplicated in this repository.

## KNHANES
KNHANES participant-level data are not redistributed in this repository. Researchers should obtain the relevant survey files directly from the Korea Disease Control and Prevention Agency under its current data-access rules.

## Why the public repository excludes KNHANES participant-level files
The public release contains:
- analysis code,
- frozen model objects,
- thresholds,
- aggregate numeric outputs,
- bootstrap summary/replicate outputs,
- figures,
- technical QA records.

It intentionally excludes analysis-ready participant-level KNHANES files.

## Conditional-P5 reference scorer
Construction of the frozen U.S.-reference conditional-P5 outcome requires the separately released ALM reference scorer:
https://github.com/xukaifs/nhanes-alm-reference-scorer

Configure its local path using `PAPER1_ALM_SCORER_ROOT` before running the cohort-preparation script.
