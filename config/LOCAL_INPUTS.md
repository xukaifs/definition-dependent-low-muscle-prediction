# Non-public/local inputs needed for full reruns

The archived Korean validation outputs are public aggregate results, but full reruns require local source inputs.

The sanitized scripts recognize the following environment variables:

- `PAPER3_KNHANES_SOURCE_ROOT` — local authorized KNHANES source/preparation directory.
- `PAPER1_ALM_SCORER_ROOT` — local copy of `nhanes-alm-reference-scorer`.
- `PAPER3_PRIVATE_DATA_ROOT` — local output directory for analysis-ready KNHANES data.
- `PAPER3_US_RESULT_ROOT` — local U.S. locked-result directory for auxiliary cutpoint files used in subgroup analyses.
- `PAPER3_US_SOURCE_ROOT` — local U.S. development-person file root used for common-support sensitivity analyses.

These local inputs must not be committed if they contain participant-level records.
