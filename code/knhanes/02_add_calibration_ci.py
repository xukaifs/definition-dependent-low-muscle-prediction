from __future__ import annotations

from pathlib import Path
import os
import hashlib
import math

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("PAPER3_PRIVATE_DATA_ROOT", ROOT / "data_private"))
TABLES = ROOT / "results" / "knhanes" / "tables"
FIGURES = ROOT / "results" / "knhanes" / "figures"
MODEL_DIR = Path(r"D:\DXA身体成分分析\R分析结果\测算数据\paper3_final_locked_reanalysis_20260930\models")
FEATURES = ["age", "sex_male", "height_cm", "weight_kg", "waist_cm"]
TARGETS = {"P5": "p5_raw", "EWGSOP2": "ewgsop2_raw", "FNIH": "fnih_raw"}
MODELS = {"Logistic": "Linear logistic", "RCS Logistic": "Spline logistic", "Neural Network": "Neural network"}
B = 500
SEED = 20261001


def wmean(x, w):
    x, w = np.asarray(x, float), np.asarray(w, float)
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    return float(np.sum(x[ok] * w[ok]) / np.sum(w[ok])) if ok.any() else math.nan


def weighted_quantile(x, w, probs):
    order = np.argsort(x)
    x, w = np.asarray(x, float)[order], np.asarray(w, float)[order]
    c = np.cumsum(w) / np.sum(w)
    return np.array([x[min(np.searchsorted(c, q, side="left"), len(x) - 1)] for q in probs])


def bootstrap_multipliers(frame, rng):
    multiplier = np.zeros(len(frame), float)
    for _, stratum in frame.groupby("strata_pool", sort=False):
        psus = stratum["psu_pool"].drop_duplicates().to_numpy()
        counts = pd.Series(rng.choice(psus, len(psus), replace=True)).value_counts()
        for psu, count in counts.items():
            multiplier[stratum.index[stratum["psu_pool"] == psu]] = float(count)
    return multiplier


def load_models():
    models = {}
    for target in TARGETS:
        for model in MODELS:
            safe = model.lower().replace(" ", "_").replace("-", "_")
            obj = joblib.load(MODEL_DIR / f"stage2_{target.lower()}_{safe}_checkpoint.joblib")
            models[(target, model)] = obj["final_model"]
    return models


cohorts = {
    "KNHANES 2008-2011": pd.read_csv(DATA / "knhanes_2008_2011_paper3_ready.csv.gz"),
    "KNHANES 2024": pd.read_csv(DATA / "knhanes_2024_paper3_ready.csv.gz"),
}
models = load_models()
all_rows = []

for cohort, frame in cohorts.items():
    x = frame[FEATURES].to_numpy(float)
    base_w = frame["survey_weight"].to_numpy(float)
    rng = np.random.default_rng(SEED + (24 if "2024" in cohort else 8))
    multipliers = [bootstrap_multipliers(frame, rng) for _ in range(B)]
    for target, outcome in TARGETS.items():
        y = frame[outcome].to_numpy(int)
        for model, label in MODELS.items():
            p = np.clip(models[(target, model)].predict_proba(x)[:, 1], 1e-7, 1 - 1e-7)
            cuts = np.unique(weighted_quantile(p, base_w, np.linspace(0, 1, 11)))
            bins = np.clip(np.searchsorted(cuts, p, side="right") - 1, 0, max(0, len(cuts) - 2))
            for bin_id in np.unique(bins):
                mask = bins == bin_id
                predicted, observed = [], []
                for multiplier in multipliers:
                    w = base_w * multiplier
                    predicted.append(wmean(p[mask], w[mask]))
                    observed.append(wmean(y[mask], w[mask]))
                all_rows.append({
                    "cohort": cohort,
                    "target": target,
                    "model": label,
                    "bin": int(bin_id + 1),
                    "participants": int(mask.sum()),
                    "weighted_population_share": float(base_w[mask].sum() / base_w.sum()),
                    "weighted_mean_prediction": wmean(p[mask], base_w[mask]),
                    "mean_prediction_ci_lower_95": float(np.nanpercentile(predicted, 2.5)),
                    "mean_prediction_ci_upper_95": float(np.nanpercentile(predicted, 97.5)),
                    "weighted_observed_prevalence": wmean(y[mask], base_w[mask]),
                    "observed_prevalence_ci_lower_95": float(np.nanpercentile(observed, 2.5)),
                    "observed_prevalence_ci_upper_95": float(np.nanpercentile(observed, 97.5)),
                    "bootstrap_valid_replicates": int(np.isfinite(observed).sum()),
                })

calibration = pd.DataFrame(all_rows)
calibration.to_csv(TABLES / "03_calibration_curve_points.csv", index=False)

# Definition prevalence and design-based confidence intervals, deduplicated from the
# model-performance table because prevalence is independent of model class.
performance = pd.read_csv(TABLES / "02_model_performance_with_ci.csv")
prevalence = performance[(performance["metric"] == "weighted_prevalence") &
                         (performance["model"] == "Linear logistic")].copy()
prevalence = prevalence[["cohort", "target", "participants", "estimate", "ci_lower_95",
                         "ci_upper_95", "bootstrap_valid_replicates"]]
prevalence = prevalence.rename(columns={"estimate": "weighted_prevalence"})
prevalence.to_csv(TABLES / "05_definition_prevalence.csv", index=False)

# Calibration plots with 95% design-based bootstrap intervals around observed risk.
colors = {"Linear logistic": "#0072B2", "Spline logistic": "#009E73", "Neural network": "#D55E00"}
for cohort in cohorts:
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.7))
    for axis, target in zip(axes, TARGETS):
        subset = calibration[(calibration.cohort == cohort) & (calibration.target == target)]
        upper = max(0.05, subset[["weighted_mean_prediction", "observed_prevalence_ci_upper_95"]].to_numpy().max() * 1.08)
        axis.plot([0, upper], [0, upper], linestyle="--", color="0.45", linewidth=1.2, label="Ideal")
        for model in MODELS.values():
            d = subset[subset.model == model].sort_values("bin")
            lower_error = d.weighted_observed_prevalence - d.observed_prevalence_ci_lower_95
            upper_error = d.observed_prevalence_ci_upper_95 - d.weighted_observed_prevalence
            axis.errorbar(d.weighted_mean_prediction, d.weighted_observed_prevalence,
                          yerr=np.vstack([lower_error, upper_error]), marker="o", markersize=4,
                          linewidth=1.4, capsize=2, color=colors[model], label=model)
        axis.set(xlabel="Mean predicted probability", ylabel="Observed survey-weighted prevalence",
                 title=target, xlim=(0, upper), ylim=(0, upper))
        axis.grid(alpha=0.18)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle(f"Calibration in {cohort} (95% stratified-PSU bootstrap CI)", fontsize=14)
    fig.tight_layout(rect=(0, 0.10, 1, 0.94))
    fig.savefig(FIGURES / f"Figure_calibration_{cohort.replace(' ', '_')}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

# Refresh file hashes after all post-processing. The manifest does not hash itself.
manifest_path = ROOT / "SHA256_MANIFEST.csv"
rows = []
for path in sorted(ROOT.rglob("*")):
    if path.is_file() and path != manifest_path:
        rows.append({"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size,
                     "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
pd.DataFrame(rows).to_csv(manifest_path, index=False)
print(f"Calibration CI complete: {len(calibration)} rows; {B} replicates per cohort.")
