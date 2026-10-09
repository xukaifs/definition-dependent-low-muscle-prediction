from __future__ import annotations

import hashlib
import json
import math
import platform
import shutil
import sys
import time
from pathlib import Path
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.optimize import brentq
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve, precision_recall_curve
from scipy.special import expit, logit


ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("PAPER3_PRIVATE_DATA_ROOT", ROOT / "data_private"))
TABLES = ROOT / "results" / "knhanes" / "tables"
FIGURES = ROOT / "results" / "knhanes" / "figures"
LOGS = ROOT / "qa"
for directory in (TABLES, FIGURES, LOGS):
    directory.mkdir(parents=True, exist_ok=True)

US_RESULT = Path(r"D:\DXA身体成分分析\R分析结果\测算数据\paper3_final_locked_reanalysis_20260930")
US_SOURCE = Path(r"D:\DXA身体成分分析\R分析结果\测算数据\paper3_complete_v2_20260827")
MODEL_DIR = ROOT / "models"
FEATURES5 = ["age", "sex_male", "height_cm", "weight_kg", "waist_cm"]
FEATURES4 = ["age", "sex_male", "height_cm", "weight_kg"]
TARGETS = {"P5": "p5_raw", "EWGSOP2": "ewgsop2_raw", "FNIH": "fnih_raw"}
TARGETS_CAL = {"P5": "p5_calibrated_0946", "EWGSOP2": "ewgsop2_calibrated_0946", "FNIH": "fnih_calibrated_0946"}
MODELS = {"Logistic": "Linear logistic", "RCS Logistic": "Spline logistic", "Neural Network": "Neural network"}
SEED = 20261001
B = 500


def log(message: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    with (LOGS / "01_run_frozen_knhanes_validation.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def wmean(x, w):
    x, w = np.asarray(x, float), np.asarray(w, float)
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    return float(np.sum(x[ok] * w[ok]) / np.sum(w[ok])) if ok.any() else math.nan


def metrics(y, p, w):
    y, p, w = np.asarray(y, int), np.asarray(p, float), np.asarray(w, float)
    ok = np.isfinite(y) & np.isfinite(p) & np.isfinite(w) & (w > 0)
    y, p, w = y[ok], np.clip(p[ok], 1e-7, 1 - 1e-7), w[ok]
    prevalence, mean_p = wmean(y, w), wmean(p, w)
    out = {
        "weighted_prevalence": prevalence,
        "weighted_mean_prediction": mean_p,
        "weighted_roc_auc": math.nan,
        "weighted_pr_auc": math.nan,
        "weighted_brier": wmean((y - p) ** 2, w),
        "calibration_intercept": math.nan,
        "calibration_slope": math.nan,
    }
    if len(np.unique(y)) < 2:
        return out
    out["weighted_roc_auc"] = float(roc_auc_score(y, p, sample_weight=w))
    out["weighted_pr_auc"] = float(average_precision_score(y, p, sample_weight=w))
    lp = logit(p)
    try:
        out["calibration_intercept"] = float(brentq(lambda a: np.sum(w * (y - expit(a + lp))), -30, 30))
    except ValueError:
        pass
    try:
        fit = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000, random_state=SEED)
        fit.fit(lp.reshape(-1, 1), y, sample_weight=w / np.mean(w))
        out["calibration_slope"] = float(fit.coef_[0, 0])
    except Exception:
        pass
    return out


def operating(y, p, w, threshold):
    y, test, w = np.asarray(y, int), np.asarray(p) >= threshold, np.asarray(w, float)
    tp, fn = w[(y == 1) & test].sum(), w[(y == 1) & ~test].sum()
    tn, fp = w[(y == 0) & ~test].sum(), w[(y == 0) & test].sum()
    sensitivity = tp / (tp + fn) if tp + fn else math.nan
    specificity = tn / (tn + fp) if tn + fp else math.nan
    ppv = tp / (tp + fp) if tp + fp else math.nan
    npv = tn / (tn + fn) if tn + fn else math.nan
    return {
        "sensitivity": sensitivity, "specificity": specificity, "ppv": ppv, "npv": npv,
        "lr_positive": sensitivity / (1 - specificity) if specificity < 1 else math.inf,
        "lr_negative": (1 - sensitivity) / specificity if specificity > 0 else math.inf,
        "test_positive_proportion": (tp + fp) / (tp + fn + tn + fp),
    }


def multipliers(frame, rng):
    m = np.zeros(len(frame), float)
    for _, stratum in frame.groupby("strata_pool", sort=False):
        psus = stratum["psu_pool"].drop_duplicates().to_numpy()
        counts = pd.Series(rng.choice(psus, len(psus), replace=True)).value_counts()
        for psu, count in counts.items():
            m[stratum.index[stratum["psu_pool"] == psu]] = float(count)
    return m


def weighted_quantile(x, w, probs):
    x, w = np.asarray(x, float), np.asarray(w, float)
    order = np.argsort(x)
    x, w = x[order], w[order]
    c = np.cumsum(w) / np.sum(w)
    return np.array([x[min(np.searchsorted(c, q, side="left"), len(x) - 1)] for q in probs])


def weighted_corr(x, y, w):
    mx, my = wmean(x, w), wmean(y, w)
    return wmean((x - mx) * (y - my), w) / math.sqrt(wmean((x - mx) ** 2, w) * wmean((y - my) ** 2, w))


def load_models():
    five, four = {}, {}
    for target in TARGETS:
        for model in MODELS:
            safe = model.lower().replace(" ", "_").replace("-", "_")
            checkpoint = joblib.load(MODEL_DIR / f"stage2_{target.lower()}_{safe}_checkpoint.joblib")
            five[(target, model)] = checkpoint["final_model"]
            four_obj = joblib.load(MODEL_DIR / f"four_predictor_{target.lower()}_{safe}.joblib")
            four[(target, model)] = four_obj["final_model"]
    return five, four


def add_predictions(frame, five, four):
    x5, x4 = frame[FEATURES5].to_numpy(float), frame[FEATURES4].to_numpy(float)
    predictions5, predictions4 = {}, {}
    for key, model in five.items():
        predictions5[key] = np.clip(model.predict_proba(x5)[:, 1], 1e-7, 1 - 1e-7)
        predictions4[key] = np.clip(four[key].predict_proba(x4)[:, 1], 1e-7, 1 - 1e-7)
    return predictions5, predictions4


def point_performance(cohort, frame, predictions, targets=TARGETS, analysis="Primary raw Korean ALM"):
    rows = []
    w = frame["survey_weight"].to_numpy(float)
    for target, outcome in targets.items():
        y = frame[outcome].to_numpy(int)
        for model, label in MODELS.items():
            row = {"cohort": cohort, "analysis": analysis, "target": target, "model": label, "participants": len(frame)}
            row.update(metrics(y, predictions[(target, model)], w))
            rows.append(row)
    return pd.DataFrame(rows)


def calibration_points(cohort, frame, predictions):
    rows, w = [], frame["survey_weight"].to_numpy(float)
    for target, outcome in TARGETS.items():
        y = frame[outcome].to_numpy(int)
        for model, label in MODELS.items():
            p = predictions[(target, model)]
            cuts = np.unique(weighted_quantile(p, w, np.linspace(0, 1, 11)))
            bins = np.clip(np.searchsorted(cuts, p, side="right") - 1, 0, max(0, len(cuts) - 2))
            for b in np.unique(bins):
                mask = bins == b
                rows.append({"cohort": cohort, "target": target, "model": label, "bin": int(b + 1),
                             "participants": int(mask.sum()), "weighted_population_share": w[mask].sum() / w.sum(),
                             "weighted_mean_prediction": wmean(p[mask], w[mask]),
                             "weighted_observed_prevalence": wmean(y[mask], w[mask])})
    return pd.DataFrame(rows)


def threshold_points(cohort, frame, predictions, thresholds):
    rows, w = [], frame["survey_weight"].to_numpy(float)
    for target, outcome in TARGETS.items():
        y = frame[outcome].to_numpy(int)
        for model, label in MODELS.items():
            p = predictions[(target, model)]
            for _, threshold_row in thresholds[(thresholds.target == target) & (thresholds.model == label)].iterrows():
                threshold_value = threshold_row.frozen_probability_threshold
                vals = operating(y, p, w, threshold_value)
                for metric, estimate in vals.items():
                    rows.append({"cohort": cohort, "target": target, "model": label,
                                 "operating_point": threshold_row.operating_point, "threshold": threshold_value,
                                 "metric": metric, "estimate": estimate})
    return pd.DataFrame(rows)


def bootstrap_all(cohort, frame, predictions5, predictions4, thresholds):
    rng = np.random.default_rng(SEED + (24 if "2024" in cohort else 8))
    base_w = frame["survey_weight"].to_numpy(float)
    perf_rows, op_rows, waist_rows, subgroup_rows, agreement_rows = [], [], [], [], []
    subgroup_masks = build_subgroups(frame)
    for rep in range(1, B + 1):
        w = base_w * multipliers(frame, rng)
        for target, outcome in TARGETS.items():
            y = frame[outcome].to_numpy(int)
            for model, label in MODELS.items():
                p5, p4 = predictions5[(target, model)], predictions4[(target, model)]
                met = metrics(y, p5, w)
                for metric in ["weighted_prevalence", "weighted_mean_prediction", "weighted_roc_auc", "weighted_pr_auc", "weighted_brier", "calibration_intercept", "calibration_slope"]:
                    perf_rows.append({"cohort": cohort, "replicate": rep, "target": target, "model": label, "metric": metric, "value": met[metric]})
                m4 = metrics(y, p4, w)
                for metric in ["weighted_roc_auc", "weighted_brier"]:
                    waist_rows.append({"cohort": cohort, "replicate": rep, "target": target, "model": label, "metric": metric,
                                       "five_minus_four": met[metric] - m4[metric]})
                for _, tr in thresholds[(thresholds.target == target) & (thresholds.model == label)].iterrows():
                    for metric, value in operating(y, p5, w, tr.frozen_probability_threshold).items():
                        op_rows.append({"cohort": cohort, "replicate": rep, "target": target, "model": label,
                                        "operating_point": tr.operating_point, "metric": metric, "value": value})
                for dimension, level, mask in subgroup_masks:
                    cases = float(y[mask].sum())
                    if cases >= 20 and len(np.unique(y[mask])) == 2:
                        auc = roc_auc_score(y[mask], p5[mask], sample_weight=w[mask])
                        subgroup_rows.append({"cohort": cohort, "replicate": rep, "target": target, "model": label,
                                              "dimension": dimension, "level": level, "weighted_roc_auc": auc})
        # Definition agreement is model-independent.
        for a, b in [("P5", "EWGSOP2"), ("P5", "FNIH"), ("EWGSOP2", "FNIH")]:
            ya, yb = frame[TARGETS[a]].to_numpy(int), frame[TARGETS[b]].to_numpy(int)
            agreement = wmean((ya == yb).astype(float), w)
            pa, pb = wmean(ya, w), wmean(yb, w)
            expected = pa * pb + (1 - pa) * (1 - pb)
            intersection = w[(ya == 1) & (yb == 1)].sum()
            union = w[(ya == 1) | (yb == 1)].sum()
            for metric, value in [("weighted_percent_agreement", agreement),
                                  ("weighted_cohen_kappa", (agreement - expected) / (1 - expected)),
                                  ("weighted_jaccard_index", intersection / union)]:
                agreement_rows.append({"cohort": cohort, "replicate": rep, "definition_a": a,
                                       "definition_b": b, "metric": metric, "value": value})
        if rep % 100 == 0:
            log(f"{cohort} stratified-PSU bootstrap {rep}/{B}")
    return tuple(pd.DataFrame(x) for x in (perf_rows, op_rows, waist_rows, subgroup_rows, agreement_rows))


def summarize_ci(detail, keys, value="value"):
    return detail.groupby(keys, observed=True)[value].agg(
        ci_lower_95=lambda x: np.nanpercentile(x, 2.5),
        ci_upper_95=lambda x: np.nanpercentile(x, 97.5),
        bootstrap_valid_replicates=lambda x: int(np.isfinite(x).sum()),
    ).reset_index()


def build_subgroups(frame):
    masks = []
    for sex in ["Male", "Female"]:
        masks.append(("sex", sex, frame.sex_label.eq(sex).to_numpy()))
    for level, condition in [("20-39", frame.age.between(20, 39)), ("40-59", frame.age.between(40, 59))]:
        if condition.any(): masks.append(("age_group", level, condition.to_numpy()))
    for level, condition in [("<25", frame.bmi < 25), ("25-29.9", frame.bmi.between(25, 29.999)), (">=30", frame.bmi >= 30)]:
        masks.append(("bmi_group", level, condition.to_numpy()))
    cut_table = pd.read_csv(US_RESULT / "tables" / "09c_sex_specific_height_tertile_cutpoints.csv")
    for _, row in cut_table.iterrows():
        sex = row.sex
        sub = frame.sex_label.eq(sex)
        low, high = row.lower_cut_cm, row.upper_cut_cm
        for level, condition in [(f"{sex} Short", sub & (frame.height_cm <= low)),
                                 (f"{sex} Middle", sub & (frame.height_cm > low) & (frame.height_cm <= high)),
                                 (f"{sex} Tall", sub & (frame.height_cm > high))]:
            masks.append(("sex_specific_height_tertile", level, condition.to_numpy()))
    return masks


def subgroup_points(cohort, frame, predictions):
    rows, w = [], frame.survey_weight.to_numpy(float)
    for target, outcome in TARGETS.items():
        y = frame[outcome].to_numpy(int)
        for model, label in MODELS.items():
            p = predictions[(target, model)]
            for dimension, level, mask in build_subgroups(frame):
                case_equiv = float(y[mask].sum())
                row = {"cohort": cohort, "target": target, "model": label, "dimension": dimension, "level": level,
                       "participants": int(mask.sum()), "case_equivalents": case_equiv,
                       "weighted_prevalence": wmean(y[mask], w[mask]), "reporting_flag": "REPORTABLE" if case_equiv >= 20 else "SUPPRESSED_LT20_CASE_EQUIVALENTS",
                       "weighted_roc_auc": math.nan, "calibration_slope": math.nan}
                if case_equiv >= 20 and len(np.unique(y[mask])) == 2:
                    met = metrics(y[mask], p[mask], w[mask])
                    row["weighted_roc_auc"], row["calibration_slope"] = met["weighted_roc_auc"], met["calibration_slope"]
                rows.append(row)
    return pd.DataFrame(rows)


def agreement_points(cohort, frame):
    w = frame.survey_weight.to_numpy(float)
    prev, pairs, patterns = [], [], []
    for target, outcome in TARGETS.items():
        prev.append({"cohort": cohort, "target": target, "weighted_prevalence": wmean(frame[outcome], w)})
    for a, b in [("P5", "EWGSOP2"), ("P5", "FNIH"), ("EWGSOP2", "FNIH")]:
        ya, yb = frame[TARGETS[a]].to_numpy(int), frame[TARGETS[b]].to_numpy(int)
        agreement = wmean((ya == yb).astype(float), w)
        pa, pb = wmean(ya, w), wmean(yb, w)
        expected = pa * pb + (1 - pa) * (1 - pb)
        intersection, union = w[(ya == 1) & (yb == 1)].sum(), w[(ya == 1) | (yb == 1)].sum()
        pairs.append({"cohort": cohort, "definition_a": a, "definition_b": b,
                      "weighted_percent_agreement": agreement, "weighted_cohen_kappa": (agreement - expected) / (1 - expected),
                      "weighted_jaccard_index": intersection / union})
    labels = frame[[TARGETS[t] for t in TARGETS]].astype(int).astype(str).agg("".join, axis=1)
    mapping = {"100": "P5", "010": "EWGSOP2", "001": "FNIH", "110": "P5 + EWGSOP2",
               "101": "P5 + FNIH", "011": "EWGSOP2 + FNIH", "111": "P5 + EWGSOP2 + FNIH", "000": "None"}
    for code, index in labels.groupby(labels).groups.items():
        patterns.append({"cohort": cohort, "overlap_pattern": mapping[code], "participants": len(index),
                         "survey_weighted_population_share": w[np.asarray(list(index), int)].sum() / w.sum()})
    return pd.DataFrame(prev), pd.DataFrame(pairs), pd.DataFrame(patterns)


def cross_definition(cohort, frame, predictions):
    w, rows = frame.survey_weight.to_numpy(float), []
    nn = {target: predictions[(target, "Neural Network")] for target in TARGETS}
    for trained, p in nn.items():
        for evaluated, outcome in TARGETS.items():
            rows.append({"cohort": cohort, "trained_target": trained, "evaluated_outcome": evaluated,
                         "weighted_roc_auc": roc_auc_score(frame[outcome], p, sample_weight=w)})
    corr = []
    for a, b in [("P5", "EWGSOP2"), ("P5", "FNIH"), ("EWGSOP2", "FNIH")]:
        ra, rb = rankdata(nn[a]), rankdata(nn[b])
        corr.append({"cohort": cohort, "prediction_a": a, "prediction_b": b,
                     "spearman_unweighted": np.corrcoef(ra, rb)[0, 1],
                     "spearman_survey_weighted": weighted_corr(ra, rb, w)})
    return pd.DataFrame(rows), pd.DataFrame(corr)


def common_support(frame, us):
    keep = np.ones(len(frame), bool)
    rows = []
    for sex in ["Female", "Male"]:
        k = frame.sex_label.eq(sex)
        u = us.sex.eq(sex)
        lower = max(frame.loc[k, "height_cm"].min(), us.loc[u, "height_cm"].min())
        upper = min(frame.loc[k, "height_cm"].max(), us.loc[u, "height_cm"].max())
        keep[k] = frame.loc[k, "height_cm"].between(lower, upper)
        rows.append({"cohort": frame.cohort.iloc[0], "sex": sex, "lower_height_cm": lower, "upper_height_cm": upper})
    return frame.loc[keep].reset_index(drop=True), pd.DataFrame(rows)


def make_figures(performance, calibration, patterns, cross, subgroup):
    # ROC and PR curves.
    for cohort in performance.cohort.unique():
        frame = cohorts[cohort]
        predictions = predictions_by_cohort[cohort][0]
        w = frame.survey_weight.to_numpy(float)
        fig, axes = plt.subplots(3, 2, figsize=(10, 12))
        for i, (target, outcome) in enumerate(TARGETS.items()):
            y = frame[outcome].to_numpy(int)
            for model, label in MODELS.items():
                p = predictions[(target, model)]
                fpr, tpr, _ = roc_curve(y, p, sample_weight=w)
                precision, recall, _ = precision_recall_curve(y, p, sample_weight=w)
                axes[i, 0].plot(fpr, tpr, label=f"{label} ({roc_auc_score(y,p,sample_weight=w):.3f})")
                axes[i, 1].plot(recall, precision, label=f"{label} ({average_precision_score(y,p,sample_weight=w):.3f})")
            axes[i, 0].plot([0, 1], [0, 1], "--", color="#9CA3AF")
            axes[i, 0].set(title=f"{target}: ROC", xlabel="1 - specificity", ylabel="Sensitivity")
            axes[i, 1].axhline(wmean(y, w), linestyle="--", color="#9CA3AF")
            axes[i, 1].set(title=f"{target}: precision-recall", xlabel="Recall", ylabel="Precision")
            axes[i, 0].legend(frameon=False, fontsize=8); axes[i, 1].legend(frameon=False, fontsize=8)
        fig.suptitle(cohort); fig.tight_layout()
        safe = cohort.replace(" ", "_")
        fig.savefig(FIGURES / f"Figure_ROC_PR_{safe}.png", dpi=300, bbox_inches="tight"); plt.close(fig)

        fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
        for ax, target in zip(axes, TARGETS):
            sub = calibration[(calibration.cohort == cohort) & (calibration.target == target)]
            maximum = max(.1, 1.08 * sub[["weighted_mean_prediction", "weighted_observed_prevalence"]].max().max())
            ax.plot([0, maximum], [0, maximum], "--", color="#9CA3AF")
            for label in MODELS.values():
                d = sub[sub.model == label]
                ax.plot(d.weighted_mean_prediction, d.weighted_observed_prevalence, marker="o", label=label)
            ax.set(title=target, xlabel="Mean predicted probability", ylabel="Observed prevalence", xlim=(0, maximum), ylim=(0, maximum))
            ax.legend(frameon=False, fontsize=8)
        fig.suptitle(f"Calibration: {cohort}"); fig.tight_layout()
        fig.savefig(FIGURES / f"Figure_calibration_{safe}.png", dpi=300, bbox_inches="tight"); plt.close(fig)

    # Primary-cohort UpSet and cross-definition heat map.
    primary = "KNHANES 2008-2011"
    d = patterns[(patterns.cohort == primary) & (patterns.overlap_pattern != "None")].sort_values("survey_weighted_population_share", ascending=False)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6.5), gridspec_kw={"height_ratios": [3, 1.5]}, sharex=True)
    x = np.arange(len(d)); values = 100 * d.survey_weighted_population_share.to_numpy()
    ax1.bar(x, values, color="#4C78A8"); ax1.set_ylabel("Survey-weighted share (%)"); ax1.set_title("KNHANES 2008-2011 definition overlap")
    defs = ["P5", "EWGSOP2", "FNIH"]
    for row, definition in enumerate(defs): ax2.text(-.55, row, definition, ha="right", va="center")
    for col, label in enumerate(d.overlap_pattern):
        active = [i for i, definition in enumerate(defs) if definition in label.split(" + ")]
        ax2.scatter([col]*3, range(3), color="#D1D5DB", s=45)
        if len(active)>1: ax2.plot([col,col],[min(active),max(active)],color="#111827",lw=2)
        ax2.scatter([col]*len(active),active,color="#111827",s=55)
    ax2.set_ylim(2.5,-.5); ax2.set_yticks([]); ax2.set_xticks(x,d.overlap_pattern,rotation=25,ha="right")
    fig.tight_layout(); fig.savefig(FIGURES / "Figure_UpSet_KNHANES_2008_2011.png", dpi=300, bbox_inches="tight"); plt.close(fig)

    matrix = cross[cross.cohort == primary].pivot(index="trained_target", columns="evaluated_outcome", values="weighted_roc_auc").loc[list(TARGETS), list(TARGETS)]
    fig, ax = plt.subplots(figsize=(6,5)); im=ax.imshow(matrix,cmap="YlGnBu",vmin=.5,vmax=1)
    for i in range(3):
        for j in range(3): ax.text(j,i,f"{matrix.iloc[i,j]:.3f}",ha="center",va="center")
    ax.set_xticks(range(3),matrix.columns); ax.set_yticks(range(3),matrix.index); ax.set(xlabel="Evaluated outcome",ylabel="NN training target",title="KNHANES 2008-2011 cross-definition AUROC")
    fig.colorbar(im,ax=ax); fig.tight_layout(); fig.savefig(FIGURES / "Figure_cross_definition_KNHANES_2008_2011.png",dpi=300,bbox_inches="tight"); plt.close(fig)


log("Loading prepared Korean cohorts and frozen US models")
cohorts = {
    "KNHANES 2008-2011": pd.read_csv(DATA / "knhanes_2008_2011_paper3_ready.csv.gz"),
    "KNHANES 2024": pd.read_csv(DATA / "knhanes_2024_paper3_ready.csv.gz"),
}
for frame in cohorts.values():
    frame.index = np.arange(len(frame))
five_models, four_models = load_models()
predictions_by_cohort = {name: add_predictions(frame, five_models, four_models) for name, frame in cohorts.items()}
thresholds = pd.read_csv(US_RESULT / "tables" / "04_development_frozen_thresholds.csv")

performance = pd.concat([point_performance(name, frame, predictions_by_cohort[name][0]) for name, frame in cohorts.items()], ignore_index=True)
performance.to_csv(TABLES / "01_preliminary_primary_performance.csv", index=False)
log("PRELIMINARY POINT RESULTS READY")
print(performance[["cohort","target","model","weighted_prevalence","weighted_mean_prediction","weighted_roc_auc","weighted_pr_auc","weighted_brier","calibration_intercept","calibration_slope"]].to_string(index=False), flush=True)

calibration = pd.concat([calibration_points(name, frame, predictions_by_cohort[name][0]) for name, frame in cohorts.items()], ignore_index=True)
operating_point = pd.concat([threshold_points(name, frame, predictions_by_cohort[name][0], thresholds) for name, frame in cohorts.items()], ignore_index=True)
subgroup = pd.concat([subgroup_points(name, frame, predictions_by_cohort[name][0]) for name, frame in cohorts.items()], ignore_index=True)

prev_parts=[]; pair_parts=[]; pattern_parts=[]; cross_parts=[]; corr_parts=[]
for name, frame in cohorts.items():
    a,b,c=agreement_points(name,frame); prev_parts.append(a); pair_parts.append(b); pattern_parts.append(c)
    a,b=cross_definition(name,frame,predictions_by_cohort[name][0]); cross_parts.append(a); corr_parts.append(b)
prevalence=pd.concat(prev_parts,ignore_index=True); agreement=pd.concat(pair_parts,ignore_index=True); patterns=pd.concat(pattern_parts,ignore_index=True)
cross=pd.concat(cross_parts,ignore_index=True); correlations=pd.concat(corr_parts,ignore_index=True)

# Point estimates for waist incremental value.
waist_point=[]
for name, frame in cohorts.items():
    w=frame.survey_weight.to_numpy(float); p5,p4=predictions_by_cohort[name]
    for target,outcome in TARGETS.items():
        y=frame[outcome].to_numpy(int)
        for model,label in MODELS.items():
            m5,m4=metrics(y,p5[(target,model)],w),metrics(y,p4[(target,model)],w)
            for metric in ["weighted_roc_auc","weighted_brier"]:
                waist_point.append({"cohort":name,"target":target,"model":label,"metric":metric,"five_predictor":m5[metric],"four_predictor":m4[metric],"five_minus_four":m5[metric]-m4[metric]})
waist_point=pd.DataFrame(waist_point)

# Sensitivity: historical 0.946 factor and empirical common age/height support.
us=pd.read_csv(US_SOURCE / "data" / "development_maximum_set_b_person.csv.gz")
sens_parts=[]; support_parts=[]
for name,frame in cohorts.items():
    sens_parts.append(point_performance(name,frame,predictions_by_cohort[name][0],TARGETS_CAL,"Historical KNHANES ALM factor 0.946"))
    supported,cuts=common_support(frame,us); support_parts.append(cuts)
    support_predictions,_=add_predictions(supported,five_models,four_models)
    sens_parts.append(point_performance(name,supported,support_predictions,TARGETS,"Common empirical US-Korean age/height support"))
sensitivity=pd.concat(sens_parts,ignore_index=True); support=pd.concat(support_parts,ignore_index=True)

log("Starting paired 500-replicate stratified-PSU bootstrap")
perf_boot=[]; op_boot=[]; waist_boot=[]; subgroup_boot=[]; agreement_boot=[]
for name,frame in cohorts.items():
    parts=bootstrap_all(name,frame,*predictions_by_cohort[name],thresholds)
    for container,part in zip((perf_boot,op_boot,waist_boot,subgroup_boot,agreement_boot),parts): container.append(part)
perf_boot=pd.concat(perf_boot,ignore_index=True); op_boot=pd.concat(op_boot,ignore_index=True); waist_boot=pd.concat(waist_boot,ignore_index=True); subgroup_boot=pd.concat(subgroup_boot,ignore_index=True); agreement_boot=pd.concat(agreement_boot,ignore_index=True)

perf_ci=summarize_ci(perf_boot,["cohort","target","model","metric"])
performance_long=performance.melt(id_vars=["cohort","analysis","target","model","participants"],value_vars=["weighted_prevalence","weighted_mean_prediction","weighted_roc_auc","weighted_pr_auc","weighted_brier","calibration_intercept","calibration_slope"],var_name="metric",value_name="estimate")
performance_long=performance_long.merge(perf_ci,on=["cohort","target","model","metric"],how="left")
operating=operating_point.merge(summarize_ci(op_boot,["cohort","target","model","operating_point","metric"]),on=["cohort","target","model","operating_point","metric"],how="left")
waist=waist_point.merge(summarize_ci(waist_boot,["cohort","target","model","metric"],"five_minus_four"),on=["cohort","target","model","metric"],how="left")
subgroup=subgroup.merge(summarize_ci(subgroup_boot,["cohort","target","model","dimension","level"],"weighted_roc_auc").rename(columns={"ci_lower_95":"auc_ci_lower_95","ci_upper_95":"auc_ci_upper_95"}),on=["cohort","target","model","dimension","level"],how="left")

# Add agreement CIs metric by metric.
agreement_long=agreement.melt(id_vars=["cohort","definition_a","definition_b"],var_name="metric",value_name="estimate")
agreement_long=agreement_long.merge(summarize_ci(agreement_boot,["cohort","definition_a","definition_b","metric"]),on=["cohort","definition_a","definition_b","metric"],how="left")

performance_long.to_csv(TABLES / "02_model_performance_with_ci.csv",index=False)
calibration.to_csv(TABLES / "03_calibration_curve_points.csv",index=False)
operating.to_csv(TABLES / "04_fixed_US_threshold_operating_characteristics.csv",index=False)
prevalence.to_csv(TABLES / "05_definition_prevalence.csv",index=False)
agreement_long.to_csv(TABLES / "06_definition_agreement_with_ci.csv",index=False)
patterns.to_csv(TABLES / "07_definition_overlap_patterns.csv",index=False)
cross.to_csv(TABLES / "08_cross_definition_auc_matrix.csv",index=False)
correlations.to_csv(TABLES / "09_prediction_probability_correlations.csv",index=False)
waist.to_csv(TABLES / "10_waist_incremental_value_with_ci.csv",index=False)
subgroup.to_csv(TABLES / "11_subgroup_validation.csv",index=False)
sensitivity.to_csv(TABLES / "12_sensitivity_historical_factor_common_support.csv",index=False)
support.to_csv(TABLES / "12b_common_support_boundaries.csv",index=False)
perf_boot.to_csv(TABLES / "bootstrap_performance_replicates.csv.gz",index=False,compression="gzip")
op_boot.to_csv(TABLES / "bootstrap_operating_replicates.csv.gz",index=False,compression="gzip")
waist_boot.to_csv(TABLES / "bootstrap_waist_replicates.csv.gz",index=False,compression="gzip")
subgroup_boot.to_csv(TABLES / "bootstrap_subgroup_auc_replicates.csv.gz",index=False,compression="gzip")
agreement_boot.to_csv(TABLES / "bootstrap_agreement_replicates.csv.gz",index=False,compression="gzip")

make_figures(performance,calibration,patterns,cross,subgroup)

qa=pd.DataFrame([
    {"check":"Frozen US five-predictor models loaded","status":"PASS","observed":len(five_models)},
    {"check":"Frozen US four-predictor models loaded","status":"PASS","observed":len(four_models)},
    {"check":"No Korean refitting/recalibration/tuning","status":"PASS","observed":"predict_proba only"},
    {"check":"Primary cohort age locked 20-59","status":"PASS" if cohorts['KNHANES 2008-2011'].age.between(20,59).all() else "FAIL","observed":len(cohorts['KNHANES 2008-2011'])},
    {"check":"Contemporary cohort available age 40-59","status":"PASS" if cohorts['KNHANES 2024'].age.between(40,59).all() else "FAIL","observed":len(cohorts['KNHANES 2024'])},
    {"check":"500 bootstrap replicates per cohort","status":"PASS" if perf_boot.groupby('cohort').replicate.nunique().eq(B).all() else "FAIL","observed":B},
    {"check":"All required predictor values finite","status":"PASS" if all(np.isfinite(f[FEATURES5]).all().all() for f in cohorts.values()) else "FAIL","observed":"age sex height weight waist"},
])
qa.to_csv(TABLES / "13_final_QA.csv",index=False)

manifest={"analysis":"Frozen US Paper 3 models in KNHANES","date":"2026-10-01","primary":"KNHANES 2008-2011 age 20-59; raw Korean ALM","secondary":"KNHANES 2024 age 40-59; raw Korean ALM","predictors":FEATURES5,"models":"nine frozen US final models; no Korean refitting","thresholds":"frozen US development OOF 90% and 80% sensitivity thresholds","bootstrap":{"replicates":B,"design":"PSU with replacement within cycle-nested strata"},"sensitivity":["historical KNHANES ALM factor 0.946","common empirical US-Korean age/height support"]}
(ROOT/"analysis_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
(ROOT/"session_info.json").write_text(json.dumps({"python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__,"scipy":scipy.__version__,"sklearn":sklearn.__version__,"joblib":joblib.__version__},indent=2),encoding="utf-8")

rows=[]
for path in sorted(ROOT.rglob("*")):
    if path.is_file() and path.name!="SHA256_MANIFEST.csv": rows.append({"path":path.relative_to(ROOT).as_posix(),"bytes":path.stat().st_size,"sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
pd.DataFrame(rows).to_csv(ROOT/"SHA256_MANIFEST.csv",index=False)
log("Frozen Korean external validation completed")
