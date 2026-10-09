from pathlib import Path
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "results" / "knhanes" / "tables"
FIGURES = ROOT / "results" / "knhanes" / "figures"
TARGETS = ["P5", "EWGSOP2", "FNIH"]

patterns = pd.read_csv(TABLES / "07_definition_overlap_patterns.csv")
cross = pd.read_csv(TABLES / "08_cross_definition_auc_matrix.csv")
subgroups = pd.read_csv(TABLES / "11_subgroup_validation.csv")

for cohort in ["KNHANES 2024"]:
    d = patterns[(patterns.cohort == cohort) & patterns.overlap_pattern.notna() &
                 (patterns.overlap_pattern != "None")].sort_values(
        "survey_weighted_population_share", ascending=False)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6.5), gridspec_kw={"height_ratios": [3, 1.5]}, sharex=True)
    x = np.arange(len(d))
    ax1.bar(x, 100 * d.survey_weighted_population_share.to_numpy(), color="#4C78A8")
    ax1.set_ylabel("Survey-weighted share (%)")
    ax1.set_title(f"{cohort} definition overlap")
    for row, definition in enumerate(TARGETS):
        ax2.text(-0.55, row, definition, ha="right", va="center")
    for col, label in enumerate(d.overlap_pattern):
        active = [i for i, definition in enumerate(TARGETS) if definition in label.split(" + ")]
        ax2.scatter([col] * 3, range(3), color="#D1D5DB", s=45)
        if len(active) > 1:
            ax2.plot([col, col], [min(active), max(active)], color="#111827", lw=2)
        ax2.scatter([col] * len(active), active, color="#111827", s=55)
    ax2.set_ylim(2.5, -0.5)
    ax2.set_yticks([])
    ax2.set_xticks(x, d.overlap_pattern, rotation=25, ha="right")
    fig.tight_layout()
    fig.savefig(FIGURES / "Figure_UpSet_KNHANES_2024.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    matrix = cross[cross.cohort == cohort].pivot(index="trained_target", columns="evaluated_outcome",
                                                  values="weighted_roc_auc").loc[TARGETS, TARGETS]
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, cmap="YlGnBu", vmin=0.5, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{matrix.iloc[i, j]:.3f}", ha="center", va="center")
    ax.set_xticks(range(3), matrix.columns)
    ax.set_yticks(range(3), matrix.index)
    ax.set(xlabel="Evaluated outcome", ylabel="NN training target", title=f"{cohort} cross-definition AUROC")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(FIGURES / "Figure_cross_definition_KNHANES_2024.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

# Neural-network subgroup forest plots. Suppressed cells are retained as labels but have no estimate.
for cohort in subgroups.cohort.unique():
    data = subgroups[(subgroups.cohort == cohort) & (subgroups.model == "Neural network")].copy()
    order = data[["dimension", "level"]].drop_duplicates()
    labels = [f"{row.dimension}: {row.level}" for row in order.itertuples()]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, max(6.5, 0.38 * len(labels))))
    for ax, target in zip(axes, TARGETS):
        d = order.merge(data[data.target == target], on=["dimension", "level"], how="left")
        y = np.arange(len(d))[::-1]
        reportable = d.reporting_flag == "REPORTABLE"
        center = pd.to_numeric(d.weighted_roc_auc, errors="coerce").to_numpy(float)
        lower = pd.to_numeric(d.auc_ci_lower_95, errors="coerce").to_numpy(float)
        upper = pd.to_numeric(d.auc_ci_upper_95, errors="coerce").to_numpy(float)
        ok = reportable.to_numpy() & np.isfinite(center) & np.isfinite(lower) & np.isfinite(upper)
        ax.errorbar(center[ok], y[ok], xerr=np.vstack([center[ok] - lower[ok], upper[ok] - center[ok]]),
                    fmt="o", color="#0072B2", capsize=2, markersize=4)
        ax.axvline(0.5, linestyle="--", color="0.55", linewidth=1)
        ax.set(title=target, xlabel="Survey-weighted AUROC (95% CI)", xlim=(0.45, 1.01), yticks=y)
        if ax is axes[0]:
            ax.set_yticklabels(labels)
        else:
            ax.set_yticklabels([])
        ax.grid(axis="x", alpha=0.2)
    fig.suptitle(f"{cohort}: frozen neural-network subgroup validation")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    safe = cohort.replace(" ", "_")
    fig.savefig(FIGURES / f"Figure_subgroup_forest_{safe}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

print("Secondary UpSet, cross-definition heat map, and subgroup forest plots completed.")
