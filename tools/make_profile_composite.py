from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


SUBJECT_ORDER = ["生理/生化", "病理", "内科", "外科", "伦理/法规"]
SUBJECT_LABELS = {
    "生理/生化": "Physiology/Biochem",
    "病理": "Pathology",
    "内科": "Internal Medicine",
    "外科": "Surgery",
    "伦理/法规": "Ethics/Reg",
}


TYPE_COLORS = {
    "Proprietary": "#4C78A8",  # blue
    "Open-weight": "#F58518",  # orange
}


HEAT_COLORS = {
    0: "#B2182B",  # Robust Wrong
    1: "#FDDBC7",  # Guessing/Ambiguous
    2: "#92C5DE",  # Probabilistic
    3: "#2166AC",  # Robust Correct
}


def is_open_weight(model_id: str) -> bool:
    s = str(model_id).lower()
    if "oss" in s:
        return True
    if s.startswith("qwen3-") and "max" not in s:
        return True
    if s.startswith("deepseek"):
        return True
    return False


@dataclass(frozen=True)
class BlocksAndOrder:
    ordered_qids: List[int]
    blocks: List[Tuple[str, int, int]]  # (subject, start_idx, end_idx)


def build_question_blocks(df_cells: pd.DataFrame) -> BlocksAndOrder:
    df = df_cells[["question_id", "subject", "correct_count"]].copy()
    df["question_id"] = df["question_id"].astype(int)

    qid_subject = df.drop_duplicates(subset=["question_id"])[["question_id", "subject"]].set_index("question_id")[
        "subject"
    ]
    mean_cc = df.groupby("question_id")["correct_count"].mean()

    all_qids = sorted(int(q) for q in qid_subject.index.tolist())

    ordered_qids: List[int] = []
    blocks: List[Tuple[str, int, int]] = []

    idx = 0
    for subject in SUBJECT_ORDER:
        qids_s = [q for q in all_qids if str(qid_subject.get(q, "")) == subject]
        if not qids_s:
            continue
        qids_s = sorted(qids_s, key=lambda q: (-float(mean_cc.get(q, 0.0)), int(q)))
        ordered_qids.extend(qids_s)
        blocks.append((subject, idx, idx + len(qids_s)))
        idx += len(qids_s)

    # Any other subjects (should be empty for CMCA 165).
    other = [q for q in all_qids if str(qid_subject.get(q, "")) not in SUBJECT_ORDER]
    if other:
        other = sorted(other, key=lambda q: (-float(mean_cc.get(q, 0.0)), int(q)))
        ordered_qids.extend(other)
        blocks.append(("其他", idx, idx + len(other)))

    return BlocksAndOrder(ordered_qids=ordered_qids, blocks=blocks)


def build_category_matrix(
    df_cells: pd.DataFrame,
    model_order: List[str],
    ordered_qids: List[int],
    run_count: int,
) -> np.ndarray:
    pivot = (
        df_cells.pivot(index="model_id", columns="question_id", values="correct_count")
        .reindex(index=model_order, columns=ordered_qids)
        .fillna(0)
        .astype(int)
    )
    mat = pivot.to_numpy()

    cat = np.zeros_like(mat, dtype=int)
    cat[(mat >= 1) & (mat <= 2)] = 1
    cat[(mat >= 3) & (mat <= 4)] = 2
    cat[mat == run_count] = 3
    return cat


def _apply_rcparams() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "font.family": "Arial",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans", "sans-serif"],
            "font.weight": "normal",
            "axes.titleweight": "normal",
            "axes.labelweight": "normal",
            "figure.titleweight": "normal",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )


def _despine(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def make_profile_figure(
    *,
    df_models: pd.DataFrame,
    df_cells: pd.DataFrame,
    title_prefix: str,
    out_pdf: Path,
    out_png: Path,
    figsize: Tuple[float, float] = (18, 14),
) -> None:
    _apply_rcparams()

    df_models = df_models.copy()
    df_models["accuracy_pct"] = df_models["accuracy_mean"].astype(float) * 100.0
    df_models["ci95_half_pct"] = df_models["accuracy_ci95_half"].astype(float) * 100.0

    df_models = df_models.sort_values("accuracy_mean", ascending=False).reset_index(drop=True)

    model_order = df_models["model_id"].astype(str).tolist()
    model_labels = df_models["model_name"].astype(str).tolist()

    run_count = int(df_models["runs"].iloc[0]) if "runs" in df_models.columns and len(df_models) else 5

    blocks = build_question_blocks(df_cells)
    ordered_qids = blocks.ordered_qids
    cat = build_category_matrix(df_cells, model_order, ordered_qids, run_count=run_count)

    n_models = len(model_order)
    n_items = len(ordered_qids)

    cmap = ListedColormap([HEAT_COLORS[0], HEAT_COLORS[1], HEAT_COLORS[2], HEAT_COLORS[3]])

    fig, (ax_bar, ax_hm) = plt.subplots(
        2,
        1,
        figsize=figsize,
        sharey=True,
        gridspec_kw={"height_ratios": [1, 2.5]},
    )
    # Reserve a fixed left margin so both panels share the exact same left anchor.
    fig.subplots_adjust(left=0.25, right=0.99, top=0.95, bottom=0.10, hspace=0.16)

    # Panel A: bar chart
    y_centers = np.arange(n_models) + 0.5
    types = ["Open-weight" if is_open_weight(mid) else "Proprietary" for mid in model_order]
    colors = [TYPE_COLORS[t] for t in types]

    ax_bar.barh(
        y_centers,
        df_models["accuracy_pct"].to_numpy(),
        xerr=df_models["ci95_half_pct"].to_numpy(),
        height=0.72,
        color=colors,
        edgecolor="none",
        error_kw=dict(ecolor="#111827", elinewidth=1.2, capsize=3, capthick=1.2),
        zorder=3,
    )

    ax_bar.set_xlim(0, 100)
    ax_bar.set_ylim(n_models, 0)
    ax_bar.set_yticks(y_centers)
    ax_bar.set_yticklabels(model_labels)
    for tl in ax_bar.get_yticklabels():
        tl.set_fontsize(12)
    ax_bar.set_xlabel("Accuracy (%) (mean ± 95% CI)")
    ax_bar.grid(axis="x", color="#e5e7eb", linewidth=0.8, alpha=0.8, zorder=0)
    ax_bar.tick_params(axis="y", length=0, pad=6)
    ax_bar.tick_params(axis="x", labelsize=10)

    for y, v, ci in zip(
        y_centers,
        df_models["accuracy_pct"].to_numpy(),
        df_models["ci95_half_pct"].to_numpy(),
    ):
        # Place value labels to the right of the full error bar to avoid overlap.
        x = float(v) + float(ci) + 0.8
        x = min(x, 99.2)
        ax_bar.text(
            x,
            float(y),
            f"{float(v):.1f}",
            va="center",
            ha="left",
            fontsize=10,
            color="#111827",
        )

    ax_bar.set_title(f"{title_prefix}: Overall accuracy (165 items x {run_count} trials)", pad=10)
    _despine(ax_bar)

    # Bar legend (type)
    bar_legend = [
        Patch(facecolor=TYPE_COLORS["Proprietary"], edgecolor="none", label="Proprietary"),
        Patch(facecolor=TYPE_COLORS["Open-weight"], edgecolor="none", label="Open-weight"),
    ]
    ax_bar.legend(handles=bar_legend, loc="lower right", frameon=False, fontsize=10)

    # Panel labels A/B
    ax_bar.text(
        -0.06,
        1.02,
        "A",
        transform=ax_bar.transAxes,
        fontsize=14,
        va="bottom",
        ha="left",
    )

    # Panel B: heatmap (no vertical gridlines; only row separators + discipline separators)
    ax_hm.set_xlim(0, n_items)
    ax_hm.set_ylim(n_models, 0)
    ax_hm.pcolormesh(
        np.arange(n_items + 1),
        np.arange(n_models + 1),
        cat,
        cmap=cmap,
        vmin=-0.5,
        vmax=3.5,
        shading="flat",
        edgecolors="none",
        antialiased=False,
        rasterized=False,
    )

    # Horizontal separators between models
    ax_hm.hlines(np.arange(1, n_models), xmin=0, xmax=n_items, colors="white", linewidth=0.5, zorder=5)

    # Discipline separators and labels
    for _, start, _ in blocks.blocks[1:]:
        ax_hm.vlines(start, ymin=0, ymax=n_models, colors="white", linewidth=3.5, zorder=6)

    for subject, start, end in blocks.blocks:
        if end <= start:
            continue
        label = SUBJECT_LABELS.get(subject, str(subject))
        mid = (start + end) / 2.0
        ax_hm.text(
            mid,
            1.015,
            label,
            transform=ax_hm.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=11,
            color="#111827",
            clip_on=False,
        )

    ax_hm.set_xticks([])
    ax_hm.set_xlabel("")
    # Restore model names on Panel B and keep them identical to Panel A.
    ax_hm.set_yticks(y_centers)
    ax_hm.set_yticklabels(model_labels)
    for tl in ax_hm.get_yticklabels():
        tl.set_fontsize(12)
    ax_hm.tick_params(axis="y", length=0, pad=6)
    # Avoid a panel title here to keep the discipline headers unobstructed.

    for side in ("top", "right", "left", "bottom"):
        ax_hm.spines[side].set_visible(False)

    ax_hm.text(
        -0.06,
        1.02,
        "B",
        transform=ax_hm.transAxes,
        fontsize=14,
        va="bottom",
        ha="left",
    )

    # Heatmap legend (bottom)
    hm_legend = [
        Patch(facecolor=HEAT_COLORS[3], edgecolor="none", label=f"Robust Correct ({run_count}/{run_count})"),
        Patch(facecolor=HEAT_COLORS[2], edgecolor="none", label="Probabilistic (3-4/5)"),
        Patch(facecolor=HEAT_COLORS[1], edgecolor="none", label="Guessing/Ambiguous (1-2/5)"),
        Patch(facecolor=HEAT_COLORS[0], edgecolor="none", label=f"Robust Wrong (0/{run_count})"),
    ]
    # Slightly shrink the heatmap width so its right edge roughly aligns with the bar panel's data extent
    # and leaves extra breathing room for the rightmost discipline label (e.g., "Ethics/Reg").
    pos_bar = ax_bar.get_position()
    pos_hm = ax_hm.get_position()
    ax_hm.set_position([pos_bar.x0, pos_hm.y0, pos_bar.width * 0.94, pos_hm.height])

    # Legend: place at the very bottom, centered under the overall chart area (not the left label margin),
    # and keep it close to the heatmap.
    pos_bar = ax_bar.get_position()
    pos_hm = ax_hm.get_position()
    legend_x = float(pos_bar.x0 + pos_bar.width / 2.0)
    legend_y = max(0.01, float(pos_hm.y0 - 0.035))
    fig.legend(
        handles=hm_legend,
        loc="lower center",
        bbox_to_anchor=(legend_x, legend_y),
        ncol=4,
        frameon=False,
        fontsize=10,
        borderaxespad=0.0,
    )

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description="Make publication-ready Baseline/Agentic profile composite figure (A+B).")
    ap.add_argument("--model-metrics", required=True, type=Path, help="model_metrics.csv")
    ap.add_argument("--cell-metrics", required=True, type=Path, help="cell_metrics.csv")
    ap.add_argument("--title-prefix", required=True, help="Figure title prefix (e.g., Baseline / Agentic)")
    ap.add_argument("--out-pdf", required=True, type=Path)
    ap.add_argument("--out-png", required=True, type=Path)
    ap.add_argument("--figsize", default="18,14", help="Figure size in inches, e.g. 18,14")
    args = ap.parse_args()

    w_s, h_s = (args.figsize.split(",") + ["14"])[:2]
    figsize = (float(w_s), float(h_s))

    df_models = pd.read_csv(args.model_metrics)
    df_cells = pd.read_csv(args.cell_metrics)

    # Normalize types
    df_cells["question_id"] = df_cells["question_id"].astype(int)
    df_cells["correct_count"] = df_cells["correct_count"].astype(int)
    df_cells["model_id"] = df_cells["model_id"].astype(str)
    df_cells["subject"] = df_cells["subject"].astype(str)

    make_profile_figure(
        df_models=df_models,
        df_cells=df_cells,
        title_prefix=str(args.title_prefix),
        out_pdf=args.out_pdf,
        out_png=args.out_png,
        figsize=figsize,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
