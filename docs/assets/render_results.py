"""Regenerate README result figures from the transcribed result table.

Run from the repository root: python docs/assets/render_results.py
Requires matplotlib. Does not execute notebooks, models, or benchmark grading.
"""

import csv
from collections import Counter
from pathlib import Path

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


HERE = Path(__file__).resolve().parent
with (HERE.parent / "research/results_by_competition.csv").open() as stream:
    ROWS = list(csv.DictReader(stream))
COHORTS = ["R1", "R2", "R3", "without-search"]
LABELS = [
    "R1 · search enabled",
    "R2 · search enabled",
    "R3 · search enabled",
    "Without external retrieval",
]
COLORS = {
    "gold": "#E4D19A",
    "silver": "#D3D9DE",
    "bronze": "#D9C0AD",
    "none": "#FFFFFF",
    "invalid": "#EAEAEA",
}
HATCHES = {"gold": "", "silver": "", "bronze": "", "none": "", "invalid": "////"}
SYMBOLS = {"gold": "G", "silver": "S", "bronze": "B", "none": "-", "invalid": "X"}


def status(row):
    return row["medal"] if row["valid_submission"] == "true" else "invalid"


def save(fig, name):
    fig.savefig(HERE / f"{name}.svg", facecolor="white", metadata={"Date": None})
    fig.savefig(HERE / f"{name}.png", dpi=220, facecolor="white")
    plt.close(fig)


def render_cohorts():
    fig, ax = plt.subplots(figsize=(8.8, 3.9))
    fig.subplots_adjust(left=0.30, right=0.97, top=0.77, bottom=0.30)
    fig.text(
        0.035,
        0.925,
        "Recorded outcomes across 22 competitions",
        fontsize=12,
    )
    fig.text(
        0.035,
        0.865,
        "Historical cohorts · reported setup: Gemini 3 Flash, Colab L4/T4",
        fontsize=9,
    )
    for i, cohort in enumerate(COHORTS):
        entries = [r for r in ROWS if r["cohort"] == cohort]
        assert len(entries) == 22
        counts = Counter(status(r) for r in entries)
        left = 0
        for category, color in COLORS.items():
            count = counts[category]
            if not count:
                continue
            ax.barh(
                i,
                count,
                left=left,
                height=0.56,
                color=color,
                edgecolor="#333333",
                linewidth=0.5,
                hatch=HATCHES[category],
            )
            ax.text(
                left + count / 2,
                i,
                str(count),
                ha="center",
                va="center",
                fontsize=9,
                bbox={"facecolor": color, "edgecolor": "none", "pad": 0.5},
            )
            left += count
    ax.set_yticks(range(4), LABELS, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 22)
    ax.set_xticks([0, 5, 10, 15, 20, 22])
    ax.set_xlabel(
        "Competitions (invalid submissions stay in the denominator)", fontsize=9, labelpad=7
    )
    ax.spines[["left", "right", "top"]].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.tick_params(axis="y", length=0, pad=8)
    ax.tick_params(axis="x", length=3, width=0.6, labelsize=8)
    fig.legend(
        handles=[
            Patch(
                facecolor=v,
                edgecolor="#333333",
                linewidth=0.5,
                hatch=HATCHES[k],
                label=k.title() if k != "none" else "Valid, no medal",
            )
            for k, v in COLORS.items()
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.09),
        ncol=5,
        frameon=False,
        fontsize=9,
        handlelength=1.5,
        columnspacing=1.8,
    )
    fig.text(
        0.035,
        0.045,
        "Descriptive results; cohorts are not controlled replicates. Source: accompanying result table; see evidence notes.",
        fontsize=8,
    )
    save(fig, "cohort-results")


def render_matrix():
    names = [r["competition_id"] for r in ROWS if r["cohort"] == "R1"]
    short = [
        "Aerial Cactus",
        "APTOS 2019",
        "Denoising Documents",
        "Detecting Insults",
        "Dog Breed",
        "Dogs vs. Cats Redux",
        "Histopathologic Cancer",
        "Jigsaw Toxic Comments",
        "Leaf Classification",
        "MLSP 2013 Birds",
        "NYC Taxi Fare",
        "NOMAD 2018",
        "Plant Pathology 2020",
        "Random Acts of Pizza",
        "RANZCR CLiP",
        "SIIM-ISIC Melanoma",
        "Spooky Author",
        "TPS December 2021",
        "TPS May 2022",
        "Text Normalization (EN)",
        "Text Normalization (RU)",
        "Right Whale Redux",
    ]
    lookup = {(r["competition_id"], r["cohort"]): status(r) for r in ROWS}
    fig, ax = plt.subplots(figsize=(7.4, 7.8))
    fig.subplots_adjust(left=0.05, right=0.95, top=0.89, bottom=0.13)
    fig.text(0.05, 0.965, "Outcomes by competition and cohort", fontsize=12)
    fig.text(
        0.05,
        0.937,
        "22 tasks / 4 historical cohorts · threshold-based outcomes",
        fontsize=9,
    )
    columns = [3.25, 4.15, 5.05, 6.1]
    ax.text(0, -1.43, "Competition", va="center", fontsize=9)
    for x, label in zip(
        columns,
        ["R1\nSearch\nenabled", "R2\nSearch\nenabled", "R3\nSearch\nenabled", "Without\nsearch"],
        strict=True,
    ):
        ax.text(x, -1.43, label, ha="center", va="center", fontsize=9, linespacing=1.2)
    for i, name in enumerate(names):
        ax.text(0, i, short[i], va="center", fontsize=9)
        for j, cohort in enumerate(COHORTS):
            category = lookup[name, cohort]
            ax.text(
                columns[j],
                i,
                SYMBOLS[category],
                ha="center",
                va="center",
                fontsize=9,
            )
    ax.hlines([i + 0.5 for i in range(21)], 0, 6.65, color="#DDDDDD", linewidth=0.35)
    ax.hlines([-2.35, -0.5, 21.5], 0, 6.65, color="black", linewidth=[0.7, 0.5, 0.7])
    ax.set_xlim(0, 6.65)
    ax.set_ylim(21.6, -2.45)
    ax.set_axis_off()
    fig.text(
        0.05,
        0.098,
        "G  Gold       S  Silver       B  Bronze       -  No medal       X  Invalid",
        fontsize=9,
    )
    fig.text(
        0.05,
        0.046,
        "Medals are historical MLE-bench thresholds, not live Kaggle awards.\n"
        "Raw scores and caveats accompany the CSV.",
        fontsize=8,
        linespacing=1.4,
    )
    save(fig, "competition-matrix")


if __name__ == "__main__":
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "text.color": "black",
            "axes.labelcolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "hatch.linewidth": 0.35,
            "svg.fonttype": "none",
            "svg.hashsalt": "kaggle-agents-readme",
        }
    )
    render_cohorts()
    render_matrix()
