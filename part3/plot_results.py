#!/usr/bin/env python3
"""Create report-ready Part 3 figures from results.csv."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results.csv"
OUTPUT = ROOT.parent / "output"
DISTANCES = [20, 40, 80, 120]
COLORS = {0.1: "#2a6fbb", 0.5: "#d56a22"}


def load_rows() -> list[dict]:
    with RESULTS.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["distance_cm"] = int(row["distance_cm"])
        row["delta_alpha"] = float(row["delta_alpha"])
        row["n_bits"] = int(row["n_bits"])
        row["n_correct"] = int(row["n_correct"])
        row["ber"] = float(row["ber"])
        row["data_rate_bps"] = float(row["data_rate_bps"])
        row["sync_score"] = float(row["sync_score"])
    return sorted(rows, key=lambda row: (row["delta_alpha"], row["distance_cm"]))


def rows_for_alpha(rows: list[dict], alpha: float) -> list[dict]:
    selected = [row for row in rows if row["delta_alpha"] == alpha]
    if [row["distance_cm"] for row in selected] != DISTANCES:
        raise ValueError(f"Expected distances {DISTANCES} for alpha {alpha}")
    return selected


def style_axes(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.8)
    ax.set_xticks(DISTANCES)


def plot_ber(rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for alpha, marker in ((0.1, "o"), (0.5, "s")):
        selected = rows_for_alpha(rows, alpha)
        values = [row["ber"] for row in selected]
        ax.plot(
            DISTANCES,
            values,
            color=COLORS[alpha],
            marker=marker,
            markersize=9 if alpha == 0.1 else 6,
            linewidth=3 if alpha == 0.1 else 1.8,
            linestyle="-" if alpha == 0.1 else "--",
            markerfacecolor="white" if alpha == 0.1 else COLORS[alpha],
            markeredgewidth=2,
            zorder=2 if alpha == 0.1 else 3,
            label=f"Δα = {alpha}",
        )
    ax.set_title("Bit-error rate versus phone-to-screen distance", pad=14)
    ax.set_xlabel("Distance (cm)")
    ax.set_ylabel("Bit-error rate")
    ax.set_ylim(-0.002, 0.052)
    ax.set_yticks([0.00, 0.01, 0.02, 0.03, 0.04, 0.05])
    ax.text(
        0.5,
        0.88,
        "All 448 transmitted bits were decoded correctly",
        transform=ax.transAxes,
        ha="center",
        fontsize=10,
    )
    style_axes(ax)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUTPUT / "part3_ber_vs_distance.png", dpi=180)
    plt.close(fig)


def plot_sync(rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for alpha, marker in ((0.1, "o"), (0.5, "s")):
        selected = rows_for_alpha(rows, alpha)
        values = [row["sync_score"] for row in selected]
        ax.plot(
            DISTANCES,
            values,
            color=COLORS[alpha],
            marker=marker,
            markersize=7,
            linewidth=2,
            label=f"Δα = {alpha}",
        )
    ax.axhline(0.35, color="#666666", linestyle="--", linewidth=1.2, label="Sync threshold")
    ax.set_title("Packet synchronization score versus distance", pad=14)
    ax.set_xlabel("Distance (cm)")
    ax.set_ylabel("Normalized synchronization score")
    ax.set_ylim(0.3, 1.02)
    style_axes(ax)
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUTPUT / "part3_sync_vs_distance.png", dpi=180)
    plt.close(fig)


def plot_table(rows: list[dict]) -> None:
    headers = ["Δ alpha", "Distance", "Correct bits", "BER", "Data rate", "Sync score", "Decoded"]
    cells = [
        [
            f"{row['delta_alpha']:.1f}",
            f"{row['distance_cm']} cm",
            f"{row['n_correct']} / {row['n_bits']}",
            f"{row['ber']:.3f}",
            f"{row['data_rate_bps']:.2f} bits/s",
            f"{row['sync_score']:.3f}",
            row["decoded"],
        ]
        for row in rows
    ]
    fig, ax = plt.subplots(figsize=(10.8, 4.1))
    ax.axis("off")
    ax.set_title("Screen–camera BFSK experimental results", fontsize=15, pad=16)
    table = ax.table(cellText=cells, colLabels=headers, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1, 1.55)
    for (row_index, column_index), cell in table.get_celld().items():
        cell.set_edgecolor("#d5d5d5")
        cell.set_linewidth(0.7)
        if row_index == 0:
            cell.set_facecolor("#242424")
            cell.set_text_props(color="white", weight="bold")
        elif row_index % 2 == 0:
            cell.set_facecolor("#f2f4f6")
    fig.tight_layout()
    fig.savefig(OUTPUT / "part3_results_table.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(exist_ok=True)
    rows = load_rows()
    plot_ber(rows)
    plot_sync(rows)
    plot_table(rows)
    print("Created:")
    for name in (
        "part3_ber_vs_distance.png",
        "part3_sync_vs_distance.png",
        "part3_results_table.png",
    ):
        print(f"  {OUTPUT / name}")


if __name__ == "__main__":
    main()
