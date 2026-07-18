"""Generate all figures for the Block 1 slide deck.

Usage (from the data-mining/ directory):
    uv run python lecture_1/make_figures.py

Outputs PDF figures into lecture_1/figures/. Deterministic: uses the
synthetic course portfolio with its fixed seed.
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

# Formats money axes as "50k" so tick labels never collide.
KPLN = FuncFormatter(lambda x, _: f"{x/1000:.0f}k")

from finance_data import load_credit

logger = logging.getLogger(__name__)

FIG_DIR = Path(__file__).parent / "figures"

# Deck palette: blue carries data, red is reserved for default/risk accents.
BLUE = "#2E6DA4"
DARK = "#003366"
RED = "#C83C28"
GRAY = "#8C959E"
LIGHT = "#D7E3F0"

plt.rcParams.update(
    {
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "figure.autolayout": False,
        "savefig.bbox": "tight",
    }
)


def save(fig: plt.Figure, name: str) -> None:
    """Save a figure as PDF into the figures directory."""
    path = FIG_DIR / f"{name}.pdf"
    fig.savefig(path)
    plt.close(fig)
    logger.info("wrote %s", path)


def fig_target_balance(df: pd.DataFrame) -> None:
    counts = df["default"].value_counts().sort_index()
    share = counts / counts.sum() * 100
    fig, ax = plt.subplots(figsize=(6.8, 2.6))
    labels = ["repaid (0)", "default (1)"]
    colors = [GRAY, RED]
    bars = ax.barh(labels, counts.values, color=colors, height=0.55)
    for bar, c, s in zip(bars, counts.values, share.values):
        ax.text(c + counts.max() * 0.015, bar.get_y() + bar.get_height() / 2,
                f"{c:,}  ({s:.0f}%)", va="center", fontsize=12)
    ax.set_xlim(0, counts.max() * 1.22)
    ax.set_xlabel("customers")
    ax.grid(axis="y", visible=False)
    ax.invert_yaxis()
    save(fig, "fig_target_balance")


def fig_income_hist(df: pd.DataFrame) -> None:
    inc = df["monthly_income_pln"].dropna()
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.hist(inc, bins=60, color=BLUE, edgecolor="white", linewidth=0.3)
    ax.axvline(inc.mean(), color=RED, linestyle="--", linewidth=1.6)
    ax.axvline(inc.median(), color=DARK, linestyle="-", linewidth=1.6)
    ax.text(0.97, 0.88, f"mean = {inc.mean()/1000:.1f}k (dashed)",
            color=RED, fontsize=11, ha="right", transform=ax.transAxes)
    ax.text(0.97, 0.74, f"median = {inc.median()/1000:.1f}k",
            color=DARK, fontsize=11, ha="right", transform=ax.transAxes)
    ax.xaxis.set_major_formatter(KPLN)
    ax.set_xlabel("monthly income (k PLN)")
    ax.set_ylabel("customers")
    ax.grid(axis="x", visible=False)
    save(fig, "fig_income_hist")


def fig_income_log(df: pd.DataFrame) -> None:
    inc = df["monthly_income_pln"].dropna()
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0))
    axes[0].hist(inc, bins=50, color=BLUE, edgecolor="white", linewidth=0.3)
    axes[0].set_title("raw scale", fontsize=12)
    axes[0].xaxis.set_major_formatter(KPLN)
    axes[0].set_xlabel("monthly income (k PLN)")
    axes[1].hist(np.log10(inc), bins=50, color=BLUE, edgecolor="white",
                 linewidth=0.3)
    axes[1].set_title("log10 scale", fontsize=12)
    axes[1].set_xlabel("log10(income)")
    for ax in axes:
        ax.grid(axis="x", visible=False)
        ax.set_ylabel("customers")
    fig.tight_layout()
    save(fig, "fig_income_log")


def fig_purpose_counts(df: pd.DataFrame) -> None:
    counts = df["purpose"].value_counts()
    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    bars = ax.barh(counts.index[::-1], counts.values[::-1], color=BLUE,
                   height=0.62)
    for bar, c in zip(bars, counts.values[::-1]):
        ax.text(c + counts.max() * 0.012, bar.get_y() + bar.get_height() / 2,
                f"{c:,}", va="center", fontsize=11)
    ax.set_xlim(0, counts.max() * 1.15)
    ax.set_xlabel("customers")
    ax.grid(axis="y", visible=False)
    save(fig, "fig_purpose_counts")


def fig_default_by_utilization(df: pd.DataFrame) -> None:
    bins = pd.qcut(df["credit_utilization"], 10, duplicates="drop")
    rate = df.groupby(bins, observed=True)["default"].mean() * 100
    centers = [iv.mid for iv in rate.index]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.bar(range(len(rate)), rate.values, color=BLUE, width=0.7)
    avg = df["default"].mean() * 100
    ax.axhline(avg, color=GRAY, linestyle="--", linewidth=1.4)
    ax.set_ylim(0, rate.max() * 1.18)
    ax.text(0.02, 0.94, f"dashed line = portfolio average ({avg:.0f}%)",
            color="#5B6570", fontsize=11, transform=ax.transAxes)
    ax.set_xticks(range(len(rate)))
    ax.set_xticklabels([f"{c:.2f}" for c in centers], fontsize=10)
    ax.set_xlabel("credit utilisation (decile mid-points)")
    ax.set_ylabel("default rate (%)")
    ax.grid(axis="x", visible=False)
    save(fig, "fig_default_by_utilization")


def fig_default_by_checking(df: pd.DataFrame) -> None:
    order = ["none", "low", "medium", "high"]
    rate = df.groupby("checking_status")["default"].mean().reindex(order) * 100
    colors = [RED if c == "none" else BLUE for c in order]
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    bars = ax.bar(order, rate.values, color=colors, width=0.6)
    for bar, r in zip(bars, rate.values):
        ax.text(bar.get_x() + bar.get_width() / 2, r + 0.6, f"{r:.0f}%",
                ha="center", fontsize=11)
    avg = df["default"].mean() * 100
    ax.axhline(avg, color=GRAY, linestyle="--", linewidth=1.4)
    ax.set_ylim(0, rate.max() * 1.22)
    ax.text(0.98, 0.94, f"dashed line = portfolio average ({avg:.0f}%)",
            color="#5B6570", fontsize=11, ha="right", transform=ax.transAxes)
    ax.set_xlabel("checking-account status")
    ax.set_ylabel("default rate (%)")
    ax.grid(axis="x", visible=False)
    save(fig, "fig_default_by_checking")


def fig_corr_heatmap(df: pd.DataFrame) -> None:
    cols = [
        "age", "monthly_income_pln", "employment_length_years",
        "num_dependents", "num_existing_loans", "loan_amount_pln",
        "loan_term_months", "credit_utilization", "debt_to_income",
        "savings_balance_pln", "default",
    ]
    short = [
        "age", "income", "employment", "dependents", "loans#", "loan amt",
        "term", "utilisation", "DTI", "savings", "default",
    ]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(short)))
    ax.set_yticks(range(len(short)))
    ax.set_xticklabels(short, rotation=45, ha="right", fontsize=12)
    ax.set_yticklabels(short, fontsize=12)
    # Annotate only meaningful cells; a number in every cell is unreadable noise.
    for i in range(len(short)):
        for j in range(len(short)):
            v = corr.iloc[i, j]
            if i != j and abs(v) >= 0.13:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=10,
                        color="white" if abs(v) > 0.6 else "#333333")
    ax.grid(visible=False)
    fig.colorbar(im, ax=ax, shrink=0.75, label="Pearson correlation")
    save(fig, "fig_corr_heatmap")


def fig_anscombe() -> None:
    # Anscombe (1973), the classic quartet.
    x123 = [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5]
    ys = {
        "I": [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68],
        "II": [9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74],
        "III": [7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73],
    }
    x4 = [8, 8, 8, 8, 8, 8, 8, 19, 8, 8, 8]
    y4 = [6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 12.50, 5.56, 7.91, 6.89]
    sets = [("I", x123, ys["I"]), ("II", x123, ys["II"]),
            ("III", x123, ys["III"]), ("IV", x4, y4)]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.4), sharex=True, sharey=True)
    for ax, (name, x, y) in zip(axes.flat, sets):
        ax.scatter(x, y, color=BLUE, s=28, zorder=3)
        b, a = np.polyfit(x, y, 1)
        xs = np.array([3, 20])
        ax.plot(xs, a + b * xs, color=RED, linewidth=1.5)
        ax.set_title(f"dataset {name}", fontsize=11)
        ax.grid(alpha=0.2)
    fig.suptitle(
        "Four datasets, identical summaries: mean, variance, corr = 0.816, "
        "same regression line", fontsize=11, y=1.00)
    fig.tight_layout()
    save(fig, "fig_anscombe")


def fig_outliers_box(df: pd.DataFrame) -> None:
    inc = df["monthly_income_pln"].dropna()
    q1, q3 = inc.quantile([0.25, 0.75])
    iqr = q3 - q1
    fence = q3 + 1.5 * iqr
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    bp = ax.boxplot(inc, orientation="horizontal", widths=0.5, patch_artist=True,
                    flierprops=dict(marker="o", markersize=3.5,
                                    markerfacecolor=RED, markeredgecolor=RED,
                                    alpha=0.5))
    bp["boxes"][0].set(facecolor=LIGHT, edgecolor=DARK)
    for element in ("whiskers", "caps", "medians"):
        for line in bp[element]:
            line.set(color=DARK, linewidth=1.3)
    ax.axvline(fence, color=RED, linestyle=":", linewidth=1.4)
    ax.text(fence * 1.01, 1.32, f"upper fence = Q3 + 1.5·IQR ≈ {fence/1000:.0f}k",
            color=RED, fontsize=10.5)
    ax.set_yticks([])
    ax.xaxis.set_major_formatter(KPLN)
    ax.set_xlabel("monthly income (k PLN)")
    ax.grid(axis="y", visible=False)
    save(fig, "fig_outliers_box")


def fig_missingness(df: pd.DataFrame) -> None:
    miss = df.isna().mean() * 100
    miss = miss[miss > 0].sort_values()
    mechanism = {
        "monthly_income_pln": "MNAR — high earners decline to state income",
        "employment_length_years": "MAR — renters skip the question",
        "months_since_last_delinquency": "structural — never delinquent",
    }
    fig, ax = plt.subplots(figsize=(7.4, 2.8))
    bars = ax.barh(miss.index, miss.values, color=BLUE, height=0.55)
    for bar, (col, v) in zip(bars, miss.items()):
        ax.text(v + 1.2, bar.get_y() + bar.get_height() / 2,
                f"{v:.0f}%  ({mechanism[col]})", va="center", fontsize=10.5)
    ax.set_xlim(0, 100)
    ax.set_xlabel("share of rows missing (%)")
    ax.grid(axis="y", visible=False)
    save(fig, "fig_missingness")


def fig_missing_matrix(df: pd.DataFrame) -> None:
    sample = df.sample(300, random_state=1).reset_index(drop=True)
    present = sample.notna().astype(int)
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.imshow(present.T, aspect="auto", cmap="Blues", vmin=-0.4, vmax=1.4,
              interpolation="nearest")
    ax.set_yticks(range(len(present.columns)))
    ax.set_yticklabels(present.columns, fontsize=10.5)
    ax.set_xlabel("customers (sample of 300)", fontsize=12)
    ax.grid(visible=False)
    ax.set_title("white = missing value", fontsize=13)
    save(fig, "fig_missing_matrix")


def fig_structural_signal(df: pd.DataFrame) -> None:
    is_nan = df["months_since_last_delinquency"].isna()
    rates = [
        df.loc[is_nan, "default"].mean() * 100,
        df.loc[~is_nan, "default"].mean() * 100,
    ]
    labels = ["never delinquent\n(structural NaN)", "has delinquency\nhistory"]
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    bars = ax.bar(labels, rates, color=[BLUE, RED], width=0.5)
    for bar, r in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, r + 0.7, f"{r:.0f}%",
                ha="center", fontsize=12)
    ax.set_ylabel("default rate (%)")
    ax.grid(axis="x", visible=False)
    save(fig, "fig_structural_signal")


def fig_interaction(df: pd.DataFrame) -> None:
    rent = df["housing"].eq("rent")
    none = df["checking_status"].eq("none")
    groups = [
        ("whole portfolio", df["default"].mean(), GRAY),
        ("renter", df.loc[rent, "default"].mean(), BLUE),
        ("no checking acct", df.loc[none, "default"].mean(), BLUE),
        ("renter AND\nno checking acct", df.loc[rent & none, "default"].mean(), RED),
    ]
    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    labels = [g[0] for g in groups]
    rates = [g[1] * 100 for g in groups]
    colors = [g[2] for g in groups]
    bars = ax.bar(labels, rates, color=colors, width=0.55)
    for bar, r in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, r + 0.7, f"{r:.0f}%",
                ha="center", fontsize=11)
    ax.set_ylim(0, max(rates) * 1.18)
    ax.set_ylabel("default rate (%)")
    ax.tick_params(axis="x", labelsize=10)
    ax.grid(axis="x", visible=False)
    save(fig, "fig_interaction")


def fig_simpson() -> None:
    # Illustrative (constructed) data: within each income band, larger loans
    # mean higher default; in aggregate, larger loans mean lower default,
    # because high-income customers take the large loans.
    x_low = np.linspace(5, 40, 8)            # loan amount, k PLN
    y_low = 22 + 0.55 * x_low                # default %, rising in amount
    x_high = np.linspace(40, 80, 8)
    y_high = 2 + 0.30 * x_high

    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    ax.scatter(x_low, y_low, color=BLUE, s=36, label="low-income band")
    ax.scatter(x_high, y_high, color=DARK, s=36, marker="s",
               label="high-income band")
    for x, y in [(x_low, y_low), (x_high, y_high)]:
        b, a = np.polyfit(x, y, 1)
        ax.plot(x, a + b * x, color=GRAY, linewidth=1.4)
    x_all = np.concatenate([x_low, x_high])
    y_all = np.concatenate([y_low, y_high])
    b, a = np.polyfit(x_all, y_all, 1)
    xs = np.array([5, 80])
    ax.plot(xs, a + b * xs, color=RED, linewidth=1.8, linestyle="--",
            label="aggregate trend")
    ax.set_xlabel("loan amount (k PLN)")
    ax.set_ylabel("default rate (%)")
    ax.legend(fontsize=10, frameon=False)
    save(fig, "fig_simpson")


def fig_time_split() -> None:
    phases = [
        "Understand the business question",
        "Gather, clean & prepare data",
        "Explore (EDA)",
        "Model",
        "Evaluate & communicate",
    ]
    share = [10, 45, 20, 10, 15]
    colors = [BLUE, RED, BLUE, BLUE, BLUE]
    fig, ax = plt.subplots(figsize=(7.0, 2.9))
    bars = ax.barh(phases[::-1], share[::-1], color=colors[::-1], height=0.55)
    for bar, s in zip(bars, share[::-1]):
        ax.text(s + 1, bar.get_y() + bar.get_height() / 2, f"~{s}%",
                va="center", fontsize=11)
    ax.set_xlim(0, 55)
    ax.set_xlabel("share of a typical project's time (stylised)")
    ax.grid(axis="y", visible=False)
    save(fig, "fig_time_split")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    FIG_DIR.mkdir(exist_ok=True)
    df = load_credit(source="synthetic", verbose=True)
    fig_target_balance(df)
    fig_income_hist(df)
    fig_income_log(df)
    fig_purpose_counts(df)
    fig_default_by_utilization(df)
    fig_default_by_checking(df)
    fig_corr_heatmap(df)
    fig_anscombe()
    fig_outliers_box(df)
    fig_missingness(df)
    fig_missing_matrix(df)
    fig_structural_signal(df)
    fig_interaction(df)
    fig_simpson()
    fig_time_split()
    logger.info("All figures written to %s", FIG_DIR)


if __name__ == "__main__":
    main()
