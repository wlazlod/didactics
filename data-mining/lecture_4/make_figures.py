"""Generate all figures for the Block 4 slide deck (clustering & PCA).

Usage (from the data-mining/ directory):
    uv run python lecture_4/make_figures.py

Outputs PDF figures into lecture_4/figures/. Deterministic. Also prints the
k=4 segment profile table used on the slides and in the script.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "lecture_1"))
from finance_data import load_taiwan  # noqa: E402

logger = logging.getLogger(__name__)

FIG_DIR = Path(__file__).parent / "figures"
SEED = 42

BLUE = "#2E6DA4"
DARK = "#003366"
RED = "#C83C28"
GRAY = "#8C959E"
LIGHT = "#D7E3F0"
GREEN = "#4C956C"
ORANGE = "#E8A33D"
PURPLE = "#7B5CA6"

# Fixed segment colors (identity follows the segment, never recycled)
SEG_COLORS = [BLUE, ORANGE, GREEN, PURPLE, RED, GRAY]

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
        "savefig.bbox": "tight",
    }
)


def save(fig: plt.Figure, name: str) -> None:
    path = FIG_DIR / f"{name}.pdf"
    fig.savefig(path)
    plt.close(fig)
    logger.info("wrote %s", path)


# ---------------------------------------------------------------------
# Data: clustering feature set (NO target, NO leakage column)
# ---------------------------------------------------------------------
def prepare_data():
    """Clustering features: behaviour + capacity, protected attrs excluded
    on purpose (sex/marriage out; the profile step may still LOOK at them)."""
    df = load_taiwan(verbose=False)
    pay_cols = [c for c in df.columns if c.startswith("pay_status_")]
    Xc = pd.DataFrame(
        {
            "log_limit": np.log10(df["limit_bal"]),
            "age": df["age"],
            "utilisation": (df["bill_amt_sep"] / df["limit_bal"]).clip(-0.5, 2.0),
            "months_delayed": (df[pay_cols] > 0).sum(axis=1),
            "pay_ratio": (df["pay_amt_sep"]
                          / df["bill_amt_sep"].clip(lower=1)).clip(0, 2.0),
            "bill_trend": ((df["bill_amt_sep"] - df["bill_amt_apr"])
                           / df["limit_bal"]).clip(-2, 2),
        }
    )
    return Xc, df


def scaled(Xc: pd.DataFrame) -> np.ndarray:
    from sklearn.preprocessing import StandardScaler

    return StandardScaler().fit_transform(Xc)


# ---------------------------------------------------------------------
def fig_scaling_matters(Xc: pd.DataFrame) -> None:
    from sklearn.cluster import KMeans

    from matplotlib.ticker import FuncFormatter

    raw = Xc[["age"]].assign(limit=10 ** Xc["log_limit"]).to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.3))
    for ax, data, title in [
        (axes[0], raw, "raw units: limit (NT$) dominates"),
        (axes[1], (raw - raw.mean(0)) / raw.std(0), "standardised: both count"),
    ]:
        km = KMeans(n_clusters=3, n_init=10, random_state=SEED).fit(data)
        for k in range(3):
            m = km.labels_ == k
            ax.scatter(data[m, 0], data[m, 1], s=6, alpha=0.55,
                       color=SEG_COLORS[k], edgecolors="none")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("age" + (" (years)" if data is raw else " (SD)"))
    axes[0].yaxis.set_major_formatter(
        FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
    axes[0].set_ylabel("limit (k NT$)")
    axes[1].set_ylabel("limit (SD)")
    fig.tight_layout()
    save(fig, "fig_scaling_matters")


def fig_kmeans_steps() -> None:
    from sklearn.datasets import make_blobs

    X, _ = make_blobs(n_samples=350, centers=3, cluster_std=1.1,
                      random_state=7)
    rng = np.random.default_rng(3)
    centers = X[rng.choice(len(X), 3, replace=False)]

    def assign(X, C):
        return np.argmin(((X[:, None, :] - C[None]) ** 2).sum(-1), axis=1)

    fig, axes = plt.subplots(1, 3, figsize=(9.0, 2.9))
    C = centers.copy()
    for it, ax in enumerate(axes):
        lab = assign(X, C)
        for k in range(3):
            ax.scatter(X[lab == k, 0], X[lab == k, 1], s=7, alpha=0.6,
                       color=SEG_COLORS[k], edgecolors="none")
        ax.scatter(C[:, 0], C[:, 1], marker="X", s=160, color="black",
                   edgecolors="white", linewidths=1.6, zorder=5)
        ax.set_title(["random start", "after 1 update", "converged"][it],
                     fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        ax.grid(visible=False)
        # update step for next panel
        for _ in range(1 if it == 0 else 8):
            lab = assign(X, C)
            C = np.array([X[lab == k].mean(0) for k in range(3)])
    fig.tight_layout()
    save(fig, "fig_kmeans_steps")


def _kmeans_labels(Xs: np.ndarray, k: int = 4):
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(Xs)
    return km


def fig_kmeans_portfolio(Xc: pd.DataFrame, Xs: np.ndarray) -> None:
    km = _kmeans_labels(Xs, 4)
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    for k in range(4):
        m = km.labels_ == k
        ax.scatter(Xc.loc[m, "utilisation"], Xc.loc[m, "log_limit"], s=9,
                   alpha=0.7, color=SEG_COLORS[k], edgecolors="none",
                   label=f"segment {k}")
    ax.set_xlabel("credit utilisation")
    ax.set_ylabel("log10 credit limit")
    ax.legend(frameon=False, fontsize=10.5, markerscale=2.6, ncol=4,
              loc="lower center", bbox_to_anchor=(0.5, 1.02))
    save(fig, "fig_kmeans_portfolio")


def fig_elbow_silhouette(Xs: np.ndarray) -> None:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    ks = range(2, 11)
    inertia, sil = [], []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(Xs)
        inertia.append(km.inertia_)
        sil.append(silhouette_score(Xs, km.labels_,
                                    sample_size=2000, random_state=SEED))
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.1))
    axes[0].plot(list(ks), inertia, "-o", ms=4, color=BLUE)
    axes[0].set_xlabel("k"); axes[0].set_ylabel("inertia (within-cluster SS)")
    axes[0].set_title("elbow: kink ≈ diminishing returns", fontsize=11)
    axes[1].plot(list(ks), sil, "-s", ms=4, color=DARK)
    axes[1].set_xlabel("k"); axes[1].set_ylabel("mean silhouette")
    axes[1].set_title("silhouette: higher = cleaner split", fontsize=11)
    best = list(ks)[int(np.argmax(sil))]
    axes[1].axvline(best, color=GRAY, linestyle=":", linewidth=1.2)
    fig.tight_layout()
    save(fig, "fig_elbow_silhouette")
    logger.info("silhouette best k=%d (values: %s)", best,
                [round(s, 3) for s in sil])


def fig_kmeans_fails() -> None:
    from sklearn.cluster import KMeans
    from sklearn.datasets import make_blobs

    rng = np.random.default_rng(SEED)
    # anisotropic
    X1, _ = make_blobs(n_samples=500, centers=3, random_state=170)
    X1 = X1 @ np.array([[0.6, -0.64], [-0.4, 0.85]])
    # unequal variance
    X2, _ = make_blobs(n_samples=500, centers=3,
                       cluster_std=[1.0, 2.8, 0.4], random_state=170)
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.1))
    for ax, X, title in [(axes[0], X1, "stretched clusters"),
                         (axes[1], X2, "unequal spreads")]:
        km = KMeans(n_clusters=3, n_init=10, random_state=SEED).fit(X)
        for k in range(3):
            m = km.labels_ == k
            ax.scatter(X[m, 0], X[m, 1], s=6, alpha=0.5,
                       color=SEG_COLORS[k], edgecolors="none")
        ax.set_title(f"k-means vs {title}", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        ax.grid(visible=False)
    fig.tight_layout()
    save(fig, "fig_kmeans_fails")


def fig_dendrogram(Xs: np.ndarray, df: pd.DataFrame) -> None:
    from scipy.cluster.hierarchy import dendrogram, linkage

    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(Xs), 60, replace=False)
    Z = linkage(Xs[idx], method="ward")
    fig, ax = plt.subplots(figsize=(8.6, 3.4))
    dendrogram(Z, ax=ax, color_threshold=7.0, no_labels=True,
               above_threshold_color=GRAY)
    ax.axhline(7.0, color=RED, linestyle="--", linewidth=1.3)
    ax.text(0.01, 0.93, "cut here → 4 segments", color=RED, fontsize=11,
            ha="left", transform=ax.transAxes)
    ax.set_ylabel("merge distance (Ward)")
    ax.grid(visible=False)
    save(fig, "fig_dendrogram")


def fig_dbscan_vs_kmeans() -> None:
    from sklearn.cluster import DBSCAN, KMeans
    from sklearn.datasets import make_moons

    X, _ = make_moons(n_samples=600, noise=0.07, random_state=SEED)
    km = KMeans(n_clusters=2, n_init=10, random_state=SEED).fit(X)
    db = DBSCAN(eps=0.12, min_samples=10).fit(X)
    n_found = len(set(db.labels_) - {-1})
    logger.info("DBSCAN clusters found: %d, noise points: %d",
                n_found, (db.labels_ == -1).sum())

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.1))
    for k in range(2):
        axes[0].scatter(X[km.labels_ == k, 0], X[km.labels_ == k, 1], s=8,
                        alpha=0.7, color=SEG_COLORS[k], edgecolors="none")
    axes[0].set_title("k-means: distance to a centre", fontsize=11)
    labs = db.labels_
    for k in sorted(set(labs) - {-1}):
        axes[1].scatter(X[labs == k, 0], X[labs == k, 1], s=8, alpha=0.7,
                        color=SEG_COLORS[k], edgecolors="none")
    noise = labs == -1
    axes[1].scatter(X[noise, 0], X[noise, 1], s=42, marker="x", color=RED,
                    linewidths=1.8, label="noise points")
    axes[1].set_title("DBSCAN: density, with noise", fontsize=11)
    axes[1].legend(frameon=False, fontsize=10, loc="upper right")
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
        ax.grid(visible=False)
    fig.tight_layout()
    save(fig, "fig_dbscan_vs_kmeans")


def fig_pca_arrows(Xc: pd.DataFrame) -> None:
    from sklearn.decomposition import PCA

    d = np.column_stack([Xc["utilisation"], Xc["bill_trend"]])
    d = (d - d.mean(0)) / d.std(0)
    pca = PCA(2).fit(d)
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(d), 4000, replace=False)
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.scatter(d[idx, 0], d[idx, 1], s=5, alpha=0.2, color=BLUE,
               edgecolors="none")
    for i, (comp, var) in enumerate(zip(pca.components_,
                                        pca.explained_variance_)):
        v = comp * np.sqrt(var) * 2.4
        ax.annotate("", xy=(v[0], v[1]), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", lw=2.4,
                                    color=RED if i == 0 else DARK))
        ax.text(v[0] * 1.25, v[1] * 1.25, f"PC{i+1}",
                color=RED if i == 0 else DARK, fontsize=12,
                ha="center")
    ax.set_xlabel("utilisation (SD)")
    ax.set_ylabel("bill trend (SD)")
    ax.set_xlim(-2.2, 3.2)
    ax.set_ylim(-2.6, 2.8)
    save(fig, "fig_pca_arrows")


def fig_scree(Xs: np.ndarray) -> None:
    from sklearn.decomposition import PCA

    pca = PCA().fit(Xs)
    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)
    xs = np.arange(1, len(evr) + 1)
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    ax.bar(xs, evr, color=BLUE, width=0.6, label="per component")
    ax.plot(xs, cum, "-o", ms=4, color=RED, label="cumulative")
    ax.axhline(0.9, color=GRAY, linestyle=":", linewidth=1.2)
    ax.text(1.0, 0.915, "90%", color="#5B6570", fontsize=10, ha="left")
    ax.set_xticks(xs)
    ax.set_xlabel("principal component")
    ax.set_ylabel("share of variance")
    ax.legend(frameon=False, fontsize=10.5, loc="center right")
    save(fig, "fig_scree")
    logger.info("explained variance: %s (cum %s)",
                [round(v, 3) for v in evr], [round(v, 3) for v in cum])


def fig_loadings(Xc: pd.DataFrame, Xs: np.ndarray) -> None:
    from sklearn.decomposition import PCA

    pca = PCA(4).fit(Xs)
    load = pd.DataFrame(pca.components_.T,
                        index=Xc.columns,
                        columns=[f"PC{i+1}" for i in range(4)])
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    im = ax.imshow(load, cmap="RdBu_r", vmin=-0.8, vmax=0.8)
    ax.set_xticks(range(4)); ax.set_xticklabels(load.columns, fontsize=11)
    ax.set_yticks(range(len(load))); ax.set_yticklabels(load.index, fontsize=10.5)
    for i in range(load.shape[0]):
        for j in range(load.shape[1]):
            v = load.iloc[i, j]
            if abs(v) >= 0.15:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=9, color="white" if abs(v) > 0.55 else "#333")
    ax.grid(visible=False)
    fig.colorbar(im, shrink=0.8, label="loading")
    save(fig, "fig_loadings")
    logger.info("PC1 loadings: %s", load["PC1"].round(2).to_dict())


def fig_pca_segments(Xs: np.ndarray, df: pd.DataFrame) -> None:
    from sklearn.decomposition import PCA

    km = _kmeans_labels(Xs, 4)
    proj = PCA(2).fit_transform(Xs)
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    for k in range(4):
        m = km.labels_ == k
        rate = df.loc[m, "default"].mean()
        ax.scatter(proj[m, 0], proj[m, 1], s=9, alpha=0.65,
                   color=SEG_COLORS[k], edgecolors="none",
                   label=f"segment {k} (PD {rate:.1%})")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.legend(frameon=False, fontsize=10.5, markerscale=2.6, ncol=2,
              loc="lower center", bbox_to_anchor=(0.5, 1.02))
    save(fig, "fig_pca_segments")


def fig_noise_clusters() -> None:
    from sklearn.cluster import KMeans

    rng = np.random.default_rng(SEED)
    X = rng.uniform(0, 1, size=(800, 2))
    km = KMeans(n_clusters=4, n_init=10, random_state=SEED).fit(X)
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    for k in range(4):
        m = km.labels_ == k
        ax.scatter(X[m, 0], X[m, 1], s=10, alpha=0.85, color=SEG_COLORS[k],
                   edgecolors="none")
    ax.set_xticks([]); ax.set_yticks([])
    ax.grid(visible=False)
    ax.set_title("k-means, k=4, on pure uniform noise", fontsize=11)
    save(fig, "fig_noise_clusters")


def segment_profile(Xc: pd.DataFrame, Xs: np.ndarray, df: pd.DataFrame) -> None:
    km = _kmeans_labels(Xs, 4)
    prof = Xc.assign(segment=km.labels_).groupby("segment").mean().round(2)
    prof["limit_ntd"] = (10 ** prof["log_limit"]).round(-3)
    prof["size"] = pd.Series(km.labels_).value_counts().sort_index()
    prof["default_rate"] = df.groupby(km.labels_)["default"].mean().round(3)
    logger.info("\n=== k=4 segment profile ===\n%s", prof.to_string())


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    FIG_DIR.mkdir(exist_ok=True)
    Xc, df = prepare_data()
    Xs = scaled(Xc)
    fig_scaling_matters(Xc)
    fig_kmeans_steps()
    fig_kmeans_portfolio(Xc, Xs)
    fig_elbow_silhouette(Xs)
    fig_kmeans_fails()
    fig_dendrogram(Xs, df)
    fig_dbscan_vs_kmeans()
    fig_pca_arrows(Xc)
    fig_scree(Xs)
    fig_loadings(Xc, Xs)
    fig_pca_segments(Xs, df)
    fig_noise_clusters()
    segment_profile(Xc, Xs, df)
    logger.info("All figures written to %s", FIG_DIR)


if __name__ == "__main__":
    main()
