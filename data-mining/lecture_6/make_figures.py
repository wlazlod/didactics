"""Generate all figures for the Block 6 slide deck (neural networks &
responsible ML).

Usage (from the data-mining/ directory):
    uv run python lecture_6/make_figures.py

Outputs PDF figures into lecture_6/figures/. Deterministic. Logs the final
tournament, fairness gaps, and proxy-removal numbers used on the slides.
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
from finance_data import load_taiwan, taiwan_features  # noqa: E402

logger = logging.getLogger(__name__)

FIG_DIR = Path(__file__).parent / "figures"
SEED = 42

BLUE = "#2E6DA4"
DARK = "#003366"
RED = "#C83C28"
GRAY = "#8C959E"
LIGHT = "#D7E3F0"
ORANGE = "#E8A33D"

GBDT_KW = dict(learning_rate=0.03, max_leaf_nodes=7, min_samples_leaf=60,
               early_stopping=True, random_state=SEED)

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
def prepare_data():
    df = load_taiwan(verbose=False)
    return taiwan_features(df), df["default"], df


def split(X, y):
    from sklearn.model_selection import train_test_split

    return train_test_split(X, y, test_size=0.3, random_state=SEED, stratify=y)


def _mlp(**kw):
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    defaults = dict(hidden_layer_sizes=(32, 16), early_stopping=True,
                    random_state=SEED, max_iter=500)
    defaults.update(kw)
    return make_pipeline(StandardScaler(), MLPClassifier(**defaults))


# ---------------------------------------------------------------------
def fig_activations() -> None:
    z = np.linspace(-4, 4, 300)
    fig, axes = plt.subplots(1, 3, figsize=(8.8, 2.7))
    for ax, f, name in [
        (axes[0], 1 / (1 + np.exp(-z)), "sigmoid"),
        (axes[1], np.tanh(z), "tanh"),
        (axes[2], np.maximum(0, z), "ReLU"),
    ]:
        ax.plot(z, f, color=BLUE, linewidth=2.2)
        ax.axhline(0, color="#C4CAD1", linewidth=0.8)
        ax.axvline(0, color="#C4CAD1", linewidth=0.8)
        ax.set_title(name, fontsize=12)
    fig.tight_layout()
    save(fig, "fig_activations")


def fig_mlp_boundary() -> None:
    from sklearn.datasets import make_moons
    from sklearn.linear_model import LogisticRegression

    X, y = make_moons(n_samples=600, noise=0.18, random_state=SEED)
    lo = LogisticRegression().fit(X, y)
    nn = _mlp(hidden_layer_sizes=(16, 16), early_stopping=False).fit(X, y)

    xx, yy = np.meshgrid(np.linspace(-1.8, 2.8, 250),
                         np.linspace(-1.3, 1.8, 250))
    grid = np.c_[xx.ravel(), yy.ravel()]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.1))
    for ax, model, title in [(axes[0], lo, "logistic: one line"),
                             (axes[1], nn, "MLP: a learned curve")]:
        zz = model.predict_proba(grid)[:, 1].reshape(xx.shape)
        ax.contourf(xx, yy, zz, levels=[0, 0.5, 1],
                    colors=["#EAF1F8", "#F6DEDB"])
        ax.contour(xx, yy, zz, levels=[0.5], colors=[DARK], linewidths=1.5)
        ax.scatter(X[y == 0, 0], X[y == 0, 1], s=6, color=GRAY, alpha=0.55,
                   edgecolors="none")
        ax.scatter(X[y == 1, 0], X[y == 1, 1], s=6, color=RED, alpha=0.55,
                   edgecolors="none")
        ax.set_title(title, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        ax.grid(visible=False)
    fig.tight_layout()
    save(fig, "fig_mlp_boundary")


def fig_capacity() -> None:
    from sklearn.datasets import make_moons

    X, y = make_moons(n_samples=600, noise=0.18, random_state=SEED)
    xx, yy = np.meshgrid(np.linspace(-1.8, 2.8, 220),
                         np.linspace(-1.3, 1.8, 220))
    grid = np.c_[xx.ravel(), yy.ravel()]
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 2.8))
    for ax, h in zip(axes, [2, 16, 256]):
        nn = _mlp(hidden_layer_sizes=(h,), early_stopping=False,
                  max_iter=2000).fit(X, y)
        zz = nn.predict_proba(grid)[:, 1].reshape(xx.shape)
        ax.contourf(xx, yy, zz, levels=[0, 0.5, 1],
                    colors=["#EAF1F8", "#F6DEDB"])
        ax.contour(xx, yy, zz, levels=[0.5], colors=[DARK], linewidths=1.4)
        ax.scatter(X[:, 0], X[:, 1], s=4, c=np.where(y, RED, GRAY), alpha=0.4,
                   edgecolors="none")
        ax.set_title(f"{h} hidden units", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        ax.grid(visible=False)
    fig.tight_layout()
    save(fig, "fig_capacity")


def fig_loss_curve(X, y) -> None:
    X_tr, _, y_tr, _ = split(X, y)
    pipe = _mlp(validation_fraction=0.15)
    pipe.fit(X_tr, y_tr)
    nn = pipe.named_steps["mlpclassifier"]
    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    ax.plot(nn.loss_curve_, color=BLUE, linewidth=2.0, label="training loss")
    ax2 = ax.twinx()
    ax2.plot(nn.validation_scores_, color=RED, linewidth=2.0,
             label="validation accuracy")
    ax2.spines["right"].set_visible(True)
    best = int(np.argmax(nn.validation_scores_))
    ax.axvline(best, color=GRAY, linestyle=":", linewidth=1.4)
    ax.text(best - 0.6, max(nn.loss_curve_) * 0.98, "early stopping\npoint",
            fontsize=10, color="#5B6570", ha="right", va="top")
    from matplotlib.ticker import MaxNLocator

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("epoch")
    ax.set_ylabel("training loss", color=BLUE)
    ax2.set_ylabel("validation accuracy", color=RED)
    ax2.grid(visible=False)
    save(fig, "fig_loss_curve")


def fig_lr_effect(X, y) -> None:
    X_tr, _, y_tr, _ = split(X, y)
    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    for lr, color, label in [
        (3.0, RED, "too high (3.0): overshoots, settles worse"),
        (0.3, BLUE, "sensible (0.3)"),
        (0.00001, GRAY, "too low (0.00001): asleep"),
    ]:
        pipe = _mlp(solver="sgd", learning_rate_init=lr,
                    early_stopping=False, max_iter=80)
        pipe.fit(X_tr, y_tr)
        ax.plot(pipe.named_steps["mlpclassifier"].loss_curve_, color=color,
                linewidth=1.9, label=label)
    ax.set_ylim(0.40, 0.72)
    ax.set_xlabel("epoch")
    ax.set_ylabel("training loss (SGD)")
    ax.legend(frameon=False, fontsize=10)
    save(fig, "fig_lr_effect")


def fig_final_tournament(X, y) -> None:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X_tr, X_te, y_tr, y_te = split(X, y)
    models = {
        "logistic\n(Block 2)": make_pipeline(StandardScaler(),
                                             LogisticRegression(max_iter=2000)),
        "GBDT tuned\n(Block 3)": HistGradientBoostingClassifier(**GBDT_KW),
        "MLP (32,16)\n(Block 6)": _mlp(),
    }
    aucs = {}
    for name, m in models.items():
        m.fit(X_tr, y_tr)
        aucs[name] = roc_auc_score(y_te, m.predict_proba(X_te)[:, 1])
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    names = list(aucs)
    bars = ax.bar(names, [aucs[n] for n in names],
                  color=[GRAY, BLUE, DARK], width=0.5)
    for b, n in zip(bars, names):
        ax.text(b.get_x() + b.get_width() / 2, aucs[n] + 0.003,
                f"{aucs[n]:.3f}", ha="center", fontsize=11.5)
    ax.set_ylim(0.5, max(aucs.values()) * 1.06)
    ax.set_ylabel("test AUC")
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="x", labelsize=10.5)
    save(fig, "fig_final_tournament")
    logger.info("final tournament: %s", {k.split(chr(10))[0]: round(v, 3)
                                         for k, v in aucs.items()})


def _group_rates(pred, y_true, group):
    out = {}
    for g, name in [(group, "female"), (~group, "male")]:
        yt = y_true[g]
        pg = pred[g]
        out[name] = {
            "decline": pg.mean(),
            "FPR": ((pg == 1) & (yt == 0)).sum() / max((yt == 0).sum(), 1),
            "FNR": ((pg == 0) & (yt == 1)).sum() / max((yt == 1).sum(), 1),
            "base": yt.mean(),
        }
    return out


def fig_fairness(X, y, df) -> None:
    from sklearn.ensemble import HistGradientBoostingClassifier

    X_tr, X_te, y_tr, y_te = split(X, y)
    female = (df.loc[X_te.index, "sex"] == 2).to_numpy()
    y_arr = y_te.to_numpy()

    m1 = HistGradientBoostingClassifier(**GBDT_KW).fit(X_tr, y_tr)
    pred1 = m1.predict_proba(X_te)[:, 1] >= 0.26
    r1 = _group_rates(pred1, y_arr, female)

    X2 = X.drop(columns=["female"])
    m2 = HistGradientBoostingClassifier(**GBDT_KW).fit(
        X2.loc[X_tr.index], y_tr)
    pred2 = m2.predict_proba(X2.loc[X_te.index])[:, 1] >= 0.26
    r2 = _group_rates(pred2, y_arr, female)

    # --- gap chart (with age) ---
    metrics = ["decline", "FPR", "FNR"]
    labels = ["decline rate", "FPR (good customers\nwrongly declined)",
              "FNR (defaulters\nmissed)"]
    xs = np.arange(len(metrics))
    w = 0.36
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    ax.bar(xs - w / 2, [r1["female"][m] * 100 for m in metrics], w,
           color=ORANGE, label="female clients")
    ax.bar(xs + w / 2, [r1["male"][m] * 100 for m in metrics], w,
           color=BLUE, label="male clients")
    for i, m in enumerate(metrics):
        for dx, grp in [(-w / 2, "female"), (w / 2, "male")]:
            v = r1[grp][m] * 100
            ax.text(i + dx, v + 1.2, f"{v:.0f}%", ha="center", fontsize=10.5)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("rate (%)")
    ax.set_ylim(0, 85)
    ax.legend(frameon=False, fontsize=10.5)
    ax.grid(axis="x", visible=False)
    save(fig, "fig_fairness")
    logger.info("with sex: %s", {g: {k: round(v, 3) for k, v in d.items()}
                                 for g, d in r1.items()})

    # --- proxy-removal chart ---
    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    xs2 = np.arange(2)
    ax.bar(xs2 - w / 2, [r1["female"]["decline"] * 100,
                         r2["female"]["decline"] * 100], w,
           color=ORANGE, label="female clients")
    ax.bar(xs2 + w / 2, [r1["male"]["decline"] * 100,
                         r2["male"]["decline"] * 100], w,
           color=BLUE, label="male clients")
    for i, r in enumerate([r1, r2]):
        for dx, grp in [(-w / 2, "female"), (w / 2, "male")]:
            v = r[grp]["decline"] * 100
            ax.text(i + dx, v + 1.2, f"{v:.0f}%", ha="center", fontsize=10.5)
    ax.set_xticks(xs2)
    ax.set_xticklabels(["sex IN the model", "sex REMOVED"], fontsize=11)
    ax.set_ylabel("decline rate (%)")
    ax.set_ylim(0, 85)
    ax.legend(frameon=False, fontsize=10.5)
    ax.grid(axis="x", visible=False)
    save(fig, "fig_proxy")
    logger.info("sex removed: %s", {g: {k: round(v, 3) for k, v in d.items()}
                                    for g, d in r2.items()})


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    FIG_DIR.mkdir(exist_ok=True)
    X, y, df = prepare_data()
    fig_activations()
    fig_mlp_boundary()
    fig_capacity()
    fig_loss_curve(X, y)
    fig_lr_effect(X, y)
    fig_final_tournament(X, y)
    fig_fairness(X, y, df)
    logger.info("All figures written to %s", FIG_DIR)


if __name__ == "__main__":
    main()
