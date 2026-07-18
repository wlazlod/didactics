"""Generate all figures for the Block 3 slide deck (trees, ensembles,
honest evaluation, leakage).

Usage (from the data-mining/ directory):
    uv run python lecture_3/make_figures.py

Outputs PDF figures into lecture_3/figures/. Deterministic.
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
GREEN = "#4C956C"
ORANGE = "#E8A33D"

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
# Shared data & models (same features as Block 2)
# ---------------------------------------------------------------------
def prepare_data():
    df = load_taiwan(verbose=False)
    return taiwan_features(df), df["default"], df


def split(X, y):
    from sklearn.model_selection import train_test_split

    return train_test_split(X, y, test_size=0.3, random_state=SEED, stratify=y)


# Lightly tuned GBDT (defaults overfit this small portfolio badly).
GBDT_KW = dict(learning_rate=0.03, max_leaf_nodes=7, min_samples_leaf=60,
               early_stopping=True, random_state=SEED)


def make_models():
    from sklearn.ensemble import (HistGradientBoostingClassifier,
                                  RandomForestClassifier)
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier

    return {
        "logistic": make_pipeline(StandardScaler(),
                                  LogisticRegression(max_iter=2000)),
        "tree (depth 4)": DecisionTreeClassifier(max_depth=4,
                                                 min_samples_leaf=50,
                                                 random_state=SEED),
        "random forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=10, random_state=SEED, n_jobs=-1
        ),
        "GBDT (tuned)": HistGradientBoostingClassifier(**GBDT_KW),
    }


# ---------------------------------------------------------------------
def fig_tree_example(X, y) -> None:
    from sklearn.tree import DecisionTreeClassifier, plot_tree

    X_tr, _, y_tr, _ = split(X, y)
    tree = DecisionTreeClassifier(max_depth=2, min_samples_leaf=50,
                                  random_state=SEED).fit(X_tr, y_tr)
    fig, ax = plt.subplots(figsize=(10.5, 3.8))
    plot_tree(tree, feature_names=list(X.columns), class_names=["repaid", "default"],
              filled=True, impurity=False, proportion=True, rounded=True,
              fontsize=11, ax=ax)
    ax.grid(visible=False)
    save(fig, "fig_tree_example")


def fig_boundaries(X, y) -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier

    cols = ["utilisation", "months_delayed_6m"]
    X2 = X[cols]
    X_tr, _, y_tr, _ = split(X2, y)
    lo = LogisticRegression().fit(X_tr, y_tr)
    tr = DecisionTreeClassifier(max_depth=3, min_samples_leaf=60,
                                random_state=SEED).fit(X_tr, y_tr)

    xx, yy = np.meshgrid(np.linspace(0, 1.4, 250), np.linspace(-0.3, 6.3, 250))
    grid = pd.DataFrame({cols[0]: xx.ravel(), cols[1]: yy.ravel()})

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4), sharey=True)
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(X_tr), 700, replace=False)
    for ax, model, title in [(axes[0], lo, "logistic: one straight cut"),
                             (axes[1], tr, "tree: axis-aligned rectangles")]:
        zz = model.predict_proba(grid)[:, 1].reshape(xx.shape)
        ax.contourf(xx, yy, zz, levels=[0, 0.35, 1], colors=["#EAF1F8", "#F6DEDB"])
        ax.contour(xx, yy, zz, levels=[0.35], colors=[DARK], linewidths=1.4)
        sub = X_tr.iloc[idx]
        ysub = y_tr.iloc[idx]
        ax.scatter(sub[cols[0]][ysub == 0], sub[cols[1]][ysub == 0], s=5,
                   color=GRAY, alpha=0.5, edgecolors="none")
        ax.scatter(sub[cols[0]][ysub == 1], sub[cols[1]][ysub == 1], s=5,
                   color=RED, alpha=0.5, edgecolors="none")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("utilisation")
        ax.grid(visible=False)
    axes[0].set_ylabel("months delayed (6m)")
    fig.tight_layout()
    save(fig, "fig_boundaries")


def fig_tree_depth(X, y) -> None:
    from sklearn.metrics import roc_auc_score
    from sklearn.tree import DecisionTreeClassifier

    X_tr, X_te, y_tr, y_te = split(X, y)
    depths = range(1, 17)
    tr_auc, te_auc = [], []
    for d in depths:
        t = DecisionTreeClassifier(max_depth=d, random_state=SEED).fit(X_tr, y_tr)
        tr_auc.append(roc_auc_score(y_tr, t.predict_proba(X_tr)[:, 1]))
        te_auc.append(roc_auc_score(y_te, t.predict_proba(X_te)[:, 1]))
    fig, ax = plt.subplots(figsize=(6.8, 3.3))
    ax.plot(depths, tr_auc, "-o", ms=4, color=BLUE, label="training AUC")
    ax.plot(depths, te_auc, "-s", ms=4, color=RED, label="test AUC")
    best = list(depths)[int(np.argmax(te_auc))]
    ax.axvline(best, color=GRAY, linestyle=":", linewidth=1.3)
    ax.set_xlabel("max_depth (the tree's leash)")
    ax.set_ylabel("AUC")
    ax.legend(frameon=False, fontsize=11)
    save(fig, "fig_tree_depth")
    logger.info("tree best depth=%d test AUC=%.3f", best, max(te_auc))


def fig_rf_ntrees(X, y) -> None:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score

    X_tr, X_te, y_tr, y_te = split(X, y)
    ns = [1, 2, 5, 10, 25, 50, 100, 200, 400]
    aucs = []
    for n in ns:
        rf = RandomForestClassifier(n_estimators=n, min_samples_leaf=5,
                                    random_state=SEED, n_jobs=-1).fit(X_tr, y_tr)
        aucs.append(roc_auc_score(y_te, rf.predict_proba(X_te)[:, 1]))
    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    ax.plot(ns, aucs, "-o", ms=4.5, color=BLUE)
    ax.set_xscale("log")
    ax.set_xlabel("number of trees")
    ax.set_ylabel("test AUC")
    save(fig, "fig_rf_ntrees")


def _cv_aucs(X, y):
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    X_tr, _, y_tr, _ = split(X, y)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    out = {}
    for name, model in make_models().items():
        out[name] = cross_val_score(model, X_tr, y_tr, cv=cv, scoring="roc_auc",
                                    n_jobs=-1)
    return out


def fig_cv_spread(cv_results) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 3.3))
    names = list(cv_results)
    for i, n in enumerate(names):
        vals = cv_results[n]
        ax.scatter([i] * len(vals), vals, s=40, color=BLUE, alpha=0.7, zorder=3)
        ax.plot([i - 0.18, i + 0.18], [vals.mean()] * 2, color=RED, linewidth=2.2,
                zorder=4)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=10.5)
    ax.set_ylabel("fold AUC (5-fold CV)")
    ax.grid(axis="x", visible=False)
    save(fig, "fig_cv_spread")
    for n, v in cv_results.items():
        logger.info("CV %s: %.3f +/- %.3f", n, v.mean(), v.std())


def fig_model_comparison(X, y) -> None:
    from sklearn.metrics import roc_auc_score

    X_tr, X_te, y_tr, y_te = split(X, y)
    aucs = {}
    for name, model in make_models().items():
        model.fit(X_tr, y_tr)
        aucs[name] = roc_auc_score(y_te, model.predict_proba(X_te)[:, 1])
    fig, ax = plt.subplots(figsize=(7.0, 3.1))
    names = list(aucs)
    colors = [GRAY, BLUE, BLUE, DARK]
    bars = ax.bar(names, [aucs[n] for n in names], color=colors, width=0.55)
    for b, n in zip(bars, names):
        ax.text(b.get_x() + b.get_width() / 2, aucs[n] + 0.004,
                f"{aucs[n]:.3f}", ha="center", fontsize=11)
    ax.set_ylim(0.5, max(aucs.values()) * 1.06)
    ax.set_ylabel("test AUC")
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="x", labelsize=10.5)
    save(fig, "fig_model_comparison")
    logger.info("test AUCs: %s", {k: round(v, 3) for k, v in aucs.items()})


def _fitted_probas(X, y):
    from sklearn.metrics import roc_auc_score

    X_tr, X_te, y_tr, y_te = split(X, y)
    models = make_models()
    probas, aucs = {}, {}
    for name in ["logistic", "GBDT (tuned)"]:
        m = models[name]
        m.fit(X_tr, y_tr)
        probas[name] = m.predict_proba(X_te)[:, 1]
        aucs[name] = roc_auc_score(y_te, probas[name])
    return y_te.to_numpy(), probas, aucs


def fig_roc(y_te, probas, aucs) -> None:
    from sklearn.metrics import roc_curve

    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    for name, color in [("logistic", BLUE), ("GBDT (tuned)", ORANGE)]:
        fpr, tpr, _ = roc_curve(y_te, probas[name])
        ax.plot(fpr, tpr, color=color, linewidth=2.0,
                label=f"{name} (AUC {aucs[name]:.3f})")
    ax.plot([0, 1], [0, 1], color=GRAY, linestyle="--", linewidth=1.2,
            label="coin flip (AUC 0.5)")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate (recall)")
    ax.legend(frameon=False, fontsize=11, loc="lower right")
    save(fig, "fig_roc")


def fig_pr(y_te, probas) -> None:
    from sklearn.metrics import average_precision_score, precision_recall_curve

    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    base = y_te.mean()
    for name, color in [("logistic", BLUE), ("GBDT (tuned)", ORANGE)]:
        prec, rec, _ = precision_recall_curve(y_te, probas[name])
        ap = average_precision_score(y_te, probas[name])
        ax.plot(rec, prec, color=color, linewidth=2.0,
                label=f"{name} (AP {ap:.3f})")
    ax.axhline(base, color=GRAY, linestyle="--", linewidth=1.2,
               label=f"random (base rate {base:.0%})")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=11, loc="upper right")
    save(fig, "fig_pr")


def fig_calibration(y_te, probas) -> None:
    from sklearn.calibration import calibration_curve

    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    for name, color, marker in [("logistic", BLUE, "o"),
                                ("GBDT (tuned)", ORANGE, "s")]:
        frac, mean_pred = calibration_curve(y_te, probas[name], n_bins=10,
                                            strategy="quantile")
        ax.plot(mean_pred, frac, marker=marker, ms=5, color=color,
                linewidth=1.8, label=name)
    ax.plot([0, 0.75], [0, 0.75], color=GRAY, linestyle="--", linewidth=1.2,
            label="perfect calibration")
    ax.set_xlim(0, 0.75)
    ax.set_ylim(0, 0.75)
    ax.set_xlabel("mean predicted PD (bin)")
    ax.set_ylabel("observed default rate (bin)")
    ax.legend(frameon=False, fontsize=11, loc="upper left")
    save(fig, "fig_calibration")


def fig_leakage(X, y, df) -> None:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score

    X_leak = X.assign(collections_contact=df["collections_contact"])
    aucs = {}
    for label, data in [("honest features", X), ("+ collections_contact", X_leak)]:
        X_tr, X_te, y_tr, y_te = split(data, y)
        m = HistGradientBoostingClassifier(**GBDT_KW).fit(X_tr, y_tr)
        aucs[label] = roc_auc_score(y_te, m.predict_proba(X_te)[:, 1])
    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    names = list(aucs)
    bars = ax.barh(names, [aucs[n] for n in names], color=[BLUE, RED], height=0.5)
    for b, n in zip(bars, names):
        ax.text(aucs[n] + 0.008, b.get_y() + b.get_height() / 2, f"{aucs[n]:.3f}",
                va="center", fontsize=12)
    ax.set_xlim(0.5, 1.05)
    ax.set_xlabel("test AUC (GBDT)")
    ax.grid(axis="y", visible=False)
    ax.invert_yaxis()
    save(fig, "fig_leakage")
    logger.info("leakage AUCs: %s", {k: round(v, 3) for k, v in aucs.items()})


def fig_importance(X, y) -> None:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.inspection import permutation_importance

    X_tr, X_te, y_tr, y_te = split(X, y)
    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                random_state=SEED, n_jobs=-1).fit(X_tr, y_tr)
    imp = pd.Series(rf.feature_importances_, index=X.columns)
    perm = permutation_importance(rf, X_te, y_te, n_repeats=10,
                                  random_state=SEED, scoring="roc_auc",
                                  n_jobs=-1)
    pim = pd.Series(perm.importances_mean, index=X.columns)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    for ax, series, title in [
        (axes[0], imp.sort_values(), "impurity importance (train-side)"),
        (axes[1], pim.sort_values(), "permutation importance (test AUC drop)"),
    ]:
        ax.barh(series.index, series.values, color=BLUE, height=0.6)
        ax.set_title(title, fontsize=12)
        ax.tick_params(axis="y", labelsize=10.5)
        ax.tick_params(axis="x", labelsize=10)
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, "fig_importance")


def fig_svm_margin() -> None:
    """Maximum margin on nearly separable 2-D data, support vectors circled."""
    from sklearn.datasets import make_blobs
    from sklearn.svm import SVC

    X, y = make_blobs(n_samples=120, centers=2, cluster_std=1.05,
                      random_state=11)
    svc = SVC(kernel="linear", C=10).fit(X, y)

    xx = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 200)
    yy = np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 200)
    XX, YY = np.meshgrid(xx, yy)
    Z = svc.decision_function(np.c_[XX.ravel(), YY.ravel()]).reshape(XX.shape)

    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.scatter(X[y == 0, 0], X[y == 0, 1], s=22, color=GRAY, alpha=0.8,
               edgecolors="none")
    ax.scatter(X[y == 1, 0], X[y == 1, 1], s=22, color=RED, alpha=0.8,
               edgecolors="none")
    ax.contour(XX, YY, Z, levels=[-1, 0, 1], colors=[BLUE, DARK, BLUE],
               linestyles=["--", "-", "--"], linewidths=[1.3, 2.0, 1.3])
    sv = svc.support_vectors_
    ax.scatter(sv[:, 0], sv[:, 1], s=130, facecolors="none",
               edgecolors=DARK, linewidths=1.6, label="support vectors")
    ax.set_xticks([]); ax.set_yticks([])
    ax.grid(visible=False)
    ax.legend(frameon=False, fontsize=10.5, loc="upper left")
    save(fig, "fig_svm_margin")


def fig_svm_kernel() -> None:
    """Linear vs RBF SVM on the course's crescents."""
    from sklearn.datasets import make_moons
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    X, y = make_moons(n_samples=600, noise=0.18, random_state=SEED)
    xx, yy = np.meshgrid(np.linspace(-1.8, 2.8, 220),
                         np.linspace(-1.3, 1.8, 220))
    grid = np.c_[xx.ravel(), yy.ravel()]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.1))
    for ax, kernel, title in [(axes[0], "linear", "linear kernel: still a line"),
                              (axes[1], "rbf", "RBF kernel: the lifted view")]:
        m = make_pipeline(StandardScaler(), SVC(kernel=kernel)).fit(X, y)
        zz = m.decision_function(grid).reshape(xx.shape)
        ax.contourf(xx, yy, zz, levels=[-99, 0, 99],
                    colors=["#EAF1F8", "#F6DEDB"])
        ax.contour(xx, yy, zz, levels=[0], colors=[DARK], linewidths=1.5)
        ax.scatter(X[:, 0], X[:, 1], s=5, c=np.where(y, RED, GRAY),
                   alpha=0.45, edgecolors="none")
        ax.set_title(title, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        ax.grid(visible=False)
    fig.tight_layout()
    save(fig, "fig_svm_kernel")


def fig_losses() -> None:
    """Zero-one, hinge, and logistic loss against the margin y*f(x)."""
    m = np.linspace(-2.5, 3, 400)
    zero_one = (m < 0).astype(float)
    hinge = np.maximum(0, 1 - m)
    logistic = np.log2(1 + np.exp(-m))
    fig, ax = plt.subplots(figsize=(6.6, 3.3))
    ax.plot(m, zero_one, color=GRAY, linewidth=2.0, linestyle=":",
            label="0/1 loss (what we mean)")
    ax.plot(m, hinge, color=RED, linewidth=2.0,
            label="hinge (SVM)")
    ax.plot(m, logistic, color=BLUE, linewidth=2.0,
            label="logistic (Block 2)")
    ax.axvline(1, color="#C4CAD1", linewidth=0.9, linestyle="--")
    ax.text(1.05, 2.6, "margin = 1", fontsize=9.5, color="#5B6570")
    ax.set_xlabel(r"margin $y \cdot f(x)$  (right and confident $\rightarrow$)")
    ax.set_ylabel("loss")
    ax.set_ylim(-0.1, 3.2)
    ax.legend(frameon=False, fontsize=10)
    save(fig, "fig_losses")


def svm_tournament_entry(X, y) -> None:
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    X_tr, X_te, y_tr, y_te = split(X, y)
    sub = X_tr.sample(6000, random_state=SEED)
    m = make_pipeline(StandardScaler(), SVC(kernel="rbf")).fit(sub, y_tr.loc[sub.index])
    auc = roc_auc_score(y_te, m.decision_function(X_te))
    logger.info("SVM (RBF) test AUC: %.3f", auc)


def fig_expected_cost(y_te, probas) -> None:
    proba = probas["GBDT (tuned)"]
    thresholds = np.linspace(0.02, 0.9, 120)
    C_FN, C_FP = 3.0, 1.0
    costs = []
    for t in thresholds:
        pred = proba >= t
        fn = ((pred == 0) & (y_te == 1)).sum()
        fp = ((pred == 1) & (y_te == 0)).sum()
        costs.append((fn * C_FN + fp * C_FP) / len(y_te))
    i = int(np.argmin(costs))
    fig, ax = plt.subplots(figsize=(6.8, 3.3))
    ax.plot(thresholds, costs, color=BLUE, linewidth=2.0)
    ax.axvline(thresholds[i], color=RED, linestyle=":", linewidth=1.4)
    ax.text(thresholds[i] - 0.02, max(costs) * 0.94,
            f"optimal threshold ≈ {thresholds[i]:.2f}", color=RED,
            fontsize=11, ha="right")
    ax.axvline(0.5, color=GRAY, linestyle="--", linewidth=1.2)
    ax.text(0.515, min(costs) * 0.82, "default 0.5", color="#5B6570",
            fontsize=10)
    ax.set_ylim(min(costs) * 0.72, max(costs) * 1.05)
    ax.set_xlabel("decision threshold")
    ax.set_ylabel("expected cost per customer\n(FN = 3 × FP)")
    save(fig, "fig_expected_cost")
    logger.info("optimal threshold=%.2f cost=%.3f (cost@0.5=%.3f)",
                thresholds[i], costs[i],
                costs[int(np.argmin(np.abs(thresholds - 0.5)))])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    FIG_DIR.mkdir(exist_ok=True)
    X, y, df = prepare_data()
    fig_tree_example(X, y)
    fig_boundaries(X, y)
    fig_tree_depth(X, y)
    fig_rf_ntrees(X, y)
    cv = _cv_aucs(X, y)
    fig_cv_spread(cv)
    fig_model_comparison(X, y)
    y_te, probas, aucs = _fitted_probas(X, y)
    fig_roc(y_te, probas, aucs)
    fig_pr(y_te, probas)
    fig_calibration(y_te, probas)
    fig_leakage(X, y, df)
    fig_importance(X, y)
    fig_expected_cost(y_te, probas)
    fig_svm_margin()
    fig_svm_kernel()
    fig_losses()
    svm_tournament_entry(X, y)
    logger.info("All figures written to %s", FIG_DIR)


if __name__ == "__main__":
    main()
