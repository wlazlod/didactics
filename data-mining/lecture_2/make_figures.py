"""Generate all figures for the Block 2 slide deck (regression).

Usage (from the data-mining/ directory):
    uv run python lecture_2/make_figures.py

Outputs PDF figures into lecture_2/figures/. Deterministic: uses the
synthetic course portfolio with its fixed seed plus fixed local seeds.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

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
# Shared modelling data
# ---------------------------------------------------------------------
def prepare_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Return (X, y_default, df) for the real Taiwan portfolio."""
    df = load_taiwan(verbose=False)
    return taiwan_features(df), df["default"], df


# ---------------------------------------------------------------------
# Overfitting / bias-variance (constructed 1-D example)
# ---------------------------------------------------------------------
def _toy_curve(n: int = 60, seed: int = SEED):
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(0, 1, n))
    y = np.sin(2.2 * np.pi * x) * 0.7 + 0.5 + rng.normal(0, 0.18, n)
    return x, y


def fig_overfitting() -> None:
    x, y = _toy_curve()
    grid = np.linspace(0, 1, 300)
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.9), sharey=True)
    for ax, deg, title in [
        (axes[0], 1, "degree 1 — underfit"),
        (axes[1], 4, "degree 4 — about right"),
        (axes[2], 15, "degree 15 — overfit"),
    ]:
        coefs = np.polyfit(x, y, deg)
        ax.scatter(x, y, s=14, color=GRAY, alpha=0.8)
        ax.plot(grid, np.polyval(coefs, grid), color=RED if deg == 15 else BLUE,
                linewidth=1.8)
        ax.set_ylim(-0.6, 1.7)
        ax.set_title(title, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(visible=False)
    fig.tight_layout()
    save(fig, "fig_overfitting")


def fig_bias_variance() -> None:
    x, y = _toy_curve(n=120)
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(x))
    tr, te = idx[:80], idx[80:]
    degrees = range(1, 16)
    tr_err, te_err = [], []
    for d in degrees:
        c = np.polyfit(x[tr], y[tr], d)
        tr_err.append(np.mean((np.polyval(c, x[tr]) - y[tr]) ** 2))
        te_err.append(np.mean((np.polyval(c, x[te]) - y[te]) ** 2))
    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    ax.plot(degrees, tr_err, "-o", color=BLUE, markersize=4, label="training error")
    ax.plot(degrees, te_err, "-s", color=RED, markersize=4, label="test error")
    best = np.argmin(te_err)
    ax.axvline(list(degrees)[best], color=GRAY, linestyle=":", linewidth=1.3)
    ax.set_xlabel("model complexity (polynomial degree)")
    ax.set_ylabel("mean squared error")
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=11)
    save(fig, "fig_bias_variance")


# ---------------------------------------------------------------------
# Linear regression on the portfolio
# ---------------------------------------------------------------------
def _warmup_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Warm-up regression data: September payment vs August bill (payers)."""
    d = df[(df["pay_amt_sep"] > 0) & (df["bill_amt_aug"] > 0)]
    return d


def fig_linear_fit(df: pd.DataFrame) -> None:
    d = _warmup_frame(df)
    lx = np.log10(d["bill_amt_aug"])
    ly = np.log10(d["pay_amt_sep"])
    b, a = np.polyfit(lx, ly, 1)
    r2 = np.corrcoef(lx, ly)[0, 1] ** 2
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(d), 4000, replace=False)
    ax.scatter(lx.iloc[idx], ly.iloc[idx], s=5, alpha=0.15, color=BLUE,
               edgecolors="none")
    xs = np.array([lx.min(), lx.max()])
    ax.plot(xs, a + b * xs, color=RED, linewidth=2.0)
    ax.text(0.03, 0.93, f"$\\hat{{y}} = {a:.2f} + {b:.2f}\\,x$",
            transform=ax.transAxes, fontsize=12, color=RED)
    ax.set_xlabel("log10 August bill (NT\\$)")
    ax.set_ylabel("log10 September payment (NT\\$)")
    save(fig, "fig_linear_fit")
    logger.info("linear warm-up: slope=%.2f intercept=%.2f R2=%.3f (n=%d)",
                b, a, r2, len(d))


def fig_residuals(df: pd.DataFrame) -> None:
    d = _warmup_frame(df).sample(5000, random_state=SEED)
    x = np.log10(d["bill_amt_aug"]).to_numpy().reshape(-1, 1)
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.1))
    for ax, target, title in [
        (axes[0], d["pay_amt_sep"].to_numpy(), "target: raw payment"),
        (axes[1], np.log10(d["pay_amt_sep"]).to_numpy(),
         "target: log payment"),
    ]:
        X1 = np.hstack([np.ones_like(x), x])
        beta, *_ = np.linalg.lstsq(X1, target, rcond=None)
        fitted = X1 @ beta
        resid = target - fitted
        ax.scatter(fitted, resid, s=6, alpha=0.25, color=BLUE, edgecolors="none")
        ax.axhline(0, color=GRAY, linewidth=1.2)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("fitted values")
        ax.set_ylabel("residuals")
    fig.tight_layout()
    save(fig, "fig_residuals")


# ---------------------------------------------------------------------
# Logistic regression
# ---------------------------------------------------------------------
def fig_lpm_vs_logistic(df: pd.DataFrame) -> None:
    from sklearn.linear_model import LogisticRegression

    X_all = taiwan_features(df)
    sub = X_all.sample(5000, random_state=SEED)
    x = sub["utilisation"].clip(-0.15, 1.55).to_numpy()
    y = df.loc[sub.index, "default"].to_numpy()
    rng = np.random.default_rng(SEED)
    jitter = y + rng.normal(0, 0.02, len(y))

    b, a = np.polyfit(x, y, 1)
    grid = np.linspace(-0.15, 1.55, 300)
    lr = LogisticRegression().fit(x.reshape(-1, 1), y)
    proba = lr.predict_proba(grid.reshape(-1, 1))[:, 1]

    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    ax.scatter(x, jitter, s=5, alpha=0.12, color=GRAY, edgecolors="none")
    ax.plot(grid, a + b * grid, color=RED, linewidth=1.8, linestyle="--",
            label="linear fit (LPM)")
    ax.plot(grid, proba, color=BLUE, linewidth=2.2, label="logistic fit")
    ax.axhline(0, color="#B9BEC4", linewidth=0.8)
    ax.axhline(1, color="#B9BEC4", linewidth=0.8)
    ax.set_xlim(-0.15, 1.55)
    ax.set_ylim(-0.25, 1.25)
    ax.set_xlabel("credit utilisation")
    ax.set_ylabel("default (0/1) / predicted probability")
    ax.legend(frameon=False, fontsize=11, loc="lower right")
    save(fig, "fig_lpm_vs_logistic")


def fig_sigmoid() -> None:
    z = np.linspace(-6, 6, 300)
    p = 1 / (1 + np.exp(-z))
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.plot(z, p, color=BLUE, linewidth=2.2)
    ax.axhline(0.5, color=GRAY, linestyle=":", linewidth=1.2)
    ax.axvline(0, color=GRAY, linestyle=":", linewidth=1.2)
    ax.annotate("$z=0 \\Rightarrow p=0.5$", xy=(0, 0.5), xytext=(1.3, 0.36),
                fontsize=11, arrowprops=dict(arrowstyle="->", color="#5B6570"))
    ax.set_xlabel("$z = \\beta_0 + \\beta_1 x_1 + \\dots + \\beta_k x_k$")
    ax.set_ylabel("$\\sigma(z) = 1/(1+e^{-z})$")
    save(fig, "fig_sigmoid")


def fig_logit_coefs(X: pd.DataFrame, y: pd.Series) -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    Xs = StandardScaler().fit_transform(X)
    lr = LogisticRegression(max_iter=2000).fit(Xs, y)
    coefs = pd.Series(lr.coef_[0], index=X.columns).sort_values()
    colors = [RED if c < 0 else BLUE for c in coefs.values]
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    ax.barh(coefs.index, coefs.values, color=colors, height=0.6)
    ax.axvline(0, color="#5B6570", linewidth=1.0)
    ax.set_xlabel("standardised coefficient (log-odds per 1 SD)")
    ax.grid(axis="y", visible=False)
    save(fig, "fig_logit_coefs")


def _fit_test_proba(X: pd.DataFrame, y: pd.Series):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=SEED, stratify=y
    )
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    pipe.fit(X_tr, y_tr)
    return y_te.to_numpy(), pipe.predict_proba(X_te)[:, 1]


def fig_customer_contrib(X: pd.DataFrame, y: pd.Series) -> None:
    """One test customer's prediction, decomposed into per-feature log-odds
    contributions -- the exact, free version of what SHAP approximates."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X_tr, X_te, y_tr, _ = train_test_split(
        X, y, test_size=0.3, random_state=SEED, stratify=y
    )
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    pipe.fit(X_tr, y_tr)
    proba = pipe.predict_proba(X_te)[:, 1]

    # deterministic pick: the test customer whose PD is closest to 0.65
    i = int(np.argmin(np.abs(proba - 0.65)))
    scaler = pipe.named_steps["standardscaler"]
    lr = pipe.named_steps["logisticregression"]
    xs = scaler.transform(X_te.iloc[[i]])[0]
    contrib = pd.Series(lr.coef_[0] * xs, index=X.columns).sort_values()
    z = lr.intercept_[0] + contrib.sum()
    p = 1 / (1 + np.exp(-z))

    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    colors = [RED if c > 0 else BLUE for c in contrib.values]
    ax.barh(contrib.index, contrib.values, color=colors, height=0.6)
    ax.axvline(0, color="#5B6570", linewidth=1.0)
    ax.set_xlabel("contribution to log-odds (this customer)")
    ax.grid(axis="y", visible=False)
    ax.text(0.98, 0.06,
            f"intercept {lr.intercept_[0]:+.2f}   →   z = {z:+.2f}"
            f"   →   p = {p:.0%}",
            transform=ax.transAxes, ha="right", fontsize=11.5,
            color="#333333")
    save(fig, "fig_customer_contrib")
    logger.info("contrib customer: z=%.3f p=%.3f", z, p)


def fig_pred_prob_hist(X: pd.DataFrame, y: pd.Series) -> None:
    y_te, proba = _fit_test_proba(X, y)
    fig, ax = plt.subplots(figsize=(7.0, 3.3))
    bins = np.linspace(0, 1, 35)
    ax.hist(proba[y_te == 0], bins=bins, alpha=0.75, color=GRAY,
            label="repaid (0)", edgecolor="white", linewidth=0.3)
    ax.hist(proba[y_te == 1], bins=bins, alpha=0.75, color=RED,
            label="default (1)", edgecolor="white", linewidth=0.3)
    ax.axvline(0.5, color=DARK, linestyle="--", linewidth=1.5)
    ax.text(0.51, ax.get_ylim()[1] * 0.9, "threshold 0.5", color=DARK,
            fontsize=10.5)
    ax.set_xlabel("predicted probability of default (test set)")
    ax.set_ylabel("customers")
    ax.legend(frameon=False, fontsize=11)
    save(fig, "fig_pred_prob_hist")


def fig_threshold_tradeoff(X: pd.DataFrame, y: pd.Series) -> None:
    y_te, proba = _fit_test_proba(X, y)
    thresholds = np.linspace(0.05, 0.75, 50)
    prec, rec = [], []
    for t in thresholds:
        pred = proba >= t
        tp = ((pred == 1) & (y_te == 1)).sum()
        fp = ((pred == 1) & (y_te == 0)).sum()
        fn = ((pred == 0) & (y_te == 1)).sum()
        prec.append(tp / (tp + fp) if tp + fp else np.nan)
        rec.append(tp / (tp + fn))
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    ax.plot(thresholds, prec, color=BLUE, linewidth=2.0, label="precision")
    ax.plot(thresholds, rec, color=RED, linewidth=2.0, label="recall")
    ax.axvline(0.5, color=GRAY, linestyle=":", linewidth=1.2)
    ax.set_xlabel("decision threshold")
    ax.set_ylabel("metric value")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=11)
    save(fig, "fig_threshold_tradeoff")


# ---------------------------------------------------------------------
# Regularisation
# ---------------------------------------------------------------------
def _regression_problem(df: pd.DataFrame):
    d = _warmup_frame(df)
    X = pd.DataFrame(
        {
            "log_bill_aug": np.log10(d["bill_amt_aug"]),
            "log_limit": np.log10(d["limit_bal"]),
            "age": d["age"],
            "education": d["education"],
            "female": (d["sex"] == 2).astype(int),
            "married": (d["marriage"] == 1).astype(int),
            "months_delayed": (d[[c for c in d.columns
                                  if c.startswith("pay_status_")]] > 0
                               ).sum(axis=1),
        }
    )
    y = np.log10(d["pay_amt_sep"])
    return X, y


def _coef_paths(model_cls, X, y, alphas):
    from sklearn.preprocessing import StandardScaler

    Xs = StandardScaler().fit_transform(X)
    paths = []
    for a in alphas:
        m = model_cls(alpha=a, max_iter=50_000)
        m.fit(Xs, y)
        paths.append(m.coef_)
    return np.array(paths)


# Fixed, CVD-distinguishable series colors (identity follows the feature).
PATH_COLORS = {
    "log_bill_aug": "#2E6DA4",
    "log_limit": "#E8A33D",
    "age": "#7B5CA6",
    "education": "#C83C28",
    "female": "#4C956C",
    "married": "#8C959E",
    "months_delayed": "#003366",
}


def _plot_paths(paths, alphas, cols, fname):
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    for j, c in enumerate(cols):
        lw = 2.2 if c == "log_bill_aug" else 1.4
        ax.plot(alphas, paths[:, j], linewidth=lw, label=c,
                color=PATH_COLORS[c])
    ax.set_xscale("log")
    ax.axhline(0, color="#B9BEC4", linewidth=0.8)
    ax.set_xlabel(r"regularisation strength $\alpha$")
    ax.set_ylabel("standardised coefficient")
    ax.legend(fontsize=9, frameon=False, ncol=2, loc="upper right")
    save(fig, fname)


def fig_ridge_lasso_paths(df: pd.DataFrame) -> None:
    from sklearn.linear_model import Lasso, Ridge

    X, y = _regression_problem(df)
    alphas_r = np.logspace(-2, 4, 40)
    alphas_l = np.logspace(-4, 0, 40)
    _plot_paths(_coef_paths(Ridge, X, y, alphas_r), alphas_r, X.columns,
                "fig_ridge_paths")
    _plot_paths(_coef_paths(Lasso, X, y, alphas_l), alphas_l, X.columns,
                "fig_lasso_paths")


def fig_interactions() -> None:
    """Illustrative: additive effects (parallel) vs an interaction."""
    x = np.linspace(0, 1, 50)
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.0), sharey=True)

    axes[0].plot(x, 12 + 25 * x, color=BLUE, linewidth=2.0, label="never delayed")
    axes[0].plot(x, 22 + 25 * x, color=RED, linewidth=2.0, label="delayed history")
    axes[0].set_title("no interaction: one slope,\nshifted intercepts",
                      fontsize=11)

    axes[1].plot(x, 12 + 18 * x, color=BLUE, linewidth=2.0, label="never delayed")
    axes[1].plot(x, 16 + 42 * x, color=RED, linewidth=2.0, label="delayed history")
    axes[1].set_title("interaction: utilisation bites\nharder for renters",
                      fontsize=11)

    for ax in axes:
        ax.set_xlabel("credit utilisation")
        ax.legend(frameon=False, fontsize=10, loc="upper left")
    axes[0].set_ylabel("default rate (%)")
    fig.suptitle("Illustrative data", fontsize=9, x=0.98, ha="right",
                 color="#8C959E")
    fig.tight_layout()
    save(fig, "fig_interactions")


def fig_l1_l2_geometry() -> None:
    """Classic constraint-region picture: why L1 produces exact zeros."""
    A = np.array([[1.0, 0.35], [0.35, 0.55]])
    b_star = np.array([1.7, 0.4])

    def loss(b1, b2):
        d1, d2 = b1 - b_star[0], b2 - b_star[1]
        return A[0, 0] * d1**2 + 2 * A[0, 1] * d1 * d2 + A[1, 1] * d2**2

    r = 0.8
    theta = np.linspace(0, 2 * np.pi, 2000)

    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.6), sharey=True)
    grid = np.linspace(-1.6, 2.9, 300)
    B1, B2 = np.meshgrid(grid, grid)
    Z = loss(B1, B2)

    # L2 ball: contact point on the circle
    circ = np.stack([r * np.cos(theta), r * np.sin(theta)])
    i2 = np.argmin(loss(circ[0], circ[1]))
    p2 = circ[:, i2]

    # L1 ball: contact point on the diamond
    t = np.linspace(0, 1, 500)
    edges = []
    for s1, s2 in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
        edges.append(np.stack([s1 * r * t, s2 * r * (1 - t)]))
    diam = np.concatenate(edges, axis=1)
    i1 = np.argmin(loss(diam[0], diam[1]))
    p1 = diam[:, i1]

    for ax, region, p, name in [
        (axes[0], circ, p2, "L2 (ridge): $\\beta_1^2+\\beta_2^2 \\leq t$"),
        (axes[1], diam, p1, "L1 (lasso): $|\\beta_1|+|\\beta_2| \\leq t$"),
    ]:
        ax.contour(B1, B2, Z, levels=sorted(
            [loss(*p) * f for f in (0.35, 1.0, 2.2, 4.0)]),
            colors="#9AA5B1", linewidths=0.9)
        ax.fill(region[0], region[1], color=LIGHT, alpha=0.9, zorder=2)
        ax.plot(region[0], region[1], color=DARK, linewidth=1.4, zorder=3)
        ax.plot(*b_star, marker="x", color="#5B6570", markersize=8, zorder=4)
        ax.annotate("OLS $\\hat\\beta$", b_star, xytext=(b_star[0] - 0.15,
                    b_star[1] + 0.28), fontsize=10, color="#5B6570")
        ax.plot(*p, marker="o", color=RED, markersize=7, zorder=5)
        ax.axhline(0, color="#C4CAD1", linewidth=0.8)
        ax.axvline(0, color="#C4CAD1", linewidth=0.8)
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("$\\beta_1$")
        ax.set_aspect("equal")
        ax.grid(visible=False)
        ax.set_xlim(-1.5, 2.8)
        ax.set_ylim(-1.2, 1.7)
    axes[0].set_ylabel("$\\beta_2$")
    fig.tight_layout()
    save(fig, "fig_l1_l2_geometry")


def fig_validation_curve(df: pd.DataFrame) -> None:
    """Train/validation error vs alpha with a genuine interior optimum.

    A high-variance setting on purpose: few training rows, polynomially
    expanded features -- so that weak regularisation visibly overfits and the
    validation curve has a real U shape.
    """
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler

    X, y = _regression_problem(df)
    Xp = PolynomialFeatures(degree=2, include_bias=False).fit_transform(X)
    X_tr, X_te, y_tr, y_te = train_test_split(
        Xp, y, train_size=80, random_state=SEED
    )
    sc = StandardScaler().fit(X_tr)
    alphas = np.logspace(-4, 4, 60)
    tr_err, te_err = [], []
    for a in alphas:
        m = Ridge(alpha=a).fit(sc.transform(X_tr), y_tr)
        tr_err.append(np.mean((m.predict(sc.transform(X_tr)) - y_tr) ** 2))
        te_err.append(np.mean((m.predict(sc.transform(X_te)) - y_te) ** 2))
    fig, ax = plt.subplots(figsize=(6.8, 3.3))
    ax.plot(alphas, tr_err, color=BLUE, linewidth=2.0, label="training error")
    ax.plot(alphas, te_err, color=RED, linewidth=2.0, label="validation error")
    i_best = int(np.argmin(te_err))
    best = alphas[i_best]
    ax.axvline(best, color=GRAY, linestyle=":", linewidth=1.3)
    ax.text(best * 1.5, te_err[i_best] * 1.6,
            f"best $\\alpha \\approx {best:.2g}$", fontsize=10.5,
            color="#5B6570")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"regularisation strength $\alpha$")
    ax.set_ylabel("mean squared error")
    ax.legend(frameon=False, fontsize=11, loc="upper left")
    save(fig, "fig_validation_curve")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    FIG_DIR.mkdir(exist_ok=True)
    X, y, df = prepare_data()
    fig_overfitting()
    fig_bias_variance()
    fig_linear_fit(df)
    fig_residuals(df)
    fig_lpm_vs_logistic(df)
    fig_sigmoid()
    fig_logit_coefs(X, y)
    fig_customer_contrib(X, y)
    fig_pred_prob_hist(X, y)
    fig_threshold_tradeoff(X, y)
    fig_ridge_lasso_paths(df)
    fig_interactions()
    fig_l1_l2_geometry()
    fig_validation_curve(df)
    logger.info("All figures written to %s", FIG_DIR)


if __name__ == "__main__":
    main()
