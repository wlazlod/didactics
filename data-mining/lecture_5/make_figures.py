"""Generate all figures for the Block 5 slide deck (pattern discovery &
text mining).

Usage (from the data-mining/ directory):
    uv run python lecture_5/make_figures.py

Outputs PDF figures into lecture_5/figures/. Deterministic. Also logs the
top association rules and classifier metrics used on the slides.
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
from finance_data import load_baskets, load_complaints  # noqa: E402

logger = logging.getLogger(__name__)

FIG_DIR = Path(__file__).parent / "figures"
SEED = 42

BLUE = "#2E6DA4"
DARK = "#003366"
RED = "#C83C28"
GRAY = "#8C959E"
LIGHT = "#D7E3F0"
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
# Baskets
# ---------------------------------------------------------------------
def fig_basket_sample(baskets: pd.DataFrame) -> None:
    baskets = baskets[baskets.mean().sort_values(ascending=False).index]
    sample = baskets.sample(40, random_state=SEED)
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    ax.imshow(sample.T, aspect="auto", cmap="Blues", vmin=-0.35, vmax=1.35,
              interpolation="nearest")
    ax.set_yticks(range(baskets.shape[1]))
    ax.set_yticklabels(baskets.columns, fontsize=10)
    ax.set_xlabel("customers (sample of 40)")
    ax.set_title("dark = customer holds the product", fontsize=11)
    ax.grid(visible=False)
    save(fig, "fig_basket_sample")


def fig_penetration(baskets: pd.DataFrame) -> None:
    pen = baskets.mean().sort_values()
    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    bars = ax.barh(pen.index, pen.values, color=BLUE, height=0.6)
    for b, v in zip(bars, pen.values):
        ax.text(v + 0.01, b.get_y() + b.get_height() / 2, f"{v:.0%}",
                va="center", fontsize=10)
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("share of customers holding the product (support)")
    ax.grid(axis="y", visible=False)
    save(fig, "fig_penetration")


def _mine_rules(baskets: pd.DataFrame):
    from mlxtend.frequent_patterns import apriori, association_rules

    freq = apriori(baskets, min_support=0.03, use_colnames=True)
    rules = association_rules(freq, metric="confidence", min_threshold=0.3)
    rules["ante"] = rules["antecedents"].apply(lambda s: " + ".join(sorted(s)))
    rules["cons"] = rules["consequents"].apply(lambda s: " + ".join(sorted(s)))
    rules["label"] = rules["ante"] + " → " + rules["cons"]
    return rules


def fig_rules_scatter(rules: pd.DataFrame) -> None:
    r = rules[(rules["antecedents"].apply(len) == 1)
              & (rules["consequents"].apply(len) == 1)]
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    sc = ax.scatter(r["support"], r["confidence"], c=r["lift"], cmap="RdBu_r",
                    vmin=0, vmax=4, s=42, edgecolors="#5B6570", linewidths=0.4)
    top = r.nlargest(2, "lift")
    for _, row in top.iterrows():
        ax.annotate(row["label"], (row["support"], row["confidence"]),
                    xytext=(8, -12), textcoords="offset points", fontsize=9.5)
    cb = fig.colorbar(sc)
    cb.set_label("lift", labelpad=10)
    ax.set_xlabel("support")
    ax.set_ylabel("confidence")
    save(fig, "fig_rules_scatter")


def fig_top_rules(rules: pd.DataFrame) -> None:
    r = rules[(rules["antecedents"].apply(len) <= 2)
              & (rules["consequents"].apply(len) == 1)]
    # drop mirror duplicates, keep highest-lift ten with support >= 5%
    r = r[r["support"] >= 0.05].nlargest(10, "lift")
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    bars = ax.barh(r["label"][::-1], r["lift"][::-1], color=BLUE, height=0.62)
    ax.axvline(1.0, color=RED, linestyle="--", linewidth=1.6, zorder=5)
    ax.text(1.0, 1.01, "lift = 1: independence", color=RED, fontsize=10,
            ha="center", transform=ax.get_xaxis_transform())
    ax.set_xlabel("lift")
    ax.tick_params(axis="y", labelsize=10)
    ax.grid(axis="y", visible=False)
    save(fig, "fig_top_rules")
    logger.info("top rules:\n%s",
                r[["label", "support", "confidence", "lift"]]
                .round(3).to_string(index=False))


# ---------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------
def fig_zipf(complaints: pd.DataFrame) -> None:
    from sklearn.feature_extraction.text import CountVectorizer

    cv = CountVectorizer()
    X = cv.fit_transform(complaints["text"])
    counts = np.asarray(X.sum(0)).ravel()
    ranked = np.sort(counts)[::-1]
    fig, ax = plt.subplots(figsize=(6.2, 3.3))
    ax.loglog(np.arange(1, len(ranked) + 1), ranked, color=BLUE, linewidth=1.8)
    words = np.array(cv.get_feature_names_out())
    order = np.argsort(counts)[::-1]
    for rank in [0, 9, 99]:
        ax.annotate(words[order[rank]], (rank + 1, ranked[rank]),
                    xytext=(8, 8), textcoords="offset points", fontsize=10)
    ax.set_xlabel("word rank (log)")
    ax.set_ylabel("frequency (log)")
    save(fig, "fig_zipf")


def fig_idf_words(complaints: pd.DataFrame) -> None:
    from sklearn.feature_extraction.text import TfidfVectorizer

    tv = TfidfVectorizer()
    tv.fit(complaints["text"])
    words = ["account", "bank", "card", "mortgage", "fee", "collector",
             "skimmed", "casino"]
    vocab = {w: i for i, w in enumerate(tv.get_feature_names_out())}
    present = [w for w in words if w in vocab]
    idf = [tv.idf_[vocab[w]] for w in present]
    order = np.argsort(idf)
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    ax.barh(np.array(present)[order], np.array(idf)[order], color=BLUE,
            height=0.6)
    ax.set_xlabel("idf (rarity weight)")
    ax.grid(axis="y", visible=False)
    save(fig, "fig_idf_words")


def _text_model(complaints: pd.DataFrame):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline

    X_tr, X_te, y_tr, y_te = train_test_split(
        complaints["text"], complaints["category"], test_size=0.3,
        random_state=SEED, stratify=complaints["category"])
    pipe = make_pipeline(TfidfVectorizer(),
                         LogisticRegression(max_iter=2000))
    pipe.fit(X_tr, y_tr)
    return pipe, X_te, y_te


def fig_confusion_text(pipe, X_te, y_te) -> None:
    from sklearn.metrics import accuracy_score, confusion_matrix

    pred = pipe.predict(X_te)
    cats = sorted(y_te.unique())
    cm = confusion_matrix(y_te, pred, labels=cats)
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    ax.imshow(cm, cmap="Blues", vmin=0)
    ax.set_xticks(range(len(cats))); ax.set_xticklabels(cats, rotation=30,
                                                        ha="right", fontsize=10)
    ax.set_yticks(range(len(cats))); ax.set_yticklabels(cats, fontsize=10)
    for i in range(len(cats)):
        for j in range(len(cats)):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14,
                    fontweight="bold" if i != j and cm[i, j] else "normal",
                    color="white" if cm[i, j] > cm.max() * 0.6
                    else (RED if i != j and cm[i, j] else "#333333"))
    ax.set_xlabel("predicted"); ax.set_ylabel("actual")
    ax.grid(visible=False)
    save(fig, "fig_confusion_text")
    logger.info("text clf test accuracy: %.3f", accuracy_score(y_te, pred))


def fig_class_words(complaints: pd.DataFrame) -> None:
    """Display model with English stopwords removed, so the chart shows
    content words (the routing model itself keeps everything)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline

    X_tr, _, y_tr, _ = train_test_split(
        complaints["text"], complaints["category"], test_size=0.3,
        random_state=SEED, stratify=complaints["category"])
    pipe = make_pipeline(TfidfVectorizer(stop_words="english"),
                         LogisticRegression(max_iter=2000))
    pipe.fit(X_tr, y_tr)
    tv = pipe.named_steps["tfidfvectorizer"]
    lr = pipe.named_steps["logisticregression"]
    words = np.array(tv.get_feature_names_out())
    xmax = lr.coef_.max() * 1.12
    fig, axes = plt.subplots(1, 5, figsize=(10.4, 3.0))
    for ax, cat, coefs in zip(axes, lr.classes_, lr.coef_):
        top = np.argsort(coefs)[-5:]
        ax.barh(words[top], coefs[top], color=BLUE, height=0.6)
        ax.set_xlim(0, xmax)
        ax.set_title(cat, fontsize=11)
        ax.tick_params(axis="y", labelsize=10.5)
        ax.tick_params(axis="x", labelsize=8.5)
        ax.grid(axis="y", visible=False)
        logger.info("top words %s: %s", cat, list(words[top][::-1]))
    fig.suptitle("strongest words per category (stopwords removed for display)",
                 fontsize=11, y=1.04)
    fig.tight_layout()
    save(fig, "fig_class_words")


def fig_ngram_effect(complaints: pd.DataFrame) -> None:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import make_pipeline

    settings = {
        "unigrams": dict(ngram_range=(1, 1)),
        "uni+bigrams": dict(ngram_range=(1, 2)),
        "uni+bi, min_df=3": dict(ngram_range=(1, 2), min_df=3),
    }
    accs, dims = [], []
    for kw in settings.values():
        tv = TfidfVectorizer(**kw)
        Xv = tv.fit_transform(complaints["text"])
        dims.append(Xv.shape[1])
        pipe = make_pipeline(TfidfVectorizer(**kw),
                             LogisticRegression(max_iter=2000))
        accs.append(cross_val_score(pipe, complaints["text"],
                                    complaints["category"], cv=5).mean())
    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    bars = ax.bar(list(settings), accs, color=BLUE, width=0.5)
    for b, a, d in zip(bars, accs, dims):
        ax.text(b.get_x() + b.get_width() / 2, a + 0.004,
                f"{a:.3f}\n({d:,} feats)", ha="center", fontsize=9.5)
    ax.set_ylim(0.8, 1.0)
    ax.set_ylabel("CV accuracy")
    ax.grid(axis="x", visible=False)
    save(fig, "fig_ngram_effect")
    logger.info("ngram accs: %s dims: %s",
                [round(a, 3) for a in accs], dims)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    FIG_DIR.mkdir(exist_ok=True)
    baskets = load_baskets()
    complaints = load_complaints()
    fig_basket_sample(baskets)
    fig_penetration(baskets)
    rules = _mine_rules(baskets)
    fig_rules_scatter(rules)
    fig_top_rules(rules)
    fig_zipf(complaints)
    fig_idf_words(complaints)
    pipe, X_te, y_te = _text_model(complaints)
    fig_confusion_text(pipe, X_te, y_te)
    fig_class_words(complaints)
    fig_ngram_effect(complaints)
    logger.info("All figures written to %s", FIG_DIR)


if __name__ == "__main__":
    main()
