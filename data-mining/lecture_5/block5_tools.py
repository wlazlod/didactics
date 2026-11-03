"""block5_tools.py - the experiments of Block 5 (association rules and text mining).

Data Mining, SGH Warsaw School of Economics (223121-0090).

The Block 5 notebook imports this file as

    import block5_tools as lab

Each experiment is ONE function with ONE control (the value you change), and
it draws ONE figure. The slides are drawn by the same functions, so the deck
shows exactly the pictures you make in the notebook. You do not need to read
this file; the notebook shows the few mlxtend and scikit-learn lines behind
each experiment ("The code behind it").

Everything is deterministic. Rules: one or two products on the left, one on
the right, a minimum support of 3% unless you change it (mlxtend's ``apriori``
and ``association_rules``). Texts: the course split of Block 2 (70% training /
30% test, stratified, ``random_state=42``: 840 / 360 complaints),
``TfidfVectorizer()`` with its defaults, ``LogisticRegression(max_iter=2000)``,
out-of-fold probabilities with ``StratifiedKFold(5, shuffle=True,
random_state=42)``, the confidence bar 0.60. The vectoriser and the model are
fitted on the training texts only; the test texts are read in step 4 and once
in step 5. Results are computed on first use and kept in memory until the
kernel restarts; every one takes well under a second.

Four worlds, never mixed in one picture:
    T  ten made-up customers 1-10 x four products (TOY_BASKETS)
    S  six short made-up complaints S1-S6 (SENTENCES; TOY_TOP: made-up top
       probabilities for them)
    R  our 4,000 made-up customers x ten products (baskets(), load_baskets())
    C  our 1,200 made-up complaints in five topics (complaints(), load_complaints())

In the notebook (each slider cell has a plain twin: set the value, re-run)
---------------------------------------------------------------------------
    lab.setup_check()      "Setup OK ...", a version note if needed, "Sliders OK"
    D1  lab.rules_by_hand(); lab.toy_shuffled(0)
    D2  lab.support_bar(min_support=0.03)
    E1  lab.confidence_against_everyone(right_side="debit_card")
        lab.slider(lab.confidence_against_everyone, right_side=lab.E1_PRODUCTS)
    D3  lab.rule_table(); lab.customers_to_approach("mortgage", "home_insurance")
    E2  lab.shuffle_baskets(shuffle="none", min_support=0.03)
        lab.slider(lab.shuffle_baskets, shuffle=lab.E2_SHUFFLES)
        done early: lab.shuffle_baskets(shuffle, min_support=0.01)
        lab.SENTENCES; lab.toy_bag_of_words(); lab.toy_idf(); lab.weights_table(2)
    E3  lab.spread_word(in_sentences=2);   lab.slider(lab.spread_word, in_sentences=lab.E3_COUNTS)
    D4  lab.idf_ours(); lab.text_row(lab.EXAMPLE_TEXT)
    D5  lab.classifier_report()
    D6  lab.rare_topic("fees")
    E4  lab.confidence_bar(bar=0.40);      lab.slider(lab.confidence_bar, bar=lab.E4_BARS)
        lab.test_check(0.60)
    E5  lab.new_topic(left_out="card_fraud"); lab.slider(lab.new_topic, left_out=lab.E5_TOPICS)
    D7  lab.watch_stream(share_new=0.10)
    Homework A: lab.check("A1", penetration=[...], ...)   (lab.check("A1") lists what it takes)
    Homework B: train, test = lab.split(); lab.oof_proba(); lab.copy_in_other_fold()

Public API
----------
Setup and sliders (as in Blocks 3-4)
    setup_check()                loads the data; prints "Setup OK: ...", the version note
                                 (if any) and sliders_status().
    sliders_ok() -> bool; sliders_status() -> str; version_note() -> str ("" for
                                 scikit-learn 1.8 or newer and mlxtend 0.23 or newer; else
                                 a calm Note, or a WARNING below scikit-learn 1.4)
    slider(func, **controls)     ipywidgets.interact (redraws when you let go). (lo, hi, step)
                                 -> slider; a list of numbers -> slider over them; a list of
                                 strings, or of strings and numbers -> dropdown. Without
                                 ipywidgets: a line saying to use the plain cell.
    VERBOSE = True               kept for make_figures.py (nothing here is slow enough to
                                 print a "Computing ..." line).
    clear_cache()                forget cached results and the trails of Experiments 2 and 4.
    SEED = 42; colour tokens BLUE RED DARK GRAY LIGHT LIGHT_RED LIGHT_BLUE MID_BLUE MUTED
    SHADE and NEAR_BLACK (#222222, the text colour; Block 4 has the same value under another
    name); STYLE; PRODUCT_LABELS, TOPIC_LABELS (plain names for figures).

Data
    PRODUCTS                     the ten products by penetration: checking, debit_card,
                                 savings, credit_card, personal_loan, home_insurance,
                                 term_deposit, mortgage, car_loan, brokerage
    baskets()                    load_baskets() with its columns in that order (a copy)
    TOY_BASKETS                  10 x 4 True/False, index customer 1-10 (checking, savings,
                                 mortgage, home_insurance)
    toy_shuffled(seed=0)         TOY_BASKETS with each column shuffled on its own
    SENTENCES                    the six sentences S1-S6 (a list)
    SPREAD_ORDER = [4, 5, 3, 1, 6]   the sentences that get " twice" in Experiment 3, in order
    TOY_TOP                      made-up top probabilities of S1-S6 and right / wrong (slide 53)
    complaints()                 load_complaints() (a copy): text, category
    TOPICS                       the five topics, alphabetical (scikit-learn's classes_)
    split() -> train, test       70/30, stratified, random_state=42 (copies; columns text,
                                 category; 840 and 360 rows)
    EXAMPLE_TEXT                 the mortgage text of slides 43 and 48 (a test text with no
                                 copy among the training texts)
    MIXED_TEXT                   the text of slide 51 (a test text; collections, labelled
                                 mortgage)

Rules (our data or any True/False table)
    rules(table, min_support=0.03) -> DataFrame   rule, support, confidence, lift,
                                 customers; one or two products on the left, one on the
                                 right; the order of the key below. Our table and the
                                 shuffles are cached per minimum support.
    levels(table, min_support) -> DataFrame       per level (products in an itemset):
                                 all, counted (by Apriori), kept
    toy_counts(); toy_levels(bar=3); rules_by_hand(); rule_table(min_support=0.03);
    customers_to_approach(a, b) -> the rows of the customers with a but not b

Texts
    toy_bag_of_words(); toy_idf(); weights_table(sentence); idf_ours(); text_row(text);
    classifier_report(); rare_topic(topic="fees"); test_check(bar=0.60);
    oof_proba() -> DataFrame (840 x 5, cached); copy_in_other_fold() -> Series of True /
    False (the training text has a copy in another fold); watch_stream(share_new=0.10)

Target-free orders (the same on every run)
    products: by penetration (PRODUCTS, a fixed list); topics: alphabetical (TOPICS).
    rules: by the key (-round(lift, 6), -round(support, 6), rule text), where the rule
           text is "A => B" with column names and A's products joined by " + " in
           alphabetical order (rounding makes exact ties such as A => B and B => A
           deterministic). The "best rule" of a table = its first rule by this key.
    Experiment 1's bars: by lift, then A's name. Words: by weight, then alphabet.
    shuffles: np.random.default_rng(shuffle), one permutation per column in the
           fixed product order.

Experiments (X(control) shows the figure and returns None; X_figure(control, ...,
figsize=None, suptitle=True) -> Figure; draw_*(ax, ...) single panels; X_numbers())
    E1 confidence_against_everyone(right_side="debit_card")   E1_PRODUCTS = PRODUCTS
       draw_confidence_bars(ax, right_side); draw_lift_bars(ax, right_side);
       confidence_against_everyone_figure(right_side); confidence_numbers()
    E2 shuffle_baskets(shuffle="none", min_support=0.03)   E2_SHUFFLES = ["none"] + 1-20;
       keeps a trail of the tables tried (shuffle_baskets_reset()).
       draw_rule_cloud(ax, shuffle, min_support=0.03); draw_best_lifts(ax, tried,
       current=None, min_support=0.03, ours_grey=None);
       shuffle_baskets_figure(shuffle, min_support, tried=None);
       shuffle_numbers(min_support=0.03)
    E3 spread_word(in_sentences=2)   E3_COUNTS = [1, ..., 6]
       draw_idf_curve(ax, in_sentences); draw_s2_weights(ax, in_sentences, title=None);
       spread_word_figure(in_sentences); spread_numbers()
    E4 confidence_bar(bar=0.40)      E4_BARS = (0.30, 0.90, 0.05); E4_BAR_LIST (the 13
       values); the bar is rounded to 2 decimals first; trail (confidence_bar_reset()).
       draw_top_proba(ax, bar); draw_bar_tradeoff(ax, bar, tried);
       confidence_bar_figure(bar, tried=None); bar_numbers()
    E5 new_topic(left_out="card_fraud")   E5_TOPICS = TOPICS
       draw_new_topic_labels(ax, left_out); draw_new_topic_proba(ax, left_out);
       new_topic_figure(left_out); new_topic_numbers()

Demos (replay at home)
    D1 rules_by_hand() + toy_shuffled()   D2 support_bar(min_support=0.03)
    D3 rule_table() + customers_to_approach()   D4 idf_ours() + text_row()
    D5 classifier_report()   D6 rare_topic()   D7 watch_stream()

Slide drawing helpers (make_figures.py; each returns the numbers it shows)
    draw_toy_grid(ax, table=None, frame=None, ring=None, shade=None),
    draw_toy_levels(ax, stage "a" | "b"), draw_compare_bars(ax, a, b, table=None),
    draw_penetration(ax), draw_levels(ax, table, min_support), draw_idf_words(ax),
    draw_text_row(ax, text), draw_five_probs(ax, text), draw_confusion(ax),
    draw_rare_topic(ax, topic), draw_watch(ax, current=None), draw_zipf(ax)

Homework A (by hand)
    HW_CLUB (members 1-8), HW_REVIEWS (R1-R4), HW_CONFUSION, HW_TOP
    check(item, **answers)       soft check: says right / not yet with a hint; never
                                 raises; check(item) lists the answers it takes
    homework_answers(item) -> dict   the worked numbers (for the solutions notebook)

All figure text is at least 8.5 pt; every '$' in a figure string is written '\\$'
(matplotlib would read a pair of them as mathematics).
"""
from __future__ import annotations

import math
import re
import sys
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, Rectangle

try:  # Colab: finance_data.py uploaded next to the notebook
    import finance_data as _finance_data
except ImportError:  # course folder: the loader lives in ../lecture_1
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lecture_1"))
    import finance_data as _finance_data
from finance_data import load_baskets, load_complaints

SEED = 42
VERBOSE = True
MIN_SUPPORT = 0.03
BAR = 0.60

# ---------------------------------------------------------------------------
# Colours and style (the course palette, as in Blocks 2-4)
# ---------------------------------------------------------------------------
BLUE = "#2E6DA4"
RED = "#C83C28"
DARK = "#003366"
GRAY = "#8C959E"
LIGHT = "#D7E3F0"
LIGHT_RED = "#F2C9C2"
LIGHT_BLUE = "#BFD3E8"
MID_BLUE = "#7FA7CF"
NEAR_BLACK = "#222222"
MUTED = "#5B6570"
SHADE = "#EEF0F3"

STYLE = {
    "font.size": 9.5,
    "axes.titlesize": 9.5,
    "axes.labelsize": 9.5,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#6B7580",
    "axes.labelcolor": NEAR_BLACK,
    "xtick.color": "#6B7580",
    "ytick.color": "#6B7580",
    "xtick.labelcolor": NEAR_BLACK,
    "ytick.labelcolor": NEAR_BLACK,
    "text.color": NEAR_BLACK,
    "axes.grid": False,
    "axes.unicode_minus": False,
    "legend.frameon": False,
    "figure.constrained_layout.h_pad": 0.03,
    "figure.constrained_layout.w_pad": 0.03,
    "figure.constrained_layout.hspace": 0.03,
    "figure.constrained_layout.wspace": 0.03,
}

PRODUCTS = ["checking", "debit_card", "savings", "credit_card", "personal_loan",
            "home_insurance", "term_deposit", "mortgage", "car_loan", "brokerage"]
PRODUCT_LABELS = {p: p.replace("_", " ") for p in PRODUCTS}
TOPICS = ["app_service", "card_fraud", "collections", "fees", "mortgage"]
TOPIC_LABELS = {"app_service": "app & service", "card_fraud": "card fraud",
                "collections": "collections", "fees": "fees", "mortgage": "mortgage"}
ARROW = "⇒"   # the arrow of a rule in figures: A => B


def _style():
    return plt.rc_context(STYLE)


def _new_fig(figsize, nrows=1, ncols=1, **kw):
    return plt.subplots(nrows, ncols, figsize=figsize, layout="constrained", **kw)


def _show(fig) -> None:
    """Show a figure in the notebook (inline backend) and free it. Under a
    non-interactive backend (a script, Agg) there is nothing to show."""
    if plt.get_backend().lower() not in ("agg", "pdf", "svg", "ps", "cairo"):
        plt.show()
    plt.close(fig)


def _n(v) -> str:
    """A whole number with thousands separators: 1918 -> 1,918."""
    return f"{int(round(float(v))):,}"


def _two(v) -> str:
    """Two decimals without a negative zero."""
    return f"{round(float(v), 2) + 0.0:.2f}"


def _pct(v, d: int = 1) -> str:
    """A share (0-1) as a percent: 0.8475 -> 84.7%."""
    return f"{100 * float(v):.{d}f}%"


def _label(name: str) -> str:
    """The plain name of a product or topic (for figures)."""
    return PRODUCT_LABELS.get(name, TOPIC_LABELS.get(name, str(name).replace("_", " ")))


def _rule_label(left, right) -> str:
    """A rule in plain words for figures: mortgage => home insurance."""
    left = [left] if isinstance(left, str) else list(left)
    return " + ".join(_label(a) for a in sorted(left)) + f" {ARROW} " + _label(right)


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------
_CACHE: dict = {}


def _cached(key, fn):
    """fn() once per session (every result here takes well under a second)."""
    if key not in _CACHE:
        _CACHE[key] = fn()
    return _CACHE[key]


def clear_cache() -> None:
    """Forget every cached result and the trails of Experiments 2 and 4."""
    _CACHE.clear()
    _E2_TRAIL.clear()
    _E4_TRAIL.clear()


# ---------------------------------------------------------------------------
# Sliders and the setup check (as in Blocks 3-4)
# ---------------------------------------------------------------------------
def sliders_ok() -> bool:
    """True if ipywidgets is installed and we run inside a Jupyter kernel."""
    try:
        import ipywidgets  # noqa: F401
        from IPython import get_ipython
        ip = get_ipython()
        return ip is not None and "IPKernelApp" in ip.config
    except Exception:
        return False


def sliders_status() -> str:
    if sliders_ok():
        return "Sliders OK"
    return ("No sliders here (ipywidgets is not available) - that is fine: above each "
            "slider cell there is a plain cell; change the value in it and re-run it.")


def _version(text: str) -> tuple:
    found = re.match(r"(\d+)\.(\d+)", str(text))
    return tuple(int(n) for n in found.groups()) if found else (0, 0)


def _mlxtend_version():
    """mlxtend's version as text, or None if it is not installed."""
    try:
        import mlxtend
        return str(getattr(mlxtend, "__version__", "0.0"))
    except Exception:
        return None


def version_note() -> str:
    """The lines about the package versions, for the setup cell ("" if
    nothing needs saying). The notebook was tested with scikit-learn 1.9 and
    mlxtend 0.25."""
    import sklearn
    notes = []
    sk = _version(sklearn.__version__)
    if sk < (1, 4):
        return (f"WARNING: scikit-learn {sklearn.__version__} is older than 1.4, the oldest "
                "version this notebook works with.\nFix: run  %pip install -q -U scikit-learn  "
                "in a new cell, then restart\n(JupyterLab: Kernel -> Restart Kernel; Colab: "
                "Runtime -> Restart session)\nand run the setup cell again.")
    if sk < (1, 8):
        notes.append(f"Note: scikit-learn {sklearn.__version__} is older than 1.8, so a few "
                     "numbers may differ from the slides in the third decimal - that is fine, "
                     "nothing to do.")
    ml = _mlxtend_version()
    if ml is None:
        notes.append("Note: the package mlxtend is not installed. The experiments still work "
                     "(this file then counts the rules itself, with the same numbers); for "
                     "'the code behind it' run  %pip install -q -U mlxtend  in a new cell, "
                     "then restart the kernel.")
    elif _version(ml) < (0, 23):
        notes.append(f"Note: mlxtend {ml} is older than 0.23. The experiments still work; "
                     "if 'the code behind it' stops with an error, run  "
                     "%pip install -q -U mlxtend  in a new cell, then restart the kernel.")
    return "\n".join(notes)


def setup_check() -> None:
    """The setup cell's lines: loads the data, then prints "Setup OK: ...",
    the version note if there is one, and the sliders status."""
    import sklearn
    b, c = baskets(), complaints()
    note = version_note()
    if note.startswith("WARNING"):
        print(note)
        print("Setup OK - but see the WARNING above")
    else:
        print(f"Setup OK: {len(b):,} customers × {b.shape[1]} products; {len(c):,} "
              f"complaints in {c['category'].nunique()} topics - scikit-learn "
              f"{sklearn.__version__}")
        if note:
            print(note)
    print(sliders_status())


def _say(msg: str) -> None:
    """A one-line note to the student (never an error)."""
    print(msg, flush=True)


def _plain(value):
    return value.item() if isinstance(value, np.generic) else value


def _decimal_comma(value, what: str):
    """Python reads 0,3 as a pair of numbers, (0, 3). Such a pair becomes the
    decimal it was meant to be (0.3), with a note; any other value is unchanged."""
    if (isinstance(value, tuple) and len(value) == 2
            and all(isinstance(v, (int, np.integer)) and not isinstance(v, bool) for v in value)
            and value[1] >= 0):
        a, b = int(value[0]), int(value[1])
        meant = float(f"{a}.{b}")
        _say(f"Python writes decimals with a dot: {a}.{b}, not {a},{b} - showing "
             f"{what} = {meant:g}.")
        return meant
    return value


def _as_number(value):
    """value as a number if it is one (or a string that reads as one); anything
    else unchanged."""
    value = _plain(value)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, str):
        try:
            return float(value.strip().rstrip("%"))
        except ValueError:
            return value
    return value


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and np.isfinite(v)


def _key(v) -> str:
    return "".join(ch for ch in str(v).lower() if ch.isalnum())


def _choice(value, choices: list, what: str, aliases: dict | None = None) -> str:
    """One of `choices` for what was typed (case, spaces and signs do not
    matter); anything else -> the first choice, with a note."""
    value = _plain(value)
    if value in choices:
        return value
    for c in choices:
        if _key(c) == _key(value):
            return c
    for k, c in (aliases or {}).items():
        if _key(value) == k:
            return c
    _say(f"{what} {value!r} is not one of " + ", ".join(f'"{c}"' for c in choices)
         + f': showing "{choices[0]}".')
    return choices[0]


def _whole(value, lo: int, hi: int, default: int, what: str) -> int:
    """value as a whole number from lo to hi; anything else -> the nearest
    allowed one (or default), with a note."""
    value = _decimal_comma(_plain(value), what)
    v = _as_number(value)
    if _is_number(v) and float(v).is_integer() and lo <= v <= hi:
        return int(v)
    new = int(min(max(round(v), lo), hi)) if _is_number(v) else default
    shown = f"{value:g}" if _is_number(value) else repr(value)
    _say(f"{what} {shown} is not a whole number from {lo} to {hi}: showing {what} = {new}.")
    return new


def _share(value, lo: float, hi: float, default: float, what: str) -> float:
    """value as a share from lo to hi (a percent such as 3 or "3%" is read as
    0.03 when that fits); anything else -> the nearest allowed one, with a note."""
    value = _decimal_comma(_plain(value), what)
    v = _as_number(value)
    if not _is_number(v):
        _say(f"{what} {value!r} is not a number: showing {what} = {default:g}.")
        return default
    v = float(v)
    if v >= 1 and lo <= v / 100 <= hi:
        _say(f"{what} {v:g} read as {v:g}% = {v / 100:g}.")
        v = v / 100
    if not lo <= v <= hi:
        new = min(max(v, lo), hi)
        _say(f"{what} {v:g} is outside {lo:g}-{hi:g}: showing {what} = {new:g}.")
        v = new
    return v


def _widget(name, spec, default):
    import ipywidgets as w
    common = dict(description=name, continuous_update=False,
                  style={"description_width": "initial"}, layout=w.Layout(width="480px"))
    if isinstance(spec, tuple) and len(spec) in (2, 3) and all(
            isinstance(v, (int, float)) for v in spec):
        lo, hi = spec[0], spec[1]
        is_int = all(isinstance(v, int) for v in spec)
        step = spec[2] if len(spec) == 3 else (1 if is_int else (hi - lo) / 20)
        value = default if isinstance(default, (int, float)) and lo <= default <= hi else lo
        cls = w.IntSlider if is_int else w.FloatSlider
        return cls(min=lo, max=hi, step=step, value=value, **common)
    if isinstance(spec, list) and spec and any(isinstance(v, str) for v in spec):
        common.pop("continuous_update")
        options = [(str(v), v) for v in spec]
        return w.Dropdown(options=options, value=default if default in spec else spec[0],
                          **common)
    if isinstance(spec, list) and spec:
        options = [(f"{v:g}", v) for v in spec]
        values = [v for _, v in options]
        return w.SelectionSlider(options=options, value=default if default in values
                                 else values[0], **common)
    return w.fixed(spec)


def slider(func, **controls) -> None:
    """Draw func's figure with a slider (or dropdown) for each control.

    Example: lab.slider(lab.confidence_bar, bar=lab.E4_BARS)
    The picture is redrawn when you let go of the slider.
    """
    if not sliders_ok():
        print("Sliders are not available here. Use the plain cell above instead: "
              "change the value in it and re-run it.")
        return
    import inspect

    import ipywidgets as w
    params = inspect.signature(func).parameters
    widgets = {}
    for name, spec in controls.items():
        default = params[name].default if name in params else None
        if default is inspect.Parameter.empty:
            default = None
        widgets[name] = _widget(name, spec, default)
    w.interact(lambda **kw: func(**kw), **widgets)


# ---------------------------------------------------------------------------
# Data: the four worlds
# ---------------------------------------------------------------------------
def baskets() -> pd.DataFrame:
    """Our 4,000 customers: one row per customer, one True/False column per
    product, the columns in PRODUCTS order (by penetration). A copy."""
    def build():
        b = load_baskets()
        return b[PRODUCTS].astype(bool)
    return _cached("baskets", build).copy()


def _b() -> pd.DataFrame:
    """Our baskets without a copy (private, read only)."""
    baskets()
    return _CACHE["baskets"]


TOY_BASKETS = pd.DataFrame(
    {"checking": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
     "savings": [0, 0, 1, 1, 0, 0, 1, 1, 1, 1],
     "mortgage": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
     "home_insurance": [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]},
    index=pd.Index(range(1, 11), name="customer")).astype(bool)
_TOY = TOY_BASKETS.copy()          # private copy: editing lab.TOY_BASKETS changes nothing


def toy_shuffled(seed: int = 0) -> pd.DataFrame:
    """The ten customers with each product column shuffled on its own:
    np.random.default_rng(seed), one permutation per column in the order
    checking, savings, mortgage, home_insurance. Every column keeps its
    holders; the links between columns are gone."""
    rng = np.random.default_rng(int(seed))
    return pd.DataFrame({p: rng.permutation(_TOY[p].to_numpy()) for p in _TOY.columns},
                        index=_TOY.index)


SENTENCES = [
    "my card was stolen and my account emptied",
    "the app charged my card twice",
    "my card fee was charged without notice",
    "the app crashes and my transfer failed",
    "the mortgage installment on my account went up",
    "the collector calls my phone every day",
]
_S = list(SENTENCES)               # private copy
SPREAD_ORDER = [4, 5, 3, 1, 6]     # Experiment 3: these sentences get " twice", in this order

TOY_TOP = pd.DataFrame(
    {"top probability": [0.71, 0.45, 0.52, 0.93, 0.58, 0.90],
     "label right?": [True, False, False, True, True, True]},
    index=pd.Index([f"S{i}" for i in range(1, 7)], name="text"))

EXAMPLE_TEXT = ("my mortgage installment jumped by 480 pln without clear explanation the "
                "schedule i signed said something different")
MIXED_TEXT = ("i keep receiving collection calls about a loan installment i already paid the "
              "confirmation number is attached and another thing my mortgage installment "
              "jumped by 480 pln without clear explanation the schedule i signed said "
              "something different")


def complaints() -> pd.DataFrame:
    """Our 1,200 made-up complaints (text, category): five topics, 240 each. A copy."""
    return _cached("complaints", load_complaints).copy()


def _split() -> tuple:
    def build():
        from sklearn.model_selection import train_test_split
        c = _cached("complaints", load_complaints)
        tr, te = train_test_split(c, test_size=0.3, random_state=SEED, stratify=c["category"])
        train_texts, test_texts = set(tr["text"]), set(te["text"])
        assert EXAMPLE_TEXT in test_texts and EXAMPLE_TEXT not in train_texts, \
            "EXAMPLE_TEXT must be a test text with no copy among the training texts"
        assert MIXED_TEXT in test_texts, "MIXED_TEXT must be a test text"
        return tr, te
    return _cached("split", build)


def split() -> tuple:
    """train, test: the course split of Block 2 (70/30, stratified by topic,
    random_state=42): 840 and 360 complaints, columns text and category. Copies."""
    tr, te = _split()
    return tr.copy(), te.copy()


# ---------------------------------------------------------------------------
# Rules: itemsets, support, confidence, lift (mlxtend, with a fallback)
# ---------------------------------------------------------------------------
def _association_rules(sets: pd.DataFrame, n: int) -> pd.DataFrame:
    """mlxtend's association_rules(sets, metric="confidence", min_threshold=0.0);
    versions that require the number of rows get num_itemsets=n."""
    from mlxtend.frequent_patterns import association_rules
    try:
        return association_rules(sets, metric="confidence", min_threshold=0.0)
    except TypeError:
        return association_rules(sets, num_itemsets=n, metric="confidence",
                                 min_threshold=0.0)


def _rule_text(left, right) -> str:
    return " + ".join(sorted(left)) + " => " + right


def _order(t: pd.DataFrame) -> pd.DataFrame:
    """Rules in the order of the key (-round(lift, 6), -round(support, 6), rule text)."""
    key = pd.DataFrame({"a": -t["lift"].round(6), "b": -t["support"].round(6),
                        "c": t["rule"]}, index=t.index)
    order = key.sort_values(["a", "b", "c"], kind="mergesort").index
    return t.loc[order].reset_index(drop=True)


def _rules_mlxtend(table: pd.DataFrame, min_support: float) -> pd.DataFrame:
    from mlxtend.frequent_patterns import apriori
    sets = apriori(table, min_support=min_support, use_colnames=True)
    r = _association_rules(sets, len(table))
    r = r[(r["consequents"].apply(len) == 1) & (r["antecedents"].apply(len) <= 2)
          & (r["support"] >= min_support)]
    left = r["antecedents"].apply(lambda s: tuple(sorted(s)))
    right = r["consequents"].apply(lambda s: next(iter(s)))
    return pd.DataFrame({"left": left.to_numpy(), "right": right.to_numpy(),
                         "support": r["support"].to_numpy(float),
                         "confidence": r["confidence"].to_numpy(float),
                         "lift": r["lift"].to_numpy(float)})


def _rules_own(table: pd.DataFrame, min_support: float) -> pd.DataFrame:
    """The same rules without mlxtend: frequent itemsets of one to three
    products level by level, then every split with one product on the right."""
    X = table.to_numpy(bool)
    cols = list(table.columns)
    sup = {}
    prev = []
    for k in (1, 2, 3):
        cands = [(j,) for j in range(len(cols))] if k == 1 else \
            [c for c in combinations(range(len(cols)), k)
             if all(s in sup for s in combinations(c, k - 1))]
        prev = []
        for c in cands:
            s = X[:, list(c)].all(axis=1).mean()
            if s >= min_support:
                sup[c] = s
                prev.append(c)
    rows = []
    for c, s in sup.items():
        if len(c) < 2:
            continue
        for r in c:
            left = tuple(j for j in c if j != r)
            conf = s / sup[left]
            rows.append({"left": tuple(sorted(cols[j] for j in left)), "right": cols[r],
                         "support": s, "confidence": conf, "lift": conf / sup[(r,)]})
    return pd.DataFrame(rows, columns=["left", "right", "support", "confidence", "lift"])


def _rules_full(table: pd.DataFrame, min_support: float) -> pd.DataFrame:
    table = table.astype(bool)
    try:
        r = _rules_mlxtend(table, min_support)
    except ImportError:
        r = _rules_own(table, min_support)
    r["rule"] = [_rule_text(a, b) for a, b in zip(r["left"], r["right"])]
    r["customers"] = (r["support"] * len(table)).round().astype(int)
    r = r[["rule", "left", "right", "support", "confidence", "lift", "customers"]]
    return _order(r)


def _shuffled(shuffle: int) -> pd.DataFrame:
    """Our baskets with each product column shuffled on its own (cached):
    np.random.default_rng(shuffle), one permutation per column in PRODUCTS order."""
    def build():
        b = _b()
        rng = np.random.default_rng(int(shuffle))
        return pd.DataFrame({p: rng.permutation(b[p].to_numpy()) for p in PRODUCTS},
                            index=b.index)
    return _cached(("shuffled", int(shuffle)), build)


def _ms(min_support) -> float:
    return round(float(min_support), 6)


def _rules_of(which, min_support: float = MIN_SUPPORT) -> pd.DataFrame:
    """Cached rules of our baskets (which = "none") or of shuffle `which`."""
    ms = _ms(min_support)
    table = _b() if which == "none" else _shuffled(which)
    return _cached(("rules", which, ms), lambda: _rules_full(table, ms))


def rules(table=None, min_support: float = MIN_SUPPORT) -> pd.DataFrame:
    """The rules of a True/False table (default: our baskets) with one or two
    products on the left and one on the right, held by at least min_support of
    the rows: rule, support, confidence, lift, customers (with A and B). The
    order of the key -lift, -support, rule text. Our table: cached."""
    cols = ["rule", "support", "confidence", "lift", "customers"]
    if table is None:
        return _rules_of("none", min_support)[cols].copy()
    return _rules_full(pd.DataFrame(table), _ms(min_support))[cols]


def _support_of(X: np.ndarray, cols) -> float:
    return float(X[:, list(cols)].all(axis=1).mean())


def _levelwise(table: pd.DataFrame, min_support: float) -> dict:
    """Apriori by hand: per level, every itemset (as column positions) with
    its support and status: "kept", "infrequent" (counted, below the bar) or
    "not counted" (it contains an infrequent itemset of the level below)."""
    X = table.to_numpy(bool)
    p = X.shape[1]
    status, support = {}, {}
    kept_prev = None
    for k in range(1, p + 1):
        kept_now = set()
        for c in combinations(range(p), k):
            support[c] = _support_of(X, c)
            if k > 1 and not all(s in kept_prev for s in combinations(c, k - 1)):
                status[c] = "not counted"
            elif support[c] >= min_support - 1e-12:
                status[c] = "kept"
                kept_now.add(c)
            else:
                status[c] = "infrequent"
        kept_prev = kept_now
    return {"status": status, "support": support, "columns": list(table.columns)}


def levels(table=None, min_support: float = MIN_SUPPORT) -> pd.DataFrame:
    """Apriori level by level (default: our baskets): for each number of
    products in an itemset, how many itemsets there are, how many Apriori
    counts (every smaller itemset inside was kept) and how many it keeps."""
    def build(t, ms):
        lw = _levelwise(t, ms)
        rows = []
        for k in range(1, t.shape[1] + 1):
            st = [s for c, s in lw["status"].items() if len(c) == k]
            rows.append({"all": len(st), "counted": sum(s != "not counted" for s in st),
                         "kept": sum(s == "kept" for s in st)})
        return pd.DataFrame(rows, index=pd.Index(range(1, t.shape[1] + 1),
                                                 name="products in the itemset"))
    ms = _ms(min_support)
    if table is None:
        return _cached(("levels", ms), lambda: build(_b(), ms)).copy()
    return build(pd.DataFrame(table).astype(bool), ms)


def toy_counts() -> pd.DataFrame:
    """Every itemset of the ten customers (15): its level, how many customers
    hold all of it, and its support."""
    X = _TOY.to_numpy(bool)
    cols = list(_TOY.columns)
    rows = []
    for k in range(1, 5):
        for c in combinations(range(4), k):
            n = int(X[:, list(c)].all(axis=1).sum())
            rows.append({"itemset": " + ".join(cols[j] for j in c), "level": k,
                         "customers": n, "support": n / len(X)})
    return pd.DataFrame(rows).set_index("itemset")


def toy_levels(bar: int = 3) -> pd.DataFrame:
    """Apriori on the ten customers with a minimum support of `bar` customers:
    every itemset with its level, its count and its status (kept, infrequent,
    not counted)."""
    bar = _whole(bar, 1, 10, 3, "bar")
    lw = _levelwise(_TOY, bar / len(_TOY))
    t = toy_counts()
    t["status"] = [lw["status"][c] for k in range(1, 5) for c in combinations(range(4), k)]
    return t[["level", "customers", "status"]]


_LEVEL_NAMES = {1: "one product", 2: "pairs", 3: "sets of three", 4: "all four"}


def _toy_rule_rows(table: pd.DataFrame, pairs) -> pd.DataFrame:
    n = len(table)
    rows = []
    for x, z in pairs:
        for a, b in ((x, z), (z, x)):
            both = int((table[a] & table[b]).sum())
            n_a = int(table[a].sum())
            conf = both / n_a
            rows.append({"rule": _rule_text((a,), b), "support": both / n, "confidence": conf,
                         "lift": conf / table[b].mean(), "penetration of B": table[b].mean(),
                         "with A but not B": n_a - both})
    return _order(pd.DataFrame(rows))


def rules_by_hand() -> pd.DataFrame:
    """Demo 1: steps 1-4 on the ten customers. Prints the counting level by
    level (minimum support 3 customers) and returns the eight rules of the four
    kept pairs: support, confidence, penetration of B, lift (in the key's order)."""
    t = toy_levels(3)
    total_counted = total_kept = 0
    for k in range(1, 5):
        lv = t[t["level"] == k]
        counted = lv[lv["status"] != "not counted"]
        kept = lv[lv["status"] == "kept"]
        low = lv[lv["status"] == "infrequent"]
        skipped = lv[lv["status"] == "not counted"]
        total_counted += len(counted)
        total_kept += len(kept)
        parts = [f"Level {k} ({_LEVEL_NAMES[k]}): counted {len(counted)}"]
        if len(counted):
            parts[0] += " - " + ", ".join(f"{i} {v}" for i, v in counted["customers"].items())
        if len(low):
            parts.append("below 3 customers: " + ", ".join(low.index))
        if len(skipped):
            parts.append(f"not counted {len(skipped)} (each contains an infrequent itemset)")
        parts.append(f"kept {len(kept)}")
        print("; ".join(parts))
    print(f"Counted {total_counted} of 15 itemsets; kept {total_kept}.")
    kept_pairs = [tuple(i.split(" + ")) for i in t[(t["level"] == 2) &
                                                     (t["status"] == "kept")].index]
    r = _toy_rule_rows(_TOY, kept_pairs)
    out = pd.DataFrame({"support (%)": (100 * r["support"]).round(0).astype(int),
                        "confidence (%)": (100 * r["confidence"]).round(0).astype(int),
                        "penetration of B (%)": (100 * r["penetration of B"]).round(0).astype(int),
                        "lift": r["lift"].round(2),
                        "with A but not B": r["with A but not B"]})
    out.index = pd.Index(r["rule"], name="rule")
    return out


_RULE_TABLE = [(("savings",), "checking"), (("brokerage", "home_insurance"), "mortgage"),
               (("mortgage",), "home_insurance"), (("brokerage",), "term_deposit")]


def _rule_numbers(table: pd.DataFrame, left, right) -> dict:
    left = [left] if isinstance(left, str) else list(left)
    a = table[left].all(axis=1)
    both = int((a & table[right]).sum())
    n_a = int(a.sum())
    conf = both / n_a if n_a else float("nan")
    pen = float(table[right].mean())
    return {"rule": _rule_text(left, right), "customers with A": n_a,
            "customers with A and B": both, "support": both / len(table), "confidence": conf,
            "penetration of B": pen, "lift": conf / pen, "with A, not B": n_a - both}


def rule_table(min_support: float = MIN_SUPPORT) -> pd.DataFrame:
    """Demo 3 (slide 24): four of our rules sorted by confidence, with B's
    penetration and the customers with A but not B. Rules held by fewer than
    min_support of the customers are left out (with a note)."""
    ms = _share(min_support, 0.0, 0.5, MIN_SUPPORT, "min_support")
    b = _b()
    rows = [_rule_numbers(b, a, c) for a, c in _RULE_TABLE]
    keep = [r for r in rows if r["support"] >= ms - 1e-12]
    for r in rows:
        if r not in keep:
            _say(f"{r['rule']}: held by {r['customers with A and B']:,} customers "
                 f"({_pct(r['support'])}), below the minimum support of {ms:g} - left out.")
    t = pd.DataFrame(keep).sort_values("confidence", ascending=False, kind="mergesort")
    out = pd.DataFrame({"customers with A and B": t["customers with A and B"].to_numpy(),
                        "confidence (%)": (100 * t["confidence"]).round(1).to_numpy(),
                        "B's penetration (%)": (100 * t["penetration of B"]).round(1).to_numpy(),
                        "lift": t["lift"].round(2).to_numpy(),
                        "with A, not B": t["with A, not B"].to_numpy()},
                       index=pd.Index(t["rule"], name="rule"))
    return out


def customers_to_approach(a="mortgage", b="home_insurance") -> pd.DataFrame:
    """The customers who hold a (one product or a list) but not b: the rows
    of our baskets. len(...) counts them (mortgage, home_insurance: 122)."""
    t = _b()
    left = [a] if isinstance(a, str) else list(a)
    left = [_choice(x, PRODUCTS, "product") for x in left]
    b = _choice(b, PRODUCTS, "product")
    mask = t[left].all(axis=1) & ~t[b]
    return t[mask].copy()


# ---------------------------------------------------------------------------
# Slide pictures of the ten customers and of our baskets
# ---------------------------------------------------------------------------
def _ax_size_pt(ax) -> tuple:
    """The axes' width and height in points once the layout is done."""
    fig = ax.figure
    fig.canvas.draw()
    bb = ax.get_window_extent()
    return bb.width * 72 / fig.dpi, bb.height * 72 / fig.dpi


def _text_pt(ax, text: str, fontsize: float = 9) -> float:
    """The width (points) of the widest line of a text at this font size."""
    r = ax.figure.canvas.get_renderer()
    out = 0.0
    for line in str(text).split("\n"):
        t = ax.text(0, 0, line, fontsize=fontsize)
        out = max(out, t.get_window_extent(r).width * 72 / ax.figure.dpi)
        t.remove()
    return out


def draw_toy_grid(ax, table=None, frame=None, ring=None, shade=None) -> dict:
    """The ten customers' baskets: a row per customer 1-10, a column per
    product; a dark square = holds it, an empty square = does not; under the
    grid the holders of each product. frame = product columns framed in red;
    ring = customers ringed in red (their cells in the framed columns, or
    their number when nothing is framed); shade = customers whose row is
    shaded light blue. The column names stand on one line when there is
    room, else on two lines ("home / insurance"), else on two staggered rows
    (every second name one line higher).
    Laid out in points: create the figure at its final size (and set any
    panel title) before calling it."""
    t = (_TOY if table is None else pd.DataFrame(table)).astype(bool)
    cols = list(t.columns)
    custs = list(t.index)
    n, m = len(custs), len(cols)
    frame = [p for p in (frame or []) if p in cols]
    ring = list(ring or [])
    shade = list(shade or [])
    ax.axis("off")
    W, H = _ax_size_pt(ax)
    gap = 12.0                                 # a number's right edge to its first square's centre
    left = _text_pt(ax, "holders") + gap
    one = [_label(c) for c in cols]
    two = [x.replace(" ", "\n") for x in one]
    w1 = [_text_pt(ax, x) for x in one]
    w2 = [_text_pt(ax, x) for x in two]
    lines2 = max(x.count("\n") + 1 for x in two)
    pitch = lambda right: (W - left - right) / max(m - 1, 1)   # noqa: E731
    # the room two neighbouring names need: one-line names stand clearly apart
    need = lambda w, sep: max([(w[j] + w[j + 1]) / 2 + sep for j in range(m - 1)] or [0])  # noqa: E731
    if m < 2 or pitch(w1[-1] / 2) >= need(w1, 14):
        mode, names, right, nlines, k = "one line", one, w1[-1] / 2, 1, need(w1, 14)
    elif pitch(w2[-1] / 2) >= need(w2, 7):
        mode, names, right, nlines, k = "two lines", two, w2[-1] / 2, lines2, need(w2, 7)
    else:
        mode, names, right, nlines, k = "staggered", two, w2[-1] / 2, lines2 + 1, 0.0
    p = min(pitch(right), max(60.0, k))        # a wide panel: no wider than needed
    shift = (W - left - right - p * (m - 1)) / 2   # centre a grid that has room to spare
    line = 10.5
    names_pt = line * nlines + 4.0
    q = (H - names_pt) / (n + 1.9 - 0.45)      # points per row
    num_fs = 9 if q >= 10.0 else 8.5           # a row's number never touches the next
    size = max(5.0, min(11.0, 0.6 * q))
    y_lo, y_top = -n - 1.9, -0.45
    y_hi = y_top + names_pt / q
    x_lo = -(left + shift) / p
    x_hi = (m - 1) + (right + shift) / p
    xs = np.arange(m)
    for i, c in enumerate(custs):
        y = -(i + 1)
        if c in shade:
            ax.add_patch(Rectangle((-0.5 * size / p - 0.25, y - 0.48), m - 1 + size / p + 0.5,
                                   0.96, color=LIGHT, lw=0, zorder=0))
        hold = t.loc[c].to_numpy(bool)
        ax.scatter(xs[hold], [y] * int(hold.sum()), marker="s", s=size ** 2, color=DARK,
                   edgecolor=DARK, linewidths=0.8, zorder=2)
        ax.scatter(xs[~hold], [y] * int((~hold).sum()), marker="s", s=size ** 2,
                   facecolor="white", edgecolor=GRAY, linewidths=0.8, zorder=2)
        circle = dict(boxstyle="circle,pad=0.2", fc="none", ec=RED, lw=1.3) \
            if (c in ring and not frame) else None
        ax.annotate(str(c), (0, y), xytext=(-gap, 0), textcoords="offset points",
                    ha="right", va="center", fontsize=num_fs, bbox=circle)
        if c in ring and frame:
            fx = [cols.index(pp) for pp in frame]
            ax.scatter(fx, [y] * len(fx), marker="o", s=(1.6 * size) ** 2,
                       facecolor="none", edgecolor=RED, linewidths=1.3, zorder=3)
    half = 1.25 * size / p
    for pp in frame:
        j = cols.index(pp)
        ax.add_patch(Rectangle((j - half, -n - 0.62), 2 * half, n + 0.24, fill=False,
                               edgecolor=RED, lw=1.5, zorder=4))
    holders = t.sum().astype(int)
    for j, pp in enumerate(cols):
        upper = mode == "staggered" and j % 2 == 1
        ax.annotate(names[j], (j, y_top), xytext=(0, 2 + (line if upper else 0)),
                    textcoords="offset points", ha="center", va="bottom", fontsize=9,
                    linespacing=1.0)
        ax.text(j, -n - 1.3, str(holders[pp]), ha="center", va="center", fontsize=9,
                color=DARK, fontweight="bold")
    ax.annotate("holders", (0, -n - 1.3), xytext=(-gap, 0), textcoords="offset points",
                ha="right", va="center", fontsize=9, color=MUTED)
    for tx in ax.texts:   # the layout is in points: texts must not move the axes
        tx.set_in_layout(False)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    return {"holders": holders.to_dict(), "ringed": ring, "framed": frame}


def draw_toy_levels(ax, stage: str = "a") -> pd.DataFrame:
    """Apriori on the ten customers, four rows of boxes (level 1 at the bottom):
    stage "a" - levels 1-2 counted: kept dark, infrequent with a red border;
    levels 3-4 grey outlines with "?". Stage "b" - the supersets of an
    infrequent pair dashed grey, "not counted", with a thin grey line to the
    infrequent itemset below them; checking + mortgage + home insurance
    counted and kept. The sets of three stand above the pairs they contain,
    so every line is short. Laid out in points: draw it at its final size."""
    stage = _choice(str(stage).lower(), ["a", "b"], "stage")
    lw = _levelwise(_TOY, 0.3)
    status = lw["status"]
    cols = list(_TOY.columns)
    ax.axis("off")
    W, H = _ax_size_pt(ax)
    lh, pad = 10.2, 3.5
    lines = {1: 2, 2: 4, 3: 3, 4: 1}
    # a product name too wide for its level-1 box (3 pt to spare on each side)
    # goes on two lines
    wrap1 = {c: _text_pt(ax, _label(c), 8.5) > (W - 60.0) / 4 - 3 - 6 for c in cols}
    if any(wrap1.values()):
        lines[1] = 3
    h = {k: lines[k] * lh + 2 * pad for k in lines}
    spare = H - sum(h.values())
    gaps = {1: 0.30 * spare, 2: 0.45 * spare, 3: 0.25 * spare}   # above level k
    ys, y = {}, 0.0
    for k in (1, 2, 3, 4):
        ys[k] = y + h[k] / 2
        y += h[k] + gaps.get(k, 0)
    x0 = 60.0
    area = W - x0
    names = {1: "level 1:\none product", 2: "level 2:\npairs", 3: "level 3:\nsets of three",
             4: "level 4:\nall four"}
    pairs = list(combinations(range(4), 2))
    slot2 = area / len(pairs)
    pair_x = {c: x0 + slot2 * (i + 0.5) for i, c in enumerate(pairs)}
    # Sets of three, left to right: the kept one, then each superset of an
    # infrequent pair right above that pair (every line short, none crossing).
    kept3 = [c for c in combinations(range(4), 3) if status[c] == "kept"]
    rest3 = [c for c in combinations(range(4), 3) if status[c] != "kept"]
    w3 = {"kept": 96.0, "not counted": 42.0}
    aim = {c: float(np.mean([pair_x[s] for s in combinations(c, 2) if status[s] != "kept"]))
           for c in rest3}
    rest3.sort(key=lambda c: (aim[c], c))
    x3 = {}
    if rest3:
        sep = w3["not counted"] + 8
        mid = float(np.mean([aim[c] for c in rest3]))
        mid = min(mid, W - 2 - w3["not counted"] / 2 - sep * (len(rest3) - 1) / 2)
        for i, c in enumerate(rest3):
            x3[c] = mid + sep * (i - (len(rest3) - 1) / 2)
    first = min(x3.values()) - w3["not counted"] / 2 if x3 else W
    for c in kept3:
        x3[c] = min(float(np.mean([pair_x[s] for s in combinations(c, 2)])),
                    first - 10 - w3["kept"] / 2)
    x4 = x3[rest3[len(rest3) // 2]] if rest3 else x0 + area / 2
    boxes, out = {}, []

    def box(c, x, yc, w, hh, text, kind):
        style = {"kept": (DARK, DARK, "white", "-"), "infrequent": ("white", RED, RED, "-"),
                 "not counted": ("white", GRAY, MUTED, (0, (3, 2))),
                 "unknown": ("white", "#C3C9CF", "#9AA3AC", "-")}[kind]
        fc, ec, tc, ls = style
        ax.add_patch(FancyBboxPatch((x - w / 2, yc - hh / 2), w, hh,
                                    boxstyle="round,pad=0,rounding_size=3", fc=fc, ec=ec,
                                    lw=1.1, ls=ls, zorder=3))
        ax.text(x, yc, text, ha="center", va="center", fontsize=8.5, color=tc,
                linespacing=1.0, zorder=4)
        boxes[c] = (x, yc, w, hh)

    for k in (1, 2, 3, 4):
        ax.text(0, ys[k], names[k], ha="left", va="center", fontsize=9, color=MUTED,
                linespacing=1.0)
        sets = list(combinations(range(4), k))
        if k == 3:
            sets = kept3 + rest3
        for i, c in enumerate(sets):
            st = status[c]
            cnt = int(round(lw["support"][c] * 10))
            if k == 1:
                s1 = area / 4
                name1 = _label(cols[c[0]])
                if wrap1[cols[c[0]]]:
                    name1 = name1.replace(" ", "\n")
                box(c, x0 + s1 * (i + 0.5), ys[k], s1 - 3, h[k], f"{name1}\n{cnt}", st)
            elif k == 2:
                text = "\n".join(_label(cols[j]).replace(" ", "\n") for j in c)
                box(c, pair_x[c], ys[k], slot2 - 3.5, h[k], f"{text}\n{cnt}", st)
            else:
                x = x3[c] if k == 3 else x4
                w = (w3["kept"] if st == "kept" else w3["not counted"]) if k == 3 else 74.0
                if stage == "a":   # the boxes of stage b, still empty
                    box(c, x, ys[k], w, h[k], "?", "unknown")
                    st = "unknown"
                elif st == "not counted":
                    box(c, x, ys[k], w, h[k], "not\ncounted" if k == 3 else "not counted", st)
                else:
                    names3 = [_label(cols[j]) for j in c]
                    box(c, x, ys[k], w, h[k],
                        f"{' + '.join(names3[:2])}\n+ {' + '.join(names3[2:])}\n{cnt}", st)
            out.append({"itemset": " + ".join(cols[j] for j in c), "level": k, "shown": st,
                        "customers": cnt if st in ("kept", "infrequent") else None})
    if stage == "b":   # each superset that is not counted, to an infrequent itemset in it
        for c in [c for c in boxes if len(c) >= 3 and status[c] == "not counted"]:
            k = len(c)
            below = [s for s in combinations(c, k - 1)
                     if status[s] in ("infrequent", "not counted")]
            if k == 4:   # the nearest set of three below it
                below = sorted(below, key=lambda s: abs(boxes[s][0] - boxes[c][0]))[:1]
            xa, ya, wa, ha_ = boxes[c]
            for s in below:
                xb, yb, wb, hb = boxes[s]
                xa1 = float(np.clip(xb, xa - wa / 2 + 5, xa + wa / 2 - 5))
                xb1 = float(np.clip(xa1, xb - wb / 2 + 5, xb + wb / 2 - 5))
                ax.plot([xa1, xb1], [ya - ha_ / 2, yb + hb / 2], color="#A9B0B7", lw=0.9,
                        zorder=1)
    for t in ax.texts:   # the layout is in points: texts must not move the axes
        t.set_in_layout(False)
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    return pd.DataFrame(out)


def draw_compare_bars(ax, a: str = "mortgage", b: str = "home_insurance", table=None) -> dict:
    """Two bars for product b: the share of a's holders who hold it (red) and
    the share of all customers who hold it, b's penetration (grey); a bracket
    with the ratio, the lift."""
    t = (_TOY if table is None else pd.DataFrame(table)).astype(bool)
    r = _rule_numbers(t, a, b)
    conf, pen, lift = 100 * r["confidence"], 100 * r["penetration of B"], r["lift"]
    n = len(t)
    ax.bar([0, 1], [conf, pen], width=0.6, color=[RED, GRAY])
    for x, v in ((0, conf), (1, pen)):
        ax.text(x, v + 2, f"{v:.0f}%", ha="center", va="bottom", fontsize=9)
    lo, hi = sorted((conf, pen))
    xb = 1.48
    ax.plot([xb - 0.07, xb, xb, xb - 0.07], [lo, lo, hi, hi], color=NEAR_BLACK, lw=1)
    lift_txt = f"{lift:.0f}" if abs(lift - round(lift)) < 1e-9 else f"{lift:.2f}"
    ax.text(xb + 0.08, (lo + hi) / 2, f"× {lift_txt}", ha="left", va="center",
            fontsize=9.5, color=NEAR_BLACK)
    ax.set_xticks([0, 1])
    who = "all ten\ncustomers" if n == 10 else f"all {n:,}\ncustomers"
    ax.set_xticklabels([f"{_label(a)}\nholders", who])
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.55, 2.15)
    ax.set_ylim(0, 115)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel(f"hold {_label(b)} (%)")
    ax.set_title(_rule_label(a, b))
    return {"confidence": r["confidence"], "penetration": r["penetration of B"],
            "lift": lift, "customers with A": r["customers with A"],
            "customers with A and B": r["customers with A and B"]}


def draw_penetration(ax) -> pd.Series:
    """Our ten products: the share of the 4,000 customers who hold each (its
    penetration), largest at the top."""
    pen = 100 * _b().mean()
    pen = pen.loc[PRODUCTS]
    ys = np.arange(len(pen))[::-1]
    ax.barh(ys, pen.to_numpy(), height=0.7, color=BLUE)
    for y, v in zip(ys, pen.to_numpy()):
        ax.text(v + 1.5, y, f"{v:.1f}%", ha="left", va="center", fontsize=9)
    ax.set_yticks(ys)
    ax.set_yticklabels([_label(p) for p in pen.index])
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 112)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("customers who hold it (%)")
    return pen / 100


def draw_levels(ax, table=None, min_support: float = MIN_SUPPORT) -> pd.DataFrame:
    """Apriori level by level: for each number of products in an itemset,
    all itemsets (grey outline), those counted (light) and those kept (dark)."""
    t = levels(table, min_support)
    x = t.index.to_numpy()
    ax.bar(x, t["all"], width=0.78, facecolor="white", edgecolor=GRAY, lw=1,
           label="all itemsets")
    ax.bar(x, t["counted"], width=0.78, color=LIGHT_BLUE, label="counted")
    ax.bar(x, t["kept"], width=0.44, color=DARK, label="kept")
    ax.set_xticks(x)
    ax.set_xlabel("products in the itemset")
    ax.set_ylabel("itemsets")
    ax.set_ylim(0, 1.12 * t["all"].max())
    ax.legend(loc="upper right", handlelength=1.2)
    return t


# ---------------------------------------------------------------------------
# Demo 2 - the minimum support
# ---------------------------------------------------------------------------
def support_bar(min_support: float = MIN_SUPPORT) -> pd.Series:
    """Demo 2: Apriori on our baskets at a minimum support (0.01 to 0.10):
    the figure shows, per level, all itemsets, those counted and those kept;
    returns the minimum support, the customers it means, the itemsets counted
    and kept, and the rules (one or two products left, one right)."""
    ms = _share(min_support, 0.01, 0.10, MIN_SUPPORT, "min_support")
    lv = levels(None, ms)
    r = _rules_of("none", ms)
    with _style():
        fig, ax = _new_fig((6.2, 3.2))
        draw_levels(ax, None, ms)
        ax.set_title(f"minimum support {ms:g} = {int(math.ceil(ms * len(_b()) - 1e-9)):,} "
                     f"customers: {lv['counted'].sum()} itemsets counted, "
                     f"{lv['kept'].sum()} kept, {len(r)} rules")
    _show(fig)
    return pd.Series({"minimum support": ms,
                      "at least ... customers": int(math.ceil(ms * len(_b()) - 1e-9)),
                      "itemsets counted": int(lv["counted"].sum()),
                      "itemsets kept": int(lv["kept"].sum()), "rules": len(r)}, dtype=object)


# ---------------------------------------------------------------------------
# E1 - Confidence against everyone
# ---------------------------------------------------------------------------
E1_PRODUCTS = list(PRODUCTS)
_PRODUCT_ALIASES = {"insurance": "home_insurance", "deposit": "term_deposit",
                    "loan": "personal_loan", "card": "debit_card", "debit": "debit_card",
                    "credit": "credit_card", "car": "car_loan", "account": "checking"}


def _product(value, what: str = "right_side") -> str:
    return _choice(value, PRODUCTS, what, _PRODUCT_ALIASES)


def confidence_numbers() -> pd.DataFrame:
    """All 90 rules A => B with one product on each side: for each right side
    B (in PRODUCTS order) its nine rules in the order of the bars (lift, then
    A's name)."""
    def build():
        b = _b()
        rows = []
        for right in PRODUCTS:
            part = []
            for left in PRODUCTS:
                if left == right:
                    continue
                r = _rule_numbers(b, left, right)
                part.append({"B (right side)": right, "A (left side)": left,
                             "customers with A and B": r["customers with A and B"],
                             "support (%)": 100 * r["support"],
                             "confidence (%)": 100 * r["confidence"],
                             "B's penetration (%)": 100 * r["penetration of B"],
                             "lift": r["lift"]})
            part.sort(key=lambda d: (-round(d["lift"], 6), d["A (left side)"]))
            rows += part
        return pd.DataFrame(rows)
    return _cached("e1", build).copy()


def _e1(right_side: str) -> pd.DataFrame:
    t = confidence_numbers()
    return t[t["B (right side)"] == right_side].reset_index(drop=True)


def draw_confidence_bars(ax, right_side: str = "debit_card", labels: bool = True) -> pd.DataFrame:
    """Each rule A => B for one B: its confidence (bars, %) and B's
    penetration (dashed line, with its value). Rules sorted by lift, largest
    on top."""
    right_side = _product(right_side)
    t = _e1(right_side)
    ys = np.arange(len(t))[::-1]
    conf = t["confidence (%)"].to_numpy()
    pen = float(t["B's penetration (%)"].iloc[0])
    ax.barh(ys, conf, height=0.68, color=BLUE)
    for y, v in zip(ys, conf):
        ax.text(max(v, pen) + 2, y, f"{v:.1f}", ha="left", va="center", fontsize=8.5)
    top = len(t) - 0.35
    ax.plot([pen, pen], [-0.6, top + 0.45], ls="--", color=DARK, lw=1.1)
    lab = f"everyone {pen:.1f}%"
    if pen > 55:
        ax.text(pen - 1.5, top + 0.55, lab, ha="right", va="center", fontsize=8.5, color=DARK)
    else:
        ax.text(pen + 1.5, top + 0.55, lab, ha="left", va="center", fontsize=8.5, color=DARK)
    ax.set_yticks(ys)
    ax.set_yticklabels([_label(a) for a in t["A (left side)"]] if labels else [])
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.6, len(t) + 0.45)
    ax.set_xlim(0, 118)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("confidence (%)")
    ax.set_title(f"A {ARROW} {_label(right_side)}")
    return t


def draw_lift_bars(ax, right_side: str = "debit_card", labels: bool = True) -> pd.DataFrame:
    """Each rule A => B for one B: its lift (bars; 1.5 or more in red), the
    dashed line at 1 (no link). The same order as draw_confidence_bars."""
    right_side = _product(right_side)
    t = _e1(right_side)
    ys = np.arange(len(t))[::-1]
    lift = t["lift"].to_numpy()
    ax.barh(ys, lift, height=0.68, color=[RED if v >= 1.5 else BLUE for v in lift])
    for y, v in zip(ys, lift):
        ax.text(max(v, 1) + 0.08, y, f"{v:.2f}", ha="left", va="center", fontsize=8.5)
    ax.axvline(1, ls="--", color=DARK, lw=1.1)
    ax.set_yticks(ys)
    ax.set_yticklabels([_label(a) for a in t["A (left side)"]] if labels else [])
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.6, len(t) + 0.45)
    ax.set_xlim(0, 4.6)
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_xlabel("lift")
    ax.set_title("lift = confidence / everyone")
    return t


def confidence_against_everyone_figure(right_side: str = "debit_card", figsize=None,
                                       suptitle: bool = True):
    right_side = _product(right_side)
    t = _e1(right_side)
    with _style():
        fig, axes = _new_fig(figsize or (8.0, 3.5), 1, 2, width_ratios=[1.25, 1])
        draw_confidence_bars(axes[0], right_side)
        draw_lift_bars(axes[1], right_side, labels=False)
        if suptitle:
            best = t.iloc[0]
            pen = float(t["B's penetration (%)"].iloc[0])
            fig.suptitle(f"right side: {_label(right_side)}, held by {pen:.1f}% of everyone; "
                         f"highest lift: {_rule_label(best['A (left side)'], right_side)}, "
                         f"{best['lift']:.2f}", fontsize=9.5)
    return fig


def confidence_against_everyone(right_side: str = "debit_card") -> None:
    """E1: pick a product B, the right side; each bar is a rule A => B, one
    for each of the other nine products A: its confidence against B's
    penetration (left) and its lift (right)."""
    _show(confidence_against_everyone_figure(_product(right_side)))


# ---------------------------------------------------------------------------
# E2 - Shuffle the baskets
# ---------------------------------------------------------------------------
E2_SHUFFLES = ["none"] + list(range(1, 21))
_E2_TRAIL: list = []


def _shuffle_value(value):
    """ "none" (our baskets) or a whole number 1-20."""
    value = _plain(value)
    if isinstance(value, str) and _key(value) in ("none", "ours", "our", "ourbaskets", "no",
                                                   "original", ""):
        return "none"
    v = _as_number(value)
    if isinstance(v, str) or v is None:
        _say(f'shuffle {value!r} is not "none" or a number from 1 to 20: showing "none".')
        return "none"
    if _is_number(v) and float(v) == 0:
        _say('shuffle 0 means no shuffle: showing "none" (our baskets).')
        return "none"
    return _whole(v, 1, 20, 1, "shuffle")


def _e2_summary(which, min_support: float) -> dict:
    r = _rules_of(which, min_support)
    table = _b() if which == "none" else _shuffled(which)
    return {"rules": len(r), "best lift": float(r["lift"].iloc[0]) if len(r) else float("nan"),
            "best rule": r["rule"].iloc[0] if len(r) else "",
            "lift 1.5 or more": int((r["lift"] >= 1.5).sum()),
            "mortgage + home insurance": int((table["mortgage"] & table["home_insurance"]).sum())}


def shuffle_numbers(min_support: float = MIN_SUPPORT) -> pd.DataFrame:
    """Our baskets ("none") and shuffles 1-20: the rules that pass, the best
    lift and its rule (the first by the key), the rules with lift 1.5 or more
    and the customers holding mortgage and home insurance."""
    ms = _ms(min_support)
    t = pd.DataFrame([_e2_summary(s, ms) for s in E2_SHUFFLES],
                     index=pd.Index(E2_SHUFFLES, name="shuffle", dtype=object))
    return t[["rules", "best lift", "best rule", "lift 1.5 or more",
              "mortgage + home insurance"]]


def draw_rule_cloud(ax, shuffle="none", min_support: float = MIN_SUPPORT) -> pd.DataFrame:
    """Every rule that passes as a dot: support across, lift up; the dashed
    line at lift 1. A shuffle's rules in red with ours in grey; "none": ours
    in blue."""
    shuffle = _shuffle_value(shuffle)
    ms = _ms(min_support)
    ours = _rules_of("none", ms)
    if shuffle == "none":
        ax.scatter(100 * ours["support"], ours["lift"], s=10, color=BLUE, alpha=0.75,
                   linewidths=0, zorder=3, label=f"our baskets ({len(ours)})")
        shown = ours
    else:
        sh = _rules_of(shuffle, ms)
        ax.scatter(100 * ours["support"], ours["lift"], s=9, color="#B4BBC2", alpha=0.8,
                   linewidths=0, zorder=2, label=f"our baskets ({len(ours)})")
        ax.scatter(100 * sh["support"], sh["lift"], s=10, color=RED, alpha=0.75,
                   linewidths=0, zorder=3, label=f"shuffle {shuffle} ({len(sh)})")
        shown = sh
    ax.axhline(1, ls="--", color=DARK, lw=1)
    ax.set_xlim(0, 62)
    ax.set_ylim(0.4, 4.7)
    ax.set_xlabel("support (%)")
    ax.set_ylabel("lift")
    ax.legend(loc="upper right", handletextpad=0.2, markerscale=1.6)
    ax.set_title("every rule that passes")
    return shown[["rule", "support", "confidence", "lift", "customers"]]


def draw_best_lifts(ax, tried=None, current=None, min_support: float = MIN_SUPPORT,
                    ours_grey=None) -> pd.Series:
    """The best lift of every table tried: our baskets first, then the
    shuffles in order (red; the current one darker); dashed line at 1.
    The bar of our baskets is grey (#B4BBC2, as "our baskets" in
    draw_rule_cloud) when ours_grey is True, or when ours_grey is None and
    current is a shuffle (1-20); blue otherwise (current "none")."""
    ms = _ms(min_support)
    tried = [_shuffle_value(s) if not (isinstance(s, (int, np.integer)) and 1 <= s <= 20)
             else int(s) for s in (tried or [])]
    if current is not None:
        current = _shuffle_value(current)
        if current not in tried:
            tried.append(current)
    shuffles = sorted({s for s in tried if s != "none"})
    order = ["none"] + shuffles
    best = pd.Series({s: _e2_summary(s, ms)["best lift"] for s in order})
    xs = np.array([0.0] + [i + 0.8 for i in range(1, len(order))])
    many = len(order) > 8
    if ours_grey is None:
        ours_grey = current is not None and current != "none"
    colours = [("#B4BBC2" if ours_grey else BLUE) if s == "none" else
               (RED if (s == current or current is None or current == "none") else LIGHT_RED)
               for s in order]
    ax.bar(xs, best.to_numpy(), width=0.7, color=colours)
    ax.axhline(1, ls="--", color=DARK, lw=1)
    for x, s in zip(xs, order):
        if s == "none":
            ax.text(x + 0.45, best[s], f"{best[s]:.2f}", ha="left", va="center", fontsize=8.5)
        elif s == current or not many:
            ax.text(x, best[s] + 0.08, f"{best[s]:.2f}", ha="center", va="bottom",
                    fontsize=8.5)
    if not many:
        ticks = list(range(len(order)))
    else:
        ticks = [0] + [i for i, s in enumerate(order) if s != "none" and
                       (s in (5, 10, 15, 20) or s == current)]
    ax.set_xticks(xs[ticks])
    ax.set_xticklabels(["ours" if order[i] == "none" else str(order[i]) for i in ticks])
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.7, max(xs[-1], 2.5) + 0.7)
    ax.set_ylim(0, 4.85)
    ax.set_ylabel("best lift")
    ax.set_xlabel("table (shuffle)")
    ax.set_title("the tables you tried")
    best.index = ["ours" if s == "none" else s for s in order]
    return best


def shuffle_baskets_figure(shuffle="none", min_support: float = MIN_SUPPORT, tried=None,
                           figsize=None, suptitle: bool = True):
    shuffle = _shuffle_value(shuffle)
    ms = _ms(min_support)
    with _style():
        fig, axes = _new_fig(figsize or (8.0, 3.5), 1, 2, width_ratios=[1.25, 1])
        draw_rule_cloud(axes[0], shuffle, ms)
        draw_best_lifts(axes[1], tried if tried is not None else [shuffle], shuffle, ms)
        if suptitle:
            s = _e2_summary(shuffle, ms)
            bar = "" if abs(ms - MIN_SUPPORT) < 1e-9 else f" (minimum support {ms:g})"
            if shuffle == "none":
                fig.suptitle(f"our baskets: {s['rules']} rules pass{bar}; best lift "
                             f"{s['best lift']:.2f}; {s['lift 1.5 or more']} rules with lift "
                             "1.5 or more", fontsize=9.5)
            else:
                o = _e2_summary("none", ms)
                fig.suptitle(f"shuffle {shuffle}: {s['rules']} rules pass{bar}; best lift "
                             f"{s['best lift']:.2f} (ours {o['best lift']:.2f})", fontsize=9.5)
    return fig


def shuffle_baskets(shuffle="none", min_support: float = MIN_SUPPORT) -> None:
    """E2: our baskets with each product column shuffled on its own (shuffle
    1-20; "none" = our baskets), then the rules that pass the minimum support;
    the best lift of every table you tried stays on the right
    (shuffle_baskets_reset() clears it)."""
    s = _shuffle_value(shuffle)
    ms = _share(min_support, 0.005, 0.5, MIN_SUPPORT, "min_support")
    if s not in _E2_TRAIL:
        _E2_TRAIL.append(s)
    _show(shuffle_baskets_figure(s, ms, list(_E2_TRAIL)))


def shuffle_baskets_reset() -> None:
    _E2_TRAIL.clear()


# ---------------------------------------------------------------------------
# Six sentences: words, bag of words, idf, TF-IDF
# ---------------------------------------------------------------------------
def _tfidf(texts):
    from sklearn.feature_extraction.text import TfidfVectorizer
    tv = TfidfVectorizer()
    X = tv.fit_transform(texts).toarray()
    return tv, X


def toy_bag_of_words() -> pd.DataFrame:
    """Step 2 on the six sentences: the bag of words, one row per sentence
    S1-S6, one column per word of the vocabulary (27, alphabetical as
    scikit-learn orders them); each cell counts the word in the sentence."""
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer()
    X = cv.fit_transform(_S).toarray()
    return pd.DataFrame(X, index=pd.Index([f"S{i}" for i in range(1, 7)], name="text"),
                        columns=cv.get_feature_names_out())


def toy_idf() -> pd.DataFrame:
    """Step 3 on the six sentences: for each word, in how many sentences it
    occurs (df) and its idf = ln((1 + 6) / (1 + df)) + 1 (scikit-learn's
    formula); lowest idf first, ties alphabetical."""
    bag = toy_bag_of_words()
    df = (bag > 0).sum()
    idf = np.log((1 + len(bag)) / (1 + df)) + 1
    t = pd.DataFrame({"df": df, "idf": idf.round(2)})
    t = t.reset_index(names="word").sort_values(["idf", "word"], kind="mergesort")
    return t.set_index("word")


def weights_table(sentence=2) -> pd.DataFrame:
    """Steps 3-4 for one sentence (1-6 or "S2"): each word's tf (count in the
    sentence), df, idf, tf x idf and its weight after scaling the row to
    length 1 (largest first, ties alphabetical). Prints the row's length."""
    if isinstance(sentence, str) and sentence.strip().upper().startswith("S"):
        sentence = sentence.strip()[1:]
    k = _whole(sentence, 1, 6, 2, "sentence")
    tv, X = _tfidf(_S)
    words = tv.get_feature_names_out()
    idf = dict(zip(words, tv.idf_))
    bag = toy_bag_of_words()
    tf = bag.iloc[k - 1]
    tf = tf[tf > 0]
    df = (bag > 0).sum()
    raw = {w: tf[w] * idf[w] for w in tf.index}
    row = dict(zip(words, X[k - 1]))
    t = pd.DataFrame({"tf": tf, "df": df[tf.index], "idf": [idf[w] for w in tf.index],
                      "tf x idf": [raw[w] for w in tf.index],
                      "weight": [row[w] for w in tf.index]}, index=tf.index)
    t = t.reset_index(names="word")
    t["_k"] = -t["weight"].round(9)
    t = t.sort_values(["_k", "word"], kind="mergesort").drop(columns="_k").set_index("word")
    ss = sum(v ** 2 for v in raw.values())
    print(f"S{k}: '{_S[k - 1]}' - length of its row = √{ss:.2f} = {math.sqrt(ss):.2f}; "
          "weight = tf x idf / length")
    return t.round(3)


# ---------------------------------------------------------------------------
# E3 - Spread a word
# ---------------------------------------------------------------------------
E3_COUNTS = [1, 2, 3, 4, 5, 6]


def _spread(k: int) -> dict:
    """The six sentences with " twice" appended to the first k - 1 sentences
    of SPREAD_ORDER; idf of every word and S2's row (cached)."""
    def build():
        texts = list(_S)
        for j in SPREAD_ORDER[:k - 1]:
            texts[j - 1] = texts[j - 1] + " twice"
        tv, X = _tfidf(texts)
        words = tv.get_feature_names_out()
        dfs = {w: sum(w in s.split() for s in texts) for w in words}
        s2 = {w: X[1][i] for i, w in enumerate(words) if X[1][i] > 0}
        return {"texts": texts, "idf": dict(zip(words, tv.idf_)), "df": dfs, "s2": s2}
    return _cached(("e3", k), build)


def spread_numbers() -> pd.DataFrame:
    """For 1 to 6 sentences with 'twice': its idf, its weight in S2 and S2's
    other five weights."""
    rows = {}
    for k in E3_COUNTS:
        r = _spread(k)
        rows[k] = {"idf of 'twice'": r["idf"]["twice"], "weight of 'twice' in S2": r["s2"]["twice"],
                   **{f"weight of '{w}' in S2": r["s2"][w]
                      for w in sorted(r["s2"]) if w != "twice"}}
    return pd.DataFrame(rows).T.infer_objects().rename_axis("sentences with 'twice'")


def _sorted_weights(d: dict) -> list:
    return sorted(d.items(), key=lambda kv: (-round(kv[1], 9), kv[0]))


def draw_idf_curve(ax, in_sentences: int = 2) -> dict:
    """Each word's idf against the number of sentences that contain it: the
    formula ln(7 / (1 + df)) + 1 as a thin grey line, a grey dot per df that
    has words ('my', 'the', 'card' named), 'twice' as a red dot."""
    k = _whole(in_sentences, 1, 6, 2, "in_sentences")
    r = _spread(k)
    grid = np.linspace(1, 6, 101)
    ax.plot(grid, np.log(7 / (1 + grid)) + 1, color="#B8BEC5", lw=1, zorder=1)
    others = {w: d for w, d in r["df"].items() if w != "twice"}
    dfs = sorted(set(others.values()))
    ax.scatter(dfs, [np.log(7 / (1 + d)) + 1 for d in dfs], s=34, color=GRAY, zorder=2,
               linewidths=0)
    for w in ("my", "the", "card"):
        d = others[w]
        ax.annotate(w, (d, r["idf"][w]), xytext=(-5, -5), textcoords="offset points",
                    ha="right", va="top", fontsize=9, color=MUTED)
    idf = r["idf"]["twice"]
    ax.scatter([k], [idf], s=60, color=RED, zorder=4, linewidths=0)
    ax.text(0.97, 0.95, f"'twice': idf {idf:.2f}", ha="right", va="top", color=RED,
            fontsize=9.5, transform=ax.transAxes)
    ax.set_xlim(0.6, 6.4)
    ax.set_xticks(range(1, 7))
    ax.set_ylim(0.7, 2.45)
    ax.set_xlabel("sentences that contain the word (df)")
    ax.set_ylabel("idf")
    ax.set_title("idf of each word")
    return {"in_sentences": k, "idf of twice": idf,
            "idf": {w: r["idf"][w] for w in ("my", "the", "card")}}


def draw_s2_weights(ax, in_sentences: int = 2, title=None) -> pd.Series:
    """S2's six weights (its TF-IDF row) as bars, largest on top (ties
    alphabetical); 'twice' red, the others blue; the value at each bar's end."""
    k = _whole(in_sentences, 1, 6, 2, "in_sentences")
    r = _spread(k)
    items = _sorted_weights(r["s2"])
    ys = np.arange(len(items))[::-1]
    vals = [v for _, v in items]
    ax.barh(ys, vals, height=0.68, color=[RED if w == "twice" else BLUE for w, _ in items])
    for y, (w, v) in zip(ys, items):
        ax.text(v + 0.012, y, f"{v:.2f}", ha="left", va="center", fontsize=9,
                color=RED if w == "twice" else NEAR_BLACK)
    ax.set_yticks(ys)
    ax.set_yticklabels([w for w, _ in items])
    for lbl, (w, _) in zip(ax.get_yticklabels(), items):
        if w == "twice":
            lbl.set_color(RED)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 0.68)
    ax.set_xticks([0, 0.2, 0.4, 0.6])
    ax.set_xlabel("weight in S2")
    if title is None:
        idf = r["idf"]["twice"]
        title = (f"'twice' in 1 sentence: idf {idf:.2f}" if k == 1 else
                 f"'twice' in all 6: idf {idf:.2f}" if k == 6 else
                 f"'twice' in {k} sentences: idf {idf:.2f}")
    ax.set_title(title)
    return pd.Series(dict(items), name=f"S2, 'twice' in {k}")


def spread_word_figure(in_sentences: int = 2, figsize=None, suptitle: bool = True):
    k = _whole(in_sentences, 1, 6, 2, "in_sentences")
    r = _spread(k)
    with _style():
        fig, axes = _new_fig(figsize or (8.0, 3.4), 1, 2, width_ratios=[1.15, 1])
        draw_idf_curve(axes[0], k)
        draw_s2_weights(axes[1], k, title="S2's weights")
        if suptitle:
            fig.suptitle(f"'twice' in {k} of the 6 sentences: idf {r['idf']['twice']:.2f}; "
                         f"its weight in S2 {r['s2']['twice']:.2f}", fontsize=9.5)
    return fig


def spread_word(in_sentences: int = 2) -> None:
    """E3: the word 'twice' (only in S2) is added to the end of more and more
    of the other sentences; its idf (left) and S2's TF-IDF row (right)."""
    _show(spread_word_figure(_whole(in_sentences, 1, 6, 2, "in_sentences")))


# ---------------------------------------------------------------------------
# The classifier: the pipeline, out-of-fold probabilities, copies
# ---------------------------------------------------------------------------
def _pipe(**lr):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    return make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=2000, **lr))


def _model():
    """The pipeline fitted on the 840 training texts (cached)."""
    def build():
        tr, _ = _split()
        return _pipe().fit(tr["text"], tr["category"])
    return _cached("model", build)


def _test() -> dict:
    """The test texts: probabilities, labels, top probability, right / wrong,
    copy of a training text (cached)."""
    def build():
        tr, te = _split()
        m = _model()
        proba = m.predict_proba(te["text"])
        label = m.classes_[proba.argmax(axis=1)]
        y = te["category"].to_numpy()
        return {"proba": proba, "label": label, "top": proba.max(axis=1), "y": y,
                "right": label == y, "copy": te["text"].isin(set(tr["text"])).to_numpy()}
    return _cached("test", build)


def _cv():
    from sklearn.model_selection import StratifiedKFold
    return StratifiedKFold(5, shuffle=True, random_state=SEED)


def oof_proba() -> pd.DataFrame:
    """The out-of-fold probabilities of the 840 training texts: each text
    scored by the pipeline fitted on the other four folds
    (StratifiedKFold(5, shuffle=True, random_state=42)). One column per topic
    (alphabetical); the index of the training rows. Cached; a copy."""
    def build():
        from sklearn.model_selection import cross_val_predict
        tr, _ = _split()
        p = cross_val_predict(_pipe(), tr["text"], tr["category"], cv=_cv(),
                              method="predict_proba")
        return pd.DataFrame(p, index=tr.index, columns=TOPICS)
    return _cached("oof", build).copy()


def copy_in_other_fold() -> pd.Series:
    """True for a training text that has a word-for-word copy in another fold
    of the same StratifiedKFold (373 of the 840)."""
    def build():
        tr, _ = _split()
        text = tr["text"].reset_index(drop=True)
        mask = np.zeros(len(tr), bool)
        for fit_idx, held_idx in _cv().split(text, tr["category"]):
            mask[held_idx] = text.iloc[held_idx].isin(set(text.iloc[fit_idx])).to_numpy()
        return pd.Series(mask, index=tr.index, name="copy in another fold")
    return _cached("oof_copy", build).copy()


def _oof() -> dict:
    def build():
        tr, _ = _split()
        p = oof_proba().to_numpy()
        label = np.array(TOPICS)[p.argmax(axis=1)]
        return {"top": p.max(axis=1), "right": label == tr["category"].to_numpy()}
    return _cached("oof_summary", build)


def _asides():
    """For each complaint, the topic of the aside mixed into it (None if
    there is none): the loader's steps replayed, checked against its texts.
    None if the loader has changed."""
    def build():
        fd = _finance_data
        try:
            seed = fd.BASKET_SEED + 1
            rng = np.random.default_rng(seed)
            rng_seam = np.random.default_rng(seed + 7)
            rows = []
            for cat in fd.COMPLAINT_CATEGORIES:
                temps = fd._COMPLAINT_TEMPLATES[cat]
                for _ in range(240):
                    text = fd._fill(temps[rng.integers(len(temps))], rng)
                    if rng.random() < 0.5:
                        text += " " + fd._FILLERS[rng.integers(len(fd._FILLERS))]
                    aside_cat = None
                    if rng.random() < 0.25:
                        other = fd.COMPLAINT_CATEGORIES[rng.integers(
                            len(fd.COMPLAINT_CATEGORIES))]
                        if other != cat:
                            aside_cat = other
                            aside = fd._fill(fd._COMPLAINT_TEMPLATES[other][
                                rng.integers(len(fd._COMPLAINT_TEMPLATES[other]))], rng)
                            if rng_seam.random() < 0.5:
                                conn = fd._ASIDE_AFTER[rng_seam.integers(len(fd._ASIDE_AFTER))]
                                text = " ".join(t for t in (text, conn, aside) if t)
                            else:
                                conn = fd._ASIDE_BEFORE[rng_seam.integers(
                                    len(fd._ASIDE_BEFORE))]
                                text = " ".join(t for t in (aside, conn, text) if t)
                    rows.append({"text": text, "category": cat, "aside": aside_cat})
            out = pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)
            ref = _cached("complaints", load_complaints)
            if not (out["text"].equals(ref["text"]) and out["category"].equals(ref["category"])):
                return None
            return out["aside"]
        except Exception:
            return None
    return _cached("asides", build)


# ---------------------------------------------------------------------------
# Demo 4 - our complaints as numbers
# ---------------------------------------------------------------------------
_IDF_WORDS = ["the", "my", "and", "account", "app", "fee", "card", "mortgage", "bank",
              "collector", "skimmed", "casino"]


def _corpus() -> dict:
    """The vocabulary of all 1,200 complaints: counts, df, idf (cached)."""
    def build():
        from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
        c = _cached("complaints", load_complaints)
        cv = CountVectorizer()
        X = cv.fit_transform(c["text"])
        words = cv.get_feature_names_out()
        tv = TfidfVectorizer().fit(c["text"])
        return {"words": words, "count": np.asarray(X.sum(axis=0)).ravel(),
                "df": np.asarray((X > 0).sum(axis=0)).ravel(),
                "idf": pd.Series(tv.idf_, index=tv.get_feature_names_out()),
                "zeros": 1 - X.nnz / np.prod(X.shape), "per_text": X.nnz / X.shape[0]}
    return _cached("corpus", build)


def draw_idf_words(ax) -> pd.DataFrame:
    """The idf of twelve words of our 1,200 complaints (lowest at the top,
    ties alphabetical), each labelled with the number of texts that contain it."""
    k = _corpus()
    df = pd.Series(k["df"], index=k["words"])
    t = pd.DataFrame({"texts": df[_IDF_WORDS], "idf": k["idf"][_IDF_WORDS]})
    t = t.reset_index(names="word")
    t["_k"] = t["idf"].round(6)
    t = t.sort_values(["_k", "word"], kind="mergesort").drop(columns="_k").set_index("word")
    ys = np.arange(len(t))[::-1]
    ax.barh(ys, t["idf"], height=0.66, color=BLUE)
    for y, (v, n) in zip(ys, zip(t["idf"], t["texts"])):
        ax.text(v + 0.08, y, f"{v:.2f}  ({n:,} texts)", ha="left", va="center", fontsize=8.5)
    ax.set_yticks(ys)
    ax.set_yticklabels(t.index)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.6, len(t) - 0.4)
    ax.set_xlim(0, 6.9)
    ax.xaxis.set_visible(False)        # every bar carries its idf
    ax.spines["bottom"].set_visible(False)
    return t


def idf_ours() -> pd.DataFrame:
    """Demo 4: the four steps on our 1,200 complaints (fitted on all of them,
    to describe the data): the idf of twelve words and the number of texts
    that contain each."""
    k = _corpus()
    print(f"{len(k['words'])} different words; a complaint uses {k['per_text']:.1f} different "
          f"words on average, so {100 * k['zeros']:.1f}% of the table's cells are 0.")
    with _style():
        fig, ax = _new_fig((5.6, 3.4))
        t = draw_idf_words(ax)
        ax.set_title("idf of twelve words of our 1,200 complaints")
    _show(fig)
    return t


def _row_of(text: str) -> pd.Series:
    tv = _model().named_steps["tfidfvectorizer"]
    row = pd.Series(tv.transform([text]).toarray().ravel(), index=tv.get_feature_names_out())
    row = row[row > 0]
    order = sorted(row.items(), key=lambda kv: (-round(kv[1], 9), kv[0]))
    return pd.Series(dict(order), name="weight")


def draw_text_row(ax, text: str = EXAMPLE_TEXT) -> pd.Series:
    """One complaint's row of the TF-IDF table (the vectoriser fitted on the
    840 training texts): its non-zero weights, largest at the top."""
    row = _row_of(str(text))
    ys = np.arange(len(row))[::-1]
    ax.barh(ys, row.to_numpy(), height=0.66, color=BLUE)
    for y, v in zip(ys, row.to_numpy()):
        ax.text(v + 0.006, y, f"{v:.2f}", ha="left", va="center", fontsize=8.5)
    ax.set_yticks(ys)
    ax.set_yticklabels(row.index, fontsize=8.5)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.6, len(row) - 0.4)
    ax.set_xlim(0, max(0.45, row.max() + 0.08))
    ax.xaxis.set_visible(False)        # every bar carries its weight
    ax.spines["bottom"].set_visible(False)
    return row


def text_row(text: str = EXAMPLE_TEXT) -> pd.Series:
    """Demo 4: a text as the classifier sees it - its row of the TF-IDF table
    (vectoriser fitted on the training texts): the words that are not 0."""
    row = _row_of(str(text))
    n = len(_model().named_steps["tfidfvectorizer"].vocabulary_)
    print(f"'{text}'")
    print(f"{len(row)} of the {n} numbers are not 0.")
    with _style():
        fig, ax = _new_fig((5.0, max(2.4, 0.22 * len(row) + 0.9)))
        draw_text_row(ax, text)
        ax.set_title("its non-zero weights")
    _show(fig)
    return row


# ---------------------------------------------------------------------------
# Demo 5 - the classifier, steps 1-4
# ---------------------------------------------------------------------------
def _topic_tick(name: str) -> str:
    return {"app_service": "app &\nservice", "card_fraud": "card\nfraud"}.get(name, _label(name))


def draw_five_probs(ax, text: str = EXAMPLE_TEXT) -> pd.Series:
    """The five probabilities of one text (topics alphabetical), the largest
    in red."""
    m = _model()
    p = pd.Series(m.predict_proba([str(text)])[0], index=m.classes_)
    top = p.idxmax()
    xs = np.arange(len(p))
    ax.bar(xs, p.to_numpy(), width=0.62, color=[RED if t == top else BLUE for t in p.index])
    for x, v in zip(xs, p.to_numpy()):
        ax.text(x, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels([_label(t) for t in p.index], rotation=30, ha="right",
                       rotation_mode="anchor", fontsize=9)
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0, 0.5, 1])
    ax.set_ylabel("probability")
    return p


def _confusion() -> pd.DataFrame:
    from sklearn.metrics import confusion_matrix
    t = _test()
    cm = confusion_matrix(t["y"], t["label"], labels=TOPICS)
    return pd.DataFrame(cm, index=pd.Index(TOPICS, name="actual"),
                        columns=pd.Index(TOPICS, name="predicted"))


def draw_confusion(ax) -> pd.DataFrame:
    """The 360 test texts: one row per actual topic, one column per predicted
    topic (alphabetical); the diagonal (right labels) dark, the errors light,
    every count written."""
    cm = _confusion()
    M = cm.to_numpy()
    k = len(M)
    off_max = max(1, max(M[i, j] for i in range(k) for j in range(k) if i != j))
    for i in range(k):
        for j in range(k):
            v = M[i, j]
            if i == j:
                fc, tc = DARK, "white"
            else:
                a = v / off_max
                fc = tuple(np.array([1, 1, 1]) * (1 - a) + np.array(
                    [0xBF, 0xD3, 0xE8]) / 255 * a) if v else "white"
                tc = NEAR_BLACK
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fc=fc, ec="white", lw=1.5))
            ax.text(j, i, str(v), ha="center", va="center", fontsize=9.5, color=tc,
                    fontweight="bold" if i == j else "normal")
    ax.set_xlim(-0.5, k - 0.5)
    ax.set_ylim(k - 0.5, -0.5)
    ax.set_xticks(range(k))
    ax.set_xticklabels([_label(t) for t in TOPICS], rotation=30, ha="right",
                       rotation_mode="anchor", fontsize=9)
    ax.set_yticks(range(k))
    ax.set_yticklabels([_label(t) for t in TOPICS])
    ax.tick_params(length=0)
    ax.set_xlabel("predicted topic")
    ax.set_ylabel("actual topic")
    for s in ax.spines.values():
        s.set_visible(False)
    return cm


def _per_topic() -> pd.DataFrame:
    cm = _confusion().to_numpy()
    return pd.DataFrame({"texts": cm.sum(axis=1), "right": np.diag(cm),
                         "given the label": cm.sum(axis=0),
                         "recall": np.diag(cm) / cm.sum(axis=1),
                         "precision": np.diag(cm) / cm.sum(axis=0)},
                        index=pd.Index(TOPICS, name="topic"))


def _wrong_examples(n: int = 3) -> pd.DataFrame:
    """The wrong test texts shown by classifier_report: MIXED_TEXT first, then
    the shortest other wrong texts (fewest words; ties by the text)."""
    _, te = _split()
    t = _test()
    rows = pd.DataFrame({"text": te["text"].to_numpy(), "actual": t["y"], "label": t["label"],
                         "top": t["top"], "right": t["right"]}, index=te.index)
    wrong = rows[~rows["right"]].drop_duplicates("text")
    wrong = wrong.assign(words=wrong["text"].str.split().str.len())
    first = wrong[wrong["text"] == MIXED_TEXT]
    rest = wrong[wrong["text"] != MIXED_TEXT].sort_values(["words", "text"], kind="mergesort")
    return pd.concat([first, rest]).head(n)


def classifier_report() -> pd.DataFrame:
    """Demo 5: steps 1-4 of the classifier. Prints the split, the example's
    five probabilities and the accuracy, draws the confusion matrix, prints
    where the errors come from and three wrong texts; returns each topic's
    recall and precision."""
    tr, te = _split()
    m = _model()
    t = _test()
    n_words = len(m.named_steps["tfidfvectorizer"].vocabulary_)
    print(f"Step 1: {len(tr)} training texts ({len(tr) // 5} per topic), {len(te)} test texts "
          f"({len(te) // 5} per topic).")
    print(f"Step 2: the vectoriser, fitted on the training texts only, keeps {n_words} words.")
    p = pd.Series(m.predict_proba([EXAMPLE_TEXT])[0], index=m.classes_).sort_values(
        ascending=False, kind="mergesort")
    print(f"Step 3: '{EXAMPLE_TEXT}'\n        " +
          ", ".join(f"{_label(k)} {v:.2f}" for k, v in p.items()))
    acc = t["right"].mean()
    print(f"Step 4: test accuracy {acc:.3f} ({int(t['right'].sum())} of {len(te)}); always "
          f"guessing one topic: {1 / len(TOPICS):.3f}.")
    with _style():
        fig, ax = _new_fig((5.4, 4.0))
        draw_confusion(ax)
    _show(fig)
    asides = _asides()
    if asides is not None:
        a = asides.loc[te.index].to_numpy()
        has = pd.notna(a)
        wrong = ~t["right"]
        same = np.array([lab == x for lab, x in zip(t["label"], a)])
        print(f"Of the {int(has.sum())} test texts with an aside about another topic, "
              f"{int((wrong & has).sum())} got a wrong label"
              + (", each time the aside's topic" if (same[wrong]).all() else "")
              + f"; of the {int((~has).sum())} without one, "
              + (f"{int((wrong & ~has).sum())}." if (wrong & ~has).any() else "none."))
    print(f"Wrong labels have low top probabilities: median {np.median(t['top'][~t['right']]):.2f}, "
          f"against {np.median(t['top'][t['right']]):.2f} for right ones.")
    print("Three wrong labels:")
    for _, r in _wrong_examples(3).iterrows():
        print(f"  actual {_label(r['actual'])}, labelled {_label(r['label'])} (top probability "
              f"{r['top']:.2f}):\n    '{r['text']}'")
    return _per_topic()[["texts", "given the label", "recall", "precision"]].round(3)


# ---------------------------------------------------------------------------
# Demo 6 - a rare class
# ---------------------------------------------------------------------------
def _rare(topic: str) -> dict:
    """The topic rare in training AND test (1 in 10 of its texts kept in each,
    sample(frac=0.1, random_state=42)); refitted without and with class
    weights (cached)."""
    def build():
        from sklearn.metrics import accuracy_score
        tr, te = _split()
        tr2 = pd.concat([tr[tr["category"] != topic],
                         tr[tr["category"] == topic].sample(frac=0.1, random_state=SEED)])
        te2 = pd.concat([te[te["category"] != topic],
                         te[te["category"] == topic].sample(frac=0.1, random_state=SEED)])
        out = {"train": len(tr2), "train_topic": int((tr2["category"] == topic).sum()),
               "test": len(te2), "test_topic": int((te2["category"] == topic).sum())}
        y = te2["category"].to_numpy()
        for name, kw in (("rare", {}), ("weighted", {"class_weight": "balanced"})):
            pr = _pipe(**kw).fit(tr2["text"], tr2["category"]).predict(te2["text"])
            hit = int(((y == topic) & (pr == topic)).sum())
            got = pd.Series(pr[y == topic]).value_counts()
            got = got.reset_index().set_axis(["label", "n"], axis=1).sort_values(
                ["n", "label"], ascending=[False, True], kind="mergesort")
            out[name] = {"accuracy": accuracy_score(y, pr), "hit": hit,
                         "recall": hit / out["test_topic"],
                         "labels": dict(zip(got["label"], got["n"]))}
        t = _test()
        out["balanced"] = {"accuracy": float(t["right"].mean()),
                           "recall": float(_per_topic().loc[topic, "recall"]),
                           "hit": int(_per_topic().loc[topic, "right"]), "n": 72}
        return out
    return _cached(("rare", topic), build)


def draw_rare_topic(ax, topic: str = "fees") -> dict:
    """Two groups of two bars, the accuracy and the topic's recall: all five
    topics equally common (light) and the topic rare in training and test (dark)."""
    topic = _choice(topic, TOPICS, "topic", {"app": "app_service", "fraud": "card_fraud",
                                             "collection": "collections", "fee": "fees"})
    r = _rare(topic)
    vals = [(r["balanced"]["accuracy"], r["rare"]["accuracy"]),
            (r["balanced"]["recall"], r["rare"]["recall"])]
    for g, (a, b) in enumerate(vals):
        ax.bar(g - 0.19, a, width=0.36, color=LIGHT_BLUE,
               label="all five equally common" if g == 0 else None)
        ax.bar(g + 0.19, b, width=0.36, color=DARK,
               label=f"{_label(topic)} rare: 1 in 10 kept" if g == 0 else None)
        ax.text(g - 0.19, a + 0.02, f"{a:.2f}", ha="center", va="bottom", fontsize=9)
        ax.text(g + 0.19, b + 0.02, f"{b:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["accuracy", f"{_label(topic)} recall"])
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(0, 1.5)                # the legend stands above the bars, inside the panel
    ax.set_yticks([0, 0.5, 1])
    ax.spines["left"].set_bounds(0, 1)
    ax.legend(loc="upper right", ncol=1, handlelength=1.2, borderaxespad=0.1)
    return {"topic": topic, "accuracy balanced": vals[0][0], "accuracy rare": vals[0][1],
            "recall balanced": vals[1][0], "recall rare": vals[1][1],
            "rare: hit": r["rare"]["hit"], "rare: test texts of the topic": r["test_topic"],
            "rare: labels": r["rare"]["labels"]}


def rare_topic(topic: str = "fees") -> pd.DataFrame:
    """Demo 6: suppose one topic were rare - keep 1 in 10 of its texts in
    training and in test, train again. Accuracy and the topic's recall against
    the model with all five topics equally common; then with class weights."""
    topic = _choice(topic, TOPICS, "topic", {"app": "app_service", "fraud": "card_fraud",
                                             "collection": "collections", "fee": "fees"})
    r = _rare(topic)
    name = _label(topic)
    print(f"{name} rare: {r['train_topic']} of {r['train']} training texts, {r['test_topic']} of "
          f"{r['test']} test texts.")
    print(f"Accuracy {r['balanced']['accuracy']:.3f} -> {r['rare']['accuracy']:.3f}; {name} "
          f"recall {r['balanced']['recall']:.2f} -> {r['rare']['hit']} of {r['test_topic']} "
          f"({r['rare']['recall']:.2f}).")
    print(f"Its {r['test_topic']} test texts are labelled: " +
          ", ".join(f"{_label(k)} {v}" for k, v in r["rare"]["labels"].items()) + ".")
    print(f"With class_weight=\"balanced\" (each topic the same total weight in training): "
          f"{name} recall {r['weighted']['hit']} of {r['test_topic']}, accuracy "
          f"{r['weighted']['accuracy']:.3f}.")
    with _style():
        fig, ax = _new_fig((4.6, 3.1))
        draw_rare_topic(ax, topic)
        ax.set_title(f"{name} rare in training and test")
    _show(fig)
    return pd.DataFrame(
        {"training texts": [840, r["train"]], "test texts": [360, r["test"]],
         "accuracy": [round(r["balanced"]["accuracy"], 3), round(r["rare"]["accuracy"], 3)],
         f"{name} recall": [f"{r['balanced']['hit']} of 72", f"{r['rare']['hit']} of "
                                                              f"{r['test_topic']}"]},
        index=pd.Index(["all five equally common", f"{name} rare"], name="model"))


# ---------------------------------------------------------------------------
# E4 - Set the confidence bar
# ---------------------------------------------------------------------------
E4_BARS = (0.30, 0.90, 0.05)
E4_BAR_LIST = [round(0.30 + 0.05 * i, 2) for i in range(13)]
_E4_TRAIL: list = []


def _bar_value(bar) -> float:
    value = _decimal_comma(_plain(bar), "bar")
    v = _as_number(value)
    if not _is_number(v):
        _say(f"bar {value!r} is not a number: showing bar = 0.40.")
        return 0.40
    v = float(v)
    if 10 <= v <= 100:
        _say(f"bar {v:g} read as {v:g}% = {v / 100:g}.")
        v = v / 100
    if not 0.30 <= round(v, 2) <= 0.90:
        new = min(max(v, 0.30), 0.90)
        _say(f"bar {v:g} is outside 0.30-0.90: showing bar = {new:.2f}.")
        v = new
    return round(v, 2)


def _bar_row(bar: float, top: np.ndarray, right: np.ndarray) -> dict:
    auto = top >= bar
    n_auto = int(auto.sum())
    wrong = int((auto & ~right).sum())
    return {"labelled automatically": n_auto, "labelled automatically (%)": 100 * auto.mean(),
            "wrong": wrong, "wrong among them (%)": 100 * wrong / n_auto if n_auto else 0.0,
            "to a person (%)": 100 * (1 - auto.mean()),
            "wrong labels caught": int((~auto & ~right).sum()),
            "at most 2 in 100 wrong": (100 * wrong / n_auto if n_auto else 0.0) <= 2.0}


def bar_numbers() -> pd.DataFrame:
    """The 13 bars 0.30-0.90 on the out-of-fold probabilities of the 840
    training texts: labelled automatically, wrong among them, to a person,
    wrong labels caught (of 97)."""
    o = _oof()
    return pd.DataFrame({b: _bar_row(b, o["top"], o["right"]) for b in E4_BAR_LIST}).T \
        .infer_objects() \
        .rename_axis("bar")


def draw_top_proba(ax, bar: float = 0.40) -> dict:
    """The top probability of the 840 out-of-fold texts, right labels (blue)
    and wrong labels (red) stacked, bins of 0.05; the bar dashed, the area left
    of it shaded: those texts go to a person."""
    bar = _bar_value(bar)
    o = _oof()
    bins = np.round(np.arange(0.20, 1.0001, 0.05), 2)
    r_counts, _ = np.histogram(o["top"][o["right"]], bins)
    w_counts, _ = np.histogram(o["top"][~o["right"]], bins)
    lefts = bins[:-1]
    top = 1.18 * (r_counts + w_counts).max()
    ax.axvspan(0.2, max(bar, 0.2), color=SHADE, lw=0, zorder=0)
    ax.bar(lefts, r_counts, width=0.05, align="edge", color=BLUE, edgecolor="white", lw=0.5,
           zorder=2)
    ax.bar(lefts, w_counts, width=0.05, align="edge", bottom=r_counts, color=RED,
           edgecolor="white", lw=0.5, zorder=2)
    ax.axvline(bar, ls="--", color=NEAR_BLACK, lw=1.1, zorder=3)
    if bar >= 0.25:
        ax.text(bar - 0.012, 0.97 * top, "to a person", rotation=90, ha="right", va="top",
                fontsize=8.5, color=MUTED, zorder=4)
    ax.set_xlim(0.2, 1.0)
    ax.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylim(0, top)
    ax.set_xlabel("top probability")
    ax.set_ylabel("texts")
    ax.set_title("right labels blue, wrong red")
    return {"bar": bar, **_bar_row(bar, o["top"], o["right"])}


def _label_points(ax, pts: list, labels: list, current: str) -> None:
    """Label the dots of draw_bar_tradeoff with their bars. Dots with no wrong
    label: labelled below, neighbours closer than 16 pt share one label
    ("0.65-0.80"). Other dots: those on top of each other share a label, which
    stands left of the dot (below and right of it under the 2-in-100 line,
    with its top at least 3 pt below that line)."""
    fig = ax.figure
    fig.canvas.draw()
    scale = 72 / fig.dpi
    disp = ax.transData.transform(np.array(pts, float)) * scale
    zero = [i for i, (_, y) in enumerate(pts) if y == 0]
    other = [i for i, (_, y) in enumerate(pts) if y != 0]
    groups = []
    for idx, near in ((sorted(zero, key=lambda i: pts[i][0]), 16.0),
                      (sorted(other, key=lambda i: pts[i][0]), 6.0)):
        for i in idx:
            g = groups[-1] if groups and groups[-1][0] is idx else None
            if g and np.hypot(*(disp[i] - disp[g[1][-1]])) < near:
                g[1].append(i)
            else:
                groups.append((idx, [i]))
    placed = []   # (x, y) in points of the labels left of their dots, highest first
    spots = []
    for idx, g in groups:
        x = float(np.mean([pts[i][0] for i in g]))
        y = float(np.mean([pts[i][1] for i in g]))
        names = sorted({labels[i] for i in g}, key=float)
        text = names[0] if len(names) == 1 else f"{names[0]}-{names[-1]}"
        spots.append((x, y, text, DARK if current in names else MUTED))
    for x, y, text, colour in sorted(spots, key=lambda s: -s[1]):
        if y == 0:
            xy, ha, va = (0, -6), "center", "top"
        elif y < 2:
            # below-right of the dot: its top 4 pt under the dot, which is
            # itself under the line at 2
            xy, ha, va = (6, -4), "left", "top"
        else:
            # a label left of its dot moves down until it is 11 pt below any
            # label above it that it would touch (the bars 0.30-0.45 sit close)
            px, py = ax.transData.transform((x, y)) * scale
            dy = 0.0
            for qx, qy in placed:
                if abs(qx - px) < 40 and py + dy > qy - 11:
                    dy = qy - 11 - py
            placed.append((px, py + dy))
            xy, ha, va = (-6, dy), "right", "center"
        ax.annotate(text, (x, y), xytext=xy, textcoords="offset points", ha=ha, va=va,
                    fontsize=8.5, color=colour, annotation_clip=False)


def draw_bar_tradeoff(ax, bar: float = 0.40, tried=None) -> pd.DataFrame:
    """One dot per bar tried: the share labelled automatically (across)
    against the share wrong among them (up), labelled with its bar; the
    current bar dark; dashed line at 2 in 100."""
    bar = _bar_value(bar)
    o = _oof()
    tried = sorted({_bar_value(b) for b in (tried or [])} | {bar})
    rows = pd.DataFrame({b: _bar_row(b, o["top"], o["right"]) for b in tried}).T
    xs = rows["labelled automatically (%)"].astype(float).to_numpy()
    ys = rows["wrong among them (%)"].astype(float).to_numpy()
    if len(tried) > 1:
        ax.plot(xs, ys, color="#C3C9CF", lw=1, zorder=1)
    colours = [DARK if b == bar else MID_BLUE for b in tried]
    ax.scatter(xs, ys, s=[46 if b == bar else 26 for b in tried], color=colours, zorder=3,
               linewidths=0, clip_on=False)
    ax.axhline(2, ls="--", color=RED, lw=1)
    ax.text(31, 2.3, "2 in 100", ha="left", va="bottom", fontsize=8.5, color=RED)
    ax.set_xlim(24, 103)
    ax.set_ylim(-3.4, 14.2)
    ax.set_xticks([40, 60, 80, 100])
    ax.set_yticks([0, 5, 10])
    ax.set_xlabel("labelled automatically (%)")
    ax.set_ylabel("wrong among them (%)")
    ax.set_title("one dot per confidence bar")
    _label_points(ax, list(zip(xs, ys)), [f"{b:.2f}" for b in tried], f"{bar:.2f}")
    return rows


def confidence_bar_figure(bar: float = 0.40, tried=None, figsize=None, suptitle: bool = True):
    bar = _bar_value(bar)
    with _style():
        fig, axes = _new_fig(figsize or (8.0, 3.5), 1, 2, width_ratios=[1.1, 1])
        r = draw_top_proba(axes[0], bar)
        draw_bar_tradeoff(axes[1], bar, tried if tried is not None else [bar])
        if suptitle:
            fig.suptitle(f"bar {bar:.2f}: {r['labelled automatically (%)']:.1f}% labelled "
                         f"automatically, {r['wrong among them (%)']:.1f}% of them wrong; "
                         f"{r['to a person (%)']:.1f}% to a person", fontsize=9.5)
    return fig


def confidence_bar(bar: float = 0.40) -> None:
    """E4: a text whose top probability (out-of-fold) reaches the bar is
    labelled automatically, the others go to a person; the bars you tried
    stay on the right (confidence_bar_reset() clears them)."""
    b = _bar_value(bar)
    if b not in _E4_TRAIL:
        _E4_TRAIL.append(b)
    _show(confidence_bar_figure(b, list(_E4_TRAIL)))


def confidence_bar_reset() -> None:
    _E4_TRAIL.clear()


def test_check(bar: float = BAR) -> pd.DataFrame:
    """Step 5 (slide 56): the bar chosen on the out-of-fold texts, checked
    once on the test texts - all 360, and the 174 that are not word-for-word
    copies of training texts."""
    bar = _bar_value(bar)
    o = _oof()
    ob = _bar_row(bar, o["top"], o["right"])
    print(f"Chosen on the 840 training texts (out-of-fold): bar {bar:.2f} labels "
          f"{ob['labelled automatically (%)']:.1f}% automatically, "
          f"{ob['wrong among them (%)']:.1f}% of them wrong.")
    t = _test()
    cols = {}
    for name, m in ((f"all {len(t['y'])} test texts", np.ones(len(t["y"]), bool)),
                    (f"the {int((~t['copy']).sum())} that are not copies", ~t["copy"])):
        r = _bar_row(bar, t["top"][m], t["right"][m])
        cols[name] = {"accuracy, no bar": f"{t['right'][m].mean():.2f} ({int(t['right'][m].sum())})",
                      f"labelled automatically at {bar:.2f}":
                          f"{r['labelled automatically']} ({r['labelled automatically (%)']:.1f}%)",
                      "wrong among them": f"{r['wrong']} ({r['wrong among them (%)']:.1f}%)",
                      "to a person": str(int(m.sum()) - r["labelled automatically"])}
    print(f"{int(t['copy'].sum())} of the {len(t['y'])} test texts are word-for-word copies of "
          "training texts.")
    return pd.DataFrame(cols)


# ---------------------------------------------------------------------------
# E5 - A topic the model never saw
# ---------------------------------------------------------------------------
E5_TOPICS = list(TOPICS)
_TOPIC_ALIASES = {"app": "app_service", "service": "app_service", "fraud": "card_fraud",
                  "card": "card_fraud", "collection": "collections", "fee": "fees",
                  "home": "mortgage", "loan": "mortgage"}


def _topic(value, what: str = "left_out") -> str:
    return _choice(value, TOPICS, what, _TOPIC_ALIASES)


def _e5(left_out: str) -> dict:
    """The pipeline refitted without one topic's training texts, then the 360
    test texts (cached)."""
    def build():
        tr, te = _split()
        keep = tr["category"] != left_out
        m = _pipe().fit(tr.loc[keep, "text"], tr.loc[keep, "category"])
        new = te[te["category"] == left_out]
        old = te[te["category"] != left_out]
        pn, po = m.predict_proba(new["text"]), m.predict_proba(old["text"])
        labels = pd.Series(m.classes_[pn.argmax(axis=1)]).value_counts()
        labels = labels.reset_index().set_axis(["label", "n"], axis=1).sort_values(
            ["n", "label"], ascending=[False, True], kind="mergesort")
        return {"model": m, "new_top": pn.max(axis=1), "old_top": po.max(axis=1),
                "labels": dict(zip(labels["label"], labels["n"])),
                "old_right": m.classes_[po.argmax(axis=1)] == old["category"].to_numpy(),
                "n_new": len(new), "n_old": len(old)}
    return _cached(("e5", left_out), build)


def new_topic_numbers() -> pd.DataFrame:
    """Per topic left out: the labels its 72 test texts get, the share below
    the bar 0.60 (its texts and the 288 familiar ones), the median top
    probabilities and the accuracy on the familiar texts."""
    rows = {}
    for tp in TOPICS:
        r = _e5(tp)
        rows[tp] = {"its texts labelled": ", ".join(f"{k} {v}" for k, v in r["labels"].items()),
                    "below the bar, its texts (%)": 100 * (r["new_top"] < BAR).mean(),
                    "below the bar, familiar (%)": 100 * (r["old_top"] < BAR).mean(),
                    "median top, its texts": float(np.median(r["new_top"])),
                    "median top, familiar": float(np.median(r["old_top"])),
                    "accuracy, familiar": float(r["old_right"].mean())}
    return pd.DataFrame(rows).T.infer_objects().rename_axis("left out")


def draw_new_topic_labels(ax, left_out: str = "card_fraud") -> dict:
    """The labels the left-out topic's 72 test texts get: the four known
    topics (by count, ties alphabetical), each count written."""
    left_out = _topic(left_out)
    r = _e5(left_out)
    items = list(r["labels"].items())
    for tp in TOPICS:
        if tp != left_out and tp not in r["labels"]:
            items.append((tp, 0))
    ys = np.arange(len(items))[::-1]
    ax.barh(ys, [v for _, v in items], height=0.62, color=BLUE)
    for y, (_, v) in zip(ys, items):
        ax.text(v + 1, y, str(v), ha="left", va="center", fontsize=9)
    ax.set_yticks(ys)
    ax.set_yticklabels([_label(k) for k, _ in items])
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 80)
    ax.set_xticks([0, 20, 40, 60])
    ax.set_xlabel("texts")
    ax.set_title(f"labels of the {r['n_new']} {_label(left_out)} texts")
    return dict(items)


def draw_new_topic_proba(ax, left_out: str = "card_fraud") -> dict:
    """The top probability of the familiar test texts (blue) and of the
    left-out topic's texts (red), each as a share of its own group, bins of
    0.05; the bar 0.60 dashed."""
    left_out = _topic(left_out)
    r = _e5(left_out)
    bins = np.round(np.arange(0.20, 1.0001, 0.05), 2)
    old, _ = np.histogram(r["old_top"], bins)
    new, _ = np.histogram(r["new_top"], bins)
    old = 100 * old / r["n_old"]
    new = 100 * new / r["n_new"]
    lefts = bins[:-1]
    ax.bar(lefts + 0.003, old, width=0.022, align="edge", color=BLUE,
           label=f"familiar ({r['n_old']})")
    ax.bar(lefts + 0.025, new, width=0.022, align="edge", color=RED,
           label=f"{_label(left_out)} ({r['n_new']})")
    # the bar line stops below the legend, which sits above the highest bar
    ax.axvline(BAR, ymax=0.68, ls="--", color=NEAR_BLACK, lw=1.1)
    ax.set_xlim(0.2, 1.0)
    ax.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylim(0, max(old.max(), new.max()) * 1.55)
    ax.set_xlabel("top probability")
    ax.set_ylabel("share of its group (%)")
    ax.legend(loc="upper left", handlelength=1.1, borderaxespad=0.2)
    ax.set_title("top probability, bar 0.60")
    return {"below the bar, its texts": float((r["new_top"] < BAR).mean()),
            "below the bar, familiar": float((r["old_top"] < BAR).mean()),
            "median top, its texts": float(np.median(r["new_top"])),
            "median top, familiar": float(np.median(r["old_top"]))}


def new_topic_figure(left_out: str = "card_fraud", figsize=None, suptitle: bool = True):
    left_out = _topic(left_out)
    r = _e5(left_out)
    with _style():
        fig, axes = _new_fig(figsize or (8.0, 3.4), 1, 2, width_ratios=[0.9, 1.2])
        # a wider gap between the panels, so that the left title stays clear
        # of the right panel's y-label on a short (slide) figure
        fig.get_layout_engine().set(wspace=0.15)
        draw_new_topic_labels(axes[0], left_out)
        draw_new_topic_proba(axes[1], left_out)
        if suptitle:
            fig.suptitle(f"{_label(left_out)} left out: all {r['n_new']} texts get an old "
                         f"label; {100 * (r['new_top'] < BAR).mean():.0f}% below the bar "
                         f"(familiar {100 * (r['old_top'] < BAR).mean():.0f}%)", fontsize=9.5)
    return fig


def new_topic(left_out: str = "card_fraud") -> None:
    """E5: the model trained again without one topic's training texts, then
    the 360 test texts, among them the 72 of the left-out topic: the labels
    they get (left) and their top probabilities against the bar 0.60 (right)."""
    _show(new_topic_figure(_topic(left_out)))


# ---------------------------------------------------------------------------
# Demo 7 - watch the share below the bar
# ---------------------------------------------------------------------------
WATCH_SHARES = [0.0, 0.05, 0.10, 0.20]


def _stream(share_new: float) -> dict:
    """The model of Experiment 5 without mortgage; a stream of the 288
    familiar test texts plus the first k mortgage test texts, k = share x 288
    / (1 - share), rounded."""
    def build():
        _, te = _split()
        r = _e5("mortgage")
        old = te[te["category"] != "mortgage"]
        new = te[te["category"] == "mortgage"]
        k = int(round(share_new * len(old) / (1 - share_new)))
        stream = pd.concat([old, new.head(k)])
        top = r["model"].predict_proba(stream["text"]).max(axis=1)
        return {"share": share_new, "new": k, "texts": len(stream),
                "below": float((top < BAR).mean())}
    return _cached(("stream", round(float(share_new), 4)), build)


def draw_watch(ax, current=None) -> pd.DataFrame:
    """The share of texts below the bar 0.60 as mortgage complaints arrive
    (0, 5, 10, 20% of the incoming texts), for the model trained without
    mortgage; dashed: its usual level, with no new topic."""
    shares = list(WATCH_SHARES)
    if current is not None and round(float(current), 4) not in [round(s, 4) for s in shares]:
        shares = sorted(shares + [float(current)])
    rows = pd.DataFrame([_stream(s) for s in shares])
    x = 100 * rows["share"].to_numpy()
    y = 100 * rows["below"].to_numpy()
    base = 100 * _stream(0.0)["below"]
    ax.axhline(base, ls="--", color=GRAY, lw=1)
    ax.text(21, base - 0.7, f"usual level for this model: {base:.1f}%", ha="right", va="top",
            fontsize=8.5, color=MUTED)
    ax.plot(x, y, "-", color=MID_BLUE, lw=1.3, zorder=2)
    cur = None if current is None else round(100 * float(current), 4)
    ax.scatter(x, y, s=[60 if (cur is not None and abs(v - cur) < 1e-6) else 34 for v in x],
               color=[DARK if (cur is None or abs(v - cur) < 1e-6) else MID_BLUE for v in x],
               zorder=3, linewidths=0)
    for xv, yv in zip(x, y):
        if xv == 0:
            continue   # its value is the usual level, written on the dashed line
        ax.annotate(f"{yv:.1f}%", (xv, yv), xytext=(-5, 4), textcoords="offset points",
                    ha="right", va="bottom", fontsize=8.5)
    ax.set_xlim(-1.5, 21.5)
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_ylim(min(base, y.min()) - 5, max(y.max(), base) + 4)
    ax.set_xlabel("new topic among incoming texts (%)")
    ax.set_ylabel("texts below the bar (%)")
    return rows.rename(columns={"share": "share of the new topic", "new": "new-topic texts",
                                "texts": "texts in the stream", "below": "share below the bar"})


def watch_stream(share_new: float = 0.10) -> pd.Series:
    """Demo 7: the model trained without mortgage labels a stream of the 288
    familiar test texts plus mortgage complaints (share_new of the stream,
    0 to 0.20); how many texts fall below the bar 0.60?"""
    s = _share(share_new, 0.0, 0.20, 0.10, "share_new")
    r = _stream(s)
    with _style():
        fig, ax = _new_fig((5.0, 3.1))
        draw_watch(ax, current=s)
        ax.set_title("the model trained without mortgage, bar 0.60")
    _show(fig)
    return pd.Series({"share of the new topic": s, "new-topic texts": r["new"],
                      "texts in the stream": r["texts"],
                      "below the bar (%)": round(100 * r["below"], 1),
                      "usual level (%)": round(100 * _stream(0.0)["below"], 1)}, dtype=object)


# ---------------------------------------------------------------------------
# Backup: a few words are everywhere
# ---------------------------------------------------------------------------
def draw_zipf(ax) -> dict:
    """The words of our 1,200 complaints by rank (most frequent first) against
    their count, both axes logarithmic; the fitted line and its slope."""
    from matplotlib.ticker import FuncFormatter
    k = _corpus()
    order = np.argsort(-k["count"], kind="stable")
    counts = k["count"][order]
    words = k["words"][order]
    ranks = np.arange(1, len(counts) + 1)
    slope, icpt = np.polyfit(np.log(ranks), np.log(counts), 1)
    ax.scatter(ranks, counts, s=12, color=BLUE, linewidths=0, zorder=3)
    fit = np.exp(icpt) * ranks.astype(float) ** slope
    ax.plot(ranks, fit, ls="--", color=GRAY, lw=1, zorder=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    plain = FuncFormatter(lambda v, _: f"{v:,.0f}")
    ax.xaxis.set_major_formatter(plain)
    ax.yaxis.set_major_formatter(plain)
    total = counts.sum()
    ax.text(0.03, 0.05, f"fitted slope ${slope:.2f}$\nthe 10 most frequent words: "
            f"{100 * counts[:10].sum() / total:.1f}% of all words", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=9, color=MUTED, linespacing=1.15)
    ax.set_xlabel("rank of the word (1 = most frequent)")
    ax.set_ylabel("times it occurs")
    return {"slope": float(slope), "words": len(counts), "tokens": int(total),
            "top 10 share": float(counts[:10].sum() / total),
            "top 50 share": float(counts[:50].sum() / total),
            "top 3": list(zip(words[:3], counts[:3].tolist()))}


# ---------------------------------------------------------------------------
# Homework A: the tables and the soft checks
# ---------------------------------------------------------------------------
HW_CLUB = pd.DataFrame(
    [[1, 1, 0, 0], [1, 1, 1, 0], [1, 0, 1, 0], [1, 1, 1, 0],
     [0, 1, 0, 1], [1, 0, 0, 1], [1, 1, 1, 0], [0, 0, 1, 0]],
    index=pd.Index(range(1, 9), name="member"),
    columns=["gym", "swimming", "yoga", "tennis"]).astype(bool)
HW_REVIEWS = ["the train was late again", "the seat was broken", "the train was clean",
              "the staff were kind"]
HW_CONFUSION = pd.DataFrame([[40, 6, 4], [5, 30, 5], [3, 2, 5]],
                            index=pd.Index(["room", "breakfast", "location"], name="actual"),
                            columns=pd.Index(["room", "breakfast", "location"],
                                             name="predicted"))
HW_TOP = pd.DataFrame(
    {"top probability": [0.95, 0.91, 0.88, 0.83, 0.72, 0.66, 0.58, 0.52, 0.45, 0.41],
     "label right?": [True, True, True, True, True, False, True, False, False, True]},
    index=pd.Index(range(1, 11), name="text"))

A9_MODEL_ANSWER = (
    "Each spa review gets one of the three old labels (room, breakfast or location): the "
    "classifier has no 'none of these'. Its accuracy on the old topics need not change, so "
    "nothing signals the problem - a silent failure. But spa reviews mostly get low top "
    "probabilities, so the share of reviews below the confidence bar rises. Watch that "
    "share, and the share of each label, every month; read the reviews below the bar, label "
    "some spa reviews, add the topic and train again.")


def _hw_levels() -> dict:
    lw = _levelwise(HW_CLUB, 3 / len(HW_CLUB))
    cols = list(HW_CLUB.columns)
    name = lambda c: tuple(cols[j] for j in c)   # noqa: E731
    st = lw["status"]
    return {"frequent": [cols[c[0]] for c, s in st.items() if len(c) == 1 and s == "kept"],
            "pairs_counted": [name(c) for c, s in st.items() if len(c) == 2 and s != "not counted"],
            "triples_counted": [name(c) for c, s in st.items()
                                if len(c) == 3 and s != "not counted"],
            "counted": sum(s != "not counted" for s in st.values()),
            "kept": sum(s == "kept" for s in st.values())}


def homework_answers(item: str) -> dict:
    """The worked numbers of a homework item A1-A9 (for the solutions)."""
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    item = str(item).strip().upper()
    if item == "A1":
        return {"penetration": {k: 100 * v for k, v in HW_CLUB.mean().items()}, **_hw_levels()}
    if item == "A2":
        n, a, b, ab = 2000, 400, 300, 180
        return {"support": 100 * ab / n, "confidence": 100 * ab / a, "penetration": 100 * b / n,
                "lift": (ab / a) / (b / n), "reverse_confidence": 100 * ab / b,
                "reverse_lift": (ab / b) / (a / n), "approach": a - ab}
    if item == "A3":
        n = 10000
        rows = {"drama => news": (4000, 3280, 0.81), "cooking => travel": (1250, 600, 0.15),
                "opera => ballet": (140, 98, 0.10), "sport => films": (3600, 1980, 0.60)}
        return {"support": {k: 100 * ab / n for k, (na, ab, pb) in rows.items()},
                "confidence": {k: 100 * ab / na for k, (na, ab, pb) in rows.items()},
                "lift": {k: (ab / na) / pb for k, (na, ab, pb) in rows.items()},
                "act_on": "cooking => travel", "approach": 1250 - 600}
    if item == "A4":
        cv = CountVectorizer().fit(HW_REVIEWS)
        X = pd.DataFrame(cv.transform(HW_REVIEWS).toarray(), columns=cv.get_feature_names_out(),
                         index=[f"R{i}" for i in range(1, 5)])
        return {"vocabulary": list(X.columns), "counts": X, "df": (X > 0).sum().to_dict()}
    if item == "A5":
        tv = TfidfVectorizer().fit(HW_REVIEWS)
        idf = dict(zip(tv.get_feature_names_out(), tv.idf_))
        words = ["the", "was", "train", "clean"]
        raw = {w: idf[w] for w in words}
        row = dict(zip(tv.get_feature_names_out(), tv.transform([HW_REVIEWS[2]]).toarray()[0]))
        return {"idf": raw, "sum_of_squares": sum(v ** 2 for v in raw.values()),
                "length": math.sqrt(sum(v ** 2 for v in raw.values())),
                "row": {w: row[w] for w in words}, "idf_one": "the", "removed": False}
    if item == "A6":
        v = set(CountVectorizer().fit(HW_REVIEWS[:2]).vocabulary_)
        return {"vocabulary": sorted(v),
                "r3_keeps": [w for w in HW_REVIEWS[2].split() if w in v],
                "r4_keeps": [w for w in HW_REVIEWS[3].split() if w in v]}
    if item == "A7":
        cm = HW_CONFUSION.to_numpy()
        names = list(HW_CONFUSION.index)
        rec = np.diag(cm) / cm.sum(axis=1)
        prec = np.diag(cm) / cm.sum(axis=0)
        return {"accuracy": np.trace(cm) / cm.sum(), "recall": dict(zip(names, rec)),
                "precision": dict(zip(names, prec)), "worst": names[int(np.argmin(rec))],
                "always_room": cm[0].sum() / cm.sum()}
    if item == "A8":
        top = HW_TOP["top probability"].to_numpy()
        right = HW_TOP["label right?"].to_numpy()
        bars = [0.5, 0.6, 0.7]
        lab = [int((top >= b).sum()) for b in bars]
        wrong = [int(((top >= b) & ~right).sum()) for b in bars]
        ok = [b for b, w in zip(bars, wrong) if w == 0]
        return {"labelled": lab, "wrong": wrong, "to_person": [10 - v for v in lab],
                "lowest_bar": min(ok)}
    if item == "A9":
        return {"model answer": A9_MODEL_ANSWER}
    raise ValueError(f"unknown homework item {item!r}: use A1-A9")


# (key, kind, what it is, hint when it is not right yet)
_CHECKS = {
    "A1": [("penetration", "percents", "the penetration of gym, swimming, yoga, tennis (in %)",
            "holders / 8 members, for each activity"),
           ("frequent", "words", "the frequent activities (held by at least 3 members)",
            "count the 1s in each column; keep the activities with 3 or more"),
           ("pairs_counted", "itemsets", "the pairs that must be counted",
            "only pairs of two frequent activities: a pair with an infrequent one is never "
            "counted (downward closure)"),
           ("triples_counted", "itemsets", "the itemsets of three that are counted",
            "only sets of three whose three pairs were all kept"),
           ("counted", "int", "how many itemsets are counted", "4 singles, then the pairs and "
            "sets of three that must be counted"),
           ("kept", "int", "how many itemsets are kept", "the counted ones held by 3 or more")],
    "A2": [("support", "percent", "the support of travel guide => phrasebook (%)",
            "customers with both / all 2,000"),
           ("confidence", "percent", "its confidence (%)", "customers with both / travel-guide "
            "buyers"),
           ("penetration", "percent", "the phrasebook's penetration (%)",
            "phrasebook buyers / all 2,000"),
           ("lift", "number", "the lift", "confidence / the phrasebook's penetration"),
           ("reverse_confidence", "percent", "the confidence of phrasebook => travel guide (%)",
            "customers with both / phrasebook buyers"),
           ("reverse_lift", "number", "its lift", "confidence / the travel guide's penetration"),
           ("approach", "int", "how many customers you would approach",
            "travel-guide buyers without a phrasebook")],
    "A3": [("support", "percents", "the support of the four rules (in %, in the order given)",
            "viewers of both / 10,000 subscribers"),
           ("confidence", "percents", "their confidence (in %)",
            "viewers of both / viewers of the left side"),
           ("lift", "numbers", "their lift", "confidence / the share who watch the right side"),
           ("act_on", "rule", "the rule you would act on (as text, e.g. \"drama => news\")",
            "frequent (at least 300 subscribers) and a lift well above 1"),
           ("approach", "int", "how many subscribers you would approach",
            "viewers of the left side who do not watch the right side")],
    "A4": [("vocabulary", "words", "the vocabulary (a list of words)",
            "every different word of the four reviews, each once"),
           ("df", "numbers", "in how many reviews each word occurs (a dict, word: count)",
            "count the reviews that contain the word, not how often it occurs"),
           ("counts", "table", "the count table (4 rows, one column per word, alphabetical)",
            "one row per review; each cell counts the word in that review")],
    "A5": [("idf", "numbers", "the idf of the, was, train, clean (a dict or a list)",
            "ln((1 + 4) / (1 + df)) + 1, with df = in how many reviews the word occurs"),
           ("length", "number", "the length of R3's row of tf x idf",
            "the square root of the sum of the four squared tf x idf values"),
           ("row", "numbers", "R3's row scaled to length 1 (the, was, train, clean)",
            "divide each tf x idf by the length"),
           ("idf_one", "word", "the word with idf 1", "the word that is in every review"),
           ("removed", "bool", "is that word removed (True / False)?",
            "idf lowers a weight; it removes no word")],
    "A6": [("vocabulary", "words", "the vocabulary fitted on R1 and R2",
            "the different words of R1 and R2 only"),
           ("r3_keeps", "words", "the words of R3 the vectoriser keeps",
            "the words of R3 that are in that vocabulary"),
           ("r4_keeps", "words", "the words of R4 it keeps",
            "the words of R4 that are in that vocabulary")],
    "A7": [("accuracy", "share", "the accuracy (0.80 or 80)", "the diagonal / all 100 reviews"),
           ("recall", "shares", "the recall of room, breakfast, location",
            "along a row: the diagonal cell / the row's total"),
           ("precision", "shares", "the precision of room, breakfast, location",
            "down a column: the diagonal cell / the column's total"),
           ("worst", "word", "the topic handled worst", "the lowest recall and precision")],
    "A8": [("labelled", "numbers", "how many are labelled automatically at 0.5, 0.6, 0.7",
            "texts whose top probability is at least the bar"),
           ("wrong", "numbers", "how many of them are wrong, at each bar",
            "among the texts at or above the bar, count the wrong labels"),
           ("to_person", "numbers", "how many go to a person, at each bar",
            "the texts below the bar"),
           ("lowest_bar", "number", "the lowest of the three bars with no wrong automatic label",
            "the first bar at which every text at or above it is right")],
}


def _close(x, y) -> bool:
    return abs(float(x) - float(y)) <= max(0.02, 0.005 * abs(float(y))) + 1e-9


def _close_percent(given, g, e) -> bool:
    """A percent g (given as typed: given) against the expected e: close, or
    a whole number within 0.5 of e (62 or 63 for 62.5, 1 for 0.98)."""
    return _close(g, e) or (float(given).is_integer() and abs(g - e) <= 0.5 + 1e-9)


def _norm_word(v) -> str:
    return " ".join(str(v).lower().replace("_", " ").split())


def _as_list(given, expected):
    """given as a list in the order of expected (a dict: by its keys)."""
    if isinstance(expected, dict):
        keys = list(expected)
        if isinstance(given, (dict, pd.Series)):
            g = {_key(k): v for k, v in dict(given).items()}
            return [g.get(_key(k)) for k in keys], [expected[k] for k in keys]
        return list(np.ravel(given)), [expected[k] for k in keys]
    if isinstance(given, (dict, pd.Series)):
        given = list(dict(given).values())
    return list(np.ravel(given)), list(np.ravel(expected))


def _itemset(v) -> frozenset:
    if isinstance(v, str):
        parts = re.split(r"\s*(?:\+|,|&|\band\b)\s*", v)
    else:
        parts = list(v)
    return frozenset(_norm_word(p) for p in parts if str(p).strip())


def _right(kind: str, given, expected) -> bool:
    if kind == "number":
        return _close(_as_number(given), expected)
    if kind == "percent":
        g0 = g = float(_as_number(given))
        if abs(g) <= 1.0001 and abs(float(expected)) > 1.0001:
            g = 100 * g
        return _close_percent(g0, g, float(expected))
    if kind == "share":   # a share: 0.75 or 75 (a percent) are both right
        g = float(_as_number(given))
        if abs(g) > 1.0001 and abs(float(expected)) <= 1.0001:
            g = g / 100
        return _close(g, expected)
    if kind == "shares":
        g, e = _as_list(given, expected)
        if any(v is None for v in g):
            return False
        g = [float(_as_number(v)) for v in g]
        e = [float(v) for v in e]
        if g and max(abs(v) for v in g) > 1.0001 and max(abs(v) for v in e) <= 1.0001:
            g = [v / 100 for v in g]
        return len(g) == len(e) and all(_close(a, b) for a, b in zip(g, e))
    if kind in ("numbers", "percents"):
        g, e = _as_list(given, expected)
        if any(v is None for v in g):
            return False
        g0 = g = [float(_as_number(v)) for v in g]
        e = [float(v) for v in e]
        if kind == "percents" and g and max(abs(v) for v in g) <= 1.0001 < max(abs(v) for v in e):
            g = [100 * v for v in g]
        if kind == "percents":
            return len(g) == len(e) and all(_close_percent(a0, a, b)
                                            for a0, a, b in zip(g0, g, e))
        return len(g) == len(e) and all(_close(a, b) for a, b in zip(g, e))
    if kind == "int":
        return float(_as_number(given)) == float(expected)
    if kind == "word":
        return _norm_word(given) == _norm_word(expected)
    if kind == "words":
        g = [given] if isinstance(given, str) else list(given)
        return sorted({_norm_word(v) for v in g}) == sorted({_norm_word(v) for v in expected})
    if kind == "itemsets":
        g = [given] if isinstance(given, str) else list(given)
        return {_itemset(v) for v in g} == {_itemset(v) for v in expected}
    if kind == "rule":
        parts = re.split(r"\s*(?:=>|⇒|->|=)\s*", str(given).strip().lower())
        if len(parts) == 1:
            return _key(parts[0]) == _key(str(expected).split("=>")[0])
        e = str(expected).split("=>")
        return len(parts) == 2 and _key(parts[0]) == _key(e[0]) and _key(parts[1]) == _key(e[1])
    if kind == "bool":
        words = {"yes": True, "true": True, "no": False, "false": False}
        g = words.get(str(given).strip().lower(), given) if isinstance(given, str) else given
        return bool(g) == bool(expected)
    if kind == "table":
        if isinstance(given, pd.DataFrame):
            g = given.reindex(columns=expected.columns).to_numpy(float)
        else:
            g = np.array(given, float)
        return g.shape == expected.shape and np.allclose(g, expected.to_numpy(float))
    return False


def check(item: str, **answers) -> None:
    """A soft check of a homework answer: one line per part, "right" or "not
    yet" with a hint. It never stops the notebook. check("A1") lists the
    parts it takes; leave a part as ... until you have it."""
    key = str(item).strip().upper()
    if key == "A9":
        _say("A9 is a written answer: compare yours with the model answer in the "
             "solutions (and with the slides on a topic the model never saw).")
        return
    if key not in _CHECKS:
        _say(f"{item!r} is not one of the items A1-A9.")
        return
    spec = _CHECKS[key]
    if not answers:
        _say(f"{key} takes: " + "; ".join(f"{k} = {what}" for k, _, what, _ in spec))
        return
    try:
        expected = homework_answers(key)
    except Exception as error:  # never stop the notebook
        _say(f"{key}: the check could not run ({error}).")
        return
    known = {k for k, *_ in spec}
    for k in answers:
        if k not in known:
            _say(f"{key}: {k!r} is not one of its parts ({', '.join(sorted(known))}).")
    n_right = n_given = 0
    for k, kind, what, hint in spec:
        if k not in answers:
            continue
        given = answers[k]
        if given is ... or given is None:
            _say(f"{key} {k}: not filled in yet ({what}).")
            continue
        n_given += 1
        try:
            ok = _right(kind, given, expected[k])
        except Exception:
            comma = (" - decimals take a dot: 0.09, not 0,09"
                     if isinstance(given, str) and "," in given else "")
            _say(f"{key} {k}: could not read {given!r} - it should be {what}{comma}.")
            continue
        n_right += ok
        if (not ok and kind in ("numbers", "percents", "shares")
                and isinstance(given, (list, tuple)) and isinstance(expected[k], (list, tuple, dict))
                and len(given) != len(expected[k])):
            hint += " (decimals take a dot: [0.80, 0.75], not [0,80, 0,75])"
        _say(f"{key} {k}: right." if ok else f"{key} {k}: not yet - {hint}.")
    if n_given:
        _say(f"{key}: {n_right} of {n_given} right.")
