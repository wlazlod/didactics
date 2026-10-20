"""block3_tools.py - the experiments of Block 3 (trees, ensembles, honest evaluation).

Data Mining, SGH Warsaw School of Economics (223121-0090).

The Block 3 notebook imports this file as

    import block3_tools as lab

Each experiment is ONE function with ONE control (the number you change), and
it draws ONE figure. The slides are drawn by the same functions, so the deck
shows exactly the pictures you make in the notebook. You do not need to read
this file; the notebook shows the few scikit-learn lines behind each
experiment ("The code behind it").

Everything is deterministic (seed 42) and uses the course split of Block 2:
the 12 columns of ``taiwan_features``, 70% training / 30% test, stratified,
``random_state=42`` (21,000 / 9,000 clients), and 5-fold cross-validation on
the training rows with ``StratifiedKFold(5, shuffle=True, random_state=42)``.
Test rows are read only by Experiment 2 (last week's model, already chosen),
by ``expert_decline_counts`` and by ``expert_tree_table(rows="test")``;
nothing here is chosen on them. Slow
results are computed once, with a line "Computing ... (about N s, only the
first time)", and kept in memory until the kernel restarts.

In the notebook (each slider cell has a plain twin: set the number, re-run)
---------------------------------------------------------------------------
    print(lab.sliders_status())                  "Sliders OK" or what to do instead
    lab.show_expert_tree(); lab.show_learned_tree()          opening (pictures)
    lab.expert_tree_table(); lab.learned_tree_table()        opening (numbers)
    D1  lab.split_search("pay_delay_recent", cut=1.5)
        lab.slider(lab.split_search, feature="age", cut=lab.cut_range("age"))
    E1  lab.grow_tree(3);            lab.slider(lab.grow_tree, depth=lab.E1_DEPTHS)
    D2  lab.units_check()
    E2  lab.trace_roc(0.5);          lab.slider(lab.trace_roc, threshold=lab.E2_THRESHOLDS)
    E3  lab.cv_depth(3);             lab.slider(lab.cv_depth, depth=lab.E3_DEPTHS)
    D3  lab.cv_leaf(20);             lab.slider(lab.cv_leaf, min_samples_leaf=lab.D3_LEAVES)
    E4  lab.crowd_and_relay(1);      lab.slider(lab.crowd_and_relay, n_trees=lab.E4_N)
    D4  lab.rare_positives(0.02);    lab.slider(lab.rare_positives, share=lab.D4_SHARES)
    E5  lab.bend_probabilities(2);   lab.slider(lab.bend_probabilities, bend=lab.E5_BENDS)
    D5  lab.prices(3, 1, bend=1);    lab.slider(lab.prices, miss_cost=[1, 2, 3, 5, 10],
                                                false_alarm_cost=[1, 2, 3], bend=[0.5, 1, 2])
    E6  lab.too_good("honest");      lab.slider(lab.too_good, setup=lab.SETUPS)

Public API
----------
Setup and sliders
    sliders_ok() -> bool         True if ipywidgets is installed and a Jupyter kernel runs.
    sliders_status() -> str      "Sliders OK", or the line that says to use the plain cells.
    version_note() -> str        "" from scikit-learn 1.8 on; for 1.4-1.7 a note that needs
                                 no action (third-decimal differences); below 1.4 a WARNING
                                 with the upgrade line.
    slider(func, **controls)     ipywidgets.interact with continuous_update=False (redraws
                                 when you let go). A control is (lo, hi) or (lo, hi, step)
                                 -> slider; a list of numbers (None shown as "no limit")
                                 -> slider over those values; a list of strings -> dropdown;
                                 anything else is passed as a fixed value. Without
                                 ipywidgets it prints how to use the plain cell instead.
    VERBOSE = True               False silences the "Computing ..." lines (make_figures.py).
    clear_cache()                Forget all cached results and the E2/E6 trails.

Data and models
    course_data() -> CourseData  .df (raw table incl. collections_contact), .X, .y, .X_train,
                                 .X_test, .y_train, .y_test, .folds
    course_split()               -> X_train, X_test, y_train, y_test
    folds()                      -> StratifiedKFold(5, shuffle=True, random_state=42)
    model(name)                  unfitted: "logistic" (StandardScaler + LogisticRegression(
                                 max_iter=2000)), "tree" (max_depth=5), "forest" (300 trees,
                                 min_samples_leaf=10), "boosting" (HistGradientBoosting:
                                 learning_rate=0.03, max_leaf_nodes=7, min_samples_leaf=60,
                                 max_iter=1000, early_stopping=True)
    crescents()                  -> X, y, X_new, y_new: make_moons(200, noise=0.3,
                                 random_state=42) and 20,000 new points (random_state=43)
    wave_truth(x)                the 1-D toy's true curve, y = sin(x) + 0.3 x (x from 0 to 6)
    wave_samples(k=20)           -> k samples (x, y) of 80 points, noise SD 0.45, seed 42 + i
    copies_data()                -> X_rows, y_rows, client: 6,000 training clients x 3
                                 identical rows (18,000 rows)
    impurity(share)              2 share (1 - share)

Opening: an expert's tree, then a learned tree
    expert_tree_table(rows="train")   leaves decline / manual review / approve: clients,
                                      defaulted, share of clients, share defaulted
    expert_decline_counts()           "decline" leaf alone on the test rows: TP, FP, FN,
                                      TN, precision, recall (read only)
    learned_tree_table(depth=2)       leaves of DecisionTreeClassifier(max_depth=depth)
    expert_tree_spec(), learned_tree_spec(depth=2)   dicts for draw_tree_diagram
    draw_tree_diagram(ax, spec, fontsize=9)          boxes and yes/no branches
    show_expert_tree(figsize), show_learned_tree(depth=2, figsize)

Figures: for every experiment X there are
    X(control)                          the notebook call: draws and shows, returns None
    X_figure(control, ..., figsize=None, suptitle=True) -> Figure (not shown, for slides)
    draw_*(ax, control, ...)            single panels, to compose other layouts
    X_numbers() / tables                the numbers behind the picture
All text is at least 8.5 pt and fits a slide 5 in wide.
The notebook calls never raise for a value typed in a plain cell: a value that
is not on the experiment's list is moved to the nearest one (None = no limit
stays None; 0 or below -> the smallest), a name that is not one of the choices
falls back to the starting one, and one line says what is shown instead.

D1 Try every cut (training rows)
    split_search(feature="pay_delay_recent", cut=None)    cut=None: the best cut
    split_search_figure(feature, cut=None, figsize=None, suptitle=True)
    draw_split_bars(ax, feature, cut=None) -> dict; draw_drop_curve(ax, feature, cut=None)
    draw_best_drops(ax, features=D1_FEATURES); best_cuts(features=None) -> DataFrame
    cut_range(feature) -> (lo, hi, step);  D1_FEATURES (the six of the slide)
E1 Grow a tree (crescents)
    grow_tree(depth=2)                 depth in E1_DEPTHS = 1..12 and None (no limit)
    grow_tree_figure(depth=2, figsize=None, suptitle=True)
    draw_tree_regions(ax, depth=2, points="training" | "new"); draw_tree_accuracy(ax, depth=2)
    grow_tree_numbers() -> DataFrame (levels used, leaves, training / new-point accuracy)
D2 Units do not matter to a tree
    units_check(n_show=5) -> DataFrame (prints the largest differences)
E2 Trace the ROC curve (last week's model, test rows, read only; decline when p > t)
    trace_roc(threshold=0.5)           keeps a trail of the thresholds tried; trace_roc_reset()
    trace_roc_figure(threshold=0.5, trail=None, figsize=None, suptitle=True)
    draw_confusion(ax, threshold) -> counts; draw_roc_trail(ax, threshold, trail=None)
    roc_points(thresholds) -> DataFrame (TP, FN, FP, TN, false-alarm rate, recall, precision)
    baseline_test_p() -> (p, y);  E2_THRESHOLDS = 0.95, 0.90, ..., 0.05
E3 Let cross-validation choose the depth;  D3 leaf size (no depth limit)
    cv_depth(depth=3)                  depth in E3_DEPTHS = 1..20 and None
    cv_depth_figure(depth=3, figsize=None, suptitle=True)
    draw_cv_curve(ax, depth); draw_cv_rounds(ax, depth, param="max_depth")
    cv_depth_numbers() -> DataFrame (training AUC, validation AUC, spread, rounds)
    cv_leaf(min_samples_leaf=1)        D3_LEAVES = 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000
    cv_leaf_figure(...); cv_leaf_numbers(); draw_cv_rounds(ax, leaf, "min_samples_leaf")
    logistic_cv() -> the logistic model's 5 validation AUCs
E4 Crowd and relay (1-D wave toy; 5 lines drawn, meters measured on 20 samples)
    crowd_and_relay(n_trees=1)         n_trees in E4_N = 1, 2, 5, 10, 30, 100, 300
    crowd_and_relay_figure(n_trees=1, figsize=None, suptitle=True)
    draw_crowd(ax, n_trees, lines=5, points=False, band=True)   average of n deep
                                       bootstrap trees; band=True adds the band mean +/- SD
                                       of the 20 samples' lines (spread) and their mean
                                       line (the middle; its distance from the truth = miss)
    draw_relay(ax, n_trees, lines=5, points=False, band=True)   n stumps added, step
                                       E4_STEP = 0.3; band as in draw_crowd
    draw_meters(ax, kind="crowd" | "relay", n_trees)  spread and miss bars (0-0.5)
    crowd_and_relay_numbers() -> DataFrame
Tournament and importance (the same five rounds)
    tournament_scores() -> DataFrame (5 rounds x logistic / tree / forest / boosting)
    tournament_summary() -> DataFrame (mean, spread = SD with ddof=0, rounds ahead)
    boosting_oof() -> (p, y): out-of-fold p of the 21,000 training clients
    importance(setup="honest", top=None) -> Series: AUC drop when one column is shuffled
                                 in round 1's validation rows (copies: forest, held-out 30%)
E5 Bend the probabilities (p -> p ** bend; the order is kept)
    bend_probabilities(bend=1)         bend in E5_BENDS = 0.5, 0.7, 1, 1.5, 2
    bend_figure(bend=1, figsize=None, suptitle=True)
    draw_reliability(ax, bend, notes=True); draw_bend_roc(ax, bend); draw_count(ax, bend)
    bend_numbers() -> DataFrame (AUC, predicted count = sum of p, actual count, mean gap)
    reliability_points(p, y, groups=10) -> DataFrame (equal-size groups sorted by p;
                                 p and y may be arrays, lists or pandas Series)
D4 Rare positives (every repayer kept, defaulters thinned; 20 random draws)
    rare_positives(share=0.22)         share in D4_SHARES = 0.22, 0.10, 0.05, 0.02
    rare_positives_figure(...); draw_rare_roc(ax, share); draw_rare_pr(ax, share)
    rare_positives_numbers() -> DataFrame (mean AUC, mean AP, coin AP)
D5 Pricing the threshold (out-of-fold p of boosting; decline when p > t)
    prices(miss_cost=3, false_alarm_cost=1, bend=1)
    prices_figure(...); draw_prices(ax, miss_cost, false_alarm_cost, bend) -> dict
    price_numbers(miss_cost, false_alarm_cost, bend) -> dict (cheapest t and cost,
                                 t* = false_alarm_cost / (false_alarm_cost + miss_cost),
                                 cost at t*, cost at 0.5, extra cost at t*)
E6 Too good to be true
    too_good(setup="honest")           setup in SETUPS = "honest", "+ collections_contact",
                                       "copies: split by row", "copies: split by client";
                                       keeps the setups tried; too_good_reset()
    too_good_figure(setup, tried=None, figsize=None, suptitle=True)
    draw_scoreboard(ax, tried, current=None); draw_importance(ax, setup, top=5)
    too_good_numbers() -> DataFrame; copies_one_row() -> AUC with one row per client

Colours (course palette): BLUE repaid / negative, RED default / positive and
held-out (validation, new) data, DARK, GRAY, LIGHT; tints LIGHT_RED, LIGHT_BLUE,
MID_BLUE, SHADE; text INK, MUTED.
"""
from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter, MaxNLocator, PercentFormatter

try:  # Colab: finance_data.py uploaded next to the notebook
    import finance_data as _finance_data
except ImportError:  # course folder: the loader lives in ../lecture_1
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lecture_1"))
    import finance_data as _finance_data
from finance_data import load_taiwan, taiwan_features

SEED = 42
VERBOSE = True

# ---------------------------------------------------------------------------
# Colours and style (the course palette, as in Block 2)
# ---------------------------------------------------------------------------
BLUE = "#2E6DA4"    # repaid / negative class
RED = "#C83C28"     # default / positive class; held-out (validation, new) data
DARK = "#003366"
GRAY = "#8C959E"
LIGHT = "#D7E3F0"
# derived tints (no new meanings)
LIGHT_RED = "#F2C9C2"
LIGHT_BLUE = "#BFD3E8"
MID_BLUE = "#7FA7CF"
INK = "#222222"
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
    "axes.labelcolor": INK,
    "xtick.color": "#6B7580",
    "ytick.color": "#6B7580",
    "xtick.labelcolor": INK,
    "ytick.labelcolor": INK,
    "text.color": INK,
    "axes.grid": False,
    "axes.unicode_minus": False,
    "legend.frameon": False,
    "figure.constrained_layout.h_pad": 0.03,
    "figure.constrained_layout.w_pad": 0.03,
    "figure.constrained_layout.hspace": 0.03,
    "figure.constrained_layout.wspace": 0.03,
}
PCT0 = PercentFormatter(1.0, decimals=0)

LABELS = {  # plain names (Block 2's), used on axes and bars
    "months_delayed_6m": "months late (of last 6)",
    "pay_delay_recent": "latest status",
    "limit_bal": "credit limit",
    "married": "married",
    "bill_trend": "bill trend",
    "age": "age",
    "edu_graduate": "graduate school",
    "pay_ratio": "payment / bill",
    "edu_high_school": "high school",
    "female": "female",
    "utilisation": "utilisation",
    "edu_other": "other education",
    "collections_contact": "collections_contact",
}
NO_LIMIT = "no limit"


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


def _depth_name(d) -> str:
    return NO_LIMIT if d is None else str(d)


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------
_CACHE: dict = {}


def _cached(key, about_s: float, what: str, fn):
    """fn() once per session; the first call prints how long it will take."""
    if key not in _CACHE:
        if VERBOSE and about_s >= 1:
            print(f"Computing {what} (about {about_s:.0f} s, only the first time) ...", flush=True)
        _CACHE[key] = fn()
    return _CACHE[key]


def clear_cache() -> None:
    """Forget every cached result and the trails of Experiments 2 and 6."""
    _CACHE.clear()
    _E2_TRAIL.clear()
    _E6_TRIED.clear()


# ---------------------------------------------------------------------------
# Sliders
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
            "slider cell there is a plain cell; change the number in it and re-run it.")


def version_note() -> str:
    """One line about the scikit-learn version, for the setup cell. The notebook
    was tested with 1.9; 1.4 (the course minimum) to 1.7 give the same results up
    to a few third decimals, so that needs no action."""
    import re

    import sklearn
    found = re.match(r"(\d+)\.(\d+)", sklearn.__version__)
    version = tuple(int(n) for n in found.groups()) if found else (0, 0)
    if version >= (1, 8):
        return ""
    if version >= (1, 4):
        return (f"Note: scikit-learn {sklearn.__version__} is older than 1.8, so a few "
                "numbers may differ from the slides in the third decimal - that is fine, "
                "nothing to do.")
    return (f"WARNING: scikit-learn {sklearn.__version__} is older than 1.4, the oldest "
            "version this notebook works with.\nFix: run  %pip install -q -U scikit-learn  "
            "in a new cell, then restart\n(JupyterLab: Kernel -> Restart Kernel; Colab: "
            "Runtime -> Restart session)\nand run the setup cell again.")


def _say(msg: str) -> None:
    """A one-line note to the student (never an error)."""
    print(msg, flush=True)


def _plain(value):
    """A numpy number (np.int64(5), np.float64(0.3)) as the plain Python number."""
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
    """value as a number if it is one (or a string that reads as one), "none"
    / "no limit" as None, anything else unchanged."""
    value = _plain(value)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, str):
        s = value.strip().lower()
        if s in ("none", "no limit", "nolimit", ""):
            return None
        try:
            return float(s)
        except ValueError:
            return value
    return value


def _snap(value, allowed: list, what: str, default, quiet: bool = False):
    """The allowed value nearest to `value` (never an error). None stays None
    if None is allowed; below the smallest -> the smallest; above the
    largest -> the largest; not a number -> `default`. Prints one line when
    the value had to change, e.g. "depth 15 is not on this experiment's list
    (1-12 or None): showing 12." Returns the value to use."""
    value = _decimal_comma(_plain(value), what)
    nums = [v for v in allowed if v is not None]
    has_none = None in allowed
    v = _as_number(value)
    if v is None and has_none:
        return None
    if isinstance(v, (int, float)) and not isinstance(v, bool) and np.isfinite(v):
        for a in nums:
            if v == a:
                return a
        new = min(nums, key=lambda a: (abs(a - v), a))
    else:
        new = default
    if len(nums) > 4 and all(isinstance(a, int) for a in nums) and \
            nums == list(range(nums[0], nums[-1] + 1)):
        listed = f"{nums[0]}-{nums[-1]}"
    elif len(nums) > 10:
        listed = f"{nums[0]:g}, {nums[1]:g}, ..., {nums[-1]:g}"
    else:
        listed = ", ".join(f"{a:g}" for a in nums)
    listed += " or None" if has_none else ""
    if quiet:
        return new
    shown = NO_LIMIT if new is None else f"{new:g}"
    _say(f"{what} {value!r} is not on this experiment's list ({listed}): showing {shown}."
         if not isinstance(value, (int, float)) or isinstance(value, bool) else
         f"{what} {value:g} is not on this experiment's list ({listed}): showing {shown}.")
    return new


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
    if isinstance(spec, list) and spec and all(isinstance(v, str) for v in spec):
        common.pop("continuous_update")
        return w.Dropdown(options=spec, value=default if default in spec else spec[0], **common)
    if isinstance(spec, list) and spec:
        options = [(NO_LIMIT if v is None else f"{v:g}", NO_LIMIT if v is None else v)
                   for v in spec]
        values = [v for _, v in options]
        d = NO_LIMIT if default is None else default
        return w.SelectionSlider(options=options, value=d if d in values else values[0], **common)
    return w.fixed(spec)


def slider(func, **controls) -> None:
    """Draw func's figure with a slider (or dropdown) for each control.

    Example: lab.slider(lab.grow_tree, depth=[1, 2, 3, 4, 5, 6, 8, 10, 12, None])
    The picture is redrawn when you let go of the slider.
    """
    if not sliders_ok():
        print("Sliders are not available here. Use the plain cell above instead: "
              "change the number in it and re-run it.")
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

    def call(**kw):
        kw = {k: (None if isinstance(v, str) and v == NO_LIMIT else v) for k, v in kw.items()}
        func(**kw)

    w.interact(call, **widgets)


def _positive(value, default, what: str):
    """value as a positive number; anything else -> default, with a note."""
    value = _decimal_comma(_plain(value), what)
    v = _as_number(value)
    if isinstance(v, (int, float)) and not isinstance(v, bool) and np.isfinite(v) and v > 0:
        return v
    _say(f"{what} {value!r} is not a positive number: showing {what} = {default:g}.")
    return default


# ---------------------------------------------------------------------------
# Data: the course split, the models, the toys
# ---------------------------------------------------------------------------
@dataclass
class CourseData:
    df: pd.DataFrame        # the raw table (30,000 clients), incl. collections_contact
    X: pd.DataFrame         # the 12 features of taiwan_features
    y: pd.Series            # default (1) / repaid (0)
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    folds: object           # StratifiedKFold(5, shuffle=True, random_state=42)


def folds():
    from sklearn.model_selection import StratifiedKFold
    return StratifiedKFold(5, shuffle=True, random_state=SEED)


def course_data() -> CourseData:
    """The course data and split (the same as Block 2's)."""
    def build():
        from sklearn.model_selection import train_test_split
        cache = getattr(_finance_data, "_TAIWAN_CACHE", None)
        if VERBOSE and cache is not None and not Path(cache).exists():
            _say("Downloading the course data from OpenML (about 15 s, only the first "
                 "time) ...")
        df = load_taiwan(verbose=False)
        X, y = taiwan_features(df), df["default"]
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y,
                                                  random_state=SEED)
        return CourseData(df, X, y, X_tr, X_te, y_tr, y_te, folds())
    return _cached("data", 0, "the data", build)


def course_split():
    """X_train, X_test, y_train, y_test of the course split."""
    d = course_data()
    return d.X_train, d.X_test, d.y_train, d.y_test


BOOSTING = dict(learning_rate=0.03, max_iter=1000, max_leaf_nodes=7, min_samples_leaf=60,
                early_stopping=True, random_state=SEED)
FOREST = dict(n_estimators=300, min_samples_leaf=10, random_state=SEED, n_jobs=-1)


def model(name: str):
    """An unfitted model of the tournament: logistic, tree, forest or boosting."""
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier
    if name == "logistic":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    if name == "tree":
        return DecisionTreeClassifier(max_depth=5, random_state=SEED)
    if name == "forest":
        return RandomForestClassifier(**FOREST)
    if name == "boosting":
        return HistGradientBoostingClassifier(**BOOSTING)
    raise ValueError(f"unknown model {name!r}: use logistic, tree, forest or boosting")


def crescents():
    """The made-up crescents: 200 training points and 20,000 new points."""
    def build():
        from sklearn.datasets import make_moons
        X, y = make_moons(n_samples=200, noise=0.3, random_state=SEED)
        X_new, y_new = make_moons(n_samples=20_000, noise=0.3, random_state=SEED + 1)
        return X, y, X_new, y_new
    return _cached("crescents", 0, "the crescents", build)


WAVE_RANGE = (0.0, 6.0)
WAVE_POINTS, WAVE_NOISE = 80, 0.45


def wave_truth(x):
    """The made-up curve of the 1-D toy: y = sin(x) + 0.3 x."""
    return np.sin(x) + 0.3 * np.asarray(x)


def wave_samples(k: int = 20):
    """k samples from the same recipe: 80 points, x uniform on 0-6, y = curve + noise."""
    out = []
    for i in range(k):
        rng = np.random.default_rng(SEED + i)
        x = np.sort(rng.uniform(*WAVE_RANGE, WAVE_POINTS))
        out.append((x, wave_truth(x) + rng.normal(0, WAVE_NOISE, WAVE_POINTS)))
    return out


def copies_data():
    """6,000 training clients, each in 3 identical rows (e.g. three monthly snapshots).
    Returns X_rows (18,000 x 12), y_rows and client (the client id of each row),
    as copies: changing them does not change what the lab keeps."""
    def build():
        d = course_data()
        clients = d.X_train.sample(6000, random_state=SEED).index
        base, yb = d.X_train.loc[clients], d.y_train.loc[clients]
        X_rows = pd.concat([base] * 3, ignore_index=True)
        y_rows = pd.concat([yb] * 3, ignore_index=True)
        client = pd.Series(np.tile(np.asarray(clients), 3), name="client")
        return X_rows, y_rows, client
    X_rows, y_rows, client = _cached("copies", 0, "the copies", build)
    return X_rows.copy(), y_rows.copy(), client.copy()


def impurity(share):
    """Impurity of a group with this share of positives: 2 share (1 - share)."""
    share = np.asarray(share, dtype=float)
    return 2 * share * (1 - share)


# ---------------------------------------------------------------------------
# Opening: the expert's tree and the learned tree
# ---------------------------------------------------------------------------
def _expert_leaf(X: pd.DataFrame) -> pd.Series:
    late = X["pay_delay_recent"] >= 2
    high_use = X["utilisation"] > 0.9
    return pd.Series(np.where(late, "decline", np.where(high_use, "manual review", "approve")),
                     index=X.index)


def expert_tree_table(rows: str = "train") -> pd.DataFrame:
    """The expert's rule list walked on our data (training rows by default):
    2+ months late on the latest bill -> decline; else utilisation > 0.9 ->
    manual review; else approve."""
    d = course_data()
    X, y = (d.X_train, d.y_train) if rows == "train" else (d.X_test, d.y_test)
    leaf = _expert_leaf(X)
    tab = pd.DataFrame({"leaf": leaf, "default": y}).groupby("leaf")["default"].agg(
        clients="size", defaulted="sum")
    tab["share of clients"] = tab["clients"] / len(X)
    tab["share defaulted"] = tab["defaulted"] / tab["clients"]
    return tab.loc[["decline", "manual review", "approve"]]


def expert_decline_counts() -> dict:
    """The expert's "decline" leaf as a classifier, on the test rows (read only)."""
    d = course_data()
    dec = (_expert_leaf(d.X_test) == "decline").to_numpy()
    yv = d.y_test.to_numpy()
    c = {"TP": int((dec & (yv == 1)).sum()), "FP": int((dec & (yv == 0)).sum()),
         "FN": int((~dec & (yv == 1)).sum()), "TN": int((~dec & (yv == 0)).sum())}
    c["precision"] = c["TP"] / (c["TP"] + c["FP"])
    c["recall"] = c["TP"] / (c["TP"] + c["FN"])
    return c


def _learned_tree(depth: int = 2):
    from sklearn.tree import DecisionTreeClassifier
    d = course_data()
    return _cached(("learned", depth), 0, "the tree",
                   lambda: DecisionTreeClassifier(max_depth=depth, random_state=SEED).fit(
                       d.X_train, d.y_train))


def _yes_is_right(feature: str, thr: float) -> bool:
    """True if the picture draws the right branch (value > cut) as "yes" (on the
    left): the questions about being late, as in learned_tree_spec."""
    key = next((k for k in _PLAIN_Q if k[0] == feature and abs(k[1] - thr) < 0.01), None)
    return key is None or _PLAIN_Q[key][1] == "no-yes"


def learned_tree_table(depth: int = 2) -> pd.DataFrame:
    """Leaves of a tree learned from the 21,000 training clients: the path of
    questions, the share of clients and the share who defaulted (= its p).
    The leaves are numbered 1, 2, ... from left to right, as in the picture."""
    tree = _learned_tree(depth)
    d = course_data()
    t, names = tree.tree_, list(d.X_train.columns)
    leaf_of_row = tree.apply(d.X_train)
    yv = d.y_train.to_numpy()
    rows = []

    def rec(node, path):
        if t.children_left[node] == -1:
            m = leaf_of_row == node
            rows.append({"leaf": len(rows) + 1, "path": " and ".join(path) or "(all)",
                         "clients": int(m.sum()), "share of clients": m.mean(),
                         "share defaulted": yv[m].mean()})
            return
        f, thr = names[t.feature[node]], t.threshold[node]
        left = (t.children_left[node], path + [f"{f} <= {thr:.3g}"])
        right = (t.children_right[node], path + [f"{f} > {thr:.3g}"])
        for child in ((right, left) if _yes_is_right(f, thr) else (left, right)):
            rec(*child)

    rec(0, [])
    return pd.DataFrame(rows).set_index("leaf")


def expert_tree_spec() -> dict:
    """The expert's tree as a dict for draw_tree_diagram (numbers: training rows)."""
    tab = expert_tree_table()

    fills = {"decline": LIGHT_RED, "manual review": SHADE, "approve": LIGHT_BLUE}

    def leaf(name):
        r = tab.loc[name]
        return {"leaf": name, "clients": r["share of clients"], "defaulted": r["share defaulted"],
                "colour": fills[name]}

    return {"question": "2+ months late on\nthe latest bill?",
            "yes": leaf("decline"),
            "no": {"question": "card used above 90%\nof its limit?",
                   "yes": leaf("manual review"), "no": leaf("approve")}}


_PLAIN_Q = {  # plain questions for the learned depth-2 tree's cuts
    ("pay_delay_recent", 1.5): ("2+ months late on\nthe latest bill?", "no-yes"),
    ("pay_ratio", 0.67): ("paid at most 2/3\nof the last bill?", "yes-no"),
    ("months_delayed_6m", 0.5): ("late in any of the\nlast 6 months?", "no-yes"),
}


def learned_tree_spec(depth: int = 2) -> dict:
    """The learned tree as a dict for draw_tree_diagram. The right branch of a
    cut (value > cut) is drawn on the "yes" side when the question asks about
    being late, as on the slides; leaves show p = share defaulted."""
    tree = _learned_tree(depth)
    d = course_data()
    t, names = tree.tree_, list(d.X_train.columns)
    leaf_of_row = tree.apply(d.X_train)
    yv = d.y_train.to_numpy()

    def rec(node):
        if t.children_left[node] == -1:
            m = leaf_of_row == node
            return {"leaf": f"p = {yv[m].mean():.3f}", "clients": m.mean(),
                    "defaulted": yv[m].mean()}
        f, thr = names[t.feature[node]], t.threshold[node]
        left, right = rec(t.children_left[node]), rec(t.children_right[node])
        key = next((k for k in _PLAIN_Q if k[0] == f and abs(k[1] - thr) < 0.01), None)
        if key is None:
            return {"question": f"{LABELS.get(f, f)}\nabove {thr:g}?", "yes": right, "no": left}
        q, order = _PLAIN_Q[key]
        return ({"question": q, "yes": right, "no": left} if order == "no-yes"
                else {"question": q, "yes": left, "no": right})

    return rec(0)


def _leaf_colour(share: float):
    """Leaf fill by the share who defaulted: light blue below 30%, light grey
    30-50%, light red above 50%."""
    return LIGHT_BLUE if share < 0.3 else (SHADE if share <= 0.5 else LIGHT_RED)


def draw_tree_diagram(ax, spec: dict, *, fontsize: float = 9) -> None:
    """Draw a small yes/no tree: question boxes (white, dark edge), leaf boxes
    filled by the share who defaulted (or by spec["colour"]); "yes" is the left
    branch. spec = {"question": str, "yes": spec, "no": spec} or a leaf
    {"leaf": str, "clients": share, "defaulted": share[, "colour": colour]}."""
    leaves, nodes = [], []

    def depth_of(s):
        return 0 if "leaf" in s else 1 + max(depth_of(s["yes"]), depth_of(s["no"]))

    def place(s, level):
        if "leaf" in s:
            x = float(len(leaves))
            leaves.append(s)
        else:
            x = (place(s["yes"], level + 1) + place(s["no"], level + 1)) / 2
        nodes.append((s, x, level))
        return x

    D = depth_of(spec)
    place(spec, 0)
    texts = {}
    for s, x, lev in nodes:
        if "leaf" in s:
            txt = f"{s['leaf']}\n{s['clients']:.1%} of clients\n{s['defaulted']:.1%} defaulted"
            fc, ec = s.get("colour", _leaf_colour(s["defaulted"])), GRAY
        else:
            txt, fc, ec = s["question"], "white", DARK
        texts[id(s)] = ax.text(x, -lev, txt, ha="center", va="center", fontsize=fontsize,
                               linespacing=1.15, zorder=3,
                               bbox=dict(boxstyle="round,pad=0.35", fc=fc, ec=ec, lw=1.1))
    ax.set_xlim(-0.6, len(leaves) - 0.4)
    ax.set_ylim(-D - 0.5, 0.5)
    ax.axis("off")
    # measure the drawn boxes, so the branches run from box edge to box edge
    ax.figure.canvas.draw()
    inv = ax.transData.inverted()

    def edges(s):
        bb = texts[id(s)].get_bbox_patch().get_window_extent()
        (_, y0), (_, y1) = inv.transform([[bb.x0, bb.y0], [bb.x1, bb.y1]])
        return y0, y1

    pos = {id(s): x for s, x, _ in nodes}
    for s, x, lev in nodes:
        if "leaf" in s:
            continue
        bottom = edges(s)[0]
        for side in ("yes", "no"):
            child = s[side]
            cx, top = pos[id(child)], edges(child)[1]
            ax.plot([x, cx], [bottom, top], color=GRAY, lw=1.2, zorder=1)
            ax.text(x + 0.3 * (cx - x) + (-0.05 if side == "yes" else 0.05),
                    bottom + 0.3 * (top - bottom), side, ha="right" if side == "yes" else "left",
                    va="center", fontsize=fontsize, color=MUTED)


def show_expert_tree(figsize=(6.0, 3.0)) -> None:
    with _style():
        fig, ax = _new_fig(figsize)
        draw_tree_diagram(ax, expert_tree_spec())
        ax.set_title("An expert's rule list as a tree (a made-up example), "
                     "walked on the 21,000 training clients")
    _show(fig)


def show_learned_tree(depth: int = 2, figsize=(7.5, 3.2)) -> None:
    with _style():
        fig, ax = _new_fig(figsize)
        draw_tree_diagram(ax, learned_tree_spec(depth))
        ax.set_title(f"A tree learned from the 21,000 training clients (depth {depth})")
    _show(fig)


# ---------------------------------------------------------------------------
# D1 - Try every cut
# ---------------------------------------------------------------------------
D1_FEATURES = ["pay_delay_recent", "months_delayed_6m", "limit_bal", "pay_ratio",
               "utilisation", "age"]


def _drop_curve(feature: str):
    """Every cut between two neighbouring values of feature (training rows):
    cuts, share of clients on the left, the impurity drop."""
    def build():
        d = course_data()
        x = d.X_train[feature].to_numpy(dtype=float)
        yv = d.y_train.to_numpy()
        order = np.argsort(x, kind="stable")
        xs, ys = x[order], yv[order]
        vals, first = np.unique(xs, return_index=True)
        counts = np.diff(np.append(first, len(xs)))
        pos = np.add.reduceat(ys, first)
        n, P = len(xs), ys.sum()
        nL, pL = np.cumsum(counts)[:-1], np.cumsum(pos)[:-1]
        nR, pR = n - nL, P - pL
        cuts = (vals[:-1] + vals[1:]) / 2
        weighted = nL / n * impurity(pL / nL) + nR / n * impurity(pR / nR)
        drop = impurity(P / n) - weighted
        return cuts, nL / n, drop
    return _cached(("drop", feature), 0, "the cuts", build)


def _split_at(feature: str, cut: float) -> dict:
    d = course_data()
    x, yv = d.X_train[feature].to_numpy(), d.y_train.to_numpy()
    left = x <= cut
    n = len(x)
    out = {"cut": cut, "left share": left.mean(),
           "left rate": yv[left].mean() if left.any() else np.nan,
           "right rate": yv[~left].mean() if (~left).any() else np.nan, "parent rate": yv.mean()}
    w = out["left share"]
    imp_l = impurity(out["left rate"]) if left.any() else 0.0
    imp_r = impurity(out["right rate"]) if (~left).any() else 0.0
    out["left impurity"], out["right impurity"] = float(imp_l), float(imp_r)
    out["parent impurity"] = float(impurity(yv.mean()))
    out["drop"] = out["parent impurity"] - (w * imp_l + (1 - w) * imp_r)
    out["n"] = n
    return out


def best_cuts(features=None) -> pd.DataFrame:
    """For every feature: its best cut on the training rows and the impurity drop."""
    d = course_data()
    rows = []
    for f in (features or list(d.X_train.columns)):
        cuts, left, drop = _drop_curve(f)
        i = int(np.argmax(drop))
        s = _split_at(f, cuts[i])
        rows.append({"feature": f, "best cut": cuts[i], "drop": drop[i],
                     "left share": s["left share"], "left rate": s["left rate"],
                     "right rate": s["right rate"]})
    return pd.DataFrame(rows).set_index("feature").sort_values("drop", ascending=False)


def cut_range(feature: str):
    """(lo, hi, step) for a cut slider on this feature (a name that is not one
    of the 12 features: the note of split_search, and pay_delay_recent)."""
    feature = _feature_name(feature)
    d = course_data()
    x = d.X_train[feature]
    if feature in ("pay_delay_recent", "months_delayed_6m"):
        return (float(x.min()) + 0.5, float(x.max()) - 0.5, 1.0)
    if feature == "limit_bal":
        return (20_000.0, 500_000.0, 10_000.0)
    if feature == "age":
        return (22.5, 70.5, 1.0)
    lo, hi = np.quantile(x, [0.01, 0.99])
    return (round(float(lo), 2) + 0.0, round(float(hi), 2) + 0.0, 0.02)


def _fmt_cut(feature: str, cut: float) -> str:
    if feature == "limit_bal":
        return f"{cut:,.0f}"
    if feature in ("pay_delay_recent", "months_delayed_6m", "age"):
        return f"{cut:g}"
    return f"{cut:.3g}"


def draw_split_bars(ax, feature: str = "pay_delay_recent", cut=None) -> dict:
    """Two bars, one per side of the cut: height = share who defaulted,
    width = share of the training clients on that side. Returns the numbers."""
    if cut is None:
        cuts, _, drop = _drop_curve(feature)
        cut = float(cuts[np.argmax(drop)])
    s = _split_at(feature, cut)
    name, c = LABELS.get(feature, feature), _fmt_cut(feature, cut)
    wl, wr = s["left share"], 1 - s["left share"]
    gap = 0.03
    sides = ((0, wl, s["left rate"], s["left impurity"], f"\u2264 {c}", "left"),
             (wl + gap, wr, s["right rate"], s["right impurity"], f"> {c}", "right"))
    # in a narrow panel the two labels would touch: the wider group's label goes lower
    narrow = ax.figure.get_figwidth() * ax.get_position().width < 2.6
    for x0, w, rate, imp, label, where in sides:
        if w <= 0:
            continue
        ax.add_patch(Rectangle((x0, 0), w, rate, facecolor=RED, edgecolor="white", lw=0.8))
        low = narrow and w == max(wl, wr) and rate < 0.3
        ax.text(0 if where == "left" else 1 + gap, 0.6 if low else 1.0,
                f"{label}\n{w:.1%} of clients\n{rate:.1%} defaulted\nimpurity {imp:.3f}",
                ha=where, va="top", fontsize=8.5, color=INK, linespacing=1.1)
    ax.axhline(s["parent rate"], color=DARK, ls="--", lw=1)
    ax.text(1.01, s["parent rate"], f"all\n{s['parent rate']:.1%}", ha="left", va="center",
            fontsize=8.5, color=DARK, transform=ax.get_yaxis_transform(), linespacing=1.0)
    ax.set_xlim(0, 1 + gap)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.yaxis.set_major_formatter(PCT0)
    ax.set_ylabel("share who defaulted")
    ax.set_xlabel("bar width = share of clients")
    ax.set_title(f"{name}, cut at {c}: drop {s['drop']:.3f}")
    return s


def draw_drop_curve(ax, feature: str = "pay_delay_recent", cut=None) -> None:
    """The impurity drop of every cut of feature; the best cut and (optionally)
    the current cut marked."""
    cuts, _, drop = _drop_curve(feature)
    i = int(np.argmax(drop))
    discrete = len(cuts) <= 30
    if discrete:
        ax.plot(cuts, drop, "-o", color=DARK, ms=3.5, lw=1.2)
    else:
        ax.plot(cuts, drop, "-", color=DARK, lw=1.2)
    ax.plot(cuts[i], drop[i], "o", color=DARK, ms=6.5, zorder=4)
    ax.annotate(f"best cut {_fmt_cut(feature, cuts[i])}\ndrop {drop[i]:.3f}", (cuts[i], drop[i]),
                xytext=(7, 2), textcoords="offset points", fontsize=8.5, va="bottom", color=DARK,
                linespacing=1.0)
    if cut is not None:
        s = _split_at(feature, cut)
        ax.axvline(cut, color=GRAY, lw=1, ls=":")
        ax.plot(cut, s["drop"], "o", ms=7, mfc="none", mec=INK, zorder=5)
    if not discrete:
        lo, hi = np.quantile(course_data().X_train[feature], [0.005, 0.995])
        ax.set_xlim(lo, hi)
    if feature == "limit_bal":
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:.0f}"))
    ax.set_ylim(0, max(0.065, drop.max() * 1.25))
    ax.set_xlabel(f"cut: {LABELS.get(feature, feature)}"
                  + (" (thousand NT$)" if feature == "limit_bal" else ""))
    ax.set_ylabel("impurity drop")
    ax.set_title("every cut, tried")


def draw_best_drops(ax, features=None) -> None:
    """Horizontal bars: the best impurity drop of each feature."""
    tab = best_cuts(features or D1_FEATURES)
    names = [LABELS.get(f, f) for f in tab.index][::-1]
    vals = tab["drop"].to_numpy()[::-1]
    ax.barh(names, vals, color=[DARK if v == vals.max() else GRAY for v in vals], height=0.6)
    for k, v in enumerate(vals):
        ax.text(v + 0.001, k, f"{v:.3f}", va="center", fontsize=8.5)
    ax.set_xlim(0, vals.max() * 1.25)
    ax.set_xlabel("best drop")
    ax.tick_params(axis="y", length=0)


def _feature_name(feature) -> str:
    """feature if it is one of the 12 columns; else "pay_delay_recent", with a note."""
    cols = list(course_data().X_train.columns)
    if feature in cols:
        return feature
    _say(f"{feature!r} is not one of the 12 features ({', '.join(cols)}): "
         'showing "pay_delay_recent".')
    return "pay_delay_recent"


def _cut_inside(feature: str, cut) -> bool:
    """True if cut is a number that leaves training clients on both sides."""
    if not (isinstance(cut, (int, float)) and not isinstance(cut, bool) and np.isfinite(cut)):
        return False
    x = course_data().X_train[feature]
    return float(x.min()) <= cut < float(x.max())


def split_search_figure(feature: str = "pay_delay_recent", cut=None, figsize=None,
                        suptitle=True):
    feature = feature if feature in course_data().X_train.columns else "pay_delay_recent"
    cut = _as_number(cut)
    cut = cut if _cut_inside(feature, cut) else None    # None or outside: the best cut
    with _style():
        fig, axes = _new_fig(figsize or (8.0, 3.2), 1, 2, width_ratios=[1.1, 1])
        if cut is None:
            cuts, _, drop = _drop_curve(feature)
            cut = float(cuts[np.argmax(drop)])
        draw_split_bars(axes[0], feature, cut)
        draw_drop_curve(axes[1], feature, cut)
        if suptitle:
            fig.suptitle("One question, two groups: the 21,000 training clients", fontsize=9.5)
    return fig


def split_search(feature: str = "pay_delay_recent", cut=None) -> None:
    """D1: cut one feature in two; see both groups and the impurity drop of every cut.
    cut=None (or a cut outside the feature's values) shows the best cut."""
    feature = _feature_name(feature)
    cut = _decimal_comma(_plain(cut), "cut")
    c = _as_number(cut)
    name = LABELS.get(feature, feature)
    if cut is not None and not (isinstance(c, (int, float)) and not isinstance(c, bool)
                                and np.isfinite(c)):
        _say(f"cut {cut!r} is not a number: showing the best cut.")
        cut = None
    elif cut is not None and not _cut_inside(feature, c):
        x = course_data().X_train[feature]
        lo, hi = _fmt_cut(feature, float(x.min())), _fmt_cut(feature, float(x.max()))
        shown = f"{c:,.0f}" if abs(c) >= 1000 else f"{c:g}"
        _say(f"cut {shown} is outside {name} ({lo} to {hi}), so one group "
             "would be empty: showing the best cut.")
        cut = None
    cuts, _, drop = _drop_curve(feature)
    if drop.max() < 0.005:
        best = best_cuts(["pay_delay_recent"])["drop"].iloc[0]
        _say(f"{name} hardly separates defaulters from repayers: even its best cut lowers "
             f"the impurity by only {drop.max():.3f} (the latest status: {best:.3f}).")
    _show(split_search_figure(feature, cut))


# ---------------------------------------------------------------------------
# E1 - Grow a tree (crescents)
# ---------------------------------------------------------------------------
E1_DEPTHS = list(range(1, 13)) + [None]


def _e1_tree(depth):
    from sklearn.tree import DecisionTreeClassifier
    X, y, _, _ = crescents()
    return _cached(("e1", depth), 0, "the tree",
                   lambda: DecisionTreeClassifier(max_depth=depth, random_state=SEED).fit(X, y))


def grow_tree_numbers() -> pd.DataFrame:
    """Depth, leaves, share correct on the 200 training points and the 20,000 new points."""
    X, y, Xn, yn = crescents()
    rows = []
    for d in E1_DEPTHS:
        t = _e1_tree(d)
        rows.append({"depth": _depth_name(d), "levels used": t.get_depth(),
                     "leaves": t.get_n_leaves(), "training points": t.score(X, y),
                     "new points": t.score(Xn, yn)})
    return pd.DataFrame(rows).set_index("depth")


def _crescent_window():
    X, *_ = crescents()
    lo, hi = X.min(0) - 0.25, X.max(0) + 0.25
    return [lo[0], hi[0], lo[1], hi[1]]


def _leaf_boxes(tree, window):
    t = tree.tree_
    out = []

    def rec(node, box):
        if t.children_left[node] == -1:
            v = t.value[node][0]
            out.append((box, v[1] / v.sum()))
            return
        f, thr = t.feature[node], t.threshold[node]
        lb, rb = list(box), list(box)
        if f == 0:
            lb[1], rb[0] = min(box[1], thr), max(box[0], thr)
        else:
            lb[3], rb[2] = min(box[3], thr), max(box[2], thr)
        rec(t.children_left[node], lb)
        rec(t.children_right[node], rb)

    rec(0, list(window))
    return out


def draw_tree_regions(ax, depth=2, points: str = "training") -> None:
    """The crescents with the tree's rectangles: light red = the tree answers
    "red", light blue = "blue"; white lines = the edges of its regions (leaves)."""
    depth = _snap(depth, E1_DEPTHS, "depth", 2, quiet=True)
    X, y, Xn, yn = crescents()
    tree = _e1_tree(depth)
    win = _crescent_window()
    for (x0, x1, y0, y1), p1 in _leaf_boxes(tree, win):
        if x1 > x0 and y1 > y0:
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                                   facecolor=LIGHT_RED if p1 >= 0.5 else LIGHT_BLUE,
                                   edgecolor="white", lw=0.9, zorder=1))
    if points == "new":
        P, Q = Xn[:600], yn[:600]
    else:
        P, Q = X, y
    for cls, col in ((0, BLUE), (1, RED)):
        m = Q == cls
        ax.scatter(P[m, 0], P[m, 1], s=13, color=col, edgecolor="white", linewidth=0.4,
                   zorder=3)
    ax.set_xlim(win[0], win[1])
    ax.set_ylim(win[2], win[3])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("feature 1")
    ax.set_ylabel("feature 2")
    n = tree.get_n_leaves()
    ax.set_title((f"max depth {depth}" if depth is not None else "no depth limit")
                 + f": {n} region{'s' if n > 1 else ''}")


def draw_tree_accuracy(ax, depth=2) -> None:
    """Share classified correctly by depth, drawn up to the chosen depth:
    grey = the 200 training points, red = 20,000 new points."""
    depth = _snap(depth, E1_DEPTHS, "depth", 2, quiet=True)
    tab = grow_tree_numbers()
    xs = list(range(1, 13)) + [13.6]
    names = [_depth_name(d) for d in E1_DEPTHS]
    k = len(E1_DEPTHS) if depth is None else E1_DEPTHS.index(depth) + 1
    for col, colour, lab in (("training points", GRAY, "training points (200)"),
                             ("new points", RED, "new points (20,000)")):
        v = tab[col].to_numpy()
        ax.plot(xs[:min(k, 12)], v[:min(k, 12)], "-o", color=colour, ms=3.5, lw=1.5, label=lab)
        if k == 13:
            ax.plot(xs[11:13], v[11:13], ":", color=colour, lw=1.2)
            ax.plot(xs[12], v[12], "o", color=colour, ms=3.5)
        ax.plot(xs[k - 1], v[k - 1], "o", color=colour, ms=8, zorder=4)
        ax.annotate(f"{v[k - 1]:.1%}", (xs[k - 1], v[k - 1]),
                    xytext=(6, 5 if col == "training points" else -11), textcoords="offset points",
                    fontsize=8.5, color=colour if colour != GRAY else MUTED)
    ax.set_xticks(xs)
    ax.set_xticklabels([n if n in ("1", "2", "4", "6", "8", "10", "12") else
                        ("no\nlimit" if n == NO_LIMIT else "") for n in names], fontsize=8.5)
    ax.set_xlim(0.4, 14.4)
    ax.set_ylim(0.75, 1.02)
    ax.yaxis.set_major_formatter(PCT0)
    ax.set_xlabel("max depth")
    ax.set_ylabel("share correct")
    ax.legend(loc="lower right", fontsize=8.5, handlelength=1.5)


def grow_tree_figure(depth=2, figsize=None, suptitle=True):
    depth = _snap(depth, E1_DEPTHS, "depth", 2, quiet=True)
    with _style():
        fig, axes = _new_fig(figsize or (8.0, 3.4), 1, 2, width_ratios=[1, 1.1])
        draw_tree_regions(axes[0], depth)
        draw_tree_accuracy(axes[1], depth)
        if suptitle:
            fig.suptitle("Made-up crescents: 200 training points, 20,000 new points",
                         fontsize=9.5)
    return fig


def grow_tree(depth=2) -> None:
    """E1: a tree of this max depth on the crescents (depth 1-12, or None = no limit).
    Another value is moved to the nearest one on the list, with a note."""
    _show(grow_tree_figure(_snap(depth, E1_DEPTHS, "depth", 2)))


# ---------------------------------------------------------------------------
# D2 - Units do not matter to a tree
# ---------------------------------------------------------------------------
def units_check(n_show: int = 5) -> pd.DataFrame:
    """The same depth-5 tree and the same logistic regression, fitted on the
    training rows in three units: raw, standardised (z-scores), and with
    log(credit limit). Shows p for the first training clients and prints the
    largest difference over all 21,000 of them.

    The logistic regression uses solver="newton-cholesky", which reaches the
    best coefficients on raw columns too (the default solver, lbfgs, can stop
    early when the columns have very different sizes). So the differences it
    prints come from the model itself: standardising changes p only through the
    default penalty (about 0.01), and the log of the limit changes the shape of
    the relation that the straight line sees (about 0.09)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier
    d = course_data()
    X = d.X_train
    versions = {"raw": X,
                "standardised": pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns,
                                             index=X.index),
                "log limit": X.assign(limit_bal=np.log(X["limit_bal"]))}
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, Xv in versions.items():
            out[f"tree: {name}"] = DecisionTreeClassifier(max_depth=5, random_state=SEED).fit(
                Xv, d.y_train).predict_proba(Xv)[:, 1]
        for name, Xv in versions.items():
            out[f"logistic: {name}"] = LogisticRegression(solver="newton-cholesky").fit(
                Xv, d.y_train).predict_proba(Xv)[:, 1]
    tab = pd.DataFrame(out, index=X.index)
    def diff(a, b):
        return np.abs(tab[a] - tab[b]).max()
    print("Largest difference in p over the 21,000 training clients:")
    print(f"  tree:     raw vs standardised {diff('tree: raw', 'tree: standardised'):.3f}, "
          f"raw vs log limit {diff('tree: raw', 'tree: log limit'):.3f}")
    print(f"  logistic: raw vs standardised "
          f"{diff('logistic: raw', 'logistic: standardised'):.3f}, "
          f"raw vs log limit {diff('logistic: raw', 'logistic: log limit'):.3f}")
    return tab.head(n_show).round(3)


# ---------------------------------------------------------------------------
# E2 - Trace the ROC curve (last week's model on the test rows, read only)
# ---------------------------------------------------------------------------
E2_THRESHOLDS = [round(float(t), 2) for t in np.arange(0.95, 0.04, -0.05)]
_E2_TRAIL: list = []

CELL = {  # (column, row): row 0 = actually default, column 0 = decline (Block 2 layout)
    "TP": (0, 0, RED, "white", "caught"),
    "FN": (1, 0, LIGHT_RED, INK, "missed"),
    "FP": (0, 1, LIGHT_BLUE, INK, "falsely\ndeclined"),
    "TN": (1, 1, BLUE, "white", "rightly\napproved"),
}


def baseline_test_p():
    """Last week's model (scaler + logistic regression, fitted on the training
    rows): its p on the 9,000 test rows, and their outcomes."""
    def build():
        d = course_data()
        m = model("logistic").fit(d.X_train, d.y_train)
        return m.predict_proba(d.X_test)[:, 1], d.y_test.to_numpy()
    return _cached("baseline", 0, "last week's model", build)


def _counts(p, yv, t) -> dict:
    dec = p > t   # decline when p > t (Block 2's rule)
    return {"TP": int((dec & (yv == 1)).sum()), "FN": int((~dec & (yv == 1)).sum()),
            "FP": int((dec & (yv == 0)).sum()), "TN": int((~dec & (yv == 0)).sum())}


def roc_points(thresholds=(0.7, 0.5, 0.3)) -> pd.DataFrame:
    """Confusion counts, false-alarm rate (FP / all repayers) and recall
    (TP / all defaulters) of last week's model on the test rows."""
    p, yv = baseline_test_p()
    rows = []
    for t in thresholds:
        c = _counts(p, yv, t)
        rows.append({"threshold": t, **c, "false-alarm rate": c["FP"] / (c["FP"] + c["TN"]),
                     "recall": c["TP"] / (c["TP"] + c["FN"]),
                     "precision": c["TP"] / max(c["TP"] + c["FP"], 1)})
    return pd.DataFrame(rows).set_index("threshold")


def draw_confusion(ax, threshold=0.5) -> dict:
    """The confusion matrix at this threshold, in the slide layout."""
    p, yv = baseline_test_p()
    c = _counts(p, yv, threshold)
    for key, (col, row, fc, tc, words) in CELL.items():
        ax.add_patch(Rectangle((col, row), 1, 1, facecolor=fc, edgecolor="white", linewidth=2))
        ax.text(col + 0.5, row + 0.3, f"{c[key]:,}", ha="center", va="center", fontsize=10.5,
                fontweight="bold", color=tc)
        ax.text(col + 0.5, row + 0.68, f"{key}: {words}", ha="center", va="center",
                fontsize=8.5, color=tc, linespacing=1.0)
    ax.set_xlim(0, 2)
    ax.set_ylim(2, 0)
    ax.set_aspect("equal")
    ax.xaxis.tick_top()
    ax.tick_params(length=0, pad=2)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks([0.5, 1.5])
    ax.set_xticklabels(["p > t\ndecline", "p \u2264 t\napprove"], fontsize=8.5, linespacing=1.0)
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels(["actually\ndefault", "actually\nrepaid"], fontsize=8.5)
    ax.set_xlabel(f"threshold t = {threshold:.2f}", fontsize=9.5)
    return c


def draw_roc_trail(ax, threshold=0.5, trail=None) -> None:
    """The ROC curve of last week's model (faint), the dots of the thresholds
    tried (trail) and the current threshold's dot."""
    from sklearn.metrics import roc_auc_score, roc_curve
    p, yv = baseline_test_p()
    fpr, tpr, _ = roc_curve(yv, p)
    auc = roc_auc_score(yv, p)
    ax.plot([0, 1], [0, 1], ls="--", color=GRAY, lw=1)
    ax.text(0.66, 0.58, "coin", color=MUTED, fontsize=8.5)
    ax.plot(fpr, tpr, color=LIGHT_BLUE, lw=2.2, zorder=1)
    pts = roc_points(sorted(set(trail or []) | {threshold}))
    if trail:
        tr = pts.loc[sorted(set(trail))]
        ax.plot(tr["false-alarm rate"], tr["recall"], "o", color=RED, alpha=0.45, ms=5, zorder=3)
    cur = pts.loc[threshold]
    fx, ry = cur["false-alarm rate"], cur["recall"]
    ax.plot(fx, ry, "o", color=RED, ms=9, zorder=4, clip_on=False)
    right = fx > 0.55   # near the right edge: write the label on the left
    low = ry < 0.15     # near the bottom: write the label above the dot, clear of the axis
    ax.annotate(f"t = {threshold:.2f}\n({fx:.3f}, {ry:.3f})", (fx, ry),
                xytext=(-10 if right else 10, 8 if low else -6), textcoords="offset points",
                fontsize=8.5, color=RED, va="bottom" if low else "top",
                ha="right" if right else "left", zorder=5,
                bbox=dict(fc="white", ec="none", pad=0.2, alpha=0.85))
    ax.text(0.97, 0.05, f"AUC = {auc:.3f}", ha="right", va="bottom",
            fontsize=9.5, color=DARK, transform=ax.transAxes)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.0)
    ax.set_aspect("equal")
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel("false-alarm rate\n= FP / all repayers", linespacing=1.0)
    ax.set_ylabel("recall\n= TP / all defaulters", linespacing=1.0)
    ax.set_title("one dot per threshold")


def trace_roc_figure(threshold=0.5, trail=None, figsize=None, suptitle=True):
    threshold = round(float(threshold), 2)
    with _style():
        fig, axes = _new_fig(figsize or (7.5, 3.6), 1, 2, width_ratios=[1, 1])
        draw_confusion(axes[0], threshold)
        draw_roc_trail(axes[1], threshold, trail)
        if suptitle:
            fig.suptitle("Last week's model, test rows, read only", fontsize=9.5)
    return fig


def trace_roc(threshold=0.5) -> None:
    """E2: move the threshold; the dot moves along the ROC curve. The dots of
    the thresholds you tried stay on the picture (trace_roc_reset() clears them)."""
    t = _as_number(threshold)
    if not (isinstance(t, (int, float)) and 0 <= t <= 1):
        t = _snap(threshold, E2_THRESHOLDS, "threshold", 0.5)
    threshold = round(float(t), 2)
    if threshold not in _E2_TRAIL:
        _E2_TRAIL.append(threshold)
    _show(trace_roc_figure(threshold, list(_E2_TRAIL)))


def trace_roc_reset() -> None:
    _E2_TRAIL.clear()


# ---------------------------------------------------------------------------
# E3 - Let cross-validation choose the depth (and D3, the leaf size)
# ---------------------------------------------------------------------------
E3_DEPTHS = list(range(1, 21)) + [None]
D3_LEAVES = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
D2_LEAVES = D3_LEAVES   # the name before the demos were renumbered (20.10); older copies


def logistic_cv() -> np.ndarray:
    """Validation AUC of last week's model in the five rounds."""
    def build():
        from sklearn.model_selection import cross_val_score
        d = course_data()
        return cross_val_score(model("logistic"), d.X_train, d.y_train, cv=d.folds,
                               scoring="roc_auc")
    return _cached("logistic_cv", 0, "last week's model's cross-validation", build)


def _cv_tree(param: str, values: list, about: float) -> pd.DataFrame:
    def build():
        from sklearn.model_selection import cross_validate
        from sklearn.tree import DecisionTreeClassifier
        d = course_data()
        rows = []
        for v in values:
            kw = {param: v} if param == "max_depth" else {"max_depth": None, param: v}
            r = cross_validate(DecisionTreeClassifier(random_state=SEED, **kw), d.X_train,
                               d.y_train, cv=d.folds, scoring="roc_auc", return_train_score=True)
            rows.append({"value": v, "train": r["train_score"], "valid": r["test_score"]})
        return rows
    rows = _cached(("cv", param), about, "the cross-validation for every "
                   + ("depth" if param == "max_depth" else "leaf size"), build)
    return rows


def _cv_table(rows, name) -> pd.DataFrame:
    out = []
    for r in rows:
        out.append({name: _depth_name(r["value"]) if name == "depth" else r["value"],
                    "training AUC": r["train"].mean(), "validation AUC": r["valid"].mean(),
                    "spread": r["valid"].std(),
                    "rounds": np.round(r["valid"], 3).tolist()})
    return pd.DataFrame(out).set_index(name)


def cv_depth_numbers() -> pd.DataFrame:
    """Mean training and validation AUC by depth, the spread (SD of the five
    validation AUCs) and the five rounds."""
    return _cv_table(_cv_tree("max_depth", E3_DEPTHS, 6), "depth")


def cv_leaf_numbers() -> pd.DataFrame:
    return _cv_table(_cv_tree("min_samples_leaf", D3_LEAVES, 3), "min_samples_leaf")


def _draw_cv(ax, rows, values, value, xs, xticklabels, xlabel, legend_loc="upper left") -> None:
    k = values.index(value) + 1
    tr = np.array([r["train"].mean() for r in rows])
    va = np.array([r["valid"].mean() for r in rows])
    gap_at = len(xs) - 1 if values[-1] is None else None
    # red first, grey on top of it (thinner): where the two curves coincide
    # (small depths) the grey one stays visible
    handles = {}
    for v, colour, lab, lw, ms, z in ((va, RED, "validation part", 2.4, 4.2, 2),
                                      (tr, GRAY, "training part", 1.1, 2.6, 3)):
        m = min(k, gap_at) if gap_at is not None else k
        handles[lab], = ax.plot(xs[:m], v[:m], "-o", color=colour, ms=ms, lw=lw, label=lab,
                                zorder=z)
        if gap_at is not None and k == len(xs):
            ax.plot(xs[-2:], v[-2:], ":", color=colour, lw=1.2, zorder=z)
            ax.plot(xs[-1], v[-1], "o", color=colour, ms=3, zorder=z)
    lg = logistic_cv()
    handles["logistic"] = ax.axhline(lg.mean(), color=DARK, ls="--", lw=1.1,
                                     label=f"logistic (last week) {lg.mean():.3f}")
    five = rows[k - 1]["valid"]
    ax.plot(np.full(5, xs[k - 1]), five, "o", ms=5, mfc="white", mec=RED, mew=1.2, zorder=5)
    ax.set_xticks(xs)
    ax.set_xticklabels(xticklabels, fontsize=8.5)
    ax.set_ylim(0.58, 1.02)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("AUC, mean of 5 rounds")
    ax.legend(handles=[handles[n] for n in ("training part", "validation part", "logistic")],
              loc=legend_loc, fontsize=8.5, handlelength=1.5)


def draw_cv_curve(ax, depth=3) -> None:
    """Grey = mean training AUC, red = mean validation AUC, drawn up to the chosen
    depth; white dots = the five rounds at that depth; dashed = logistic."""
    depth = _snap(depth, E3_DEPTHS, "depth", 3, quiet=True)
    rows = _cv_tree("max_depth", E3_DEPTHS, 6)
    xs = list(range(1, 21)) + [22.2]
    labels = [str(d) if d in (1, 5, 10, 15, 20) else "" for d in range(1, 21)] + ["no\nlimit"]
    _draw_cv(ax, rows, E3_DEPTHS, depth, xs, labels, "max depth of the tree")
    ax.set_xlim(0.3, 23)


def draw_cv_rounds(ax, depth=3, param="max_depth") -> None:
    """Zoom on the five rounds at one setting: the dots, their mean and the
    spread band (mean +/- SD). The y-range is always 0.08 wide."""
    rows = (_cv_tree("max_depth", E3_DEPTHS, 6) if param == "max_depth"
            else _cv_tree("min_samples_leaf", D3_LEAVES, 3))
    values = E3_DEPTHS if param == "max_depth" else D3_LEAVES
    depth = _snap(depth, values, "value", values[0], quiet=True)
    five = rows[values.index(depth)]["valid"]
    m, s = five.mean(), five.std()
    ax.axhspan(m - s, m + s, color=LIGHT_RED, alpha=0.7, lw=0)
    ax.axhline(m, color=RED, lw=1.2)
    ax.plot(np.arange(1, 6), five, "o", ms=6, mfc="white", mec=RED, mew=1.4, zorder=4)
    ax.set_xticks(np.arange(1, 6))
    ax.set_xlim(0.4, 5.6)
    ax.set_ylim(m - 0.04, m + 0.04)
    ax.set_xlabel("round")
    ax.set_ylabel("validation AUC")
    ax.yaxis.set_major_locator(MaxNLocator(4))
    name = ((f"depth {depth}" if depth is not None else NO_LIMIT) if param == "max_depth"
            else f"leaf size {depth}")
    ax.set_title(f"{name}\n{m:.3f} \u00b1 {s:.3f}", fontsize=9.5)


def cv_depth_figure(depth=3, figsize=None, suptitle=True):
    depth = _snap(depth, E3_DEPTHS, "depth", 3, quiet=True)
    with _style():
        fig, axes = _new_fig(figsize or (8.0, 3.4), 1, 2, width_ratios=[2.8, 1])
        draw_cv_curve(axes[0], depth)
        draw_cv_rounds(axes[1], depth)
        if suptitle:
            fig.suptitle("5-fold cross-validation, training rows only", fontsize=9.5)
    return fig


def cv_depth(depth=3) -> None:
    """E3: cross-validated AUC of a tree of this max depth (1-20, or None).
    Another value is moved to the nearest one on the list, with a note."""
    _show(cv_depth_figure(_snap(depth, E3_DEPTHS, "depth", 3)))


def cv_leaf_figure(min_samples_leaf=1, figsize=None, suptitle=True):
    min_samples_leaf = _snap(min_samples_leaf, D3_LEAVES, "leaf size", 1, quiet=True)
    rows = _cv_tree("min_samples_leaf", D3_LEAVES, 3)
    with _style():
        fig, axes = _new_fig(figsize or (8.0, 3.4), 1, 2, width_ratios=[2.8, 1])
        xs = list(range(1, len(D3_LEAVES) + 1))
        _draw_cv(axes[0], rows, D3_LEAVES, min_samples_leaf, xs, [str(v) for v in D3_LEAVES],
                 "min_samples_leaf (no depth limit)",
                 legend_loc="lower right")
        axes[0].set_xlim(0.4, len(xs) + 0.6)
        draw_cv_rounds(axes[1], min_samples_leaf, param="min_samples_leaf")
        if suptitle:
            fig.suptitle("5-fold cross-validation, training rows only", fontsize=9.5)
    return fig


def cv_leaf(min_samples_leaf=1) -> None:
    """D3 (E3's done-early cell): the leaf size as the leash, no depth limit.
    Another value is moved to the nearest one on the list, with a note."""
    _show(cv_leaf_figure(_snap(min_samples_leaf, D3_LEAVES, "leaf size", 1)))


# ---------------------------------------------------------------------------
# E4 - Crowd and relay (1-D wave toy)
# ---------------------------------------------------------------------------
E4_N = [1, 2, 5, 10, 30, 100, 300]
E4_STEP = 0.3        # the relay's step size (learning rate)
E4_METER_SAMPLES = 20


def _e4_paths():
    """For each of 20 samples: the crowd's line after 1-300 averaged deep
    bootstrap trees, and the relay's line after 1-300 stumps (step 0.3)."""
    def build():
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.tree import DecisionTreeRegressor
        grid = np.linspace(*WAVE_RANGE, 300)
        crowd, relay = [], []
        for i, (x, y) in enumerate(wave_samples(E4_METER_SAMPLES)):
            # bagging by hand (what BaggingRegressor does): bootstrap rows, grow a
            # deep tree, average; the running average gives every n_trees at once
            rng = np.random.default_rng(SEED + i)
            P = np.empty((300, len(grid)))
            for b in range(300):
                rows = rng.integers(0, len(x), len(x))
                P[b] = DecisionTreeRegressor(random_state=SEED).fit(
                    x[rows, None], y[rows]).predict(grid[:, None])
            crowd.append(np.cumsum(P, 0) / np.arange(1, 301)[:, None])
            gb = GradientBoostingRegressor(n_estimators=300, max_depth=1, learning_rate=E4_STEP,
                                           random_state=SEED).fit(x[:, None], y)
            relay.append(np.array(list(gb.staged_predict(grid[:, None]))))
        return grid, np.array(crowd), np.array(relay)   # (20, 300, 300) each
    return _cached("e4", 3, "the crowd and the relay for 20 samples", build)


def _meters(lines, truth):
    """spread = how far the lines are from each other (SD across samples,
    averaged along x); miss = how far their middle is from the truth."""
    return lines.std(0).mean(), np.abs(lines.mean(0) - truth).mean()


def crowd_and_relay_numbers() -> pd.DataFrame:
    grid, crowd, relay = _e4_paths()
    T = wave_truth(grid)
    rows = []
    for n in E4_N:
        cs, cm = _meters(crowd[:, n - 1], T)
        rs, rm = _meters(relay[:, n - 1], T)
        rows.append({"trees": n, "crowd spread": cs, "crowd miss": cm,
                     "relay spread": rs, "relay miss": rm})
    return pd.DataFrame(rows).set_index("trees")


def draw_meters(ax, kind: str = "crowd", n_trees: int = 1) -> None:
    """Two bars for one panel: spread (how far the lines are from each other)
    and miss (how far their middle is from the true curve), on a fixed 0-0.5 scale."""
    grid, crowd, relay = _e4_paths()
    spread, miss = _meters((crowd if kind == "crowd" else relay)[:, n_trees - 1], wave_truth(grid))
    ax.barh([1, 0], [spread, miss], color=[MID_BLUE, DARK], height=0.55)
    for k, v in zip([1, 0], [spread, miss]):
        ax.text(v + 0.015, k, f"{v:.2f}", va="center", fontsize=9)
    ax.set_yticks([1, 0])
    ax.set_yticklabels(["spread", "miss"], fontsize=9)
    ax.set_xlim(0, 0.5)
    ax.set_ylim(-0.7, 1.7)
    ax.set_xticks([0, 0.25, 0.5])
    ax.set_xticklabels(["0", "", "0.5"], fontsize=8.5)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)


def _draw_wave(ax, paths, n, lines, points, title, band=True):
    """lines thin lines (one per sample); band=True also draws, over all 20
    samples, the band mean +/- SD (light blue: how far apart the lines are =
    the spread) and their middle, the mean line (dark: its distance from the
    dashed true curve = the miss)."""
    grid = _e4_paths()[0]
    L = paths[:, n - 1]
    if points:
        x, y = wave_samples(1)[0]
        ax.scatter(x, y, s=9, color=GRAY, zorder=1)
    for i in range(lines):
        ax.plot(grid, L[i], color=MID_BLUE, lw=0.7 if band else 1.1,
                alpha=0.45 if band else 0.95, zorder=2)
    if band:
        # drawn over the thin lines (half see-through, with thin edges) so that
        # even a narrow band - many trees in the crowd - stays visible
        mid, sd = L.mean(0), L.std(0)
        ax.fill_between(grid, mid - sd, mid + sd, color=LIGHT_BLUE, alpha=0.8,
                        lw=0, zorder=2.5)
        for edge in (mid - sd, mid + sd):
            ax.plot(grid, edge, color=MID_BLUE, lw=0.5, zorder=2.6)
        ax.plot(grid, mid, color=DARK, lw=2.0, zorder=3)
    ax.plot(grid, wave_truth(grid), ls="--", color=INK, lw=1.5, zorder=4)
    ax.set_xlim(*WAVE_RANGE)
    ax.set_ylim(-1.3, 3.0)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y (made-up number)")


def draw_crowd(ax, n_trees=1, lines=5, points=False, band=True) -> None:
    """The crowd: each line = the average of n_trees deep trees, each grown on a
    bootstrap sample (rows drawn again with replacement) of one training sample;
    one line per training sample (lines=1..20). Dashed = the true curve.
    band=True: light band = mean +/- SD of the 20 samples' lines (the spread),
    dark line = their mean (the middle). points=True also shows the 80 points
    of sample 1."""
    _, crowd, _ = _e4_paths()
    _draw_wave(ax, crowd, n_trees, lines, points,
               f"crowd: {n_trees} deep tree{'s' if n_trees > 1 else ''} averaged", band)


def draw_relay(ax, n_trees=1, lines=5, points=False, band=True) -> None:
    """The relay: start at the average y, then add n_trees stumps one after
    another, each a step of 0.3 towards what is still wrong; one line per
    training sample. Dashed = the true curve; band and dark line as in
    draw_crowd."""
    _, _, relay = _e4_paths()
    _draw_wave(ax, relay, n_trees, lines, points,
               f"relay: {n_trees} stump{'s' if n_trees > 1 else ''} added", band)


def crowd_and_relay_figure(n_trees=1, figsize=None, suptitle=True):
    n_trees = int(min(max(round(float(n_trees)), 1), 300))
    with _style():
        fig, axes = _new_fig(figsize or (8.0, 4.8), 2, 2, width_ratios=[4, 1])
        draw_crowd(axes[0, 0], n_trees)
        draw_relay(axes[1, 0], n_trees)
        axes[0, 0].set_xlabel("")
        axes[0, 0].sharex(axes[1, 0])
        axes[0, 0].tick_params(labelbottom=False)
        for row, kind in enumerate(("crowd", "relay")):
            draw_meters(axes[row, 1], kind, n_trees)
        axes[0, 1].set_title("meters", fontsize=9.5)
        fig.legend(handles=[Line2D([], [], color=MID_BLUE, lw=1.0,
                                   label="one line per training sample (5 of 20)"),
                            Rectangle((0, 0), 1, 1, fc=LIGHT_BLUE, ec="none",
                                      label="band: how far apart the 20 lines are"),
                            Line2D([], [], color=DARK, lw=2.0, label="their middle"),
                            Line2D([], [], color=INK, ls="--", lw=1.5, label="true curve")],
                   loc="outside lower center", ncol=2, fontsize=8.5, handlelength=1.8,
                   columnspacing=1.2)
        if suptitle:
            fig.suptitle("spread = how wide the band is;  miss = how far the middle is "
                         "from the true curve", fontsize=9.5)
    return fig


def crowd_and_relay(n_trees=1) -> None:
    """E4: one slider for both panels: trees averaged (crowd) / stumps added (relay).
    Another value is moved to the nearest one on the list, with a note."""
    _show(crowd_and_relay_figure(_snap(n_trees, E4_N, "n_trees", 1)))


# ---------------------------------------------------------------------------
# Tournament, out-of-fold boosting, permutation importance
# ---------------------------------------------------------------------------
def _boosting_cv(leak: bool = False) -> dict:
    """Boosting in the five rounds: out-of-fold p, the five validation AUCs, and
    the round-1 model with its validation rows (for permutation importance)."""
    def build():
        from sklearn.metrics import roc_auc_score
        d = course_data()
        X = d.X_train.assign(collections_contact=d.df.loc[d.X_train.index, "collections_contact"]) \
            if leak else d.X_train
        yv = d.y_train.to_numpy()
        oof = np.zeros(len(yv))
        aucs, first = [], None
        for k, (tr, va) in enumerate(d.folds.split(X, yv)):
            m = model("boosting").fit(X.iloc[tr], yv[tr])
            oof[va] = m.predict_proba(X.iloc[va])[:, 1]
            aucs.append(roc_auc_score(yv[va], oof[va]))
            if k == 0:
                first = (m, X.iloc[va], yv[va])
        return {"oof": oof, "y": yv, "aucs": np.array(aucs), "round1": first}
    return _cached(("boosting", leak), 4 if leak else 3,
                   "boosting in 5 rounds" + (" with collections_contact" if leak else ""), build)


def boosting_oof():
    """Out-of-fold p of boosting for the 21,000 training clients (each p comes
    from the round in which that client was in the validation part) and y."""
    b = _boosting_cv(False)
    return b["oof"], b["y"]


def tournament_scores() -> pd.DataFrame:
    """Validation AUC of the four models in the same five rounds."""
    def build():
        from sklearn.model_selection import cross_val_score
        d = course_data()
        out = {"logistic": logistic_cv(),
               "tree (depth 5)": cross_val_score(model("tree"), d.X_train, d.y_train,
                                                 cv=d.folds, scoring="roc_auc"),
               "forest": cross_val_score(model("forest"), d.X_train, d.y_train, cv=d.folds,
                                         scoring="roc_auc"),
               "boosting": _boosting_cv(False)["aucs"]}
        return pd.DataFrame(out, index=pd.Index(range(1, 6), name="round"))
    return _cached("tournament", 20, "the tournament: 4 models x 5 rounds", build)


def tournament_summary() -> pd.DataFrame:
    """Mean and spread (SD of the five rounds, numpy's std, ddof=0) per model."""
    t = tournament_scores()
    return pd.DataFrame({"mean": t.mean(), "spread": t.std(ddof=0),
                         "rounds ahead of logistic": (t.gt(t["logistic"], axis=0)).sum()})


def _perm(m, X, y, repeats=5) -> pd.Series:
    from sklearn.inspection import permutation_importance
    r = permutation_importance(m, X, y, scoring="roc_auc", n_repeats=repeats,
                               random_state=SEED)
    return pd.Series(r.importances_mean, index=X.columns).sort_values(ascending=False)


SETUPS = ["honest", "+ collections_contact", "copies: split by row", "copies: split by client"]


def _copies_split(by: str) -> dict:
    """Forest on the copies data, one 70/30 split by row or by client."""
    def build():
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import GroupShuffleSplit, train_test_split
        X_rows, y_rows, client = copies_data()
        if by == "row":
            a, b = train_test_split(np.arange(len(X_rows)), test_size=0.3, random_state=SEED,
                                    stratify=y_rows)
        else:
            a, b = next(GroupShuffleSplit(1, test_size=0.3, random_state=SEED).split(
                X_rows, y_rows, client))
        m = model("forest").fit(X_rows.iloc[a], y_rows.iloc[a])
        auc = roc_auc_score(y_rows.iloc[b], m.predict_proba(X_rows.iloc[b])[:, 1])
        shared = np.isin(client.iloc[b].to_numpy(), client.iloc[a].to_numpy()).mean()
        imp = _perm(m, X_rows.iloc[b], y_rows.iloc[b], repeats=3)
        return {"auc": auc, "shared": shared, "importance": imp}
    return _cached(("copies", by), 3, f"the forest on the copies, split by {by}", build)


def copies_one_row() -> float:
    """The same 6,000 clients, one row each, one 70/30 split: the forest's AUC."""
    def build():
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import train_test_split
        X_rows, y_rows, _ = copies_data()
        base, yb = X_rows.iloc[:6000], y_rows.iloc[:6000]
        a, b = train_test_split(np.arange(6000), test_size=0.3, random_state=SEED, stratify=yb)
        m = model("forest").fit(base.iloc[a], yb.iloc[a])
        return roc_auc_score(yb.iloc[b], m.predict_proba(base.iloc[b])[:, 1])
    return _cached("copies_one", 1, "the forest on one row per client", build)


def importance(setup: str = "honest", top=None) -> pd.Series:
    """Permutation importance: shuffle one column in the validation rows and
    measure how much the AUC drops. Boosting, round 1 of the five (honest or
    with collections_contact); for the copies setups: the forest on its
    held-out 30%. Another setup name -> "honest", with a note."""
    setup = _setup_name(setup)
    if setup in ("honest", "+ collections_contact"):
        b = _boosting_cv(setup != "honest")
        key = ("imp", setup)
        imp = _cached(key, 2, "the permutation importance", lambda: _perm(*b["round1"]))
    else:
        imp = _copies_split("row" if setup.endswith("row") else "client")["importance"]
    return imp if top is None else imp.head(top)


# ---------------------------------------------------------------------------
# E5 - Bend the probabilities
# ---------------------------------------------------------------------------
E5_BENDS = [0.5, 0.7, 1, 1.5, 2]


def reliability_points(p, y, groups: int = 10) -> pd.DataFrame:
    """Sort the clients by p, cut them into equal groups; per group the average p
    and the share who defaulted. p and y may be arrays, lists or pandas Series."""
    p, y = np.asarray(p, dtype=float), np.asarray(y, dtype=float)
    order = np.argsort(p, kind="stable")
    rows = [{"average p": p[ix].mean(), "share defaulted": y[ix].mean(), "clients": len(ix)}
            for ix in np.array_split(order, groups)]
    return pd.DataFrame(rows)


def _bent(bend):
    p, y = boosting_oof()
    return p ** bend, y


def bend_numbers() -> pd.DataFrame:
    from sklearn.metrics import roc_auc_score
    rows = []
    for b in E5_BENDS:
        q, y = _bent(b)
        r = reliability_points(q, y)
        gap = (r["share defaulted"] - r["average p"]).mean()
        rows.append({"bend": b, "AUC": roc_auc_score(y, q), "predicted count": q.sum(),
                     "actual count": int(y.sum()), "mean gap (actual - predicted)": gap})
    return pd.DataFrame(rows).set_index("bend")


def _bend_label(bend) -> str:
    return "p unchanged" if bend == 1 else f"every p becomes $p^{{{bend:g}}}$"


def draw_reliability(ax, bend=1, notes: bool = True) -> None:
    """Reliability diagram: 10 groups of 2,100 clients sorted by p; x = their
    average p, y = the share who defaulted. Diagonal = honest. notes=True writes
    what above and below the diagonal mean."""
    q, y = _bent(bend)
    r = reliability_points(q, y)
    ax.plot([0, 1], [0, 1], ls="--", color=GRAY, lw=1)
    ax.plot(r["average p"], r["share defaulted"], "-o", color=RED, ms=4, lw=1.4)
    if notes:
        ax.text(0.03, 0.97, "above:\nmore defaults\nthan predicted", transform=ax.transAxes,
                fontsize=8.5, color=MUTED, va="top", linespacing=1.0)
        ax.text(0.97, 0.03, "below:\nfewer defaults\nthan predicted", transform=ax.transAxes,
                fontsize=8.5, color=MUTED, ha="right", va="bottom", linespacing=1.0)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel("average p of the group")
    ax.set_ylabel("share who defaulted")
    ax.set_title("is 20% really 20%?")


def draw_bend_roc(ax, bend=1) -> None:
    from sklearn.metrics import roc_auc_score, roc_curve
    q, y = _bent(bend)
    fpr, tpr, _ = roc_curve(y, q)
    ax.plot([0, 1], [0, 1], ls="--", color=GRAY, lw=1)
    ax.plot(fpr, tpr, color=DARK, lw=1.6)
    ax.text(0.97, 0.04, f"AUC\n{roc_auc_score(y, q):.3f}", transform=ax.transAxes, ha="right",
            fontsize=9.5, color=DARK, linespacing=1.0)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel("false-alarm rate")
    ax.set_ylabel("recall")
    ax.set_title("ranking: ROC")


def draw_count(ax, bend=1) -> None:
    """Predicted count (the sum of all p) against the actual number of defaulters."""
    q, y = _bent(bend)
    vals = [q.sum(), y.sum()]
    ax.bar([0, 1], vals, color=[DARK, RED], width=0.62)
    for k, v in enumerate(vals):
        ax.text(k, v + 150, f"{v:,.0f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["sum\nof p", "actual"], fontsize=8.5)
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(0, 11_000)
    ax.set_yticks([0, 5000, 10000])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_ylabel("defaulters")
    ax.set_title("counting")
    ax.tick_params(axis="x", length=0)


def bend_figure(bend=1, figsize=None, suptitle=True):
    figsize = figsize or (8.0, 3.2)
    with _style():
        fig, axes = _new_fig(figsize, 1, 3, width_ratios=[1, 1, 0.75])
        draw_reliability(axes[0], bend, notes=figsize[0] >= 6.5)
        draw_bend_roc(axes[1], bend)
        draw_count(axes[2], bend)
        if suptitle:
            fig.suptitle(f"Out-of-fold p of boosting: {_bend_label(bend)} (same order)",
                         fontsize=9.5)
    return fig


def bend_probabilities(bend=1) -> None:
    """E5: replace every p by p ** bend (0.5-2) - the order of the clients is kept."""
    _show(bend_figure(_positive(bend, 1, "bend")))


# ---------------------------------------------------------------------------
# D4 - Rare positives
# ---------------------------------------------------------------------------
D4_SHARES = [0.22, 0.10, 0.05, 0.02]
D3_SHARES = D4_SHARES   # the name before the demos were renumbered (20.10); older copies
D4_SUBSAMPLES = 20


def _rare_index(share: float, k: int):
    """Keep every repayer and a random subset of defaulters so that defaulters
    make up `share` of the rows (0.22 = all of them)."""
    p, y = boosting_oof()
    neg, pos = np.where(y == 0)[0], np.where(y == 1)[0]
    if share >= 0.22:
        return np.arange(len(y))
    n_pos = int(round(share * len(neg) / (1 - share)))
    rng = np.random.default_rng(SEED + k)
    return np.concatenate([neg, rng.choice(pos, n_pos, replace=False)])


def rare_positives_numbers() -> pd.DataFrame:
    """Mean AUC and AP (area under the precision-recall curve) of 20 random
    subsamples at each share of positives."""
    def build():
        from sklearn.metrics import average_precision_score, roc_auc_score
        p, y = boosting_oof()
        rows = []
        for s in D4_SHARES:
            a, ap = [], []
            for k in range(1 if s >= 0.22 else D4_SUBSAMPLES):
                ix = _rare_index(s, k)
                a.append(roc_auc_score(y[ix], p[ix]))
                ap.append(average_precision_score(y[ix], p[ix]))
            rows.append({"share of positives": s, "AUC": np.mean(a), "AP": np.mean(ap),
                         "coin AP": y[_rare_index(s, 0)].mean()})
        return pd.DataFrame(rows).set_index("share of positives")
    return _cached("d3", 1, "the rare-positive subsamples", build)


def _rare_curves(share):
    from sklearn.metrics import precision_recall_curve, roc_curve
    p, y = boosting_oof()
    ix = _rare_index(share, 0)
    fpr, tpr, _ = roc_curve(y[ix], p[ix])
    prec, rec, _ = precision_recall_curve(y[ix], p[ix])
    return fpr, tpr, prec, rec


def draw_rare_roc(ax, share=0.22) -> None:
    tab = rare_positives_numbers()
    f0, t0, _, _ = _rare_curves(0.22)
    f, t, _, _ = _rare_curves(share)
    ax.plot([0, 1], [0, 1], ls="--", color=GRAY, lw=1)
    if share < 0.22:
        ax.plot(f0, t0, color=LIGHT_BLUE, lw=2.4, label="22% positives")
    ax.plot(f, t, color=DARK, lw=1.5, label=f"{share:.0%} positives")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("false-alarm rate")
    ax.set_ylabel("recall")
    ax.set_title(f"ROC: AUC {tab.loc[share, 'AUC']:.3f}")
    ax.legend(loc="lower right", fontsize=8.5)


def draw_rare_pr(ax, share=0.22) -> None:
    tab = rare_positives_numbers()
    _, _, p0, r0 = _rare_curves(0.22)
    _, _, p, r = _rare_curves(share)
    coin = tab.loc[share, "coin AP"]
    if share < 0.22:
        ax.plot(r0, p0, color=LIGHT_BLUE, lw=2.4, label="22% positives")
    ax.plot(r, p, color=DARK, lw=1.5, label=f"{share:.0%} positives")
    ax.axhline(coin, ls=":", color=GRAY, lw=1.2)
    ax.text(0.02, coin + 0.012, f"coin: {coin:.2f}", ha="left", va="bottom", fontsize=8.5,
            color=MUTED, transform=ax.get_yaxis_transform())
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title(f"PR curve: AP {tab.loc[share, 'AP']:.3f}")
    ax.legend(loc="upper right", fontsize=8.5)


def rare_positives_figure(share=0.22, figsize=None, suptitle=True):
    share = _snap(share, D4_SHARES, "share", 0.22, quiet=True)
    with _style():
        fig, axes = _new_fig(figsize or (7.0, 3.6), 1, 2)
        draw_rare_roc(axes[0], share)
        draw_rare_pr(axes[1], share)
        if suptitle:
            fig.suptitle("Out-of-fold p of boosting, all training clients (22% defaulted)"
                         if share >= 0.22 else
                         f"Defaulters thinned to {share:.0%} of the rows "
                         f"(mean of {D4_SUBSAMPLES} draws)", fontsize=9.5)
    return fig


def rare_positives(share=0.22) -> None:
    """D4: make defaulters rare (22%, 10%, 5%, 2%): ROC holds, precision-recall falls."""
    _show(rare_positives_figure(_snap(share, D4_SHARES, "share", 0.22)))


# ---------------------------------------------------------------------------
# D5 - Pricing the threshold
# ---------------------------------------------------------------------------
D5_T = np.round(np.arange(0.01, 1.0, 0.01), 2)


def price_numbers(miss_cost=3, false_alarm_cost=1, bend=1) -> dict:
    """Average cost per client of "decline when p > t" on the out-of-fold p:
    the cheapest t, the formula t* = false_alarm_cost / (false_alarm_cost +
    miss_cost), and the costs at both and at 0.5."""
    q, y = _bent(bend)
    pos, neg = y == 1, y == 0

    def cost_at(t):
        # decline when p > t; a miss = a defaulter approved, a false alarm = a repayer declined
        misses, false_alarms = ((q <= t) & pos).sum(), ((q > t) & neg).sum()
        return (miss_cost * misses + false_alarm_cost * false_alarms) / len(y)

    costs = np.array([cost_at(t) for t in D5_T])
    t_star = false_alarm_cost / (false_alarm_cost + miss_cost)
    i = int(np.argmin(costs))
    return {"t": D5_T, "cost": costs, "cheapest t": float(D5_T[i]), "cheapest cost": costs[i],
            "t*": t_star, "cost at t*": cost_at(t_star), "cost at 0.5": cost_at(0.5),
            "extra at t*": cost_at(t_star) / costs[i] - 1}


def draw_prices(ax, miss_cost=3, false_alarm_cost=1, bend=1) -> dict:
    r = price_numbers(miss_cost, false_alarm_cost, bend)
    # the y-axis shows the region around the cheapest cost (expensive ends run off the top)
    top = min(r["cost"].max(), max(2.5 * r["cheapest cost"], 1.15 * r["cost at t*"],
                                   1.15 * r["cost at 0.5"]))
    ax.plot(r["t"], r["cost"], color=DARK, lw=1.8, zorder=3)
    ax.axvline(r["t*"], color=DARK, ls="--", lw=1)
    ax.axvline(0.5, color=GRAY, ls=":", lw=1)
    ax.text(0.51, top * 0.03, "0.5", fontsize=8.5, color=MUTED)
    left = r["t*"] > 0.6
    ax.text(r["t*"] + (-0.012 if left else 0.012), top * 1.02,
            f"formula t* = {r['t*']:.2f}\ncosts {r['cost at t*']:.3f} "
            f"(+{r['extra at t*']:.1%})" if r["extra at t*"] < 0.1 else
            f"formula t* = {r['t*']:.2f}\ncosts {r['cost at t*']:.3f} (+{r['extra at t*']:.0%})",
            va="top", ha="right" if left else "left", fontsize=8.5, color=DARK,
            bbox=dict(fc="white", ec="none", pad=0.3, alpha=0.85))
    ax.plot(r["t*"], r["cost at t*"], "o", ms=6, mfc="white", mec=DARK, mew=1.4, zorder=4)
    ax.plot(r["cheapest t"], r["cheapest cost"], "o", color=DARK, ms=6.5, zorder=5)
    ct = r["cheapest t"]   # near an edge, the label runs inwards (clear of the y axis)
    ha, dx = ("left", 6) if ct < 0.2 else (("right", -6) if ct > 0.8 else ("center", 0))
    ax.annotate(f"cheapest: t = {ct:.2f}, cost {r['cheapest cost']:.3f}",
                (ct, r["cheapest cost"]), xytext=(dx, -9),
                textcoords="offset points", ha=ha, va="top", fontsize=8.5, color=DARK,
                bbox=dict(fc="white", ec="none", pad=0.3))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, top * 1.05)
    ax.set_xlabel("threshold t (decline when p > t)")
    ax.set_ylabel("average cost per client")
    ax.set_title(f"a miss costs {miss_cost:g}, a false alarm {false_alarm_cost:g}"
                 + ("" if bend == 1 else f"; {_bend_label(bend)}"))
    return r


def prices_figure(miss_cost=3, false_alarm_cost=1, bend=1, figsize=None, suptitle=True):
    with _style():
        fig, ax = _new_fig(figsize or (6.5, 3.2))
        draw_prices(ax, miss_cost, false_alarm_cost, bend)
        if suptitle:
            fig.suptitle("Out-of-fold p of boosting, 21,000 training clients", fontsize=9.5)
    return fig


def prices(miss_cost=3, false_alarm_cost=1, bend=1) -> None:
    """D5: the cost of every threshold for these prices; bend the p and the
    cheapest t runs away from the formula."""
    miss_cost = _positive(miss_cost, 3, "miss_cost")
    false_alarm_cost = _positive(false_alarm_cost, 1, "false_alarm_cost")
    bend = _positive(bend, 1, "bend")
    _show(prices_figure(miss_cost, false_alarm_cost, bend))


# ---------------------------------------------------------------------------
# E6 - Too good to be true
# ---------------------------------------------------------------------------
_E6_TRIED: list = []


def _setup_result(setup: str) -> dict:
    if setup == "honest":
        a = _boosting_cv(False)["aucs"]
        return {"score": a.mean(), "spread": a.std(), "rounds": a, "kind": "boosting, CV"}
    if setup == "+ collections_contact":
        a = _boosting_cv(True)["aucs"]
        return {"score": a.mean(), "spread": a.std(), "rounds": a, "kind": "boosting, CV"}
    r = _copies_split("row" if setup.endswith("row") else "client")
    return {"score": r["auc"], "spread": np.nan, "rounds": None, "kind": "forest, one split",
            "shared": r["shared"]}


def too_good_numbers() -> pd.DataFrame:
    rows = []
    for s in SETUPS:
        r = _setup_result(s)
        imp = importance(s)
        rows.append({"setup": s, "model": r["kind"], "AUC": r["score"], "spread": r["spread"],
                     "rounds": None if r["rounds"] is None else np.round(r["rounds"], 3).tolist(),
                     "held-out rows with a copy in training": r.get("shared", np.nan),
                     "top column": imp.index[0], "its importance": imp.iloc[0],
                     "next importance": imp.iloc[1]})
    out = pd.DataFrame(rows).set_index("setup")
    out.attrs["one row per client"] = copies_one_row()
    return out


def _setup_name(setup) -> str:
    """One of SETUPS for what was typed (case, spaces, "+" and ":" do not
    matter); anything else -> "honest", with a note."""
    if setup in SETUPS:
        return setup
    def key(v):
        return "".join(ch for ch in str(v).lower() if ch.isalnum())
    for s in SETUPS:
        if key(s) == key(setup):
            return s
    _say(f"setup {setup!r} is not one of the four ("
         + ", ".join(f'"{s}"' for s in SETUPS) + '): showing "honest".')
    return "honest"


_SETUP_LABEL = {"honest": "honest", "+ collections_contact": "+ collections_contact",
                "copies: split by row": "copies, split by row",
                "copies: split by client": "copies, split by client"}


def draw_scoreboard(ax, tried, current=None) -> None:
    """One bar per setup tried (dotted slots for the others), its name above it;
    the current setup dark. White dots = the five rounds of a cross-validated setup."""
    ys = {s: len(SETUPS) - 1 - k for k, s in enumerate(SETUPS)}
    for s in SETUPS:
        y = ys[s]
        ax.text(0.502, y + 0.24, _SETUP_LABEL[s], va="bottom", fontsize=8.5,
                color=INK if s in tried else MUTED)
        if s not in tried:
            ax.add_patch(Rectangle((0.5, y - 0.2), 0.5, 0.4, fill=False, ec=GRAY, ls=":", lw=1))
            ax.text(0.75, y, "not tried yet", ha="center", va="center", fontsize=8.5, color=MUTED,
                    zorder=3, bbox=dict(fc="white", ec="none", pad=0.15))
            continue
        r = _setup_result(s)
        ax.barh(y, r["score"] - 0.5, left=0.5, height=0.4,
                color=DARK if s == current else MID_BLUE)
        if r["rounds"] is not None:
            ax.plot(r["rounds"], np.full(5, y), "o", ms=4, mfc="white", mec=INK, mew=0.8,
                    zorder=4)
        ax.text(0.51, y, f"{r['score']:.3f}", va="center", fontsize=9, color="white",
                fontweight="bold")
    if any(s.startswith("copies") for s in tried):
        one = copies_one_row()
        ax.plot([one, one], [-0.3, 1.25], ls="--", color=INK, lw=1)
        ax.text(one + 0.006, -0.62, f"one row per client: {one:.3f}",
                ha="left", va="center", fontsize=8.5, color=INK)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_ylim(-0.85, len(SETUPS) - 0.25)
    ax.set_xlim(0.5, 1.0)
    ax.set_xticks([0.5, 0.75, 1.0])
    ax.set_xlabel("AUC, held-out rows")
    ax.set_title("the setups you tried")


def draw_importance(ax, setup: str = "honest", top: int = 5) -> None:
    """The top columns by permutation importance (AUC drop when shuffled);
    collections_contact in dark."""
    imp = importance(setup, top)
    names = [LABELS.get(c, c) for c in imp.index][::-1]
    vals = imp.to_numpy()[::-1]
    ax.barh(names, vals, color=[DARK if c == "collections_contact" else GRAY
                                for c in imp.index][::-1], height=0.6)
    for k, v in enumerate(vals):
        ax.text(v + 0.004, k, f"{v:.3f}", va="center", fontsize=8.5)
    ax.set_xlim(0, max(0.12, vals.max() * 1.35))
    ax.xaxis.set_major_locator(MaxNLocator(3))
    ax.set_xlabel("AUC drop when the\ncolumn is shuffled", linespacing=1.0)
    ax.tick_params(axis="y", length=0)
    ax.set_title("columns it uses")


_SETUP_HOW = {"honest": "boosting, 5-fold CV on the training rows",
              "+ collections_contact": "boosting, 5-fold CV on the training rows",
              "copies: split by row": "forest, 6,000 clients \u00d7 3 rows, one 70/30 split",
              "copies: split by client": "forest, 6,000 clients \u00d7 3 rows, one 70/30 split"}


def too_good_figure(setup: str = "honest", tried=None, figsize=None, suptitle=True):
    setup = setup if setup in SETUPS else "honest"
    tried = [s for s in tried if s in SETUPS] if tried is not None else [setup]
    if setup not in tried:
        tried.append(setup)
    with _style():
        fig, axes = _new_fig(figsize or (8.0, 3.2), 1, 2, width_ratios=[1, 1])
        draw_scoreboard(axes[0], tried, setup)
        draw_importance(axes[1], setup)
        if suptitle:
            fig.suptitle(f"{setup}: {_SETUP_HOW[setup]}", fontsize=9.5)
    return fig


def too_good(setup: str = "honest") -> None:
    """E6: pick a setup; the scoreboard keeps every setup you tried
    (too_good_reset() clears it)."""
    setup = _setup_name(setup)
    if setup not in _E6_TRIED:
        _E6_TRIED.append(setup)
    _show(too_good_figure(setup, list(_E6_TRIED)))


def too_good_reset() -> None:
    _E6_TRIED.clear()
