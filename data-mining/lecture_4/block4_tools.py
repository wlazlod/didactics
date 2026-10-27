"""block4_tools.py - the experiments of Block 4 (PCA and clustering).

Data Mining, SGH Warsaw School of Economics (223121-0090).

The Block 4 notebook imports this file as

    import block4_tools as lab

Each experiment is ONE function with ONE control (the value you change), and
it draws ONE figure. The slides are drawn by the same functions, so the deck
shows exactly the pictures you make in the notebook. You do not need to read
this file; the notebook shows the few scikit-learn lines behind each
experiment ("The code behind it").

Everything is deterministic and uses the course split of Block 2: the 12
columns of ``taiwan_features``, 70% training / 30% test, stratified,
``random_state=42`` (21,000 / 9,000 clients). Clustering uses five columns
of the training rows (``five_columns``), each standardised with a scaler
fitted on the training rows only. Default is never used to build or number a
group: it is read only afterwards, as the outside check (Experiment 5's
right panel, ``outside_check``, Demo 5). Test rows are read only by
``segments`` / ``outside_check`` (assigned to the nearest centre with
``predict``); nothing here is chosen on them. Slow results are computed once,
with a line "Computing ... once (a few seconds ...)", and kept in memory until
the kernel restarts.

Three worlds: the ten made-up clients A-J (two bills, thousand NT$), the
made-up crescents with 8 stray points, and our data. A fourth, 300 random
points, appears on one slide only. Before our clients' segments, 178 wines
(scikit-learn's bundled copy of the UCI Wine data, no download) show the recipe
where real groups exist.

In the notebook (each slider cell has a plain twin: set the value, re-run)
---------------------------------------------------------------------------
    lab.setup_check()      "Setup OK ...", a version note if needed, "Sliders OK"
    lab.WORLD              the ten clients
    E1  lab.turn_direction(angle=0);          lab.slider(lab.turn_direction, angle=lab.E1_ANGLES)
    D1  lab.pca_columns("bills"); lab.pca_columns("payments")
    E2  lab.start_kmeans(start=0);            lab.slider(lab.start_kmeans, start=lab.E2_STARTS)
        done early: lab.start_kmeans(start=0, n_init=10)
    W   lab.wine_groups(standardise=True)  the 178 wines: elbow, PC1-PC2,
        each group's wines by variety; done early: standardise=False
    E3  lab.units_kmeans(units="NT$ and months"); lab.slider(lab.units_kmeans, units=lab.E3_UNITS)
        done early: lab.units_kmeans(units="standardised", k=4)
    lab.five_columns().describe()
    D2  lab.elbow_ours()
    D3  lab.segment_table()
    D4  lab.dendrogram_ours(linkage="complete")
    E4  lab.grow_dbscan(eps=0.2);             lab.slider(lab.grow_dbscan, eps=lab.E4_EPS)
        done early: lab.grow_dbscan(eps=0.2, min_samples=3)  (or 10)
    E5  lab.clusters_in_noise(data="ours");   lab.slider(lab.clusters_in_noise, data=lab.E5_DATA)
    D5  lab.stability_and_outside()
    Homework A: lab.check("A1", shadows_a=[...], ...)   (lab.check("A1") lists what it takes)

Public API
----------
Setup and sliders (as in Block 3)
    setup_check()                loads the data; prints "Setup OK: ...", the version note
                                 (if any) and sliders_status().
    sliders_ok() -> bool; sliders_status() -> str; version_note() -> str ("" from
                                 scikit-learn 1.8 on; a calm Note for 1.4-1.7; a WARNING below 1.4)
    slider(func, **controls)     ipywidgets.interact (redraws when you let go). (lo, hi, step)
                                 -> slider; a list of numbers -> slider over them; a list of
                                 strings -> dropdown. Without ipywidgets: a line saying to use
                                 the plain cell.
    VERBOSE = True               False silences the "Computing ..." lines (make_figures.py).
    clear_cache()                forget cached results and the E1 / E2 trails.
    SEED = 42; colour tokens BLUE RED DARK GRAY LIGHT LIGHT_RED LIGHT_BLUE MID_BLUE INK
    MUTED SHADE; STYLE; LABELS (plain names); TOY_COLOUR (A-D blue, E-H and J red, I grey);
    SEG_COLOURS (segments 1-4: BLUE, MID_BLUE, LIGHT_RED, RED).

Data
    course_data() -> CourseData  .df, .X, .y, .X_train, .X_test, .y_train, .y_test (copies
                                 are the caller's job: the notebook takes .copy())
    WORLD                        DataFrame, index A-J, columns august, september (thousand NT$)
    five_columns(rows="train")   log_limit, card_use, months_late, pay_ratio, bill_trend
    standardised_five(rows="train")  the same, standardised with the training rows' scaler
    FIVE_ABOUT                   what each of the five columns is (slide 48)
    bills(rows="train"), payments(rows="train")   the six monthly columns, NT$, Sep -> Apr
    crescents_with_strays() -> X (308 x 2), crescent (0 / 1; -1 for the 8 strays)
    random_points() -> 300 x 2 uniform points (slide 65)
    segments() -> train_labels, test_labels, profile   (four segments, numbered 1-4 by the
                                 target-free rule below; profile = slide 50's columns)
    pca_fixed(Z, which) -> fitted PCA with fixed signs ("ten", "bills", "payments", "five",
                                 "model"); its transform gives the sign-fixed scores
    null_tables() -> {"ours", "shuffled", "noise"}: three 21,000 x 5 arrays
    agree(a, b) -> share of rows in the same group after matching the group numbers

Target-free rules (never scikit-learn's label order)
    segments: 1 = highest mean payment ratio, 4 = highest mean months late, 3 = the higher
              median card use of the other two, 2 = the last one.
    PC signs: the ten: PC1 positive on September, PC2 positive on August; bills and
              payments: PC1 with a positive sum of loadings, PC2 positive on September;
              five columns: PC1 positive on card use, PC2 positive on months late; the
              12 model columns: positive sums.
    other groups (Experiment 3, null tables, the tree of Demo 4): numbered by size,
              largest first. Groups of the ten clients are coloured by position: the
              group with the smallest mean bills blue, the largest red, the middle grey.
    wines: "strong" = the highest median alcohol, "dark" = the higher
           median colour intensity of the other two, "light" = the last
           one (standardised); without standardising: by size, largest
           first. PC signs: PC1 positive on flavanoids, PC2 positive on
           colour intensity.

Experiments (X(control) shows the figure and returns None; X_figure(control, ...,
figsize=None, suptitle=True) -> Figure; draw_*(ax, ...) single panels; X_numbers())
    E1 turn_direction(angle=0)           E1_ANGLES = (0, 180, 1); keeps a trail of the angles
                                         tried (turn_direction_reset()).
       draw_direction(ax, angle); draw_angle_curve(ax, angle, trail=None, full=False)
       shadows(direction) -> Series; shadow_variance(angle) -> float;
       turn_direction_numbers(step=15) -> DataFrame
    E2 start_kmeans(start=0, n_init=1)   E2_STARTS = 0-19; start s picks three clients with
                                         np.random.RandomState(s).choice(10, 3, replace=False,
                                         p=np.full(10, 0.1)), then KMeans(3, init=those,
                                         n_init=1); trail of the starts tried
                                         (start_kmeans_reset()). n_init > 1: KMeans(3,
                                         init="random", n_init=n_init, random_state=start) -
                                         n_init random-client starts, the smallest inertia kept;
                                         only the final centres are drawn.
       draw_start_groups(ax, start, n_init=1); draw_start_bars(ax, start, trail, dark_best)
       start_clients(start); start_kmeans_numbers(); groupings_found(); draw_grouping(ax, labels)
    E3 units_kmeans(units="NT$ and months", k=3)   E3_UNITS; fit on all 21,000 training
                                         clients (limit in NT$, months late), 2,000 drawn
       draw_units_scatter(ax, units, k=3); units_kmeans_numbers(k=3)
    E4 grow_dbscan(eps=0.2, min_samples=5)   E4_EPS; left k-means K = 2, right DBSCAN
       draw_kmeans_crescents(ax); draw_dbscan(ax, eps, min_samples); grow_dbscan_numbers()
    E5 clusters_in_noise(data="ours")    E5_DATA = "ours", "shuffled", "noise"; K = 2-8,
                                         KMeans(n_init=10, random_state=42), silhouette on
                                         sample_size=5000, random_state=42; each table is
                                         computed once and cached.
       draw_null_silhouettes(ax, data, compare); draw_null_defaults(ax, tables);
       clusters_in_noise_numbers()

Demos (replay at home)
    W  wine_groups(standardise=True)     178 wines, 13 standardised columns,
                                         KMeans(3, n_init=10, random_state=42);
                                         returns wine_table(standardise)
    D1 pca_columns(which="bills")        scree, loadings, biplot; returns pca_table(which)
    D2 elbow_ours()                      inertia K = 1-10; returns elbow_numbers()
    D3 segment_table()                   segments on PC1-PC2; returns the profile table
    D4 dendrogram_ours(linkage="complete")   300 clients (default_rng(42)); returns the
                                         crosstab tree group x segment
    D5 stability_and_outside()           other seeds, five random halves (own
                                         default_rng(42)), default per segment on training and
                                         test rows; returns outside_check()

Homework A (by hand)
    HW_CLIENTS (clients 1-6), HW_A3, HW_A3_SD, HW_A5_INERTIA, HW_LINE
    check(item, **answers)               soft check: says right / not yet with a hint; never
                                         raises; check(item) lists the answers it takes
    homework_answers(item) -> dict       the worked numbers (for the solutions notebook)

Slide drawing helpers (make_figures.py; each returns the numbers it shows)
    draw_world(ax, colours=True), draw_world_table(ax, centred=False), draw_centred(ax),
    draw_shadows(ax, direction), draw_three_directions(axes), draw_pc1_scores(ax),
    draw_pc_lines(ax), draw_pc_coords(ax), draw_pve(ax), draw_distance(ax),
    draw_kmeans_goal(ax), draw_kmeans_step(ax, step "1" | "2" | "3a" | "3b"),
    draw_inertia_steps(ax), draw_two_starts(axes), draw_toy_elbow(ax),
    draw_units_toy(axes), draw_merges(ax), draw_linkage(ax),
    draw_dendrogram_toy(ax, stage "a" | "b"), draw_eps_circles(ax),
    draw_core_border_noise(ax), draw_dbscan_grow(ax), draw_silhouette_client(ax, who="I"),
    draw_silhouette_bars(ax), draw_noise_kmeans(ax);
    real data: draw_scree(ax, which), draw_bills_vs_payments(axes), draw_pc1_default(ax),
    draw_loadings(ax, which), draw_biplot(ax, which), draw_elbow_ours(ax),
    draw_segments_pcs(ax), draw_silhouette_ours(ax), draw_default_by_segment(ax),
    draw_stability(ax), draw_dendrogram_ours(ax, linkage),
    draw_wine_elbow(ax), draw_wine_pcs(ax), draw_wine_check(ax)
    numbers: toy_pca(), toy_kmeans_run(start), toy_elbow(), toy_units_groups(units),
    toy_linkage(method), toy_dbscan(eps, min_samples), toy_silhouette(k=2),
    pca_table(which), pc1_default_numbers(), elbow_numbers(), segment_centres_pcs(),
    outside_check(), stability_numbers(), hierarchy_numbers(linkage),
    wine_numbers(), wine_table()

All figure text is at least 8.5 pt. Every "$" in a figure string is written
"\\$" (matplotlib would read a pair of them as mathematics).
"""
from __future__ import annotations

import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator, PercentFormatter

try:  # Colab: finance_data.py uploaded next to the notebook
    import finance_data as _finance_data
except ImportError:  # course folder: the loader lives in ../lecture_1
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lecture_1"))
    import finance_data as _finance_data
from finance_data import load_taiwan, taiwan_features

SEED = 42
VERBOSE = True

# ---------------------------------------------------------------------------
# Colours and style (the course palette, as in Blocks 2-3)
# ---------------------------------------------------------------------------
BLUE = "#2E6DA4"
RED = "#C83C28"
DARK = "#003366"
GRAY = "#8C959E"
LIGHT = "#D7E3F0"
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
PCT0 = PercentFormatter(100, decimals=0)

FIVE = ["log_limit", "card_use", "months_late", "pay_ratio", "bill_trend"]
MONTHS = ["sep", "aug", "jul", "jun", "may", "apr"]
_MONTH_NAME = {"sep": "September", "aug": "August", "jul": "July", "jun": "June",
               "may": "May", "apr": "April"}

LABELS = {  # plain names, used on axes, bars and tables
    "log_limit": "credit limit (log)",
    "card_use": "card use",
    "months_late": "months late",
    "pay_ratio": "payment ratio",
    "bill_trend": "bill trend",
    "limit_bal": "credit limit",
    "months_delayed_6m": "months late",
    "utilisation": "card use",
    "august": "August bill",
    "september": "September bill",
    **{f"bill_amt_{m}": f"{_MONTH_NAME[m]} bill" for m in MONTHS},
    **{f"pay_amt_{m}": f"{_MONTH_NAME[m]} payment" for m in MONTHS},
    "pay_delay_recent": "latest status",
    "married": "married",
    "age": "age",
    "edu_graduate": "graduate school",
    "edu_high_school": "high school",
    "edu_other": "other education",
    "female": "female",
}

FIVE_ABOUT = pd.DataFrame(
    {"what it is": [
        "the most the client may owe; log10, so equal steps are equal ratios "
        "(50k -> 100k as far as 100k -> 200k)",
        "September bill / credit limit (capped at -0.5 and 2; below 0: the client paid "
        "more than was owed)",
        "in how many of the last six months the payment was late",
        "September payment / August bill (1 = paid in full; capped at 2; no August bill "
        "and nothing paid counts as 0)",
        "(September bill - April bill) / credit limit (capped at -2 and 2)"],
     "built from": ["limit_bal", "utilisation", "months_delayed_6m", "pay_ratio",
                    "bill_trend"]},
    index=pd.Index([LABELS[c] for c in FIVE], name="column"))

SEG_COLOURS = {1: BLUE, 2: MID_BLUE, 3: LIGHT_RED, 4: RED}
SEGMENT_NAMES = {1: "pays the whole bill", 2: "pays a small part or has no bill",
                 3: "card nearly full", 4: "often late"}
_CLUSTER_CYCLE = [BLUE, RED, DARK, MID_BLUE, LIGHT_RED, GRAY, "#5E8C4A", "#B07AA1",
                  "#E0A040", "#4BA3A3"]


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
    """A whole number with thousands separators: 10153 -> 10,153."""
    return f"{int(round(float(v))):,}"


def _two(v) -> str:
    """Two decimals without a negative zero."""
    return f"{round(float(v), 2) + 0.0:.2f}"


def _dist_text(d) -> str:
    """A distance as on the slides: 2.0, 1.41, 3.16."""
    d = float(d)
    return f"{d:.1f}" if abs(d - round(d, 1)) < 1e-9 else f"{d:.2f}"


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------
_CACHE: dict = {}


def _cached(key, what: str, fn, slow: bool = False):
    """fn() once per session; a slow one first prints a line saying so."""
    if key not in _CACHE:
        if VERBOSE and slow:
            print(f"Computing {what} once (a few seconds; kept until the kernel "
                  "restarts) ...", flush=True)
        _CACHE[key] = fn()
    return _CACHE[key]


def clear_cache() -> None:
    """Forget every cached result and the trails of Experiments 1 and 2."""
    _CACHE.clear()
    _E1_TRAIL.clear()
    _E2_TRAIL.clear()


# ---------------------------------------------------------------------------
# Sliders and the setup check (as in Block 3)
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


def setup_check() -> None:
    """The setup cell's lines: loads the data (and downloads it the first
    time), then prints "Setup OK: ...", the version note if there is one, and
    the sliders status."""
    import sklearn
    d = course_data()
    note = version_note()
    if note.startswith("WARNING"):
        print(note)
        print("Setup OK - but see the WARNING above")
    else:
        print(f"Setup OK: {len(d.X_train):,} training clients, {len(d.X_test):,} test "
              f"clients, ten made-up clients A-J - scikit-learn {sklearn.__version__}")
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
            return float(value.strip())
        except ValueError:
            return value
    return value


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and np.isfinite(v)


def _snap(value, allowed: list, what: str, default):
    """The allowed value nearest to `value` (never an error); not a number ->
    `default`. Prints one line when the value had to change."""
    value = _decimal_comma(_plain(value), what)
    v = _as_number(value)
    if _is_number(v):
        for a in allowed:
            if v == a:
                return a
        new = min(allowed, key=lambda a: (abs(a - v), a))
    else:
        new = default
    if len(allowed) > 4 and all(isinstance(a, int) for a in allowed) and \
            allowed == list(range(allowed[0], allowed[-1] + 1)):
        listed = f"{allowed[0]}-{allowed[-1]}"
    else:
        listed = ", ".join(f"{a:g}" for a in allowed)
    shown = f"{value:g}" if _is_number(value) else repr(value)
    _say(f"{what} {shown} is not on this experiment's list ({listed}): showing {new:g}.")
    return new


def _choice(value, choices: list, what: str, aliases: dict | None = None) -> str:
    """One of `choices` for what was typed (case, spaces and signs do not
    matter); anything else -> the first choice, with a note."""
    if value in choices:
        return value

    def key(v):
        return "".join(ch for ch in str(v).lower() if ch.isalnum())
    for c in choices:
        if key(c) == key(value):
            return c
    for k, c in (aliases or {}).items():
        if key(value) == k:
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
        # a <select> shows its options as plain text: "NT$" needs no escaping here
        return w.Dropdown(options=spec, value=default if default in spec else spec[0], **common)
    if isinstance(spec, list) and spec:
        options = [(f"{v:g}", v) for v in spec]
        values = [v for _, v in options]
        return w.SelectionSlider(options=options, value=default if default in values
                                 else values[0], **common)
    return w.fixed(spec)


def slider(func, **controls) -> None:
    """Draw func's figure with a slider (or dropdown) for each control.

    Example: lab.slider(lab.turn_direction, angle=lab.E1_ANGLES)
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
# Data: the course split, the five columns, the bills
# ---------------------------------------------------------------------------
@dataclass
class CourseData:
    df: pd.DataFrame        # the raw table (30,000 clients)
    X: pd.DataFrame         # the 12 features of taiwan_features
    y: pd.Series            # default (1) / repaid (0)
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


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
        return CourseData(df, X, y, X_tr, X_te, y_tr, y_te)
    return _cached("data", "the data", build)


def _rows(rows: str) -> str:
    return _choice(rows, ["train", "test"], "rows",
                   {"training": "train", "trainingrows": "train", "testrows": "test"})


def _five(X: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"log_limit": np.log10(X["limit_bal"].astype(float)),
                         "card_use": X["utilisation"],
                         "months_late": X["months_delayed_6m"],
                         "pay_ratio": X["pay_ratio"],
                         "bill_trend": X["bill_trend"]}, index=X.index)


def _real() -> dict:
    """The five columns of the training and test rows, the scaler (fitted on
    the training rows only) and the standardised arrays."""
    def build():
        from sklearn.preprocessing import StandardScaler
        d = course_data()
        F_tr, F_te = _five(d.X_train), _five(d.X_test)
        scaler = StandardScaler().fit(F_tr.to_numpy(float))
        return {"F_tr": F_tr, "F_te": F_te, "scaler": scaler,
                "Z_tr": scaler.transform(F_tr.to_numpy(float)),
                "Z_te": scaler.transform(F_te.to_numpy(float)),
                "y_tr": d.y_train.to_numpy(), "y_te": d.y_test.to_numpy()}
    return _cached("real", "the five columns", build)


def five_columns(rows: str = "train") -> pd.DataFrame:
    """The five clustering columns (slide 48) of the training or test rows:
    log_limit = log10(credit limit), card_use = utilisation, months_late =
    months_delayed_6m, pay_ratio, bill_trend. A copy."""
    r = _real()
    return (r["F_tr"] if _rows(rows) == "train" else r["F_te"]).copy()


def standardised_five(rows: str = "train") -> pd.DataFrame:
    """The five columns, standardised (minus the mean, divided by the SD) with
    the scaler fitted on the 21,000 training rows; test rows are transformed
    with that same scaler. A copy."""
    r = _real()
    train = _rows(rows) == "train"
    F = r["F_tr"] if train else r["F_te"]
    return pd.DataFrame(r["Z_tr"] if train else r["Z_te"], index=F.index, columns=FIVE)


def bills(rows: str = "train") -> pd.DataFrame:
    """The six monthly bills (NT$), September back to April. A copy."""
    d = course_data()
    idx = d.X_train.index if _rows(rows) == "train" else d.X_test.index
    return d.df.loc[idx, [f"bill_amt_{m}" for m in MONTHS]].copy()


def payments(rows: str = "train") -> pd.DataFrame:
    """The six monthly payments (NT$), September back to April. A copy."""
    d = course_data()
    idx = d.X_train.index if _rows(rows) == "train" else d.X_test.index
    return d.df.loc[idx, [f"pay_amt_{m}" for m in MONTHS]].copy()


def default_rate() -> float:
    """The share of the 21,000 training clients who defaulted (0.221)."""
    return float(_real()["y_tr"].mean())


def agree(a, b) -> float:
    """The share of rows that two groupings put in the same group, after the
    second grouping's numbers are renamed to agree with the first as often as
    possible (the Hungarian method on their crosstab)."""
    from scipy.optimize import linear_sum_assignment
    a, b = np.asarray(a), np.asarray(b)
    M = pd.crosstab(a, b).to_numpy()
    r, c = linear_sum_assignment(-M)
    return float(M[r, c].sum() / len(a))


def _by_size(labels) -> np.ndarray:
    """Group numbers 1, 2, ... by size, largest first (ties: the old order)."""
    labels = np.asarray(labels)
    ks, counts = np.unique(labels, return_counts=True)
    order = ks[np.argsort(-counts, kind="stable")]
    mp = {k: i + 1 for i, k in enumerate(order)}
    return np.array([mp[v] for v in labels])


# ---------------------------------------------------------------------------
# PCA with fixed signs
# ---------------------------------------------------------------------------
_SIGN_RULES = {  # (component, ("col", name / position) or "sum")
    "ten": [(0, ("september", 1)), (1, ("august", 0))],
    "bills": [(0, "sum"), (1, ("bill_amt_sep", 0))],
    "payments": [(0, "sum"), (1, ("pay_amt_sep", 0))],
    "five": [(0, ("card_use", 1)), (1, ("months_late", 2))],
    "model": [(0, "sum"), (1, "sum")],
}


def pca_fixed(Z, which: str = "five"):
    """PCA().fit(Z) with the component signs fixed by a rule (the sign of a
    component is arbitrary, and scikit-learn's can change between versions):
    "ten" - PC1 positive on September, PC2 positive on August; "bills" and
    "payments" - PC1 with a positive sum of loadings, PC2 positive on
    September; "five" - PC1 positive on card use, PC2 positive on months late;
    "model" (the 12 columns) - positive sums. Returns the fitted PCA; its
    .transform(Z) gives the sign-fixed scores."""
    from sklearn.decomposition import PCA
    which = _choice(which, list(_SIGN_RULES), "which",
                    {"world": "ten", "toy": "ten", "tenclients": "ten", "fivecolumns": "five",
                     "12": "model", "twelve": "model"})
    cols = list(Z.columns) if isinstance(Z, pd.DataFrame) else None
    p = PCA().fit(np.asarray(Z, float))
    for comp, rule in _SIGN_RULES[which]:
        if comp >= p.components_.shape[0]:
            continue
        row = p.components_[comp]
        if rule == "sum":
            ref = row.sum()
        else:
            name, pos = rule
            ref = row[cols.index(name)] if cols and name in cols else row[pos]
        if ref < 0:
            p.components_[comp] = -row
    return p


# ---------------------------------------------------------------------------
# The ten made-up clients A-J (bills in thousand NT$)
# ---------------------------------------------------------------------------
WORLD = pd.DataFrame(
    {"august": [1, 2, 2, 3, 5, 7, 8, 8, 9, 5],
     "september": [0, 1, 3, 2, 7, 8, 8, 10, 3, 8]},
    index=pd.Index(list("ABCDEFGHIJ"), name="client"))
_NAMES = np.array(list("ABCDEFGHIJ"))
_WX = WORLD.to_numpy(float)                 # private copies: editing lab.WORLD changes nothing
_WC = _WX - _WX.mean(axis=0)
TOTAL_VARIANCE = float((_WC ** 2).mean(axis=0).sum())   # 19.0 (divided by n, as the book does)
TOY_COLOUR = {c: BLUE if c in "ABCD" else (GRAY if c == "I" else RED) for c in _NAMES}
_TOY_PALETTE = {1: [DARK], 2: [BLUE, RED], 3: [BLUE, GRAY, RED]}


def _idx(letters) -> list:
    return [int(np.where(_NAMES == c)[0][0]) for c in letters]


def _groups_text(labels, names=_NAMES) -> str:
    """'ABCD | EFGHJ | I': the groups' letters, sorted."""
    labels = np.asarray(labels)
    return " | ".join(sorted("".join(names[labels == k]) for k in np.unique(labels)))


def _group_colours(labels, X=None) -> dict:
    """Colours of the ten clients' groups by position: the group with the
    smallest mean of the two bills blue, the largest red, the middle grey."""
    X = _WX if X is None else X
    labels = np.asarray(labels)
    ks = list(np.unique(labels))
    ks.sort(key=lambda k: X[labels == k].sum(axis=1).mean())
    pal = _TOY_PALETTE.get(len(ks), _CLUSTER_CYCLE)
    return {k: pal[i % len(pal)] for i, k in enumerate(ks)}


def _toy_frame(ax, centred: bool = False, lim=(-7, 7), labels: bool = True) -> None:
    if centred:
        lo, hi = lim
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.axhline(0, color=GRAY, lw=0.8, zorder=0)
        ax.axvline(0, color=GRAY, lw=0.8, zorder=0)
        ticks = [t for t in (-6, -3, 0, 3, 6) if lo <= t <= hi]
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        if labels:
            ax.set_xlabel("August, centred (thousand NT\\$)")
            ax.set_ylabel("September, centred (thousand NT\\$)")
    else:
        ax.set_xlim(-0.6, 10.6)
        ax.set_ylim(-0.6, 10.6)
        ax.set_xticks(range(0, 11, 2))
        ax.set_yticks(range(0, 11, 2))
        if labels:
            ax.set_xlabel("August bill (thousand NT\\$)")
            ax.set_ylabel("September bill (thousand NT\\$)")
    ax.set_aspect("equal")


def _toy_dots(ax, X=None, colours=None, letters: bool = True, s: float = 36,
              names=_NAMES, which=None, fontsize: float = 9, zorder: float = 3) -> None:
    """The ten clients as dots with their letters. colours: None (the world's
    colours), one colour, or {letter: colour}."""
    X = _WX if X is None else np.asarray(X, float)
    for i, c in enumerate(names):
        if which is not None and c not in which:
            continue
        col = (TOY_COLOUR[c] if colours is None else
               colours.get(c, INK) if isinstance(colours, dict) else colours)
        ax.scatter(X[i, 0], X[i, 1], s=s, color=col, zorder=zorder, edgecolor="white",
                   linewidth=0.6)
        if letters:
            ax.annotate(c, (X[i, 0], X[i, 1]), xytext=(4, 3), textcoords="offset points",
                        fontsize=fontsize, color=INK, zorder=zorder + 1, annotation_clip=False)


def _label_colours(labels) -> dict:
    gc = _group_colours(labels)
    return {c: gc[k] for c, k in zip(_NAMES, np.asarray(labels))}


def _centre_marks(ax, centres, open_: bool = False, s: float = 95, zorder: float = 5) -> None:
    centres = np.atleast_2d(centres)
    if open_:
        ax.scatter(centres[:, 0], centres[:, 1], marker="X", s=s * 1.5, facecolor="none",
                   edgecolor=INK, linewidth=1.1, zorder=zorder)
    else:
        ax.scatter(centres[:, 0], centres[:, 1], marker="X", s=s, color=INK,
                   edgecolor="white", linewidth=0.6, zorder=zorder)


# ----- the first principal component, by hand --------------------------------
def _unit(angle) -> np.ndarray:
    r = math.radians(float(angle))
    return np.array([math.cos(r), math.sin(r)])


def shadows(direction=(0.6, 0.8)) -> pd.Series:
    """The ten clients' shadows (projections) on a direction through the middle:
    w1 x centred August + w2 x centred September. A direction that is not of
    length 1 is first scaled to length 1 (with a note)."""
    u = np.asarray(direction, float)
    norm = float(np.hypot(*u))
    if norm == 0:
        _say("direction (0, 0) has no direction: showing (0.6, 0.8).")
        u, norm = np.array([0.6, 0.8]), 1.0
    if abs(norm - 1) > 1e-9:
        _say(f"direction ({u[0]:g}, {u[1]:g}) has length {norm:.3f}, not 1: using "
             f"({u[0] / norm:.3f}, {u[1] / norm:.3f}).")
        u = u / norm
    return pd.Series(_WC @ u, index=pd.Index(_NAMES, name="client"), name="shadow")


def shadow_variance(angle) -> float:
    """The variance of the ten shadows (mean of their squares) on the line at
    this angle (0 = the August axis, 90 = the September axis)."""
    return float(np.mean((_WC @ _unit(angle)) ** 2))


def _draw_projection(ax, u, *, drops: bool = True, ticks: bool = True, line_colour=DARK,
                     letters: bool = True, colours=None, label_shadows=(), line_lw=1.4,
                     s: float = 36) -> pd.Series:
    u = np.asarray(u, float)
    nvec = np.array([-u[1], u[0]])
    sh = _WC @ u
    ax.plot([-14 * u[0], 14 * u[0]], [-14 * u[1], 14 * u[1]], color=line_colour, lw=line_lw,
            zorder=1)
    feet = np.outer(sh, u)
    if drops:
        for p, f in zip(_WC, feet):
            ax.plot([p[0], f[0]], [p[1], f[1]], ls=":", color=MUTED, lw=0.9, zorder=1)
    if ticks:
        for f in feet:
            a, b = f - 0.3 * nvec, f + 0.3 * nvec
            ax.plot([a[0], b[0]], [a[1], b[1]], color=line_colour, lw=1.3, zorder=2)
    _toy_dots(ax, _WC, colours, letters=letters, s=s)
    sides = label_shadows if isinstance(label_shadows, dict) else {w: 1 for w in label_shadows}
    for who, side in sides.items():
        i = _idx(who)[0]
        f, p = feet[i], _WC[i]
        away = f - p
        away = away / np.hypot(*away) if np.hypot(*away) > 1e-9 else nvec
        away = away * side
        ax.annotate(f"{sh[i]:.1f}", f, xytext=(away[0] * 13, away[1] * 13),
                    textcoords="offset points", ha="center", va="center", fontsize=9,
                    color=line_colour, zorder=6,
                    bbox=dict(fc="white", ec="none", pad=0.15, alpha=0.9))
    return pd.Series(sh, index=_NAMES)


# ---------------------------------------------------------------------------
# E1 - Turn the direction
# ---------------------------------------------------------------------------
E1_ANGLES = (0, 180, 1)
_E1_TRAIL: list = []


def turn_direction_numbers(step: int = 15) -> pd.DataFrame:
    """The variance of the shadows every `step` degrees, and its share of 19.0."""
    angles = list(range(0, 180, step))
    v = [shadow_variance(a) for a in angles]
    return pd.DataFrame({"variance": v, "share of 19.0": np.array(v) / TOTAL_VARIANCE},
                        index=pd.Index(angles, name="angle"))


def _e1_extremes():
    angles = np.arange(0, 180)
    v = np.array([shadow_variance(a) for a in angles])
    return int(angles[v.argmax()]), float(v.max()), int(angles[v.argmin()]), float(v.min())


def draw_direction(ax, angle=0, labels: bool = True) -> dict:
    """The centred clients, the line at this angle, dotted drops and the
    shadows as ticks."""
    u = _unit(angle)
    _draw_projection(ax, u)
    _toy_frame(ax, centred=True, labels=labels)
    ax.set_title(f"direction ({_two(u[0])}, {_two(u[1])})")
    return {"angle": angle, "direction": u, "variance": shadow_variance(angle)}


def draw_angle_curve(ax, angle=0, trail=None, full: bool = False) -> dict:
    """The variance of the shadows at the angles tried (dots; the current one
    dark) against the total 19.0 (dashed). full=True also draws the curve at
    every angle and marks its highest and lowest point (the debrief)."""
    ax.axhline(TOTAL_VARIANCE, ls="--", color=GRAY, lw=1)
    ax.text(178, TOTAL_VARIANCE + 0.35, f"total {TOTAL_VARIANCE:.1f}", ha="right",
            va="bottom", fontsize=8.5, color=MUTED)
    out = {}
    if full:
        grid = np.arange(0, 180.01, 0.5)
        ax.plot(grid, [shadow_variance(a) for a in grid], color=MID_BLUE, lw=1.6, zorder=1)
        amax, vmax, amin, vmin = _e1_extremes()
        ax.plot([amax, amin], [vmax, vmin], "o", color=DARK, ms=6, zorder=4)
        ax.annotate(f"{amax}°: {vmax:.1f}", (amax, vmax), xytext=(0, 7),
                    textcoords="offset points", ha="center", va="bottom", fontsize=9, color=DARK)
        ax.annotate(f"{amin}°: {vmin:.1f}", (amin, vmin), xytext=(0, -7),
                    textcoords="offset points", ha="center", va="top", fontsize=9,
                    color=DARK)
        out.update(max_angle=amax, max_variance=vmax, min_angle=amin, min_variance=vmin)
    tried = sorted(set(trail or []) - {angle})
    if tried:
        ax.plot(tried, [shadow_variance(a) for a in tried], "o", color=MID_BLUE, ms=5,
                zorder=3, alpha=0.9, clip_on=False)
    v = shadow_variance(angle)
    if not full:
        ax.plot([angle], [v], "o", color=DARK, ms=8, zorder=4, clip_on=False)
        ax.annotate(f"{v:.1f}", (angle, v), xytext=(0, 8), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, color=DARK, zorder=5,
                    bbox=dict(fc="white", ec="none", pad=0.1, alpha=0.85))
    ax.set_xlim(0, 180)
    ax.set_xticks([0, 45, 90, 135, 180])
    ax.set_ylim(0, 21)
    ax.set_xlabel("angle (degrees)")
    ax.set_ylabel("variance of the shadows")
    ax.set_title("every angle" if full else "the angles you tried")
    out.update(angle=angle, variance=v)
    return out


def turn_direction_figure(angle=0, trail=None, figsize=None, suptitle: bool = True):
    with _style():
        fig, axes = _new_fig(figsize or (8.0, 3.7), 1, 2, width_ratios=[1, 1.1])
        draw_direction(axes[0], angle)
        draw_angle_curve(axes[1], angle, trail)
        if suptitle:
            v = shadow_variance(angle)
            fig.suptitle(f"angle {angle:g}°: variance {v:.1f} of {TOTAL_VARIANCE:.1f} "
                         f"({v / TOTAL_VARIANCE:.0%})", fontsize=9.5)
    return fig


def turn_direction(angle=0) -> None:
    """E1: turn the line through the middle of the ten centred clients; the
    variance of their shadows at every angle you tried stays on the right
    (turn_direction_reset() clears it)."""
    value = _decimal_comma(_plain(angle), "angle")
    a = _as_number(value)
    if not _is_number(a):
        _say(f"angle {value!r} is not a number of degrees: showing angle = 0.")
        a = 0
    if not 0 <= a <= 180:
        new = a % 180
        _say(f"angle {a:g} gives the same line as {new:g} (a line repeats every 180 "
             f"degrees): showing {new:g}.")
        a = new
    a = int(a) if float(a).is_integer() else round(float(a), 2)
    if a not in _E1_TRAIL:
        _E1_TRAIL.append(a)
    _show(turn_direction_figure(a, list(_E1_TRAIL)))


def turn_direction_reset() -> None:
    _E1_TRAIL.clear()


# ----- components, scores, PVE (the ten) ------------------------------------
def toy_pca() -> dict:
    """PCA of the ten clients (signs fixed): components, PVE, the variances
    (divided by n, as the book does) and each client's two scores."""
    p = pca_fixed(WORLD, "ten")
    S = p.transform(_WX)
    var_n = p.explained_variance_ * (len(_WX) - 1) / len(_WX)
    return {"components": p.components_.copy(), "pve": p.explained_variance_ratio_.copy(),
            "variance": var_n,
            "angle_pc1": math.degrees(math.atan2(p.components_[0, 1], p.components_[0, 0])),
            "scores": pd.DataFrame(S, index=pd.Index(_NAMES, name="client"),
                                   columns=["PC1", "PC2"])}


def draw_pc1_scores(ax) -> pd.Series:
    """Each client's PC1 score on one line."""
    sc = toy_pca()["scores"]["PC1"]
    ax.axhline(0, color=GRAY, lw=1, zorder=0)
    order = sc.sort_values().index
    for k, c in enumerate(order):
        x = sc[c]
        ax.scatter(x, 0, s=48, color=TOY_COLOUR[c], zorder=3, edgecolor="white", linewidth=0.6)
        up = k % 2 == 0
        ax.annotate(f"{c}\n{x:.1f}" if up else f"{x:.1f}\n{c}", (x, 0),
                    xytext=(0, 7 if up else -7), textcoords="offset points", ha="center",
                    va="bottom" if up else "top", fontsize=9, linespacing=1.0)
    ax.set_xlim(-7.5, 7.5)
    ax.set_ylim(-1, 1)
    ax.set_yticks([])
    for side in ("left", "top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_xticks(range(-6, 7, 2))
    ax.set_xlabel("PC1 score")
    return sc


def draw_pc_lines(ax) -> dict:
    """The centred clients with the PC1 line (dark) and the PC2 line (grey);
    I's shadow on PC2 marked."""
    t = toy_pca()
    c1, c2 = t["components"]
    _draw_projection(ax, c2, drops=False, ticks=False, line_colour=GRAY, letters=False)
    _draw_projection(ax, c1, drops=False, ticks=False, line_colour=DARK)
    i = _idx("I")[0]
    s2 = float(t["scores"].loc["I", "PC2"])
    foot = s2 * c2
    ax.plot([_WC[i, 0], foot[0]], [_WC[i, 1], foot[1]], ls=":", color=MUTED, lw=1)
    ax.plot([foot[0]], [foot[1]], "o", color=GRAY, ms=4, zorder=4)
    ax.annotate(f"{s2:.1f}", foot, xytext=(-10, 8), textcoords="offset points", ha="right",
                fontsize=9, color=MUTED,
                bbox=dict(fc="white", ec="none", pad=0.1, alpha=0.9))
    ax.annotate("PC1", 8 * c1, xytext=(5, 0), textcoords="offset points", color=DARK,
                fontsize=9, ha="left", va="center")
    ax.text(6.3 * c2[0], 6.3 * c2[1] - 0.2, "PC2", color=MUTED, fontsize=9, ha="left",
            va="top")
    _toy_frame(ax, centred=True)
    return {"pc1": c1, "pc2": c2, "I_pc2": s2}


def draw_pc_coords(ax) -> pd.DataFrame:
    """Each client at (PC1 score, PC2 score), equal scales."""
    S = toy_pca()["scores"]
    _toy_dots(ax, S.to_numpy())
    ax.axhline(0, color=GRAY, lw=0.8, zorder=0)
    ax.axvline(0, color=GRAY, lw=0.8, zorder=0)
    ax.set_xlim(-7.5, 7.5)
    ax.set_ylim(-3.5, 5.5)
    ax.set_aspect("equal")
    ax.set_xticks(range(-6, 7, 3))
    ax.set_yticks([-3, 0, 3])
    ax.set_xlabel("PC1 score")
    ax.set_ylabel("PC2 score")
    return S


def draw_pve(ax) -> np.ndarray:
    """Bars of the PVE of PC1 and PC2; dotted = the running total."""
    pve = 100 * toy_pca()["pve"]
    ax.bar([1, 2], pve, width=0.6, color=[BLUE, MID_BLUE])
    for x, v in zip([1, 2], pve):
        ax.text(x, v / 2 if v > 30 else v + 2, f"{v:.0f}%", ha="center", va="center"
                if v > 30 else "bottom", color="white" if v > 30 else INK, fontsize=9.5,
                fontweight="bold")
    run = np.cumsum(pve)
    ax.plot([1, 2], run, ls=":", color=INK, marker="o", ms=4, lw=1.2)
    ax.annotate(f"{run[-1]:.0f}%", (2, run[-1]), xytext=(0, 5), textcoords="offset points",
                ha="center", va="bottom", fontsize=9)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["PC1", "PC2"])
    ax.set_xlim(0.4, 2.6)
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("share of the variance (%)")
    return pve


# ----- slide pictures of the world -----------------------------------------
def draw_world(ax, colours: bool = True) -> None:
    """The ten clients: August bill across, September bill up."""
    _toy_dots(ax, colours=None if colours else "#555555")
    _toy_frame(ax)


def draw_world_table(ax, centred: bool = False) -> pd.DataFrame:
    """The ten clients as a small table (client, August, September), or the
    centred values ("August - 5", "September - 5")."""
    T = pd.DataFrame(_WC if centred else _WX, index=_NAMES, columns=["a", "s"]).astype(int)
    head = ["client", "August - 5", "September - 5"] if centred else \
        ["client", "August", "September"]
    ax.axis("off")
    xs = [0.12, 0.5, 0.86]
    rows = len(T) + 1
    for j, h in enumerate(head):
        ax.text(xs[j], 1 - 0.5 / rows, h, ha="center", va="center", fontsize=9,
                fontweight="bold", transform=ax.transAxes)
    ax.plot([0.0, 1.0], [1 - 1 / rows] * 2, color=GRAY, lw=0.8, transform=ax.transAxes)
    for k, c in enumerate(T.index):
        y = 1 - (k + 1.5) / rows
        ax.text(xs[0], y, c, ha="center", va="center", fontsize=9, color=TOY_COLOUR[c],
                fontweight="bold", transform=ax.transAxes)
        for j, col in enumerate(["a", "s"]):
            ax.text(xs[j + 1], y, f"{T.loc[c, col]:d}", ha="center", va="center", fontsize=9,
                    transform=ax.transAxes)
    return T


def draw_centred(ax) -> None:
    """The ten clients after centring: the axes cross at the average client."""
    _toy_dots(ax, _WC)
    _toy_frame(ax, centred=True)


def draw_shadows(ax, direction=(0.6, 0.8), label=None) -> pd.Series:
    """The centred clients, one direction (an arrow of length 1), dotted drops
    to the line and the shadows as ticks; the shadows of `label` written
    ({letter: 1 = beyond the line, -1 = on the client's side}; default A and I)."""
    label = {"A": 1, "I": -1} if label is None else label
    u = np.asarray(direction, float)
    u = u / np.hypot(*u)
    sh = _draw_projection(ax, u, label_shadows=label)
    ax.annotate("", xy=(u[0], u[1]), xytext=(0, 0), zorder=6,
                arrowprops=dict(arrowstyle="-|>", color=DARK, lw=1.6, mutation_scale=10))
    ax.annotate(f"({u[0]:g}, {u[1]:g})", (u[0], u[1]), xytext=(-14, 2),
                textcoords="offset points", ha="right", va="bottom", fontsize=9, color=DARK,
                bbox=dict(fc="white", ec="none", pad=0.1, alpha=0.9), zorder=7)
    _toy_frame(ax, centred=True)
    return sh


def draw_three_directions(axes) -> list:
    """Three panels: the shadows on (1, 0), (0, 1) and (0.6, 0.8); the
    variance of each set of shadows under the panel."""
    out = []
    for ax, (u, title) in zip(axes, [((1, 0), "August only (1, 0)"),
                                      ((0, 1), "September only (0, 1)"),
                                      ((0.6, 0.8), "(0.6, 0.8)")]):
        u = np.asarray(u, float)
        sh = _draw_projection(ax, u, drops=False, letters=False, s=26)
        v = float(np.mean(sh ** 2))
        _toy_frame(ax, centred=True, labels=False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title)
        ax.set_xlabel(f"variance {v:.1f}")
        out.append(v)
    return out


def draw_distance(ax, legs: bool = True) -> dict:
    """C's distance to D and to E as two right triangles."""
    _toy_dots(ax, colours="#777777", s=30)
    C, D, E = (_WX[i] for i in _idx("CDE"))
    out = {}
    for P, colour in ((D, DARK), (E, BLUE)):
        corner = np.array([P[0], C[1]])
        ax.plot([C[0], corner[0], P[0]], [C[1], corner[1], P[1]], color=colour, lw=1, ls="--")
        ax.plot([C[0], P[0]], [C[1], P[1]], color=colour, lw=1.8)
        d = float(np.hypot(*(P - C)))
        mid = (C + P) / 2
        txt = f"{d:.2f}" if abs(d - round(d)) > 1e-9 else f"{d:.0f}"
        ax.annotate(txt, mid, xytext=(-8, 6) if P is E else (-6, -6), textcoords="offset points",
                    ha="right", va="bottom" if P is E else "top",
                    fontsize=9, color=colour, fontweight="bold",
                    bbox=dict(fc="white", ec="none", pad=0.1, alpha=0.9))
        if legs and P is E:
            ax.text((C[0] + corner[0]) / 2 + 0.3, C[1] + 0.15, f"{abs(P[0] - C[0]):.0f}",
                    ha="center", va="bottom", fontsize=9, color=colour)
            ax.text(corner[0] + 0.2, (C[1] + P[1]) / 2, f"{abs(P[1] - C[1]):.0f}",
                    ha="left", va="center", fontsize=9, color=colour)
        out["C-" + ("D" if P is D else "E")] = d
    _toy_frame(ax)
    return out


# ----- k-means on the ten -----------------------------------------------------
def toy_kmeans_run(start=("A", "B")) -> list:
    """K-means by hand on the ten clients from starting centres on the given
    clients: a list of steps {"step": "assign" | "move", "labels", "centres",
    "inertia", "groups"}, until an assign changes nothing."""
    from scipy.spatial.distance import cdist
    cent = _WX[_idx(start)].astype(float)
    prev, steps = None, []
    while True:
        d2 = cdist(_WX, cent, "sqeuclidean")
        lab = d2.argmin(axis=1)
        steps.append({"step": "assign", "labels": lab, "centres": cent.copy(),
                      "inertia": float(d2.min(axis=1).sum()), "groups": _groups_text(lab),
                      "sq_distances": d2})
        if prev is not None and (lab == prev).all():
            return steps
        cent = np.array([_WX[lab == k].mean(axis=0) for k in range(len(cent))])
        steps.append({"step": "move", "labels": lab, "centres": cent.copy(),
                      "inertia": float(((_WX - cent[lab]) ** 2).sum()),
                      "groups": _groups_text(lab)})
        prev = lab


def _inertia_text(v) -> str:
    return f"{v:.0f}" if v >= 90 else f"{v:.1f}"


def draw_kmeans_goal(ax) -> dict:
    """ABCD and EFGHIJ, their centres (crosses) and each client's line to its
    own centre."""
    lab = np.array([0] * 4 + [1] * 6)
    cent = np.array([_WX[lab == k].mean(axis=0) for k in (0, 1)])
    col = _label_colours(lab)
    for i, c in enumerate(_NAMES):
        ax.plot([_WX[i, 0], cent[lab[i], 0]], [_WX[i, 1], cent[lab[i], 1]], color=col[c],
                lw=0.8, alpha=0.7, zorder=1)
    _toy_dots(ax, colours=col)
    _centre_marks(ax, cent)
    _toy_frame(ax)
    parts = [float(((_WX[lab == k] - cent[k]) ** 2).sum()) for k in (0, 1)]
    return {"centres": cent, "inertia_parts": parts, "inertia": sum(parts)}


def draw_kmeans_step(ax, step: str = "1") -> dict:
    """The run from A and B: "1" start and assign (crosses on A and B), "2" move
    centre 2 (arrow), "3a" assign again (the clients that switch ringed), "3b"
    move both centres (arrows)."""
    steps = toy_kmeans_run(("A", "B"))
    a0, m0, a1, m1 = steps[0], steps[1], steps[2], steps[3]
    step = str(step)
    arrow = dict(arrowstyle="-|>", color=INK, lw=1.2, mutation_scale=10, shrinkA=4, shrinkB=6)
    if step == "1":
        col = _label_colours(a0["labels"])
        for i, c in enumerate(_NAMES):
            k = a0["labels"][i]
            ax.plot([_WX[i, 0], a0["centres"][k, 0]], [_WX[i, 1], a0["centres"][k, 1]],
                    color=col[c], lw=0.8, alpha=0.7, zorder=1)
        _toy_dots(ax, colours=col)
        _centre_marks(ax, a0["centres"])
        out = {"inertia": a0["inertia"], "groups": a0["groups"]}
    elif step == "2":
        col = _label_colours(a0["labels"])
        _toy_dots(ax, colours=col)
        ax.annotate("", xy=m0["centres"][1], xytext=a0["centres"][1], arrowprops=arrow,
                    zorder=6)
        _centre_marks(ax, m0["centres"])
        out = {"inertia": m0["inertia"], "centres": m0["centres"]}
    elif step == "3a":
        col = _label_colours(a1["labels"])
        _toy_dots(ax, colours=col)
        switch = [i for i in range(10) if a1["labels"][i] != a0["labels"][i]]
        ax.scatter(_WX[switch, 0], _WX[switch, 1], s=230, facecolor="none", edgecolor=INK,
                   linewidth=1.1, zorder=4)
        _centre_marks(ax, a1["centres"])
        out = {"inertia": a1["inertia"], "groups": a1["groups"],
               "switch": "".join(_NAMES[switch]),
               "D_sq_distances": a1["sq_distances"][_idx("D")[0]]}
    elif step == "3b":
        col = _label_colours(a1["labels"])
        _toy_dots(ax, colours=col)
        for k in (0, 1):
            ax.annotate("", xy=m1["centres"][k], xytext=a1["centres"][k], arrowprops=arrow,
                        zorder=6)
        _centre_marks(ax, a1["centres"], open_=True, s=60)
        _centre_marks(ax, m1["centres"])
        out = {"inertia": m1["inertia"], "centres": m1["centres"]}
    else:
        raise ValueError('step must be "1", "2", "3a" or "3b"')
    _toy_frame(ax)
    return out


def draw_inertia_steps(ax) -> list:
    """The inertia after each step of the run from A and B."""
    steps = toy_kmeans_run(("A", "B"))
    vals = [s["inertia"] for s in steps]
    names = ["1 assign", "2 move", "3 assign", "3 move", "4 assign"][:len(vals)]
    xs = np.arange(len(vals))
    ax.bar(xs, vals, width=0.6, color=[MID_BLUE if s["step"] == "assign" else BLUE
                                       for s in steps])
    for x, v in zip(xs, vals):
        ax.text(x, v + 8, _inertia_text(v), ha="center", va="bottom", fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels([n.replace(" ", "\n") for n in names], linespacing=1.0)
    ax.set_ylim(0, 490)
    ax.set_ylabel("inertia")
    return vals


def draw_two_starts(axes) -> list:
    """K = 3 from A, C, E (left) and from A, E, I (right): open crosses = the
    starting clients, filled = the final centres."""
    out = []
    for ax, start in zip(axes, [("A", "C", "E"), ("A", "E", "I")]):
        last = toy_kmeans_run(start)[-1]
        _toy_dots(ax, colours=_label_colours(last["labels"]))
        _centre_marks(ax, _WX[_idx(start)], open_=True)
        _centre_marks(ax, last["centres"])
        _toy_frame(ax)
        ax.set_title(f"start {', '.join(start)}\ninertia {last['inertia']:.1f}",
                     linespacing=1.1)
        out.append({"start": start, "groups": last["groups"], "inertia": last["inertia"]})
    return out


def toy_elbow(ks=range(1, 7)) -> pd.Series:
    """The smallest inertia k-means finds on the ten clients for each K
    (n_init=50)."""
    from sklearn.cluster import KMeans
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        v = [KMeans(k, n_init=50, random_state=SEED).fit(_WX).inertia_ for k in ks]
    return pd.Series(v, index=pd.Index(list(ks), name="K"), name="inertia")


def draw_toy_elbow(ax) -> pd.Series:
    el = toy_elbow()
    ax.plot(el.index, el.values, "-o", color=DARK, ms=5, lw=1.4)
    for k, v in el.items():
        ax.annotate(_inertia_text(v) if v >= 90 else f"{v:.1f}", (k, v), xytext=(5, 4),
                    textcoords="offset points", fontsize=9, ha="left", va="bottom")
    ax.text(6.4, 30, "K = 10: 0", ha="right", fontsize=9, color=MUTED)
    ax.set_xticks(list(el.index))
    ax.set_xlim(0.6, 6.6)
    ax.set_ylim(0, 215)
    ax.set_xlabel("K, the number of groups")
    ax.set_ylabel("smallest inertia")
    return el


def toy_units_groups(units: str = "thousand NT$") -> np.ndarray:
    """K-means (K = 3) on the ten clients with August in thousand NT$, in NT$, or
    both standardised; group labels 0-2."""
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    units = _choice(units, ["thousand NT$", "NT$", "standardised"], "units",
                    {"augustinnt": "NT$", "raw": "thousand NT$", "standardized": "standardised"})
    X = _WX.copy()
    if units == "NT$":
        X[:, 0] *= 1000
    elif units == "standardised":
        X = StandardScaler().fit_transform(X)
    return KMeans(3, n_init=10, random_state=SEED).fit(X).labels_


def draw_units_toy(axes) -> dict:
    """K-means with K = 3: both bills in thousand NT$ (left) and August in NT$
    (right)."""
    out = {}
    C, B, D = (_WX[i] for i in _idx("CBD"))
    for ax, units in zip(axes, ["thousand NT$", "NT$"]):
        lab = toy_units_groups(units)
        scale = 1000 if units == "NT$" else 1
        X = _WX.copy()
        X[:, 0] *= scale
        _toy_dots(ax, X, colours=_label_colours(lab))
        pairs = [(D, "1.41")] if scale == 1 else [(B, "2"), (D, "1,000")]
        for P, txt in pairs:
            ax.plot([C[0] * scale, P[0] * scale], [C[1], P[1]], color=INK, lw=1.2, zorder=2)
            above = txt == "1,000"
            ax.annotate(txt, ((C[0] + P[0]) / 2 * scale, (C[1] + P[1]) / 2),
                        xytext=(6, 5) if above else (-6, 0), textcoords="offset points",
                        ha="left" if above else "right", va="bottom" if above else "center",
                        fontsize=9,
                        bbox=dict(fc="white", ec="none", pad=0.1, alpha=0.9))
        ax.set_xlim(-0.6 * scale, 10.6 * scale)
        ax.set_ylim(-0.6, 10.6)
        ax.set_yticks(range(0, 11, 2))
        if scale == 1:
            ax.set_aspect("equal")
            ax.set_xticks(range(0, 11, 2))
            ax.set_xlabel("August bill (thousand NT\\$)")
            ax.set_title("both in thousand NT\\$")
        else:
            ax.set_box_aspect(1)
            ax.set_xticks([0, 5000, 10000])
            ax.set_xticklabels(["0", "5,000", "10,000"])
            ax.set_xlabel("August bill (NT\\$)")
            ax.set_title("August in NT\\$")
        ax.set_ylabel("September bill (thousand NT\\$)")
        out[units] = _groups_text(lab)
    return out


# ----- hierarchical clustering on the ten -----------------------------------
def _toy_tree(method: str = "complete") -> dict:
    """The merges of the ten clients; children ordered so that the child with
    the alphabetically first client is on the left (leaves A B C D E J F G H I
    for complete linkage)."""
    from scipy.cluster.hierarchy import linkage
    Z = linkage(_WX, method)
    n = len(_WX)
    members = {i: [i] for i in range(n)}
    height = {i: 0.0 for i in range(n)}
    children = {}
    for j, (a, b, h, _) in enumerate(Z):
        a, b = int(a), int(b)
        if min(members[a]) > min(members[b]):
            a, b = b, a
        children[n + j] = (a, b)
        members[n + j] = members[a] + members[b]
        height[n + j] = float(h)
    order = members[2 * n - 2]
    pos = {leaf: k for k, leaf in enumerate(order)}

    def xpos(node):
        if node < n:
            return float(pos[node])
        a, b = children[node]
        return (xpos(a) + xpos(b)) / 2
    return {"Z": Z, "children": children, "members": members, "height": height,
            "order": order, "x": xpos, "n": n}


def toy_linkage(method: str = "complete") -> pd.DataFrame:
    """The merges in order: the two groups joined, the new group and its
    height (the distance at the merge)."""
    t = _toy_tree(method)
    rows = []
    for node in sorted(t["children"]):
        a, b = t["children"][node]
        rows.append({"joins": "".join(sorted(_NAMES[t["members"][a]])) + " + "
                     + "".join(sorted(_NAMES[t["members"][b]])),
                     "group": "".join(sorted(_NAMES[t["members"][node]])),
                     "height": t["height"][node]})
    return pd.DataFrame(rows, index=pd.Index(range(1, len(rows) + 1), name="merge"))


_MERGE_LABEL = {"EJ": (-7, 0, "right", "center"), "FG": (0, -6, "center", "top"),
                "AB": (8, -2, "left", "top"), "CD": (-7, -2, "right", "top")}


def draw_merges(ax, k: int = 4) -> pd.DataFrame:
    """The first k merges (complete linkage) as links between the clients;
    label = order: distance."""
    t = toy_linkage("complete").head(k)
    _toy_dots(ax)
    for m, row in t.iterrows():
        a, b = row["joins"].split(" + ")
        if len(a) == 1 and len(b) == 1:
            P, Q = _WX[_idx(a)[0]], _WX[_idx(b)[0]]
            ax.plot([P[0], Q[0]], [P[1], Q[1]], color=DARK, lw=2, zorder=2)
            mid = (P + Q) / 2
            dx, dy, ha, va = _MERGE_LABEL.get(a + b, (-7, -2, "right", "top"))
            ax.annotate(f"{m}: {_dist_text(row['height'])}", mid, xytext=(dx, dy),
                        textcoords="offset points", ha=ha, va=va, fontsize=9,
                        color=DARK, bbox=dict(fc="white", ec="none", pad=0.1, alpha=0.9))
    _toy_frame(ax)
    return t


def _pad_hull(ax, pts, colour, width=22, alpha=0.18) -> None:
    """A shaded, padded outline around points (a thick round-joined line).
    One opaque light colour (the colour mixed with white), so the fill and
    the thick edge do not add up to a darker inner region."""
    from matplotlib.colors import to_rgb
    from matplotlib.patches import Polygon
    pts = np.asarray(pts, float)
    if len(pts) >= 3:
        from scipy.spatial import ConvexHull
        pts = pts[ConvexHull(pts).vertices]
    light = tuple(1 - alpha * (1 - c) for c in to_rgb(colour))
    poly = Polygon(pts, closed=True, facecolor=light, edgecolor=light,
                   linewidth=width, joinstyle="round", capstyle="round", zorder=0)
    ax.add_patch(poly)


def draw_linkage(ax) -> dict:
    """{A, B} and {C, D} and the distances of their four pairs."""
    A, B, C, D = (_WX[i] for i in _idx("ABCD"))
    _pad_hull(ax, [A, B], BLUE)
    _pad_hull(ax, [C, D], MID_BLUE)
    # A, B and D lie on one straight line, so A-D is drawn as an arc below B;
    # each label names its pair: (label position, ha, va)
    from matplotlib.patches import FancyArrowPatch
    pairs = {"A-C": (A, C, (1.33, 1.55), "right", "center"),
             "A-D": (A, D, (2.42, 0.48), "left", "top"),
             "B-C": (B, C, (2.08, 2.45), "left", "center"),
             "B-D": (B, D, (3.1, 1.5), "left", "center")}
    out = {}
    for name, (P, Q, xy, ha, va) in pairs.items():
        d = float(np.hypot(*(P - Q)))
        if name == "A-D":
            ax.add_patch(FancyArrowPatch(tuple(P), tuple(Q), arrowstyle="-", color=INK, lw=1,
                                         ls="--", connectionstyle="arc3,rad=0.35",
                                         shrinkA=0, shrinkB=0, zorder=2))
        else:
            ax.plot([P[0], Q[0]], [P[1], Q[1]], color=INK, lw=1,
                    ls="--" if name != "B-D" else "-", zorder=2)
        ax.text(xy[0], xy[1], f"{name} {_dist_text(d)}", ha=ha, va=va, fontsize=9, zorder=5)
        out[name] = d
    _toy_dots(ax, which="ABCD", s=40)
    ax.set_xlim(-0.2, 4.6)
    ax.set_ylim(-0.8, 4.0)
    ax.set_aspect("equal")
    ax.set_xticks(range(0, 5))
    ax.set_yticks(range(0, 5))
    ax.set_xlabel("August bill (thousand NT\\$)")
    ax.set_ylabel("September bill (thousand NT\\$)")
    v = np.array(list(out.values()))
    out.update(single=float(v.min()), complete=float(v.max()), average=float(v.mean()))
    return out


def draw_dendrogram_toy(ax, stage: str = "b", method: str = "complete",
                        heights: bool = True) -> pd.DataFrame:
    """The dendrogram of the ten clients: stage "a" the first four joins only,
    "b" the full tree with dashed cuts at 5 and 10. The y axis is the same in
    both stages, so the two overlays line up."""
    t = _toy_tree(method)
    x, h, ch, n = t["x"], t["height"], t["children"], t["n"]
    limit = sorted(h[k] for k in ch)[3] + 1e-9 if stage == "a" else np.inf
    for node in sorted(ch):
        if h[node] > limit:
            continue
        a, b = ch[node]
        xa, xb, H = x(a), x(b), h[node]
        ax.plot([xa, xa, xb, xb], [h[a], H, H, h[b]], color=INK, lw=1.4)
        if heights:   # right of the join's middle, where the line to the parent rises
            ax.text((xa + xb) / 2 + 0.1, H + 0.15, _dist_text(H), ha="left", va="bottom",
                    fontsize=8.5, color=MUTED)
    if stage != "a":
        for cut, words in ((5, "3 clusters"), (10, "2 clusters")):
            ax.plot([-0.6, n - 0.4], [cut, cut], ls="--", color=RED, lw=1)
            ax.text(n - 0.25, cut, words, ha="left", va="center", fontsize=9, color=RED)
    ax.set_xticks(range(n))
    ax.set_xticklabels([_NAMES[i] for i in t["order"]])
    for lbl, i in zip(ax.get_xticklabels(), t["order"]):
        lbl.set_color(TOY_COLOUR[_NAMES[i]])
        lbl.set_fontweight("bold")
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.6, n + 1.9)   # room on the right for the cut labels (both stages)
    ax.spines["bottom"].set_bounds(-0.6, n - 0.4)
    ax.set_ylim(0, 13.5)
    ax.set_ylabel("height = distance at the merge")
    return toy_linkage(method)


# ----- DBSCAN on the ten ------------------------------------------------------
def toy_dbscan(eps: float = 2.0, min_samples: int = 3) -> pd.DataFrame:
    """Per client: how many clients are inside its circle (at most eps away,
    itself included), core / border / noise, and its cluster (-1 = noise).
    The same as scikit-learn's DBSCAN."""
    from scipy.spatial.distance import pdist, squareform
    from sklearn.cluster import DBSCAN
    D = squareform(pdist(_WX))
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(_WX)
    core = np.zeros(len(_WX), bool)
    core[db.core_sample_indices_] = True
    kind = np.where(core, "core", np.where(db.labels_ < 0, "noise", "border"))
    return pd.DataFrame({"inside": (D <= eps + 1e-9).sum(axis=1), "kind": kind,
                         "cluster": db.labels_}, index=pd.Index(_NAMES, name="client"))


def _toy_cluster_colours(t: pd.DataFrame) -> dict:
    col = {}
    for c, k in t["cluster"].items():
        col[c] = INK if k < 0 else [BLUE, RED, DARK, MID_BLUE][int(k) % 4]
    return col


def _toy_eps_frame(ax) -> None:
    ax.set_xlim(-1.5, 11.5)
    ax.set_ylim(-1.8, 12.0)
    ax.set_aspect("equal")
    ax.set_xticks(range(0, 11, 2))
    ax.set_yticks(range(0, 11, 2))
    ax.set_xlabel("August bill (thousand NT\\$)")
    ax.set_ylabel("September bill (thousand NT\\$)")


_EPS_LABEL = {"J": (-5, 3, "right", "bottom"), "E": (-5, -3, "right", "top"),
              "F": (0, 6, "center", "bottom"), "G": (5, -3, "left", "top"),
              "H": (5, 0, "left", "center")}


def draw_eps_circles(ax, eps: float = 2.0) -> pd.DataFrame:
    """A circle of radius eps around each client; next to each dot its letter
    and the count of clients inside (itself included); thin lines = the pairs
    exactly eps apart."""
    from matplotlib.patches import Circle
    from scipy.spatial.distance import pdist, squareform
    t = toy_dbscan(eps, 3)
    for P in _WX:
        ax.add_patch(Circle(P, eps, facecolor=LIGHT, edgecolor=MID_BLUE, lw=0.6, alpha=0.35,
                            zorder=0))
    D = squareform(pdist(_WX))
    for i in range(10):
        for j in range(i + 1, 10):
            if abs(D[i, j] - eps) < 1e-9:
                ax.plot(*zip(_WX[i], _WX[j]), color=INK, lw=0.9, zorder=2)
    for i, c in enumerate(_NAMES):
        ax.scatter(*_WX[i], s=30, color=TOY_COLOUR[c], zorder=3, edgecolor="white",
                   linewidth=0.6)
        dx, dy, ha, va = _EPS_LABEL.get(c, (4, 3, "left", "bottom"))
        ax.annotate(f"{c}: {t.loc[c, 'inside']}", _WX[i], xytext=(dx, dy), ha=ha, va=va,
                    textcoords="offset points", fontsize=9, zorder=4,
                    bbox=dict(fc="white", ec="none", pad=0.1, alpha=0.7))
    _toy_eps_frame(ax)
    return t


def draw_core_border_noise(ax, eps: float = 2.0, min_samples: int = 3) -> pd.DataFrame:
    """Core clients = big dots (with their circles), border = small dots,
    noise = black x."""
    from matplotlib.patches import Circle
    t = toy_dbscan(eps, min_samples)
    col = _toy_cluster_colours(t)
    for i, c in enumerate(_NAMES):
        kind = t.loc[c, "kind"]
        if kind == "core":
            ax.add_patch(Circle(_WX[i], eps, facecolor=LIGHT, edgecolor=MID_BLUE, lw=0.6,
                                alpha=0.35, zorder=0))
            ax.scatter(*_WX[i], s=80, color=col[c], zorder=3, edgecolor="white", linewidth=0.6)
        elif kind == "border":
            ax.scatter(*_WX[i], s=22, color=col[c], zorder=3)
        else:
            ax.scatter(*_WX[i], s=55, marker="x", color=INK, linewidth=1.6, zorder=3)
        ax.annotate(c, _WX[i], xytext=(5, 3), textcoords="offset points", fontsize=9, zorder=4)
    _toy_eps_frame(ax)
    return t


def draw_dbscan_grow(ax, eps: float = 2.0, min_samples: int = 3) -> list:
    """How the clusters grow: an arrow from a core client to each client in its
    circle that joins the cluster through it; shaded = the clusters."""
    from scipy.spatial.distance import pdist, squareform
    t = toy_dbscan(eps, min_samples)
    D = squareform(pdist(_WX))
    core = (t["kind"] == "core").to_numpy()
    seen, arrows = set(), []
    for start in range(10):
        if not core[start] or start in seen:
            continue
        seen.add(start)
        queue = [start]
        while queue:
            i = queue.pop(0)
            for j in range(10):
                if j not in seen and D[i, j] <= eps + 1e-9:
                    seen.add(j)
                    arrows.append((str(_NAMES[i]), str(_NAMES[j])))
                    if core[j]:
                        queue.append(j)
    col = _toy_cluster_colours(t)
    for k in sorted(set(t["cluster"]) - {-1}):
        members = [i for i, c in enumerate(_NAMES) if t.loc[c, "cluster"] == k]
        _pad_hull(ax, _WX[members], [BLUE, RED, DARK, MID_BLUE][k % 4], width=20, alpha=0.15)
    for a, b in arrows:
        ax.annotate("", xy=_WX[_idx(b)[0]], xytext=_WX[_idx(a)[0]], zorder=2,
                    arrowprops=dict(arrowstyle="-|>", color=INK, lw=1, mutation_scale=9,
                                    shrinkA=5, shrinkB=5))
    for i, c in enumerate(_NAMES):
        if t.loc[c, "kind"] == "noise":
            ax.scatter(*_WX[i], s=55, marker="x", color=INK, linewidth=1.6, zorder=3)
        else:
            ax.scatter(*_WX[i], s=80 if core[i] else 22, color=col[c], zorder=3,
                       edgecolor="white", linewidth=0.6)
        ax.annotate(c, _WX[i], xytext=(5, 3), textcoords="offset points", fontsize=9, zorder=4)
    _toy_frame(ax)
    return arrows


# ----- the silhouette on the ten ---------------------------------------------
def toy_silhouette(k: int = 2) -> pd.DataFrame:
    """k-means (n_init=10, seed 42) on the ten clients; per client its group,
    a (mean distance to its own group), b (mean distance to the nearest other
    group) and s = (b - a) / max(a, b). The mean of s is the silhouette."""
    from scipy.spatial.distance import pdist, squareform
    from sklearn.cluster import KMeans
    lab = KMeans(k, n_init=10, random_state=SEED).fit(_WX).labels_
    D = squareform(pdist(_WX))
    rows = []
    for i in range(10):
        own = [j for j in range(10) if lab[j] == lab[i] and j != i]
        a = D[i, own].mean() if own else 0.0
        b = min(D[i, lab == g].mean() for g in set(lab) if g != lab[i])
        rows.append({"group": "".join(_NAMES[lab == lab[i]]), "a": a, "b": b,
                     "s": 0.0 if not own else (b - a) / max(a, b)})
    return pd.DataFrame(rows, index=pd.Index(_NAMES, name="client"))


def draw_silhouette_client(ax, who: str = "I", k: int = 2) -> dict:
    """One client's lines: solid to the others in its own group, dashed to the
    clients of the nearest other group."""
    from sklearn.cluster import KMeans
    lab = KMeans(k, n_init=10, random_state=SEED).fit(_WX).labels_
    i = _idx(who)[0]
    col = _label_colours(lab)
    t = toy_silhouette(k)
    other = min((g for g in set(lab) if g != lab[i]),
                key=lambda g: np.hypot(*(_WX[lab == g] - _WX[i]).T).mean())
    for j in range(10):
        if j == i:
            continue
        if lab[j] == lab[i]:
            ax.plot(*zip(_WX[i], _WX[j]), color=col[_NAMES[i]], lw=1.2, zorder=1)
        elif lab[j] == other:
            ax.plot(*zip(_WX[i], _WX[j]), color=col[_NAMES[j]], lw=1.1, ls="--", zorder=1)
    _toy_dots(ax, colours=col)
    ax.scatter(*_WX[i], s=200, facecolor="none", edgecolor=INK, linewidth=1.1, zorder=4)
    _toy_frame(ax)
    return t.loc[who].to_dict()


def draw_silhouette_bars(ax, k: int = 2) -> pd.Series:
    """The silhouette s of each client; dashed = their mean."""
    from sklearn.cluster import KMeans
    lab = KMeans(k, n_init=10, random_state=SEED).fit(_WX).labels_
    col = _label_colours(lab)
    s = toy_silhouette(k)["s"]
    xs = np.arange(10)
    ax.bar(xs, s.values, width=0.65, color=[col[c] for c in _NAMES])
    for x, v in zip(xs, s.values):
        ax.text(x, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=8.5)
    m = float(s.mean())
    ax.axhline(m, ls="--", color=INK, lw=1)
    ax.text(0.99, 0.98, f"dashed: mean {m:.2f}", ha="right", va="top", fontsize=9,
            transform=ax.transAxes)
    ax.set_xticks(xs)
    ax.set_xticklabels(_NAMES)
    ax.set_xlim(-0.6, 9.6)
    ax.set_ylim(0, 1)
    ax.set_ylabel("silhouette s")
    return s


# ---------------------------------------------------------------------------
# E2 - Start k-means anywhere
# ---------------------------------------------------------------------------
E2_STARTS = list(range(20))
_E2_TRAIL: list = []


def start_clients(start: int) -> list:
    """The three starting clients of start `start` (the lab's own draw:
    numpy's RandomState stream does not change between versions)."""
    idx = np.random.RandomState(int(start)).choice(10, 3, replace=False, p=np.full(10, 0.1))
    return [str(c) for c in _NAMES[idx]]


def _e2_run(start: int, n_init: int = 1) -> dict:
    from sklearn.cluster import KMeans
    if n_init == 1:
        letters = start_clients(start)
        km = KMeans(3, init=_WX[_idx(letters)], n_init=1).fit(_WX)
    else:
        letters = None
        km = KMeans(3, init="random", n_init=n_init, random_state=int(start)).fit(_WX)
    return {"start": start, "clients": letters, "labels": km.labels_,
            "centres": km.cluster_centers_, "inertia": float(km.inertia_),
            "groups": _groups_text(km.labels_)}


def start_kmeans_numbers() -> pd.DataFrame:
    """The 20 starts: the three starting clients, the groups where the loop
    stopped and their inertia."""
    def build():
        rows = []
        for s in E2_STARTS:
            r = _e2_run(s)
            rows.append({"starting clients": ", ".join(r["clients"]), "groups": r["groups"],
                         "inertia": round(r["inertia"], 1)})
        return pd.DataFrame(rows, index=pd.Index(E2_STARTS, name="start"))
    return _cached("e2", "the 20 starts", build).copy()


def groupings_found() -> pd.DataFrame:
    """The different groupings the 20 starts end in, smallest inertia first."""
    t = start_kmeans_numbers()
    g = t.reset_index().groupby("groups").agg(inertia=("inertia", "first"),
                                              starts=("start", list))
    g["how many starts"] = g["starts"].map(len)
    return g.sort_values("inertia")


def draw_grouping(ax, labels, title=None, letters: bool = True, s: float = 30,
                  frame: bool = True) -> None:
    """The ten clients coloured by a grouping (a small picture)."""
    labels = np.asarray(labels)
    if labels.dtype.kind in "US":   # a text like "ABCD | EFGHJ | I"
        parts = [p.strip() for p in str(labels.item() if labels.ndim == 0 else labels[0])
                 .split("|")]
        labels = np.array([next(k for k, p in enumerate(parts) if c in p) for c in _NAMES])
    _toy_dots(ax, colours=_label_colours(labels), letters=letters, s=s)
    cent = np.array([_WX[labels == k].mean(axis=0) for k in np.unique(labels)])
    _centre_marks(ax, cent, s=50)
    _toy_frame(ax, labels=frame)
    if not frame:
        ax.set_xticks([])
        ax.set_yticks([])
    if title:
        ax.set_title(title)


def draw_start_groups(ax, start: int = 0, n_init: int = 1) -> dict:
    """The groups where k-means stopped: open crosses = the three starting
    clients, filled crosses = the final centres. n_init > 1: only the final
    centres."""
    r = _e2_run(start, n_init)
    _toy_dots(ax, colours=_label_colours(r["labels"]))
    if n_init == 1:
        _centre_marks(ax, _WX[_idx(r["clients"])], open_=True)
    _centre_marks(ax, r["centres"])
    _toy_frame(ax)
    ax.set_title("where the loop stopped" if n_init == 1 else
                 f"the best of {n_init} starts")
    return r


def draw_start_bars(ax, start=None, trail=None, dark_best: bool = False) -> pd.DataFrame:
    """One bar per start tried (its inertia); the current start dark, or with
    dark_best=True the starts with the smallest inertia dark."""
    t = start_kmeans_numbers()
    tried = sorted(set(trail or []) | ({start} if start is not None else set()))
    best = t["inertia"].min()
    for s in tried:
        v = t.loc[s, "inertia"]
        dark = abs(v - best) < 1e-9 if dark_best else s == start
        ax.bar(s, v, width=0.72, color=DARK if dark else MID_BLUE)
        if not dark_best and len(tried) <= 6:
            ax.text(s, v + 1, f"{v:.1f}", ha="center", va="bottom", fontsize=8.5)
        elif not dark_best:
            ax.text(s, v + 1, f"{v:.1f}", ha="center", va="bottom", fontsize=8.5,
                    rotation=90)
    ax.set_xlim(-0.8, 19.8)
    ax.set_xticks([0, 5, 10, 15, 19])
    ax.set_ylim(0, 65)
    ax.set_xlabel("start")
    ax.set_ylabel("inertia")
    ax.set_title("the inertia of every start" if dark_best else "the starts you tried")
    return t.loc[tried]


def start_kmeans_figure(start: int = 0, n_init: int = 1, trail=None, figsize=None,
                        suptitle: bool = True):
    with _style():
        fig, axes = _new_fig(figsize or (8.0, 3.6), 1, 2, width_ratios=[1, 1.15])
        r = draw_start_groups(axes[0], start, n_init)
        draw_start_bars(axes[1], start if n_init == 1 else None, trail)
        if suptitle:
            if n_init == 1:
                fig.suptitle(f"start {start}: {', '.join(r['clients'])} - inertia "
                             f"{r['inertia']:.1f}", fontsize=9.5)
            else:
                fig.suptitle(f"start {start}, n_init = {n_init}: {n_init} starts, the "
                             f"smallest inertia kept - inertia {r['inertia']:.1f}",
                             fontsize=9.5)
    return fig


def start_kmeans(start: int = 0, n_init: int = 1) -> None:
    """E2: k-means with K = 3 on the ten clients from three starting clients;
    the inertia of every start you tried stays on the right
    (start_kmeans_reset() clears it)."""
    s = _snap(start, E2_STARTS, "start", 0) if not (
        isinstance(_plain(start), (int, np.integer)) and 0 <= int(start) <= 19) else int(start)
    k = _whole(n_init, 1, 50, 1, "n_init")
    if k == 1 and s not in _E2_TRAIL:
        _E2_TRAIL.append(s)
    _show(start_kmeans_figure(s, k, list(_E2_TRAIL)))


def start_kmeans_reset() -> None:
    _E2_TRAIL.clear()


# ---------------------------------------------------------------------------
# E3 - Units and k-means (our data: credit limit in NT$, months late)
# ---------------------------------------------------------------------------
E3_UNITS = ["NT$ and months", "standardised"]
_E3_ALIASES = {"nt": "NT$ and months", "raw": "NT$ and months", "ntmonths": "NT$ and months",
               "ntandmonth": "NT$ and months", "rawunits": "NT$ and months",
               "standardized": "standardised", "standard": "standardised",
               "scaled": "standardised"}


def _e3_colours(k: int) -> list:
    return {2: [BLUE, RED], 3: [BLUE, MID_BLUE, RED],
            4: [BLUE, MID_BLUE, LIGHT_RED, RED]}.get(k, _CLUSTER_CYCLE)


def _e3(units: str, k: int) -> np.ndarray:
    """Group numbers 1..k (largest group first) of the 21,000 training clients."""
    def build():
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        d = course_data()
        M = d.X_train[["limit_bal", "months_delayed_6m"]].to_numpy(float)
        Z = M if units == "NT$ and months" else StandardScaler().fit_transform(M)
        return _by_size(KMeans(k, n_init=10, random_state=SEED).fit(Z).labels_)
    return _cached(("e3", units, k), "the groups", build)


def _e3_sample():
    def build():
        rng = np.random.default_rng(SEED)
        idx = rng.choice(len(course_data().X_train), 2000, replace=False)
        return idx, rng.uniform(-0.25, 0.25, 2000)
    return _cached("e3_sample", "the sample", build)


def units_kmeans_numbers(k: int = 3) -> pd.DataFrame:
    """Per setting and group (largest first): clients, credit limit min-max
    (thousand NT$), months late min-max."""
    d = course_data()
    M = d.X_train[["limit_bal", "months_delayed_6m"]]
    rows = []
    for units in E3_UNITS:
        lab = _e3(units, k)
        for g in range(1, k + 1):
            m = M[lab == g]
            rows.append({"units": units, "group": g, "clients": int(len(m)),
                         "limit min (thousand)": m.limit_bal.min() / 1000,
                         "limit max (thousand)": m.limit_bal.max() / 1000,
                         "months late min": int(m.months_delayed_6m.min()),
                         "months late max": int(m.months_delayed_6m.max())})
    return pd.DataFrame(rows).set_index(["units", "group"])


def draw_units_scatter(ax, units: str = "NT$ and months", k: int = 3) -> list:
    """2,000 of the training clients: credit limit across, months late up
    (jittered), coloured by the k-means groups of all 21,000."""
    lab = _e3(units, k)
    idx, jit = _e3_sample()
    d = course_data()
    lim = d.X_train["limit_bal"].to_numpy(float)[idx] / 1000
    late = d.X_train["months_delayed_6m"].to_numpy(float)[idx] + jit
    cols = _e3_colours(k)
    sizes = [int((lab == g).sum()) for g in range(1, k + 1)]
    for g in range(1, k + 1):
        m = lab[idx] == g
        ax.scatter(lim[m], late[m], s=6, color=cols[(g - 1) % len(cols)], alpha=0.75,
                   linewidth=0, label=f"{sizes[g - 1]:,} clients")
    leg = ax.legend(loc="upper right", markerscale=2.2, handletextpad=0.2, borderaxespad=0.2,
                    fontsize=8.5)
    for h in leg.legend_handles:
        h.set_alpha(1)
    ax.set_xlim(0, 1020)
    ax.set_ylim(-0.5, 6.5)
    ax.set_yticks(range(7))
    ax.set_xlabel("credit limit (thousand NT\\$)")
    ax.set_ylabel("months late (of the last 6)")
    ax.set_title("NT\\$ and months" if units == "NT$ and months" else "standardised")
    return sizes


def units_kmeans_figure(units: str = "NT$ and months", k: int = 3, figsize=None,
                        suptitle: bool = True):
    with _style():
        fig, ax = _new_fig(figsize or (5.6, 3.7))
        sizes = draw_units_scatter(ax, units, k)
        if suptitle:
            fig.suptitle(f"K = {k}: group sizes " + " / ".join(f"{s:,}" for s in sizes),
                         fontsize=9.5)
    return fig


def units_kmeans(units: str = "NT$ and months", k: int = 3) -> None:
    """E3: k-means on the credit limit and the months late of our 21,000
    training clients, in their own units or standardised."""
    units = _choice(units, E3_UNITS, "units", _E3_ALIASES)
    k = _whole(k, 2, 8, 3, "k")
    _show(units_kmeans_figure(units, k))


# ---------------------------------------------------------------------------
# E4 - Grow DBSCAN (crescents with 8 strays)
# ---------------------------------------------------------------------------
E4_EPS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7]
_CRESCENT_WINDOW = ((-1.5, 2.5), (-1.0, 1.5))


def crescents_with_strays():
    """Made-up crescents: make_moons(300, noise=0.06, random_state=42) and 8 stray
    points (default_rng(7): 40 uniform points drawn, the first 8 that lie more
    than 0.35 from every crescent point). Returns X (308 x 2) and crescent
    (0 / 1 for the crescents, -1 for the 8 strays). Copies."""
    def build():
        from scipy.spatial.distance import cdist
        from sklearn.datasets import make_moons
        Xm, ym = make_moons(300, noise=0.06, random_state=SEED)
        rng = np.random.default_rng(7)
        out = np.column_stack([rng.uniform(-1.4, 2.4, 40), rng.uniform(-0.9, 1.4, 40)])
        out = out[np.min(cdist(out, Xm), axis=1) > 0.35][:8]
        return np.vstack([Xm, out]), np.concatenate([ym, np.full(len(out), -1)])
    X, c = _cached("crescents", "the crescents", build)
    return X.copy(), c.copy()


def random_points() -> np.ndarray:
    """300 random points, uniform in the unit square (default_rng(3)): data with
    no groups (slide 65)."""
    return np.random.default_rng(3).uniform(0, 1, (300, 2))


def _dbscan(eps: float, min_samples: int):
    def build():
        from sklearn.cluster import DBSCAN
        X, _ = crescents_with_strays()
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
        core = np.zeros(len(X), bool)
        core[db.core_sample_indices_] = True
        return db.labels_, core
    return _cached(("dbscan", round(float(eps), 6), int(min_samples)), "DBSCAN", build)


def grow_dbscan_numbers(min_samples: int = 5, eps_list=None) -> pd.DataFrame:
    """For each eps: clusters, noise points, strays marked noise (of 8) and
    crescent points marked noise."""
    _, c = crescents_with_strays()
    rows = []
    for eps in (eps_list or E4_EPS):
        lab, _ = _dbscan(eps, min_samples)
        rows.append({"clusters": len(set(lab) - {-1}), "noise": int((lab == -1).sum()),
                     "strays as noise": int((lab[c == -1] == -1).sum()),
                     "crescent points as noise": int((lab[c >= 0] == -1).sum())})
    return pd.DataFrame(rows, index=pd.Index(list(eps_list or E4_EPS), name="eps"))


def _crescent_frame(ax) -> None:
    ax.set_xlim(*_CRESCENT_WINDOW[0])
    ax.set_ylim(*_CRESCENT_WINDOW[1])
    ax.set_aspect("equal")
    ax.set_xticks([-1, 0, 1, 2])
    ax.set_yticks([-1, 0, 1])


def _cluster_order(X, lab) -> list:
    ks = [k for k in set(lab) if k >= 0]
    return sorted(ks, key=lambda k: (-(lab == k).sum(), X[lab == k, 0].mean()))


def draw_kmeans_crescents(ax) -> dict:
    """K-means with K = 2 on the crescents and strays (every point in a cluster)."""
    from sklearn.cluster import KMeans
    X, c = crescents_with_strays()
    km = KMeans(2, n_init=10, random_state=SEED).fit(X)
    order = sorted([0, 1], key=lambda k: X[km.labels_ == k, 0].mean())
    for k, col in zip(order, (BLUE, RED)):
        m = km.labels_ == k
        ax.scatter(X[m, 0], X[m, 1], s=10, color=col, linewidth=0)
    _centre_marks(ax, km.cluster_centers_, s=70)
    _crescent_frame(ax)
    ax.set_title("k-means, K = 2")
    y = c[c >= 0]
    lab = km.labels_[c >= 0]
    return {"with majority": max((lab == y).mean(), (lab != y).mean()),
            "strays in a cluster": int((c == -1).sum())}


def draw_dbscan(ax, eps: float = 0.2, min_samples: int = 5) -> dict:
    """DBSCAN on the crescents: colours = clusters, big dots = core points,
    small dots = border points, black x = noise."""
    X, _ = crescents_with_strays()
    lab, core = _dbscan(eps, min_samples)
    for i, k in enumerate(_cluster_order(X, lab)):
        col = _CLUSTER_CYCLE[i % len(_CLUSTER_CYCLE)]
        for is_core, size in ((True, 14), (False, 4)):
            m = (lab == k) & (core == is_core)
            ax.scatter(X[m, 0], X[m, 1], s=size, color=col, linewidth=0)
    m = lab == -1
    ax.scatter(X[m, 0], X[m, 1], s=16, marker="x", color=INK, linewidth=0.9)
    _crescent_frame(ax)
    n_c, n_n = len(set(lab) - {-1}), int(m.sum())
    ax.set_title(f"DBSCAN, eps {eps:.2f}")
    return {"eps": eps, "min_samples": min_samples, "clusters": n_c, "noise": n_n}


def _dbscan_title(eps, min_samples, n_c, n_n) -> str:
    words = (f"eps {eps:.2f}: {n_c} cluster{'s' if n_c != 1 else ''}, {n_n} noise "
             f"point{'s' if n_n != 1 else ''}")
    return words if min_samples == 5 else words + f" (min_samples = {min_samples})"


def grow_dbscan_figure(eps: float = 0.2, min_samples: int = 5, figsize=None,
                       suptitle: bool = True):
    with _style():
        fig, axes = _new_fig(figsize or (8.0, 3.2), 1, 2)
        draw_kmeans_crescents(axes[0])
        r = draw_dbscan(axes[1], eps, min_samples)
        if suptitle:
            fig.suptitle(_dbscan_title(eps, min_samples, r["clusters"], r["noise"]),
                         fontsize=9.5)
    return fig


def grow_dbscan(eps: float = 0.2, min_samples: int = 5) -> None:
    """E4: DBSCAN on the crescents and 8 strays; on the left, for comparison,
    k-means with K = 2."""
    value = _decimal_comma(_plain(eps), "eps")
    e = _as_number(value)
    if not (_is_number(e) and 0 < e <= 3):
        new = 0.2 if not _is_number(e) else float(min(max(e, 0.01), 3))
        shown = f"{value:g}" if _is_number(value) else repr(value)
        _say(f"eps {shown} is not a radius from 0.01 to 3: showing eps = {new:g}.")
        e = new
    ms = _whole(min_samples, 1, 50, 5, "min_samples")
    _show(grow_dbscan_figure(round(float(e), 4), ms))


# ---------------------------------------------------------------------------
# The four segments (k-means K = 4 on the five standardised columns)
# ---------------------------------------------------------------------------
def _renumber(raw, F: pd.DataFrame) -> dict:
    """The target-free rule: 1 = highest mean payment ratio, 4 = highest mean
    months late, 3 = the higher median card use of the other two, 2 = the last."""
    g = F.assign(s=raw).groupby("s")
    s1 = g["pay_ratio"].mean().idxmax()
    late = g["months_late"].mean().drop(s1)
    s4 = late.idxmax()
    rest = [k for k in sorted(set(raw)) if k not in (s1, s4)]
    med = g["card_use"].median()
    s3 = max(rest, key=lambda k: med[k])
    s2 = [k for k in rest if k != s3][0]
    return {s1: 1, s2: 2, s3: 3, s4: 4}


def _segments() -> dict:
    def build():
        from sklearn.cluster import KMeans
        r = _real()
        km = KMeans(4, n_init=10, random_state=SEED).fit(r["Z_tr"])
        mp = _renumber(km.labels_, r["F_tr"])
        train = np.array([mp[v] for v in km.labels_])
        test = np.array([mp[v] for v in km.predict(r["Z_te"])])
        return {"model": km, "map": mp, "raw": km.labels_, "train": train, "test": test}
    return _cached("segments", "the four segments", build)


def _profile() -> pd.DataFrame:
    s = _segments()
    r = _real()
    F = r["F_tr"].assign(s=s["train"])
    no_aug = (course_data().df.loc[r["F_tr"].index, "bill_amt_aug"] <= 0).to_numpy()
    g = F.groupby("s")
    prof = pd.DataFrame({
        "name": pd.Series(SEGMENT_NAMES),
        "clients": g.size(),
        "credit limit (median, NT$)": (10 ** g["log_limit"].median()).round(-3).astype(int),
        "card use (median)": g["card_use"].median(),
        "months late (mean)": g["months_late"].mean(),
        "pays the whole bill (share)": g["pay_ratio"].apply(lambda v: (v >= 0.99).mean()),
        "bill trend (median)": g["bill_trend"].median(),
        "no August bill (share)": pd.Series(no_aug).groupby(s["train"]).mean(),
    })
    prof.index.name = "segment"
    return prof


def segments():
    """The four segments: KMeans(4, n_init=10, random_state=42) on the five
    standardised columns of the training rows, numbered 1-4 by the target-free
    rule (1 = highest mean payment ratio, 4 = highest mean months late, 3 = the
    higher median card use of the other two, 2 = the last). Returns
    train_labels (21,000), test_labels (9,000, assigned with predict) and the
    profile table (slide 50; default is not in it). Copies."""
    s = _segments()
    return s["train"].copy(), s["test"].copy(), _profile()


def outside_check() -> pd.DataFrame:
    """The outside check: the share who defaulted in each segment, on the
    training rows and on the test rows assigned to the nearest centre."""
    s = _segments()
    r = _real()
    out = pd.DataFrame({
        "training clients": pd.Series(s["train"]).value_counts().sort_index(),
        "default, training (%)": 100 * pd.Series(r["y_tr"]).groupby(s["train"]).mean(),
        "test clients": pd.Series(s["test"]).value_counts().sort_index(),
        "default, test (%)": 100 * pd.Series(r["y_te"]).groupby(s["test"]).mean(),
    })
    out.index.name = "segment"
    return out


def segment_centres_pcs() -> pd.DataFrame:
    """Each segment's centre in PC1 and PC2 scores (the five columns' PCA)."""
    p, S = _pca_five()
    return pd.DataFrame(S[:, :2], columns=["PC1", "PC2"]).groupby(
        _segments()["train"]).mean().rename_axis("segment")


def _pca_five():
    def build():
        Z = standardised_five()
        p = pca_fixed(Z, "five")
        return p, p.transform(Z.to_numpy())
    return _cached("pca_five", "the PCA of the five columns", build)


# ---------------------------------------------------------------------------
# PCA on our data (Demo 1, slides 23-25, 51)
# ---------------------------------------------------------------------------
_PCA_WHICH = ["bills", "payments", "five", "model"]


def _pca_data(which: str):
    def build():
        from sklearn.preprocessing import StandardScaler
        if which == "bills":
            B = bills()
        elif which == "payments":
            B = payments()
        elif which == "five":
            B = five_columns()
        else:
            B = course_data().X_train.copy()
        Z = pd.DataFrame(StandardScaler().fit_transform(B.to_numpy(float)), index=B.index,
                         columns=B.columns)
        p = pca_fixed(Z, which)
        return p, p.transform(Z.to_numpy()), list(B.columns)
    return _cached(("pca", which), f"the PCA of the {which}", build)


def pca_table(which: str = "bills") -> pd.DataFrame:
    """PCA of the standardised training columns ("bills", "payments", "five" or
    "model" = the 12 columns), signs fixed: per component its PVE (%), the
    running total (%) and its loadings (one column per input column)."""
    which = _choice(which, _PCA_WHICH, "which", {"bill": "bills", "payment": "payments",
                                                  "fivecolumns": "five", "12": "model"})
    p, _, cols = _pca_data(which)
    pve = 100 * p.explained_variance_ratio_
    t = pd.DataFrame(p.components_, columns=[LABELS.get(c, c) for c in cols],
                     index=pd.Index([f"PC{k + 1}" for k in range(len(pve))], name="component"))
    t.insert(0, "running total %", np.cumsum(pve))
    t.insert(0, "PVE %", pve)
    return t


def draw_scree(ax, which: str = "bills", running: bool = True, title=None,
               bar_labels: bool = True) -> np.ndarray:
    """A scree plot: bars = the PVE of each component; dotted = the running total."""
    t = pca_table(which)
    pve, run = t["PVE %"].to_numpy(), t["running total %"].to_numpy()
    xs = np.arange(1, len(pve) + 1)
    ax.bar(xs, pve, width=0.62, color=BLUE)
    if bar_labels:
        for x, v in zip(xs, pve):
            if v > 15:   # inside the bar: the running total starts on its top
                ax.text(x, v / 2, f"{v:.0f}%", ha="center", va="center", color="white",
                        fontsize=9, fontweight="bold")
            else:
                ax.text(x, v + 1.5, f"{v:.0f}%", ha="center", va="bottom", fontsize=8.5)
    if running:
        ax.plot(xs, run, ls=":", color=INK, marker="o", ms=3.5, lw=1.2)
    ax.set_xticks(xs)
    ax.set_xlim(0.4, len(pve) + 0.6)
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlabel("component")
    ax.set_ylabel("share of the variance (%)")
    if title:
        ax.set_title(title)
    return pve


def draw_bills_vs_payments(axes) -> dict:
    """The scree plots of the six bills (left) and the six payments (right)."""
    return {"bills": draw_scree(axes[0], "bills", title="six bills"),
            "payments": draw_scree(axes[1], "payments", title="six payments")}


def draw_loadings(ax, which: str = "bills") -> pd.DataFrame:
    """The PC1 and PC2 loadings of each column (for the bills: month by month,
    latest first)."""
    t = pca_table(which)
    cols = [c for c in t.columns if c not in ("PVE %", "running total %")]
    L = t.loc[["PC1", "PC2"], cols]
    xs = np.arange(len(cols))
    ax.axhline(0, color=GRAY, lw=0.8)
    ax.plot(xs, L.loc["PC1"], "-o", color=DARK, ms=4, lw=1.4, label="PC1")
    ax.plot(xs, L.loc["PC2"], "-o", color=MID_BLUE, ms=4, lw=1.4, label="PC2")
    short = [c.replace(" bill", "").replace(" payment", "")[:3] if which in
             ("bills", "payments") else c for c in cols]
    ax.set_xticks(xs)
    ax.set_xticklabels(short, rotation=0 if which in ("bills", "payments") else 30,
                       ha="center" if which in ("bills", "payments") else "right")
    ax.set_ylim(-0.8, 0.8)
    ax.set_ylabel("loading")
    ax.legend(loc="lower left", ncol=2)
    ax.set_title("loadings" + (" (latest month first)" if which in ("bills", "payments")
                               else ""))
    return L


def draw_biplot(ax, which: str = "bills", n: int = 2000) -> None:
    """2,000 training clients at their PC1 and PC2 scores, and one arrow per
    column (its two loadings, scaled)."""
    t = pca_table(which)
    _, S, cols = _pca_data(_choice(which, _PCA_WHICH, "which"))
    idx = np.random.default_rng(SEED).choice(len(S), n, replace=False)
    P = S[idx, :2]
    ax.scatter(P[:, 0], P[:, 1], s=4, color=GRAY, alpha=0.5, linewidth=0)
    lo = np.percentile(P, 1, axis=0)
    hi = np.percentile(P, 99, axis=0)
    names = [c for c in t.columns if c not in ("PVE %", "running total %")]
    L = t.loc[["PC1", "PC2"], names].to_numpy().T
    scale = 0.8 * min(hi[0] - 0, hi[1] - 0, 0 - lo[0], 0 - lo[1]) / np.abs(L).max() \
        if min(hi[0], hi[1], -lo[0], -lo[1]) > 0 else 1.0
    for (a, b), name in zip(L, names):
        ax.annotate("", xy=(a * scale, b * scale), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=DARK, lw=1.1, mutation_scale=8))
        short = name.replace(" bill", "").replace(" payment", "")[:3] if which in (
            "bills", "payments") else name
        ax.text(a * scale * 1.12, b * scale * 1.12, short, fontsize=8.5, color=DARK,
                ha="center", va="center")
    pad = 0.15 * (hi - lo)
    ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
    ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
    ax.set_xlabel("PC1 score")
    ax.set_ylabel("PC2 score")
    ax.set_title("clients and loadings")


def pca_columns(which: str = "bills") -> pd.DataFrame:
    """D1: PCA of the six bills (or "payments", "five", "model"), standardised:
    the scree plot, the PC1 and PC2 loadings, and a biplot. Returns
    pca_table(which)."""
    which = _choice(which, _PCA_WHICH, "which", {"bill": "bills", "payment": "payments",
                                                  "fivecolumns": "five", "12": "model"})
    with _style():
        fig, axes = _new_fig((10.0, 3.4), 1, 3)
        draw_scree(axes[0], which, title="scree plot")
        draw_loadings(axes[1], which)
        draw_biplot(axes[2], which)
        words = {"bills": "the six monthly bills", "payments": "the six monthly payments",
                 "five": "the five clustering columns", "model": "the 12 model columns"}
        fig.suptitle(f"PCA of {words[which]}, 21,000 training clients, standardised",
                     fontsize=9.5)
    _show(fig)
    return pca_table(which)


def pc1_default_numbers() -> dict:
    """The share who defaulted in each quarter of the bills' PC1 score (lowest
    first) and the AUC of PC1 alone."""
    from sklearn.metrics import roc_auc_score
    _, S, _ = _pca_data("bills")
    y = _real()["y_tr"]
    q = pd.qcut(S[:, 0], 4, labels=False)
    auc = roc_auc_score(y, S[:, 0])
    return {"default by quarter (%)": (100 * pd.Series(y).groupby(q).mean()).to_numpy(),
            "AUC of PC1": max(auc, 1 - auc), "all training clients (%)": 100 * y.mean()}


def draw_pc1_default(ax) -> dict:
    """The share who defaulted in each quarter of the bills' PC1 score."""
    r = pc1_default_numbers()
    v = r["default by quarter (%)"]
    xs = np.arange(4)
    ax.bar(xs, v, width=0.6, color=BLUE)
    for x, val in zip(xs, v):
        ax.text(x, val - 1.5, f"{val:.1f}%", ha="center", va="top", fontsize=9,
                color="white", fontweight="bold")
    base = r["all training clients (%)"]
    ax.axhline(base, ls="--", color=INK, lw=1)
    ax.text(0.99, 0.98, f"dashed: all training clients, {base:.1f}%", ha="right", va="top",
            fontsize=8.5, transform=ax.transAxes)
    ax.set_xticks(xs)
    ax.set_xticklabels(["lowest", "second", "third", "highest"])
    ax.set_xlabel("quarter of the PC1 score of the six bills")
    ax.set_ylabel("share who defaulted (%)")
    ax.set_ylim(0, 35)
    return r


# ---------------------------------------------------------------------------
# D2 - The elbow on our data; D3 - the segment table
# ---------------------------------------------------------------------------
def elbow_numbers() -> pd.DataFrame:
    """Inertia of KMeans(K, n_init=10, random_state=42) on the five standardised
    columns for K = 1 to 10, the drop to the next K, and the drop as a share of
    the inertia at K = 1."""
    def build():
        from sklearn.cluster import KMeans
        Z = _real()["Z_tr"]
        return [KMeans(k, n_init=10, random_state=SEED).fit(Z).inertia_ for k in range(1, 11)]
    ins = _cached("elbow", "the elbow (K = 1 to 10)", build, slow=True)
    t = pd.DataFrame({"inertia": ins}, index=pd.Index(range(1, 11), name="K"))
    t["drop to the next K"] = t["inertia"].diff(-1)
    t["drop, % of K = 1"] = 100 * t["drop to the next K"] / ins[0]
    return t


def draw_elbow_ours(ax, mark: int = 4, n_drops: int = 6) -> pd.DataFrame:
    """Inertia for K = 1 to 10 (thousands); the drops written between the dots;
    K = mark ringed."""
    t = elbow_numbers()
    v = t["inertia"].to_numpy() / 1000
    ks = t.index.to_numpy()
    ax.plot(ks, v, "-o", color=DARK, ms=4.5, lw=1.4, zorder=2)
    ax.set_xlim(0.5, 10.5)
    ax.set_ylim(0, 115)
    for k in range(min(n_drops, len(ks) - 1)):
        d = v[k] - v[k + 1]
        # each drop (in whole thousands, as the slide says them) next to the
        # middle of its own segment, at a right angle to it: above the steep
        # segments, below the flat ones (computed on the page, at any size)
        p, q = ax.transData.transform([(ks[k], v[k]), (ks[k + 1], v[k + 1])])
        n = np.array([p[1] - q[1], q[0] - p[0]])   # up and to the right
        n = n / np.hypot(*n)
        steep = abs(p[1] - q[1]) > 0.6 * abs(q[0] - p[0])
        off = n * 9 if steep else -n * 9
        ax.annotate(f"{d:.0f}k", ((ks[k] + ks[k + 1]) / 2, (v[k] + v[k + 1]) / 2),
                    xytext=tuple(off), textcoords="offset points", ha="center", va="center",
                    fontsize=8.5, color=MUTED)
    i = list(ks).index(mark)
    ax.scatter([mark], [v[i]], s=170, facecolor="none", edgecolor=RED, linewidth=1.4, zorder=3)
    ax.annotate(f"K = {mark}", (mark, v[i]), xytext=(-12, -14), textcoords="offset points",
                ha="right", va="top", fontsize=9, color=RED)
    ax.set_xticks(ks)
    ax.set_xlim(0.5, 10.5)
    ax.set_ylim(0, 115)
    ax.set_xlabel("K, the number of segments")
    ax.set_ylabel("inertia (thousands)")
    return t


def elbow_ours() -> pd.DataFrame:
    """D2: the elbow on our 21,000 training clients and five columns."""
    t = elbow_numbers()
    with _style():
        fig, ax = _new_fig((5.6, 3.4))
        draw_elbow_ours(ax)
        fig.suptitle("k-means (n_init = 10) on the five standardised columns", fontsize=9.5)
    _show(fig)
    return t


def draw_segments_pcs(ax, n: int = 2000) -> pd.DataFrame:
    """2,000 training clients at their PC1 and PC2 scores (the five columns'
    PCA), coloured by segment; the segment numbers at the segment centres."""
    _, S = _pca_five()
    seg = _segments()["train"]
    idx = np.random.default_rng(SEED).choice(len(S), n, replace=False)
    for k in (1, 2, 3, 4):
        m = seg[idx] == k
        ax.scatter(S[idx][m, 0], S[idx][m, 1], s=6, color=SEG_COLOURS[k], linewidth=0.2,
                   edgecolor="#D9978C" if k == 3 else SEG_COLOURS[k], alpha=0.85)
    cen = segment_centres_pcs()
    for k, row in cen.iterrows():
        ax.text(row["PC1"], row["PC2"], str(k), ha="center", va="center", fontsize=10,
                fontweight="bold", color=INK, zorder=5,
                bbox=dict(boxstyle="circle,pad=0.25", fc="white", ec=INK, lw=0.8))
    P = S[idx, :2]
    lo, hi = np.percentile(P, 0.5, axis=0), np.percentile(P, 99.5, axis=0)
    ax.set_xlim(lo[0] - 0.3, hi[0] + 0.3)
    ax.set_ylim(lo[1] - 0.3, hi[1] + 0.3)
    ax.set_xlabel("PC1 score")
    ax.set_ylabel("PC2 score")
    return cen


def segment_table() -> pd.DataFrame:
    """D3: the four segments on PC1 and PC2; returns the profile table
    (slide 50), with months late rounded to one decimal as on the slide
    (segment 1's 0.249 shows as 0.2, not 0.25)."""
    prof = _profile()
    prof["months late (mean)"] = prof["months late (mean)"].round(1)
    with _style():
        fig, ax = _new_fig((5.6, 3.8))
        draw_segments_pcs(ax)
        pve = 100 * _pca_five()[0].explained_variance_ratio_
        fig.suptitle(f"our four segments on PC1 ({pve[0]:.0f}%) and PC2 ({pve[1]:.0f}%) "
                     "of the five columns", fontsize=9.5)
    _show(fig)
    return prof


# ---------------------------------------------------------------------------
# D4 - The tree of merges on 300 of our clients
# ---------------------------------------------------------------------------
_LINKAGES = ["complete", "average", "single", "ward"]


def _tree_ours(linkage: str):
    def build():
        from scipy.cluster.hierarchy import fcluster
        from scipy.cluster.hierarchy import linkage as scipy_linkage
        idx = np.random.default_rng(SEED).choice(len(_real()["Z_tr"]), 300, replace=False)
        Zs = _real()["Z_tr"][idx]
        L = scipy_linkage(Zs, linkage)
        groups = _by_size(fcluster(L, 4, "maxclust"))
        return idx, L, groups
    return _cached(("tree", linkage), "the tree", build)


def hierarchy_numbers(linkage: str = "complete") -> dict:
    """The tree of 300 sampled training clients cut into 4 groups: group sizes
    (largest first) and the share in the same group as the k-means segments
    (after matching)."""
    linkage = _choice(linkage, _LINKAGES, "linkage")
    idx, _, groups = _tree_ours(linkage)
    return {"sizes": np.bincount(groups)[1:], "same as k-means": agree(
        _segments()["train"][idx], groups)}


def draw_dendrogram_ours(ax, linkage: str = "complete") -> dict:
    """The dendrogram of 300 sampled training clients; dashed = the cut into 4."""
    from scipy.cluster.hierarchy import dendrogram, set_link_color_palette
    linkage = _choice(linkage, _LINKAGES, "linkage")
    _, L, _ = _tree_ours(linkage)
    cut = (L[-4, 2] + L[-3, 2]) / 2
    set_link_color_palette([BLUE, RED, DARK, MID_BLUE])
    try:
        dendrogram(L, ax=ax, no_labels=True, color_threshold=cut, above_threshold_color=GRAY)
    finally:
        set_link_color_palette(None)
    ax.axhline(cut, ls="--", color=INK, lw=1)
    ax.text(ax.get_xlim()[1], cut, " cut: 4 groups", ha="right", va="bottom", fontsize=8.5)
    ax.set_ylabel("height = distance at the merge")
    ax.set_xlabel("300 training clients")
    ax.spines["bottom"].set_visible(False)
    return {"cut": cut, **hierarchy_numbers(linkage)}


def dendrogram_ours(linkage: str = "complete") -> pd.DataFrame:
    """D4: the tree of merges of 300 sampled training clients (standardised
    five columns), cut into 4 groups; returns the crosstab tree group x
    k-means segment."""
    linkage = _choice(linkage, _LINKAGES, "linkage")
    idx, _, groups = _tree_ours(linkage)
    with _style():
        fig, ax = _new_fig((7.5, 3.4))
        r = draw_dendrogram_ours(ax, linkage)
        fig.suptitle(f"{linkage} linkage, 300 training clients: groups of "
                     + " / ".join(str(s) for s in r["sizes"]), fontsize=9.5)
    _show(fig)
    _say(f"The tree's 4 groups and the k-means segments agree for "
         f"{r['same as k-means']:.0%} of these 300 clients (after matching).")
    return pd.crosstab(pd.Series(groups, name="tree group"),
                       pd.Series(_segments()["train"][idx], name="k-means segment"))


# ---------------------------------------------------------------------------
# E5 - Clusters in noise? (ours, a shuffled copy, random noise)
# ---------------------------------------------------------------------------
E5_DATA = ["ours", "shuffled", "noise"]
E5_KS = list(range(2, 9))
_E5_NAME = {"ours": "our clients", "shuffled": "the shuffled copy", "noise": "random noise"}
BANDS = [(0.71, 1.0, "0.71-1 strong"), (0.51, 0.70, "0.51-0.70 reasonable"),
         (0.26, 0.50, "0.26-0.50 weak, could be artificial"),
         (0.0, 0.25, "0-0.25 no substantial structure")]


def null_tables() -> dict:
    """Three tables of 21,000 rows and five columns: "ours" (the standardised
    five), "shuffled" (each column shuffled on its own) and "noise" (random
    normal numbers). One np.random.default_rng(42): the five permutations
    first, then standard_normal. Copies."""
    def build():
        Z = _real()["Z_tr"]
        rng = np.random.default_rng(SEED)
        sh = np.column_stack([rng.permutation(Z[:, j]) for j in range(Z.shape[1])])
        return {"ours": Z, "shuffled": sh, "noise": rng.standard_normal(Z.shape)}
    t = _cached("nulls", "the null tables", build)
    return {k: v.copy() for k, v in t.items()}


def _e5(data: str) -> dict:
    def build():
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        M = null_tables()[data]
        sil, lab4 = [], None
        for k in E5_KS:
            lab = KMeans(k, n_init=10, random_state=SEED).fit(M).labels_
            sil.append(silhouette_score(M, lab, sample_size=5000, random_state=SEED))
            if k == 4:
                lab4 = lab
        groups = np.array([_segments()["map"][v] for v in lab4]) if data == "ours" \
            else _by_size(lab4)
        y = _real()["y_tr"]
        return {"silhouette": np.array(sil), "groups": groups,
                "default": (100 * pd.Series(y).groupby(groups).mean()).to_numpy(),
                "sizes": np.bincount(groups)[1:]}
    return _cached(("e5", data), f"the clusters of {_E5_NAME[data]}", build, slow=True)


def clusters_in_noise_numbers() -> pd.DataFrame:
    """Per table: the mean silhouette at K = 2 to 8 (5,000 sampled rows) and the
    default rate (%) of each of its four groups at K = 4."""
    rows = {}
    for d in E5_DATA:
        r = _e5(d)
        rows[d] = {**{f"silhouette K={k}": v for k, v in zip(E5_KS, r["silhouette"])},
                   **{f"default % group {g}": v for g, v in enumerate(r["default"], 1)}}
    return pd.DataFrame(rows).T.rename_axis("table")


def _bands(ax, labels: bool = True, x_text=None) -> None:
    for k, (lo, hi, words) in enumerate(BANDS):
        if k % 2 == 0:
            ax.axhspan(lo, hi, color=SHADE, zorder=0, lw=0)
        if labels:   # at the top of the band (the lowest band: at its bottom), clear of the lines
            y, va = (lo + 0.012, "bottom") if lo == 0 else (hi - 0.012, "top")
            ax.text(x_text if x_text is not None else ax.get_xlim()[1], y,
                    words + " ", ha="right", va=va, fontsize=8.5, color=MUTED, zorder=1)


def draw_null_silhouettes(ax, data: str = "ours", compare=("ours",), bands: bool = True,
                          band_labels: bool = False) -> dict:
    """The mean silhouette at K = 2 to 8: `data` dark, the tables in `compare`
    grey; grey bands = the usual reading (Kaufman and Rousseeuw)."""
    ax.set_xlim(1.6, 9.2)   # room for the tables' names after K = 8
    ax.set_ylim(0, 1)
    if bands:
        _bands(ax, band_labels)
    out = {}
    for d in [c for c in compare if c != data] + ([data] if data else []):
        r = _e5(d)
        dark = d == data
        ax.plot(E5_KS, r["silhouette"], "-o", color=DARK if dark else GRAY,
                lw=1.8 if dark else 1.1, ms=4.5 if dark else 3.5, zorder=3 if dark else 2)
        ax.annotate(d, (E5_KS[-1], r["silhouette"][-1]), xytext=(4, 0),
                    textcoords="offset points", ha="left", va="center", fontsize=9,
                    color=DARK if dark else MUTED, annotation_clip=False)
        out[d] = r["silhouette"]
    ax.set_xticks(E5_KS)
    ax.set_xlabel("K, the number of groups")
    ax.set_ylabel("mean silhouette")
    return out


def draw_silhouette_ours(ax) -> np.ndarray:
    """Slide 64: the mean silhouette of our segments, K = 2 to 8, on a fixed
    0-1 axis with the reading bands."""
    draw_null_silhouettes(ax, "ours", compare=(), band_labels=True)
    return _e5("ours")["silhouette"]


def draw_null_defaults(ax, tables=("ours",), base: bool = True) -> dict:
    """The share who defaulted in each of the four groups at K = 4, for each
    table; dashed = all training clients (22.1%)."""
    tables = list(tables)
    out = {}
    width = 0.8 / max(len(tables), 1) if len(tables) > 1 else 0.6
    for j, d in enumerate(tables):
        r = _e5(d)
        if len(tables) == 1:
            xs = np.arange(1, 5)
        else:
            xs = np.arange(4) + j * 5
        cols = [SEG_COLOURS[g] for g in range(1, 5)] if d == "ours" else [GRAY] * 4
        ax.bar(xs, r["default"], width=0.7, color=cols,
               edgecolor=["#D9978C" if (d == "ours" and g == 3) else c
                          for g, c in zip(range(1, 5), cols)], linewidth=0.6)
        for x, v, c in zip(xs, r["default"], cols):
            ax.text(x, v - 1.2, f"{v:.0f}", ha="center", va="top", fontsize=8.5,
                    color=INK if c == LIGHT_RED else "white", fontweight="bold")
        out[d] = r["default"]
    if base:
        b = 100 * default_rate()
        ax.axhline(b, ls="--", color=INK, lw=1, label=f"everyone, {b:.1f}%")
        ax.legend(loc="upper left", handlelength=1.8)
    if len(tables) == 1:
        ax.set_xticks(np.arange(1, 5))
        ax.set_xlabel("segment" if tables[0] == "ours" else "group (largest first)")
    else:
        ax.set_xticks([1.5 + 5 * j for j in range(len(tables))])
        ax.set_xticklabels(tables)
        ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, 70)
    ax.set_ylabel("share who defaulted (%)")
    return out


def clusters_in_noise_figure(data: str = "ours", figsize=None, suptitle: bool = True):
    r = _e5(data)
    with _style():
        fig, axes = _new_fig(figsize or (8.0, 3.4), 1, 2, width_ratios=[1.15, 1])
        draw_null_silhouettes(axes[0], data, compare=("ours",), band_labels=True)
        axes[0].set_title("mean silhouette, K = 2 to 8")
        draw_null_defaults(axes[1], (data,))
        axes[1].set_title("default in each group, K = 4")
        if suptitle:
            sil4 = r["silhouette"][list(E5_KS).index(4)]
            fig.suptitle(f"{_E5_NAME[data]}: 21,000 rows, five columns - silhouette "
                         f"{sil4:.2f} at K = 4", fontsize=9.5)
    return fig


def clusters_in_noise(data: str = "ours") -> None:
    """E5: k-means on our five columns, on a shuffled copy and on random noise
    (21,000 rows each). Each table is computed once (a few seconds), then
    kept."""
    data = _choice(data, E5_DATA, "data", {"our": "ours", "ourclients": "ours",
                                            "shuffle": "shuffled", "random": "noise",
                                            "randomnoise": "noise"})
    _show(clusters_in_noise_figure(data))


# ---------------------------------------------------------------------------
# D5 - Do the segments come back? The outside check
# ---------------------------------------------------------------------------
def stability_numbers() -> dict:
    """The share of the 21,000 clients in the same segment (after matching):
    other random_state (1-5, n_init=10), and five fits on a random half of the
    clients (its own default_rng(42)) with all 21,000 then assigned."""
    def build():
        from sklearn.cluster import KMeans
        Z = _real()["Z_tr"]
        base = _segments()["raw"]
        seeds = [agree(base, KMeans(4, n_init=10, random_state=s).fit(Z).labels_)
                 for s in (1, 2, 3, 4, 5)]
        rng = np.random.default_rng(SEED)
        halves = []
        for _ in range(5):
            idx = rng.choice(len(Z), len(Z) // 2, replace=False)
            km = KMeans(4, n_init=10, random_state=SEED).fit(Z[idx])
            halves.append(agree(base, km.predict(Z)))
        return {"other random_state": np.array(seeds), "random halves": np.array(halves)}
    r = _cached("stability", "the stability checks (ten more k-means runs)", build, slow=True)
    return {k: v.copy() for k, v in r.items()}


def draw_stability(ax) -> dict:
    """One bar per check: the share of clients in the same segment."""
    r = stability_numbers()
    vals = list(100 * r["other random_state"]) + list(100 * r["random halves"])
    xs = list(range(5)) + list(range(6, 11))
    ax.bar(xs, vals, width=0.7, color=[MID_BLUE] * 5 + [BLUE] * 5)
    for x, v in zip(xs, vals):
        ax.text(x, v - 1.5, f"{v:.1f}", ha="center", va="top", fontsize=8.5, rotation=90,
                color="white")
    ax.set_xticks([2, 8])
    ax.set_xticklabels(["random_state 1-5", "random halves 1-5"])
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("clients in the same segment (%)")
    return r


def draw_default_by_segment(ax) -> pd.DataFrame:
    """The share who defaulted per segment: training rows (light) and test rows
    assigned to the nearest centre (dark); dashed = all training clients."""
    t = outside_check()
    xs = np.arange(1, 5)
    ax.bar(xs - 0.19, t["default, training (%)"], width=0.36, color=LIGHT_RED,
           label="training clients")
    ax.bar(xs + 0.19, t["default, test (%)"], width=0.36, color=RED, label="test clients")
    for x, a, b in zip(xs, t["default, training (%)"], t["default, test (%)"]):
        ax.text(x - 0.19, a - 1, f"{a:.1f}", ha="center", va="top", fontsize=8.5,
                rotation=90, color=INK)
        ax.text(x + 0.19, b - 1, f"{b:.1f}", ha="center", va="top", fontsize=8.5,
                rotation=90, color="white")
    base = 100 * default_rate()
    ax.axhline(base, ls="--", color=INK, lw=1, label=f"everyone, {base:.1f}%")
    ax.set_xticks(xs)
    ax.set_xlim(0.45, 4.55)
    ax.set_ylim(0, 70)
    ax.set_xlabel("segment")
    ax.set_ylabel("share who defaulted (%)")
    ax.legend(loc="upper left")
    return t


def stability_and_outside() -> pd.DataFrame:
    """D5: do the segments come back (other seeds, random halves)? And the
    outside check: default per segment on training and test rows. Returns
    outside_check()."""
    r = stability_numbers()
    with _style():
        fig, axes = _new_fig((8.5, 3.4), 1, 2)
        draw_stability(axes[0])
        axes[0].set_title("do the segments come back?")
        draw_default_by_segment(axes[1])
        axes[1].set_title("the outside check: default per segment")
    _show(fig)
    _say("Other random_state: " + ", ".join(f"{v:.1%}" for v in r["other random_state"])
         + " of the 21,000 clients keep their segment.")
    _say("Fitted on a random half, then all assigned: "
         + ", ".join(f"{v:.1%}" for v in r["random halves"]) + ".")
    return outside_check()


# ---------------------------------------------------------------------------
# W - Where groups exist: 178 wines (slides 43-44, the notebook's wine demo)
# ---------------------------------------------------------------------------
# scikit-learn's bundled copy of the UCI Wine data (Forina et al., PARVUS,
# Genoa; UCI Machine Learning Repository, doi 10.24432/C5PC7J, CC BY 4.0):
# 178 wines from one region of Italy, three grape varieties, 13 lab
# measurements. No download, so it works offline and in Colab. The variety
# is never used to make or name a group: it is read only afterwards, to check
# the groups (as default is for our clients).
WINE_LABELS = {  # plain names of the 13 columns
    "alcohol": "alcohol",
    "malic_acid": "malic acid",
    "ash": "ash",
    "alcalinity_of_ash": "alkalinity of ash",
    "magnesium": "magnesium",
    "total_phenols": "total phenols",
    "flavanoids": "flavanoids",
    "nonflavanoid_phenols": "non-flavanoid phenols",
    "proanthocyanins": "proanthocyanins",
    "color_intensity": "colour intensity",
    "hue": "hue",
    "od280/od315_of_diluted_wines": "OD280/OD315",
    "proline": "proline",
}
WINE_GROUPS = ["strong", "light", "dark"]          # the order of every table and figure
WINE_COLOURS = {"strong": RED, "light": MID_BLUE, "dark": DARK}
_WINE_RAW_COLOURS = {1: BLUE, 2: LIGHT_RED, 3: _CLUSTER_CYCLE[8]}   # not standardised: by size
WINE_KS = list(range(1, 9))                       # the elbow, K = 1 to 8
WINE_PROFILE = ["alcohol", "colour intensity", "flavanoids", "proline"]
_WINE_SHUFFLES = 20                                # shuffled copies, default_rng(0) ... (19)


def wine_data():
    """The 178 wines: X (178 x 13 lab measurements, plain column names) and
    variety (1, 2 or 3, the grape variety; never used to make the groups).
    Copies."""
    def build():
        from sklearn.datasets import load_wine
        d = load_wine(as_frame=True)
        X = d.data.rename(columns=WINE_LABELS)
        return X, pd.Series(d.target.to_numpy() + 1, index=X.index, name="variety")
    X, v = _cached("wine", "the wines", build)
    return X.copy(), v.copy()


def _wine_names(raw, X: pd.DataFrame) -> np.ndarray:
    """The target-free names: "strong" = the highest median alcohol, "dark" =
    the higher median colour intensity of the other two, "light" = the last."""
    g = X.assign(g=raw).groupby("g")
    strong = g["alcohol"].median().idxmax()
    colour = g["colour intensity"].median().drop(strong)
    dark = colour.idxmax()
    light = [k for k in colour.index if k != dark][0]
    mp = {strong: "strong", dark: "dark", light: "light"}
    return np.array([mp[v] for v in raw])


def _wine_fit(standardise: bool = True) -> dict:
    """K-means (n_init=10, random_state=42) on the 13 columns, standardised
    (StandardScaler) or as they are; the elbow K = 1 to 8, the silhouette K = 2
    to 8, and the groups at K = 3 (names, or 1-3 by size without the scaler)."""
    def build():
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        from sklearn.preprocessing import StandardScaler
        X, _ = wine_data()
        M = StandardScaler().fit_transform(X) if standardise else X.to_numpy(float)
        fits = {k: KMeans(k, n_init=10, random_state=SEED).fit(M) for k in WINE_KS}
        raw = fits[3].labels_
        groups = _wine_names(raw, X) if standardise else _by_size(raw)
        return {"M": M, "raw": raw, "groups": groups,
                "inertia": pd.Series({k: f.inertia_ for k, f in fits.items()}, name="inertia"),
                "silhouette": pd.Series({k: silhouette_score(M, f.labels_)
                                         for k, f in fits.items() if k > 1}, name="silhouette")}
    return _cached(("wine_fit", bool(standardise)), "the wine groups", build)


def _wine_order(standardise: bool = True) -> list:
    return WINE_GROUPS if standardise else [1, 2, 3]


def _wine_colour(g) -> str:
    return WINE_COLOURS[g] if isinstance(g, str) else _WINE_RAW_COLOURS[g]


def _wine_pca():
    """PCA of the 13 standardised columns, signs fixed: PC1 positive on
    flavanoids, PC2 positive on colour intensity. Returns (pca, scores)."""
    def build():
        from sklearn.decomposition import PCA
        X, _ = wine_data()
        Z = _wine_fit(True)["M"]
        p = PCA().fit(Z)
        cols = list(X.columns)
        for comp, name in ((0, "flavanoids"), (1, "colour intensity")):
            if p.components_[comp, cols.index(name)] < 0:
                p.components_[comp] = -p.components_[comp]
        return p, p.transform(Z)
    return _cached("wine_pca", "the PCA of the wines", build)


def wine_crosstab(standardise: bool = True) -> pd.DataFrame:
    """Wines per k-means group (rows) and variety (columns 1-3)."""
    _, v = wine_data()
    g = _wine_fit(standardise)["groups"]
    t = pd.crosstab(pd.Series(g, name="k-means group"), v.rename("variety").to_numpy())
    t.columns.name = "variety"
    return t.reindex(_wine_order(standardise)).fillna(0).astype(int)


def wine_table(standardise: bool = True) -> pd.DataFrame:
    """Per k-means group: its wines, how many of each variety, and its typical
    (median) alcohol (%), colour intensity, flavanoids and proline."""
    X, _ = wine_data()
    g = _wine_fit(standardise)["groups"]
    ct = wine_crosstab(standardise)
    t = pd.DataFrame({"wines": ct.sum(axis=1)})
    for c in ct.columns:
        t[f"variety {c}"] = ct[c]
    med = X.assign(g=g).groupby("g")[WINE_PROFILE].median().reindex(ct.index)
    for c in WINE_PROFILE:
        t[f"{c} (median)"] = med[c]
    t.index.name = "k-means group"
    return t


def _own_variety(ct: pd.DataFrame) -> int:
    """Wines in the group whose most common variety is their own."""
    return int(ct.max(axis=1).sum())


def wine_numbers() -> dict:
    """Every number of slides 43-44 (and of the notebook demo)."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    X, v = wine_data()
    std, raw = _wine_fit(True), _wine_fit(False)
    p, S = _wine_pca()
    ct, ct_raw = wine_crosstab(True), wine_crosstab(False)

    def shuffled(seed):
        rng = np.random.default_rng(seed)
        M = std["M"]
        Ms = np.column_stack([rng.permutation(M[:, j]) for j in range(M.shape[1])])
        return silhouette_score(Ms, KMeans(3, n_init=10, random_state=SEED).fit(Ms).labels_)
    sh = _cached("wine_shuffled", "the shuffled wines",
                 lambda: np.array([shuffled(s) for s in range(_WINE_SHUFFLES)]))
    seeds = {s: agree(std["raw"], KMeans(3, n_init=10, random_state=s).fit(std["M"]).labels_)
             for s in range(1, 6)}
    # not standardised: proline alone gives the same groups (its share of the raw variance)
    proline = agree(raw["raw"], KMeans(3, n_init=10, random_state=SEED)
                    .fit(X[["proline"]].to_numpy(float)).labels_)
    var = X.var(ddof=0)
    cen = pd.DataFrame(S[:, :2], columns=["PC1", "PC2"]).groupby(std["groups"]).mean()
    own, own_raw = _own_variety(ct), _own_variety(ct_raw)
    return {
        "wines": len(X), "columns": X.shape[1],
        "variety sizes": v.value_counts().sort_index().to_dict(),
        "inertia": std["inertia"], "drops": -std["inertia"].diff().dropna(),
        "silhouette": std["silhouette"],
        "best K (silhouette)": int(std["silhouette"].idxmax()),
        "shuffled silhouette K=3": sh,
        "group sizes": ct.sum(axis=1).to_dict(),
        "crosstab": ct, "own variety": own, "own variety share": own / len(X),
        "crosstab, not standardised": ct_raw, "own variety, not standardised": own_raw,
        "own variety share, not standardised": own_raw / len(X),
        "proline alone = not standardised": proline,
        "proline share of raw variance": var["proline"] / var.sum(),
        "other random_state 1-5": seeds,
        "PVE": 100 * p.explained_variance_ratio_,
        "loadings PC1-PC2": pd.DataFrame(p.components_[:2].T, index=X.columns,
                                         columns=["PC1", "PC2"]),
        "centres PC1-PC2": cen.reindex(WINE_GROUPS),
        "profile (median)": wine_table(True)[[f"{c} (median)" for c in WINE_PROFILE]],
        "variety profile (median)": X.groupby(v.to_numpy())[WINE_PROFILE].median(),
        "column ranges": X[["alcohol", "colour intensity", "proline"]].agg(["min", "max", "std"]),
    }


def draw_wine_elbow(ax, standardise: bool = True, mark: int = 3) -> pd.Series:
    """Inertia of k-means on the wines for K = 1 to 8 (thousands); K = mark
    ringed."""
    ins = _wine_fit(standardise)["inertia"]
    scale = 1000 if standardise else 1e6
    v = ins.to_numpy() / scale
    ax.plot(WINE_KS, v, "-o", color=DARK, ms=4, lw=1.4, zorder=2)
    i = WINE_KS.index(mark)
    ax.scatter([mark], [v[i]], s=150, facecolor="none", edgecolor=RED, linewidth=1.4, zorder=3)
    ax.annotate(f"K = {mark}", (mark, v[i]), xytext=(9, 9), textcoords="offset points",
                ha="left", va="bottom", fontsize=9, color=RED)
    ax.set_xticks(WINE_KS)
    ax.set_xlim(0.5, 8.5)
    ax.set_ylim(0, 1.1 * v.max())
    ax.yaxis.set_major_locator(MaxNLocator(5, steps=[1, 2, 5, 10]))
    ax.set_xlabel("K, the number of groups")
    ax.set_ylabel("inertia (thousands)" if standardise else "inertia (millions)")
    return ins


def draw_wine_pcs(ax, standardise: bool = True, s: float = 14) -> pd.DataFrame:
    """The 178 wines at their PC1 and PC2 scores (13 standardised columns),
    coloured by k-means group; each group's name (or number) at its centre."""
    p, S = _wine_pca()
    g = _wine_fit(standardise)["groups"]
    for k in _wine_order(standardise):
        m = g == k
        ax.scatter(S[m, 0], S[m, 1], s=s, color=_wine_colour(k), linewidth=0.3,
                   edgecolor="white", alpha=0.9, zorder=2)
    cen = pd.DataFrame(S[:, :2], columns=["PC1", "PC2"]).groupby(g).mean()
    cen = cen.reindex(_wine_order(standardise))
    for k, row in cen.iterrows():
        ax.text(row["PC1"], row["PC2"], k if isinstance(k, str) else f"group {k}",
                ha="center", va="center", fontsize=9, color=INK, zorder=5,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=_wine_colour(k), lw=1.0))
    from matplotlib.ticker import MultipleLocator
    pve = 100 * p.explained_variance_ratio_
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.yaxis.set_major_locator(MultipleLocator(2))
    ax.set_xlabel(f"PC1 score ({pve[0]:.0f}% of the variance)")
    ax.set_ylabel(f"PC2 score ({pve[1]:.0f}%)")
    ax.set_xlim(S[:, 0].min() - 0.4, S[:, 0].max() + 0.4)
    ax.set_ylim(S[:, 1].min() - 0.4, S[:, 1].max() + 0.4)
    return cen


def draw_wine_check(ax, standardise: bool = True) -> pd.DataFrame:
    """The check against the variety: for each k-means group, its wines by
    variety (1, 2, 3). The group's most common variety in the group's colour, the others
    grey; the count above each bar; the group's name under its three bars."""
    ct = wine_crosstab(standardise)
    top = max(ct.to_numpy().max(), 1)
    for j, (k, row) in enumerate(ct.iterrows()):
        xs = 4 * j + np.arange(3)
        main = row.idxmax()
        cols = [_wine_colour(k) if c == main else GRAY for c in row.index]
        ax.bar(xs, row.to_numpy(), width=0.8, color=cols, zorder=2)
        for x, n in zip(xs, row.to_numpy()):
            ax.annotate(f"{n}", (x, n), xytext=(0, 2), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8.5, color=INK)
        ax.annotate(k if isinstance(k, str) else f"group {k}", (4 * j + 1, 0),
                    xycoords=ax.get_xaxis_transform(), xytext=(0, -15),
                    textcoords="offset points", ha="center", va="top", fontsize=9,
                    color=INK, annotation_clip=False)
    ax.set_xticks([4 * j + i for j in range(len(ct)) for i in range(3)])
    ax.set_xticklabels([str(c) for _ in range(len(ct)) for c in ct.columns])
    ax.tick_params(axis="x", length=0, pad=2)
    ax.set_xlim(-0.7, 4 * len(ct) - 1.3)
    ax.set_ylim(0, 1.15 * top)
    ax.set_ylabel("wines")
    return ct


def wine_figure(standardise: bool = True, figsize=None, suptitle: bool = True):
    with _style():
        fig, axes = _new_fig(figsize or (10.0, 3.4), 1, 3, width_ratios=[0.8, 1.25, 1.05])
        draw_wine_elbow(axes[0], standardise)
        axes[0].set_title("the elbow")
        draw_wine_pcs(axes[1], standardise)
        axes[1].set_title("the wines on PC1 and PC2, K = 3")
        draw_wine_check(axes[2], standardise)
        axes[2].set_title("each group's wines by variety (1, 2, 3)")
        if suptitle:
            fig.suptitle("178 wines, 13 columns " + ("standardised" if standardise else
                         "as they are (not standardised)") + ": k-means with K = 3, ten starts",
                         fontsize=9.5)
    return fig


def wine_groups(standardise: bool = True) -> pd.DataFrame:
    """W: the recipe on 178 wines with three known varieties: the elbow, the
    groups on PC1 and PC2, and the check against the variety (each group's
    wines by variety; k-means never saw the variety). standardise=False runs the same
    steps on the raw columns. Returns wine_table(standardise)."""
    if not isinstance(standardise, (bool, np.bool_)):
        _say("standardise is True or False; using True.")
        standardise = True
    _show(wine_figure(bool(standardise)))
    ct = wine_crosstab(bool(standardise))
    own = _own_variety(ct)
    _say(f"{own} of the {ct.to_numpy().sum()} wines ({own / ct.to_numpy().sum():.1%}) sit in "
         "the group whose most common variety is their own.")
    return wine_table(bool(standardise))


# ---------------------------------------------------------------------------
# The 300 random points (slide 65)
# ---------------------------------------------------------------------------
def draw_noise_kmeans(ax) -> dict:
    """300 random points with no groups, coloured by k-means with K = 3."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    U = random_points()
    lab = KMeans(3, n_init=10, random_state=SEED).fit(U).labels_
    order = sorted(range(3), key=lambda k: U[lab == k].sum(axis=1).mean())
    for k, col in zip(order, (BLUE, GRAY, RED)):
        m = lab == k
        ax.scatter(U[m, 0], U[m, 1], s=12, color=col, linewidth=0)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel("first random column")
    ax.set_ylabel("second random column")
    return {"sizes": sorted(np.bincount(lab).tolist()), "silhouette": silhouette_score(U, lab)}


# ---------------------------------------------------------------------------
# Homework A: the tables and the soft checks
# ---------------------------------------------------------------------------
HW_CLIENTS = pd.DataFrame({"august": [1, 2, 3, 7, 8, 9], "september": [2, 1, 3, 6, 8, 10]},
                          index=pd.Index([1, 2, 3, 4, 5, 6], name="client"))
HW_A3 = pd.DataFrame({"credit limit (thousand NT$)": [100, 110, 140], "months late": [0, 4, 0]},
                     index=pd.Index(["X", "Y", "Z"], name="client"))
HW_A3_SD = (40, 1.6)
HW_A5_INERTIA = [400, 180, 90, 75, 63, 55]
HW_LINE = [0, 1, 2, 3.5, 6, 7, 7.5, 10]

A9_MODEL_ANSWER = (
    "0.22 lies below the weak band (0.26-0.50) and hardly above the shuffled copy's 0.20, "
    "so the six groups are barely better than groups found in a table with no structure. "
    "They also buy concert tickets at almost the same rate (9-11%), so they do not differ "
    "on an outcome the clustering never used. Conclusion: the groups may be artificial; "
    "report 0.22 as it is and do not build on the six groups yet. Next check: stability - "
    "run k-means with other random_state values and on a random half of the listeners; if "
    "the groups do not come back, drop them.")


def homework_answers(item: str) -> dict:
    """The worked numbers of a homework item A1-A9 (for the solutions)."""
    from scipy.spatial.distance import cdist, pdist, squareform
    item = str(item).strip().upper()
    H = HW_CLIENTS.to_numpy(float)
    if item == "A1":
        Hc = H - H.mean(axis=0)
        sa, sb = Hc @ [1, 0], Hc @ [0.6, 0.8]
        total = float((Hc ** 2).mean(axis=0).sum())
        p = pca_fixed(HW_CLIENTS, "ten")
        u = p.components_[0]
        return {"means": H.mean(axis=0), "shadows_a": sa, "variance_a": float(np.mean(sa ** 2)),
                "shadows_b": sb, "variance_b": float(np.mean(sb ** 2)), "total": total,
                "share_b": float(np.mean(sb ** 2)) / total, "pc1": u,
                "pc1_angle": math.degrees(math.atan2(u[1], u[0])),
                "pc1_share": float(p.explained_variance_ratio_[0])}
    if item == "A2":
        v = np.array([2.4, 0.9, 0.5, 0.2])
        pve = 100 * v / v.sum()
        run = np.cumsum(pve)
        return {"pve": pve, "running": run, "components": int(np.argmax(run >= 80 - 1e-9) + 1)}
    if item == "A3":
        P = HW_A3.to_numpy(float)
        S = P / np.array(HW_A3_SD)
        raw = {"raw_xy": float(np.hypot(*(P[1] - P[0]))), "raw_xz": float(np.hypot(*(P[2] - P[0])))}
        std = {"std_xy": float(np.hypot(*(S[1] - S[0]))), "std_xz": float(np.hypot(*(S[2] - S[0])))}
        return {**raw, **std, "nearest_raw": "Y" if raw["raw_xy"] < raw["raw_xz"] else "Z",
                "nearest_standardised": "Y" if std["std_xy"] < std["std_xz"] else "Z"}
    if item == "A4":
        c0 = np.array([(2, 2), (5, 5)], float)
        d2 = cdist(H, c0, "sqeuclidean")
        lab = d2.argmin(axis=1)
        c1 = np.array([H[lab == k].mean(axis=0) for k in (0, 1)])
        d2b = cdist(H, c1, "sqeuclidean")
        return {"sq_distances": d2, "groups": lab + 1, "inertia_first": float(d2.min(axis=1).sum()),
                "centres": c1, "inertia_second": float(((H - c1[lab]) ** 2).sum()),
                "stops": bool((d2b.argmin(axis=1) == lab).all())}
    if item == "A5":
        v = HW_A5_INERTIA
        return {"drops": [v[i] - v[i + 1] for i in range(len(v) - 1)], "elbow": 3}
    if item == "A6":
        D = squareform(pdist(H))
        sub = D[np.ix_([0, 1], [3, 4])]
        return {"pairs": sub, "single": float(sub.min()), "complete": float(sub.max()),
                "average": float(sub.mean())}
    if item == "A7":
        from sklearn.cluster import DBSCAN
        x = np.array(HW_LINE, float)[:, None]
        db = DBSCAN(eps=1.5, min_samples=3).fit(x)
        core = np.zeros(len(x), bool)
        core[db.core_sample_indices_] = True
        kinds = np.where(core, "core", np.where(db.labels_ < 0, "noise", "border"))
        pts = [float(v) for v in HW_LINE]
        return {"inside": (np.abs(x - x.T) <= 1.5 + 1e-9).sum(axis=1), "kind": kinds,
                "core": [p for p, k in zip(pts, kinds) if k == "core"],
                "border": [p for p, k in zip(pts, kinds) if k == "border"],
                "noise": [p for p, k in zip(pts, kinds) if k == "noise"],
                "clusters": [[p for p, lb in zip(pts, db.labels_) if lb == g]
                             for g in sorted(set(db.labels_) - {-1})]}
    if item == "A8":
        D = squareform(pdist(H))
        a, b = float(D[2, [0, 1]].mean()), float(D[2, [3, 4, 5]].mean())
        return {"a": a, "b": b, "s": (b - a) / max(a, b)}
    if item == "A9":
        return {"model answer": A9_MODEL_ANSWER}
    raise ValueError(f"unknown homework item {item!r}: use A1-A9")


# (key, kind, what it is, hint when it is not right yet)
_CHECKS = {
    "A1": [("shadows_a", "numbers", "the six shadows on (1, 0)",
            "on (1, 0) a shadow is the centred August value (August minus its mean 5)"),
           ("variance_a", "number", "their variance",
            "the mean of the six squared shadows: divide by 6, as on slide 13"),
           ("shadows_b", "numbers", "the six shadows on (0.6, 0.8)",
            "0.6 x centred August + 0.8 x centred September, client by client"),
           ("variance_b", "number", "their variance",
            "the shadows have mean 0, so their variance is the mean of their squares")],
    "A2": [("pve", "percents", "the four PVEs (in %)",
            "each variance divided by the total 2.4 + 0.9 + 0.5 + 0.2 = 4.0"),
           ("components", "int", "how many components keep at least 80%",
            "add the PVEs from the largest until the running total reaches 80%")],
    "A3": [("nearest_raw", "letter", "the client nearer to X in raw units (Y or Z)",
            "compare the two raw distances: one squared difference per column"),
           ("nearest_standardised", "letter", "the client nearer to X after standardising",
            "divide each difference by its column's SD (40 and 1.6) first"),
           ("raw_xy", "number", "X-Y in raw units", "the square root of 10^2 + 4^2"),
           ("raw_xz", "number", "X-Z in raw units", "only the limit differs"),
           ("std_xy", "number", "X-Y standardised", "differences 10/40 and 4/1.6"),
           ("std_xz", "number", "X-Z standardised", "difference 40/40 in the limit only")],
    "A4": [("groups", "partition", "the group (1 or 2) of clients 1-6 after the first assign",
            "compare each client's squared distance to (2, 2) and to (5, 5)"),
           ("inertia_first", "number", "the inertia after the first assign",
            "add each client's squared distance to its own centre"),
           ("centres", "points", "the two centres after the move",
            "each centre moves to the mean of its group's clients"),
           ("inertia_second", "number", "the inertia after the move",
            "squared distances to the new centres, added up"),
           ("stops", "bool", "does the next assign change nothing (True / False)?",
            "assign again with the new centres: does any client change group?")],
    "A5": [("elbow", "int", "the K at the elbow",
            "write the drops between neighbouring K; the elbow is the K after which "
            "they become small")],
    "A6": [("single", "number", "single linkage", "the smallest of the four pair distances"),
           ("complete", "number", "complete linkage", "the largest of the four pair distances"),
           ("average", "number", "average linkage", "the mean of the four pair distances")],
    "A7": [("core", "set", "the core points",
            "count the points at distance 1.5 or less, the point itself included; core = "
            "at least 3"),
           ("border", "set", "the border points", "not core, but within 1.5 of a core point"),
           ("noise", "set", "the noise points", "neither core nor border"),
           ("clusters", "groups", "the clusters (a list of lists)",
            "join core points within 1.5 of each other, then add their border points")],
    "A8": [("a", "number", "a for client 3",
            "the mean distance from client 3 to clients 1 and 2"),
           ("b", "number", "b for client 3",
            "the mean distance from client 3 to clients 4, 5 and 6"),
           ("s", "number", "the silhouette s of client 3", "(b - a) / max(a, b)")],
}


def _close(x, y) -> bool:
    return abs(float(x) - float(y)) <= max(0.02, 0.005 * abs(float(y)))


def _right(kind: str, given, expected) -> bool:
    if kind == "number":
        return _close(given, expected)
    if kind in ("numbers", "percents"):
        g = [float(v) for v in np.ravel(given)]
        e = [float(v) for v in np.ravel(expected)]
        if kind == "percents" and g and max(abs(v) for v in g) <= 1.0001:
            g = [100 * v for v in g]
        return len(g) == len(e) and all(_close(a, b) for a, b in zip(g, e))
    if kind == "set":
        g = sorted(float(v) for v in np.ravel(given))
        e = sorted(float(v) for v in np.ravel(expected))
        return len(g) == len(e) and all(_close(a, b) for a, b in zip(g, e))
    if kind == "groups":
        g = {tuple(sorted(round(float(v), 6) for v in grp)) for grp in given}
        e = {tuple(sorted(round(float(v), 6) for v in grp)) for grp in expected}
        return g == e
    if kind == "partition":
        g, e = list(np.ravel(given)), list(np.ravel(expected))
        if len(g) != len(e):
            return False
        part = lambda lab: {frozenset(i for i, v in enumerate(lab) if v == k) for k in set(lab)}
        return part(g) == part(e)
    if kind == "points":
        g = sorted(tuple(round(float(v), 6) for v in p) for p in given)
        e = sorted(tuple(round(float(v), 6) for v in p) for p in expected)
        return len(g) == len(e) and all(_close(a, b) for pa, pb in zip(g, e)
                                        for a, b in zip(pa, pb))
    if kind == "int":
        return float(given) == float(expected)
    if kind == "letter":
        return str(given).strip().upper() == str(expected).strip().upper()
    if kind == "bool":
        words = {"yes": True, "true": True, "no": False, "false": False}
        g = words.get(str(given).strip().lower(), given) if isinstance(given, str) else given
        return bool(g) == bool(expected)
    return False


def check(item: str, **answers) -> None:
    """A soft check of a homework answer: one line per part, "right" or "not
    yet" with a hint. It never stops the notebook. check("A1") lists the
    parts it takes; leave a part as ... until you have it."""
    key = str(item).strip().upper()
    if key == "A9":
        _say("A9 is a written answer: compare yours with the model answer in the "
             "solutions (and with slides 64-69).")
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
            _say(f"{key} {k}: could not read {given!r} - it should be {what}.")
            continue
        n_right += ok
        _say(f"{key} {k}: right." if ok else f"{key} {k}: not yet - {hint}.")
    if n_given:
        _say(f"{key}: {n_right} of {n_given} right.")
