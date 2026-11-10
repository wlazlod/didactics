"""block6_tools.py - the experiments of Block 6 (neural networks and responsible ML).

Data Mining, SGH Warsaw School of Economics (223121-0090).

The Block 6 notebook imports this file as

    import block6_tools as lab

Each experiment is ONE function with ONE control (the value you change), and
it draws ONE figure. The slides are drawn by the same functions, so the deck
shows exactly the pictures you make in the notebook. You do not need to read
this file; the notebook shows the few scikit-learn lines behind each
experiment ("The code behind it").

The data, the split, the folds, the models of Blocks 2-3 and the crescents
come from Block 3's file, block3_tools.py (imported, never changed): the 12
columns of ``taiwan_features``, 70% training / 30% test, stratified,
``random_state=42`` (21,000 / 9,000 clients), and 5-fold cross-validation on
the training rows with ``StratifiedKFold(5, shuffle=True, random_state=42)``.
Everything is deterministic (seed 42). Models are chosen on the training
clients only (cross-validation, or a validation part held out from them);
the 9,000 test clients are read for models that are already fixed: the
comparison's last column, the audit (Experiments 4-5) and Demo 5.

Slow results are computed once, with a line "Computing ... (about N s, only
the first time)", and kept in memory until the kernel restarts. The cell
``lab.prepare()`` computes everything Experiments 2-5 draw (about 20 s on a
laptop, up to a minute on Colab); after it every slider redraw takes well
under a second.

Seven worlds, never mixed in one picture:
    N  clients 1-4 (made up): x1 = months late in the last 6 months, x2 = share
       of the credit limit used; outcomes 1, 2 repaid, 3, 4 defaulted (CLIENTS)
    X  the four points (made up): class 1 when exactly one of x1, x2 is 1 (POINTS)
    C  Block 3's crescents: 200 training points, 20,000 new ones (generated)
    K  one weight: the loss L(w) = (w - 3)^2, start w = 0
    E  ten epochs (made up): training loss and validation AUC (EPOCH_TOY)
    F  twenty applicants (made up), groups A and B of ten (APPLICANTS)
    R  our clients: 21,000 training / 9,000 test, 12 columns

In the notebook (each slider cell has a plain twin: set the value, re-run)
---------------------------------------------------------------------------
    X_train, X_test, y_train, y_test = lab.course_split();  lab.setup_check()
    lab.prepare()                              Experiments 2-5 computed once
    lab.neuron_table(); lab.sigmoid_table()    clients 1-4, one neuron
    Demo 1  lab.no_hidden_layer()
    lab.xor_table(); lab.xor_table(relu=False); lab.count_parameters(12, (16,))
    E1  lab.bend_boundary(1);        lab.slider(lab.bend_boundary, hidden_units=lab.E1_UNITS)
    lab.loss_table(); lab.gradient_steps(lr=0.25); lab.count_steps(21_000, 200)
    E2  lab.learning_rate(0.001);    lab.slider(lab.learning_rate, learning_rate=lab.E2_RATES)
    lab.EPOCH_TOY; lab.patience_walk(patience=3)
    E3  lab.stop_early(5);           lab.slider(lab.stop_early, stop_after=lab.E3_EPOCHS)
    Demo 2  lab.our_network()        Demo 3  lab.tournament()
    lab.toy_audit()                  the twenty applicants, steps 1-4
    E4  lab.audit_by_group("sex");   lab.slider(lab.audit_by_group, group=lab.E4_GROUPS)
    Demo 4  lab.proxy_check()
    E5  lab.remove_column("nothing"); lab.slider(lab.remove_column, removed=lab.E5_REMOVED)
    Demo 5  lab.mlp_audit()
    Homework A: lab.check("A1", z=[...], ...)   (lab.check("A1") lists what it takes)
    Homework B: lab.group_report(y_test, p_test, mask, names=(...), threshold=0.25)

Public API
----------
Reused from block3_tools (imported, never redefined)
    course_data(), folds(), model(name), crescents(), sliders_ok(), sliders_status(),
    the colour tokens BLUE RED DARK GRAY LIGHT LIGHT_RED LIGHT_BLUE MID_BLUE MUTED SHADE
    (NEAR_BLACK = Block 3's INK, #222222) and STYLE.
    course_split() -> X_train, X_test, y_train, y_test   Block 3's split, as copies (so a
                                 change you make to them never reaches the experiments).

Setup
    SEED = 42; THRESHOLD = 0.25 (Block 3's rule: decline when p > 0.25); VERBOSE = True
    (False silences the "Computing ..." lines: make_figures.py); clear_cache().
    setup_check()                loads the data; prints "Setup OK: 21,000 training and
                                 9,000 test clients, 12 columns", the version note (if any)
                                 and the sliders status.
    version_note() -> str        "" from scikit-learn 1.8 on; a calm Note for 1.4-1.7; a
                                 WARNING with the upgrade line below 1.4.
    slider(func, **controls)     ipywidgets.interact (redraws when you let go). (lo, hi,
                                 step) -> slider; a list of numbers -> slider over them,
                                 labelled with plain decimals (0.00001, not 1e-05); a list
                                 of strings -> dropdown. Without ipywidgets: a line saying
                                 to use the plain cell.
    prepare()                    Experiments 2-5 computed once, one "Computing ..." line
                                 per part with the measured seconds, then "Ready: ...".
    network()                    the deck's network, unfitted: StandardScaler + MLPClassifier
                                 (16,), early_stopping=True, max_iter=500, random_state=42.

The small worlds
    CLIENTS, W = (1, 2, -3), sigmoid(z), relu(z), neuron_table(w1=1, w2=2, b=-3),
    sigmoid_table(); POINTS, xor_table(relu=True), count_parameters(n_inputs, hidden,
    n_outputs=1); loss_table(), gradient_steps(lr=0.25, start=0.0, target=3.0, steps=3),
    count_steps(rows, batch); EPOCH_TOY, patience_walk(patience=3);
    APPLICANTS, toy_audit(threshold=0.45).

Experiments (X(control) shows the figure and returns None; X_figure(control, ...,
figsize=None, suptitle=True) -> Figure; X_numbers() -> DataFrame)
    E1 bend_boundary(hidden_units=1)      E1_UNITS = [0, 1, 2, 3, 4, 8, 16, 32, 64]
    E2 learning_rate(learning_rate=0.001) E2_RATES = [0.00001, 0.001, 0.03, 0.3, 3, 10, 20];
       keeps a trail of the rates tried (learning_rate_reset() clears it)
    E3 stop_early(stop_after=5)           E3_EPOCHS = (1, 100, 1)
    E4 audit_by_group(group="sex")        E4_GROUPS = ["sex", "age: under 25",
                                          "age: 50 and over"]; prints group_report
    E5 remove_column(removed="nothing")   E5_REMOVED = ["nothing", "female",
                                          "female, married", "female, age",
                                          "female, married, age"]
    A value that is not on the list is moved to the nearest allowed one, with a one-line
    note; a decimal comma (0,3) is read as 0.3, with a note; text values take quotes.
    gap_numbers() -> DataFrame   Experiment 4's three gaps per grouping, with their noise.

Demos (replay at home)
    no_hidden_layer() (Demo 1), our_network() (Demo 2), tournament() (Demo 3),
    proxy_check() (Demo 4), mlp_audit() (Demo 5)

Drawing functions (shared with make_figures.py; each takes ax and returns the numbers
it drew)
    draw_sigmoid(ax), draw_xor(ax), draw_hidden_space(ax), draw_boundary(ax,
    hidden_units), draw_slope(ax), draw_lr_path(ax, lr), draw_toy_epochs(ax_loss, ax_auc),
    draw_loss_curves(ax, learning_rate, tried=None), draw_epoch_curves(ax_loss, ax_auc,
    stop_after), draw_tournament(ax), draw_applicants(ax), draw_group_rates(ax, group,
    removed="nothing"), draw_error_rates(ax, group), draw_reliability(ax, name) (out-of-fold
    p of the 21,000), draw_dense_middle(ax) (Backup 15); toy_epochs_figure()

Numbers for the log (make_figures.py)
    no_hidden_layer_numbers(), our_network_numbers(), tournament_numbers(),
    calibration_numbers(name), proxy_numbers(), mlp_audit_numbers(),
    dense_middle_numbers(), applicants_numbers(), flip_numbers(removed="female") (R11),
    open_question_numbers() (the open practice question, Q)

Project helper
    group_report(y_true, p, group, names=("group 1", "group 2"), threshold=0.25)
        -> DataFrame (one row per group: n, base rate, mean p, decline rate, false-alarm
        rate, miss rate, precision, in %); below the table: the decline-rate gap and the
        base-rate gap with their 95% noise, and "decline gap minus base-rate gap".

Homework A (by hand)
    HW_MEMBERS (A1), HW_EPOCHS (A4), HW_SCHEME (A5); check(item, **answers): soft check,
    "right" / "not yet" with a hint, never raises; check(item) lists what it takes;
    homework_answers(item) -> dict (the worked numbers, for the solutions).

Target-free orders (the same on every run)
    groups: "women" before "men", "under 25" before "25 and over", "50 and over" before
    "under 50"; models: logistic regression, decision tree, random forest, GBDT, MLP
    (the comparison figure sorts them by CV AUC, ties in this order); rates ascending.

All figure text is at least 8.5 pt and in plain words (no code names).
"""
from __future__ import annotations

import math
import re
import sys
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

try:  # Colab: block3_tools.py and finance_data.py uploaded next to the notebook
    import block3_tools as b3
except ImportError:  # course folder: they live in ../lecture_3 and ../lecture_1
    _ROOT = Path(__file__).resolve().parent.parent
    for _sub in ("lecture_1", "lecture_3"):
        if str(_ROOT / _sub) not in sys.path:
            sys.path.insert(0, str(_ROOT / _sub))
    import block3_tools as b3

from block3_tools import (BLUE, DARK, GRAY, LIGHT, LIGHT_BLUE, LIGHT_RED,  # noqa: F401
                          MID_BLUE, MUTED, RED, SHADE, STYLE, course_data, crescents,
                          folds, model, sliders_ok, sliders_status)

NEAR_BLACK = b3.INK      # #222222, the text colour (Block 3 calls it INK)

SEED = 42
THRESHOLD = 0.25         # Block 3's rule: a missed default costs as much as 3 wrong declines
VERBOSE = True


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
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


def _display(obj) -> None:
    """Show a table under the figure: IPython's display in a notebook, print
    elsewhere."""
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            from IPython.display import display
            display(obj)
            return
    except Exception:
        pass
    print(repr(obj) if isinstance(obj, pd.DataFrame) else obj)


def _say(msg: str) -> None:
    """A one-line note to the student (never an error)."""
    print(msg, flush=True)


def _pct(v, d: int = 1) -> str:
    """A share (0-1) as a percent: 0.2443 -> 24.4%."""
    return "-" if v is None or not np.isfinite(v) else f"{100 * float(v):.{d}f}%"


def _pts(v, d: int = 1, sign: bool = True) -> str:
    """A difference of two shares in percentage points: -0.0429 -> -4.3."""
    x = round(100 * float(v), d) + 0.0
    return f"{x:+.{d}f}" if sign and x != 0 else f"{x:.{d}f}"


def _plain_decimal(v) -> str:
    """A number in plain decimals: 1e-05 -> 0.00001, 20.0 -> 20."""
    v = float(v)
    if v == int(v) and abs(v) < 1e15:
        return str(int(v))
    return f"{v:.10f}".rstrip("0").rstrip(".")


def _clean(v: float, d: int = 3) -> str:
    """Three decimals without a negative zero."""
    return f"{round(float(v), d) + 0.0:.{d}f}"


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------
_CACHE: dict = {}
_DEPTH = [0]   # > 0 while a "Computing ..." line is open: inner parts stay quiet


def _cached(key, about_s: float, what: str, fn):
    """fn() once per session; the first call prints how long it will take and
    then how long it took."""
    if key in _CACHE:
        return _CACHE[key]
    loud = VERBOSE and about_s >= 1 and _DEPTH[0] == 0
    if loud:
        print(f"Computing {what} (about {about_s:.0f} s, only the first time) ...", end="",
              flush=True)
    t0 = time.perf_counter()
    _DEPTH[0] += 1
    try:
        _CACHE[key] = fn()
    finally:
        _DEPTH[0] -= 1
    if loud:
        print(_took(t0), flush=True)
    return _CACHE[key]


def _took(t0: float) -> str:
    """The end of a "Computing ..." line: the seconds it took ("under a
    second" rather than "0 s")."""
    took = time.perf_counter() - t0
    return f" done in {took:.0f} s." if took >= 0.5 else " done in under a second."


def clear_cache() -> None:
    """Forget every cached result and the trail of Experiment 2."""
    _CACHE.clear()
    _E2_TRAIL.clear()


# Seconds on a quiet laptop (Colab: about 2-3 times as long), measured 04.11.2026.
_ABOUT = {"lr": 5, "epochs": 15, "gbdt": 2, "refit": 2, "refits": 6,
          "tournament": 10, "no hidden": 2, "proxy": 5, "network": 0}


# ---------------------------------------------------------------------------
# Versions, setup, sliders
# ---------------------------------------------------------------------------
def _version(text: str) -> tuple:
    found = re.match(r"(\d+)\.(\d+)", str(text))
    return tuple(int(n) for n in found.groups()) if found else (0, 0)


def version_note() -> str:
    """One line about the scikit-learn version, for the setup cell ("" if
    nothing needs saying). The notebook was tested with scikit-learn 1.9."""
    import sklearn
    sk = _version(sklearn.__version__)
    if sk >= (1, 8):
        return ""
    if sk >= (1, 4):
        return (f"Note: scikit-learn {sklearn.__version__} is older than 1.8, so a few "
                "numbers may differ from the slides in the third decimal - that is fine, "
                "nothing to do.")
    return (f"WARNING: scikit-learn {sklearn.__version__} is older than 1.4, the oldest "
            "version this notebook works with.\nFix: run  %pip install -q -U scikit-learn  "
            "in a new cell, then restart\n(JupyterLab: Kernel -> Restart Kernel; Colab: "
            "Runtime -> Restart session)\nand run the setup cell again.")


def course_split():
    """X_train, X_test, y_train, y_test of the course split (Block 3's
    course_split, the same as Block 2's), as copies."""
    X_tr, X_te, y_tr, y_te = b3.course_split()
    return X_tr.copy(), X_te.copy(), y_tr.copy(), y_te.copy()


def _split():
    """The cached split itself (read only, never changed here)."""
    return b3.course_split()


def setup_check() -> None:
    """The setup cell's lines: loads the data, then prints "Setup OK: ...",
    the version note if there is one, and the sliders status."""
    X_tr, X_te, _, _ = _split()
    note = version_note()
    if note.startswith("WARNING"):
        print(note)
        print("Setup OK - but see the WARNING above")
    else:
        print(f"Setup OK: {len(X_tr):,} training and {len(X_te):,} test clients, "
              f"{X_tr.shape[1]} columns")
        if note:
            print(note)
    if sliders_ok():
        print(sliders_status())
    else:
        print("No sliders here (ipywidgets is not available) - that is fine: above each "
              "slider cell there is a plain cell; change the value in it and re-run it.")


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
        options = [(_plain_decimal(v), v) for v in spec]
        values = [v for _, v in options]
        return w.SelectionSlider(options=options, value=default if default in values
                                 else values[0], **common)
    return w.fixed(spec)


def slider(func, **controls) -> None:
    """Draw func's figure with a slider (or dropdown) for each control.

    Example: lab.slider(lab.learning_rate, learning_rate=lab.E2_RATES)
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
# Reading a control: decimal comma, nearest allowed value, quotes
# ---------------------------------------------------------------------------
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
    """value as a number if it is one (or a string that reads as one); anything
    else unchanged."""
    value = _plain(value)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, str):
        try:
            return float(value.strip().rstrip("%").replace(" ", ""))
        except ValueError:
            return value
    return value


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and np.isfinite(v)


def _key(v) -> str:
    return "".join(ch for ch in str(v).lower() if ch.isalnum())


def _snap(value, allowed: list, what: str, default, log: bool = False, quiet: bool = False):
    """The allowed value nearest to `value` (never an error): below the
    smallest -> the smallest, above the largest -> the largest, not a number
    -> `default`. log=True measures "nearest" on a log scale (learning
    rates). Prints one line when the value had to change."""
    value = _decimal_comma(_plain(value), what)
    v = _as_number(value)
    if _is_number(v):
        for a in allowed:
            if math.isclose(float(v), a, rel_tol=1e-9, abs_tol=1e-12):
                return a
        if log:
            lv = math.log10(v) if v > 0 else -99.0
            new = min(allowed, key=lambda a: (abs(math.log10(a) - lv), a))
        else:
            new = min(allowed, key=lambda a: (abs(a - v), a))
    else:
        new = default
    if quiet:
        return new
    listed = ", ".join(_plain_decimal(a) for a in allowed)
    shown = (_plain_decimal(value) if _is_number(value) else repr(value))
    _say(f"{what} {shown} is not on this experiment's list ({listed}): showing "
         f"{what} = {_plain_decimal(new)}.")
    return new


def _whole(value, lo: int, hi: int, default: int, what: str, quiet: bool = False) -> int:
    """value as a whole number from lo to hi; anything else -> the nearest
    allowed one (or default), with a note."""
    value = _decimal_comma(_plain(value), what)
    v = _as_number(value)
    if _is_number(v) and float(v).is_integer() and lo <= v <= hi:
        return int(v)
    new = int(min(max(round(v), lo), hi)) if _is_number(v) else default
    if not quiet:
        shown = f"{value:g}" if _is_number(value) else repr(value)
        _say(f"{what} {shown} is not a whole number from {lo} to {hi}: showing "
             f"{what} = {new}.")
    return new


def _choice(value, choices: list, what: str, aliases: dict | None = None,
            quiet: bool = False) -> str:
    """One of `choices` for what was typed (case, spaces and signs do not
    matter); anything else -> the first choice, with a note."""
    value = _plain(value)
    if isinstance(value, str) and value in choices:
        return value
    for c in choices:
        if _key(c) == _key(value):
            return c
    for k, c in (aliases or {}).items():
        if _key(value) == _key(k):
            return c
    if not quiet:
        quotes = ("" if isinstance(value, str) else
                  f' Text values take quotes: {what} = "{choices[0]}".')
        _say(f"{what} {value!r} is not one of " + ", ".join(f'"{c}"' for c in choices)
             + f': showing "{choices[0]}".' + quotes)
    return choices[0]


def _share(value, default: float, what: str) -> float:
    """value as a share from 0 to 1 (a percent such as 45 or "45%" is read as
    0.45), with a note when it had to change."""
    value = _decimal_comma(_plain(value), what)
    v = _as_number(value)
    if not _is_number(v):
        _say(f"{what} {value!r} is not a number: showing {what} = {default:g}.")
        return default
    v = float(v)
    if 1 < v <= 100:
        _say(f"{what} {v:g} read as {v:g}% = {v / 100:g}.")
        v = v / 100
    if not 0 <= v <= 1:
        new = min(max(v, 0.0), 1.0)
        _say(f"{what} {v:g} is outside 0-1: showing {what} = {new:g}.")
        v = new
    return v


def _positive(value, default: float, what: str) -> float:
    value = _decimal_comma(_plain(value), what)
    v = _as_number(value)
    if _is_number(v) and v > 0:
        return float(v)
    _say(f"{what} {value!r} is not a positive number: showing {what} = {default:g}.")
    return default


# ===========================================================================
# The small worlds
# ===========================================================================
def sigmoid(z):
    """1 / (1 + e^-z): any number z becomes a p between 0 and 1."""
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


def relu(z):
    """max(0, z): keep z if it is positive, else 0."""
    return np.maximum(0.0, np.asarray(z, dtype=float))


# --- N: clients 1-4 ---------------------------------------------------------
CLIENTS = pd.DataFrame(
    {"x1": [0, 1, 1, 3], "x2": [0.5, 0.5, 1.0, 1.0],
     "outcome": ["repaid", "repaid", "defaulted", "defaulted"]},
    index=pd.Index([1, 2, 3, 4], name="client"))
"""Made up: x1 = months late in the last 6 months, x2 = share of the limit used."""
W = (1, 2, -3)          # w1, w2, b of slides 7-9 (chosen by hand)
_CL = CLIENTS.copy()    # private copy: editing lab.CLIENTS changes nothing


def neuron_table(w1=1, w2=2, b=-3) -> pd.DataFrame:
    """One neuron on clients 1-4: z = w1 x1 + w2 x2 + b, then p = sigmoid(z)."""
    w1 = float(_decimal_comma(_plain(w1), "w1"))
    w2 = float(_decimal_comma(_plain(w2), "w2"))
    b = float(_decimal_comma(_plain(b), "b"))
    t = _CL[["x1", "x2"]].copy()
    t["w1·x1"] = w1 * t["x1"]
    t["w2·x2"] = w2 * t["x2"]
    t["b"] = b
    t["z"] = t["w1·x1"] + t["w2·x2"] + b
    t["p"] = sigmoid(t["z"]).round(3)
    return t


def sigmoid_table() -> pd.DataFrame:
    """The sigmoid for z = -3 ... 3 (three decimals)."""
    z = np.arange(-3, 4)
    return pd.DataFrame({"z": z, "p": sigmoid(z).round(3)}).set_index("z")


# --- X: the four points ------------------------------------------------------
POINTS = pd.DataFrame({"x1": [0, 1, 0, 1], "x2": [0, 0, 1, 1], "class": [0, 1, 1, 0]},
                      index=pd.Index(["(0,0)", "(1,0)", "(0,1)", "(1,1)"], name="point"))
"""Made up: class 1 when exactly one of x1, x2 is 1."""
_PT = POINTS.copy()


def xor_table(relu: bool = True) -> pd.DataFrame:  # noqa: A002 - the name says what it does
    """The hand network on the four points: h1 = ReLU(x1 - x2), h2 = ReLU(x2 -
    x1), z = 4 h1 + 4 h2 - 2, p = sigmoid(z), class 1 if p > 0.5. With
    relu=False the hidden units pass their sums on unchanged, and every z is -2."""
    t = _PT[["x1", "x2"]].copy()
    t["x1 - x2"] = t["x1"] - t["x2"]
    t["x2 - x1"] = t["x2"] - t["x1"]
    if relu:
        t["h1"] = np.maximum(0, t["x1 - x2"])
        t["h2"] = np.maximum(0, t["x2 - x1"])
    else:
        t["h1"] = t["x1 - x2"]
        t["h2"] = t["x2 - x1"]
    t = t[["x1", "x2", "x1 - x2", "h1", "x2 - x1", "h2"]]
    t["z"] = 4 * t["h1"] + 4 * t["h2"] - 2
    t["p"] = sigmoid(t["z"]).round(3)
    t["class"] = (t["p"] > 0.5).astype(int)
    t["right class"] = _PT["class"]
    return t


def _layers(hidden) -> list:
    hidden = _plain(hidden)
    if hidden is None:
        return []
    if isinstance(hidden, (int, float)) and not isinstance(hidden, bool):
        return [int(hidden)] if hidden else []
    return [int(_plain(h)) for h in hidden]


def count_parameters(n_inputs, hidden, n_outputs=1) -> int:
    """Weights and biases of a network: each unit has one weight per input
    coming in, and one bias. Prints the sum layer by layer; returns the total.
    count_parameters(12, (16,)) -> 225."""
    sizes = [int(_plain(n_inputs)), *_layers(hidden), int(_plain(n_outputs))]
    total = 0
    parts = []
    for k, (a, b) in enumerate(zip(sizes[:-1], sizes[1:]), 1):
        n = a * b + b
        total += n
        name = "output" if k == len(sizes) - 1 else f"hidden layer {k}"
        parts.append(f"{name}: {a:,} × {b:,} + {b:,} = {n:,}")
    print("; ".join(parts) + f"; total {total:,}")
    return total


# --- N again: the loss -------------------------------------------------------
def loss_table() -> pd.DataFrame:
    """Block 2's log-loss of clients 1-4 (weights 1, 2, -3; made-up outcomes):
    -ln p for a defaulter, -ln(1 - p) for a repayer; the last row is the mean."""
    p = sigmoid(W[0] * _CL["x1"] + W[1] * _CL["x2"] + W[2])
    y = (_CL["outcome"] == "defaulted").astype(int)
    loss = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    t = pd.DataFrame({"p": [f"{v:.3f}" for v in p], "outcome": _CL["outcome"],
                      "loss": [f"-ln({_clean(v)}) = {_clean(l, 2)}" if d else
                               f"-ln(1 - {_clean(v)}) = {_clean(l, 2)}"
                               for v, l, d in zip(p, loss, y)],
                      "loss (number)": loss.round(3)})
    t.index = t.index.astype(str)
    t.loc["mean"] = ["", "", "", round(float(loss.mean()), 3)]
    return t


def gradient_steps(lr=0.25, start=0.0, target=3.0, steps=3) -> None:
    """Gradient descent on one weight: L(w) = (w - target)^2, slope
    2(w - target), new w = w - lr × slope. Prints each step and its loss."""
    lr = _positive(lr, 0.25, "lr")
    start = float(_as_number(_decimal_comma(_plain(start), "start")))
    target = float(_as_number(_decimal_comma(_plain(target), "target")))
    steps = _whole(steps, 1, 50, 3, "steps")
    w = start
    print(f"L(w) = (w - {target:g})², slope 2(w - {target:g}); learning rate {lr:g}; "
          f"start w = {w:g}, loss {(w - target) ** 2:.4g}")
    for k in range(1, steps + 1):
        slope = 2 * (w - target)
        new = w - lr * slope
        print(f"step {k}: slope {slope:.4g}; w = {w:.4g} - {lr:g} × ({slope:.4g}) = "
              f"{new:.4g}; loss {(new - target) ** 2:.4g}")
        w = new
    f = 1 - 2 * lr
    if math.isclose(f, 0.0, abs_tol=1e-12):
        how = "lands on the lowest point in one step"
    elif f < 0:
        if math.isclose(f, -1.0, abs_tol=1e-12):
            how = "jumps over the lowest point and back; the distance stays the same"
        elif f < -1:
            how = "jumps over the lowest point; the distance grows"
        else:
            how = "jumps over the lowest point and closes in"
    elif math.isclose(f, 1.0, abs_tol=1e-12) or f > 1:
        how = "does not move"
    else:
        how = "closes in slowly" if f >= 0.8 else "closes in"
    print(f"Each step multiplies the distance to {target:g} by 1 - 2 × {lr:g} = {f:.4g}: {how}.")


def count_steps(rows, batch) -> int:
    """Steps (mini-batches) in one epoch: rows / batch, rounded up."""
    rows = _whole(rows, 1, 10 ** 9, 21_000, "rows")
    batch = _whole(batch, 1, 10 ** 9, 200, "batch")
    n = math.ceil(rows / batch)
    print(f"{rows:,} rows in batches of {batch:,}: {n:,} steps make one epoch.")
    return n


# --- E: ten epochs -------------------------------------------------------------
EPOCH_TOY = pd.DataFrame(
    {"training loss": [0.60, 0.50, 0.45, 0.42, 0.40, 0.38, 0.36, 0.34, 0.32, 0.30],
     "validation AUC": [0.700, 0.740, 0.760, 0.772, 0.778, 0.775, 0.768, 0.760, 0.751,
                        0.742]},
    index=pd.Index(range(1, 11), name="epoch"))
"""Ten made-up epochs (the shape of the real run of Experiment 3)."""
_ET = EPOCH_TOY.copy()


def _patience(scores, patience: int):
    """Early stopping by hand: (rows, stop epoch or None, best epoch)."""
    best, best_e, waiting, rows = -np.inf, 0, 0, []
    for e, s in enumerate(scores, 1):
        if s > best:
            best, best_e, waiting = s, e, 0
        else:
            waiting += 1
        rows.append((e, s, best, best_e, waiting))
        if waiting >= patience:
            return rows, e, best_e
    return rows, None, best_e


def patience_walk(patience=3) -> None:
    """Early stopping by hand on the ten made-up epochs: after each epoch the
    validation AUC, the best so far and the epochs in a row without
    improvement; training stops once that count reaches the patience."""
    patience = _whole(patience, 1, 10, 3, "patience")
    rows, stop, best_e = _patience(_ET["validation AUC"].tolist(), patience)
    for e, s, best, be, waiting in rows:
        print(f"epoch {e:2d}: validation AUC {s:.3f}; best so far {best:.3f} (epoch {be}); "
              f"epochs without improvement: {waiting}")
    best = _ET["validation AUC"][best_e]
    if stop is None:
        print(f"With a patience of {patience} it does not stop within the 10 epochs; it keeps "
              f"epoch {best_e}'s weights (validation AUC {best:.3f}).")
    else:
        print(f"With a patience of {patience}: stops after epoch {stop}, keeps epoch "
              f"{best_e}'s weights (validation AUC {best:.3f}).")


# --- F: twenty applicants -------------------------------------------------------
def _applicants() -> pd.DataFrame:
    rows = []
    cells = {"A": [(0.25, 2, 6), (0.50, 1, 1)], "B": [(0.40, 2, 3), (0.80, 4, 1)]}
    for g, parts in cells.items():
        for score, bad, good in parts:
            rows += [(g, score, 1)] * bad + [(g, score, 0)] * good
    t = pd.DataFrame(rows, columns=["group", "score", "defaulted"])
    t.index = pd.Index(range(1, 21), name="applicant")
    return t


APPLICANTS = _applicants()
"""Twenty made-up applicants, groups A and B of ten: the model's score
(it does not use the group) and whether the applicant defaulted (1) or repaid (0)."""
_AP = APPLICANTS.copy()


def _toy_rows(threshold: float) -> pd.DataFrame:
    rows = []
    for g in ("A", "B"):
        t = _AP[_AP["group"] == g]
        dec = t["score"] > threshold
        y = t["defaulted"] == 1
        TP, FP = int((dec & y).sum()), int((dec & ~y).sum())
        FN, TN = int((~dec & y).sum()), int((~dec & ~y).sum())
        rows.append({"group": g, "applicants": len(t), "declined": int(dec.sum()),
                     "decline rate": dec.mean(), "defaulted": int(y.sum()),
                     "base rate": y.mean(), "TP": TP, "FP": FP, "FN": FN, "TN": TN,
                     "false-alarm rate": FP / (FP + TN) if FP + TN else np.nan,
                     "miss rate": FN / (TP + FN) if TP + FN else np.nan,
                     "precision": TP / (TP + FP) if TP + FP else np.nan,
                     "mean score": t["score"].mean()})
    return pd.DataFrame(rows).set_index("group")


def applicants_numbers(threshold: float = 0.45) -> pd.DataFrame:
    """The audit of the twenty applicants (rule: decline when the score is
    above the threshold), one row per group."""
    return _toy_rows(threshold)


def toy_audit(threshold=0.45) -> None:
    """The four steps of an audit by group on the twenty applicants, in the
    order of the slides: 1 decide, 2 compare the gaps, 3 errors, 4 check."""
    t = _share(threshold, 0.45, "threshold")
    rows = _toy_rows(t).to_dict("index")
    a, b = rows["A"], rows["B"]
    print(f"Rule: decline when the score is above {t:g} (made up).\n")
    print("Step 1 - decline rates and base rates")
    for g, x in rows.items():
        print(f"  group {g}: {x['applicants']} applicants; declined {x['declined']} = "
              f"{_pct(x['decline rate'], 0)}; defaulted {x['defaulted']} = "
              f"{_pct(x['base rate'], 0)} (base rate)")
    dg, bg = b["decline rate"] - a["decline rate"], b["base rate"] - a["base rate"]
    print("\nStep 2 - compare the gaps (B minus A, in percentage points)")
    print(f"  decline-rate gap {_pts(dg, 0)}; base-rate gap {_pts(bg, 0)}; decline gap minus "
          f"base-rate gap {_pts(dg - bg, 0)}")
    if abs(dg - bg) < 1e-9:
        print("  The rule reproduces the base-rate difference and does not enlarge it.")
    elif dg - bg > 0:
        print("  The rule enlarges the difference: look closer.")
    else:
        print("  The decline gap is smaller than the base-rate gap.")
    print("\nStep 3 - the two error rates (rows: what happened; columns: what the rule did)")
    for g, x in rows.items():
        print(f"  group {g}:            declined  approved")
        print(f"            defaulted  {x['TP']:8d}  {x['FN']:8d}")
        print(f"            repaid     {x['FP']:8d}  {x['TN']:8d}")
    for g, x in rows.items():
        print(f"  group {g}: false-alarm rate {x['FP']} of {x['FP'] + x['TN']} = "
              f"{_pct(x['false-alarm rate'], 0)}; miss rate {x['FN']} of {x['TP'] + x['FN']} "
              f"= {_pct(x['miss rate'], 0)}")
    print("\nStep 4 - right declines, honest scores")
    for g, x in rows.items():
        prec = ("no declines in this group" if not x["declined"] else
                f"{x['TP']} of {x['declined']} = {_pct(x['precision'], 0)}")
        print(f"  group {g}: precision {prec}; mean score {x['mean score']:.2f}; base rate "
              f"{_pct(x['base rate'], 0)}")


# ===========================================================================
# E1 - Bend the boundary (Block 3's crescents)
# ===========================================================================
E1_UNITS = [0, 1, 2, 3, 4, 8, 16, 32, 64]


def _units(value, quiet: bool = False) -> int:
    value = _plain(value)
    if isinstance(value, (tuple, list)) and len(value) != 2:
        first = value[0] if len(value) else 0
        if not quiet:
            _say(f"Experiment 1 uses one hidden layer: showing hidden_units = {first}.")
        value = first
    return _snap(value, E1_UNITS, "hidden_units", 1, quiet=quiet)


def _units_name(h: int) -> str:
    if h == 0:
        return "one neuron (no hidden layer)"
    return f"{h} hidden unit" + ("s" if h > 1 else "")


def _e1_net(h: int):
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(),
                         MLPClassifier(hidden_layer_sizes=(h,) if h else (), solver="lbfgs",
                                       max_iter=5000, random_state=SEED))


def _crescent_window():
    X, *_ = crescents()
    lo, hi = X.min(0) - 0.25, X.max(0) + 0.25
    return [lo[0], hi[0], lo[1], hi[1]]


def _e1(h: int) -> dict:
    def build():
        X, y, Xn, yn = crescents()
        net = _e1_net(h).fit(X, y)
        x0, x1, y0, y1 = _crescent_window()
        gx, gy = np.meshgrid(np.linspace(x0, x1, 200), np.linspace(y0, y1, 200))
        grid = net.predict_proba(np.c_[gx.ravel(), gy.ravel()])[:, 1].reshape(gx.shape)
        return {"net": net, "training": net.score(X, y), "new points": net.score(Xn, yn),
                "gx": gx, "gy": gy, "grid": grid}
    return _cached(("e1", h), 0, "the network", build)


def bend_boundary_numbers() -> pd.DataFrame:
    """Accuracy (share right) on the 200 training points and on the 20,000 new
    points, for every number of hidden units of Experiment 1."""
    rows = [{"hidden units": h, "parameters": 3 if h == 0 else 4 * h + 1,
             "training points": _e1(h)["training"], "new points": _e1(h)["new points"]}
            for h in E1_UNITS]
    return pd.DataFrame(rows).set_index("hidden units")


def draw_boundary(ax, hidden_units=1, title: bool = True) -> dict:
    """Block 3's crescents with a network's answer: light red where p > 0.5
    (the red class), light blue elsewhere, the boundary as a dark line; the
    title gives the accuracy on the 200 training points and on 20,000 new
    points."""
    h = _units(hidden_units, quiet=True)
    r = _e1(h)
    X, y, _, _ = crescents()
    x0, x1, y0, y1 = _crescent_window()
    ax.contourf(r["gx"], r["gy"], r["grid"], levels=[0, 0.5, 1], colors=[LIGHT_BLUE, LIGHT_RED],
                zorder=0)
    ax.contour(r["gx"], r["gy"], r["grid"], levels=[0.5], colors=[DARK], linewidths=1.3,
               zorder=2)
    for cls, col in ((0, BLUE), (1, RED)):
        m = y == cls
        ax.scatter(X[m, 0], X[m, 1], s=13, color=col, edgecolor="white", linewidth=0.4,
                   zorder=3)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("feature 1")
    ax.set_ylabel("feature 2")
    if title:
        ax.set_title(f"{_units_name(h)}\ntraining {r['training']:.3f}, new points "
                     f"{r['new points']:.3f}")
    return {"hidden units": h, "training": r["training"], "new points": r["new points"]}


def bend_boundary_figure(hidden_units=1, figsize=None, suptitle: bool = True):
    h = _units(hidden_units, quiet=True)
    with _style():
        fig, ax = _new_fig(figsize or (4.8, 4.3))
        draw_boundary(ax, h)
        if suptitle:
            fig.suptitle("Block 3's crescents: 200 training points; shaded red where p > 0.5",
                         fontsize=9.5)
    return fig


def bend_boundary(hidden_units=1) -> None:
    """E1: a network with one hidden layer of this many units (0 = one neuron,
    no hidden layer) on Block 3's crescents. Another value is moved to the
    nearest one on the list (0, 1, 2, 3, 4, 8, 16, 32, 64), with a note."""
    _show(bend_boundary_figure(_units(hidden_units)))


# ===========================================================================
# E2 - Set the learning rate (our 21,000 training clients)
# ===========================================================================
E2_RATES = [0.00001, 0.001, 0.03, 0.3, 3, 10, 20]
E2_EPOCHS = 50
_E2_TRAIL: list = []


def _rate(value, quiet: bool = False):
    return _snap(value, E2_RATES, "learning_rate", 0.001, log=True, quiet=quiet)


def _lr_runs() -> dict:
    """The (16,) network trained with plain mini-batch steps (sgd, momentum 0)
    for 50 epochs at each of the seven rates, from the same starting weights:
    the training loss after each epoch, the dead hidden units and the distinct
    p on the 21,000 training clients."""
    def build():
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.neural_network import MLPClassifier
        from sklearn.preprocessing import StandardScaler
        X_tr, _, y_tr, _ = _split()
        Xs = StandardScaler().fit_transform(X_tr)
        ya = y_tr.to_numpy()
        out = {}
        for lr in E2_RATES:
            net = MLPClassifier(hidden_layer_sizes=(16,), solver="sgd", momentum=0.0,
                                learning_rate_init=lr, alpha=1e-4, random_state=SEED)
            losses = []
            with warnings.catch_warnings(), np.errstate(over="ignore", invalid="ignore"):
                warnings.simplefilter("ignore", RuntimeWarning)
                warnings.simplefilter("ignore", ConvergenceWarning)
                for _ in range(E2_EPOCHS):
                    net.partial_fit(Xs, ya, classes=[0, 1])
                    losses.append(float(net.loss_))
                a = np.maximum(0, Xs @ net.coefs_[0] + net.intercepts_[0])
                p = net.predict_proba(Xs)[:, 1]
            out[lr] = {"loss": np.array(losses), "dead": int((a.max(axis=0) == 0).sum()),
                       "distinct p": int(np.unique(np.round(p, 6)).size)}
        return out
    return _cached("lr", _ABOUT["lr"], "the seven learning rates of Experiment 2", build)


def learning_rate_numbers() -> pd.DataFrame:
    """Training loss after epochs 1, 5, 20 and 50, dead hidden units (of 16)
    and distinct p on the training clients, for each learning rate."""
    runs = _lr_runs()
    rows = []
    for lr in E2_RATES:
        r = runs[lr]
        rows.append({"learning rate": _plain_decimal(lr), "epoch 1": r["loss"][0],
                     "epoch 5": r["loss"][4], "epoch 20": r["loss"][19],
                     "epoch 50": r["loss"][-1], "dead units (of 16)": r["dead"],
                     "distinct p (training)": r["distinct p"]})
    return pd.DataFrame(rows).set_index("learning rate")


_LOSS_LIMITS = (0.40, 0.65)


def _sci(v: float) -> str:
    """A huge number in mathtext: 1.7 × 10^58."""
    e = int(math.floor(math.log10(abs(v))))
    return f"${v / 10 ** e:.1f} \\times 10^{{{e}}}$"


def _place_labels(ax, items, min_gap_pt: float = 10.5) -> None:
    """Labels at the right end of curves (x, y, text, colour), moved apart
    vertically so that no two are closer than min_gap_pt."""
    if not items:
        return
    fig = ax.figure
    fig.canvas.draw()
    scale = 72 / fig.dpi
    pts = [(ax.transData.transform((x, y))[1] * scale, x, y, t, c) for x, y, t, c in items]
    pts.sort(key=lambda q: q[0])
    placed = []
    for py, *_ in pts:
        target = py
        if placed and target < placed[-1] + min_gap_pt:
            target = placed[-1] + min_gap_pt
        placed.append(target)
    # pull the stack back down if it ran over the top of the axes
    top = ax.transAxes.transform((0, 1))[1] * scale
    over = placed[-1] - top
    if over > 0:
        placed = [v - over for v in placed]
        for i in range(len(placed) - 2, -1, -1):
            placed[i] = min(placed[i], placed[i + 1] - min_gap_pt)
    for (py, x, y, t, c), target in zip(pts, placed):
        ax.annotate(t, (x, y), xytext=(5, target - py), textcoords="offset points",
                    ha="left", va="center", fontsize=8.5, color=c, annotation_clip=False)


def draw_loss_curves(ax, learning_rate=0.001, tried=None) -> dict:
    """The training loss after each of 50 epochs: the current rate dark, the
    rates tried before grey, each labelled at its right end; y from 0.40 to
    0.65 (a curve above it is shown as an arrow, off the chart); a dotted line
    at the loss of giving every client the same p (22%)."""
    lr = _rate(learning_rate, quiet=True)
    runs = _lr_runs()
    rates = sorted({_rate(r, quiet=True) for r in (tried or [])} | {lr})
    lo, hi = _LOSS_LIMITS
    epochs = np.arange(1, E2_EPOCHS + 1)
    _, _, y_tr, _ = _split()
    base = y_tr.mean()
    const = float(-(base * np.log(base) + (1 - base) * np.log(1 - base)))
    ax.axhline(const, color=MUTED, lw=0.9, ls=":", zorder=1)
    ax.text(26, const + 0.002, f"same p for every client: {const:.3f}", fontsize=8.5,
            color=MUTED, va="bottom", ha="center")
    labels, off = [], []
    for r in rates:
        loss = runs[r]["loss"]
        current = r == lr
        colour = DARK if current else GRAY
        if np.nanmin(loss) > hi:          # off the chart: an arrow at the top
            off.append((r, current))
            continue
        ax.plot(epochs, np.clip(loss, None, hi + 0.5), color=colour,
                lw=1.8 if current else 1.1, zorder=3 if current else 2)
        labels.append((E2_EPOCHS, float(loss[-1]), _plain_decimal(r),
                       DARK if current else MUTED))
    for k, (r, current) in enumerate(off):
        x = 6 + 12 * k
        ax.annotate("", (x, hi), xytext=(x, hi - 0.045),
                    arrowprops=dict(arrowstyle="-|>", color=DARK if current else GRAY,
                                    lw=1.4), annotation_clip=False)
        ax.text(x + 1.2, hi - 0.03, f"{_plain_decimal(r)}: off the chart, the weights blew "
                f"up\n(their penalty alone: about {_sci(runs[r]['loss'][-1])})", fontsize=8.5,
                va="center", ha="left",
                color=DARK if current else MUTED, linespacing=1.1)
    ax.set_xlim(0.5, E2_EPOCHS + 0.5)
    ax.set_ylim(lo, hi)
    ax.set_xticks([1, 10, 20, 30, 40, 50])
    ax.set_xlabel("epoch")
    ax.set_ylabel("training loss\n(lower is better)", linespacing=1.1)
    r = runs[lr]
    ax.set_title(f"learning rate {_plain_decimal(lr)}: {r['dead']} of 16 hidden units dead")
    _place_labels(ax, labels)
    return {"learning rate": lr, "loss after 50 epochs": float(r["loss"][-1]),
            "dead units": r["dead"], "distinct p": r["distinct p"],
            "rates drawn": rates}


def learning_rate_figure(learning_rate=0.001, tried=None, figsize=None, suptitle: bool = True):
    lr = _rate(learning_rate, quiet=True)
    with _style():
        fig, ax = _new_fig(figsize or (7.2, 3.6))
        r = draw_loss_curves(ax, lr, tried)
        if suptitle:
            loss = r["loss after 50 epochs"]
            if loss > 100:
                fig.suptitle(f"Our 21,000 training clients, 16 hidden units: the weights blew "
                             f"up (loss {_sci(loss)}, the weight penalty)", fontsize=9.5)
            else:
                fig.suptitle(f"Our 21,000 training clients, 16 hidden units: training loss "
                             f"{loss:.3f} after 50 epochs", fontsize=9.5)
    return fig


def learning_rate(learning_rate=0.001) -> None:  # noqa: A002
    """E2: the training loss over 50 epochs at this learning rate (0.00001,
    0.001, 0.03, 0.3, 3, 10 or 20); the rates tried before stay in grey
    (learning_rate_reset() clears them)."""
    lr = _rate(learning_rate)
    if lr not in _E2_TRAIL:
        _E2_TRAIL.append(lr)
    _show(learning_rate_figure(lr, list(_E2_TRAIL)))


def learning_rate_reset() -> None:
    """Forget the rates tried in Experiment 2."""
    _E2_TRAIL.clear()


# ===========================================================================
# E3 - Stop early (16,800 fit / 4,200 validation of the training clients)
# ===========================================================================
E3_EPOCHS = (1, 100, 1)
E3_MAX = 100


def _epoch_runs() -> pd.DataFrame:
    """The wide network (two hidden layers of 256, alpha 0, adam) trained one
    epoch at a time for 100 epochs on 16,800 training clients: the training
    loss and the AUC on the 4,200 held out for validation."""
    def build():
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import train_test_split
        from sklearn.neural_network import MLPClassifier
        from sklearn.preprocessing import StandardScaler
        X_tr, _, y_tr, _ = _split()
        X_fit, X_val, y_fit, y_val = train_test_split(X_tr, y_tr, test_size=0.2,
                                                      stratify=y_tr, random_state=SEED)
        sc = StandardScaler().fit(X_fit)
        Xf, Xv = sc.transform(X_fit), sc.transform(X_val)
        yf, yv = y_fit.to_numpy(), y_val.to_numpy()
        net = MLPClassifier(hidden_layer_sizes=(256, 256), alpha=0.0, random_state=SEED)
        rows = []
        for e in range(1, E3_MAX + 1):
            net.partial_fit(Xf, yf, classes=[0, 1])
            pv = net.predict_proba(Xv)[:, 1]
            rows.append({"epoch": e, "training loss": float(net.loss_),
                         "validation AUC": float(roc_auc_score(yv, pv))})
        return pd.DataFrame(rows).set_index("epoch")
    return _cached("epochs", _ABOUT["epochs"], "the 100 epochs of Experiment 3", build)


def stop_early_numbers() -> pd.DataFrame:
    """Experiment 3's 100 epochs: training loss, validation AUC and the best
    epoch so far."""
    t = _epoch_runs().copy()
    t["best so far (epoch)"] = [int(t["validation AUC"].iloc[:k].to_numpy().argmax()) + 1
                                for k in range(1, len(t) + 1)]
    return t


def _stop(value, quiet: bool = False) -> int:
    return _whole(value, 1, E3_MAX, 5, "stop_after", quiet=quiet)


def draw_epoch_curves(ax_loss, ax_auc, stop_after=5) -> dict:
    """Left: the training loss; right: the validation AUC, both up to epoch
    stop_after; a dot on each marks the best validation epoch so far."""
    k = _stop(stop_after, quiet=True)
    t = _epoch_runs().iloc[:k]
    e = t.index.to_numpy()
    best = int(t["validation AUC"].to_numpy().argmax()) + 1
    for ax, col, colour, lim, name in (
            (ax_loss, "training loss", GRAY, (0.28, 0.49), "training loss (lower is better)"),
            (ax_auc, "validation AUC", RED, (0.735, 0.79), "validation AUC (higher is better)")):
        ax.plot(e, t[col], color=colour if col == "validation AUC" else DARK, lw=1.5,
                marker="o" if k <= 12 else None, ms=3)
        ax.plot([best], [t[col][best]], "o", color=DARK, ms=7, zorder=4)
        ax.set_xlim(0, E3_MAX + 1)
        ax.set_ylim(*lim)
        ax.set_xticks([1, 20, 40, 60, 80, 100])
        ax.set_yticks([0.30, 0.35, 0.40, 0.45] if col == "training loss" else
                      [0.74, 0.75, 0.76, 0.77, 0.78, 0.79])
        ax.set_xlabel("epoch")
        ax.set_title(name)
    v = t["validation AUC"][best]
    ha = "left" if best < 70 else "right"
    ax_auc.annotate(f"best so far: epoch {best} ({v:.3f})", (best, v),
                    xytext=(6 if ha == "left" else -6, 8), textcoords="offset points",
                    ha=ha, va="bottom", fontsize=8.5, color=DARK)
    return {"stop after": k, "training loss": float(t["training loss"].iloc[-1]),
            "validation AUC": float(t["validation AUC"].iloc[-1]), "best epoch": best,
            "best validation AUC": float(v), "training loss at best": float(
                t["training loss"][best])}


def stop_early_figure(stop_after=5, figsize=None, suptitle: bool = True):
    k = _stop(stop_after, quiet=True)
    with _style():
        fig, axes = _new_fig(figsize or (8.0, 3.4), 1, 2)
        r = draw_epoch_curves(axes[0], axes[1], k)
        if suptitle:
            fig.suptitle(f"after {k} epoch{'s' if k > 1 else ''}: training loss "
                         f"{r['training loss']:.3f}, validation AUC {r['validation AUC']:.3f}; "
                         f"best so far: epoch {r['best epoch']} "
                         f"({r['best validation AUC']:.3f})", fontsize=9.5)
    return fig


def stop_early(stop_after=5) -> None:
    """E3: the wide network's training loss and validation AUC drawn up to
    epoch stop_after (1-100); the dot marks the best validation epoch so far."""
    _show(stop_early_figure(_stop(stop_after)))


# ===========================================================================
# The models of the comparison, the audit models (test rows: models fixed)
# ===========================================================================
MODELS = ["logistic", "tree", "forest", "GBDT", "MLP"]
MODEL_NAMES = {"logistic": "logistic regression", "tree": "decision tree",
               "forest": "random forest", "GBDT": "GBDT", "MLP": "MLP"}
_MODEL_ALIASES = {"logistic regression": "logistic", "logreg": "logistic",
                  "decision tree": "tree", "random forest": "forest", "boosting": "GBDT",
                  "gradient boosting": "GBDT", "network": "MLP", "neural network": "MLP",
                  "net": "MLP"}


def network():
    """The deck's network, unfitted: a scaler and MLPClassifier with one hidden
    layer of 16 ReLU units (225 parameters), early stopping, seed 42."""
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(),
                         MLPClassifier(hidden_layer_sizes=(16,), early_stopping=True,
                                       max_iter=500, random_state=SEED))


def _name(value) -> str:
    return _choice(value, MODELS, "name", _MODEL_ALIASES)


def _unfitted(name: str):
    return network() if name == "MLP" else model("boosting" if name == "GBDT" else name)


def _oof(name: str) -> dict:
    """Out-of-fold p of the 21,000 training clients (5 folds of b3.folds()):
    each client's p comes from the model fitted on the other four folds. Also
    each fold's AUC (the same fits as cross_val_score)."""
    def build():
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import cross_val_predict
        X_tr, _, y_tr, _ = _split()
        p = cross_val_predict(_unfitted(name), X_tr, y_tr, cv=folds(),
                              method="predict_proba")[:, 1]
        ya = y_tr.to_numpy()
        fold_auc = [roc_auc_score(ya[te], p[te]) for _, te in folds().split(X_tr, y_tr)]
        return {"p": p, "fold AUC": np.array(fold_auc), "AUC": roc_auc_score(ya, p)}
    about = {"forest": 4, "GBDT": 4, "MLP": 3}.get(name, 0)
    return _cached(("oof", name), about,
                   f"the out-of-fold p of the {MODEL_NAMES[name]}, 5 folds", build)


def _calibration_gap(p, y) -> float:
    """Mean over ten equal-size groups by p (pd.qcut) of |mean p - share
    defaulted|."""
    g = pd.qcut(p, 10, labels=False, duplicates="drop")
    t = pd.DataFrame({"p": p, "y": np.asarray(y), "g": g}).groupby("g").mean()
    return float((t["p"] - t["y"]).abs().mean())


def _fitted(name: str) -> dict:
    """The model fitted on the 21,000 training clients and its p on the 9,000
    test clients (read for models already fixed)."""
    if name == "GBDT":
        return _gbdt("nothing")

    def build():
        from sklearn.metrics import roc_auc_score
        X_tr, X_te, y_tr, y_te = _split()
        m = _unfitted(name).fit(X_tr, y_tr)
        p = m.predict_proba(X_te)[:, 1]
        return {"model": m, "p": p, "AUC": roc_auc_score(y_te, p)}
    about = {"forest": 2, "MLP": _ABOUT["network"]}.get(name, 0)
    return _cached(("fit", name), about, f"the {MODEL_NAMES[name]} on the 21,000", build)


_REMOVE_COLUMNS = {"nothing": [], "female": ["female"], "female, married": ["female", "married"],
                   "female, age": ["female", "age"],
                   "female, married, age": ["female", "married", "age"]}
E5_REMOVED = list(_REMOVE_COLUMNS)


def _gbdt(removed: str = "nothing") -> dict:
    """Our GBDT (Block 3's boosting) fitted on the 21,000 training clients
    without the removed columns; its p on the 9,000 test clients."""
    def build():
        from sklearn.metrics import roc_auc_score
        X_tr, X_te, y_tr, y_te = _split()
        cols = _REMOVE_COLUMNS[removed]
        m = model("boosting").fit(X_tr.drop(columns=cols), y_tr)
        p = m.predict_proba(X_te.drop(columns=cols))[:, 1]
        return {"model": m, "p": p, "AUC": roc_auc_score(y_te, p), "trees": int(m.n_iter_)}
    if removed == "nothing":
        return _cached(("gbdt", removed), _ABOUT["gbdt"], "our GBDT on the 21,000", build)
    return _cached(("gbdt", removed), _ABOUT["refit"],
                   f"our GBDT without {removed}", build)


def _gbdt_p(removed: str = "nothing") -> np.ndarray:
    return _gbdt(removed)["p"]


def _oof_p(name) -> np.ndarray:
    return _oof(_name(name))["p"]


def _mlp_p() -> np.ndarray:
    return _fitted("MLP")["p"]


# ===========================================================================
# The audit: groups, group_report (project helper)
# ===========================================================================
E4_GROUPS = ["sex", "age: under 25", "age: 50 and over"]
_GROUP_ALIASES = {"women": "sex", "men": "sex", "female": "sex", "gender": "sex",
                  "under 25": "age: under 25", "young": "age: under 25",
                  "age": "age: under 25", "25": "age: under 25",
                  "50 and over": "age: 50 and over", "50": "age: 50 and over",
                  "old": "age: 50 and over", "older": "age: 50 and over"}
_GROUP_NAMES = {"sex": ("women", "men"), "age: under 25": ("under 25", "25 and over"),
                "age: 50 and over": ("50 and over", "under 50")}


def _group(value, quiet: bool = False) -> str:
    return _choice(value, E4_GROUPS, "group", _GROUP_ALIASES, quiet=quiet)


def _mask(group: str) -> np.ndarray:
    """The first group of a grouping, on the 9,000 test clients."""
    _, X_te, _, _ = _split()
    if group == "sex":
        return (X_te["female"] == 1).to_numpy()
    if group == "age: under 25":
        return (X_te["age"] < 25).to_numpy()
    return (X_te["age"] >= 50).to_numpy()


def _noise(a: float, na: int, b: float, nb: int) -> float:
    """95% noise of a difference of two shares (normal approximation)."""
    if na == 0 or nb == 0:
        return float("nan")
    return 1.96 * math.sqrt(a * (1 - a) / na + b * (1 - b) / nb)


def _audit(y, p, mask, names, threshold: float = THRESHOLD) -> dict:
    """Counts of the rule "decline when p > threshold" inside each group, and
    the gaps (first group minus second) with their 95% noise."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    mask = np.asarray(mask).astype(bool)
    dec = p > threshold
    groups = {}
    for m, name in ((mask, names[0]), (~mask, names[1])):
        n = int(m.sum())
        yt, dg, pg = y[m], dec[m], p[m]
        TP, FP = int((dg & (yt == 1)).sum()), int((dg & (yt == 0)).sum())
        FN, TN = int((~dg & (yt == 1)).sum()), int((~dg & (yt == 0)).sum())
        nan = float("nan")
        groups[name] = {
            "n": n, "base rate": yt.mean() if n else nan, "mean p": pg.mean() if n else nan,
            "decline rate": dg.mean() if n else nan,
            "false-alarm rate": FP / (FP + TN) if FP + TN else nan,
            "miss rate": FN / (TP + FN) if TP + FN else nan,
            "precision": TP / (TP + FP) if TP + FP else nan,
            "TP": TP, "FP": FP, "FN": FN, "TN": TN,
            "d": dg.astype(int) - yt}
    a, b = groups[names[0]], groups[names[1]]
    gaps = {"decline gap": a["decline rate"] - b["decline rate"],
            "decline noise": _noise(a["decline rate"], a["n"], b["decline rate"], b["n"]),
            "base gap": a["base rate"] - b["base rate"],
            "base noise": _noise(a["base rate"], a["n"], b["base rate"], b["n"])}
    gaps["difference"] = gaps["decline gap"] - gaps["base gap"]
    if a["n"] > 1 and b["n"] > 1:
        gaps["difference noise"] = 1.96 * math.sqrt(np.var(a["d"], ddof=1) / a["n"]
                                                    + np.var(b["d"], ddof=1) / b["n"])
    else:
        gaps["difference noise"] = float("nan")
    return {"groups": groups, "gaps": gaps, "names": tuple(names), "threshold": threshold}


def _gap_lines(r: dict) -> list:
    g, (n1, n2) = r["gaps"], r["names"]
    if not np.isfinite(g["decline gap"]):
        return ["one of the two groups is empty: no gaps to compare"]

    def pm(v):
        return f" (± {100 * v:.1f})" if np.isfinite(v) else ""
    d, dn = g["difference"], g["difference noise"]
    reading = ""
    if np.isfinite(dn):
        reading = (": larger than the noise" if abs(d) > dn else ": within the noise")
    return [f"decline-rate gap ({n1} minus {n2}): {_pts(g['decline gap'])} points"
            f"{pm(g['decline noise'])}",
            f"base-rate gap ({n1} minus {n2}): {_pts(g['base gap'])} points"
            f"{pm(g['base noise'])}",
            f"decline gap minus base-rate gap: {_pts(d)} points{pm(dn)}{reading}"]


def _empty_lines(r: dict) -> list:
    out = []
    for name, x in r["groups"].items():
        if x["n"] == 0:
            out.append(f"{name}: this group is empty")
            continue
        if x["TP"] + x["FP"] == 0:
            out.append(f"{name}: no declines in this group (no precision)")
        if x["TP"] + x["FN"] == 0:
            out.append(f"{name}: no defaulters in this group (no miss rate)")
        if x["FP"] + x["TN"] == 0:
            out.append(f"{name}: no good clients in this group (no false-alarm rate)")
    return out


class GroupReport(pd.DataFrame):
    """group_report's table: a DataFrame that shows its gap lines below it."""

    _metadata = ["notes"]

    @property
    def _constructor(self):
        return pd.DataFrame

    def _text(self) -> str:
        return "\n".join(getattr(self, "notes", []) or [])

    def __repr__(self) -> str:
        return pd.DataFrame(self).__repr__() + ("\n" + self._text() if self._text() else "")

    def _repr_html_(self):
        from html import escape
        html = pd.DataFrame(self)._repr_html_()
        if self._text():
            html += ("<pre style='margin-top:0.4em'>" + escape(self._text()) + "</pre>")
        return html


def _as_p(p) -> np.ndarray:
    a = np.asarray(p, dtype=float)
    if a.ndim == 2 and a.shape[1] == 2:
        _say("p has two columns (predict_proba): using the second one, p of default.")
        a = a[:, 1]
    return a.ravel()


def _as_mask(group, names) -> np.ndarray:
    g = pd.Series(np.asarray(group).ravel())
    if g.dtype == bool:
        return g.to_numpy()
    values = sorted(g.dropna().unique().tolist(), key=str)
    if set(values) <= {0, 1}:
        return (g == 1).to_numpy()
    if len(values) == 2:
        if names[0] in values:
            first = names[0]
        elif names[1] in values:
            first = values[1] if values[0] == names[1] else values[0]
        else:
            raise ValueError(f"group has the values {values[0]!r} and {values[1]!r}, and "
                             f"neither is in names: write group == {values[0]!r} (True for "
                             f"the first group) and give names in that order.")
        _say(f"group has the values {values[0]!r} and {values[1]!r}: the first group is "
             f"{first!r}.")
        return (g == first).to_numpy()
    raise ValueError("group must be True / False for each client (for example "
                     "X_test[\"female\"] == 1)")


def group_report(y_true, p, group, names=("group 1", "group 2"), threshold=0.25):
    """An audit by group of the rule "decline when p > threshold".

    y_true: 1 = defaulted, 0 = repaid; p: the model's probabilities; group:
    True for the clients of the first group (e.g. X_test["female"] == 1).
    Returns one row per group: n, base rate, mean p, decline rate, false-alarm
    rate (good clients wrongly declined), miss rate (defaulters missed),
    precision (declines that default), in %. Below the table: the decline-rate
    gap and the base-rate gap (first group minus second, in percentage points)
    with their 95% noise, and the decline gap minus the base-rate gap with
    its own noise (d = declined - defaulted per client)."""
    t = _share(threshold, 0.25, "threshold")
    names = tuple(str(n) for n in names)
    y = np.asarray(y_true).ravel()
    try:
        yf = y.astype(float)
        ok = bool(np.isin(yf, (0.0, 1.0)).all())
    except (TypeError, ValueError):
        ok = False
    if not ok:
        _say("y_true must be 1 / 0 for each client (1 = the event, for example default); a "
             "text target first becomes 1 / 0, for example (y_test == \"yes\").astype(int).")
        return None
    y = yf.astype(int)
    pp = _as_p(p)
    try:
        mask = _as_mask(group, names)
    except ValueError as err:
        _say(str(err))
        return None
    if not len(y) == len(pp) == len(mask):
        _say(f"y_true, p and group must have one value per client: they have {len(y):,}, "
             f"{len(pp):,} and {len(mask):,}.")
        return None
    r = _audit(y, pp, mask, names, t)
    rows = {}
    for name, x in r["groups"].items():
        rows[name] = {"n": x["n"]} | {f"{k} (%)": round(100 * x[k], 1) if np.isfinite(x[k])
                                      else np.nan
                                      for k in ("base rate", "mean p", "decline rate",
                                                "false-alarm rate", "miss rate",
                                                "precision")}
    out = GroupReport(pd.DataFrame(rows).T.astype({"n": int}))
    out.index.name = f"decline when p > {t:g}"
    out.notes = _empty_lines(r) + _gap_lines(r)
    return out


# ===========================================================================
# E4 - Audit by group
# ===========================================================================
def _e4(group: str, removed: str = "nothing") -> dict:
    _, _, _, y_te = _split()
    return _audit(y_te, _gbdt_p(removed), _mask(group), _GROUP_NAMES[group])


def audit_by_group_numbers() -> pd.DataFrame:
    """Experiment 4 for every grouping (our GBDT, 9,000 test clients, decline
    when p > 0.25), one row per group, in %."""
    rows = []
    for g in E4_GROUPS:
        r = _e4(g)
        for name, x in r["groups"].items():
            rows.append({"grouping": g, "group": name, "n": x["n"]}
                        | {k: 100 * x[k] for k in ("base rate", "decline rate",
                                                   "false-alarm rate", "miss rate",
                                                   "precision", "mean p")})
    return pd.DataFrame(rows).set_index(["grouping", "group"])


def gap_numbers() -> pd.DataFrame:
    """Experiment 4's three gaps per grouping (first group minus second, in
    percentage points) with their 95% noise."""
    rows = []
    for g in E4_GROUPS:
        x = _e4(g)["gaps"]
        rows.append({"grouping": g, "decline gap": 100 * x["decline gap"],
                     "± decline": 100 * x["decline noise"], "base-rate gap": 100 * x["base gap"],
                     "± base": 100 * x["base noise"],
                     "decline gap minus base-rate gap": 100 * x["difference"],
                     "± difference": 100 * x["difference noise"]})
    return pd.DataFrame(rows).set_index("grouping")


def draw_group_rates(ax, group="sex", removed="nothing", title=None) -> dict:
    """Each group's base rate (grey) and decline rate (red) under our GBDT
    (without the removed columns), with both gaps in the panel title."""
    group = _group(group, quiet=True)
    removed = _removed(removed, quiet=True)
    r = _e4(group, removed)
    names = r["names"]
    xs = np.arange(2)
    w = 0.42   # wide enough that "26.7%" at 8.5 pt stays above its own bar on a slide
    for i, name in enumerate(names):
        x = r["groups"][name]
        for dx, key, colour in ((-w / 2, "base rate", GRAY), (w / 2, "decline rate", RED)):
            v = x[key]
            ax.bar(xs[i] + dx, v, width=w, color=colour, zorder=2)
            ax.text(xs[i] + dx, v + 0.008, _pct(v), ha="center", va="bottom", fontsize=8.5,
                    color=NEAR_BLACK)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{n}\n{r['groups'][n]['n']:,} clients" for n in names])
    ax.set_ylim(0, 0.50)
    ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4])
    ax.yaxis.set_major_formatter(b3.PCT0)
    ax.set_xlim(-0.5, 1.5)
    ax.legend(handles=[Rectangle((0, 0), 1, 1, color=GRAY), Rectangle((0, 0), 1, 1, color=RED)],
              labels=["base rate", "decline rate"], loc="upper right", fontsize=8.5,
              handlelength=1.0, ncols=2, columnspacing=1.0, borderaxespad=0.1)
    g = r["gaps"]
    if title is None:
        title = (f"decline gap {_pts(g['decline gap'])} points\n"
                 f"base-rate gap {_pts(g['base gap'])} points")
    if title:
        ax.set_title(title)
    return {"group": group, "removed": removed, "names": names,
            **{f"{n}: {k}": r["groups"][n][k] for n in names
               for k in ("n", "base rate", "decline rate", "mean p")},
            **g}


def draw_error_rates(ax, group="sex") -> dict:
    """Each group's two error rates under our GBDT: good clients wrongly
    declined (false-alarm rate, blue) and defaulters missed (miss rate, red)."""
    group = _group(group, quiet=True)
    r = _e4(group)
    names = r["names"]
    xs = np.arange(2)
    w = 0.42
    for i, name in enumerate(names):
        x = r["groups"][name]
        for dx, key, colour in ((-w / 2, "false-alarm rate", BLUE), (w / 2, "miss rate", RED)):
            v = x[key]
            ax.bar(xs[i] + dx, v, width=w, color=colour, zorder=2)
            ax.text(xs[i] + dx, v + 0.008, _pct(v), ha="center", va="bottom", fontsize=8.5,
                    color=NEAR_BLACK)
    ax.set_xticks(xs)
    ax.set_xticklabels(list(names))
    ax.set_ylim(0, 0.62)
    ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ax.yaxis.set_major_formatter(b3.PCT0)
    ax.set_xlim(-0.5, 1.5)
    ax.legend(handles=[Rectangle((0, 0), 1, 1, color=BLUE), Rectangle((0, 0), 1, 1, color=RED)],
              labels=["good clients wrongly declined", "defaulters missed"], loc="upper center",
              fontsize=8.5, handlelength=1.0, ncols=1, borderaxespad=0.1)
    ax.set_title("the two error rates")
    return {"group": group, **{f"{n}: {k}": r["groups"][n][k] for n in names
                               for k in ("false-alarm rate", "miss rate", "precision")}}


def audit_by_group_figure(group="sex", figsize=None, suptitle: bool = True):
    group = _group(group, quiet=True)
    with _style():
        fig, axes = _new_fig(figsize or (8.0, 3.5), 1, 2)
        draw_group_rates(axes[0], group)
        draw_error_rates(axes[1], group)
        if suptitle:
            g = _e4(group)["gaps"]
            within = ("larger than the noise" if abs(g["difference"]) > g["difference noise"]
                      else "within the noise")
            fig.suptitle(f"{group}: decline gap minus base-rate gap {_pts(g['difference'])} "
                         f"points (± {100 * g['difference noise']:.1f}), {within}",
                         fontsize=9.5)
    return fig


def audit_by_group(group="sex") -> None:
    """E4: our GBDT on the 9,000 test clients, decline when p > 0.25, audited
    by group ("sex", "age: under 25" or "age: 50 and over"); below the
    figure, group_report's table and its three gap lines."""
    group = _group(group)
    _show(audit_by_group_figure(group))
    _, _, _, y_te = _split()
    _display(group_report(y_te, _gbdt_p(), _mask(group), _GROUP_NAMES[group], THRESHOLD))


# ===========================================================================
# E5 - Remove a column
# ===========================================================================
_REMOVED_KEYS = {frozenset(v): k for k, v in _REMOVE_COLUMNS.items()}
_WORDS = {"sex": "female", "gender": "female", "women": "female", "woman": "female",
          "marriage": "married", "marital": "married", "none": "", "nothing": "",
          "no": "", "ages": "age"}


def _removed(value, quiet: bool = False) -> str:
    value = _plain(value)
    if isinstance(value, str) and value in _REMOVE_COLUMNS:
        return value
    if isinstance(value, (list, tuple, set)):
        parts = [str(v) for v in value]
    elif isinstance(value, str):
        parts = re.split(r"[,+&/;]|\band\b|\s+", value)
    else:
        parts = [str(value)]
    words = {_WORDS.get(_key(w), _key(w)) for w in parts if _key(w)}
    words.discard("")
    found = _REMOVED_KEYS.get(frozenset(words))
    if found is not None:
        return found
    if not quiet:
        quotes = "" if isinstance(value, str) else ' Text values take quotes: removed = "female".'
        _say(f"removed {value!r} is not one of " + ", ".join(f'"{c}"' for c in E5_REMOVED)
             + ': showing "nothing".' + quotes)
    return "nothing"


def _e5(removed: str) -> dict:
    _, _, _, y_te = _split()
    full, new = _gbdt("nothing"), _gbdt(removed)
    changed = int(((full["p"] > THRESHOLD) != (new["p"] > THRESHOLD)).sum())
    return {"AUC": new["AUC"], "trees": new["trees"], "changed": changed, "n": len(y_te)}


def remove_column_numbers() -> pd.DataFrame:
    """Experiment 5 for every choice: test AUC, trees in the relay, decisions
    changed (of 9,000) and the decline rates and mean p by sex and by age (%)."""
    rows = []
    for k in E5_REMOVED:
        e = _e5(k)
        s, a = _e4("sex", k)["groups"], _e4("age: under 25", k)["groups"]
        rows.append({"removed": k, "test AUC": e["AUC"], "trees": e["trees"],
                     "decisions changed": e["changed"],
                     "women declined": 100 * s["women"]["decline rate"],
                     "men declined": 100 * s["men"]["decline rate"],
                     "under 25 declined": 100 * a["under 25"]["decline rate"],
                     "25 and over declined": 100 * a["25 and over"]["decline rate"],
                     "women mean p": 100 * s["women"]["mean p"],
                     "men mean p": 100 * s["men"]["mean p"],
                     "under 25 mean p": 100 * a["under 25"]["mean p"],
                     "25 and over mean p": 100 * a["25 and over"]["mean p"]})
    return pd.DataFrame(rows).set_index("removed")


def flip_numbers(removed="female") -> pd.DataFrame:
    """[R11]: who changes when the columns are removed: by sex, the test
    clients moved from approve to decline and from decline to approve
    (decline when p > 0.25), and the mean p against the base rate (%)."""
    removed = _removed(removed, quiet=True)
    _, X_te, _, y_te = _split()
    d1, d2 = _gbdt_p() > THRESHOLD, _gbdt_p(removed) > THRESHOLD
    women = (X_te["female"] == 1).to_numpy()
    rows = []
    for m, name in ((women, "women"), (~women, "men")):
        rows.append({"group": name, "approve -> decline": int((m & ~d1 & d2).sum()),
                     "decline -> approve": int((m & d1 & ~d2).sum()),
                     "mean p (%)": 100 * _gbdt_p(removed)[m].mean(),
                     "base rate (%)": 100 * y_te.to_numpy()[m].mean()})
    return pd.DataFrame(rows).set_index("group")


def remove_column_figure(removed="nothing", figsize=None, suptitle: bool = True):
    removed = _removed(removed, quiet=True)
    with _style():
        fig, axes = _new_fig(figsize or (8.0, 3.5), 1, 2)
        draw_group_rates(axes[0], "sex", removed, title="by sex")
        draw_group_rates(axes[1], "age: under 25", removed, title="by age")
        if suptitle:
            e = _e5(removed)
            what = "nothing removed" if removed == "nothing" else f"without {removed}"
            fig.suptitle(f"{what}: test AUC {e['AUC']:.3f}; {e['changed']} of "
                         f"{e['n']:,} decisions changed", fontsize=9.5)
    return fig


def remove_column(removed="nothing") -> None:
    """E5: our GBDT trained again without some columns ("nothing", "female",
    "female, married", "female, age" or "female, married, age"); the same
    9,000 test clients and the same rule, by sex and by age."""
    _show(remove_column_figure(_removed(removed)))


# ===========================================================================
# prepare(): everything Experiments 2-5 draw, once
# ===========================================================================
def _part(what: str, about: float, keys: list, fn) -> None:
    if all(k in _CACHE for k in keys):
        return
    if VERBOSE:
        print(f"Computing {what} (about {about:.0f} s, only the first time) ...", end="",
              flush=True)
    t0 = time.perf_counter()
    _DEPTH[0] += 1
    try:
        fn()
    finally:
        _DEPTH[0] -= 1
    if VERBOSE:
        print(_took(t0), flush=True)


def prepare() -> None:
    """Compute what Experiments 2-5 draw: the seven learning rates, the 100
    epochs, our GBDT and its four refits without columns. Run it once, right
    after the setup cell (under a minute: about 20 s on a laptop, longer on
    Colab); afterwards every redraw is instant."""
    _split()
    _part("the seven learning rates of Experiment 2", _ABOUT["lr"], ["lr"], _lr_runs)
    _part("the 100 epochs of Experiment 3", _ABOUT["epochs"], ["epochs"], _epoch_runs)
    _part("our GBDT for Experiment 4", _ABOUT["gbdt"], [("gbdt", "nothing")], _gbdt)
    rest = [k for k in E5_REMOVED if k != "nothing"]
    _part("the four GBDTs without columns of Experiment 5", _ABOUT["refits"],
          [("gbdt", k) for k in rest], lambda: [_gbdt(k) for k in rest])
    print("Ready: tonight's experiments will redraw at once.")


# ===========================================================================
# Demos (replay at home)
# ===========================================================================
def _no_hidden() -> dict:
    """Demo 1: a network with no hidden layer (one sigmoid neuron) against
    Block 2's logistic regression, out of fold on the 21,000."""
    def build():
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import cross_val_predict
        from sklearn.neural_network import MLPClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        X_tr, _, y_tr, _ = _split()
        ya = y_tr.to_numpy()
        net = make_pipeline(StandardScaler(),
                            MLPClassifier(hidden_layer_sizes=(), early_stopping=False,
                                          max_iter=300, random_state=SEED))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            pn = cross_val_predict(net, X_tr, y_tr, cv=folds(), method="predict_proba")[:, 1]
        pl = _oof("logistic")["p"]
        idx = [te for _, te in folds().split(X_tr, y_tr)]
        fn = np.array([roc_auc_score(ya[te], pn[te]) for te in idx])
        fl = np.array([roc_auc_score(ya[te], pl[te]) for te in idx])
        return {"network folds": fn, "logistic folds": fl,
                "correlation": float(np.corrcoef(pn, pl)[0, 1]),
                "largest difference": float(np.abs(pn - pl).max()),
                "mean difference": float(np.abs(pn - pl).mean())}
    return _cached("no hidden", _ABOUT["no hidden"],
                   "a network with no hidden layer and logistic regression, 5 folds each",
                   build)


def no_hidden_layer_numbers() -> pd.Series:
    """[C1] and [N5b]: both CV AUCs (mean, sd of the five folds) and the
    agreement of the two out-of-fold p."""
    r = _no_hidden()
    return pd.Series({"network, no hidden layer: CV AUC": r["network folds"].mean(),
                      "network: sd": r["network folds"].std(),
                      "logistic regression: CV AUC": r["logistic folds"].mean(),
                      "logistic regression: sd": r["logistic folds"].std(),
                      "correlation of the out-of-fold p": r["correlation"],
                      "largest |difference| of the out-of-fold p": r["largest difference"],
                      "mean |difference| of the out-of-fold p": r["mean difference"]})


def no_hidden_layer() -> None:
    """Demo 1: MLPClassifier(hidden_layer_sizes=()) is one sigmoid neuron:
    the same formula as logistic regression. Both 5-fold CV AUCs, and how
    closely their out-of-fold p agree."""
    r = no_hidden_layer_numbers()
    print(f"5-fold CV AUC on the 21,000 training clients:\n"
          f"  network with no hidden layer  {r.iloc[0]:.3f} (sd {r.iloc[1]:.3f})\n"
          f"  logistic regression (Block 2) {r.iloc[2]:.3f} (sd {r.iloc[3]:.3f})")
    print(f"Their out-of-fold p: correlation {r.iloc[4]:.5f}; largest difference for one "
          f"client {r.iloc[5]:.4f}; on average {r.iloc[6]:.4f}.")


def _net_once() -> dict:
    def build():
        X_tr, _, y_tr, _ = _split()
        net = network().fit(X_tr, y_tr)
        mlp = net[-1]
        return {"parameters": int(sum(w.size for w in mlp.coefs_)
                                  + sum(b.size for b in mlp.intercepts_)),
                "epochs": int(mlp.n_iter_),
                "best epoch": int(np.argmax(mlp.validation_scores_)) + 1,
                "best validation accuracy": float(mlp.best_validation_score_)}
    return _cached("net once", _ABOUT["network"],
                   "our network on the 21,000 training clients", build)


def our_network_numbers() -> pd.Series:
    """[R2]: parameters, epochs run, best epoch and its validation accuracy,
    5-fold CV AUC (mean, sd) of the deck's network."""
    r = _net_once()
    f = _oof("MLP")["fold AUC"]
    return pd.Series({**r, "CV AUC": f.mean(), "CV sd": f.std()})


def our_network() -> None:
    """Demo 2: tonight's network on our data: its parameters, where early
    stopping ended, and its 5-fold CV AUC."""
    r = our_network_numbers()
    print(f"parameters: {int(r['parameters'])} (12 × 16 + 16 = 208, 16 + 1 = 17)")
    print(f"early stopping: ended after epoch {int(r['epochs'])}, kept epoch "
          f"{int(r['best epoch'])}'s weights (validation accuracy "
          f"{r['best validation accuracy']:.4f})")
    print(f"5-fold CV AUC: {r['CV AUC']:.3f} (sd {r['CV sd']:.3f})")


def _tournament() -> pd.DataFrame:
    """The five models: 5-fold CV AUC (mean, sd, each fold) and out-of-fold
    calibration gap and AUC on the 21,000 training clients; then, read once
    for models already fixed, the test AUC, log-loss, Brier score and mean p."""
    def build():
        from sklearn.metrics import brier_score_loss, log_loss
        _, _, y_tr, y_te = _split()
        rows = []
        for name in MODELS:
            o, f = _oof(name), _fitted(name)
            rows.append({"model": MODEL_NAMES[name], "CV AUC": o["fold AUC"].mean(),
                         "CV sd": o["fold AUC"].std(),
                         "folds": ", ".join(f"{v:.3f}" for v in o["fold AUC"]),
                         "out-of-fold AUC": o["AUC"],
                         "calibration gap (out of fold)": _calibration_gap(o["p"], y_tr),
                         "test AUC": f["AUC"], "test log-loss": log_loss(y_te, f["p"]),
                         "test Brier": brier_score_loss(y_te, f["p"]),
                         "test mean p": f["p"].mean()})
        return pd.DataFrame(rows).set_index("model")
    return _cached("tournament", _ABOUT["tournament"],
                   "the comparison of five models, 5 folds each, then one fit", build)


def tournament_numbers() -> pd.DataFrame:
    return _tournament().copy()


def draw_tournament(ax) -> pd.DataFrame:
    """One row per model, sorted by CV AUC: filled dot = CV mean with a bar
    of ± one sd (choosing); hollow dot = test AUC (read once)."""
    t = _tournament()
    order = sorted(range(len(t)), key=lambda i: (-round(t["CV AUC"].iloc[i], 4), i))
    t = t.iloc[order]
    ys = np.arange(len(t))[::-1]
    for y, (_, r) in zip(ys, t.iterrows()):
        m, s = r["CV AUC"], r["CV sd"]
        ax.plot([m - s, m + s], [y, y], color=DARK, lw=1.4, solid_capstyle="butt", zorder=2)
        ax.plot(m, y, "o", color=DARK, ms=7, zorder=3)
        ax.plot(r["test AUC"], y, "o", mfc="white", mec=RED, mew=1.5, ms=7, zorder=4)
        ax.text(m, y + 0.2, f"{m:.3f}", ha="center", va="bottom", fontsize=8.5, color=DARK)
    ax.set_yticks(ys)
    ax.set_yticklabels(list(t.index))
    ax.set_ylim(-0.6, len(t) - 0.2)
    ax.set_xlim(0.72, 0.80)
    ax.set_xticks([0.72, 0.74, 0.76, 0.78, 0.80])
    ax.set_xlabel("AUC (higher is better)")
    ax.legend(handles=[plt.Line2D([], [], color=DARK, marker="o", lw=1.4, ms=6),
                       plt.Line2D([], [], color=RED, marker="o", mfc="white", lw=0, mew=1.5,
                                  ms=6)],
              labels=["CV mean ± sd (choosing)", "test, read once"], loc="lower center",
              bbox_to_anchor=(0.5, 1.0), ncols=2, fontsize=8.5, handlelength=1.6,
              columnspacing=1.5, borderaxespad=0.2)
    return t


def tournament_figure(figsize=None, suptitle: bool = True):
    with _style():
        fig, ax = _new_fig(figsize or (6.4, 3.0))
        draw_tournament(ax)
        if suptitle:
            fig.suptitle("Five models on our data: chosen on CV (21,000 training clients)",
                         fontsize=9.5)
    return fig


def tournament() -> None:
    """Demo 3: the five models of the course on our data: 5-fold CV AUC and
    the out-of-fold calibration gap (choosing), then the test AUC, read once."""
    t = _tournament()
    show = pd.DataFrame({"CV AUC": t["CV AUC"].round(3), "sd": t["CV sd"].round(3),
                         "calibration gap (points)":
                             (100 * t["calibration gap (out of fold)"]).round(1),
                         "test AUC (read once)": t["test AUC"].round(3)})
    _show(tournament_figure())
    _display(show)


def calibration_numbers(name="logistic") -> pd.DataFrame:
    """The reliability diagram's ten equal-size groups of the out-of-fold p of
    the 21,000 training clients (Block 3's reliability_points)."""
    _, _, y_tr, _ = _split()
    return b3.reliability_points(_oof_p(name), y_tr)


def draw_reliability(ax, name="logistic", ylabel: bool = True) -> dict:
    """Reliability diagram of the out-of-fold p of the 21,000 training
    clients: ten equal-size groups by p; average p (across) against the share
    who defaulted (up); the diagonal = honest."""
    n = _name(name)
    _, _, y_tr, _ = _split()
    r = calibration_numbers(n)
    gap = _calibration_gap(_oof_p(n), y_tr)
    ax.plot([0, 1], [0, 1], ls="--", color=GRAY, lw=1)
    ax.plot(r["average p"], r["share defaulted"], "-o", color=RED, ms=4, lw=1.4)
    ax.set_xlim(0, 0.8)
    ax.set_ylim(0, 0.8)
    ax.set_aspect("equal")
    ax.set_xticks([0, 0.4, 0.8])
    ax.set_yticks([0, 0.4, 0.8])
    ax.set_xlabel("average p")
    if ylabel:
        ax.set_ylabel("share defaulted")
    ax.set_title(f"{MODEL_NAMES[n]}\nmean gap {100 * gap:.1f} points")
    return {"model": MODEL_NAMES[n], "calibration gap": gap, "groups": r}


def _proxy() -> dict:
    def build():
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.model_selection import cross_val_score
        X_tr, _, _, _ = _split()
        Xo = X_tr.drop(columns=["female"])
        s = cross_val_score(HistGradientBoostingClassifier(random_state=SEED), Xo,
                            X_tr["female"], cv=folds(), scoring="roc_auc")
        g = X_tr.groupby("female").mean()
        return {"AUC": float(s.mean()), "sd": float(s.std()), "means": g}
    return _cached("proxy", _ABOUT["proxy"],
                   "a GBDT that guesses sex from the other 11 columns, 5 folds", build)


def proxy_numbers() -> pd.DataFrame:
    """[R12]: the training clients' means by sex, and the CV AUC of guessing
    sex from the other 11 columns."""
    r = _proxy()
    g = r["means"].T.rename(columns={1: "women", 0: "men"})[["women", "men"]]
    g.loc["guessing sex: CV AUC (GBDT)"] = [r["AUC"], r["AUC"]]
    return g


_PLAIN_COLUMNS = {"married": "married (share)", "age": "age (years)",
                  "limit_bal": "credit limit (NT$)",
                  "months_delayed_6m": "months late (of last 6)",
                  "utilisation": "share of the limit used"}


def proxy_check() -> None:
    """Demo 4: can the other 11 columns guess sex? A GBDT's 5-fold CV AUC, and
    the means of a few columns for women and men (training clients)."""
    r = _proxy()
    print(f"Guessing sex from the other 11 columns: CV AUC {r['AUC']:.3f} (sd {r['sd']:.3f}); "
          "0.5 = no better than a coin, 1 = always right.")
    m = r["means"]
    t = pd.DataFrame({"women": m.loc[1, list(_PLAIN_COLUMNS)],
                      "men": m.loc[0, list(_PLAIN_COLUMNS)]})
    t.index = [_PLAIN_COLUMNS[c] for c in t.index]
    formats = {"age (years)": "{:.1f}", "credit limit (NT$)": "{:,.0f}",
               "months late (of last 6)": "{:.2f}"}
    shown = t.astype(object)
    for row in shown.index:
        shown.loc[row] = [formats.get(row, "{:.3f}").format(v) for v in t.loc[row]]
    print("Means on the 21,000 training clients:")
    _display(shown)


def mlp_audit_numbers() -> pd.DataFrame:
    """[R13]: the audit of the deck's network (test p, decline when p > 0.25),
    by sex and by age, in %."""
    _, X_te, _, y_te = _split()
    p = _mlp_p()
    rows = []
    for g in ("sex", "age: under 25"):
        r = _audit(y_te, p, _mask(g), _GROUP_NAMES[g])
        for name, x in r["groups"].items():
            rows.append({"grouping": g, "group": name, "n": x["n"]}
                        | {k: 100 * x[k] for k in ("base rate", "mean p", "decline rate",
                                                   "false-alarm rate", "miss rate",
                                                   "precision")}
                        | {"decline gap": 100 * r["gaps"]["decline gap"],
                           "base-rate gap": 100 * r["gaps"]["base gap"]})
    return pd.DataFrame(rows).set_index(["grouping", "group"])


def mlp_audit() -> None:
    """Demo 5: the same audit for tonight's network (models already fixed):
    by sex and by age, decline when p > 0.25."""
    _, _, _, y_te = _split()
    p = _mlp_p()
    from sklearn.metrics import roc_auc_score
    print(f"Our network on the 9,000 test clients: AUC {roc_auc_score(y_te, p):.3f}; "
          f"declined {_pct((p > THRESHOLD).mean())}")
    for g in ("sex", "age: under 25"):
        _display(group_report(y_te, p, _mask(g), _GROUP_NAMES[g], THRESHOLD))


# ===========================================================================
# Drawing helpers of the slides (the small worlds; Backup 15)
# ===========================================================================
def draw_sigmoid(ax) -> dict:
    """The sigmoid for z from -5 to 5, clients 1-4 as dots with their p, a
    dashed line at p = 0.5."""
    z = np.linspace(-5, 5, 300)
    ax.plot(z, sigmoid(z), color=DARK, lw=1.6, zorder=2)
    ax.axhline(0.5, color=GRAY, lw=0.9, ls="--", zorder=1)
    zs = (W[0] * _CL["x1"] + W[1] * _CL["x2"] + W[2]).to_numpy()
    ps = sigmoid(zs)
    offsets = {1: (8, -2), 2: (8, -2), 3: (-8, 4), 4: (-8, 3)}
    for c, zz, pp in zip(_CL.index, zs, ps):
        ax.plot(zz, pp, "o", color=RED, ms=6, zorder=3)
        dx, dy = offsets[c]
        ax.annotate(f"{c}: {pp:.3f}", (zz, pp), xytext=(dx, dy), textcoords="offset points",
                    ha="left" if dx > 0 else "right", va="center", fontsize=8.5,
                    color=NEAR_BLACK)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xticks([-4, -2, 0, 2, 4])
    ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel("z")
    ax.set_ylabel("p")
    return {"z": zs.tolist(), "p": ps.tolist()}


def draw_xor(ax) -> dict:
    """The four points: class 1 red filled at (1,0) and (0,1), class 0 blue
    rings at (0,0) and (1,1)."""
    for (x1, x2), cls in zip(_PT[["x1", "x2"]].to_numpy(), _PT["class"]):
        if cls == 1:
            ax.plot(x1, x2, "o", color=RED, ms=11, zorder=3)
        else:
            ax.plot(x1, x2, "o", mfc="white", mec=BLUE, mew=2.2, ms=11, zorder=3)
    ax.set_xlim(-0.35, 1.35)
    ax.set_ylim(-0.35, 1.35)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_aspect("equal")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.legend(handles=[plt.Line2D([], [], color=RED, marker="o", lw=0, ms=8),
                       plt.Line2D([], [], color=BLUE, marker="o", mfc="white", mew=2, lw=0,
                                  ms=8)],
              labels=["class 1", "class 0"], loc="center", fontsize=8.5, handlelength=1.0)
    return {"points": _PT.to_dict("index")}


def draw_hidden_space(ax) -> dict:
    """The four points in the new columns h1, h2: both class-0 points land on
    (0,0); a straight line (z = 0, that is h1 + h2 = 0.5) now separates them."""
    t = xor_table()
    h = np.array([-0.15, 0.55])
    ax.plot(h, 0.5 - h, ls="--", color=DARK, lw=1.2, zorder=1)
    ax.annotate("z = 0", (-0.15, 0.65), xytext=(2, 3), textcoords="offset points",
                ha="left", va="bottom", fontsize=8.5, color=DARK)
    for dx in (-0.045, 0.045):
        ax.plot(dx, -dx, "o", mfc="white", mec=BLUE, mew=2.0, ms=10, zorder=3)
    ax.annotate("(0,0) and (1,1)", (-0.045, -0.045), xytext=(-8, -8),
                textcoords="offset points", ha="left", va="top", fontsize=8.5,
                color=NEAR_BLACK)
    ax.plot(1, 0, "o", color=RED, ms=10, zorder=3)
    ax.plot(0, 1, "o", color=RED, ms=10, zorder=3)
    ax.annotate("(1,0)", (1, 0), xytext=(0, 10), textcoords="offset points", ha="center",
                va="bottom", fontsize=8.5)
    ax.annotate("(0,1)", (0, 1), xytext=(10, 0), textcoords="offset points", ha="left",
                va="center", fontsize=8.5)
    ax.set_xlim(-0.3, 1.3)
    ax.set_ylim(-0.42, 1.25)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_aspect("equal")
    ax.set_xlabel("h1")
    ax.set_ylabel("h2")
    return {"table": t}


def draw_slope(ax) -> dict:
    """L(w) = (w - 3)^2 for w from -1 to 7; at w = 0 (loss 9) a short tangent
    with slope -6; the lowest point is w = 3."""
    w = np.linspace(-1, 7, 300)
    ax.plot(w, (w - 3) ** 2, color=DARK, lw=1.6, zorder=2)
    tw = np.array([-0.75, 0.75])
    ax.plot(tw, 9 - 6 * tw, color=RED, lw=1.6, zorder=3)
    ax.plot(0, 9, "o", color=RED, ms=6, zorder=4)
    # inside the bowl, right of the dot: no curve and no tangent there
    ax.annotate("slope -6", (0, 9), xytext=(7, 5), textcoords="offset points",
                ha="left", va="bottom", fontsize=8.5, color=RED)
    ax.plot(3, 0, "o", color=DARK, ms=6, zorder=4)
    ax.annotate("", (3, 0.4), xytext=(3, 6.3),
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.9, shrinkA=0, shrinkB=3))
    ax.text(3, 6.5, "lowest point\nw = 3", ha="center", va="bottom", fontsize=8.5,
            color=NEAR_BLACK, linespacing=1.1)
    ax.set_xlim(-1, 7)
    ax.set_ylim(-1, 17)
    ax.set_xticks([-1, 0, 1, 2, 3, 4, 5, 6, 7])
    ax.set_yticks([0, 4, 8, 12, 16])
    ax.set_xlabel("weight w")
    ax.set_ylabel("loss")
    return {"w": 0, "loss": 9, "slope": -6, "lowest": 3}


_LR_TITLES = {0.05: "0.05: slow", 0.25: "0.25: closes in,\nhalving the distance",
              1.1: "1.1: overshoots;\nthe loss grows"}
# Label offsets of the slide's three rates (start, 1, 2, 3): (dx, dy) in points,
# each label off the curve, off the path and off the other dots; checked at the
# slide's size (0.92 of the text width, 1.45 in high).
_LR_SIDES = {0.05: [(6, 2), (-6, 0), (6, 2), (-6, 0)],
             0.25: [(6, 0), (-5, -4), (0, 6), (4, 4)],
             1.1: [(-6, 0), (0, -6), (-6, 0), (2, -6)]}


# Head room above the start dot, so that its label clears the panel title.
_LR_TOP = {0.05: 11.5}


def _lr_path(lr: float, start: float = 0.0, target: float = 3.0, steps: int = 3) -> list:
    ws = [start]
    for _ in range(steps):
        ws.append(ws[-1] - lr * 2 * (ws[-1] - target))
    return ws


def _free_labels(ax, points, labels, colours, curve, first=None) -> None:
    """Put each label next to its dot, at the first of right, below, left,
    above (first: an order of its own for the first label) where it covers no
    dot, no other label and no part of the curve, inside the axes."""
    fig = ax.figure
    fig.canvas.draw()
    k = 72 / fig.dpi
    to_pt = lambda xy: ax.transData.transform(xy) * k   # noqa: E731
    dots = [to_pt(p) for p in points]
    line = [to_pt(p) for p in curve]
    box_ax = ax.get_window_extent()
    ax0, ay0, ax1, ay1 = box_ax.x0 * k, box_ax.y0 * k, box_ax.x1 * k, box_ax.y1 * k
    placed = []
    sides = {"right": (6, 0, "left", "center"), "below": (0, -6, "center", "top"),
             "left": (-6, 0, "right", "center"), "above": (0, 6, "center", "bottom")}
    for i, (d, text, colour) in enumerate(zip(dots, labels, colours)):
        w, h = 0.62 * 8.5 * len(text), 8.5
        order = (first if i == 0 and first else ["right", "below", "left", "above"])
        best = None
        for side in order:
            dx, dy, ha, va = sides[side]
            cx = d[0] + dx + {"left": w / 2, "right": -w / 2, "center": 0}[ha]
            cy = d[1] + dy + {"bottom": h / 2, "top": -h / 2, "center": 0}[va]
            box = (cx - w / 2 - 1, cy - h / 2 - 1, cx + w / 2 + 1, cy + h / 2 + 1)
            inside = ax0 <= box[0] and box[2] <= ax1 and ay0 <= box[1] and box[3] <= ay1
            hits = any(box[0] - 3.5 < q[0] < box[2] + 3.5 and box[1] - 3.5 < q[1] < box[3] + 3.5
                       for j, q in enumerate(dots) if j != i)
            hits |= any(box[0] < q[0] < box[2] and box[1] < q[1] < box[3] for q in line)
            hits |= any(box[0] < b[2] and b[0] < box[2] and box[1] < b[3] and b[1] < box[3]
                        for b in placed)
            if inside and not hits:
                best = (side, box)
                break
        side, box = best or (order[0], (0, 0, 0, 0))
        placed.append(box)
        dx, dy, ha, va = sides[side]
        ax.annotate(text, points[i], xytext=(dx, dy), textcoords="offset points", ha=ha,
                    va=va, fontsize=8.5, color=colour)


def draw_lr_path(ax, lr=0.25, xlabel: bool = True, ylabel: bool = True) -> dict:
    """The loss (w - 3)^2 with three steps from w = 0 at this learning rate:
    the start and steps 1-3 as numbered dots joined by thin lines."""
    lr = _positive(lr, 0.25, "lr")
    ws = _lr_path(lr)
    ls = [(w - 3) ** 2 for w in ws]
    # the window: the path and the lowest point (w = 3), with a margin
    lo, hi = min(-0.6, min(ws) - 0.8), max(4.6, max(ws) + 1.2)
    if min(ws) < 0:          # room for "start" left of its dot
        lo = min(ws) - 2.7
    top = max(_LR_TOP.get(round(lr, 6), 10.5), 1.15 * max(ls))
    w = np.linspace(lo, hi, 400)
    ax.plot(w, (w - 3) ** 2, color=GRAY, lw=1.3, zorder=1)
    ax.plot(ws, ls, color=RED, lw=0.9, zorder=2)
    ax.plot(ws[0], ls[0], "o", mfc="white", mec=DARK, mew=1.3, ms=5, zorder=3)
    ax.plot(ws[1:], ls[1:], "o", color=RED, ms=5, zorder=3)
    ax.set_xlim(lo, hi)
    ax.set_ylim(-0.12 * top, top)
    step = next(k for k in (5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10_000)
                if top / k <= 5 or k == 10_000)
    ax.set_yticks(np.arange(0, top, step))
    if hi - lo <= 14:
        ax.set_xticks([0, 3] if hi < 6 else [0, 3, 6, 9])
    else:                     # a rate far too large: let matplotlib choose the ticks
        ax.xaxis.set_major_locator(plt.MaxNLocator(5, integer=True))
    if xlabel:
        ax.set_xlabel("weight w")
    if ylabel:
        ax.set_ylabel("loss")
    title = _LR_TITLES.get(round(lr, 6), f"learning rate {lr:g}")
    ax.set_title(title)
    texts, colours = ["start", "1", "2", "3"], [DARK, RED, RED, RED]
    sides = _LR_SIDES.get(round(lr, 6))
    if sides is None or len(ws) != 4:
        keep = (w - 3) ** 2 <= top
        curve = list(zip(w[keep][::3], ((w - 3) ** 2)[keep][::3]))
        for (wa, la), (wb, lb) in zip(zip(ws, ls), zip(ws[1:], ls[1:])):   # the path too
            curve += [(wa + f * (wb - wa), la + f * (lb - la)) for f in np.linspace(0, 1, 40)]
        # dots that land on the same point share one label ("1, 2, 3")
        pts, names, cols = [], [], []
        for wk, lk, text, colour in zip(ws, ls, texts, colours):
            same = [i for i, (pw, _) in enumerate(pts) if abs(pw - wk) < 1e-9]
            if same:
                names[same[0]] += f", {text}"
            else:
                pts.append((wk, lk))
                names.append(text)
                cols.append(colour)
        _free_labels(ax, pts, names, cols, curve, first=["left", "above", "below", "right"])
    else:
        for (dx, dy), wk, lk, text, colour in zip(sides, ws, ls, texts, colours):
            ax.annotate(text, (wk, lk), xytext=(dx, dy), textcoords="offset points",
                        ha="left" if dx > 0 else ("right" if dx < 0 else "center"),
                        va="bottom" if dy > 0 else ("top" if dy < 0 else "center"),
                        fontsize=8.5, color=colour)
    return {"lr": lr, "w": ws, "loss": ls}


def draw_toy_epochs(ax_loss, ax_auc) -> dict:
    """The ten made-up epochs: left the training loss (falls at every epoch),
    right the validation AUC (peaks at epoch 5)."""
    e = _ET.index.to_numpy()
    ax_loss.plot(e, _ET["training loss"], "-o", color=DARK, ms=4, lw=1.5)
    ax_loss.set_ylim(0.25, 0.65)
    ax_loss.set_title("training loss (lower is better)")
    v = _ET["validation AUC"]
    ax_auc.plot(e, v, "-o", color=RED, ms=4, lw=1.5)
    best = int(v.idxmax())
    ax_auc.plot(best, v[best], "o", color=DARK, ms=8, zorder=4)
    ax_auc.annotate(f"epoch {best}", (best, v[best]), xytext=(0, 8), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8.5, color=DARK)
    ax_auc.set_ylim(0.69, 0.80)
    ax_auc.set_title("validation AUC (higher is better)")
    for ax in (ax_loss, ax_auc):
        ax.set_xlim(0.5, 10.5)
        ax.set_xticks(range(1, 11))
        ax.set_xlabel("epoch")
    return {"best epoch": best, "best AUC": float(v[best]), "AUC at 10": float(v[10])}


def toy_epochs_figure(figsize=None, suptitle: bool = True):
    with _style():
        fig, axes = _new_fig(figsize or (8.0, 3.2), 1, 2)
        draw_toy_epochs(axes[0], axes[1])
        if suptitle:
            fig.suptitle("Ten epochs, made-up numbers", fontsize=9.5)
    return fig


def draw_applicants(ax) -> pd.DataFrame:
    """The twenty applicants: one row of ten squares per group, each showing
    its score; red = defaulted, blue = repaid."""
    for row, g in enumerate(("A", "B")):
        t = _AP[_AP["group"] == g]
        y = 1 - row
        for i, (score, bad) in enumerate(zip(t["score"], t["defaulted"])):
            ax.add_patch(Rectangle((i + 0.06, y + 0.08), 0.88, 0.84,
                                   facecolor=RED if bad else BLUE, edgecolor="white", lw=0.8))
            ax.text(i + 0.5, y + 0.5, f"{score:.2f}", ha="center", va="center", fontsize=9,
                    color="white")
        ax.text(-0.25, y + 0.5, f"group {g}", ha="right", va="center", fontsize=9.5)
    ax.text(5, -0.12, "red = defaulted, blue = repaid; the number = the model's score",
            ha="center", va="top", fontsize=8.5, color=MUTED)
    ax.set_xlim(-1.9, 10.05)
    ax.set_ylim(-0.55, 2.05)
    ax.set_axis_off()
    return applicants_numbers()


def dense_middle_numbers() -> pd.DataFrame:
    """[X15]: our GBDT's test p of the under-25s and the rest: the share above
    0.25 (declined) and the share between 0.20 and 0.30, in %."""
    _, X_te, _, _ = _split()
    p = _gbdt_p()
    young = (X_te["age"] < 25).to_numpy()
    rows = []
    for m, name in ((young, "under 25"), (~young, "25 and over")):
        q = p[m]
        rows.append({"group": name, "n": int(m.sum()), "median p": 100 * np.median(q),
                     "above 0.25 (declined)": 100 * (q > THRESHOLD).mean(),
                     "between 0.20 and 0.30": 100 * ((q > 0.20) & (q <= 0.30)).mean()})
    return pd.DataFrame(rows).set_index("group")


def draw_dense_middle(ax) -> pd.DataFrame:
    """Backup 15: the share of each age group by test p (our GBDT), with the
    line at 0.25: a group whose scores are shifted up loses more clients over
    the line."""
    _, X_te, _, _ = _split()
    p = _gbdt_p()
    young = (X_te["age"] < 25).to_numpy()
    bins = np.linspace(0, 0.8, 33)
    for m, name, colour, lw in ((young, "under 25", RED, 1.8),
                                (~young, "25 and over", DARK, 1.4)):
        h, _ = np.histogram(np.clip(p[m], 0, 0.7999), bins)
        above = _pct((p[m] > THRESHOLD).mean())
        ax.stairs(h / m.sum(), bins, color=colour, lw=lw, label=f"{name}: {above} above 0.25",
                  zorder=3 if name == "under 25" else 2)
    ax.axvline(THRESHOLD, color=NEAR_BLACK, ls="--", lw=1.0)
    ax.text(THRESHOLD + 0.01, 0.97, "decline above 0.25", transform=ax.get_xaxis_transform(),
            ha="left", va="top", fontsize=8.5, color=NEAR_BLACK)
    ax.set_xlim(0, 0.8)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
    ax.set_xlabel("p of default (our GBDT, test clients)")
    ax.set_ylabel("share of the group")
    ax.set_ylim(0, 0.17)
    ax.set_yticks([0, 0.05, 0.10, 0.15])
    ax.yaxis.set_major_formatter(b3.PCT0)
    ax.legend(loc="center right", fontsize=8.5, handlelength=1.5)
    return dense_middle_numbers()


def open_question_numbers() -> pd.DataFrame:
    """[Q]: the open practice question's made-up car insurer (2,000 test
    applicants by age): counts behind its rates, the gaps, the miss rates and
    the precision."""
    rows = []
    for name, n, base, flagged, far in (("under 30", 500, 0.20, 0.30, 0.15),
                                        ("30 and over", 1500, 0.10, 0.12, 0.06)):
        claims = round(n * base)
        flagged_n = round(n * flagged)
        false_flags = round(far * (n - claims))
        right_flags = flagged_n - false_flags
        rows.append({"group": name, "applicants": n, "claims": claims, "flagged": flagged_n,
                     "flagged, no claim": false_flags, "flagged, claim": right_flags,
                     "flag rate (%)": 100 * flagged, "base rate (%)": 100 * base,
                     "flag rate / base rate": flagged / base,
                     "miss rate (%)": 100 * (claims - right_flags) / claims,
                     "precision (%)": 100 * right_flags / flagged_n})
    t = pd.DataFrame(rows).set_index("group")
    t.attrs["flag gap"] = float(t["flag rate (%)"].iloc[0] - t["flag rate (%)"].iloc[1])
    t.attrs["base-rate gap"] = float(t["base rate (%)"].iloc[0] - t["base rate (%)"].iloc[1])
    return t


# ===========================================================================
# Homework A: the data and the soft checks
# ===========================================================================
HW_MEMBERS = pd.DataFrame({"x1": [0, 1, 3], "x2": [0, 2, 2]},
                          index=pd.Index(["X", "Y", "Z"], name="member"))
"""A1 (made up): x1 = visits per week, x2 = years as a member; w1 = -1, w2 = -0.5, b = 1."""
HW_EPOCHS = pd.DataFrame(
    {"training loss": [0.55, 0.48, 0.44, 0.41, 0.39, 0.37, 0.35, 0.33],
     "validation AUC": [0.70, 0.74, 0.76, 0.75, 0.75, 0.74, 0.73, 0.72]},
    index=pd.Index(range(1, 9), name="epoch"))
"""A4 (made up): eight epochs."""
HW_SCHEME = pd.DataFrame({"members": [20, 30], "late": [6, 6], "deposit": [8, 6],
                          "late among deposit": [5, 4]},
                         index=pd.Index(["under 30", "30 and over"], name="group"))
"""A5 (made up): a bike-rental scheme; the decision = asked for a deposit,
the event = returned a bike late."""

A6_MODEL_ANSWER = [
    "No: the late-return rates differ by age (30% against 20%), and other columns "
    "(for example the hour of the rentals, a student card) differ by age and stand in for "
    "it (proxies).",
    "Keep age outside the model and measure, after the change, the deposit rates, the error "
    "rates (false alarms and misses) and the precision by age.",
    "Ask whether the late-return gap itself is justified: where does it come from?"]


def homework_answers(item: str) -> dict:
    """The worked numbers of a homework item A1-A6 (for the solutions)."""
    item = str(item).strip().upper()
    if item == "A1":
        z = (1 - HW_MEMBERS["x1"] - 0.5 * HW_MEMBERS["x2"]).tolist()
        return {"z": z, "p": [float(sigmoid(v)) for v in z], "coefficients": [1, -1, -0.5]}
    if item == "A2":
        x1, x2 = _PT["x1"].to_numpy(), _PT["x2"].to_numpy()
        h1, h2 = np.maximum(0, x1 + x2 - 1), np.maximum(0, x1 + x2)
        z = -8 * h1 + 4 * h2 - 2
        p = sigmoid(z)
        return {"h1": h1.tolist(), "h2": h2.tolist(), "z": z.tolist(), "p": p.tolist(),
                "classes": (p > 0.5).astype(int).tolist(), "parameters": 9,
                "parameters_8": 5 * 8 + 8 + 8 + 1, "parameters_10_4": 5 * 10 + 10 + 10 * 4 + 4
                + 4 + 1}
    if item == "A3":
        out = {}
        for lr, key in ((0.1, "01"), (0.5, "05"), (1.5, "15")):
            ws = _lr_path(lr, 0.0, 2.0)[1:]
            out[f"w_{key}"] = ws
            out[f"loss_{key}"] = [(w - 2) ** 2 for w in ws]
        out.update({"slow": 0.1, "lowest": 0.5, "overshoots": 1.5, "steps": 6000 // 200})
        return out
    if item == "A4":
        rows, stop, best = _patience(HW_EPOCHS["validation AUC"].tolist(), 3)
        v = HW_EPOCHS["validation AUC"]
        return {"best_epoch": best, "stops_after": stop, "kept_epoch": best,
                "cost": round(float(v[best] - v[8]), 4)}
    if item == "A5":
        s = HW_SCHEME
        base = 100 * s["late"] / s["members"]
        dep = 100 * s["deposit"] / s["members"]
        fp = s["deposit"] - s["late among deposit"]
        fn = s["late"] - s["late among deposit"]
        return {"base_rate": base.tolist(), "deposit_rate": dep.tolist(),
                "deposit_gap": float(dep.iloc[0] - dep.iloc[1]),
                "base_gap": float(base.iloc[0] - base.iloc[1]),
                "difference": float((dep.iloc[0] - dep.iloc[1]) - (base.iloc[0] - base.iloc[1])),
                "false_alarm": (100 * fp / (s["members"] - s["late"])).tolist(),
                "miss": (100 * fn / s["late"]).tolist(),
                "precision": (100 * s["late among deposit"] / s["deposit"]).tolist()}
    if item == "A6":
        return {"model answer": A6_MODEL_ANSWER}
    raise ValueError(f"unknown homework item {item!r}: use A1-A6")


# (key, kind, what it is, hint when it is not right yet)
_CHECKS = {
    "A1": [("z", "numbers", "z of members X, Y, Z (a list)",
            "z = -1 × x1 - 0.5 × x2 + 1 for each member"),
           ("p", "numbers", "p of X, Y, Z (three decimals are enough)",
            "p = 1 / (1 + e^-z); the sigmoid table (lab.sigmoid_table()) has z = 1, -1, -3"),
           ("coefficients", "numbers", "b0, b1, b2 of the logistic regression with the same p",
            "the bias is the intercept b0, the weights are the coefficients")],
    "A2": [("h1", "numbers", "h1 at (0,0), (1,0), (0,1), (1,1)",
            "h1 = ReLU(x1 + x2 - 1): only (1,1) gives a positive sum"),
           ("h2", "numbers", "h2 at the four points", "h2 = ReLU(x1 + x2)"),
           ("z", "numbers", "z at the four points", "z = -8 × h1 + 4 × h2 - 2"),
           ("p", "numbers", "p at the four points", "p = sigmoid(z): z = -2 gives 0.119"),
           ("classes", "numbers", "the class at the four points (1 if p > 0.5)",
            "class 1 where p is above 0.5"),
           ("parameters", "int", "the number of parameters of this network",
            "hidden: 2 inputs × 2 units + 2 biases; output: 2 weights + 1 bias"),
           ("parameters_8", "int", "parameters with 5 inputs and one hidden layer of 8",
            "5 × 8 + 8 for the hidden layer, 8 + 1 for the output"),
           ("parameters_10_4", "int", "parameters with 5 inputs and hidden layers (10, 4)",
            "5 × 10 + 10, then 10 × 4 + 4, then 4 + 1")],
    "A3": [("w_01", "numbers", "w after steps 1-3 with learning rate 0.1",
            "new w = w - 0.1 × 2(w - 2), starting from w = 0"),
           ("loss_01", "numbers", "the loss after each of those steps", "loss = (w - 2)²"),
           ("w_05", "numbers", "w after steps 1-3 with learning rate 0.5",
            "new w = w - 0.5 × 2(w - 2)"),
           ("loss_05", "numbers", "the loss after each of those steps", "loss = (w - 2)²"),
           ("w_15", "numbers", "w after steps 1-3 with learning rate 1.5",
            "new w = w - 1.5 × 2(w - 2): the first step goes to 6"),
           ("loss_15", "numbers", "the loss after each of those steps", "loss = (w - 2)²"),
           ("slow", "number", "the slow learning rate", "the one whose steps are small"),
           ("lowest", "number", "the rate that reaches the lowest point",
            "the one that lands on w = 2"),
           ("overshoots", "number", "the rate that overshoots",
            "the one whose loss grows step by step"),
           ("steps", "int", "steps in one epoch: 6,000 rows in batches of 200",
            "rows / batch size")],
    "A4": [("best_epoch", "int", "the best epoch", "the epoch with the highest validation AUC"),
           ("stops_after", "int", "the epoch after which training stops (patience 3)",
            "count the epochs in a row that do not beat the best; stop when the count is 3"),
           ("kept_epoch", "int", "whose weights are kept (an epoch)",
            "early stopping keeps the best epoch's weights"),
           ("cost", "number", "the validation AUC lost by training to epoch 8",
            "the best validation AUC minus epoch 8's")],
    "A5": [("base_rate", "percents", "the base rates of under 30 and 30 and over (in %)",
            "late / members, in each group"),
           ("deposit_rate", "percents", "the deposit rates (in %)",
            "asked for a deposit / members, in each group"),
           ("deposit_gap", "number", "the deposit-rate gap (under 30 minus 30 and over, points)",
            "the difference of the two deposit rates"),
           ("base_gap", "number", "the base-rate gap (points)",
            "the difference of the two base rates"),
           ("difference", "number", "the deposit gap minus the base-rate gap (points)",
            "above 0: the rule enlarges the difference"),
           ("false_alarm", "percents", "the false-alarm rates (in %)",
            "asked but not late / all who were not late, in each group"),
           ("miss", "percents", "the miss rates (in %)",
            "late but not asked / all who were late, in each group"),
           ("precision", "percents", "the precision (in %)",
            "late among those asked / all asked, in each group")],
}


def _close(x, y) -> bool:
    return abs(float(x) - float(y)) <= max(0.02, 0.005 * abs(float(y))) + 1e-9


def _close_percent(given, g, e) -> bool:
    """A percent g (given as typed: given) against the expected e: close, or
    a whole number within 0.5 of e (62 or 63 for 62.5)."""
    return _close(g, e) or (float(given).is_integer() and abs(g - e) <= 0.5 + 1e-9)


def _as_list(given, expected):
    if isinstance(given, (dict, pd.Series)):
        given = list(dict(given).values())
    return list(np.ravel(given)), list(np.ravel(expected))


def _right(kind: str, given, expected) -> bool:
    if kind == "number":
        return _close(_as_number(given), expected)
    if kind in ("numbers", "percents"):
        g, e = _as_list(given, expected)
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
    return False


def check(item: str, **answers) -> None:
    """A soft check of a homework answer: one line per part, "right" or "not
    yet" with a hint. It never stops the notebook. check("A1") lists the
    parts it takes; leave a part as ... until you have it."""
    key = str(item).strip().upper()
    if key == "A6":
        if answers and not str(answers.get("answer", "")).strip():
            _say("A6: write your answer (3-4 sentences) between the quotes first, then run "
                 "the cell again.")
            return
        if answers:
            _say("A6 is a written answer - nothing to mark. Compare yours with the model "
                 "answer's three points:")
        else:
            _say("A6 is a written answer (3-4 sentences). The model answer's three points:")
        for k, point in enumerate(A6_MODEL_ANSWER, 1):
            _say(f"  {k}. {point}")
        return
    if key not in _CHECKS:
        _say(f"{item!r} is not one of the items A1-A6.")
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
        if kind in ("number", "int"):
            given = _decimal_comma(_plain(given), k)
        if given is ... or given is None:
            _say(f"{key} {k}: not filled in yet ({what}).")
            continue
        n_given += 1
        try:
            ok = _right(kind, given, expected[k])
        except Exception:
            comma = (f" - decimals take a dot: {given.replace(',', '.')}, not {given}"
                     if isinstance(given, str) and re.fullmatch(r"\s*-?\d+,\d+\s*", given)
                     else (" - write a list in square brackets, for example [0.1, 0.2]"
                           if isinstance(given, str) and "," in given else ""))
            _say(f"{key} {k}: could not read {given!r} - it should be {what}{comma}.")
            continue
        n_right += ok
        if not ok and kind in ("numbers", "percents") and isinstance(given, (list, tuple)):
            n_expected = len(np.ravel(expected[k]))
            if len(given) > n_expected:
                hint += " (decimals take a dot: [0.73, 0.27], not [0,73, 0,27])"
            elif len(given) < n_expected:
                hint += f" (the list needs {n_expected} values)"
        _say(f"{key} {k}: right." if ok else f"{key} {k}: not yet - {hint}.")
    if n_given:
        _say(f"{key}: {n_right} of {n_given} right.")
