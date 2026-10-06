"""
check_setup.py
==============
Checks that your computer is ready for the Data Mining labs, and says what to
do if it is not.  Run it from the ``data-mining`` folder:

    uv run python check_setup.py        (the recommended uv setup)
    python check_setup.py               (pip or conda: activate the environment first)
    !python check_setup.py              (Google Colab, after downloading this file)

It changes nothing on your computer.  The last line says either ALL GOOD or
what to fix; the step numbers refer to the setup guide
(setup_guide.pdf on MS Teams).
"""

# Plain syntax only (no f-strings), so that even a very old Python reaches the
# version check below instead of failing with a SyntaxError.
import os
import platform
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROBLEMS = []
WARNINGS = []


def ok(msg):
    print("  [ OK ] " + msg)


def warn(msg, hint):
    print("  [WARN] " + msg)
    print("         -> " + hint)
    WARNINGS.append(msg)


def fail(msg, hint):
    print("  [FAIL] " + msg)
    print("         -> " + hint)
    PROBLEMS.append(msg)


def section(title):
    print("")
    print(title)


IN_COLAB = "google.colab" in sys.modules or "COLAB_RELEASE_TAG" in os.environ

print("Data Mining - setup check")
print("  Python:  " + sys.version.split()[0] + "  (" + sys.executable + ")")
print("  System:  " + platform.platform())
print("  Folder:  " + HERE)

# ---------------------------------------------------------------------------
section("1. Python")
if sys.version_info < (3, 11):
    fail("Python %d.%d is too old; the course needs 3.11 or newer."
         % sys.version_info[:2],
         "Run the check through uv: 'uv run python check_setup.py', not "
         "'python check_setup.py' (setup guide, steps 3 and 7).")
    print("")
    print("RESULT: fix the problem above, then run this check again.")
    sys.exit(1)
ok("Python %d.%d" % sys.version_info[:2])

venv = os.path.join(HERE, ".venv")
in_course_venv = os.path.abspath(sys.prefix).startswith(os.path.abspath(venv))
if IN_COLAB:
    ok("running in Google Colab")
elif in_course_venv:
    ok("running in the course environment (.venv)")
    if sys.version_info[:2] != (3, 12):
        warn("the course environment uses Python %d.%d, not 3.12" % sys.version_info[:2],
             "Delete the .venv folder and run 'uv sync' again (setup guide, step 6).")
elif os.path.isdir(venv):
    warn("a course .venv exists, but this is a different Python",
         "Start everything with 'uv run ...' (e.g. 'uv run python check_setup.py', "
         "'uv run jupyter lab'), or in VS Code pick the .venv kernel.")
else:
    warn("not running in the course's uv environment",
         "Fine for pip/conda, but numbers may differ slightly from the slides.")

# ---------------------------------------------------------------------------
section("2. Course folder")
if IN_COLAB:
    ok("Colab: skipping folder checks")
else:
    expected = ["pyproject.toml", "uv.lock",
                os.path.join("lecture_1", "finance_data.py"),
                os.path.join("data", "taiwan_credit.csv")]
    missing = [p for p in expected if not os.path.exists(os.path.join(HERE, p))]
    if missing:
        fail("missing in this folder: " + ", ".join(missing),
             "Run this in the 'data-mining' folder of your course copy (setup guide, step 5).")
    else:
        ok("all course files present")

    low = HERE.lower()
    for word, name in (("onedrive", "OneDrive"), ("dropbox", "Dropbox"),
                       ("google drive", "Google Drive"), ("mobile documents", "iCloud Drive"),
                       ("icloud", "iCloud Drive")):
        if word in low:
            warn("the folder is inside %s" % name,
                 "Syncing the thousands of files in .venv causes errors and slowness; "
                 "download the course again into your home folder (setup guide, step 4).")
            break
    try:
        HERE.encode("ascii")
    except UnicodeEncodeError:
        warn("the folder path contains non-English letters (e.g. Polish characters)",
             "Usually fine. If anything below fails oddly, move the course to a "
             "plain path such as C:\\dm or ~/dm and run 'uv sync' again.")

# ---------------------------------------------------------------------------
section("3. Packages")
locked = {}
lock_path = os.path.join(HERE, "uv.lock")
if os.path.exists(lock_path):
    with open(lock_path, encoding="utf-8") as fh:
        for name, version in re.findall(r'\[\[package\]\]\nname = "([^"]+)"\nversion = "([^"]+)"',
                                        fh.read()):
            locked.setdefault(name, set()).add(version)

packages = [("numpy", "numpy"), ("pandas", "pandas"), ("sklearn", "scikit-learn"),
            ("scipy", "scipy"), ("matplotlib", "matplotlib"), ("mlxtend", "mlxtend"),
            ("jupyterlab", "jupyterlab"), ("ipykernel", "ipykernel")]
if IN_COLAB:
    packages = [p for p in packages if p[0] not in ("jupyterlab",)]
off_lock = []
for module, dist in packages:
    try:
        mod = __import__(module)
    except Exception as exc:  # ImportError, or a broken binary
        hint = ("Run 'uv sync' in the data-mining folder (setup guide, step 6)."
                if not IN_COLAB else "Run '!pip install %s' in a cell." % dist)
        fail("%s cannot be imported (%s: %s)" % (dist, type(exc).__name__, exc), hint)
        continue
    version = getattr(mod, "__version__", "?")
    ok("%-13s %s" % (dist, version))
    if locked.get(dist) and version not in locked[dist]:
        off_lock.append("%s %s (course: %s)" % (dist, version, "/".join(sorted(locked[dist]))))
if off_lock:
    warn("versions differ from the course's: " + "; ".join(off_lock),
         "Fine for learning; the last digits of some results may differ from the slides. "
         "The uv setup gives the exact versions.")

# ---------------------------------------------------------------------------
section("4. Course data loader")
loader_dir = os.path.join(HERE, "lecture_1")
if IN_COLAB:
    loader_dir = os.getcwd()
sys.path.insert(0, loader_dir)
try:
    import finance_data
    ok("finance_data imported")
except Exception as exc:
    finance_data = None
    fail("finance_data cannot be imported (%s: %s)" % (type(exc).__name__, exc),
         "Keep the folder structure intact: lecture_1/finance_data.py must sit next to "
         "the other lecture folders. In Colab, download it first (see the FAQ).")

if finance_data is not None and not PROBLEMS:
    try:
        credit = finance_data.load_credit(source="synthetic", verbose=False)
        ok("Block 1 portfolio: %d rows" % len(credit))
        taiwan = finance_data.load_taiwan(verbose=False)
        ok("Taiwan portfolio (Blocks 2-4, 6): %d rows" % len(taiwan))
        baskets = finance_data.load_baskets()
        ok("Block 5 baskets: %d rows" % len(baskets))
    except Exception as exc:
        fail("loading the course data failed (%s: %s)" % (type(exc).__name__, exc),
             "Run 'git pull' in the course folder (setup guide, 'Getting updates'); "
             "the 'data' folder must be next to 'lecture_1'.")

# ---------------------------------------------------------------------------
section("5. A tiny model and a plot")
if not PROBLEMS:
    try:
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        rng = np.random.default_rng(0)
        X = rng.normal(size=(200, 3))
        y = (X[:, 0] + rng.normal(size=200) > 0).astype(int)
        acc = LogisticRegression().fit(X, y).score(X, y)
        ok("scikit-learn fits a model (accuracy %.2f)" % acc)
        import matplotlib
        matplotlib.use("Agg")  # draw off-screen; Jupyter uses its own backend
        import matplotlib.pyplot as plt
        import io
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        fig.savefig(io.BytesIO(), format="png")
        plt.close(fig)
        ok("matplotlib draws a figure")
    except Exception as exc:
        fail("a basic computation failed (%s: %s)" % (type(exc).__name__, exc),
             "Delete the .venv folder and run 'uv sync' again; if it persists, "
             "send this whole output to the instructor.")
else:
    print("  (skipped until the problems above are fixed)")

# ---------------------------------------------------------------------------
section("6. Internet (needed only by the Python primer)")
try:
    from urllib.request import urlopen
    urlopen("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv",
            timeout=8).read(100)
    ok("can download the primer's example data")
except Exception as exc:
    warn("cannot reach raw.githubusercontent.com (%s)" % type(exc).__name__,
         "Only the primer's Titanic example needs this; the course labs run offline. "
         "Try another network (e.g. a phone hotspot) for the primer.")

# ---------------------------------------------------------------------------
print("")
if PROBLEMS:
    print("RESULT: %d problem(s) to fix - see the [FAIL] lines above." % len(PROBLEMS))
    sys.exit(1)
elif WARNINGS:
    print("RESULT: ALL GOOD (with %d warning(s) worth reading above)." % len(WARNINGS))
else:
    print("RESULT: ALL GOOD - you are ready for the labs.")
