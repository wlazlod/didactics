# Data Mining — SGH Warsaw School of Economics (SMMD-ADA)

**A finance-first, problem-driven Data Mining course.**
Autumn semester · Tuesdays 17:10–20:30 · 7 blocks · taught in English.

This repository holds everything you need for the course: the slide decks, a
102-page coursebook that goes deeper than the slides, and one hands-on Jupyter
notebook per class.

---

## Schedule

| # | Date | Topic | Lab notebook |
|:-:|---|---|---|
| 1 | 06.10 | Data mining & CRISP-DM; EDA; data quality; missingness | `01_eda_missingness` |
| 2 | 13.10 | Linear & logistic regression; regularisation; intro to evaluation | `02_regression_baselines` |
| 3 | 20.10 | Trees; SVM interlude; ensembles (RF, GBDT); honest evaluation; leakage | `03_trees_ensembles_evaluation` |
| 4 | 27.10 | Clustering (k-means, hierarchical, DBSCAN); PCA | `04_clustering_pca` |
| 5 | 03.11 | Pattern discovery / Market Basket; text-mining basics | `05_pattern_mining_text` |
| 6 | 10.11 | Foundations of neural networks; responsible ML | `06_neural_networks` |
| 7 | 17.11 | Project presentations + written exam (term 0) | — |

---

## What's in this repository

```
data-mining/
├── main.pdf                   the COURSEBOOK — read it after each class; it
│                              carries much more explanation than the slides
├── lecture_1/ … lecture_6/    one folder per block:
│   ├── blockN.pdf               the slide deck
│   └── NN_<topic>.ipynb         the lab notebook (run it during/after class)
├── prerequisites/
│   └── 00_python_primer.ipynb   self-paced Python warm-up — start here if
│                                your Python is rusty
└── data/                      the course dataset (already included — the
                               notebooks load it automatically, offline)
```

Every deck ends with the same sequence: lab briefing, a "block in five
sentences" summary, project status, **two practice exam questions**, and a
short reading list. The practice questions are worth your time: **one from
each block appears verbatim on the exam.**

---

## Getting started

You need Python 3.11+ and Jupyter. The recommended path uses
[`uv`](https://docs.astral.sh/uv/):

```bash
git clone <this-repo>
cd didactics/data-mining
uv sync                                   # creates the environment
cd lecture_1
uv run jupyter lab 01_eda_missingness.ipynb
```

Plain `pip` works too:

```bash
pip install pandas numpy scikit-learn matplotlib jupyter mlxtend
```

**Google Colab:** the notebooks run there unmodified. Upload
`lecture_1/finance_data.py` next to the notebook (it is the shared data
loader) and adjust the `sys.path.append("../lecture_1")` line at the top.

**Rule of thumb for every lab:** run cells top to bottom, and before you trust
any result — *Kernel → Restart & Run All*.

---

## Assessment

**Project 50% + written exam 50%. You pass with ≥ 60% combined.**

### The project

In a **team of 3**, run a full mini CRISP-DM cycle on a dataset of your
choice: business framing → data preparation → **at least two model families**
→ correct evaluation → interpretation → recommendation. It ends with a
**presentation of results in Block 7** (~6–8 minutes + questions).

**Hard deadlines:**

| Deadline | What is due |
|---|---|
| **before Block 2** (13.10, 17:10) | teams of 3 formed and registered |
| **before Block 3** (20.10, 17:10) | topic + a short written description (the decision, the dataset, the target, the planned two model families). No idea? **Consult before the deadline.** |
| **before Block 7** (17.11, 17:10) | **upload all code + the final presentation** — what you upload is what you present |
| **Block 7** (17.11) | presentation of results |

**Data:** your choice — any real dataset that can carry the full cycle: a
meaningful business decision, a predictable target, enough rows for an honest
split (Kaggle, UCI, open government data…). Declared and approved via the
topic description.

**Grading** (the weights tell you where to spend your time):
framing 10% · data prep & missingness 15% · modelling (≥2 families) 15% ·
**evaluation, leakage-aware 20%** · interpretation & recommendation 15% ·
reproducibility 5% · **presentation & defence 20%**.

**Use of AI:** assistants are allowed — but **you are responsible for every
line you submit**. The stronger the impression that the project was produced
solely by AI, the harder the questions at the defence. Expect to explain,
justify, and modify any part of your code live.

**A sensible week-by-week plan** lives on the project slides in Block 1 —
each week's class teaches exactly what that week's project step needs.

### The exam (term 0)

Second part of Block 7, **120 minutes**: ~40% multiple choice, ~60% short
open questions. It tests *selecting the right method for a described problem*
and *interpreting a given result* — not derivations. Best study kit: the six
"Block N in five sentences" slides, the worked-example slides, and the
per-block practice questions (one from each block is on the paper).

---

## Reading

Each deck closes with a short block-specific reading list built around
classic, mostly freely available articles. Course-wide references:

- James, Witten, Hastie, Tibshirani — *An Introduction to Statistical
  Learning* (free PDF at [statlearning.com](https://www.statlearning.com/))
- Aurélien Géron — *Hands-On Machine Learning with Scikit-Learn, Keras &
  TensorFlow*
- the [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
  — genuinely good prose, not just API docs

---

## Questions?

Ask during class or in the break, or by e-mail. For project-topic
consultations, ask **before** the Block 3 deadline — that is what they are
for.
