"""
finance_data.py
================
Shared data loader for the Data Mining course (SGH, SMMD-ADA).

The whole course is anchored on a single **retail-credit / PD (probability of
default)** portfolio so that every technique is taught on the same, familiar
business problem. This module gives every lesson notebook one entry point:

    from finance_data import load_credit
    df = load_credit()

Design goals
------------
* **Runs anywhere** (student laptop or Google Colab), with no manual downloads.
* **Reproducible**: fixed random seed, deterministic output.
* **Realistic teaching traps built in**:
    - three flavours of missingness (MCAR / MAR / MNAR + *structural* NaNs),
    - one deliberate **leakage** column (`collections_contact`) that is only
      known *after* the outcome — used in the ensembles/evaluation block.
* **Honest about the target**: `default` is generated from a logistic model of
  the real drivers, so downstream models genuinely have signal to find.

Real data option
-----------------
By default we try to fetch the classic **German Credit (Statlog)** dataset from
OpenML (`fetch_openml('credit-g')`). If that fails (no internet, OpenML down),
we fall back to a synthetic Polish retail-credit portfolio. Both return the
same interface. For the course you can force either path:

    load_credit(source="synthetic")   # always synthetic, offline
    load_credit(source="openml")       # force real German Credit
    load_credit(source="auto")         # try OpenML, else synthetic  (default)
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

RANDOM_SEED = 20261006  # first class date, for luck

REGIONS = ["Mazowieckie", "Małopolskie", "Śląskie", "Wielkopolskie", "Pomorskie"]
PURPOSES = ["car", "education", "business", "consolidation", "furniture", "other"]
HOUSING = ["own", "rent", "other"]
CHECKING = ["none", "low", "medium", "high"]


def _synthetic_credit(n: int = 4000, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Generate a synthetic retail-credit portfolio with a defaultable target."""
    rng = np.random.default_rng(seed)

    age = rng.integers(21, 76, size=n)
    # monthly income (PLN), heavy right tail
    monthly_income = np.round(np.exp(rng.normal(8.5, 0.45, size=n)) * 10, -1)
    employment_length = np.clip(rng.gamma(2.0, 2.5, size=n), 0, 40).round(1)
    housing = rng.choice(HOUSING, size=n, p=[0.45, 0.42, 0.13])
    purpose = rng.choice(PURPOSES, size=n, p=[0.28, 0.10, 0.12, 0.25, 0.10, 0.15])
    region = rng.choice(REGIONS, size=n, p=[0.35, 0.15, 0.20, 0.18, 0.12])
    num_dependents = rng.integers(0, 6, size=n)
    num_existing_loans = rng.poisson(1.1, size=n).clip(0, 8)

    loan_amount = np.round(rng.uniform(2_000, 80_000, size=n), -2)
    loan_term = rng.choice([12, 24, 36, 48, 60], size=n, p=[0.15, 0.25, 0.3, 0.2, 0.1])

    credit_utilization = np.clip(rng.beta(2.0, 3.0, size=n) * 1.2, 0, 1.4).round(3)
    # Savings correlate with income (realistic, and it lets the EDA lab detect
    # the MNAR income-missingness through an observed proxy).
    savings_balance = np.round(
        np.exp(0.9 * np.log(monthly_income) - 1.9 + rng.normal(0, 0.8, size=n)), -1
    )
    checking_status = rng.choice(CHECKING, size=n, p=[0.20, 0.35, 0.30, 0.15])

    # debt-to-income: coupled to loan size, income and existing loans
    monthly_instalment = loan_amount / loan_term * (1 + 0.06 * loan_term / 12)
    dti = np.clip(
        (monthly_instalment + 300 * num_existing_loans) / np.maximum(monthly_income, 800),
        0, 3.0,
    ).round(3)

    # prior delinquency: ~65% never delinquent -> structural NaN below
    ever_delinquent = rng.random(n) < 0.35
    months_since_delinquency = np.where(
        ever_delinquent, rng.integers(1, 60, size=n).astype(float), np.nan
    )

    # ---- true generative model for default (logit) ----
    z = (
        -2.4
        + 2.6 * dti
        + 1.9 * credit_utilization
        + 0.28 * num_existing_loans
        - 0.000006 * monthly_income
        - 0.03 * employment_length
        + 0.9 * ever_delinquent
        + 0.4 * (housing == "rent")
        + 0.5 * (checking_status == "none")
        - 0.15 * (age - 40) / 15
        + rng.normal(0, 0.5, size=n)
    )
    p_default = 1 / (1 + np.exp(-z))
    default = (rng.random(n) < p_default).astype(int)

    # ---- LEAKAGE TRAP (Block 3): known only AFTER outcome ----
    # Collections contacts a customer mostly when they have already defaulted.
    collections_contact = np.where(
        default == 1, rng.random(n) < 0.85, rng.random(n) < 0.04
    ).astype(int)

    df = pd.DataFrame(
        {
            "customer_id": np.arange(1, n + 1),
            "age": age,
            "monthly_income_pln": monthly_income,
            "employment_length_years": employment_length,
            "housing": housing,
            "purpose": purpose,
            "region": region,
            "num_dependents": num_dependents,
            "num_existing_loans": num_existing_loans,
            "loan_amount_pln": loan_amount,
            "loan_term_months": loan_term,
            "credit_utilization": credit_utilization,
            "debt_to_income": dti,
            "savings_balance_pln": savings_balance,
            "checking_status": checking_status,
            "months_since_last_delinquency": months_since_delinquency,
            "collections_contact": collections_contact,  # <-- leakage, drop for modelling
            "default": default,
        }
    )

    # ---- inject missingness (AFTER target is fixed) ----
    # 1) MNAR: high earners decline to state income
    p_miss_income = 0.05 + 0.25 * (df["monthly_income_pln"] > df["monthly_income_pln"].quantile(0.75))
    df.loc[rng.random(n) < p_miss_income, "monthly_income_pln"] = np.nan

    # 2) MAR: employment length missing more often for renters
    p_miss_emp = np.where(df["housing"] == "rent", 0.16, 0.04)
    df.loc[rng.random(n) < p_miss_emp, "employment_length_years"] = np.nan

    # months_since_last_delinquency is already *structurally* NaN (never delinquent)

    return df


def _openml_credit() -> pd.DataFrame:
    """Fetch German Credit (Statlog) from OpenML and rename to a friendly schema."""
    from sklearn.datasets import fetch_openml

    bunch = fetch_openml("credit-g", version=1, as_frame=True)
    df = bunch.frame.copy()
    # OpenML target 'class' in {'good','bad'} -> default = 1 if 'bad'
    df["default"] = (df["class"].astype(str) == "bad").astype(int)
    df = df.drop(columns=["class"])
    return df


def load_credit(source: str = "auto", n: int = 4000, verbose: bool = True) -> pd.DataFrame:
    """Load the course credit portfolio.

    Parameters
    ----------
    source : {"auto", "synthetic", "openml"}
        "auto" tries OpenML then falls back to synthetic (default).
    n : int
        Number of rows for the synthetic portfolio.
    verbose : bool
        Print which source was actually used.
    """
    if source == "openml":
        return _openml_credit()
    if source == "synthetic":
        if verbose:
            print(f"Loaded SYNTHETIC credit portfolio: {n} rows.")
        return _synthetic_credit(n=n)

    # auto
    try:
        df = _openml_credit()
        if verbose:
            print("Loaded REAL German Credit (Statlog) from OpenML.")
        return df
    except Exception as exc:  # offline / OpenML down
        if verbose:
            print(f"OpenML unavailable ({type(exc).__name__}); using synthetic portfolio.")
        return _synthetic_credit(n=n)


# =====================================================================
#  Real-world modelling portfolio: UCI "Default of Credit Card Clients"
#  (Taiwan, 2005; Yeh & Lien). Used from Block 2 onward.
# =====================================================================
_TAIWAN_COLUMNS = {
    "x1": "limit_bal", "x2": "sex", "x3": "education", "x4": "marriage",
    "x5": "age",
    "x6": "pay_status_sep", "x7": "pay_status_aug", "x8": "pay_status_jul",
    "x9": "pay_status_jun", "x10": "pay_status_may", "x11": "pay_status_apr",
    "x12": "bill_amt_sep", "x13": "bill_amt_aug", "x14": "bill_amt_jul",
    "x15": "bill_amt_jun", "x16": "bill_amt_may", "x17": "bill_amt_apr",
    "x18": "pay_amt_sep", "x19": "pay_amt_aug", "x20": "pay_amt_jul",
    "x21": "pay_amt_jun", "x22": "pay_amt_may", "x23": "pay_amt_apr",
}

_TAIWAN_CACHE = Path(__file__).resolve().parent.parent / "data" / "taiwan_credit.csv"


def load_taiwan(verbose: bool = True) -> pd.DataFrame:
    """Load the real Taiwan credit-card default portfolio (30,000 clients).

    Downloads from OpenML on first use and caches to ``data/taiwan_credit.csv``
    (the cache is committed to the course repo, so students work offline).
    One TEACHING TRAP is injected and documented: ``collections_contact`` is
    generated from the outcome, exactly like in the synthetic portfolio -
    known only after default, never available at decision time.
    """
    if _TAIWAN_CACHE.exists():
        df = pd.read_csv(_TAIWAN_CACHE)
        if verbose:
            print(f"Loaded REAL Taiwan credit portfolio from cache: {df.shape[0]} rows.")
        return df

    from sklearn.datasets import fetch_openml

    bunch = fetch_openml("default-of-credit-card-clients", version=1,
                         as_frame=True)
    df = bunch.frame.rename(columns=_TAIWAN_COLUMNS)
    df["default"] = bunch.target.astype(int).to_numpy()
    df = df.drop(columns=[c for c in df.columns if c.startswith("y")],
                 errors="ignore")
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ---- injected teaching trap (documented; same recipe as synthetic) ----
    rng = np.random.default_rng(20261013)  # Block 2 class date
    df["collections_contact"] = np.where(
        df["default"] == 1, rng.random(len(df)) < 0.85,
        rng.random(len(df)) < 0.04).astype(int)

    _TAIWAN_CACHE.parent.mkdir(exist_ok=True)
    df.to_csv(_TAIWAN_CACHE, index=False)
    if verbose:
        print(f"Downloaded REAL Taiwan credit portfolio: {df.shape[0]} rows "
              f"(cached to {_TAIWAN_CACHE.name}).")
    return df


def taiwan_features(df: pd.DataFrame) -> pd.DataFrame:
    """The course's shared feature matrix for the Taiwan portfolio.

    Mixes raw application-time columns with the engineered features built in
    Blocks 1-2 (ratios, flags): utilisation, payment ratio, delay counters.
    ``collections_contact`` is deliberately NOT included - Block 3 detonates
    it on purpose.
    """
    pay_cols = [c for c in df.columns if c.startswith("pay_status_")]
    bill1, limit = df["bill_amt_sep"], df["limit_bal"]
    out = pd.DataFrame(
        {
            "limit_bal": df["limit_bal"],
            "age": df["age"],
            "female": (df["sex"] == 2).astype(int),
            "education": df["education"],
            "married": (df["marriage"] == 1).astype(int),
            "pay_delay_recent": df["pay_status_sep"],
            "months_delayed_6m": (df[pay_cols] > 0).sum(axis=1),
            "utilisation": (bill1 / limit).clip(-0.5, 2.0),
            "pay_ratio": (df["pay_amt_sep"] / bill1.clip(lower=1)).clip(0, 2.0),
            "bill_trend": ((bill1 - df["bill_amt_apr"]) / limit).clip(-2, 2),
        }
    )
    return out


# =====================================================================
#  Block 5 add-ons: product cross-holding baskets & complaint corpus
# =====================================================================
PRODUCTS = [
    "checking", "savings", "debit_card", "credit_card", "personal_loan",
    "mortgage", "home_insurance", "car_loan", "brokerage", "term_deposit",
]

BASKET_SEED = 20261103  # Block 5 class date


def load_baskets(n: int = 4000, seed: int = BASKET_SEED) -> pd.DataFrame:
    """Product cross-holding baskets for the same synthetic customers.

    Returns a one-hot DataFrame (customer_id index, one column per product).
    Propensities depend on the customer profile, with a few deliberately
    strong co-holding patterns for the association-rules lab:
      * mortgage -> home_insurance (bundled at sale),
      * brokerage -> term_deposit (affluent savers),
      * credit_card + personal_loan among high-DTI customers.
    """
    df = _synthetic_credit(n=n)  # same seed => same customers as load_credit
    rng = np.random.default_rng(seed)

    income = df["monthly_income_pln"].fillna(df["monthly_income_pln"].median())
    rich = (income > income.quantile(0.75)).to_numpy()
    young = (df["age"] < 35).to_numpy()
    stretched = (df["debt_to_income"] > df["debt_to_income"].quantile(0.75)).to_numpy()

    p = {
        "checking": np.full(n, 0.92),
        "savings": 0.45 + 0.25 * rich,
        "debit_card": 0.55 + 0.25 * young,
        "credit_card": 0.35 + 0.25 * stretched + 0.10 * rich,
        "personal_loan": 0.20 + 0.35 * stretched,
        "mortgage": 0.18 + 0.15 * rich - 0.10 * young,
        "car_loan": np.full(n, 0.15),
        "brokerage": 0.06 + 0.30 * rich,
        "term_deposit": 0.10 + 0.15 * rich,
    }
    basket = {name: rng.random(n) < prob for name, prob in p.items()}

    # deliberate association patterns
    mort = basket["mortgage"]
    basket["home_insurance"] = np.where(
        mort, rng.random(n) < 0.85, rng.random(n) < 0.08)
    brok = basket["brokerage"]
    basket["term_deposit"] = np.where(
        brok, rng.random(n) < 0.65, basket["term_deposit"])

    out = pd.DataFrame({k: np.asarray(v, dtype=bool) for k in PRODUCTS
                        for v in [basket[k]]},
                       index=df["customer_id"])
    out.index.name = "customer_id"
    return out


COMPLAINT_CATEGORIES = ["card_fraud", "mortgage", "fees", "app_service",
                        "collections"]

_COMPLAINT_TEMPLATES = {
    "card_fraud": [
        "someone {verb} my {card} card and made {n} unauthorized {txn} at a {place} i never visited please block the card and refund the money",
        "i noticed unauthorized {txn} on my {card} card statement worth {amt} pln i did not make these payments this is clearly fraud",
        "my {card} card was {verb} and used online for {txn} i reported it immediately but the charges are still on my account",
        "there are fraudulent {txn} on my card from {place} i want a chargeback and a new card issued",
    ],
    "mortgage": [
        "my mortgage installment {verb2} by {amt} pln without clear explanation the schedule i signed said something different",
        "i asked for early repayment of my mortgage and the {fee} you quoted contradicts the loan agreement",
        "the bank recalculated my mortgage margin after the promotional period and nobody informed me about the new rate",
        "i submitted all documents for the mortgage {doc} three weeks ago and still have no decision the property seller is losing patience",
    ],
    "fees": [
        "i was charged a {fee} of {amt} pln on my account even though the price list says this service is free",
        "you doubled the {fee} on my account without notice i want a refund and an explanation",
        "there is a recurring {amt} pln {fee} on my statement that nobody can explain to me at the branch",
        "the advertised promotion said no {fee} for the first year yet i keep being charged every month",
    ],
    "app_service": [
        "the mobile app {verb3} every time i try to log in since the last update i cannot even check my balance",
        "i waited {n} days for anyone to answer my message in the app and the hotline disconnects after twenty minutes",
        "the transfer i made in the app {verb3} with an error but the money left my account anyway",
        "your branch closed early and the app was down so i could not authorize the payment on time",
    ],
    "collections": [
        "i keep receiving collection calls about a loan installment i already paid the confirmation number is attached",
        "your collections department calls me {n} times a day even at work this is harassment not communication",
        "i received a debt collection letter with extra {fee} for a delay caused by your own system outage",
        "the collector threatened to visit my employer although my repayment plan was approved last month",
    ],
}

_SLOTS = {
    "verb": ["stole", "skimmed", "copied", "cloned"],
    "verb2": ["increased", "jumped", "went up"],
    "verb3": ["crashes", "freezes", "fails"],
    "card": ["credit", "debit"],
    "txn": ["transactions", "payments", "purchases", "withdrawals"],
    "place": ["petrol station", "shop abroad", "website", "casino"],
    "amt": ["120", "250", "480", "999", "1500"],
    "n": ["three", "four", "five", "seven"],
    "fee": ["maintenance fee", "commission", "handling fee", "penalty fee"],
    "doc": ["application", "annex", "refinancing"],
}


_FILLERS = [
    "this is not what i expect from my bank",
    "i have been a loyal customer for over ten years",
    "please resolve this quickly or i will close my account and move to another bank",
    "nobody at the hotline or the branch could help me with this",
    "i already sent two messages through the app and got no reply",
    "my account and my money deserve better treatment than this",
]


def _fill(template: str, rng) -> str:
    text = template
    for slot, options in _SLOTS.items():
        while "{" + slot + "}" in text:
            text = text.replace(
                "{" + slot + "}", options[rng.integers(len(options))], 1)
    return text


def load_complaints(n_per_class: int = 240, seed: int = BASKET_SEED + 1
                    ) -> pd.DataFrame:
    """Synthetic consumer-complaint corpus with category labels.

    Template-based with random slot filling, generic filler sentences, and
    occasional cross-topic asides -- so a TF-IDF + linear model does well
    but not perfectly, leaving a confusion matrix worth discussing.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for cat in COMPLAINT_CATEGORIES:
        temps = _COMPLAINT_TEMPLATES[cat]
        for _ in range(n_per_class):
            text = _fill(temps[rng.integers(len(temps))], rng)
            if rng.random() < 0.5:  # generic filler
                text += " " + _FILLERS[rng.integers(len(_FILLERS))]
            if rng.random() < 0.25:  # cross-topic aside (real complaints ramble)
                other = COMPLAINT_CATEGORIES[rng.integers(len(COMPLAINT_CATEGORIES))]
                if other != cat:
                    aside = _fill(
                        _COMPLAINT_TEMPLATES[other][
                            rng.integers(len(_COMPLAINT_TEMPLATES[other]))],
                        rng)
                    text += " and while we are at it " + aside
            rows.append({"text": text, "category": cat})
    out = pd.DataFrame(rows)
    return out.sample(frac=1, random_state=seed).reset_index(drop=True)


if __name__ == "__main__":
    d = load_credit(source="synthetic")
    print(d.shape)
    print(d["default"].mean().round(3), "default rate")
    print(d.isna().mean().round(3).sort_values(ascending=False).head())
    b = load_baskets()
    print("baskets:", b.shape, "| penetration:",
          b.mean().round(2).to_dict())
    c = load_complaints()
    print("complaints:", c.shape, c["category"].value_counts().to_dict())
