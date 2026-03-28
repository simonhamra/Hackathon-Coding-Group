#!/usr/bin/env python3
"""
SS vs LL choice: logistic regression with subject-level evaluation.

By default reads the **entire** CSV (only the columns needed for modeling).
Use ``--nrows N`` for a quick subset while developing.

  python choice_model_fast.py
  python choice_model_fast.py --cv-folds 5              # robust AUC / Brier (grouped by subject)
  python choice_model_fast.py --paper                   # control for source study (paper)
  python choice_model_fast.py --premium-per-day         # extra $ per day of delay (interpretable)
  python choice_model_fast.py --no-diffs                # drop diffs (less collinearity, clearer coefs)
  python choice_model_fast.py --context --paper --cv-folds 5
  python choice_model_fast.py --out-dir ./model_outputs
  python choice_model_fast.py --perm-importance         # model-agnostic importance on hold-out

Requires: pandas, numpy, scikit-learn

Assumptions easy to forget (we print reminders at the end):
  • ``value_diff`` / ``time_diff_days`` are algebraically tied to SS/LL amounts and delays —
    coefficients are *partial* (holding others fixed), not independent “causal” effects.
  • Multiple trials per subject ⇒ a single train/test split is noisy; use ``--cv-folds`` for
    stability when reporting performance.
  • ``--paper`` absorbs cross-study heterogeneity so numeric coefs are easier to pitch as
    “within-study structure, controlling for which experiment produced the row.”
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# CSV columns to load (keeps memory down vs reading every column in the file)
# ---------------------------------------------------------------------------
BASE_USECOLS = [
    "choice",
    "subj_ident",
    "ss_value",
    "ss_time",
    "ll_value",
    "ll_time",
    "value_diff",
    "time_diff_days",
    "procedure",
    "incentivization",
    "online_study",
]

EXTRA_USECOLS = ["age", "rt"]

TIER_A = [
    "ss_value",
    "ss_time",
    "ll_value",
    "ll_time",
    "value_diff",
    "time_diff_days",
]


def load_head(path: Path, nrows: int | None, usecols: list[str]) -> pd.DataFrame:
    """Load CSV with only ``usecols``; ``nrows=None`` reads the full file."""
    t0 = time.perf_counter()
    header = pd.read_csv(path, nrows=0).columns.tolist()
    cols = [c for c in usecols if c in header]
    if "choice" not in cols or "subj_ident" not in cols:
        sys.exit(f"CSV missing required columns. Have: {cols}")

    kw: dict = dict(usecols=cols, low_memory=False, engine="c")
    if nrows is not None:
        kw["nrows"] = nrows
        print(f"Reading up to {nrows:,} rows, {len(cols)} columns ...", flush=True)
    else:
        print(f"Reading ALL rows ({len(cols)} columns) — may take a while ...", flush=True)

    df = pd.read_csv(path, **kw)
    print(f"Loaded {len(df):,} rows in {time.perf_counter() - t0:.1f}s.", flush=True)
    return df


def add_reward_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """LL / SS payout ratio."""
    out = df.copy()
    ss = pd.to_numeric(out["ss_value"], errors="coerce")
    ll = pd.to_numeric(out["ll_value"], errors="coerce")
    out["reward_ratio"] = np.where(ss.fillna(0) > 0, ll / ss, np.nan)
    return out


def add_premium_per_day(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extra delayed reward per calendar day of waiting: value_diff / time_diff_days.

    Business-facing read: “how much more do you get for each day you wait?”
    Denominator clipped to avoid divide-by-zero on identical SS/LL dates.
    """
    out = df.copy()
    vd = pd.to_numeric(out["value_diff"], errors="coerce")
    td = pd.to_numeric(out["time_diff_days"], errors="coerce")
    td_safe = td.clip(lower=np.finfo(float).eps)
    out["premium_per_day"] = vd / td_safe
    return out


def tune_threshold(y_true: np.ndarray, proba: np.ndarray, *, prefer: str = "accuracy") -> tuple[float, float]:
    """Grid-search a probability cutoff on **training** predictions only."""
    best_t, best_score = 0.5, -1.0
    for t in np.arange(0.20, 0.801, 0.01):
        pred = (proba >= t).astype(np.int8)
        if prefer == "balanced":
            s = balanced_accuracy_score(y_true, pred)
        else:
            s = accuracy_score(y_true, pred)
        if s > best_score:
            best_score, best_t = s, float(t)
    return best_t, best_score


def add_categorical_dummies(df: pd.DataFrame, colnames: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """One-hot encode named columns; missing → category ``missing``."""
    present = [c for c in colnames if c in df.columns]
    if not present:
        return df, []
    d = df.copy()
    for c in present:
        d[c] = d[c].astype(str).str.strip().replace({"nan": "missing"}).fillna("missing")
    dummies = pd.get_dummies(d[present], prefix=present, drop_first=False)
    out = pd.concat([d, dummies], axis=1)
    return out, list(dummies.columns)


def build_pipeline(feature_cols: list[str], seed: int, balance: bool) -> Pipeline:
    """Median impute + scale numeric block (all features passed as numeric matrix)."""
    prep = ColumnTransformer(
        [("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), feature_cols)],
        remainder="drop",
    )
    cw = "balanced" if balance else None
    clf = LogisticRegression(
        max_iter=5000,
        solver="lbfgs",
        random_state=seed,
        class_weight=cw,
    )
    return Pipeline([("prep", prep), ("clf", clf)])


def run_grouped_cv(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    feature_cols: list[str],
    *,
    n_splits: int,
    seed: int,
    balance: bool,
) -> dict[str, np.ndarray]:
    """Subject-level K-fold: each person appears in exactly one test fold."""
    gkf = GroupKFold(n_splits=n_splits)
    aucs, loglosses, briers, accs, baccs = [], [], [], [], []
    for tr, te in gkf.split(X, y, groups):
        pipe = build_pipeline(feature_cols, seed, balance)
        pipe.fit(X.iloc[tr], y[tr])
        proba = pipe.predict_proba(X.iloc[te])[:, 1]
        y_te = y[te]
        aucs.append(roc_auc_score(y_te, proba))
        loglosses.append(log_loss(y_te, proba))
        briers.append(brier_score_loss(y_te, proba))
        pred05 = (proba >= 0.5).astype(np.int8)
        accs.append(accuracy_score(y_te, pred05))
        baccs.append(balanced_accuracy_score(y_te, pred05))
    return {
        "auc": np.array(aucs),
        "log_loss": np.array(loglosses),
        "brier": np.array(briers),
        "accuracy": np.array(accs),
        "balanced_accuracy": np.array(baccs),
    }


def print_interpretation_caveats(*, used_diffs: bool, used_paper: bool) -> None:
    print("\n--- Interpretation caveats (for slides / appendix) ---", flush=True)
    if used_diffs:
        print(
            "• value_diff and time_diff_days are linear functions of SS/LL amounts and delays — "
            "logistic coefficients are partial effects with strong multicollinearity; "
            "prefer directional stories or try --no-diffs for a sparser, clearer model.",
            flush=True,
        )
    else:
        print("• Dropped explicit diff columns; coefs on SS/LL levels + reward_ratio are easier to narrate.", flush=True)
    if used_paper:
        print(
            "• Paper dummies absorb average differences between studies; numeric features are "
            "interpreted within the mix of studies, not as a single lab’s effect.",
            flush=True,
        )
    print(
        "• Trial-level fit does not model random subject intercepts; coefficients are "
        "population-average associations, not individual-level utility parameters.",
        flush=True,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Logistic choice model: grouped evaluation + interpretable options.")
    p.add_argument("--csv", type=Path, default=Path("clean_data_basic.csv"))
    p.add_argument(
        "--nrows",
        type=int,
        default=0,
        help="0 = entire CSV. N > 0 = first N rows only.",
    )
    p.add_argument("--context", action="store_true", help="One-hot procedure / incentivization / online_study.")
    p.add_argument(
        "--paper",
        action="store_true",
        help="One-hot source study (column 'paper' if present) — controls cross-study heterogeneity.",
    )
    p.add_argument(
        "--premium-per-day",
        action="store_true",
        help="Add premium_per_day = value_diff / time_diff_days (interpretable economic slope).",
    )
    p.add_argument(
        "--no-diffs",
        action="store_true",
        help="Drop value_diff and time_diff_days from features (reduces collinearity; clearer partial coefs).",
    )
    p.add_argument(
        "--cv-folds",
        type=int,
        default=0,
        help="If >= 2, run grouped K-fold CV by subject and print mean ± std (AUC, log-loss, Brier, acc).",
    )
    p.add_argument("--test-size", type=float, default=0.2, help="Fraction of subjects in hold-out (single split).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--extras", action="store_true", help="Add age and log(1+rt).")
    p.add_argument("--balance", action="store_true", help="class_weight='balanced'.")
    p.add_argument("--tune-threshold", action="store_true", help="Tune probability threshold on train only.")
    p.add_argument("--tune-balanced", action="store_true", help="With --tune-threshold, optimize balanced accuracy.")
    p.add_argument(
        "--perm-importance",
        action="store_true",
        help="Permutation importance on hold-out trials (subsampled; slower).",
    )
    p.add_argument(
        "--perm-samples",
        type=int,
        default=8000,
        help="Max rows used for permutation importance (speed).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="If set, write coefficients CSV and JSON metrics here.",
    )
    args = p.parse_args()

    usecols = list(BASE_USECOLS)
    if args.extras:
        usecols = usecols + [c for c in EXTRA_USECOLS if c not in usecols]
    if args.paper and "paper" not in usecols:
        usecols.append("paper")

    nrows = None if args.nrows == 0 else args.nrows
    df = load_head(args.csv, nrows, usecols)

    df = df.loc[df["choice"].isin([0, 1])].copy()
    df["choice"] = df["choice"].astype(int)
    df = add_reward_ratio(df)
    if args.premium_per_day:
        df = add_premium_per_day(df)
    df = df.dropna(subset=["subj_ident"])

    tier_cols = [c for c in TIER_A if c in df.columns]
    if args.no_diffs:
        tier_cols = [c for c in tier_cols if c not in ("value_diff", "time_diff_days")]

    feature_cols = list(tier_cols) + ["reward_ratio"]
    if args.premium_per_day:
        feature_cols.append("premium_per_day")
    if args.extras:
        if "age" in df.columns:
            feature_cols.append("age")
        if "rt" in df.columns:
            df["log_rt"] = np.log1p(
                pd.to_numeric(df["rt"], errors="coerce").clip(lower=0).fillna(0)
            )
            feature_cols.append("log_rt")
    if args.context:
        df, dnames = add_categorical_dummies(df, ["procedure", "incentivization", "online_study"])
        feature_cols.extend(dnames)
    if args.paper:
        df, pnames = add_categorical_dummies(df, ["paper"])
        feature_cols.extend(pnames)

    print("Features:", feature_cols, flush=True)
    X = df[feature_cols]
    y = df["choice"].to_numpy()
    groups = df["subj_ident"].to_numpy()

    if args.cv_folds >= 2:
        print(f"\nGrouped {args.cv_folds}-fold CV (by subject) ...", flush=True)
        t_cv = time.perf_counter()
        cv_stats = run_grouped_cv(
            X, y, groups, feature_cols, n_splits=args.cv_folds, seed=args.seed, balance=args.balance
        )
        print(f"CV done in {time.perf_counter() - t_cv:.1f}s.", flush=True)
        for name, arr in cv_stats.items():
            print(f"  {name}: {arr.mean():.4f} ± {arr.std():.4f}", flush=True)

    gss = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
    try:
        tr, te = next(gss.split(X, y, groups))
    except ValueError as e:
        sys.exit(f"Train/test split failed (need enough subjects): {e}")

    pipe = build_pipeline(feature_cols, args.seed, args.balance)

    print(f"\nFitting logistic on {len(tr):,} train trials ({np.unique(groups[tr]).size} subjects) ...", flush=True)
    t1 = time.perf_counter()
    pipe.fit(X.iloc[tr], y[tr])
    print(f"Fit done in {time.perf_counter() - t1:.1f}s.", flush=True)

    proba_tr = pipe.predict_proba(X.iloc[tr])[:, 1]
    proba = pipe.predict_proba(X.iloc[te])[:, 1]
    auc = roc_auc_score(y[te], proba)
    ll = log_loss(y[te], proba)
    brier = brier_score_loss(y[te], proba)

    y_te = y[te]
    maj = float(max(np.mean(y_te), 1.0 - np.mean(y_te)))
    pred05 = (proba >= 0.5).astype(np.int8)
    acc05 = accuracy_score(y_te, pred05)
    bacc05 = balanced_accuracy_score(y_te, pred05)

    print(f"\nHold-out subjects: {np.unique(groups[te]).size} | trials: {len(te):,}")
    print(f"Test LL rate: {np.mean(y_te)*100:.2f}%  |  majority-class baseline accuracy: {maj:.4f}")
    print(f"ROC-AUC: {auc:.4f}  |  log-loss: {ll:.4f}  |  Brier score: {brier:.4f}")
    print(
        f"Accuracy @0.5: {acc05:.4f}  |  balanced accuracy @0.5: {bacc05:.4f}  |  F1 @0.5: {f1_score(y_te, pred05):.4f}"
    )

    if args.tune_threshold:
        pref = "balanced" if args.tune_balanced else "accuracy"
        t_star, _ = tune_threshold(y[tr], proba_tr, prefer=pref)
        pred_t = (proba >= t_star).astype(np.int8)
        acc_t = accuracy_score(y_te, pred_t)
        bacc_t = balanced_accuracy_score(y_te, pred_t)
        print(
            f"Train-tuned threshold ({pref}): t={t_star:.2f}  →  test accuracy: {acc_t:.4f}  |  balanced acc: {bacc_t:.4f}  |  F1: {f1_score(y_te, pred_t):.4f}"
        )

    if args.perm_importance:
        n_perm = min(args.perm_samples, len(te))
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(te), size=n_perm, replace=False)
        te_sub = te[idx]
        print(f"\nPermutation importance (n={n_perm} hold-out trials, 8 repeats) ...", flush=True)
        t2 = time.perf_counter()
        r = permutation_importance(
            pipe,
            X.iloc[te_sub],
            y[te_sub],
            n_repeats=8,
            random_state=args.seed,
            scoring="roc_auc",
            n_jobs=-1,
        )
        print(f"Done in {time.perf_counter() - t2:.1f}s.", flush=True)
        names = pipe.named_steps["prep"].get_feature_names_out()
        imp = pd.DataFrame({"feature": names, "importance_mean": r.importances_mean, "importance_std": r.importances_std})
        print(imp.sort_values("importance_mean", ascending=False).head(12).round(5).to_string(index=False))

    names = pipe.named_steps["prep"].get_feature_names_out()
    coefs = pipe.named_steps["clf"].coef_.ravel()
    tab = pd.DataFrame({"feature": names, "coef": coefs, "odds_ratio": np.exp(coefs)})
    tab["abs_coef"] = tab["coef"].abs()
    print("\nTop coefficients (by |coef|):")
    print(tab.sort_values("abs_coef", ascending=False).drop(columns="abs_coef").head(20).round(4).to_string(index=False))

    used_diffs = not args.no_diffs and bool({"value_diff", "time_diff_days"} & set(tier_cols))
    print_interpretation_caveats(used_diffs=used_diffs, used_paper=args.paper)

    if args.out_dir is not None:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        tab.drop(columns="abs_coef", errors="ignore").to_csv(args.out_dir / "logistic_coefficients.csv", index=False)
        metrics = {
            "holdout_roc_auc": float(auc),
            "holdout_log_loss": float(ll),
            "holdout_brier": float(brier),
            "holdout_accuracy_0.5": float(acc05),
            "holdout_balanced_accuracy_0.5": float(bacc05),
            "holdout_f1_0.5": float(f1_score(y_te, pred05)),
            "n_train_trials": int(len(tr)),
            "n_test_trials": int(len(te)),
            "n_train_subjects": int(np.unique(groups[tr]).size),
            "n_test_subjects": int(np.unique(groups[te]).size),
        }
        if args.cv_folds >= 2:
            metrics["cv_folds"] = int(args.cv_folds)
            for k, arr in cv_stats.items():
                metrics[f"cv_{k}_mean"] = float(arr.mean())
                metrics[f"cv_{k}_std"] = float(arr.std())
        with open(args.out_dir / "holdout_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nWrote {args.out_dir / 'logistic_coefficients.csv'} and {args.out_dir / 'holdout_metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
