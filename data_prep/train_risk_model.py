"""
Component A: the missed-connection risk model.

    uv add scikit-learn          # once; commit pyproject.toml and uv.lock
    uv run python data_prep/train_risk_model.py --test-from 2026-01-01

Splits by date (train on the earlier days, test on the later ones, like real use),
compares three things on the same test days, and saves the best model:

  rule     the baseline: "missed if slack - departure delay - MCT < 0"
  logreg   logistic regression
  hgb      gradient boosting (HistGradientBoostingClassifier)

Writes:
  models/miss_risk.joblib          the chosen model (pipeline.py loads it)
  models/miss_risk.json            its version, features, threshold and test metrics
  eval/component_a_metrics.csv     every model's test metrics, for REPORT.md
  eval/component_a_by_month.csv    the best model vs the rule, month by month (and hub by hub)

To test on months whose season the model has already seen, train on a full year
and test on the next one: --test-from 2026-01-01
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_connections import FEATURES, LABEL  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def metrics(y, score, pred) -> dict:
    return {
        "auc": roc_auc_score(y, score),
        "avg_precision": average_precision_score(y, score),
        "brier": brier_score_loss(y, np.clip(score, 0, 1)) if score.min() >= 0 and score.max() <= 1 else np.nan,
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "flagged_share": float(np.mean(pred)),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=ROOT / "data" / "connections.csv.gz")
    p.add_argument("--test-share", type=float, default=0.25, help="share of the latest days used for testing")
    p.add_argument("--test-from", help="test on every day from this date (YYYY-MM-DD) instead; train on the days before")
    p.add_argument("--threshold", type=float, default=0.5, help="probability above which a connection is flagged")
    args = p.parse_args()

    df = pd.read_csv(args.data)
    days = np.sort(df["date"].unique())
    cut = args.test_from or days[int(len(days) * (1 - args.test_share))]
    train, test = df[df["date"] < cut], df[df["date"] >= cut]
    print(f"Train: {len(train):,} connections ({train['date'].min()} to {train['date'].max()}), {train[LABEL].mean():.1%} missed")
    print(f"Test:  {len(test):,} connections ({test['date'].min()} to {test['date'].max()}), {test[LABEL].mean():.1%} missed\n")
    X_train, y_train, X_test, y_test = train[FEATURES], train[LABEL], test[FEATURES], test[LABEL]

    results, models = [], {}
    # Baseline rule: a lower margin means more risk, so its score is -margin.
    rule_pred = (X_test["naive_margin_min"] < 0).astype(int).to_numpy()
    results.append({"model": "rule (baseline)", **metrics(y_test, -X_test["naive_margin_min"].to_numpy(), rule_pred),
                    "recall_at_rule_budget": recall_score(y_test, rule_pred, zero_division=0)})

    candidates = {
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "hgb": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0),
    }
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        row = {"model": name, **metrics(y_test, proba, (proba >= args.threshold).astype(int))}
        # Fair comparison: flag exactly as many connections as the rule does, and see who catches more misses.
        budget = max(int(rule_pred.sum()), 1)
        top = np.zeros(len(proba), dtype=int)
        top[np.argsort(-proba)[:budget]] = 1
        row["recall_at_rule_budget"] = recall_score(y_test, top, zero_division=0)
        results.append(row)
        models[name] = model

    table = pd.DataFrame(results).set_index("model").round(3)
    print(table.to_string())
    best = max(models, key=lambda n: table.loc[n, "avg_precision"])
    print(f"\nBest by average precision: {best}")

    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "eval").mkdir(exist_ok=True)
    joblib.dump(models[best], ROOT / "models" / "miss_risk.joblib")
    version = f"miss-risk-{best}-{datetime.now(timezone.utc):%Y%m%d}"
    (ROOT / "models" / "miss_risk.json").write_text(json.dumps({
        "version": version, "model": best, "features": FEATURES, "threshold": args.threshold,
        "train_days": [str(train["date"].min()), str(train["date"].max())],
        "test_days": [str(test["date"].min()), str(test["date"].max())],
        "test_metrics": table.loc[best].to_dict(),
    }, indent=2))
    table.to_csv(ROOT / "eval" / "component_a_metrics.csv")

    # Does it hold up month by month (and hub by hub)? Seasons change both the delays and the share missed.
    scored = test.assign(risk=models[best].predict_proba(X_test)[:, 1], month=test["date"].str[:7])
    keys = ["hub", "month"] if "hub" in scored.columns else ["month"]
    rows = []
    for key, g in scored.groupby(keys):
        both = g[LABEL].nunique() == 2
        rows.append({
            **dict(zip(keys, key if isinstance(key, tuple) else (key,))),
            "connections": len(g),
            "missed_rate": g[LABEL].mean(),
            "mean_predicted_risk": g["risk"].mean(),  # close to missed_rate = well calibrated that month
            f"{best}_auc": roc_auc_score(g[LABEL], g["risk"]) if both else np.nan,
            "rule_auc": roc_auc_score(g[LABEL], -g["naive_margin_min"]) if both else np.nan,
        })
    by_month = pd.DataFrame(rows).round(3)
    print("\nTest results by " + " and ".join(keys) + ":")
    print(by_month.to_string(index=False))
    by_month.to_csv(ROOT / "eval" / "component_a_by_month.csv", index=False)
    print(f"Saved models/miss_risk.joblib ({version}) and eval/component_a_metrics.csv.")


if __name__ == "__main__":
    main()
