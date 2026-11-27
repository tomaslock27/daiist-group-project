"""
Choosing the flag threshold with numbers.

    uv run python data_prep/threshold_table.py

Scores the test months (from 2026-01-01) with the saved model and prints, for each
threshold, what the desk would see: misses caught, false alarms, the value in euros a day
and how many cases a day go to review. The simple rule is shown for comparison.
Writes eval/threshold_table.csv.

The euro and volume figures use the business case's assumptions (change them with the
options): about 31,400 domestic connecting passengers a day at ORD and DEN, 629 of whom
miss, EUR 225 per miss handled after landing, 30% of that saved by acting early,
EUR 40 per false alarm.
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import pipeline  # noqa: E402  (the review band and costs come from the product itself)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=ROOT / "data" / "connections.csv.gz")
    p.add_argument("--test-from", default="2026-01-01")
    p.add_argument("--connecting-per-day", type=float, default=31_400)
    p.add_argument("--misses-per-day", type=float, default=629)
    args = p.parse_args()

    meta = json.loads((ROOT / "models" / "miss_risk.json").read_text())
    model = joblib.load(ROOT / "models" / "miss_risk.joblib")
    test = pd.read_csv(args.data)
    test = test[test["date"] >= args.test_from]
    y = test["missed"].to_numpy()
    risk = model.predict_proba(test[meta["features"]])[:, 1]
    lo, hi = pipeline.REVIEW_BAND
    in_band = float(np.mean((risk >= lo) & (risk <= hi)))
    saving = pipeline.MISS_COST_EUR * pipeline.EARLY_SAVING

    def row(name, flagged):
        caught = int(np.sum(flagged & (y == 1)))
        false_alarms = int(np.sum(flagged & (y == 0)))
        recall = caught / max(int(y.sum()), 1)
        precision = caught / max(caught + false_alarms, 1)
        caught_day = recall * args.misses_per_day
        fa_day = caught_day * (1 / precision - 1) if precision else 0.0
        return {
            "threshold": name,
            "flagged_per_1000": round(1000 * flagged.mean(), 1),
            "misses_caught_per_1000": round(1000 * caught / len(y), 1),
            "false_alarms_per_1000": round(1000 * false_alarms / len(y), 1),
            "recall": round(recall, 3),
            "precision": round(precision, 3),
            "value_eur_per_day": round(caught_day * saving - fa_day * pipeline.FALSE_ALARM_COST_EUR),
            "value_eur_per_year_m": round((caught_day * saving - fa_day * pipeline.FALSE_ALARM_COST_EUR) * 365 / 1e6, 2),
        }

    rows = [row("rule (baseline)", test["naive_margin_min"].to_numpy() < 0)]
    rows += [row(f"{t:.0%}", risk >= t) for t in (0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6)]
    table = pd.DataFrame(rows)
    print(f"Test: {len(y):,} connections ({test['date'].min()} to {test['date'].max()}), {y.mean():.1%} missed.\n")
    print(table.to_string(index=False))
    print(f"\nPer 1,000 connections there are {1000 * y.mean():.0f} real misses.")
    print(f"Break-even risk from the costs: {pipeline.BREAK_EVEN_RISK:.0%} "
          f"(acting early is cheaper than waiting above this risk).")
    print(f"Review band {lo:.0%}-{hi:.0%}: {in_band:.1%} of connections, about "
          f"{in_band * args.connecting_per_day:,.0f} cases a day at {args.connecting_per_day:,.0f} connecting passengers "
          "(plus overnights, low-confidence agent answers and untrained hubs). An upper estimate: flight pairs\n"
          f"miss more often ({y.mean():.0%}) than passengers overall (2% in the business case).")
    (ROOT / "eval").mkdir(exist_ok=True)
    table.to_csv(ROOT / "eval" / "threshold_table.csv", index=False)
    print("Wrote eval/threshold_table.csv.")


if __name__ == "__main__":
    main()
