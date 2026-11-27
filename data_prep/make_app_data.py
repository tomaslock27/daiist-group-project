"""
The small data files the app reads online (Render has 512 MB: the full year of
connections is too big to load there).

    uv run python data_prep/make_app_data.py

Reads  data/connections.csv.gz, data/schedule.csv      (the hubs the model was trained on)
       data/lax/connections.csv.gz, data/lax/schedule.csv  (the drift hub)
Writes app_data/connections.csv  connections from the TEST months only (never seen in training),
                                 a sample per hub, plus some from LAX
       app_data/schedule.csv     every departure from those hubs in the test months,
                                 for the agent's "find the next flights" tool

Commit app_data/: the online app and check.py need it.
"""

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--from-date", default="2026-01-01", help="first day of the test period")
    p.add_argument("--per-hub", type=int, default=1500, help="connections sampled per trained hub")
    p.add_argument("--drift-hub-rows", type=int, default=600, help="connections sampled from the drift hub")
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    main_conn = pd.read_csv(ROOT / "data" / "connections.csv.gz")
    lax_conn = pd.read_csv(ROOT / "data" / "lax" / "connections.csv.gz")
    main_sched = pd.read_csv(ROOT / "data" / "schedule.csv")
    lax_sched = pd.read_csv(ROOT / "data" / "lax" / "schedule.csv")

    test = main_conn[main_conn["date"] >= args.from_date]
    if test.empty:
        raise SystemExit(f"No connections on or after {args.from_date} in data/connections.csv.gz.")
    parts = []
    for hub, g in test.groupby("hub"):
        # Keep every missed connection we can (they are rare) and fill up with the rest.
        missed = g[g["missed"] == 1].sample(min(len(g[g["missed"] == 1]), args.per_hub // 4), random_state=args.seed)
        rest = g.drop(missed.index).sample(min(len(g) - len(missed), args.per_hub - len(missed)), random_state=args.seed)
        parts += [missed, rest]
    lax = lax_conn[lax_conn["date"] >= args.from_date]
    parts.append(lax.sample(min(len(lax), args.drift_hub_rows), random_state=args.seed))
    demo = pd.concat(parts).sort_values(["date", "hub", "in_sched_arr"]).reset_index(drop=True)

    sched = pd.concat([main_sched, lax_sched])
    sched = sched[(sched["date"] >= args.from_date) & (sched["date"] <= demo["date"].max())]
    sched = sched[sched["outcome_cancelled"] != 1].drop(columns=["outcome_dep_delay_min", "outcome_cancelled"])

    out = ROOT / "app_data"
    out.mkdir(exist_ok=True)
    demo.to_csv(out / "connections.csv", index=False)
    sched.to_csv(out / "schedule.csv", index=False)
    print(f"Wrote app_data/connections.csv: {len(demo):,} connections "
          f"({', '.join(f'{h} {n:,}' for h, n in demo['hub'].value_counts().items())}), {demo['missed'].mean():.1%} missed.")
    print(f"Wrote app_data/schedule.csv: {len(sched):,} departures ({demo['date'].min()} to {demo['date'].max()}).")


if __name__ == "__main__":
    main()
