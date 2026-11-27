"""
Build connecting-flight pairs at one hub from US BTS on-time data.

    uv run python data_prep/build_connections.py data/raw/*.zip --hub ORD --airline UA

Input: one or more monthly "Reporting Carrier On-Time Performance" files from
https://transtats.bts.gov (the PREZIP .zip files, or the .csv inside them).
Kaggle mirrors with the usual column names work too.

Output (in --out, default data/):
  connections.csv.gz  (compressed CSV) one row per (inbound flight -> outbound flight) pair at the hub:
                   ids for display, FEATURES known at decision time, the label `missed`,
                   and `outcome_*` columns that are NOT features (analysis only)
  schedule.csv     every departure of the airline from the hub, for the agent's
                   "find the next flight" tool

Decision moment: the inbound flight has just taken off (wheels off). So we may use
its departure delay and taxi-out, but never its arrival delay or the outbound's
actual departure. Everything is computed in minutes on the hub's local clock,
from scheduled times plus delays, so time zones and midnight never get in the way.

Assumptions (say them in REPORT.md):
  - We don't know which passengers connected: every pair with a scheduled
    connection time between --min-slack and --max-slack counts as a connection.
  - Missed = the inbound arrives later than (outbound actual departure - MCT),
    or the inbound was diverted.
  - Cancelled inbound flights are dropped (no "in-flight" decision to make);
    cancelled outbound flights are dropped (a different product: rebooking
    after a cancellation).
  - Overnight arrivals (before 05:00) are dropped to keep the clock simple.
  - Only the reporting (operating) carrier: regional partners flying for the
    airline under another code are not included.
"""

import argparse
import glob
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

# Our name -> the names it can have in BTS files or common Kaggle mirrors.
COLUMNS = {
    "date": ["FlightDate", "FL_DATE"],
    "airline": ["Reporting_Airline", "IATA_CODE_Reporting_Airline", "Operating_Airline",
                "IATA_Code_Operating_Airline", "OP_UNIQUE_CARRIER", "OP_CARRIER"],
    "flight": ["Flight_Number_Reporting_Airline", "Flight_Number_Operating_Airline", "OP_CARRIER_FL_NUM"],
    "tail_no": ["Tail_Number", "TAIL_NUM"],
    "origin": ["Origin", "ORIGIN"],
    "dest": ["Dest", "DEST"],
    "crs_dep": ["CRSDepTime", "CRS_DEP_TIME"],
    "crs_arr": ["CRSArrTime", "CRS_ARR_TIME"],
    "dep_delay": ["DepDelay", "DEP_DELAY"],
    "arr_delay": ["ArrDelay", "ARR_DELAY"],
    "taxi_out": ["TaxiOut", "TAXI_OUT"],
    "crs_elapsed": ["CRSElapsedTime", "CRS_ELAPSED_TIME"],
    "distance": ["Distance", "DISTANCE"],
    "cancelled": ["Cancelled", "CANCELLED"],
    "diverted": ["Diverted", "DIVERTED"],
}

# The model's inputs. All known when the inbound flight takes off.
FEATURES = [
    "sched_slack_min",          # scheduled connection time
    "naive_margin_min",         # slack - inbound departure delay - MCT (the baseline rule's number)
    "in_dep_delay_min",         # inbound departure delay
    "in_taxi_out_min",          # inbound taxi-out at the origin
    "in_sched_block_min",       # inbound scheduled gate-to-gate time
    "in_distance_mi",
    "in_sched_arr_hour",        # hour of scheduled arrival at the hub
    "day_of_week",              # 0 = Monday
    "same_aircraft",            # the outbound uses the inbound's plane (it will wait)
    "out_prev_leg_known",       # the outbound's incoming plane has already departed
    "out_prev_leg_dep_delay_min",  # ...and its departure delay (0 if not known / no previous leg)
    "out_first_flight_of_day",  # the outbound's plane starts its day at the hub
    "hub_recent_arr_delay_min", # mean arrival delay at the hub in the hour before the decision
    "hub_sched_arrivals_hour",  # scheduled arrivals at the hub in the inbound's arrival hour
]
LABEL = "missed"


def read_file(path: Path) -> pd.DataFrame:
    wanted = {name for names in COLUMNS.values() for name in names}
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            csv_name = next(n for n in z.namelist() if n.lower().endswith(".csv"))
            with z.open(csv_name) as f:
                df = pd.read_csv(f, usecols=lambda c: c in wanted, low_memory=False)
    elif path.suffix == ".parquet":
        df = pd.read_parquet(path)
        df = df[[c for c in df.columns if c in wanted]]
    else:
        df = pd.read_csv(path, usecols=lambda c: c in wanted, low_memory=False)
    rename = {}
    for ours, names in COLUMNS.items():
        found = next((n for n in names if n in df.columns), None)
        if found is None:
            raise SystemExit(f"{path.name}: no column for `{ours}` (looked for {', '.join(names)}).")
        rename[found] = ours
    return df.rename(columns=rename)[list(COLUMNS)]


def hhmm_to_min(hhmm: pd.Series) -> pd.Series:
    hhmm = pd.to_numeric(hhmm, errors="coerce")
    return (hhmm // 100) * 60 + hhmm % 100


def load(paths: list[Path], airline: str) -> pd.DataFrame:
    parts = []
    for p in paths:  # keep only this airline from each file, so a whole year fits in memory
        part = read_file(p)
        parts.append(part[part["airline"].astype(str).str.strip() == airline])
        print(f"  {p.name}: {len(parts[-1]):,} {airline} flights")
    df = pd.concat(parts, ignore_index=True).copy()
    if df.empty:
        raise SystemExit(f"No flights for airline {airline} in these files.")
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["flight"] = df["airline"].astype(str).str.strip() + pd.to_numeric(df["flight"], errors="coerce").astype("Int64").astype(str)
    for c in ["dep_delay", "arr_delay", "taxi_out", "crs_elapsed", "distance", "cancelled", "diverted"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["crs_dep_min"], df["crs_arr_min"] = hhmm_to_min(df["crs_dep"]), hhmm_to_min(df["crs_arr"])
    df["day"] = (df["date"] - df["date"].min()).dt.days
    return df.drop_duplicates(["date", "flight", "origin", "dest"])


def build(df: pd.DataFrame, hub: str, mct: int, min_slack: int, max_slack: int, per_inbound: int, seed: int):
    rng = np.random.default_rng(seed)

    # Arrivals at the hub, on the hub's clock.
    arr = df[(df["dest"] == hub) & (df["cancelled"] != 1)].copy()
    arr = arr[arr["crs_arr_min"] >= 300]  # drop overnight arrivals
    arr["sched_arr_abs"] = arr["day"] * 1440 + arr["crs_arr_min"]
    arr["taxi_out"] = arr["taxi_out"].fillna(arr["taxi_out"].median())
    arr["dep_delay"] = arr["dep_delay"].fillna(0)
    # When it took off, expressed on the hub's clock: scheduled arrival - block time + delay + taxi-out.
    arr["takeoff_abs"] = arr["sched_arr_abs"] - arr["crs_elapsed"] + arr["dep_delay"] + arr["taxi_out"]
    arr["actual_arr_abs"] = arr["sched_arr_abs"] + arr["arr_delay"]  # NaN if diverted
    arr = arr.dropna(subset=["crs_elapsed"]).reset_index(drop=True)

    # Departures from the hub.
    dep_all = df[df["origin"] == hub].copy()
    dep_all["sched_dep_abs"] = dep_all["day"] * 1440 + dep_all["crs_dep_min"]
    dep = dep_all[(dep_all["cancelled"] != 1) & dep_all["dep_delay"].notna()].sort_values("sched_dep_abs").reset_index(drop=True)
    dep["actual_dep_abs"] = dep["sched_dep_abs"] + dep["dep_delay"]

    # The outbound plane's previous leg: the latest arrival of the same tail at the hub before it leaves.
    legs = arr[arr["tail_no"].notna()].sort_values("sched_arr_abs")[["tail_no", "day", "sched_arr_abs", "takeoff_abs", "dep_delay"]]
    dep_sorted = dep[dep["tail_no"].notna()].sort_values("sched_dep_abs")
    prev = pd.merge_asof(
        dep_sorted[["sched_dep_abs", "tail_no", "day"]].reset_index(),
        legs.rename(columns={"sched_arr_abs": "prev_arr_abs", "takeoff_abs": "prev_takeoff_abs", "dep_delay": "prev_dep_delay", "day": "prev_day"}),
        left_on="sched_dep_abs", right_on="prev_arr_abs", by="tail_no", direction="backward", allow_exact_matches=False,
    ).set_index("index")
    prev.loc[prev["prev_day"] != prev["day"], ["prev_arr_abs", "prev_takeoff_abs", "prev_dep_delay"]] = np.nan
    dep = dep.join(prev[["prev_takeoff_abs", "prev_dep_delay"]])

    # Hub congestion: arrival delays of flights that landed before each decision, and scheduled arrivals per hour.
    landed = arr.dropna(subset=["actual_arr_abs"]).sort_values("actual_arr_abs")
    land_t, land_d = landed["actual_arr_abs"].to_numpy(), landed["arr_delay"].to_numpy()
    cum = np.concatenate([[0.0], np.cumsum(land_d)])
    hour_counts = arr.groupby(arr["sched_arr_abs"] // 60).size()

    dep_times, dep_dest = dep["sched_dep_abs"].to_numpy(), dep["dest"].to_numpy()
    rows = []
    for a in arr.itertuples(index=False):
        lo = np.searchsorted(dep_times, a.sched_arr_abs + min_slack, side="left")
        hi = np.searchsorted(dep_times, a.sched_arr_abs + max_slack, side="right")
        candidates = np.arange(lo, hi)[dep_dest[lo:hi] != a.origin]
        if len(candidates) == 0:
            continue
        chosen = rng.choice(candidates, size=min(per_inbound, len(candidates)), replace=False)

        i0 = np.searchsorted(land_t, a.takeoff_abs - 60, side="left")
        i1 = np.searchsorted(land_t, a.takeoff_abs, side="left")
        recent = (cum[i1] - cum[i0]) / (i1 - i0) if i1 > i0 else 0.0

        for i in chosen:
            o = dep.loc[i]
            slack = o.sched_dep_abs - a.sched_arr_abs
            prev_known = bool(pd.notna(o.prev_takeoff_abs) and o.prev_takeoff_abs <= a.takeoff_abs)
            if a.diverted == 1 or pd.isna(a.actual_arr_abs):
                missed = 1
            else:
                missed = int(a.actual_arr_abs + mct > o.actual_dep_abs)
            rows.append({
                # ids, for display and for the agent
                "date": a.date.date().isoformat(),
                "in_flight": a.flight, "in_origin": a.origin,
                "in_sched_arr": f"{int(a.crs_arr_min) // 60:02d}:{int(a.crs_arr_min) % 60:02d}",
                "out_flight": o.flight, "out_dest": o.dest,
                "out_sched_dep": f"{int(o.crs_dep_min) // 60:02d}:{int(o.crs_dep_min) % 60:02d}",
                # features
                "sched_slack_min": float(slack),
                "naive_margin_min": float(slack - a.dep_delay - mct),
                "in_dep_delay_min": float(a.dep_delay),
                "in_taxi_out_min": float(a.taxi_out),
                "in_sched_block_min": float(a.crs_elapsed),
                "in_distance_mi": float(a.distance),
                "in_sched_arr_hour": float(a.crs_arr_min // 60),
                "day_of_week": float(a.date.day_of_week),
                "same_aircraft": float(pd.notna(a.tail_no) and a.tail_no == o["tail_no"]),
                "out_prev_leg_known": float(prev_known),
                "out_prev_leg_dep_delay_min": float(o.prev_dep_delay) if prev_known else 0.0,
                "out_first_flight_of_day": float(pd.isna(o.prev_takeoff_abs)),
                "hub_recent_arr_delay_min": float(recent),
                "hub_sched_arrivals_hour": float(hour_counts.get(a.sched_arr_abs // 60, 0)),
                # label and outcomes (never features)
                LABEL: missed,
                "outcome_in_arr_delay_min": a.arr_delay,
                "outcome_out_dep_delay_min": o.dep_delay,
                "outcome_in_diverted": int(a.diverted == 1),
            })
    connections = pd.DataFrame(rows)
    connections.insert(0, "conn_id", [f"{hub}-{i:06d}" for i in range(len(connections))])

    schedule = dep_all.assign(
        date=dep_all["date"].dt.date.astype(str),
        sched_dep=[f"{int(m) // 60:02d}:{int(m) % 60:02d}" for m in dep_all["crs_dep_min"]],
    )[["date", "flight", "dest", "sched_dep", "crs_dep_min", "dep_delay", "cancelled"]]
    schedule = schedule.rename(columns={"dep_delay": "outcome_dep_delay_min", "cancelled": "outcome_cancelled"})
    return connections, schedule.sort_values(["date", "crs_dep_min"]).reset_index(drop=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="+", type=Path)
    p.add_argument("--hub", default="ORD", help="one hub, or several separated by commas, e.g. EWR,IAD")
    p.add_argument("--airline", default="UA", help="reporting carrier code, e.g. UA, AA, DL")
    p.add_argument("--mct", type=int, default=35, help="minimum connection time in minutes (an assumption: justify it)")
    p.add_argument("--min-slack", type=int, default=35)
    p.add_argument("--max-slack", type=int, default=120)
    p.add_argument("--per-inbound", type=int, default=3, help="outbound flights sampled per inbound flight")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=Path("data"))
    args = p.parse_args()

    # Expand wildcards ourselves: Windows PowerShell passes `data/raw/*2025_9.zip` through unexpanded.
    files = []
    for pattern in map(str, args.files):
        matches = sorted(glob.glob(pattern)) if any(ch in pattern for ch in "*?[") else [pattern]
        if not matches or not Path(matches[0]).is_file():
            raise SystemExit(f"No file matches {pattern}. Are the downloads in data/raw/?")
        files += [Path(m) for m in matches]
    print("Reading " + ", ".join(f.name for f in files))
    df = load(files, args.airline)
    print(f"{len(df):,} {args.airline} flights, {df['date'].min().date()} to {df['date'].max().date()}.")
    hubs = [h.strip().upper() for h in args.hub.split(",") if h.strip()]
    all_connections, all_schedules = [], []
    for hub in hubs:
        connections, schedule = build(df, hub, args.mct, args.min_slack, args.max_slack, args.per_inbound, args.seed)
        if connections.empty:
            raise SystemExit(f"No connections at {hub}. Is it a hub of {args.airline}?")
        connections.insert(1, "hub", hub)
        schedule.insert(0, "hub", hub)
        all_connections.append(connections)
        all_schedules.append(schedule)
    connections = pd.concat(all_connections, ignore_index=True)
    schedule = pd.concat(all_schedules, ignore_index=True)
    args.out.mkdir(parents=True, exist_ok=True)
    connections.to_csv(args.out / "connections.csv.gz", index=False)  # compressed: a year of data stays small for GitHub
    schedule.to_csv(args.out / "schedule.csv", index=False)
    print(f"Wrote {args.out / 'connections.csv.gz'}: {len(connections):,} connections, "
          f"{connections[LABEL].mean():.1%} missed.")
    print(f"Wrote {args.out / 'schedule.csv'}: {len(schedule):,} departures from {', '.join(hubs)}.")
    print("\nConnections and missed rate by hub and month:")
    table = connections.groupby(["hub", connections["date"].str[:7].rename("month")])[LABEL].agg(["count", "mean"])
    print(table.rename(columns={"count": "connections", "mean": "missed"}).round(3).to_string())

if __name__ == "__main__":
    main()
