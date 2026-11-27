"""
Connection Rescue: the two components and how one feeds the other.

    component A  tabular ML (scikit-learn): the risk that a passenger misses their connection
    component B  agentic LLM (Gemini with tools): what to do about it, and the message to the passenger

The input is a connection id from app_data/connections.csv, e.g. "DEN-001234": in real use
the airline's systems already know every flight and its delay, so the desk picks a
connection instead of typing it. Anything else is out of scope and refused.

run() keeps the template's contract (app.py, evaluate.py, main.py and check.py read it):

    run(raw_input) -> {"input_summary", "features", "outputs", "versions", "review", "tokens", "cost_eur"}

Every number that goes into the log is a plain Python int/float/bool: numpy types fail
logbook.validate().
"""

import json
import os
import re
import time
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent
APP_DATA = ROOT / "app_data"
MODEL_FILE = ROOT / "models" / "miss_risk.joblib"
MODEL_META = ROOT / "models" / "miss_risk.json"

# --- Decisions (see DECISIONS.md; justify the numbers in REPORT.md) ---------------
TRAINED_HUBS = {"ORD", "DEN"}  # the model only knows these hubs; anything else goes to a person
MCT = 35                       # minimum connection time, minutes (same as in the data)
FLAG_AT = 0.4                  # risk at or above this = "will probably miss" (chosen from data_prep/threshold_table.py)
AGENT_FROM = 0.2               # below this, with minutes to spare, the connection is safe: keep it, no Gemini call
REVIEW_BAND = (0.3, 0.5)       # the model is unsure here, around the threshold: a person checks the decision
# Costs: the same numbers as the business case (one set everywhere).
MISS_COST_EUR = 225          # a passenger rebooked after missing the connection ($250, Switchfly estimate)
EARLY_SAVING = 0.30          # share of that cost saved when acting before landing (our assumption)
FALSE_ALARM_COST_EUR = 40    # rebooking someone who would have made it (our assumption)
BREAK_EVEN_RISK = FALSE_ALARM_COST_EUR / (MISS_COST_EUR * EARLY_SAVING + FALSE_ALARM_COST_EUR)  # about 0.37

# Main model and backup (tested 27 Nov 2026: on a paid key all answer in 2-4 s; on a free key
# 3.8 and 3.7 were often busy (503/504) and only 3.6 answered). Thinking: the model's default.
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "").strip() or "gemini-3.8-flash"
GEMINI_BACKUP_MODEL = os.environ.get("GEMINI_BACKUP_MODEL", "").strip() or "gemini-3.6-flash"
GEMINI_CALL_TIMEOUT_S = 20     # one call to Gemini
GEMINI_TOTAL_BUDGET_S = 60     # all attempts together; after this the rules fallback answers
PRICE_USD_PER_M = {"input": 0.75, "output": 3.75}  # Gemini Flash, through Dec 2026; check the pricing page
USD_PER_EUR = 1.11
PROMPT_VERSION = "agent-v2"
MAX_AGENT_TURNS = 6

# What the drift check watches: continuous features only (a feature with <= 5 distinct
# values in the reference set is treated as categories by monitor.py, and any new value
# then looks like drift).
DRIFT_FEATURES = [
    "sched_slack_min", "in_dep_delay_min", "in_sched_block_min", "in_distance_mi",
    "hub_recent_arr_delay_min", "hub_sched_arrivals_hour",
]
ID_PATTERN = re.compile(r"\b([A-Z]{3}-\d{6})\b")


# --- Data and model ---------------------------------------------------------------

@lru_cache(maxsize=1)
def connections() -> pd.DataFrame:
    return pd.read_csv(APP_DATA / "connections.csv").set_index("conn_id")


@lru_cache(maxsize=1)
def schedule() -> pd.DataFrame:
    return pd.read_csv(APP_DATA / "schedule.csv")


@lru_cache(maxsize=1)
def model():
    return joblib.load(MODEL_FILE), json.loads(MODEL_META.read_text())


def _hhmm(minutes: float) -> str:
    minutes = int(round(minutes))
    day, rest = divmod(minutes, 1440)
    return f"{rest // 60:02d}:{rest % 60:02d}" + (" (+1 day)" if day > 0 else "")


def _minutes(hhmm: str) -> int:
    h, m = str(hhmm)[:5].split(":")
    return int(h) * 60 + int(m)


def demo_choices(limit: int = 400) -> list[tuple[str, str]]:
    """(label, connection id) for the App tab's dropdown: inbound flights running late, plus some from untrained hubs."""
    c = connections().reset_index()
    late = c[(c["in_dep_delay_min"] >= 15) & c["hub"].isin(TRAINED_HUBS)].sample(frac=1, random_state=1)
    other = c[~c["hub"].isin(TRAINED_HUBS)].sample(frac=1, random_state=1)
    picked = pd.concat([late.head(int(limit * 0.85)), other.head(limit - int(limit * 0.85))])
    picked = picked.sort_values(["date", "hub", "in_sched_arr"])
    return [(
        f"{r.hub} · {r.date} · {r.in_flight} from {r.in_origin}, left {int(r.in_dep_delay_min)} min late"
        f" → {r.out_flight} to {r.out_dest} at {r.out_sched_dep}"
        + ("" if r.hub in TRAINED_HUBS else "  (hub not trained)"),
        r.conn_id,
    ) for r in picked.itertuples()]


# --- Component A: risk of missing the connection -----------------------------------

def component_a(row: pd.Series) -> dict:
    clf, meta = model()
    X = pd.DataFrame([row[meta["features"]].astype(float).to_numpy()], columns=meta["features"])
    risk = float(clf.predict_proba(X)[0, 1])
    est_arrival = _minutes(row["in_sched_arr"]) + float(row["in_dep_delay_min"])
    return {
        "risk": round(risk, 3),
        "flagged": bool(risk >= FLAG_AT),
        "hub": str(row["hub"]),
        "hub_trained": bool(row["hub"] in TRAINED_HUBS),
        "estimated_arrival": _hhmm(est_arrival),
        "connection_departs": str(row["out_sched_dep"]),
        "missing_flight": str(row["out_flight"]),
        "minutes_to_spare": int(round(_minutes(row["out_sched_dep"]) - est_arrival - MCT)),
    }


# --- Component B: the rebooking agent --------------------------------------------

def _tools_for(row: pd.Series, risk: float):
    """The agent's tools, bound to this connection (hub, date, flight being missed) and its risk."""
    hub, date, dest, missing = str(row["hub"]), str(row["date"]), str(row["out_dest"]), str(row["out_flight"])

    def find_next_flights(destination: str = dest, earliest_departure: str = "00:00") -> dict:
        s = schedule()
        s = s[(s["hub"] == hub) & (s["dest"] == str(destination).upper()) & (s["flight"] != missing)]  # never the flight being missed
        today = s[(s["date"] == date) & (s["crs_dep_min"] >= _minutes(earliest_departure))].head(5)
        if len(today):
            return {"same_day": True, "flights": [{"flight": f, "departs": d} for f, d in zip(today["flight"], today["sched_dep"])]}
        nxt = s[s["date"] > date]
        nxt = nxt[nxt["date"] == nxt["date"].min()].head(3) if len(nxt) else nxt
        return {"same_day": False, "next_day": str(nxt["date"].iloc[0]) if len(nxt) else None,
                "flights": [{"flight": f, "departs": d} for f, d in zip(nxt["flight"], nxt["sched_dep"])]}

    def rebooking_policy() -> dict:
        return {
            "minimum_connection_minutes": MCT,
            "keep": "Keep the booking if the passenger is still expected to reach the connecting flight in time.",
            "rebook": f"Otherwise rebook onto the earliest flight to the same destination that leaves at least {MCT} minutes after the estimated arrival, the same day.",
            "overnight": "If no flight is left the same day, book the first flight the next morning and a hotel voucher.",
        }

    def compare_costs() -> dict:
        """Expected cost of keeping the booking vs acting now, for this passenger's risk."""
        p = float(risk)
        keep = p * MISS_COST_EUR
        act = p * MISS_COST_EUR * (1 - EARLY_SAVING) + (1 - p) * FALSE_ALARM_COST_EUR
        return {
            "risk_of_missing": round(p, 3),
            "expected_cost_if_keep_eur": round(keep, 2),
            "expected_cost_if_act_now_eur": round(act, 2),
            "cheaper_option": "keep" if keep <= act else "act now (rebook or overnight)",
            "break_even_risk": round(BREAK_EVEN_RISK, 3),
            "assumptions": {"cost_of_a_miss_handled_after_landing_eur": MISS_COST_EUR,
                            "share_saved_by_acting_before_landing": EARLY_SAVING,
                            "cost_of_rebooking_someone_who_would_have_made_it_eur": FALSE_ALARM_COST_EUR},
        }

    return {"find_next_flights": find_next_flights, "rebooking_policy": rebooking_policy, "compare_costs": compare_costs}


SYSTEM_PROMPT = """You are the rebooking assistant of United Airlines' re-accommodation desk at a hub airport.
A risk model has scored one connecting passenger whose inbound flight is running late.
Decide ONE action for them:
- keep: they will probably make the connection; change nothing.
- rebook: they will probably miss it; move them to a later flight to the same destination today.
- overnight: no flight is left today; book the first flight tomorrow and a hotel.
How to work:
1. Call rebooking_policy to read the rules.
2. Call compare_costs to see whether keeping the booking or acting now is cheaper for this risk.
3. Unless you keep the booking, call find_next_flights with the destination and the earliest departure
   the passenger can make (estimated arrival + the minimum connection time, as HH:MM).
4. Finish by calling submit_decision. Only use flights returned by find_next_flights; never invent one.
The message to the passenger: 2 to 4 short, friendly sentences, flight numbers and times included,
no names (you don't know them), no promises the policy does not make."""

FUNCTION_DECLARATIONS = [
    {"name": "rebooking_policy", "description": "The airline's rebooking rules and costs.",
     "parameters_json_schema": {"type": "object", "properties": {}}},
    {"name": "compare_costs", "description": "Expected cost in euros of keeping the booking vs acting now (rebook or overnight), given this passenger's risk of missing.",
     "parameters_json_schema": {"type": "object", "properties": {}}},
    {"name": "find_next_flights",
     "description": "Scheduled departures from this hub to a destination, leaving at or after a time today (or the first ones tomorrow if none are left).",
     "parameters_json_schema": {"type": "object", "properties": {
         "destination": {"type": "string", "description": "Airport code, e.g. BOI"},
         "earliest_departure": {"type": "string", "description": "HH:MM, local time at the hub"}},
         "required": ["destination", "earliest_departure"]}},
    {"name": "submit_decision", "description": "Submit the final decision. Call it exactly once, at the end.",
     "parameters_json_schema": {"type": "object", "properties": {
         "decision": {"type": "string", "enum": ["keep", "rebook", "overnight"]},
         "new_flight": {"type": "string", "description": "The flight number for rebook/overnight; empty for keep"},
         "message_to_passenger": {"type": "string"},
         "reason_for_desk": {"type": "string", "description": "One sentence for the desk agent"},
         "confidence": {"type": "string", "enum": ["high", "medium", "low"]}},
         "required": ["decision", "new_flight", "message_to_passenger", "reason_for_desk", "confidence"]}},
]


def _situation(row: pd.Series, a: dict) -> str:
    return (
        f"Hub: {row['hub']}. Date: {row['date']}.\n"
        f"Inbound flight {row['in_flight']} from {row['in_origin']}, scheduled to arrive {row['in_sched_arr']}, "
        f"left {int(row['in_dep_delay_min'])} minutes late; estimated arrival {a['estimated_arrival']}.\n"
        f"Connecting flight {row['out_flight']} to {row['out_dest']}, departs {row['out_sched_dep']}.\n"
        f"Minutes to spare after the {MCT}-minute minimum connection time: {a['minutes_to_spare']}.\n"
        f"Risk model: {a['risk']:.0%} probability of missing the connection "
        f"({'above' if a['flagged'] else 'below'} the {FLAG_AT:.0%} flag threshold)."
    )


RETRYABLE = ("503", "504", "429", "500", "UNAVAILABLE", "DEADLINE", "RESOURCE_EXHAUSTED", "Timeout", "timed out")


def _gemini_with_retries(row: pd.Series, a: dict) -> dict:
    """Busy or slow Gemini: retry the main model once, then try the backup model, within a time budget."""
    started, attempts = time.monotonic(), []
    for model_name in dict.fromkeys([GEMINI_MODEL, GEMINI_BACKUP_MODEL]):  # no duplicate if both are the same
        for attempt in (1, 2):
            if time.monotonic() - started > GEMINI_TOTAL_BUDGET_S:
                raise RuntimeError(f"no answer within {GEMINI_TOTAL_BUDGET_S}s; tried {attempts}")
            try:
                result = _gemini_agent(row, a, model_name)
                return result | {"model": model_name, "gemini_attempts": attempts + [f"{model_name}: ok"]}
            except Exception as error:  # noqa: BLE001
                text = f"{type(error).__name__}: {error}"
                attempts.append(f"{model_name}: {text[:120]}")
                if not any(code in text for code in RETRYABLE):
                    break  # e.g. model not found: retrying the same model will not help
                time.sleep(2)
    raise RuntimeError("; ".join(attempts))


def _gemini_agent(row: pd.Series, a: dict, model_name: str) -> dict:
    from google import genai
    from google.genai import types

    tools = _tools_for(row, a["risk"])
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"],
                          http_options=types.HttpOptions(timeout=GEMINI_CALL_TIMEOUT_S * 1000))  # ms
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        temperature=0,
        tools=[types.Tool(function_declarations=[types.FunctionDeclaration(**d) for d in FUNCTION_DECLARATIONS])],
        tool_config=types.ToolConfig(function_calling_config=types.FunctionCallingConfig(mode="ANY")),
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    contents = [types.Content(role="user", parts=[types.Part.from_text(text=_situation(row, a))])]
    tokens_in = tokens_out = 0
    calls_log, offered = [], set()
    for _ in range(MAX_AGENT_TURNS):
        response = client.models.generate_content(model=model_name, contents=contents, config=config)
        usage = response.usage_metadata
        tokens_in += int(usage.prompt_token_count or 0)
        tokens_out += int((usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0))
        calls = response.function_calls or []
        if not calls:
            break
        contents.append(response.candidates[0].content)
        replies = []
        for call in calls:
            args = dict(call.args or {})
            calls_log.append({"tool": call.name, "args": args})
            if call.name == "submit_decision":
                cost = (tokens_in * PRICE_USD_PER_M["input"] + tokens_out * PRICE_USD_PER_M["output"]) / 1e6 / USD_PER_EUR
                return {**args, "mode": "gemini", "tool_calls": calls_log, "offered_flights": sorted(offered),
                        "tokens": tokens_in + tokens_out, "cost_eur": cost}
            result = tools[call.name](**args) if call.name in tools else {"error": f"unknown tool {call.name}"}
            offered.update(f["flight"] for f in result.get("flights", []))
            replies.append(types.Part.from_function_response(name=call.name, response={"result": result}))
        contents.append(types.Content(role="user", parts=replies))
    raise RuntimeError(f"the agent did not submit a decision within {MAX_AGENT_TURNS} turns")


def _fallback_agent(row: pd.Series, a: dict, why: str) -> dict:
    """Rule-based stand-in when Gemini is unavailable: same tools, same rules, no LLM."""
    tools = _tools_for(row, a["risk"])
    if not a["flagged"]:
        decision = {"decision": "keep", "new_flight": "",
                    "message_to_passenger": f"Your inbound flight is running late, but you should still make {row['out_flight']} to {row['out_dest']} at {row['out_sched_dep']}. Please head straight to your connecting gate.",
                    "reason_for_desk": f"Risk {a['risk']:.0%} is below the flag threshold."}
    else:
        earliest = _hhmm(_minutes(a["estimated_arrival"][:5]) + MCT)[:5]
        found = tools["find_next_flights"](str(row["out_dest"]), earliest)
        if found["flights"] and found["same_day"]:
            f = found["flights"][0]
            decision = {"decision": "rebook", "new_flight": f["flight"],
                        "message_to_passenger": f"Your inbound flight is running late and you are unlikely to make {row['out_flight']}. We have moved you to {f['flight']} to {row['out_dest']}, departing at {f['departs']}.",
                        "reason_for_desk": f"Risk {a['risk']:.0%}; earliest feasible same-day flight."}
        else:
            f = found["flights"][0] if found["flights"] else {"flight": "", "departs": "tomorrow"}
            decision = {"decision": "overnight", "new_flight": f["flight"],
                        "message_to_passenger": f"Your inbound flight is running late and there is no later flight to {row['out_dest']} today. We have booked you on {f['flight']} at {f['departs']} tomorrow and arranged a hotel voucher.",
                        "reason_for_desk": f"Risk {a['risk']:.0%}; no same-day flight left."}
        decision["offered_flights"] = [x["flight"] for x in found["flights"]]
    return {**decision, "confidence": "low", "mode": f"fallback ({why})", "tool_calls": [], "tokens": 0, "cost_eur": 0.0}


def component_b(row: pd.Series, a: dict) -> dict:
    # Shortcut only when the model AND the clock agree it is safe: a low risk with no
    # minutes to spare (e.g. the connecting plane is late too) still goes to the agent.
    if a["risk"] < AGENT_FROM and a["hub_trained"] and a["minutes_to_spare"] >= 0:
        return {"decision": "keep", "new_flight": "", "confidence": "high", "mode": "low risk, no agent call",
                "message_to_passenger": f"Your connection to {row['out_dest']} on {row['out_flight']} at {row['out_sched_dep']} is on track. Please head to your connecting gate after landing.",
                "reason_for_desk": f"Risk {a['risk']:.0%} is below {AGENT_FROM:.0%}.", "tool_calls": [], "tokens": 0, "cost_eur": 0.0}
    if not os.environ.get("GEMINI_API_KEY", "").strip():
        return _fallback_agent(row, a, "GEMINI_API_KEY not set")
    try:
        return _gemini_with_retries(row, a)
    except Exception as error:  # noqa: BLE001 - the desk still gets an answer, flagged for review
        return _fallback_agent(row, a, f"Gemini unavailable: {str(error)[:300]}")


# --- The product -------------------------------------------------------------------

def review_reasons(a: dict, b: dict) -> list[str]:
    reasons = []
    if not a["hub_trained"]:
        reasons.append(f"hub {a['hub']} is outside the model's training hubs")
    if REVIEW_BAND[0] <= a["risk"] <= REVIEW_BAND[1]:
        reasons.append(f"model unsure (risk {a['risk']:.0%})")
    if b.get("decision") == "overnight":
        reasons.append("overnight stay")
    if b.get("confidence") == "low":
        reasons.append("low confidence")
    if b.get("decision") in ("rebook", "overnight") and b.get("new_flight") == a.get("missing_flight"):
        reasons.append("rebooked onto the flight being missed")
    elif b.get("decision") in ("rebook", "overnight") and b.get("new_flight") not in b.get("offered_flights", []):
        reasons.append("new flight not found in the schedule")
    return reasons


def _out_of_scope(raw_input: str) -> dict:
    return {
        "input_summary": f"not a known connection: {str(raw_input)[:60]!r}",
        "features": {},
        "outputs": {
            "component_a": {"status": "not a connection in today's list"},
            "component_b": {"decision": "out_of_scope", "reply": "I can only help with connecting passengers on today's list of late inbound flights. Please pick a connection from the list.", "mode": "refused"},
        },
        "versions": {"component_a": "-", "component_b": "-"},
        "review": False,
        "tokens": 0,
        "cost_eur": 0.0,
    }


def run(raw_input: str) -> dict:
    found = ID_PATTERN.search(str(raw_input or "").upper())
    if not found or found.group(1) not in connections().index:
        return _out_of_scope(raw_input)
    conn_id = found.group(1)
    row = connections().loc[conn_id]
    row["hub"] = str(row["hub"])

    a = component_a(row)
    b = component_b(row, a)
    reasons = review_reasons(a, b)
    b_out = {k: v for k, v in b.items() if k not in ("tokens", "cost_eur")}
    b_out["reply"] = b.get("message_to_passenger", "")
    b_out["review_reasons"] = reasons

    _, meta = model()
    return {
        "input_summary": (f"{conn_id}: {row['in_flight']} {row['in_origin']}->{row['hub']} left +{int(row['in_dep_delay_min'])} min, "
                          f"connecting {row['out_flight']}->{row['out_dest']} {row['out_sched_dep']}"),
        "features": {name: float(row[name]) for name in DRIFT_FEATURES} | {"risk": float(a["risk"])},
        "outputs": {"component_a": a, "component_b": b_out},
        "versions": {"component_a": str(meta.get("version", "unknown")),
                     "component_b": (f"{b['model']}/{PROMPT_VERSION}" if b["mode"] == "gemini"
                                     else "no-agent-low-risk" if b["mode"].startswith("low risk") else f"rules-fallback/{PROMPT_VERSION}")},
        "review": bool(reasons),
        "tokens": int(b.get("tokens", 0)),
        "cost_eur": float(round(b.get("cost_eur", 0.0), 6)),
    }
