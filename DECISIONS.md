# Decision log

Every choice that shapes the product, why we made it, and what we considered instead.
Status: **proposed** (suggested, not yet reviewed), **chosen by Tomás** (reviewed one by one on 27 Nov, to confirm with the team on Monday), **agreed** (team) or **open**.
The README asks that the decisions be the team's: review each one on Monday and change what you disagree with.

## Product and business case

| # | Decision | Why | Alternatives considered | Status |
|---|---|---|---|---|
| 1 | **Problem:** help an airline's re-accommodation desk act on connecting passengers whose inbound flight is late, *before* they land (keep, rebook or plan an overnight) | Real, frequent decision with a clear baseline and euros; public data exists | Churn retention, parcel damage claims, camera traps, coffee lots, hotel overbooking and others (see brainstorm) | proposed |
| 2 | **Airline and hubs: United at Chicago O'Hare (ORD) and Denver (DEN)** | Highest share of domestic connections among United hubs: DEN 45.8%, ORD 40.3% of domestic boardings (US DOT DB1C survey, July 2026) | Newark + Dulles: Newark only 4.1% (its connections are mostly international); JFK/Logan: mostly origin-destination airports | proposed |
| 3 | **Scope: domestic-to-domestic connections only** | The flight data covers US domestic flights only, so international connections cannot be labelled | Add international flights: needs paid flight-status data; listed as future work | proposed |
| 4 | **Positioning:** the step after United's ConnectionSaver (which decides whether to *hold* a flight) | Holding only fixes small delays; for the rest, rebooking still happens after the miss | Pitch to an airline without such a tool | proposed |
| 5 | **Two baselines:** reactive process (end to end) and the rule "slack − departure delay − 35 min < 0" (component A) | The brief asks for a baseline to beat; the rule is what a desk could do by hand | Only one baseline | proposed |
| 6 | **Value assumptions:** 30% saving per miss handled before landing, €40 per false alarm, 2% of connecting passengers miss, 160 seats × 83.1% load factor | See the business case doc for sources; the first two are our own assumptions | Ranges (15–45%) still to show | proposed |

## Data

| # | Decision | Why | Alternatives considered | Status |
|---|---|---|---|---|
| 7 | **Source:** US DOT Bureau of Transportation Statistics, Reporting Carrier On-Time Performance, Jan 2025 – Mar 2026 | Official, public, flight by flight, no personal data | Kaggle mirrors | agreed |
| 8 | **We build our own dataset:** a "connection" = an inbound and an outbound flight of the same airline at the hub with 35–120 minutes scheduled between them; up to 3 outbound flights sampled per inbound | The raw data has flights, not passengers or connections | All pairs (too many, too correlated) | proposed |
| 9 | **Label:** missed if the inbound's actual arrival + 35 min is later than the outbound's actual departure, or the inbound was diverted | Uses what really happened | — | proposed |
| 10 | **Minimum connection time = 35 minutes** | Typical domestic minimum at large hubs | 30 or 45 minutes | chosen by Tomás |
| 11 | **Dropped:** cancelled inbound flights (no in-flight decision), cancelled outbound flights (a different product), arrivals before 05:00 (clock simplicity), regional partners flying under another code | Keeps the problem clean; each is stated in REPORT.md | Keep them with extra logic | proposed |
| 12 | **Decision moment = the inbound flight has just taken off:** features use only what is known then (its departure delay and taxi-out, the outbound plane's previous leg if it has already left, recent delays at the hub) | Avoids leakage: never the inbound's arrival delay or the outbound's actual departure | Decide before departure (harder, less accurate) | proposed |
| 13 | **Split by time: train on all of 2025, test on Jan–Mar 2026** | Every season is seen in training; the test is genuinely future data, like real use | Random split (leaks the future); train Jan–Sep / test Oct–Dec (unseen holiday season) | agreed |

## Component A: the risk model

| # | Decision | Why | Alternatives considered | Status |
|---|---|---|---|---|
| 14 | **Gradient boosting (HistGradientBoostingClassifier)** | Best average precision on the test (0.841 vs 0.789 logistic regression, 0.726 rule); small enough for Render | Logistic regression | proposed |
| 15 | **No month or date feature** | The model reads current conditions (delays, slack, congestion), so it does not memorise the calendar | Add month (risks seasonal overfitting) | proposed |
| 16 | **Flag threshold 40%**, chosen from `data_prep/threshold_table.py` | Highest value (€9.99M a year, test Jan–Mar 2026), next to the 37% break-even from the costs, and it beats the simple rule on both counts: 65.3 vs 64.3 misses caught and 19.1 vs 37.4 false alarms per 1,000 connections | 45% (same value, fewer misses caught than the rule), 35% (more misses, more false alarms), 50% (fewer misses than the rule) | chosen by Tomás |

## Component B: the rebooking agent

| # | Decision | Why | Alternatives considered | Status |
|---|---|---|---|---|
| 17 | **Gemini Flash: `gemini-3.8-flash` main, `gemini-3.6-flash` backup** (both changeable in `.env`), the model's default thinking level, temperature 0, at most 6 turns | Newest model first. Tested 27 Nov with `data_prep/check_gemini.py`: on the paid key all of 3.8/3.7/3.6 answered in 2–4 s; on the free key 3.8 and 3.7 were busy (503/504) and only 3.6 answered. `.env.example` tells free-key users to set `GEMINI_MODEL=gemini-3.6-flash` | 3.6 main with 3.8 as backup (fast on any key); LOW thinking (faster, fewer tokens) | chosen by Tomás |
| 18 | **Four real tools, forced tool use:** `rebooking_policy`, `compare_costs` (expected cost of keeping vs acting now for this passenger's risk), `find_next_flights` (searches the actual schedule), and `submit_decision` (the decision comes back in a fixed structure) | Genuinely agentic; the cost tool makes its reasoning explicit and ties it to the business case | Three tools without the cost comparison; a seat-availability check (rejected: no seat data, it would be simulated) | chosen by Tomás |
| 19 | **No agent call when risk < 20% and the passenger has minutes to spare** | Saves cost and time on clearly safe connections | Always call the agent; skip only below 10% | chosen by Tomás |
| 20 | **Guardrail added after testing:** a low risk *with no minutes to spare* still goes to the agent | A test case had −33 minutes to spare but 15% risk (the outbound plane was late too); a "you're on track" message there is too risky | — | proposed |
| 21 | **The flight being missed is never offered**, and the agent may only choose flights returned by the search | A simulated run "rebooked" a passenger onto the flight they were missing | — | proposed |
| 22 | **When Gemini is busy or slow:** retry the main model once, then try the backup model; 20 s per call, 60 s in total; after that, the same rules without the LLM answer and the case goes to review. The logs record which model answered | First real runs hit `503 UNAVAILABLE` (model in high demand) and `504 DEADLINE_EXCEEDED`; the fallback worked, but the agent must answer | Switch models only; show an error | chosen by Tomás |

## The app and operations

The online app (Render) uses the paid Gemini key from 27 Nov (chosen by Tomás), for reliable testing and the presentation.

Also chosen by Tomás on 27 Nov: the agent keeps three actions (keep / rebook / overnight), writes short friendly messages in English, the screen stays as built, and the list shows 400 late connections (15% from LAX).

| # | Decision | Why | Alternatives considered | Status |
|---|---|---|---|---|
| 23 | **Input = pick a connection from a list** (its id, e.g. `DEN-001234`); anything else is refused as out of scope | In real use the airline's systems already know the flights; no personal data; easy test cases | Free text read by Gemini; list plus a free-text box | chosen by Tomás |
| 24 | **App data = a sample of the test months:** 1,500 connections per hub (a quarter of them missed, over-sampled so the demo shows risky cases) + 600 from LAX, plus the schedule. The evaluation always uses the full test set | Render has 512 MB; the full year does not fit | The real mix (about 8% missed) | chosen by Tomás |
| 25 | **Review rule:** a person checks the case if the hub is not ORD/DEN, the risk is 30–50%, the decision is an overnight, or the agent's confidence is low; plus two safety checks (a flight not in the schedule, a rebooking onto the flight being missed) | Sends the uncertain and the costly cases to a person. **Band changed from 30–70% to 30–50%**, centred on the 40% threshold: the wide band covered 3.9% of connections, up to about 1,200 cases a day | Keep 30–70%; ±5 points (35–45%) | chosen by Tomás |
| 26 | **Drift set: LAX, same months as the test** | Different flight lengths, congestion and weather; same months, so the shift is the hub, not the season | Newark during its spring-2025 crisis | chosen by Tomás |
| 27 | **Drift features: continuous ones only** (slack, departure delay, flight time, distance, recent hub delay, scheduled arrivals per hour) **plus the risk** | `monitor.py` treats a feature with 5 or fewer distinct values as categories, and any new value then looks like drift (false alarms) | Fewer, the clearest 3–4 | chosen by Tomás |
| 28 | **One set of costs everywhere:** the agent's `compare_costs` tool uses the business case's numbers (€225 per miss handled after landing, 30% saved by acting early, €40 per false alarm); break-even risk ≈ 37% | Consistent numbers in the app, the report and the presentation | Separate agent costs (€15 rebooking, €180 overnight) | chosen by Tomás |

## Still open

- Rerun `data_prep/threshold_table.py` to see how many reviews a day the 30–50% band gives (#25)
- Ranges for the value assumptions (#6)
- Product name
- Test cases (`eval/cases.csv`, 30+), reference and shifted sets, LLM judge and our 20+ labels
- RL bonus: a bandit that learns the threshold or the message style from feedback

## Limitations (for the report)

- **No seat availability:** the public data has no seat inventory, so the agent assumes later flights have free seats. In real use, `find_next_flights` would connect to the airline's booking system and only offer flights with seats.
- **Domestic only:** the BTS data covers US domestic flights, so connections to or from international flights are out of scope (future work: commercial flight-status data).
- **Connections are inferred, not real passengers:** every flight pair with 35–120 minutes between them counts as a connection; the airline would use its actual bookings.
- **United only, two hubs:** the model is trained at ORD and DEN; other hubs need their own data (the LAX drift demo shows why).
- **Estimated arrival = scheduled arrival + departure delay:** planes often recover some time in the air, so this is cautious.
- **No passenger details:** no names, loyalty status, bags or fares, so every passenger is treated the same.
- **Cost figures are partly assumptions:** the 30% saving and the €40 false-alarm cost are ours (see the business case).
