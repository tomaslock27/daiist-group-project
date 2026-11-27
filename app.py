"""
The app: an App tab and a Monitoring tab. `uv run python app.py`

Keep the logic in pipeline.py (or your own modules); this file only wires it to
the interface and the logs. Keep `serve()` and the `/run` endpoint: check.py
calls `serve()` locally and `/run` on your deployed app, with one input from
eval/cases.csv (text, or a file path).
"""

import os
import time

import gradio as gr

import env
import logbook
import monitor
import pipeline

env.load_env()
LOG_TARGET = logbook.start()

TITLE = "Connection Rescue"  # working name: the team can change it
DESCRIPTION = ("For an airline's re-accommodation desk at Chicago O'Hare and Denver: when an inbound flight runs late, "
               "it predicts which connecting passengers will miss their next flight and plans what to do before they land.")


def serve(raw_input: str) -> dict:
    """Run one request through the pipeline, log it, and return the logged record."""
    started = time.perf_counter()
    result = pipeline.run(raw_input)
    return logbook.log_request(result | {"latency_s": round(time.perf_counter() - started, 3)})


# --- App tab -----------------------------------------------------------------

DECISION_LABELS = {"keep": "Keep the booking", "rebook": "Rebook now", "overnight": "Plan an overnight", "out_of_scope": "Not a connection I can help with"}


def on_run(raw_input):
    if raw_input is None or not str(raw_input).strip():
        raise gr.Error("Pick a connection first.")
    record = serve(raw_input)
    a, b = record["outputs"]["component_a"], record["outputs"]["component_b"]
    if "risk" in a:
        risk_text = (f"### Risk of missing the connection: {a['risk']:.0%}"
                     f"{'  ⚠️ flagged' if a['flagged'] else ''}\n"
                     f"Estimated arrival **{a['estimated_arrival']}** · connection departs **{a['connection_departs']}** · "
                     f"**{a['minutes_to_spare']} min** to spare after the minimum connection time · hub {a['hub']}")
    else:
        risk_text = "### Risk: not scored\nThis input is not a connection on today's list."
    decision = DECISION_LABELS.get(b.get("decision"), str(b.get("decision")))
    flight = f" → **{b['new_flight']}**" if b.get("new_flight") else ""
    decision_text = f"### {decision}{flight}\n{b.get('reason_for_desk', '')}\n\n<small>Decided by: {b.get('mode', '-')}{' · ' + b['model'] if b.get('model') else ''}</small>"
    if record["review"]:
        status = "⚠️ **Sent to a person for review:** " + "; ".join(b.get("review_reasons", [])) + "."
    else:
        status = "✅ Handled automatically."
    status += f"  Latency {record['latency_s']} s, cost €{record['cost_eur']:.4f}."
    return risk_text, decision_text, b.get("reply", ""), record["outputs"], status, record["id"]


def on_feedback(request_id, value):
    if not request_id:
        gr.Warning("Run a request first.")
        return
    logbook.log_feedback(request_id, value)
    gr.Info("Thanks for the feedback.")


# --- Monitoring tab ----------------------------------------------------------

def on_refresh():
    logs = logbook.load()
    reference = monitor.load_reference()
    fired = monitor.alerts(logs, reference) if not logs.empty else []
    if reference is None:
        alert_text = "No reference set yet: run `uv run python main.py reference` and commit `monitoring/`."
    elif fired:
        alert_text = "### 🚨 Alerts\n" + "\n".join(f"- {message}" for message in fired)
    else:
        alert_text = "### ✅ No alerts"
    if reference is not None and len(logs) >= monitor.MIN_WINDOW:
        drift_table = monitor.drift(reference, monitor.live_features(logs))
    else:
        drift_table = monitor.drift(reference, reference.head(0)) if reference is not None else None
    return alert_text, monitor.summary(logs), monitor.quality(logs), drift_table


with gr.Blocks(title=TITLE) as demo:
    gr.Markdown(f"# {TITLE}\n{DESCRIPTION}")
    if missing := env.missing():
        gr.Markdown(f"⚠️ **Missing environment variables:** {', '.join(missing)}. See SETUP.md.")

    with gr.Tab("App"):
        request_id = gr.State("")
        raw_input = gr.Dropdown(
            choices=pipeline.demo_choices(), label="Connection to check (inbound flights running late)",
            allow_custom_value=True, filterable=True,
        )
        run_button = gr.Button("Run", variant="primary")
        with gr.Row():
            risk_box = gr.Markdown(label="Component A: risk model")
            decision_box = gr.Markdown(label="Component B: rebooking agent")
        message_box = gr.Textbox(label="Message to the passenger", lines=3)
        status = gr.Markdown()
        with gr.Row():
            up = gr.Button("👍")
            down = gr.Button("👎")
        with gr.Accordion("Details (what both components returned)", open=False):
            details = gr.JSON(label="Outputs")
        run_button.click(on_run, raw_input, [risk_box, decision_box, message_box, details, status, request_id], api_name="run")
        up.click(lambda rid: on_feedback(rid, +1), request_id, None, api_name=False)
        down.click(lambda rid: on_feedback(rid, -1), request_id, None, api_name=False)

    with gr.Tab("Monitoring"):
        refresh = gr.Button("Refresh")
        alert_box = gr.Markdown()
        summary_table = gr.Dataframe(label="Summary", interactive=False)
        quality_plot = gr.LinePlot(x="time", y="value", color="signal", label="Quality over time", height=300)
        drift_table = gr.Dataframe(label=f"Drift: PSI of the last {monitor.WINDOW} requests against the reference set", interactive=False)
        gr.Markdown(f"<small>{LOG_TARGET}</small>")
        outputs = [alert_box, summary_table, quality_plot, drift_table]
        refresh.click(on_refresh, None, outputs, api_name=False)
        demo.load(on_refresh, None, outputs, api_name=False)


if __name__ == "__main__":
    # Render gives the port in $PORT; locally it is 7860.
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
