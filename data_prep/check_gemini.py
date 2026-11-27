"""
How fast does Gemini answer with your key? A quick diagnostic, not part of the app.

    uv run python data_prep/check_gemini.py

Sends one tiny request to each model, with the default thinking and with thinking set to
LOW, and prints the time each took or the error. Costs a fraction of a cent.
"""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import env  # noqa: E402

MODELS = ["gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash"]
PROMPT = "A passenger lands at 10:05 and their connecting flight leaves at 10:20. Minimum connection time is 35 minutes. Reply with one word: keep or rebook."


def main() -> None:
    env.load_env()
    if not os.environ.get("GEMINI_API_KEY", "").strip():
        raise SystemExit("GEMINI_API_KEY is not set in .env")
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"], http_options=types.HttpOptions(timeout=60_000))
    print(f"{'model':<20} {'thinking':<9} {'seconds':>8}  result")
    for model in MODELS:
        for level in (None, "LOW"):
            config = types.GenerateContentConfig(temperature=0)
            if level:
                config.thinking_config = types.ThinkingConfig(thinking_level=level)
            started = time.perf_counter()
            try:
                response = client.models.generate_content(model=model, contents=PROMPT, config=config)
                usage = response.usage_metadata
                result = f"{(response.text or '').strip()[:20]!r}  ({usage.thoughts_token_count or 0} thinking tokens)"
            except Exception as error:  # noqa: BLE001
                result = f"ERROR {type(error).__name__}: {str(error)[:90]}"
            print(f"{model:<20} {level or 'default':<9} {time.perf_counter() - started:>8.1f}  {result}", flush=True)


if __name__ == "__main__":
    main()
