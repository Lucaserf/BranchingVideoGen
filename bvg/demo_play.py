"""Play a story against a running server without a browser, then wait for the film.

    python -m bvg.demo_play --url http://127.0.0.1:8000 --style anime --length short
"""

from __future__ import annotations

import argparse
import json
import random
import time
import urllib.request


def call(url: str, body: dict | None = None) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"}, method="POST" if body is not None else "GET")
    with urllib.request.urlopen(req) as res:
        return json.load(res)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--style", default="animated")
    parser.add_argument("--length", default="short")
    parser.add_argument("--profile", default="fast")
    parser.add_argument("--seconds-per-choice", type=float, default=20, help="pause between choices, like a viewer")
    parser.add_argument("--free-text", default=None, help="use this as the answer for act 3 instead of an option")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    meta = call(f"{args.url}/api/meta")
    s = call(f"{args.url}/api/sessions", {"style": args.style, "length": args.length, "profile": args.profile})
    print(f"session {s['id']}: {s['shots_total']} shots")
    for act_index, act in enumerate(meta["acts"]):
        time.sleep(args.seconds_per_choice)
        if args.free_text and act_index == 2:
            body = {"text": args.free_text}
        else:
            body = {"option": rng.randrange(len(act["options"]))}
        s = call(f"{args.url}/api/sessions/{s['id']}/choices", body)
        print(f"act {act_index + 1}: {s['choices'][-1]}  ({s['shots_done']}/{s['shots_total']} shots done)")
    while s["film_status"] not in ("done", "failed"):
        time.sleep(5)
        s = call(f"{args.url}/api/sessions/{s['id']}")
        print(f"  {s['shots_done']}/{s['shots_total']} shots, film {s['film_status']}, ~{s['eta_seconds']}s left")
    print(json.dumps({k: s[k] for k in ("film_status", "film", "wait_after_last_choice", "values", "error")}, indent=2))


if __name__ == "__main__":
    main()
