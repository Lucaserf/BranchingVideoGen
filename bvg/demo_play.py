"""Play a story against a running server without a browser, then wait for the film.

    python -m bvg.demo_play --url http://127.0.0.1:8000 --style anime --shots 18
"""

from __future__ import annotations

import argparse
import base64
import json
import random
import time
import urllib.request
from pathlib import Path


def call(url: str, body: dict | None = None) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"}, method="POST" if body is not None else "GET")
    with urllib.request.urlopen(req) as res:
        return json.load(res)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--style", default="animated")
    parser.add_argument("--shots", type=int, default=18, help="number of shots, each film.shot_seconds long")
    parser.add_argument("--profile", default="fast")
    parser.add_argument("--seconds-per-choice", type=float, default=20, help="pause between choices, like a viewer")
    parser.add_argument("--free-text", default=None, help="use this as the answer for act 3 instead of an option")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--voice", default="off", help="narration voice id, or off")
    parser.add_argument("--no-music", action="store_true")
    parser.add_argument("--photo", default=None, help="image file to use as the protagonist (you confirm consent)")
    parser.add_argument("--model", action="append", default=[], metavar="STAGE=MODEL")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    meta = call(f"{args.url}/api/meta")
    body = {"style": args.style, "shots": args.shots, "profile": args.profile, "voice": args.voice,
            "music": not args.no_music, "models": dict(m.split("=", 1) for m in args.model)}
    if args.photo:
        data = base64.b64encode(Path(args.photo).read_bytes()).decode()
        body.update(photo=f"data:image/jpeg;base64,{data}", photo_consent=True)
    s = call(f"{args.url}/api/sessions", body)
    print(f"session {s['id']}: {s['shots_total']} shots")
    for act_index in range(len(meta["beats"])):
        while s["current"] is None and s["film_status"] != "failed":  # the writer is still writing
            time.sleep(1)
            s = call(f"{args.url}/api/sessions/{s['id']}")
        if s["film_status"] == "failed":
            break
        print(f"\nact {act_index + 1} ({s['current']['beat']}): {s['current']['intro']}\n  {s['current']['question']}")
        for i, label in enumerate(s["current"]["options"]):
            print(f"    {i + 1}. {label}")
        time.sleep(args.seconds_per_choice)
        if args.free_text and act_index == 2:
            choice = {"text": args.free_text}
        else:
            choice = {"option": rng.randrange(len(s["current"]["options"]))}
        s = call(f"{args.url}/api/sessions/{s['id']}/choices", choice)
        print(f"  -> {s['choices'][-1]}  ({s['shots_done']}/{s['shots_total']} shots done)")
    while s["film_status"] not in ("done", "failed"):
        time.sleep(5)
        s = call(f"{args.url}/api/sessions/{s['id']}")
        print(f"  {s['shots_done']}/{s['shots_total']} shots, film {s['film_status']}, ~{s['eta_seconds']}s left")
    print(json.dumps({k: s[k] for k in ("title", "film_status", "film", "wait_after_last_choice", "values", "writer", "error")}, indent=2))


if __name__ == "__main__":
    main()
