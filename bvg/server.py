"""Render server and web app for the demo.

    python -m bvg.server --mock            # anywhere, placeholder images and pans
    python -m bvg.server --host 0.0.0.0    # on the L40

One viewer holds the GPU at a time. Each choice locks an act: its keyframes are
rendered first (so the viewer sees stills right away), then its video shots, in
the order the planner chose. After the last act the shots are stitched into the film.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import queue
import threading
import time
import uuid
import zlib
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from bvg.config import load_config, load_costs, load_profile
from bvg.engine import Engine
from bvg.models import apply_overrides
from bvg.planner import Shot, plan_film
from bvg.story import ACTS, STYLES, StoryState, acts_public

log = logging.getLogger("bvg.server")
STATIC = Path(__file__).resolve().parent / "static"

# Which shots of each act a short film keeps.
LENGTHS = {"full": (0, 1, 2, 3, 4, 5), "short": (0, 3, 5)}
KEYFRAME, VIDEO, FINISH = 0, 1, 2  # job priorities: lower runs first


@dataclass
class ShotState:
    act: int
    index: int
    render_type: str
    prompt: str = ""
    narration: str = ""
    status: str = "waiting"  # waiting -> queued -> keyframe -> rendering -> done | failed
    seconds: float | None = None


@dataclass
class Session:
    id: str
    story: StoryState
    profile: str
    length: str
    shots: dict[tuple[int, int], ShotState]
    dir: Path
    created: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    last_choice_at: float | None = None
    film_status: str = "waiting"  # waiting -> stitching -> done | failed
    film_ready_at: float | None = None
    error: str | None = None

    @property
    def finished(self) -> bool:
        return self.film_status in ("done", "failed")


class ChoiceIn(BaseModel):
    option: int | None = None
    text: str | None = None


class SessionIn(BaseModel):
    style: str = "animated"
    profile: str = "fast"
    length: str = "full"


class App:
    def __init__(self, config: dict, engine: Engine, bench: str | None):
        self.config = config
        self.engine = engine
        self.costs = load_costs(config, bench)
        server = config.get("server", {})
        self.idle_seconds = server.get("idle_lease_minutes", 15) * 60
        self.outputs = Path(server.get("outputs", "outputs/sessions"))
        self.sessions: dict[str, Session] = {}
        self.jobs: queue.PriorityQueue = queue.PriorityQueue()
        self.seq = itertools.count()
        self.lock = threading.Lock()
        threading.Thread(target=self._worker, daemon=True).start()

    # --- sessions ---------------------------------------------------------

    def active(self) -> Session | None:
        """The session holding the GPU: not finished and seen within the idle lease."""
        now = time.time()
        for s in self.sessions.values():
            if not s.finished and now - s.last_seen <= self.idle_seconds:
                return s
        return None

    def create(self, body: SessionIn) -> Session:
        if body.style not in STYLES:
            raise HTTPException(400, f"unknown style {body.style!r}")
        if body.length not in LENGTHS:
            raise HTTPException(400, f"unknown length {body.length!r}")
        with self.lock:
            busy = self.active()
            if busy:
                raise HTTPException(409, {"busy": True, "eta_seconds": self.eta(busy)})
            profile = load_profile(self.config, body.profile)
            keep = LENGTHS[body.length]
            planned = [Shot(act=a, index=i, priority=t.priority)
                       for a, act in enumerate(ACTS) for i, t in enumerate(act.shots) if i in keep]
            plan = plan_film(planned, self.costs, profile, self.config["film"]["seconds_per_choice"])
            sid = uuid.uuid4().hex[:10]
            session = Session(id=sid, story=StoryState(style=body.style), profile=body.profile, length=body.length,
                              shots={(s.act, s.index): ShotState(s.act, s.index, s.render_type) for s in plan.shots},
                              dir=self.outputs / sid)
            session.dir.mkdir(parents=True, exist_ok=True)
            self.sessions[sid] = session
            log.info("session %s: %s, %s, %s, plan %s", sid, body.style, body.profile, body.length, plan.counts())
            return session

    def get(self, sid: str) -> Session:
        session = self.sessions.get(sid)
        if session is None:
            raise HTTPException(404, "no such session")
        session.last_seen = time.time()
        return session

    def choose(self, sid: str, body: ChoiceIn) -> int:
        session = self.get(sid)
        with self.lock:
            if session.story.act >= len(ACTS):
                raise HTTPException(400, "the story is already complete")
            if body.option is None and not (body.text and body.text.strip()):
                raise HTTPException(400, "pick an option or write your own")
            if body.option is not None and not 0 <= body.option < len(ACTS[session.story.act].options):
                raise HTTPException(400, "no such option")
            act = session.story.choose(body.option, body.text)
            session.last_choice_at = time.time()
            for spec in session.story.shots(act):
                shot = session.shots.get((act, spec["index"]))
                if shot is None:
                    continue  # not part of a short film
                shot.prompt, shot.narration, shot.status = spec["prompt"], spec["narration"], "queued"
                self.jobs.put((KEYFRAME, next(self.seq), sid, act, shot.index))
                self.jobs.put((VIDEO, next(self.seq), sid, act, shot.index))
            if session.story.act == len(ACTS):
                self.jobs.put((FINISH, next(self.seq), sid, -1, -1))
            self._save(session)
            return act

    def eta(self, session: Session) -> float:
        """Seconds of GPU work left for a session, from the planner's per-shot costs."""
        left = 0.0
        for shot in session.shots.values():
            if shot.status in ("waiting", "queued"):
                left += self.costs.keyframe
            if shot.status != "done":
                left += self.costs.shot[shot.render_type]
        return left + (0 if session.finished else self.costs.finish)

    def public(self, session: Session) -> dict:
        def url(name):
            return f"/files/{session.id}/{name}"

        shots = []
        for (a, i), s in sorted(session.shots.items()):
            kf = session.dir / f"kf_{a}_{i}.png"
            shots.append({"act": a, "index": i, "render_type": s.render_type, "status": s.status,
                          "narration": s.narration, "seconds": s.seconds,
                          "keyframe": url(kf.name) if kf.exists() else None})
        done = sum(s.status == "done" for s in session.shots.values())
        return {
            "id": session.id, "style": session.story.style, "profile": session.profile, "length": session.length,
            "act": session.story.act, "choices": session.story.choices, "values": session.story.values,
            "shots": shots, "shots_done": done, "shots_total": len(session.shots),
            "film_status": session.film_status, "film": url("film.mp4") if session.film_status == "done" else None,
            "eta_seconds": round(self.eta(session)),
            "wait_after_last_choice": (round(session.film_ready_at - session.last_choice_at)
                                       if session.film_ready_at and session.last_choice_at else None),
            "error": session.error,
        }

    def _save(self, session: Session) -> None:
        (session.dir / "session.json").write_text(json.dumps(self.public(session), indent=2))

    # --- GPU worker -------------------------------------------------------

    def _worker(self) -> None:
        while True:
            kind, _, sid, act, index = self.jobs.get()
            session = self.sessions[sid]
            try:
                if kind == FINISH:
                    self._finish(session)
                else:
                    self._render(session, kind, session.shots[(act, index)])
            except Exception as e:  # keep serving other jobs
                log.exception("job failed")
                session.error = f"{type(e).__name__}: {e}"
                if kind == FINISH:
                    session.film_status = "failed"
                else:
                    session.shots[(act, index)].status = "failed"
            self._save(session)

    def _render(self, session: Session, kind: int, shot: ShotState) -> None:
        self.engine.load({s.render_type for s in session.shots.values()} - {"pan"})
        seed = zlib.crc32(f"{session.id}:{shot.act}:{shot.index}".encode())
        kf_path = session.dir / f"kf_{shot.act}_{shot.index}.png"
        start = time.time()
        if kind == KEYFRAME:
            shot.status = "keyframe"
            self.engine.keyframe(shot.prompt, seed).save(kf_path)
            shot.status = "rendering"
        else:
            from PIL import Image

            self.engine.shot(shot.render_type, Image.open(kf_path), shot.prompt, shot.narration, seed,
                             session.dir / f"shot_{shot.act}_{shot.index}.mp4")
            shot.status, shot.seconds = "done", round(time.time() - start, 1)
            log.info("session %s: shot %d.%d %s in %.1fs", session.id, shot.act + 1, shot.index + 1,
                     shot.render_type, shot.seconds)

    def _finish(self, session: Session) -> None:
        session.film_status = "stitching"
        paths = [session.dir / f"shot_{a}_{i}.mp4" for (a, i), s in sorted(session.shots.items())
                 if s.status == "done"]
        if not paths:
            raise RuntimeError("no shot rendered")
        self.engine.stitch(paths, session.dir / "film.mp4")
        session.film_status, session.film_ready_at = "done", time.time()
        log.info("session %s: film ready %.0fs after the last choice", session.id,
                 session.film_ready_at - session.last_choice_at)


def build(app_state: App) -> FastAPI:
    api = FastAPI(title="BranchingVideoGen demo")

    @api.get("/api/meta")
    def meta():
        busy = app_state.active()
        return {"styles": {k: v["label"] for k, v in STYLES.items()}, "acts": acts_public(),
                "profiles": {k: v["target_wait_minutes"] for k, v in app_state.config["profiles"].items()},
                "lengths": list(LENGTHS), "busy": bool(busy), "eta_seconds": app_state.eta(busy) if busy else 0,
                "mock": app_state.engine.mock}

    @api.post("/api/sessions", status_code=201)
    def create(body: SessionIn):
        return app_state.public(app_state.create(body))

    @api.get("/api/sessions/{sid}")
    def status(sid: str):
        return app_state.public(app_state.get(sid))

    @api.post("/api/sessions/{sid}/choices")
    def choose(sid: str, body: ChoiceIn):
        app_state.choose(sid, body)
        return app_state.public(app_state.get(sid))

    @api.get("/files/{sid}/{name}")
    def files(sid: str, name: str):
        session = app_state.get(sid)
        path = (session.dir / name).resolve()
        if path.parent != session.dir.resolve() or not path.exists():
            raise HTTPException(404, "no such file")
        return FileResponse(path)

    api.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
    return api


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--bench", default=None, help="benchmark results JSON with measured costs")
    parser.add_argument("--model", action="append", default=[], metavar="STAGE=MODEL")
    parser.add_argument("--mock", action="store_true", help="placeholder images and pans, no GPU needed")
    parser.add_argument("--preload", action="store_true", help="load the models at startup instead of on first use")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = apply_overrides(load_config(args.config) if args.config else load_config(), args.model)
    engine = Engine(config, mock=args.mock)
    if args.preload:
        engine.load({"standard"})
    import uvicorn

    uvicorn.run(build(App(config, engine, args.bench)), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
