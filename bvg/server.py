"""Render server and web app for the demo.

    python -m bvg.server --mock            # anywhere, placeholder images and pans
    python -m bvg.server --host 0.0.0.0    # on the L40

One viewer holds the GPU at a time. Each choice locks an act: its keyframes are
rendered first (so the viewer sees stills right away), then its video shots, in
the order the planner chose. Narration is spoken on the CPU in parallel. After the
last act the shots are stitched, and narration and music are mixed in.

Each film can pick its own models (image, video, quality shots, photo edit), a
narration voice, music, and a photo of the protagonist.

The story itself is written as the viewer plays (bvg.writer): an LLM writes each
act's scene, question and choices from the answers so far, then the act's shot
prompts. Writer calls run on their own thread so they never block the GPU queue.
"""

from __future__ import annotations

import argparse
import base64
import copy
import importlib.util
import io
import itertools
import json
import logging
import queue
import threading
import time
import uuid
import zlib
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import numpy as np
from PIL import Image

from bvg import audio
from bvg.config import Costs, load_config, load_costs, load_profile
from bvg.engine import Engine
from bvg.models import (TOP_LEVEL_STAGES, ModelPolicyError, apply_overrides, models_by_kind, select_model,
                        stage_settings)
from bvg.planner import Shot, plan_film
from bvg.story import BEATS, STYLES, StoryState, beats_public, split_shots
from bvg.writer import make_writer

log = logging.getLogger("bvg.server")
STATIC = Path(__file__).resolve().parent / "static"

KEYFRAME, VIDEO, FINISH = 0, 1, 2  # job priorities: lower runs first
MODEL_STAGES = ("keyframe", "standard", "hero", "photo", "story")
MAX_PHOTO_BYTES = 12 * 1024 * 1024


@dataclass
class ShotState:
    act: int
    index: int
    render_type: str
    prompt: str = ""
    photo_prompt: str | None = None
    narration: str = ""
    status: str = "waiting"  # waiting -> queued -> keyframe -> rendering -> done | failed
    seconds: float | None = None


@dataclass
class Session:
    id: str
    story: StoryState
    profile: str
    shot_total: int
    shots: dict[tuple[int, int], ShotState]
    dir: Path
    settings: dict[str, dict]  # per stage, from this film's model picks
    costs: Costs
    voice: str | None = None  # None: no narration
    music: bool = True
    captions: bool = True
    photo: Image.Image | None = None
    narration_jobs: dict[tuple[int, int], Future] = field(default_factory=dict)
    created: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    last_choice_at: float | None = None
    film_status: str = "waiting"  # waiting -> stitching -> done | failed
    film_ready_at: float | None = None
    writer: object = None  # this film's story writer (bvg.writer)
    writer_steps: list[str] = field(default_factory=list)  # "llm", "fallback" or "simple", per writer call
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
    shots: int | None = None  # number of shots (prompts), each film.shot_seconds long; None: film.shots
    models: dict[str, str] = {}  # stage -> catalog model, for this film only
    voice: str | None = None  # a narration voice id, or None / "off" for none
    music: bool = True
    captions: bool = True
    photo: str | None = None  # data URL of the protagonist's photo
    photo_consent: bool = False


class App:
    def __init__(self, config: dict, engine: Engine, bench: str | None, writer=None):
        self.config = config
        self.engine = engine
        self.writer = writer  # a fixed writer for every film (tests); otherwise one per film from story settings
        self.writer_jobs: queue.Queue = queue.Queue()
        self.bench = bench
        self.costs = load_costs(config, bench)
        self.audio = config.get("audio", {})
        self.narrator = audio.Narrator(self.audio.get("narration", "kokoro"), mock=engine.mock)
        self.cpu = ThreadPoolExecutor(max_workers=1, thread_name_prefix="narration")
        server = config.get("server", {})
        self.idle_seconds = server.get("idle_lease_minutes", 15) * 60
        self.outputs = Path(server.get("outputs", "outputs/sessions"))
        self.sessions: dict[str, Session] = {}
        self.jobs: queue.PriorityQueue = queue.PriorityQueue()
        self.seq = itertools.count()
        self.lock = threading.Lock()
        threading.Thread(target=self._worker, daemon=True).start()
        threading.Thread(target=self._writer_worker, daemon=True).start()

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
        film = self.config["film"]
        total = film["shots"] if body.shots is None else body.shots
        if not film.get("min_shots", 6) <= total <= film.get("max_shots", 72):
            raise HTTPException(400, f"shots must be between {film.get('min_shots', 6)} and {film.get('max_shots', 72)}")
        with self.lock:
            busy = self.active()
            if busy:
                raise HTTPException(409, {"busy": True, "eta_seconds": self.eta(busy)})
            settings, costs = self._film_models(body.models)
            photo = self._photo(body)
            voice = None if body.voice in (None, "", "off") else body.voice
            if voice is not None and voice not in audio.VOICES:
                raise HTTPException(400, f"unknown voice {voice!r}")
            profile = load_profile(self.config, body.profile)
            counts = split_shots(total)
            # Shot priorities aren't written yet; plan with the first and last shot of each act weighted higher.
            planned = [Shot(act=a, index=i, priority=0.8 if i in (0, n - 1) else 0.5)
                       for a, n in enumerate(counts) for i in range(n)]
            plan = plan_film(planned, costs, profile, self.config["film"]["seconds_per_choice"])
            sid = uuid.uuid4().hex[:10]
            story = StoryState(style=body.style, shot_counts=counts, has_photo=photo is not None)
            session = Session(id=sid, story=story, profile=body.profile, shot_total=total,
                              shots={(s.act, s.index): ShotState(s.act, s.index, s.render_type) for s in plan.shots},
                              dir=self.outputs / sid, settings=settings, costs=costs, voice=voice,
                              music=body.music, captions=body.captions, photo=photo,
                              writer=self.writer or make_writer(settings["story"]))
            session.dir.mkdir(parents=True, exist_ok=True)
            if photo is not None:
                photo.save(session.dir / "photo.png")
            self.sessions[sid] = session
            self.writer_jobs.put((sid, -1))
            log.info("session %s: %s, %s, %s, models %s, voice %s, photo %s, plan %s", sid, body.style,
                     body.profile, f"{total} shots", {k: v.get("model_name") for k, v in settings.items()}, voice,
                     photo is not None, plan.counts())
            return session

    def _film_models(self, picks: dict[str, str]) -> tuple[dict[str, dict], Costs]:
        """Settings per stage and planning costs for this film's model picks."""
        config = copy.deepcopy(self.config)
        try:
            for stage, model in picks.items():
                if stage not in MODEL_STAGES:
                    raise KeyError(f"unknown stage {stage!r}")
                select_model(config, stage, model)
            settings = {stage: stage_settings(config, stage) for stage in MODEL_STAGES}
        except (KeyError, ValueError, ModelPolicyError) as e:
            raise HTTPException(400, str(e).strip("'\""))
        settings["pan"] = stage_settings(config, "pan")
        return settings, load_costs(config, self.bench)

    @staticmethod
    def _photo(body: SessionIn) -> Image.Image | None:
        if not body.photo:
            return None
        if not body.photo_consent:
            raise HTTPException(400, "confirm the photo is of you, or that you have the person's permission")
        data = body.photo.split(",", 1)[-1]
        if len(data) > MAX_PHOTO_BYTES * 4 // 3:
            raise HTTPException(400, "photo too large")
        try:
            image = Image.open(io.BytesIO(base64.b64decode(data)))
            image.load()
        except Exception:
            raise HTTPException(400, "could not read the photo")
        image = image.convert("RGB")
        image.thumbnail((1024, 1024))
        return image

    def get(self, sid: str) -> Session:
        session = self.sessions.get(sid)
        if session is None:
            raise HTTPException(404, "no such session")
        session.last_seen = time.time()
        return session

    def choose(self, sid: str, body: ChoiceIn) -> int:
        session = self.get(sid)
        with self.lock:
            if session.story.act >= len(BEATS):
                raise HTTPException(400, "the story is already complete")
            current = session.story.current
            if current is None:
                raise HTTPException(409, "the next act is still being written")
            if body.option is None and not (body.text and body.text.strip()):
                raise HTTPException(400, "pick an option or write your own")
            if body.option is not None and not 0 <= body.option < len(current.options):
                raise HTTPException(400, "no such option")
            act = session.story.choose(body.option, body.text)
            session.last_choice_at = time.time()
            self.writer_jobs.put((sid, act))
            self._save(session)
            return act

    # --- writer -----------------------------------------------------------

    def _writer_worker(self) -> None:
        while True:
            sid, answered = self.writer_jobs.get()
            session = self.sessions[sid]
            try:
                if answered < 0:
                    session.writer_steps.append(session.writer.opening(session.story) or "llm")
                else:
                    self._write_act(session, answered)
            except Exception as e:
                log.exception("writer failed")
                session.error = f"The story could not be written: {e}"
                session.film_status = "failed"
            self._save(session)

    def _write_act(self, session: Session, act: int) -> None:
        """After an answer: write the next act (the viewer is waiting), then this act's shots (the GPU is)."""
        session.writer_steps.append(session.writer.next_act(session.story, act) or "llm")
        session.writer_steps.append(session.writer.shots(session.story, act) or "llm")
        for spec in session.story.prompts(act):
            shot = session.shots[(act, spec["index"])]
            shot.prompt, shot.narration, shot.status = spec["prompt"], spec["narration"], "queued"
            shot.photo_prompt = spec["photo_prompt"]
            if session.voice:
                session.narration_jobs[(act, shot.index)] = self.cpu.submit(
                    self.narrator.speak, shot.narration, session.voice)
            self.jobs.put((KEYFRAME, next(self.seq), session.id, act, shot.index))
            self.jobs.put((VIDEO, next(self.seq), session.id, act, shot.index))
        log.info("session %s: act %d written (%s)", session.id, act + 1, ", ".join(session.writer_steps[-2:]))
        if act == len(BEATS) - 1:
            self.jobs.put((FINISH, next(self.seq), session.id, -1, -1))

    def eta(self, session: Session) -> float:
        """Seconds of GPU work left for a session, from the planner's per-shot costs."""
        left = 0.0
        for shot in session.shots.values():
            if shot.status in ("waiting", "queued"):
                left += session.costs.keyframe
            if shot.status != "done":
                left += session.costs.shot[shot.render_type]
        return left + (0 if session.finished else session.costs.finish)

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
        story = session.story
        current = story.current
        return {
            "id": session.id, "style": story.style, "profile": session.profile, "shot_counts": story.shot_counts,
            "film_seconds": session.shot_total * self.config["film"]["shot_seconds"],
            "title": story.title, "act": story.act, "choices": story.choices, "values": story.values,
            "current": None if current is None else {
                "beat": BEATS[story.act].title, "intro": current.intro, "question": current.question,
                "options": [o.label for o in current.options]},
            "writing": story.act < len(BEATS) and current is None and session.film_status != "failed",
            "story": [{"beat": BEATS[i].title, "intro": a.text.intro, "answer": a.answer}
                      for i, a in enumerate(story.acts) if a.answer is not None],
            "writer": next((w for w in ("fallback", "simple", "llm") if w in session.writer_steps), None),
            "models": {k: v.get("model_name") for k, v in session.settings.items() if k != "pan"},
            "voice": session.voice, "music": session.music, "photo": session.photo is not None,
            "shots": shots, "shots_done": done, "shots_total": len(session.shots),
            "film_status": session.film_status, "film": url("film.mp4") if session.film_status == "done" else None,
            "eta_seconds": round(self.eta(session)),
            "wait_after_last_choice": (round(session.film_ready_at - session.last_choice_at)
                                       if session.film_ready_at and session.last_choice_at else None),
            "error": session.error,
        }

    def _save(self, session: Session) -> None:
        (session.dir / "session.json").write_text(json.dumps(self.public(session), indent=2))

    def _soundtrack(self, session: Session, done: list[tuple[int, int]]) -> np.ndarray:
        shot_seconds = self.config["film"]["shot_seconds"]
        seconds = len(done) * shot_seconds
        lines = []
        for n, key in enumerate(done):
            job = session.narration_jobs.get(key)
            if job is not None:
                try:
                    lines.append((n * shot_seconds + 0.4, job.result()))
                except Exception:
                    log.exception("narration failed for shot %s", key)
        music = (audio.music_bed(seconds, self.audio.get("music_dir", "assets/music"),
                                 seed=zlib.crc32(session.id.encode())) if session.music else None)
        return audio.mix(seconds, lines, music, self.audio.get("music_volume", 0.25))

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
        seed = zlib.crc32(f"{session.id}:{shot.act}:{shot.index}".encode())
        kf_path = session.dir / f"kf_{shot.act}_{shot.index}.png"
        start = time.time()
        if kind == KEYFRAME:
            shot.status = "keyframe"
            if session.photo is not None and shot.photo_prompt:
                image = self.engine.keyframe_from_photo(session.settings["photo"], session.photo,
                                                        shot.photo_prompt, seed)
            else:
                image = self.engine.keyframe(session.settings["keyframe"], shot.prompt, seed)
            image.save(kf_path)
            shot.status = "rendering"
        else:
            self.engine.shot(session.settings[shot.render_type], Image.open(kf_path), shot.prompt,
                             shot.narration if session.captions else None, seed,
                             session.dir / f"shot_{shot.act}_{shot.index}.mp4")
            shot.status, shot.seconds = "done", round(time.time() - start, 1)
            log.info("session %s: shot %d.%d %s in %.1fs", session.id, shot.act + 1, shot.index + 1,
                     shot.render_type, shot.seconds)

    def _finish(self, session: Session) -> None:
        session.film_status = "stitching"
        done = [(a, i) for (a, i), s in sorted(session.shots.items()) if s.status == "done"]
        if not done:
            raise RuntimeError("no shot rendered")
        film = session.dir / "film.mp4"
        silent = session.dir / "film_silent.mp4" if session.voice or session.music else film
        self.engine.stitch([session.dir / f"shot_{a}_{i}.mp4" for a, i in done], silent)
        if silent != film:
            wav = session.dir / "audio.wav"
            audio.write_wav(self._soundtrack(session, done), wav)
            self.engine.add_audio(silent, wav, film)
        session.film_status, session.film_ready_at = "done", time.time()
        log.info("session %s: film ready %.0fs after the last choice", session.id,
                 session.film_ready_at - session.last_choice_at)


def build(app_state: App) -> FastAPI:
    api = FastAPI(title="BranchingVideoGen demo")

    @api.get("/api/meta")
    def meta():
        busy = app_state.active()
        config = app_state.config
        defaults = {stage: (config.get(stage) if stage in TOP_LEVEL_STAGES else config["render_types"][stage])
                    .get("model") for stage in MODEL_STAGES}
        narration = app_state.audio.get("narration", "kokoro")
        narration_ok = app_state.engine.mock or (narration != "none" and importlib.util.find_spec("kokoro") is not None)
        return {"styles": {k: v["label"] for k, v in STYLES.items()}, "beats": beats_public(),
                "models": models_by_kind(config), "default_models": defaults,
                "voices": audio.VOICES if narration_ok else {},
                "default_voice": app_state.audio.get("voice", "af_heart") if narration_ok else None,
                "music_files": [p.name for p in audio.music_files(app_state.audio.get("music_dir", "assets/music"))],
                "profiles": {k: v["target_wait_minutes"] for k, v in app_state.config["profiles"].items()},
                "shots": {k: app_state.config["film"].get(k, d) for k, d in
                          (("shots", 36), ("min_shots", 6), ("max_shots", 72), ("shot_seconds", 5))},
                "busy": bool(busy), "eta_seconds": app_state.eta(busy) if busy else 0,
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
    parser.add_argument("--writer", choices=("llm", "simple"), default=None,
                        help="who writes the story (default: story.writer in the config; simple with --mock)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = apply_overrides(load_config(args.config) if args.config else load_config(), args.model)
    engine = Engine(config, mock=args.mock)
    # Mock runs use the simple writer unless --writer llm asks for the real one.
    config["story"]["writer"] = args.writer or ("simple" if args.mock else config["story"].get("writer", "llm"))
    if args.preload:
        engine.preload([stage_settings(config, "keyframe"), stage_settings(config, "standard")])
    import uvicorn

    uvicorn.run(build(App(config, engine, args.bench)), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
