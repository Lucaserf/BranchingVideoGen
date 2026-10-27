import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from bvg.story import BEATS, StoryState
from bvg.writer import FallbackWriter, LLMWriter, SimpleWriter, parse_json


def _play(writer, state, answers):
    writer.opening(state)
    for act, answer in enumerate(answers):
        assert state.current is not None
        if isinstance(answer, str):
            state.choose(free_text=answer)
        else:
            state.choose(answer)
        writer.next_act(state, act)
        writer.shots(state, act)
    assert state.current is None


def test_simple_writer_builds_prompts_from_answers():
    s = StoryState("noir", shot_counts=[3] * 6)
    _play(SimpleWriter(), s, [0, "a comet shaped like a key", 1, 0, 2, 1])
    assert len(s.acts) == len(BEATS)
    prompts = s.prompts(1)
    assert len(prompts) == 3
    assert any("a comet shaped like a key" in p["prompt"] for p in prompts)
    assert all(p["prompt"].endswith("35mm film grain") for p in prompts)
    assert s.acts[1].free and s.choices[1] == "a comet shaped like a key"


def test_parse_json_tolerates_thinking_and_fences():
    reply = '<think>{"no": 1}</think>Sure!\n```json\n{"a": {"b": 2}}\n```'
    assert parse_json(reply) == {"a": {"b": 2}}


class _FakeLLM(BaseHTTPRequestHandler):
    """An OpenAI-compatible endpoint that answers from the prompt with canned JSON."""
    calls: list = []
    broken = False

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        prompt = body["messages"][-1]["content"]
        _FakeLLM.calls.append(prompt)
        act = {"intro": "Rain hammers the glass dome of Vell.", "question": "Who are you?",
               "options": [{"label": "A glassblower", "values": {"empathy": 1}},
                           {"label": "A smuggler", "values": {"courage": 2, "order": -1}},
                           {"label": "A cartographer", "values": {"curiosity": 9}}]}
        if _FakeLLM.broken:
            content = "I cannot write JSON today."
        elif "shots that film beat" in prompt:
            n = int(prompt.split("Write the ")[1].split()[0])
            content = json.dumps({"shots": [{"prompt": f"Mara the glassblower, shot {i}", "narration": "Rain.",
                                             "shows_hero": i % 2 == 0, "priority": 0.9} for i in range(n - 1)]})
        elif "Start the story" in prompt:
            content = json.dumps({"title": "The Glass Dome", "cast": {"world": "a city under a glass dome"},
                                  "act": act})
        else:
            content = "<think>planning</think>" + json.dumps(
                {"summary": "Mara leaves Vell.", "cast": {"Mara": "a tall glassblower in a red coat"},
                 "answer_values": {"courage": 2}, "title": "Glass and Thunder", "act": act})
        out = json.dumps({"choices": [{"message": {"content": content}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


@pytest.fixture
def fake_llm():
    _FakeLLM.calls, _FakeLLM.broken = [], False
    server = HTTPServer(("127.0.0.1", 0), _FakeLLM)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield {"base_url": f"http://127.0.0.1:{server.server_port}/v1", "model": "fake", "timeout_seconds": 5}
    server.shutdown()


def test_llm_writer_writes_text_choices_and_prompts(fake_llm):
    s = StoryState("anime", shot_counts=[3] * 6, has_photo=True)
    _play(LLMWriter(fake_llm), s, [1, "I sing to the storm", 0, 0, 0, 0])
    assert s.title == "Glass and Thunder"
    assert s.acts[0].text.intro.startswith("Rain") and s.acts[0].text.options[1].label == "A smuggler"
    assert s.cast["Mara"].startswith("a tall glassblower")
    # option values are clamped; free-text answers are scored by the writer
    assert s.values["courage"] == 2 + 2 + 0 and s.values["curiosity"] == 0
    prompts = s.prompts(0)
    assert len(prompts) == 3 and prompts[2] == {**prompts[1], "index": 2}  # too few shots: last one repeated
    assert prompts[0]["photo_prompt"] and prompts[1]["photo_prompt"] is None
    assert "I sing to the storm" in _FakeLLM.calls[3]  # the answer reaches the next prompt
    assert len(_FakeLLM.calls) == 1 + 2 * len(BEATS)


def test_fallback_writer_finishes_the_story_when_the_llm_fails(fake_llm):
    _FakeLLM.broken = True
    writer = FallbackWriter(LLMWriter(fake_llm), SimpleWriter())
    s = StoryState("comic", shot_counts=[2] * 6)
    assert writer.opening(s) == "fallback"
    for act in range(len(BEATS)):
        s.choose(0)
        assert writer.next_act(s, act) == "fallback"
        writer.shots(s, act)
    assert all(len(a.shots) == 2 for a in s.acts) and writer.failures == 1 + 2 * len(BEATS)


def _client(tmp_path):
    from fastapi.testclient import TestClient

    from bvg.config import load_config
    from bvg.engine import Engine
    from bvg.server import App, build

    config = load_config()
    config["film"]["shot_seconds"] = 1
    config["server"]["outputs"] = str(tmp_path)
    config["audio"]["music_dir"] = str(tmp_path / "no-music")
    config["story"]["writer"] = "simple"
    return TestClient(build(App(config, Engine(config, mock=True), None)))


def _answer(client, sid, body):
    for _ in range(100):
        state = client.get(f"/api/sessions/{sid}").json()
        if state["current"]:
            break
        time.sleep(0.05)
    assert state["current"]["options"], state
    return client.post(f"/api/sessions/{sid}/choices", json=body)


def test_server_plays_a_short_film_in_mock_mode(tmp_path):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    client = _client(tmp_path)
    sid = client.post("/api/sessions", json={"style": "comic", "shots": 20}).json()["id"]
    assert client.post("/api/sessions", json={}).status_code == 409  # one viewer at a time
    for act in range(6):
        body = {"text": "a tiny dragon"} if act == 2 else {"option": 0}
        assert _answer(client, sid, body).status_code == 200
    assert client.post(f"/api/sessions/{sid}/choices", json={"option": 0}).status_code == 400  # story over
    state = _wait(client, sid)
    assert state["film_status"] == "done", state["error"]
    assert state["shots_done"] == state["shots_total"] == 20
    assert state["shot_counts"] == [3, 3, 3, 4, 3, 4] and state["film_seconds"] == 20
    assert state["writer"] == "simple" and state["title"]
    assert any("a tiny dragon" in a.get("answer", "") for a in state["story"])
    assert client.get(state["film"]).status_code == 200
    assert client.get(f"/files/{sid}/..%2Fsecret").status_code == 404


def _wait(client, sid):
    for _ in range(300):
        state = client.get(f"/api/sessions/{sid}").json()
        if state["film_status"] in ("done", "failed"):
            return state
        time.sleep(0.1)
    raise AssertionError("film not ready")


def test_film_with_photo_voice_music_and_model_picks(tmp_path):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    import base64
    import io
    import subprocess

    from PIL import Image

    from bvg.video import ffmpeg_exe

    client = _client(tmp_path)
    buf = io.BytesIO()
    Image.new("RGB", (300, 400), "orange").save(buf, "JPEG")
    photo = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

    body = {"shots": 6, "photo": photo, "voice": "af_heart", "music": True,
            "models": {"standard": "wan2.2-ti2v-5b", "photo": "qwen-image-edit-2509-abliterated",
                       "story": "qwen3.6-35b-a3b-abliterated"}}
    assert client.post("/api/sessions", json=body).status_code == 400  # photo without consent
    assert client.post("/api/sessions", json={**body, "photo_consent": True,
                                              "models": {"standard": "flux1-schnell"}}).status_code == 400  # wrong kind
    state = client.post("/api/sessions", json={**body, "photo_consent": True}).json()
    assert state["models"]["standard"] == "wan2.2-ti2v-5b" and state["photo"]
    assert state["models"]["story"] == "qwen3.6-35b-a3b-abliterated"
    for act in range(6):
        assert _answer(client, state["id"], {"option": 1}).status_code == 200
    state = _wait(client, state["id"])
    assert state["film_status"] == "done", state["error"]
    film = tmp_path / state["id"] / "film.mp4"
    streams = subprocess.run([ffmpeg_exe(), "-i", str(film)], capture_output=True, text=True).stderr
    assert "Video:" in streams and "Audio:" in streams


def test_mix_ducks_music_under_narration():
    import numpy as np

    from bvg import audio

    voice = np.full(audio.SAMPLE_RATE, 0.5, np.float32)
    music = np.full(3 * audio.SAMPLE_RATE, 0.5, np.float32)
    out = audio.mix(3, [(1.0, voice)], music, music_volume=0.5)
    quiet, speaking = out[int(0.2 * audio.SAMPLE_RATE)], out[int(1.5 * audio.SAMPLE_RATE)]
    assert quiet == pytest.approx(0.25, abs=0.01)  # music alone
    assert speaking > quiet  # narration on top, music ducked
    assert len(audio.synth_pad(5)) == 5 * audio.SAMPLE_RATE


def test_story_model_comes_from_the_catalog(fake_llm):
    from bvg.config import load_config
    from bvg.models import select_model, stage_settings
    from bvg.writer import make_writer

    config = load_config()
    select_model(config, "story", "qwen3.8-27b-abliterated")
    settings = stage_settings(config, "story")
    writer = make_writer({**settings, **fake_llm})
    assert writer.primary.model == "huihui_ai/Qwen3.8-abliterated:27b"
    with pytest.raises(ValueError):
        select_model(config, "story", "flux1-schnell")
        stage_settings(config, "story")
    qwen = make_writer({**stage_settings(load_config(), "story"), **fake_llm}).primary
    qwen.opening(StoryState("anime"))
    assert _FakeLLM.calls[-1].endswith(" /no_think")


def test_split_shots_is_even_and_favors_the_climax():
    from bvg.story import split_shots

    assert split_shots(36) == [6] * 6
    assert split_shots(8) == [1, 1, 1, 2, 1, 2]
    assert sum(split_shots(71)) == 71
    assert split_shots(2) == [1] * 6  # at least one shot per beat


def test_shot_count_is_bounded(tmp_path):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    client = _client(tmp_path)
    assert client.post("/api/sessions", json={"shots": 5}).status_code == 400
    assert client.post("/api/sessions", json={"shots": 500}).status_code == 400


def _snap(gpu_total=48.0, ram_free=60.0):
    return {"gpu": {"total_gb": gpu_total, "free_gb": gpu_total, "process_gb": 0},
            "ram": {"total_gb": 62.0, "free_gb": ram_free, "process_gb": 1}, "llm": []}


def test_memory_forecast_sequential_vs_swap():
    from bvg import memory
    from bvg.config import load_config
    from bvg.models import select_model, stage_settings

    config = load_config()
    stages = ["keyframe", "standard", "story"]
    settings = {s: stage_settings(config, s) for s in stages}
    seq = memory.forecast(settings, "sequential", _snap(), used_stages=["keyframe", "standard"])
    assert seq["fits"] and seq["peak_vram_gb"] == 38.5 + 5  # LTX plus the 8B writer on the GPU
    assert seq["peak_ram_gb"] == 36  # the largest model alone
    swap = memory.forecast(settings, "swap", _snap(), used_stages=["keyframe", "standard"])
    assert not swap["fits"] and swap["peak_ram_gb"] == 34 + 36  # FLUX and LTX together
    select_model(config, "story", "qwen3.8-27b-abliterated")
    big = memory.forecast({**settings, "story": stage_settings(config, "story")}, "sequential", _snap(),
                          used_stages=["keyframe", "standard"])
    assert [w.split()[0] for w in big["warnings"]] == ["flux1-schnell", "ltx-0.9.7-distilled"]  # + 18 GB > 44
    cpu = memory.forecast({**settings, "story": stage_settings(config, "story")}, "sequential", _snap(),
                          used_stages=["keyframe", "standard"], llm_device="cpu")
    assert cpu["fits"] and cpu["peak_ram_gb"] == 36 + 18


def test_scheduler_batches_jobs_by_loaded_model(tmp_path):
    pytest.importorskip("fastapi")
    from bvg.server import FINISH, KEYFRAME, VIDEO

    from bvg.config import load_config
    from bvg.engine import Engine
    from bvg.server import App, SessionIn

    config = load_config()
    config["server"]["outputs"] = str(tmp_path)
    config["story"]["writer"] = "simple"
    app = App(config, Engine(config, mock=True), None)
    session = app.create(SessionIn(shots=6, profile="quality"))
    shots = sorted(session.shots.values(), key=lambda s: s.act)
    a, b = shots[0], shots[1]
    a.status, b.status = "rendering", "queued"  # a's keyframe is done, b's isn't
    items = [(KEYFRAME, 2, session.id, b.act, b.index), (VIDEO, 1, session.id, a.act, a.index),
             (VIDEO, 3, session.id, b.act, b.index), (FINISH, 4, session.id, -1, -1)]
    std = app.engine.model_key(session.settings[a.render_type])
    app.engine._loaded[std] = object()  # pretend the video model is loaded
    assert app._pick(items) == items[1]  # video with the loaded model beats a keyframe that needs a switch
    app.engine._loaded.clear()
    assert app._pick(items) == items[0]  # nothing loaded: keyframes first
    assert app._pick([items[2], items[3]]) is None  # b's video waits for its keyframe, the finish for b
