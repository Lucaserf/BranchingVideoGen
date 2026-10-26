import time

import pytest

from bvg.story import ACTS, StoryState


def test_story_fills_shots_from_choices_and_free_text():
    s = StoryState("noir")
    s.choose(0)
    s.choose(free_text="a comet shaped like a key")
    shots = s.shots(1)
    assert len(shots) == 6
    assert "a comet shaped like a key" in shots[0]["prompt"]
    assert shots[0]["prompt"].endswith("35mm film grain")
    assert "{" not in "".join(sh["prompt"] + sh["narration"] for sh in shots)
    assert shots[1]["narration"].startswith("The courier")


def test_every_option_leaves_no_unfilled_fields():
    for picks in [(0,) * 6, (1,) * 6, (2,) * 6]:
        s = StoryState("anime")
        for a, p in enumerate(picks):
            s.choose(p)
            text = "".join(sh["prompt"] + sh["narration"] for sh in s.shots(a))
            assert "{" not in text, (a, p)
    assert s.act == len(ACTS)


def test_server_plays_a_short_film_in_mock_mode(tmp_path):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from bvg.config import load_config
    from bvg.engine import Engine
    from bvg.server import App, build

    config = load_config()
    config["film"]["shot_seconds"] = 1
    config["server"]["outputs"] = str(tmp_path)
    client = TestClient(build(App(config, Engine(config, mock=True), None)))

    sid = client.post("/api/sessions", json={"style": "comic", "length": "short"}).json()["id"]
    assert client.post("/api/sessions", json={}).status_code == 409  # one viewer at a time
    for act in range(6):
        body = {"text": "a tiny dragon"} if act == 2 else {"option": 0}
        assert client.post(f"/api/sessions/{sid}/choices", json=body).status_code == 200
    for _ in range(200):
        state = client.get(f"/api/sessions/{sid}").json()
        if state["film_status"] in ("done", "failed"):
            break
        time.sleep(0.1)
    assert state["film_status"] == "done", state["error"]
    assert state["shots_done"] == state["shots_total"] == 18
    assert client.get(state["film"]).status_code == 200
    assert client.get(f"/files/{sid}/..%2Fsecret").status_code == 404


def _client(tmp_path):
    from fastapi.testclient import TestClient

    from bvg.config import load_config
    from bvg.engine import Engine
    from bvg.server import App, build

    config = load_config()
    config["film"]["shot_seconds"] = 1
    config["server"]["outputs"] = str(tmp_path)
    config["audio"]["music_dir"] = str(tmp_path / "no-music")
    return TestClient(build(App(config, Engine(config, mock=True), None)))


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

    body = {"length": "short", "photo": photo, "voice": "af_heart", "music": True,
            "models": {"standard": "wan2.2-ti2v-5b", "photo": "qwen-image-edit-2509"}}
    assert client.post("/api/sessions", json=body).status_code == 400  # photo without consent
    assert client.post("/api/sessions", json={**body, "photo_consent": True,
                                              "models": {"standard": "flux1-schnell"}}).status_code == 400  # wrong kind
    state = client.post("/api/sessions", json={**body, "photo_consent": True}).json()
    assert state["models"]["standard"] == "wan2.2-ti2v-5b" and state["photo"]
    for act in range(6):
        client.post(f"/api/sessions/{state['id']}/choices", json={"option": 1})
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
