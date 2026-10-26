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
