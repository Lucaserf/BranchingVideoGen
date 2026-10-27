"""Writers turn the viewer's answers into the story: scene text, choices, shot prompts.

`LLMWriter` talks to any OpenAI-compatible chat endpoint (Ollama, vLLM, llama.cpp
server, or a hosted API) and asks for JSON. `SimpleWriter` needs no model: it writes
plain placeholder text from the beats and answers, and is used for mock runs and as
the fallback when the LLM fails, so a film always finishes.

Each act takes two calls. After the viewer answers act N:
  1. `next_act`: the next act's scene, question and options (what the viewer waits
     for), plus an updated summary and cast.
  2. `shots`: the shot prompts and narration for act N (what the GPU waits for).
"""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.request

from bvg.story import AXES, BEATS, STYLES, ActText, Option, ShotSpec, StoryState

log = logging.getLogger("bvg.writer")


class WriterError(RuntimeError):
    pass


# --- prompts --------------------------------------------------------------

SYSTEM = """You are the storyteller of an interactive short film that a viewer plays in a few minutes.
The film follows six fixed beats, always in this order:
{beats}

Rules:
- The viewer shapes the story with one answer per beat. Honor every answer, including surprising or
  free-text ones, and let earlier answers have consequences later.
- Write in {language}, in the second person for text the viewer reads ("you").
- Keep each beat's purpose, but make everything else specific and vivid: names, places, objects.
- Choices: exactly three, clearly different from each other, at most 12 words each.
- Two hard limits, whatever the viewer writes: no sexual content involving anyone under 18, and no sexual
  or nude depiction of the protagonist when they come from a viewer's photo (a real person).
- Reply with one JSON object only, no other text."""

VALUES_HELP = ("values: how much the choice expresses each of " + ", ".join(AXES) +
               ", as integers from -2 to 2 (omit axes that don't apply)")


def _context(state: StoryState) -> str:
    lines = [f"Visual style of the film: {STYLES[state.style]['label']}."]
    if state.has_photo:
        lines.append("The protagonist is the viewer, from a photo they uploaded: describe their clothes and look, "
                     "never their face, age or body.")
    if state.title:
        lines.append(f"Title: {state.title}")
    if state.cast:
        lines.append("Cast and places (visual descriptions): " + json.dumps(state.cast, ensure_ascii=False))
    if state.summary:
        lines.append(f"Story so far: {state.summary}")
    for i, act in enumerate(state.acts):
        lines.append(f"Beat {i + 1} ({BEATS[i].title}): scene: {act.text.intro} Asked: {act.text.question} "
                     f"Viewer answered: {act.answer or '(not yet)'}")
    return "\n".join(lines)


def _act_schema() -> str:
    return ('"act": {"intro": "2-3 sentences the viewer reads, setting the scene of this beat", '
            '"question": "the question the viewer answers", '
            '"options": [{"label": "...", "values": {"courage": 1}}, {...}, {...}]}')


def _opening_prompt(state: StoryState) -> str:
    return f"""{_context(state)}

Start the story with beat 1 ({BEATS[0].title}). Invent an original premise.
Return JSON: {{"title": "a short film title", "cast": {{"world": "visual description of the setting"}},
{_act_schema()}}}
{VALUES_HELP}."""


def _next_prompt(state: StoryState, answered: int) -> str:
    last = answered == len(BEATS) - 1
    free = state.acts[answered].free
    parts = [f"""{_context(state)}

The viewer just answered beat {answered + 1}."""]
    keys = ['"summary": "the whole story so far in at most 80 words"',
            '"cast": {"name": "visual description, repeated verbatim in image prompts"} with the protagonist, '
            'the world and every recurring character so far (keep earlier descriptions unchanged)']
    if free:
        keys.append('"answer_values": {"courage": 0, ...} scoring the viewer\'s own answer; ' + VALUES_HELP)
    if last:
        parts.append("That was the last beat. Close the story.")
        keys.append('"title": "the final film title"')
    else:
        nxt = answered + 1
        parts.append(f"Write beat {nxt + 1} ({BEATS[nxt].title}): {BEATS[nxt].purpose}")
        keys.append(_act_schema())
        keys.append(VALUES_HELP)
    parts.append("Return JSON with: " + "; ".join(keys) + ".")
    return "\n".join(parts)


def _shots_prompt(state: StoryState, act_index: int) -> str:
    n = state.shots_per_act
    return f"""{_context(state)}

Write the {n} shots that film beat {act_index + 1} ({BEATS[act_index].title}) as the viewer answered it.
Each shot is 5 seconds. Together they tell this beat clearly; the first sets the scene, the last leads on.
Return JSON: {{"shots": [{n} items of {{"prompt": "...", "narration": "...", "shows_hero": true, "priority": 0.5}}]}}
- prompt: one frame for an image model, at most 60 words: who, doing what, where, camera framing and light.
  Each prompt stands alone: repeat the full visual description of every character and place in it.
  No style words (the style is added later), no text or letters in the image.
- narration: one sentence the narrator says over the shot, at most 14 words.
- shows_hero: whether the protagonist is visible.
- priority: 0 to 1, how much the shot matters (key moments near 1)."""


# --- parsing --------------------------------------------------------------

def parse_json(text: str) -> dict:
    """The first JSON object in a model reply (tolerates code fences and thinking blocks)."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise WriterError("no JSON object in the reply")
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError as e:
        raise WriterError(f"invalid JSON: {e}") from e


def _str(d: dict, key: str, limit: int) -> str:
    value = d.get(key)
    if not isinstance(value, str) or not value.strip():
        raise WriterError(f"missing {key!r}")
    return value.strip()[:limit]


def _values(d) -> dict[str, int]:
    if not isinstance(d, dict):
        return {}
    out = {}
    for axis in AXES:
        try:
            out[axis] = max(-2, min(2, int(d.get(axis, 0))))
        except (TypeError, ValueError):
            pass
    return {k: v for k, v in out.items() if v}


def parse_act(d: dict) -> ActText:
    if not isinstance(d, dict):
        raise WriterError("missing 'act'")
    options = d.get("options")
    if not isinstance(options, list) or len(options) < 2:
        raise WriterError("an act needs options")
    parsed = []
    for o in options[:3]:
        if isinstance(o, str):
            o = {"label": o}
        parsed.append(Option(label=_str(o, "label", 120), values=_values(o.get("values"))))
    return ActText(intro=_str(d, "intro", 600), question=_str(d, "question", 200), options=parsed)


def parse_shots(d: dict, n: int) -> list[ShotSpec]:
    shots = d.get("shots")
    if not isinstance(shots, list) or not shots:
        raise WriterError("missing 'shots'")
    out = []
    for s in shots[:n]:
        try:
            priority = max(0.0, min(1.0, float(s.get("priority", 0.5))))
        except (TypeError, ValueError):
            priority = 0.5
        out.append(ShotSpec(prompt=_str(s, "prompt", 700), narration=_str(s, "narration", 160),
                            shows_hero=bool(s.get("shows_hero", False)), priority=priority))
    while len(out) < n:  # a model that wrote too few shots: hold the last one longer
        out.append(out[-1])
    return out


def _cast(d) -> dict[str, str]:
    if not isinstance(d, dict):
        return {}
    return {str(k)[:40]: str(v)[:300] for k, v in d.items() if isinstance(v, str) and v.strip()}


# --- writers --------------------------------------------------------------

class LLMWriter:
    def __init__(self, settings: dict):
        self.base_url = settings.get("base_url", "http://127.0.0.1:11434/v1").rstrip("/")
        # A catalog entry gives the server's name for the model in model_id; otherwise `model` is that name.
        self.model = settings.get("model_id") or settings.get("model", "qwen3:8b")
        self.api_key = os.environ.get(settings.get("api_key_env", "BVG_LLM_API_KEY"), "")
        self.temperature = settings.get("temperature", 0.9)
        self.timeout = settings.get("timeout_seconds", 120)
        self.max_tokens = settings.get("max_tokens", 2500)
        self.extra_body = settings.get("extra_body") or {}
        self.prompt_suffix = settings.get("prompt_suffix", "")  # e.g. " /no_think" to turn off Qwen3 thinking
        self.language = settings.get("language", "English")
        self._json_mode = True

    def _system(self) -> str:
        beats = "\n".join(f"{i + 1}. {b.title}: {b.purpose}" for i, b in enumerate(BEATS))
        return SYSTEM.format(beats=beats, language=self.language)

    def _post(self, body: dict) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(f"{self.base_url}/chat/completions", data=json.dumps(body).encode(),
                                     headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as res:
            return json.load(res)

    def ask(self, prompt: str) -> dict:
        body = {"model": self.model, "temperature": self.temperature, "max_tokens": self.max_tokens,
                "messages": [{"role": "system", "content": self._system()}, {"role": "user", "content": prompt + self.prompt_suffix}],
                **self.extra_body}
        last_error: Exception | None = None
        for attempt in range(2):
            if self._json_mode:
                body["response_format"] = {"type": "json_object"}
            else:
                body.pop("response_format", None)
            try:
                reply = self._post(body)
                return parse_json(reply["choices"][0]["message"]["content"] or "")
            except urllib.error.HTTPError as e:
                last_error = e
                if e.code == 400 and self._json_mode:  # server without JSON mode: ask in plain text
                    self._json_mode = False
                    continue
            except (WriterError, KeyError, IndexError, TypeError, OSError) as e:
                last_error = e
            log.warning("writer attempt %d failed: %s", attempt + 1, last_error)
        raise WriterError(f"the LLM failed: {last_error}")

    def opening(self, state: StoryState) -> None:
        d = self.ask(_opening_prompt(state))
        act = parse_act(d.get("act"))
        state.title = str(d.get("title") or "")[:120]
        state.cast.update(_cast(d.get("cast")))
        state.acts.append(_record(act))

    def next_act(self, state: StoryState, answered: int) -> None:
        d = self.ask(_next_prompt(state, answered))
        act = parse_act(d.get("act")) if answered < len(BEATS) - 1 else None
        if state.acts[answered].free:
            state.add_values(_values(d.get("answer_values")))
        state.summary = str(d.get("summary") or state.summary)[:1200]
        state.cast.update(_cast(d.get("cast")))
        if d.get("title"):
            state.title = str(d["title"])[:120]
        if act is not None:
            state.acts.append(_record(act))

    def shots(self, state: StoryState, act_index: int) -> None:
        d = self.ask(_shots_prompt(state, act_index))
        state.acts[act_index].shots = parse_shots(d, state.shots_per_act)


def _record(act: ActText):
    from bvg.story import ActRecord

    return ActRecord(text=act)


class SimpleWriter:
    """Plain text from the beats and answers, with no model. For mock runs and as a fallback."""

    OPTIONS = (
        ("A courier in a city of copper towers", "A lighthouse keeper on a stormy island", "A thief in a desert town"),
        ("A glowing map falls from the sky", "A wounded stranger brings a warning", "The sun does not rise"),
        ("A talking fox", "A disgraced knight", "A small lost robot"),
        ("Save the village", "Chase the villain", "Make a deal with the villain"),
        ("Lose the ally", "Lose the memories of home", "Lose the power"),
        ("Return home", "Stay in the new world", "Become the guardian"),
    )

    def opening(self, state: StoryState) -> None:
        state.title = "An Untold Journey"
        state.cast.setdefault("world", "a wide land of hills, rivers and old towers")
        state.acts.append(_record(self._act(0)))
        return "simple"

    def next_act(self, state: StoryState, answered: int) -> None:
        answers = state.choices
        state.summary = " Then ".join(answers)
        if answered == 0:
            state.cast.setdefault("hero", f"the protagonist: {answers[0]}")
        if answered < len(BEATS) - 1:
            state.acts.append(_record(self._act(answered + 1)))
        return "simple"

    def shots(self, state: StoryState, act_index: int) -> None:
        answer = state.acts[act_index].answer or ""
        hero = state.cast.get("hero", "the protagonist")
        world = state.cast.get("world", "")
        beat = BEATS[act_index].title.lower()
        frames = (f"wide shot of {world}, {beat}", f"{hero} in {world}, {answer}", f"close-up of {hero}, {answer}",
                  f"{hero} reacting, {beat}, {world}", f"{answer}, dramatic moment in {world}",
                  f"{hero} looking ahead, {world}, {beat}")
        state.acts[act_index].shots = [
            ShotSpec(prompt=frames[i % len(frames)], narration=(answer if i == 1 else BEATS[act_index].title) + ".",
                     shows_hero="hero" in frames[i % len(frames)] or hero in frames[i % len(frames)],
                     priority=0.8 if i in (0, state.shots_per_act - 1) else 0.5)
            for i in range(state.shots_per_act)]
        return "simple"

    def _act(self, i: int) -> ActText:
        beat = BEATS[i]
        return ActText(intro=f"{beat.title}. {beat.purpose}", question="What happens next?",
                       options=[Option(label) for label in self.OPTIONS[i]])


class FallbackWriter:
    """Use the LLM, and the simple writer for any step the LLM can't do."""

    def __init__(self, primary, fallback):
        self.primary, self.fallback = primary, fallback
        self.failures = 0

    def _do(self, step: str, *args) -> str:
        try:
            getattr(self.primary, step)(*args)
            return "llm"
        except Exception as e:
            self.failures += 1
            log.warning("writer step %s failed, using the simple writer: %s", step, e)
            getattr(self.fallback, step)(*args)
            return "fallback"

    def opening(self, state):
        return self._do("opening", state)

    def next_act(self, state, answered):
        return self._do("next_act", state, answered)

    def shots(self, state, act_index):
        return self._do("shots", state, act_index)


def make_writer(settings: dict, mock: bool = False):
    if mock or settings.get("writer", "llm") == "simple":
        return SimpleWriter()
    llm = LLMWriter(settings)
    return FallbackWriter(llm, SimpleWriter()) if settings.get("fallback_to_simple", True) else llm
