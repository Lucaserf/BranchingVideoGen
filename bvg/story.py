"""The story: six fixed beats, everything else written per viewer.

The beats (ordinary world, the call, the ally, the test, the cost, the new world)
are the comparable skeleton: every film has them in the same order and length. What
happens in each beat, the text the viewer reads, the choices offered and the shot
prompts are written by a writer (bvg.writer) from the viewer's answers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

STYLES = {
    "animated": {
        "label": "3D animated",
        "prompt": "3D animated feature film still, soft global illumination, expressive characters, vibrant colors",
    },
    "anime": {
        "label": "Anime",
        "prompt": "anime film, hand-drawn cel shading, painted backgrounds, dramatic lighting",
    },
    "watercolor": {
        "label": "Watercolor",
        "prompt": "watercolor illustration, soft washes, visible paper texture, gentle pastel palette",
    },
    "noir": {
        "label": "Film noir",
        "prompt": "black and white film noir, high contrast, hard shadows, 35mm film grain",
    },
    "comic": {
        "label": "Comic book",
        "prompt": "comic book art, bold ink lines, halftone shading, saturated flat colors",
    },
    "claymation": {
        "label": "Claymation",
        "prompt": "stop-motion claymation, handmade clay figures, miniature sets, tactile textures",
    },
}

AXES = ("courage", "empathy", "curiosity", "order")


@dataclass(frozen=True)
class Beat:
    title: str
    purpose: str  # what the act must accomplish, given to the writer


BEATS: tuple[Beat, ...] = (
    Beat("The ordinary world", "Introduce the protagonist and their everyday world, and hint at what they lack. "
                               "The choice decides who the protagonist is and where they live."),
    Beat("The call", "Something disrupts the ordinary world and calls the protagonist to adventure. "
                     "The choice decides what that disruption is or how they answer it."),
    Beat("The ally", "On the road, the protagonist meets someone who will help or challenge them. "
                     "The choice decides who joins them and why."),
    Beat("The test", "The hardest dilemma: the antagonist forces a choice between two things that matter. "
                     "The choice is a real moral decision with a cost either way."),
    Beat("The cost", "The consequence of the test: something is lost or changed, and the protagonist hits bottom "
                     "before finding the will to go on. The choice decides what is lost."),
    Beat("The new world", "The final confrontation and the resolution: how the story ends and who the protagonist "
                          "has become. The choice decides the ending."),
)


@dataclass
class Option:
    label: str
    values: dict[str, int] = field(default_factory=dict)


@dataclass
class ActText:
    """What the viewer reads and chooses from in one act."""
    intro: str
    question: str
    options: list[Option]


@dataclass
class ShotSpec:
    prompt: str  # what the frame shows, without style words
    narration: str
    shows_hero: bool = False
    priority: float = 0.5


@dataclass
class ActRecord:
    text: ActText
    answer: str | None = None
    free: bool = False  # answered in the viewer's own words; the writer scores the values
    shots: list[ShotSpec] = field(default_factory=list)


@dataclass
class StoryState:
    style: str
    shots_per_act: int = 6
    has_photo: bool = False
    title: str = ""
    # Visual descriptions the writer repeats in every prompt, so characters and places stay recognizable.
    cast: dict[str, str] = field(default_factory=dict)
    summary: str = ""
    acts: list[ActRecord] = field(default_factory=list)
    values: dict[str, int] = field(default_factory=lambda: {a: 0 for a in AXES})

    @property
    def act(self) -> int:
        """Index of the next act to choose (the number of answered acts)."""
        return sum(a.answer is not None for a in self.acts)

    @property
    def current(self) -> ActText | None:
        """The act waiting for an answer, once the writer has written it."""
        if self.act < len(self.acts):
            return self.acts[self.act].text
        return None

    @property
    def choices(self) -> list[str]:
        return [a.answer for a in self.acts if a.answer is not None]

    def choose(self, option_index: int | None = None, free_text: str | None = None) -> int:
        """Record the answer to the current act. Returns that act's index."""
        text = self.current
        if text is None:
            raise ValueError("no act is waiting for an answer")
        record = self.acts[self.act]
        if free_text and free_text.strip():
            record.answer, record.free = free_text.strip()[:200], True
        else:
            option = text.options[option_index]
            record.answer = option.label
            self.add_values(option.values)
        return self.acts.index(record)

    def add_values(self, values: dict[str, int]) -> None:
        for axis, delta in values.items():
            if axis in self.values:
                self.values[axis] += max(-2, min(2, int(delta)))

    def prompts(self, act_index: int) -> list[dict]:
        """Full prompts for each shot of an act, with the style appended."""
        style = STYLES[self.style]["prompt"]
        out = []
        for i, shot in enumerate(self.acts[act_index].shots):
            photo_prompt = None
            if self.has_photo and shot.shows_hero:
                photo_prompt = (f"{shot.prompt}. The protagonist is the person in the reference photo: keep the "
                                f"same face and identity. {style}")
            out.append({"act": act_index, "index": i, "priority": shot.priority, "prompt": f"{shot.prompt}, {style}",
                        "photo_prompt": photo_prompt, "narration": shot.narration})
        return out


def beats_public() -> list[dict]:
    return [{"title": b.title} for b in BEATS]
