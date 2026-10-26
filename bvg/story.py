"""The hero's journey: six acts, one choice each, and the shot list each choice produces.

Every act has a fixed purpose, question and six shot templates; the viewer's choice
fills the templates. An option is a set of story fields ({hero}, {world}, ...); a
free-text answer fills the act's fields with the viewer's own words. Each option
also adds to the value fingerprint used to compare runs.
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
class Option:
    label: str
    fields: dict[str, str]
    values: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class ShotTemplate:
    prompt: str
    narration: str
    priority: float = 0.5


@dataclass(frozen=True)
class Act:
    title: str
    question: str
    options: tuple[Option, ...]
    free_text_fields: tuple[str, ...]  # fields a "write your own" answer fills
    shots: tuple[ShotTemplate, ...]


def _shots(*items) -> tuple[ShotTemplate, ...]:
    return tuple(ShotTemplate(*item) for item in items)


ACTS: tuple[Act, ...] = (
    Act(
        title="The ordinary world",
        question="Who is your hero, and where do they live?",
        options=(
            Option("A courier in a city of copper towers",
                   {"hero": "a young courier with a satchel and a mechanical arm",
                    "world": "a city of copper towers and sky bridges", "name": "the courier"}, {"curiosity": 1}),
            Option("A lighthouse keeper on a stormy island",
                   {"hero": "an old lighthouse keeper with a white beard and a heavy coat",
                    "world": "a rocky island with a lighthouse in a stormy sea", "name": "the keeper"}, {"order": 1}),
            Option("A thief in a desert market town",
                   {"hero": "a quick-handed young thief in a hooded scarf",
                    "world": "a desert market town of sandstone domes and silk awnings", "name": "the thief"}, {"courage": 1}),
        ),
        free_text_fields=("hero", "world"),
        shots=_shots(
            ("wide establishing shot of {world} at dawn", "Every story starts somewhere ordinary.", 0.8),
            ("{hero} waking up at home in {world}, morning light", "This one starts with {hero}."),
            ("{hero} walking through the busy streets of {world}, tracking shot", "Every day the same paths."),
            ("close-up of {hero}'s hands at their daily work", "The same small tasks."),
            ("{hero} looking at the far horizon beyond {world}", "And the same quiet wish for something more."),
            ("{hero} silhouetted against the sunset over {world}", "Tomorrow, that wish would be answered.", 0.8),
        ),
    ),
    Act(
        title="The call",
        question="What breaks the routine?",
        options=(
            Option("A glowing map falls from the sky",
                   {"call": "a glowing map that falls from the sky",
                    "call_visual": "a glowing golden map drifting down from the sky"}, {"curiosity": 2}),
            Option("A wounded stranger brings a warning",
                   {"call": "a wounded stranger with a warning",
                    "call_visual": "a wounded hooded stranger collapsing at the door"}, {"empathy": 2}),
            Option("The sun does not rise",
                   {"call": "a morning when the sun does not rise",
                    "call_visual": "a dark sky at morning with no sun, people holding lanterns"}, {"courage": 1}),
        ),
        free_text_fields=("call", "call_visual"),
        shots=_shots(
            ("{call_visual}, in {world}", "Then came {call}.", 0.9),
            ("{hero} staring in disbelief, close-up of the face", "{hero} could not look away."),
            ("people of {world} gathering in fear and wonder", "Everyone saw it. Only one would act."),
            ("{hero} hesitating at the doorway of home", "Staying would be easier."),
            ("{hero} packing a bag by candlelight", "But some calls cannot be ignored."),
            ("{hero} stepping out of {world} onto an unknown road, wide shot", "So the journey began.", 0.8),
        ),
    ),
    Act(
        title="The ally",
        question="Who joins the hero on the road?",
        options=(
            Option("A talking fox who knows the old roads",
                   {"ally": "a clever red fox with bright eyes", "ally_name": "the fox"}, {"curiosity": 1}),
            Option("A disgraced knight seeking redemption",
                   {"ally": "a tired knight in dented armor", "ally_name": "the knight"}, {"order": 2}),
            Option("A small robot that lost its memory",
                   {"ally": "a small round robot with one glowing eye", "ally_name": "the robot"}, {"empathy": 2}),
        ),
        free_text_fields=("ally",),
        shots=_shots(
            ("{hero} walking alone through a misty forest", "The road was long and lonely.", 0.6),
            ("{ally} appearing from the shadows on the road", "Until {ally} appeared.", 0.9),
            ("{hero} and {ally} facing each other warily", "Neither trusted the other."),
            ("{hero} and {ally} sharing food by a campfire at night", "But hunger and cold make friends of strangers."),
            ("{hero} and {ally} laughing together by the fire", "By morning, they were companions."),
            ("{hero} and {ally} walking side by side toward mountains, wide shot", "Together, they went on."),
        ),
    ),
    Act(
        title="The test",
        question="The villain attacks a village. What does the hero do?",
        options=(
            Option("Save the village and lose the trail",
                   {"test": "saving the village", "test_visual": "{hero} pulling villagers from a burning house"},
                   {"empathy": 2, "courage": 1}),
            Option("Chase the villain and leave the village",
                   {"test": "chasing the villain", "test_visual": "{hero} running after a shadowy villain"},
                   {"courage": 2, "order": -1}),
            Option("Make a deal with the villain",
                   {"test": "a deal with the villain", "test_visual": "{hero} shaking hands with a shadowy villain"},
                   {"curiosity": 1, "order": 1}),
        ),
        free_text_fields=("test", "test_visual"),
        shots=_shots(
            ("a village burning at night, a dark villain watching from a hill", "Then the shadow struck.", 0.9),
            ("{hero} and {ally} running toward the village, dramatic low angle", "There was no time to think."),
            ("the villain turning to flee into the dark", "And only time for one choice."),
            ("{test_visual}", "{hero} chose {test}.", 1.0),
            ("{ally} watching {hero}, close-up", "{ally} saw what the hero was made of."),
            ("smoke rising over the village at dawn", "When the smoke cleared, nothing was the same.", 0.6),
        ),
    ),
    Act(
        title="The cost",
        question="Every choice has a price. What does the hero lose?",
        options=(
            Option("Their ally, captured by the villain",
                   {"cost": "{ally}", "cost_visual": "{ally} being dragged away into darkness"}, {"courage": 1}),
            Option("Their memories of home",
                   {"cost": "the memories of home", "cost_visual": "{hero}'s memories dissolving like smoke"},
                   {"curiosity": 1}),
            Option("The power that made them special",
                   {"cost": "their only power", "cost_visual": "light fading from {hero}'s hands"}, {"empathy": 1}),
        ),
        free_text_fields=("cost", "cost_visual"),
        shots=_shots(
            ("{cost_visual}", "The price was {cost}.", 0.9),
            ("{hero} kneeling in the rain, alone", "For a while, {hero} could not go on."),
            ("an empty road under a grey sky", "The world felt very big and very quiet."),
            ("{hero} finding a small reminder of the journey in their bag", "Then something small changed everything."),
            ("{hero} standing up, determined, close-up", "{hero} stood up."),
            ("{hero} walking into the storm toward the villain's tower", "There was one thing left to do.", 0.8),
        ),
    ),
    Act(
        title="The new world",
        question="How does the story end?",
        options=(
            Option("Return home as a hero",
                   {"ending": "returning home", "ending_visual": "{hero} welcomed home by cheering crowds in {world}"},
                   {"order": 1}),
            Option("Stay in the new world",
                   {"ending": "a new life far away",
                    "ending_visual": "{hero} building a new home in a green valley far away"}, {"curiosity": 1}),
            Option("Become the new guardian",
                   {"ending": "becoming the guardian",
                    "ending_visual": "{hero} standing guard on a high tower at sunrise"}, {"courage": 1, "empathy": 1}),
        ),
        free_text_fields=("ending", "ending_visual"),
        shots=_shots(
            ("{hero} facing the villain at the top of a tower, lightning", "At the top of the tower, they met.", 1.0),
            ("the villain's shadow breaking apart into light", "And the shadow was finally undone.", 0.9),
            ("the sun rising over the land", "Morning came back to the world."),
            ("{ending_visual}", "The story ended with {ending}.", 0.9),
            ("{hero} smiling, close-up, warm light", "{hero} was not the same person who left."),
            ("{hero} walking into the sunset, wide shot", "And that is how this story was told. Only this once.", 0.8),
        ),
    ),
)


def _fill(template: str, fields: dict[str, str]) -> str:
    # Fields can reference other fields ("{hero} pulling villagers..."), so fill twice.
    for _ in range(2):
        template = template.format_map(_Defaulting(fields))
    return template


def _capitalize(text: str) -> str:
    return text[:1].upper() + text[1:]


class _Defaulting(dict):
    def __missing__(self, key):
        return "{" + key + "}"


@dataclass
class StoryState:
    style: str
    # name and ally_name are the short forms used in narration; free-text answers keep these defaults.
    fields: dict[str, str] = field(default_factory=lambda: {"name": "our hero", "ally_name": "the companion"})
    choices: list[str] = field(default_factory=list)  # the label (or free text) chosen per act
    values: dict[str, int] = field(default_factory=lambda: {a: 0 for a in AXES})

    @property
    def act(self) -> int:
        """Index of the next act to choose."""
        return len(self.choices)

    def choose(self, option_index: int | None = None, free_text: str | None = None) -> int:
        """Lock the current act with an option or a free-text answer. Returns the act index locked."""
        if self.act >= len(ACTS):
            raise ValueError("the story is already complete")
        act = ACTS[self.act]
        if free_text:
            text = free_text.strip()[:200]
            for name in act.free_text_fields:
                self.fields[name] = text
            self.choices.append(text)
        else:
            option = act.options[option_index]
            self.fields.update(option.fields)
            self.choices.append(option.label)
            for axis, delta in option.values.items():
                self.values[axis] += delta
        return self.act - 1

    def shots(self, act_index: int) -> list[dict]:
        """Prompt and narration for each shot of a locked act."""
        style = STYLES[self.style]["prompt"]
        spoken = {**self.fields, "hero": self.fields["name"], "ally": self.fields["ally_name"]}
        return [{"act": act_index, "index": i, "priority": t.priority,
                 "prompt": f"{_fill(t.prompt, self.fields)}, {style}",
                 "narration": _capitalize(_fill(t.narration, spoken))}
                for i, t in enumerate(ACTS[act_index].shots)]


def acts_public() -> list[dict]:
    """What the web app shows for each act."""
    return [{"title": a.title, "question": a.question, "options": [o.label for o in a.options]} for a in ACTS]
