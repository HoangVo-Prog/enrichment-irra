"""Fixed cue ontology plus optional JSON vocabulary loading."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class Cue:
    name: str
    category: str
    patterns: tuple[str, ...]

    def regex(self) -> re.Pattern[str]:
        joined = "|".join(self.patterns)
        return re.compile(rf"\b(?:{joined})\b", flags=re.IGNORECASE)


def _color_patterns(color: str) -> str:
    variants = {
        "gray": "(?:gray|grey)",
        "dark": "(?:dark|black)",
        "light": "(?:light|white)",
    }
    return variants.get(color, re.escape(color))


def _colored(category: str, colors: Iterable[str], nouns: Iterable[str]) -> list[Cue]:
    cues = []
    noun_pattern = "(?:" + "|".join(re.escape(noun) for noun in nouns) + ")"
    for color in colors:
        color_pattern = _color_patterns(color)
        name = f"{color} {next(iter(nouns))}"
        cues.append(
            Cue(
                name=name,
                category=category,
                patterns=(
                    rf"{color_pattern}\s+{noun_pattern}",
                    rf"{noun_pattern}\s+(?:is\s+)?{color_pattern}",
                ),
            )
        )
    return cues


def default_cues() -> list[Cue]:
    colors = ("black", "white", "red", "blue", "green", "yellow", "gray", "brown", "pink", "purple")
    cues: list[Cue] = []
    cues.extend(_colored("upper_body_color", colors, ("shirt", "top", "jacket", "coat", "hoodie", "sweater", "upper")))
    cues.extend(_colored("lower_body_color", colors, ("pants", "trousers", "jeans", "shorts", "skirt", "lower")))
    cues.extend(_colored("footwear_color", ("black", "white", "red", "blue", "gray", "brown"), ("shoes", "sneakers", "boots", "footwear")))
    cues.extend(
        [
            Cue("bag", "carried_item", (r"bag", r"bags")),
            Cue("backpack", "carried_item", (r"backpack", r"rucksack")),
            Cue("handbag", "carried_item", (r"handbag", r"purse")),
            Cue("shoulder bag", "carried_item", (r"shoulder\s+bag", r"crossbody\s+bag")),
            Cue("hat", "accessory", (r"hat", r"hats")),
            Cue("cap", "accessory", (r"cap", r"baseball\s+cap")),
            Cue("glasses", "accessory", (r"glasses", r"sunglasses", r"eyeglasses")),
            Cue("umbrella", "accessory", (r"umbrella",)),
            Cue("striped clothing", "pattern", (r"striped", r"stripes", r"stripe")),
            Cue("plaid clothing", "pattern", (r"plaid", r"checkered", r"checked")),
            Cue("patterned clothing", "pattern", (r"patterned", r"printed", r"floral")),
            Cue("plain clothing", "pattern", (r"plain", r"solid\s+color", r"solid-colored")),
        ]
    )
    return cues


def load_cues(path: str | None) -> list[Cue]:
    if path is None:
        return default_cues()
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    cues: list[Cue] = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, str):
                cues.append(Cue(item, "custom", (re.escape(item),)))
            elif isinstance(item, dict):
                name = str(item["name"])
                category = str(item.get("category", "custom"))
                patterns = tuple(str(p) for p in item.get("patterns", [re.escape(name)]))
                cues.append(Cue(name, category, patterns))
    elif isinstance(data, dict):
        for category, values in data.items():
            for item in values:
                if isinstance(item, str):
                    cues.append(Cue(item, str(category), (re.escape(item),)))
                elif isinstance(item, dict):
                    name = str(item["name"])
                    patterns = tuple(str(p) for p in item.get("patterns", [re.escape(name)]))
                    cues.append(Cue(name, str(category), patterns))
    else:
        raise ValueError("Cue vocabulary JSON must be a list or category mapping")
    if not cues:
        raise ValueError("Cue vocabulary is empty")
    return cues


def cue_map(cues: Iterable[Cue]) -> dict[str, Cue]:
    return {cue.name: cue for cue in cues}


def cue_prompts(cue: str, templates: Iterable[str]) -> list[str]:
    return [template.format(cue=cue) for template in templates]
