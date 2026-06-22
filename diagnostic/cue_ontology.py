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


def _phrase_pattern(phrase: str) -> str:
    return re.escape(phrase).replace(r"\ ", r"\s+")


def _simple_cue(name: str, category: str, aliases: Iterable[str]) -> Cue:
    return Cue(name, category, tuple(_phrase_pattern(alias) for alias in aliases))


def _append_unique(cues: list[Cue], additions: Iterable[Cue]) -> None:
    seen = {cue.name for cue in cues}
    for cue in additions:
        if cue.name in seen:
            continue
        cues.append(cue)
        seen.add(cue.name)


def default_cues() -> list[Cue]:
    colors = ("black", "white", "red", "blue", "green", "yellow", "gray", "brown", "pink", "purple")
    cues: list[Cue] = []
    cues.extend(_colored("upper_body_color", colors, ("shirt", "top", "jacket", "coat", "hoodie", "sweater", "upper")))
    cues.extend(_colored("lower_body_color", colors, ("pants", "trousers", "jeans", "shorts", "skirt", "lower")))
    cues.extend(_colored("footwear_color", ("black", "white", "red", "blue", "gray", "brown"), ("shoes", "sneakers", "boots", "footwear")))
    case_colors = ("black", "white", "red", "blue", "green", "yellow", "gray", "brown", "pink", "purple", "orange")
    upper_nouns = ("shirt", "top", "jacket", "coat", "hoodie", "sweater", "vest", "blazer", "suit", "t-shirt", "tshirt", "upper")
    lower_nouns = ("pants", "trousers", "jeans", "shorts", "skirt", "dress", "leggings", "lower")
    footwear_nouns = ("shoes", "sneakers", "boots", "sandals", "heels", "footwear")
    _append_unique(cues, _colored("upper_body_color", case_colors, upper_nouns))
    for color in case_colors:
        color_pattern = _color_patterns(color)
        noun_pattern = "(?:" + "|".join(re.escape(noun) for noun in upper_nouns) + ")"
        cues.append(
            Cue(
                f"{color} upper-body clothing",
                "upper_body_color",
                (
                    rf"{color_pattern}\s+{noun_pattern}",
                    rf"{noun_pattern}\s+(?:is\s+)?{color_pattern}",
                ),
            )
        )
    _append_unique(cues, _colored("lower_body_color", case_colors, lower_nouns))
    for color in case_colors:
        color_pattern = _color_patterns(color)
        noun_pattern = "(?:" + "|".join(re.escape(noun) for noun in lower_nouns) + ")"
        cues.append(
            Cue(
                f"{color} lower-body clothing",
                "lower_body_color",
                (
                    rf"{color_pattern}\s+{noun_pattern}",
                    rf"{noun_pattern}\s+(?:is\s+)?{color_pattern}",
                ),
            )
        )
    _append_unique(cues, _colored("footwear_color", ("black", "white", "red", "blue", "gray", "brown"), footwear_nouns))
    for color in ("black", "white", "red", "blue", "gray", "brown"):
        color_pattern = _color_patterns(color)
        noun_pattern = "(?:" + "|".join(re.escape(noun) for noun in footwear_nouns) + ")"
        cues.append(
            Cue(
                f"{color} footwear",
                "footwear_color",
                (
                    rf"{color_pattern}\s+{noun_pattern}",
                    rf"{noun_pattern}\s+(?:is\s+)?{color_pattern}",
                ),
            )
        )
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
    _append_unique(
        cues,
        [
            _simple_cue("shirt", "upper_body_garment", ("shirt", "shirts")),
            Cue("t-shirt", "upper_body_garment", (r"t-?shirt", r"t\s+shirt", r"tee\s+shirt")),
            _simple_cue("jacket", "upper_body_garment", ("jacket", "jackets")),
            _simple_cue("coat", "upper_body_garment", ("coat", "coats")),
            _simple_cue("hoodie", "upper_body_garment", ("hoodie", "hooded sweatshirt")),
            _simple_cue("sweater", "upper_body_garment", ("sweater", "sweatshirt")),
            _simple_cue("vest", "upper_body_garment", ("vest",)),
            _simple_cue("blazer", "upper_body_garment", ("blazer",)),
            _simple_cue("suit", "upper_body_garment", ("suit",)),
            _simple_cue("top", "upper_body_garment", ("top",)),
            _simple_cue("pants", "lower_body_garment", ("pants", "trousers")),
            _simple_cue("jeans", "lower_body_garment", ("jeans",)),
            _simple_cue("shorts", "lower_body_garment", ("shorts",)),
            _simple_cue("skirt", "lower_body_garment", ("skirt",)),
            _simple_cue("dress", "lower_body_garment", ("dress",)),
            _simple_cue("leggings", "lower_body_garment", ("leggings",)),
            _simple_cue("shoes", "footwear", ("shoes", "shoe")),
            _simple_cue("sneakers", "footwear", ("sneakers", "trainers")),
            _simple_cue("boots", "footwear", ("boots", "boot")),
            _simple_cue("sandals", "footwear", ("sandals", "sandal")),
            _simple_cue("heels", "footwear", ("heels", "high heels")),
            _simple_cue("tote bag", "carried_item", ("tote bag",)),
            _simple_cue("shopping bag", "carried_item", ("shopping bag",)),
            _simple_cue("briefcase", "carried_item", ("briefcase",)),
            _simple_cue("suitcase", "carried_item", ("suitcase", "luggage")),
            _simple_cue("sunglasses", "accessory", ("sunglasses", "sun glasses")),
            _simple_cue("scarf", "accessory", ("scarf",)),
            _simple_cue("mask", "accessory", ("mask", "face mask")),
            _simple_cue("long sleeves", "sleeve_length", ("long sleeves", "long-sleeved", "long sleeve")),
            Cue("checkered clothing", "pattern", (r"checkered", r"checked")),
            Cue("floral clothing", "pattern", (r"floral", r"flowered")),
            Cue("camouflage clothing", "pattern", (r"camouflage", r"camo")),
            Cue("logo clothing", "pattern", (r"logo",)),
            Cue("printed clothing", "pattern", (r"printed", r"print")),
            Cue("polka-dot clothing", "pattern", (r"polka[\s-]?dot", r"polka[\s-]?dotted")),
        ],
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
