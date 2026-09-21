import re
from dataclasses import dataclass, field
from typing import Dict, List

COLOR_RGB = {"red": (220, 40, 40), "green": (40, 170, 70), "blue": (45, 90, 210), "yellow": (220, 190, 30), "black": (25, 25, 25), "white": (235, 235, 235), "orange": (230, 120, 30), "pink": (220, 100, 150), "brown": (130, 80, 45)}
NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}

@dataclass
class PromptRequirements:
    objects: Dict[str, int] = field(default_factory=dict)
    colors: List[str] = field(default_factory=list)

def normalize_label(value: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", value.lower()).strip()

def extract_prompt_requirements(prompt: str, expected_objects: Dict[str, int] | None = None) -> PromptRequirements:
    objects = {normalize_label(k): int(v) for k, v in (expected_objects or {}).items()}
    lower = prompt.lower()
    for word, count in NUMBER_WORDS.items():
        for match in re.finditer(rf"\b{word}\s+([a-z][a-z -]*)", lower):
            phrase = match.group(1).split(" ")[0]
            if phrase not in {"on", "in", "with", "and", "beside", "near", "of"}:
                objects.setdefault(normalize_label(phrase.rstrip("s")), count)
    colors = [color for color in COLOR_RGB if re.search(rf"\b{color}\b", lower)]
    return PromptRequirements(objects=objects, colors=colors)

