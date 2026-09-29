import re
from dataclasses import dataclass, field
from typing import Dict, List

COLOR_RGB = {"red": (220, 40, 40), "green": (40, 170, 70), "blue": (45, 90, 210), "yellow": (220, 190, 30), "black": (25, 25, 25), "white": (235, 235, 235), "orange": (230, 120, 30), "pink": (220, 100, 150), "brown": (130, 80, 45)}
NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
STOP_WORDS = {"on", "in", "with", "and", "beside", "near", "of", "at", "under", "by"}
# Adjectives that commonly sit between a count and its noun ("two small red cups").
MODIFIERS = set(COLOR_RGB) | {"small", "large", "big", "little", "old", "new", "tall", "short", "young", "decorated", "colorful", "traditional"}

@dataclass
class PromptRequirements:
    objects: Dict[str, int] = field(default_factory=dict)
    colors: List[str] = field(default_factory=list)

def normalize_label(value: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", value.lower()).strip()

def _singular(word: str) -> str:
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith(("ches", "shes", "xes", "sses", "oes")):
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word

def extract_prompt_requirements(prompt: str, expected_objects: Dict[str, int] | None = None) -> PromptRequirements:
    objects = {normalize_label(k): int(v) for k, v in (expected_objects or {}).items()}
    lower = prompt.lower()
    for word, count in NUMBER_WORDS.items():
        for match in re.finditer(rf"\b{word}\s+([a-z][a-z -]*)", lower):
            words = match.group(1).split()
            while words and words[0] in MODIFIERS:
                words.pop(0)
            if words and words[0] not in STOP_WORDS:
                objects.setdefault(normalize_label(_singular(words[0])), count)
    colors = [color for color in COLOR_RGB if re.search(rf"\b{color}\b", lower)]
    return PromptRequirements(objects=objects, colors=colors)
