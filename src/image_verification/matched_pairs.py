"""Matched prompt pairs for the evaluator-reliability study.

Every India-centric prompt is paired with a control prompt that keeps the same
sentence structure, object requirements and difficulty, and changes only the
cultural or script content. Differences in evaluator reliability between the
two sides of a pair can then be attributed to that content rather than to
harder prompts.

Two tracks:

* **Script track (3 conditions).** Each Indic signage prompt appears as
  ``india_indic`` (Indian city, Indic script), ``india_latin`` (same Indian
  city, English text) and ``global_latin`` (Western city, English text).
  Comparing ``india_indic`` with ``india_latin`` isolates the script effect;
  ``india_latin`` with ``global_latin`` isolates the locale effect.
* **Cultural track (2 conditions).** Hand-written ``india`` / ``global`` pairs
  across agriculture, public services, festivals and everyday scenes.

Code-mixed signage prompts are kept as an unpaired descriptive track.
"""
from __future__ import annotations

import argparse
from typing import Dict, List, Tuple

from .aikosh import agriculture_source_metadata
from .prompt_bank import CODE_MIXED_PROMPTS, INDIC_SCRIPT_PROMPTS, PromptSpec, save_prompt_bank

BANK_VERSION = "matched_v1"

INDIA_CENTRIC_CONDITIONS = {"india_indic", "india"}
CONTROL_CONDITIONS = {"global_latin", "global"}

# English rendering of each signage concept, keyed by the scene noun used in
# INDIC_SCRIPT_PROMPTS. "clinic" prompts ask for the word for hospital and
# "station" prompts ask for a welcome sign.
SCENE_TO_ENGLISH = {
    "tea stall": "TEA",
    "tea shop": "TEA",
    "school": "SCHOOL",
    "clinic": "HOSPITAL",
    "market": "MARKET",
    "station": "WELCOME",
}

# English-speaking cities so the control text is natural signage there.
GLOBAL_CITIES = ["London", "Boston", "Sydney", "Toronto", "Chicago", "Dublin", "Seattle", "Manchester", "Auckland"]

# (category, india prompt, india metadata, control prompt, control metadata).
# Controls keep sentence structure and object requirements identical.
CULTURAL_PAIRS: List[Tuple[str, str, Dict, str, Dict]] = [
    ("public_service",
     "an Indian public health poster about vaccination at a rural clinic",
     {"expected_objects": {"poster": 1, "clinic": 1}, "cultural_markers": ["rural clinic", "public health poster", "India"]},
     "an American public health poster about vaccination at a rural clinic",
     {"expected_objects": {"poster": 1, "clinic": 1}, "cultural_markers": ["rural clinic", "public health poster", "United States"]}),
    ("public_service",
     "a clean illustration of an Aadhaar enrollment desk in a government office",
     {"expected_objects": {"desk": 1, "person": 1}, "cultural_markers": ["Aadhaar", "government office", "India"]},
     "a clean illustration of a Social Security enrollment desk in a government office",
     {"expected_objects": {"desk": 1, "person": 1}, "cultural_markers": ["Social Security", "government office", "United States"]}),
    ("public_service",
     "a public-service poster encouraging water conservation in Rajasthan",
     {"expected_objects": {"poster": 1}, "cultural_markers": ["Rajasthan", "water conservation", "public-service poster"]},
     "a public-service poster encouraging water conservation in Arizona",
     {"expected_objects": {"poster": 1}, "cultural_markers": ["Arizona", "water conservation", "public-service poster"]}),
    ("public_service",
     "an Indian railway platform information board with passengers waiting",
     {"expected_objects": {"person": 2, "train": 1}, "cultural_markers": ["Indian railway", "platform", "information board"]},
     "a British railway platform information board with passengers waiting",
     {"expected_objects": {"person": 2, "train": 1}, "cultural_markers": ["British railway", "platform", "information board"]}),
    ("public_service",
     "a Swachh Bharat cleanliness poster on a city wall",
     {"expected_objects": {"poster": 1}, "cultural_markers": ["Swachh Bharat", "cleanliness poster", "city wall"]},
     "a Keep Britain Tidy cleanliness poster on a city wall",
     {"expected_objects": {"poster": 1}, "cultural_markers": ["Keep Britain Tidy", "cleanliness poster", "city wall"]}),
    ("agriculture",
     "an Indian farmer inspecting rice paddy fields during monsoon",
     {"expected_objects": {"person": 1}, "cultural_markers": ["farmer", "rice paddy", "monsoon", "India"]},
     "an American farmer inspecting rice paddy fields during spring rains in Arkansas",
     {"expected_objects": {"person": 1}, "cultural_markers": ["farmer", "rice paddy", "spring rains", "Arkansas"]}),
    ("agriculture",
     "a cotton field in Maharashtra with a farmer holding a mobile phone",
     {"expected_objects": {"person": 1, "phone": 1}, "cultural_markers": ["cotton field", "Maharashtra", "farmer"]},
     "a cotton field in Texas with a farmer holding a mobile phone",
     {"expected_objects": {"person": 1, "phone": 1}, "cultural_markers": ["cotton field", "Texas", "farmer"]}),
    ("agriculture",
     "a small dairy farm in Gujarat with two cows and a farmer",
     {"expected_objects": {"cow": 2, "person": 1}, "cultural_markers": ["dairy farm", "Gujarat", "cow"]},
     "a small dairy farm in Wisconsin with two cows and a farmer",
     {"expected_objects": {"cow": 2, "person": 1}, "cultural_markers": ["dairy farm", "Wisconsin", "cow"]}),
    ("agriculture",
     "a millet field in rural Karnataka under a clear sky",
     {"expected_objects": {"field": 1}, "cultural_markers": ["millet", "Karnataka", "rural India"]},
     "a wheat field in rural Kansas under a clear sky",
     {"expected_objects": {"field": 1}, "cultural_markers": ["wheat", "Kansas", "rural America"]}),
    ("agriculture",
     "a vegetable mandi in India with crates of tomatoes and onions",
     {"expected_objects": {"tomato": 1, "onion": 1}, "cultural_markers": ["mandi", "tomatoes", "onions", "India"]},
     "a farmers market in the United States with crates of tomatoes and onions",
     {"expected_objects": {"tomato": 1, "onion": 1}, "cultural_markers": ["farmers market", "tomatoes", "onions", "United States"]}),
    ("festival",
     "a Diwali scene with diyas arranged outside an Indian home",
     {"expected_objects": {"diya": 2}, "cultural_markers": ["Diwali", "diyas", "Indian home"]},
     "a Christmas scene with candles arranged outside an American home",
     {"expected_objects": {"candle": 2}, "cultural_markers": ["Christmas", "candles", "American home"]}),
    ("festival",
     "a Pongal celebration with a decorated clay pot and sugarcane",
     {"expected_objects": {"pot": 1, "sugarcane": 1}, "cultural_markers": ["Pongal", "clay pot", "sugarcane"]},
     "a Thanksgiving celebration with a decorated pumpkin and corn",
     {"expected_objects": {"pumpkin": 1, "corn": 1}, "cultural_markers": ["Thanksgiving", "pumpkin", "corn"]}),
    ("festival",
     "a Durga Puja pandal decorated with flowers in Kolkata",
     {"expected_objects": {"pandal": 1}, "cultural_markers": ["Durga Puja", "Kolkata", "flowers"]},
     "an Oktoberfest tent decorated with flowers in Munich",
     {"expected_objects": {"tent": 1}, "cultural_markers": ["Oktoberfest", "Munich", "flowers"]}),
    ("festival",
     "an Onam pookalam flower arrangement outside a Kerala home",
     {"expected_objects": {"flower arrangement": 1}, "cultural_markers": ["Onam", "pookalam", "Kerala"]},
     "a May Day flower wreath outside an English cottage",
     {"expected_objects": {"wreath": 1}, "cultural_markers": ["May Day", "flower wreath", "English cottage"]}),
    ("festival",
     "a traditional Indian wedding entrance decorated with marigold garlands",
     {"expected_objects": {"garland": 1}, "cultural_markers": ["Indian wedding", "marigold", "entrance"]},
     "a traditional American wedding entrance decorated with rose garlands",
     {"expected_objects": {"garland": 1}, "cultural_markers": ["American wedding", "roses", "entrance"]}),
    ("everyday_scene",
     "a small kirana store with shelves of snacks and a shopkeeper",
     {"expected_objects": {"store": 1, "person": 1}, "cultural_markers": ["kirana store", "snacks", "shopkeeper"]},
     "a small convenience store with shelves of snacks and a shopkeeper",
     {"expected_objects": {"store": 1, "person": 1}, "cultural_markers": ["convenience store", "snacks", "shopkeeper"]}),
    ("everyday_scene",
     "an auto rickshaw waiting beside a busy street in Mumbai",
     {"expected_objects": {"auto rickshaw": 1}, "cultural_markers": ["auto rickshaw", "Mumbai", "street"]},
     "a taxi cab waiting beside a busy street in New York",
     {"expected_objects": {"taxi": 1}, "cultural_markers": ["taxi cab", "New York", "street"]}),
    ("everyday_scene",
     "a chai stall beside an Indian railway station in the morning",
     {"expected_objects": {"stall": 1}, "cultural_markers": ["chai stall", "railway station", "India"]},
     "a coffee cart beside a British railway station in the morning",
     {"expected_objects": {"cart": 1}, "cultural_markers": ["coffee cart", "railway station", "United Kingdom"]}),
    ("everyday_scene",
     "a family shopping at a colorful Indian street market",
     {"expected_objects": {"person": 2, "market": 1}, "cultural_markers": ["street market", "India", "family"]},
     "a family shopping at a colorful European street market",
     {"expected_objects": {"person": 2, "market": 1}, "cultural_markers": ["street market", "Europe", "family"]}),
    ("everyday_scene",
     "a school bus arriving at a neighborhood road in Bengaluru",
     {"expected_objects": {"bus": 1}, "cultural_markers": ["school bus", "Bengaluru", "neighborhood road"]},
     "a school bus arriving at a neighborhood road in Chicago",
     {"expected_objects": {"bus": 1}, "cultural_markers": ["school bus", "Chicago", "neighborhood road"]}),
]


def _article(word: str) -> str:
    return "an" if word[:1].lower() in "aeiou" else "a"


def _parse_signage_prompt(prompt: str, region: str, text: str) -> str:
    """Return the scene noun from prompts like 'a Delhi tea stall sign that says चाय'."""
    head = prompt.rsplit(f" sign that says {text}", 1)[0]
    scene = head.split(f"{region} ", 1)[1]
    if scene not in SCENE_TO_ENGLISH:
        raise ValueError(f"No English control text for signage scene {scene!r} in prompt {prompt!r}.")
    return scene


def _spec(pair_id: str, condition: str, track: str, category: str, prompt: str, metadata: Dict) -> PromptSpec:
    meta = dict(metadata)
    expected_objects = meta.pop("expected_objects", {}) or {}
    meta.update({
        "bank_version": BANK_VERSION,
        "pair_id": pair_id,
        "condition": condition,
        "is_india_centric": condition in INDIA_CENTRIC_CONDITIONS,
        "is_control": condition in CONTROL_CONDITIONS,
        "track": track,
    })
    prompt_id = f"{pair_id}__{condition}"
    return PromptSpec(prompt_id, prompt, category, dict(expected_objects), "hard", meta)


def build_script_track() -> List[PromptSpec]:
    specs: List[PromptSpec] = []
    for index, (language, script, region, text, prompt) in enumerate(INDIC_SCRIPT_PROMPTS, 1):
        pair_id = f"script_{index:03d}"
        scene = _parse_signage_prompt(prompt, region, text)
        english = SCENE_TO_ENGLISH[scene]
        city = GLOBAL_CITIES[(index - 1) % len(GLOBAL_CITIES)]
        shared = {"concept": scene, "indic_language": language, "indic_script": script}
        specs.append(_spec(pair_id, "india_indic", "script", "script_rendering", prompt, {
            **shared, "expected_text": [text], "language": language, "script": script, "region": region,
            "cultural_markers": [region, "Indian signage"],
        }))
        specs.append(_spec(pair_id, "india_latin", "script", "script_rendering",
                           f"{_article(region)} {region} {scene} sign that says {english}", {
            **shared, "expected_text": [english], "language": "English", "script": "Latin", "region": region,
            "cultural_markers": [region, "Indian signage"],
        }))
        specs.append(_spec(pair_id, "global_latin", "script", "script_rendering",
                           f"{_article(city)} {city} {scene} sign that says {english}", {
            **shared, "expected_text": [english], "language": "English", "script": "Latin", "region": city,
            "cultural_markers": [city, "signage"],
        }))
    return specs


def build_cultural_track() -> List[PromptSpec]:
    specs: List[PromptSpec] = []
    for index, (category, india_prompt, india_meta, control_prompt, control_meta) in enumerate(CULTURAL_PAIRS, 1):
        pair_id = f"culture_{index:03d}"
        india_extra = agriculture_source_metadata() if category == "agriculture" else {}
        specs.append(_spec(pair_id, "india", "cultural", category, india_prompt, {**india_meta, **india_extra}))
        specs.append(_spec(pair_id, "global", "cultural", category, control_prompt, control_meta))
    return specs


def build_code_mixed_track() -> List[PromptSpec]:
    specs: List[PromptSpec] = []
    for index, (language, script, prompt) in enumerate(CODE_MIXED_PROMPTS, 1):
        texts = prompt.split(" says ", 1)[-1].split(" and ")
        spec = _spec(f"codemixed_{index:03d}", "india_code_mixed", "code_mixed", "code_mixed_signage", prompt, {
            "expected_text": texts, "language": language, "script": script,
            "cultural_markers": ["Indian bilingual signage"],
        })
        # Unpaired: used only for descriptive script analyses.
        spec.metadata["pair_id"] = None
        specs.append(spec)
    return specs


def build_matched_bank(include_code_mixed: bool = True) -> List[PromptSpec]:
    specs = build_script_track() + build_cultural_track()
    if include_code_mixed:
        specs += build_code_mixed_track()
    return specs


def main() -> None:
    parser = argparse.ArgumentParser(description="Write the matched India/control prompt bank.")
    parser.add_argument("--output", default="data/image_verification/matched_prompt_bank_v1.jsonl")
    parser.add_argument("--no-code-mixed", action="store_true")
    args = parser.parse_args()
    specs = build_matched_bank(include_code_mixed=not args.no_code_mixed)
    save_prompt_bank(specs, args.output)
    counts: Dict[str, int] = {}
    for spec in specs:
        counts[spec.metadata["condition"]] = counts.get(spec.metadata["condition"], 0) + 1
    print(f"[matched-bank] wrote {len(specs)} prompts -> {args.output} {counts}")


if __name__ == "__main__":
    main()
