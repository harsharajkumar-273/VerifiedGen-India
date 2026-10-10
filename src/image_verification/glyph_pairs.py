"""Glyph minimal pairs: do image evaluators read Indic scripts?

Each pair is two images rendered with identical layout, font, colours and
noise. They differ only in the sign text: the *clean* image shows the
requested word correctly; the *corrupt* image shows one controlled error.
Ground truth comes by construction, so no human labels are needed to score
an evaluator. Native readers only spot-check that each error type really is
an error (see ``docs/GLYPH_PAIRS.md``).

An evaluator "passes" a pair when it scores the clean image above the corrupt
one for the same prompt. Pairwise accuracy is compared between Indic scripts
and a Latin-script control on the error types the two share.

**Prompt mirroring.** Many errors change the image in ways an evaluator can
exploit without reading: dropping a letter leaves less ink, a pseudo-word
looks less familiar. So every pair is also scored in the *mirrored* direction:
the same two images, with a prompt that asks for the corrupted string. The
image that matched the prompt is now the other one. Any preference that
ignores the prompt (ink, aesthetics, a prior for real words) scores exactly
0.5 on the forward and mirrored pairs together; only an evaluator that
compares the requested text with the rendered text scores above it.
Forward minus mirrored accuracy measures that prompt-independent preference.
``unshaped`` and ``null_font`` cannot be mirrored (the requested text is the
same on both sides) and are reported forward only, against prompt-blind
baselines.

Error types (``ERROR_TYPES``):

* ``drop_vowel_sign`` – remove one dependent vowel sign (Latin: one vowel letter).
* ``swap_vowel_sign`` – replace one vowel sign with another (Latin: swap a vowel).
* ``lookalike`` – replace one letter with a visually or phonetically confusable one.
* ``drop_akshara`` – remove one whole syllable cluster (Latin: one letter).
* ``pseudo_word`` – random well-formed syllables of the same length in the same script.
* ``unshaped`` – the correct text rendered *without* complex-script shaping, so
  pre-base vowel signs land after their consonant and conjuncts fall apart.
  Indic only; a realistic generator and software failure.
* ``script_swap`` – the word transliterated by code point into a neighbouring
  script (Devanagari to Gujarati, Kannada to Telugu, Tamil to Malayalam). Indic only.
* ``transliterated`` – the word in Latin letters instead of the requested script. Indic only.
* ``null_font`` – *not an error*: the correct text in a different font. Accuracy
  should be 0.5; it measures the evaluator's noise floor.
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
import random
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

LEXICON_PATH = Path(__file__).resolve().parents[2] / "data" / "glyph_pairs" / "lexicon_v1.json"
FONT_DIR = Path(__file__).resolve().parents[2] / "data" / "fonts"

FONT_BASE_URL = "https://raw.githubusercontent.com/google/fonts/main/ofl"
FONT_FILES = {
    "NotoSans": "notosans/NotoSans%5Bwdth,wght%5D.ttf",
    "NotoSerif": "notoserif/NotoSerif%5Bwdth,wght%5D.ttf",
    "NotoSansDevanagari": "notosansdevanagari/NotoSansDevanagari%5Bwdth,wght%5D.ttf",
    "NotoSerifDevanagari": "notoserifdevanagari/NotoSerifDevanagari%5Bwdth,wght%5D.ttf",
    "NotoSansKannada": "notosanskannada/NotoSansKannada%5Bwdth,wght%5D.ttf",
    "NotoSerifKannada": "notoserifkannada/NotoSerifKannada%5Bwght%5D.ttf",
    "NotoSansTamil": "notosanstamil/NotoSansTamil%5Bwdth,wght%5D.ttf",
    "NotoSerifTamil": "notoseriftamil/NotoSerifTamil%5Bwdth,wght%5D.ttf",
    "NotoSansGujarati": "notosansgujarati/NotoSansGujarati%5Bwdth,wght%5D.ttf",
    "NotoSansTelugu": "notosanstelugu/NotoSansTelugu%5Bwdth,wght%5D.ttf",
    "NotoSansMalayalam": "notosansmalayalam/NotoSansMalayalam%5Bwdth,wght%5D.ttf",
}


@dataclass(frozen=True)
class ScriptSpec:
    code: str
    name: str
    language: str  # language named in the prompt
    bcp47: str  # passed to the shaper
    block_start: Optional[int]  # Unicode block for ISCII-aligned Indic scripts
    fonts: Tuple[str, ...]
    swap_to: Optional[str] = None
    # Confusable letter pairs. Unvalidated: native readers must confirm each
    # pair is a plausible, visible error before the benchmark is frozen.
    lookalikes: Tuple[Tuple[str, str], ...] = ()
    # Tamil writes a visible pulli instead of forming conjuncts, so a consonant
    # after the virama starts a new syllable for a Tamil reader.
    conjuncts: bool = True

    @property
    def is_indic(self) -> bool:
        return self.block_start is not None


SCRIPTS: Dict[str, ScriptSpec] = {
    "latn": ScriptSpec("latn", "Latin", "English", "en", None, ("NotoSans", "NotoSerif"),
                       lookalikes=(("E", "F"), ("C", "G"), ("O", "Q"), ("P", "R"), ("U", "V"), ("M", "N"), ("I", "L"))),
    "deva": ScriptSpec("deva", "Devanagari", "Hindi", "hi", 0x0900, ("NotoSansDevanagari", "NotoSerifDevanagari"), "gujr",
                       (("ब", "व"), ("घ", "ध"), ("म", "भ"), ("प", "ष"), ("थ", "य"), ("ड", "ङ"))),
    "knda": ScriptSpec("knda", "Kannada", "Kannada", "kn", 0x0C80, ("NotoSansKannada", "NotoSerifKannada"), "telu",
                       (("ತ", "ಥ"), ("ದ", "ಧ"), ("ಡ", "ಢ"), ("ಬ", "ಭ"), ("ಪ", "ಫ"), ("ಣ", "ನ"))),
    "taml": ScriptSpec("taml", "Tamil", "Tamil", "ta", 0x0B80, ("NotoSansTamil", "NotoSerifTamil"), "mlym",
                       (("ன", "ண"), ("ல", "ள"), ("ர", "ற"), ("ழ", "ள")), conjuncts=False),
    # Swap targets only.
    "gujr": ScriptSpec("gujr", "Gujarati", "Gujarati", "gu", 0x0A80, ("NotoSansGujarati",)),
    "telu": ScriptSpec("telu", "Telugu", "Telugu", "te", 0x0C00, ("NotoSansTelugu",)),
    "mlym": ScriptSpec("mlym", "Malayalam", "Malayalam", "ml", 0x0D00, ("NotoSansMalayalam",)),
}
STUDY_SCRIPTS = ("deva", "knda", "taml", "latn")

# Offsets inside an ISCII-aligned Indic block.
CONSONANT_OFFSETS = range(0x15, 0x3A)
VOWEL_SIGN_OFFSETS = (0x3E, 0x3F, 0x40, 0x41, 0x42, 0x46, 0x47, 0x48, 0x4A, 0x4B, 0x4C)
VIRAMA_OFFSET = 0x4D
LATIN_VOWELS = "AEIOU"
JOINERS = {"‌", "‍"}

MIRRORABLE_EXCLUDED = ("unshaped", "null_font")
SHARED_ERROR_TYPES = ("drop_vowel_sign", "swap_vowel_sign", "lookalike", "drop_akshara", "pseudo_word")
INDIC_ONLY_ERROR_TYPES = ("unshaped", "script_swap", "transliterated")
ERROR_TYPES = SHARED_ERROR_TYPES + INDIC_ONLY_ERROR_TYPES + ("null_font",)


# ---------------------------------------------------------------- text level

def _offset(char: str, spec: ScriptSpec) -> Optional[int]:
    if spec.block_start is None:
        return None
    value = ord(char) - spec.block_start
    return value if 0 <= value < 0x80 else None


def is_vowel_sign(char: str, spec: ScriptSpec) -> bool:
    if not spec.is_indic:
        return char in LATIN_VOWELS
    return _offset(char, spec) in VOWEL_SIGN_OFFSETS and _assigned(char)


def is_consonant(char: str, spec: ScriptSpec) -> bool:
    if not spec.is_indic:
        return char.isalpha() and char not in LATIN_VOWELS
    return _offset(char, spec) in CONSONANT_OFFSETS and _assigned(char)


def _assigned(char: str) -> bool:
    return unicodedata.name(char, "") != ""


# Devanagari short e/o signs (U+0946, U+094A) exist only for transcribing
# Dravidian languages; a Hindi reader would not expect them on a sign.
RARE_SIGN_OFFSETS = {"deva": {0x46, 0x4A}}


def vowel_sign_pool(spec: ScriptSpec) -> List[str]:
    if not spec.is_indic:
        return list(LATIN_VOWELS)
    rare = RARE_SIGN_OFFSETS.get(spec.code, set())
    return [chr(spec.block_start + off) for off in VOWEL_SIGN_OFFSETS
            if off not in rare and _assigned(chr(spec.block_start + off))]


def aksharas(text: str, join_virama: bool = True) -> List[str]:
    """Split into orthographic syllables (grapheme clusters, joined across a virama).

    Spaces are returned as their own clusters so callers can skip them. For
    Latin text every letter is its own cluster. ``join_virama=False`` keeps
    ``C + virama`` as its own cluster, which is how Tamil is read.
    """
    clusters: List[str] = []
    for char in text:
        joins = bool(clusters) and not clusters[-1].isspace() and (
            unicodedata.category(char) in ("Mn", "Mc")
            or char in JOINERS
            or (join_virama and unicodedata.name(clusters[-1][-1], "").endswith("SIGN VIRAMA"))
            or clusters[-1][-1] in JOINERS
        )
        if joins and not char.isspace():
            clusters[-1] += char
        else:
            clusters.append(char)
    return clusters


def syllable_count(text: str, spec: ScriptSpec) -> int:
    """Length used for binning: aksharas for Indic, vowel groups for Latin.

    Letters and aksharas are not comparable units, so Latin words are counted
    in (written) syllables to keep the length bins roughly aligned.
    """
    if spec.is_indic:
        return sum(1 for cluster in aksharas(text, spec.conjuncts) if not cluster.isspace())
    count, previous = 0, False
    for char in text:
        vowel = char in LATIN_VOWELS + "Y"
        count += vowel and not previous
        previous = vowel
    return max(1, count)


def _positions(text: str, predicate: Callable[[str], bool], skip_first: bool = False) -> List[int]:
    found = [index for index, char in enumerate(text) if predicate(char)]
    return [index for index in found if index > 0] if skip_first else found


def drop_vowel_sign(text: str, spec: ScriptSpec, rng: random.Random) -> Optional[str]:
    # Latin: never drop the first letter, so the word stays anchored.
    spots = _positions(text, lambda c: is_vowel_sign(c, spec), skip_first=not spec.is_indic)
    if not spots:
        return None
    index = rng.choice(spots)
    return text[:index] + text[index + 1:]


def swap_vowel_sign(text: str, spec: ScriptSpec, rng: random.Random) -> Optional[str]:
    spots = _positions(text, lambda c: is_vowel_sign(c, spec))
    if not spots:
        return None
    index = rng.choice(spots)
    options = [c for c in vowel_sign_pool(spec) if c != text[index]]
    return text[:index] + rng.choice(options) + text[index + 1:]


def lookalike(text: str, spec: ScriptSpec, rng: random.Random) -> Optional[str]:
    table: Dict[str, List[str]] = {}
    for a, b in spec.lookalikes:
        table.setdefault(a, []).append(b)
        table.setdefault(b, []).append(a)
    spots = [index for index, char in enumerate(text) if char in table]
    if not spots:
        return None
    index = rng.choice(spots)
    return text[:index] + rng.choice(table[text[index]]) + text[index + 1:]


def drop_akshara(text: str, spec: ScriptSpec, rng: random.Random) -> Optional[str]:
    clusters = aksharas(text, spec.conjuncts)
    spots = [i for i, cluster in enumerate(clusters) if not cluster.isspace()]
    if len(spots) < 2:
        return None
    # Keep the first cluster (Latin: first letter) so the word stays anchored.
    index = rng.choice(spots[1:])
    return "".join(clusters[:index] + clusters[index + 1:])


def pseudo_word(text: str, spec: ScriptSpec, rng: random.Random) -> Optional[str]:
    if spec.is_indic:
        consonants = [chr(spec.block_start + off) for off in CONSONANT_OFFSETS if _assigned(chr(spec.block_start + off))]
        signs = vowel_sign_pool(spec)

        def syllable() -> str:
            return rng.choice(consonants) + (rng.choice(signs) if rng.random() < 0.6 else "")
    else:
        letters = [chr(c) for c in range(ord("A"), ord("Z") + 1)]

        def syllable() -> str:
            return rng.choice(letters)

    words = text.split(" ")
    for _ in range(20):
        size = (lambda w: sum(1 for c in aksharas(w, spec.conjuncts) if not c.isspace())) if spec.is_indic else len
        candidate = " ".join("".join(syllable() for _ in range(max(1, size(word)))) for word in words)
        if candidate != text:
            return candidate
    return None


def script_swap(text: str, spec: ScriptSpec, rng: random.Random) -> Optional[str]:
    if not spec.is_indic or not spec.swap_to:
        return None
    target = SCRIPTS[spec.swap_to]
    out = []
    for char in text:
        offset = _offset(char, spec)
        if offset is None:
            out.append(char)
            continue
        mapped = chr(target.block_start + offset)
        if not _assigned(mapped):
            return None
        out.append(mapped)
    return "".join(out)


TEXT_CORRUPTIONS: Dict[str, Callable[[str, ScriptSpec, random.Random], Optional[str]]] = {
    "drop_vowel_sign": drop_vowel_sign,
    "swap_vowel_sign": swap_vowel_sign,
    "lookalike": lookalike,
    "drop_akshara": drop_akshara,
    "pseudo_word": pseudo_word,
    "script_swap": script_swap,
}


def applicable(error_type: str, spec: ScriptSpec) -> bool:
    if error_type in INDIC_ONLY_ERROR_TYPES:
        return spec.is_indic
    return True


# ---------------------------------------------------------------- pair specs

@dataclass
class GlyphPair:
    pair_id: str
    script: str
    concept: str
    error_type: str
    is_error: bool
    prompt: str
    target_text: str
    corrupt_text: str
    clean_font: str
    corrupt_font: str
    corrupt_shaped: bool
    variant: int
    seed: int
    n_syllables: int
    length_bin: str
    direction: str = "forward"  # "mirrored": the prompt asks for the corrupted string
    base_pair_id: str = ""
    clean_image: str = ""
    corrupt_image: str = ""
    checks: Dict[str, object] = field(default_factory=dict)


def length_bin(n: int) -> str:
    return "1-2" if n <= 2 else ("3-4" if n <= 4 else "5+")


def make_prompt(text: str) -> str:
    # No language name: the mirrored prompt must be built the same way, and
    # a corrupted string has no language.
    return f'A shop sign that reads "{text}".'


def mirror(pair: GlyphPair) -> Optional[GlyphPair]:
    """The same two images, with the prompt asking for the corrupted string."""
    if pair.error_type in MIRRORABLE_EXCLUDED or pair.corrupt_text == pair.target_text:
        return None
    return GlyphPair(
        pair_id=f"{pair.pair_id}~m", script=pair.script, concept=pair.concept, error_type=pair.error_type,
        is_error=pair.is_error, prompt=make_prompt(pair.corrupt_text), target_text=pair.corrupt_text,
        corrupt_text=pair.target_text, clean_font=pair.corrupt_font, corrupt_font=pair.clean_font,
        corrupt_shaped=True, variant=pair.variant, seed=pair.seed, n_syllables=pair.n_syllables,
        length_bin=pair.length_bin, direction="mirrored", base_pair_id=pair.pair_id,
        clean_image=pair.corrupt_image, corrupt_image=pair.clean_image, checks=dict(pair.checks),
    )


def _seed(*parts: object) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest()[:8], 16)


def load_lexicon(path: Path = LEXICON_PATH) -> List[Dict[str, str]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))["concepts"]


def plan_pairs(lexicon: Sequence[Dict[str, str]], scripts: Sequence[str] = STUDY_SCRIPTS,
               error_types: Sequence[str] = ERROR_TYPES, variants: int = 1, base_seed: int = 0) -> List[GlyphPair]:
    """Decide the text of every pair (no rendering). Inapplicable combinations are skipped."""
    pairs: List[GlyphPair] = []
    for entry in lexicon:
        for code in scripts:
            spec = SCRIPTS[code]
            word = unicodedata.normalize("NFC", entry[code])
            for error_type in error_types:
                if not applicable(error_type, spec):
                    continue
                for variant in range(variants):
                    seed = _seed(base_seed, entry["concept"], code, error_type, variant)
                    rng = random.Random(seed)
                    clean_font = spec.fonts[rng.randrange(len(spec.fonts))]
                    corrupt_font = clean_font
                    shaped = True
                    if error_type in TEXT_CORRUPTIONS:
                        corrupt = TEXT_CORRUPTIONS[error_type](word, spec, rng)
                        if error_type == "script_swap" and corrupt is not None:
                            corrupt_font = SCRIPTS[spec.swap_to].fonts[0]
                    elif error_type == "unshaped":
                        corrupt, shaped = word, False
                    elif error_type == "transliterated":
                        corrupt = entry.get(f"{code}_roman")
                        corrupt_font = "NotoSans"
                    elif error_type == "null_font":
                        if len(spec.fonts) < 2:
                            continue
                        corrupt = word
                        corrupt_font = [f for f in spec.fonts if f != clean_font][0]
                    else:
                        raise ValueError(f"Unknown error type {error_type!r}")
                    if corrupt is None or (corrupt == word and error_type not in ("unshaped", "null_font")):
                        continue
                    n = syllable_count(word, spec)
                    pairs.append(GlyphPair(
                        pair_id=f"{entry['concept']}-{code}-{error_type}-v{variant}",
                        script=code, concept=entry["concept"], error_type=error_type,
                        is_error=error_type != "null_font", prompt=make_prompt(word),
                        target_text=word, corrupt_text=corrupt, clean_font=clean_font,
                        corrupt_font=corrupt_font, corrupt_shaped=shaped, variant=variant, seed=seed,
                        n_syllables=n, length_bin=length_bin(n), base_pair_id=f"{entry['concept']}-{code}-{error_type}-v{variant}",
                    ))
    return pairs


# ---------------------------------------------------------------- rendering

PALETTES = [  # (board, text, border)
    ((196, 32, 38), (255, 255, 255), (250, 210, 60)),
    ((250, 204, 21), (20, 20, 20), (20, 20, 20)),
    ((22, 78, 160), (255, 255, 255), (230, 230, 230)),
    ((245, 245, 240), (25, 60, 140), (25, 60, 140)),
    ((18, 110, 60), (255, 255, 255), (255, 255, 255)),
    ((30, 30, 30), (255, 215, 0), (255, 215, 0)),
]


def font_path(name: str, font_dir: Path = FONT_DIR) -> Path:
    return Path(font_dir) / f"{name}.ttf"


def fetch_fonts(font_dir: Path = FONT_DIR, names: Optional[Sequence[str]] = None) -> List[Path]:
    """Download the Noto fonts (SIL OFL) into ``font_dir``; existing files are kept."""
    import urllib.request

    font_dir = Path(font_dir)
    font_dir.mkdir(parents=True, exist_ok=True)
    fetched = []
    for name in names or FONT_FILES:
        target = font_path(name, font_dir)
        if not target.exists():
            with urllib.request.urlopen(f"{FONT_BASE_URL}/{FONT_FILES[name]}", timeout=60) as response:
                target.write_bytes(response.read())
        fetched.append(target)
    return fetched


@functools.lru_cache(maxsize=None)
def _cmap(path: str) -> frozenset:
    from fontTools.ttLib import TTFont

    with TTFont(path, lazy=True) as font:
        return frozenset(font.getBestCmap())


def missing_codepoints(text: str, path: Path) -> List[str]:
    cmap = _cmap(str(path))
    return [f"U+{ord(c):04X}" for c in set(text) if not c.isspace() and c not in JOINERS and ord(c) not in cmap]


def perspective_coeffs(output_pts: Sequence[Tuple[float, float]], input_pts: Sequence[Tuple[float, float]]) -> List[float]:
    """Coefficients for ``Image.transform(PERSPECTIVE)``: maps output points to input points."""
    import numpy as np

    rows, rhs = [], []
    for (xo, yo), (xi, yi) in zip(output_pts, input_pts):
        rows.append([xo, yo, 1, 0, 0, 0, -xi * xo, -xi * yo])
        rows.append([0, 0, 0, xo, yo, 1, -yi * xo, -yi * yo])
        rhs += [xi, yi]
    return [float(v) for v in np.linalg.solve(np.array(rows, dtype=float), np.array(rhs, dtype=float))]


def apply_perspective(coeffs: Sequence[float], point: Tuple[float, float]) -> Tuple[float, float]:
    a, b, c, d, e, f, g, h = coeffs
    x, y = point
    w = g * x + h * y + 1
    return (a * x + b * y + c) / w, (d * x + e * y + f) / w


class PairRenderer:
    def __init__(self, font_dir: Path = FONT_DIR, size: int = 512, background_dir: Optional[Path] = None,
                 perspective: bool = True):
        from PIL import features

        if not features.check("raqm"):
            raise RuntimeError("Pillow was built without libraqm; complex-script shaping is unavailable.")
        self.font_dir = Path(font_dir)
        self.size = size
        self.perspective = perspective
        self.backgrounds = sorted(p for p in Path(background_dir).glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}) if background_dir else []

    @functools.lru_cache(maxsize=256)
    def _font(self, name: str, px: int, shaped: bool):
        from PIL import ImageFont

        engine = ImageFont.Layout.RAQM if shaped else ImageFont.Layout.BASIC
        return ImageFont.truetype(str(font_path(name, self.font_dir)), px, layout_engine=engine)

    def _bbox(self, text: str, name: str, px: int, shaped: bool, lang: Optional[str]):
        from PIL import Image, ImageDraw

        draw = ImageDraw.Draw(Image.new("L", (8, 8)))
        return draw.textbbox((0, 0), text, font=self._font(name, px, shaped), anchor="mm", language=lang if shaped else None)

    def _background(self, rng: random.Random):
        import numpy as np
        from PIL import Image

        if self.backgrounds:
            image = Image.open(rng.choice(self.backgrounds)).convert("RGB")
            scale = self.size / min(image.size)
            image = image.resize((max(self.size, round(image.width * scale)), max(self.size, round(image.height * scale))))
            left, top = rng.randrange(image.width - self.size + 1), rng.randrange(image.height - self.size + 1)
            return image.crop((left, top, left + self.size, top + self.size))
        top = np.array([rng.randrange(120, 230) for _ in range(3)], dtype=float)
        bottom = top * rng.uniform(0.45, 0.8)
        ramp = np.linspace(0, 1, self.size)[:, None, None]
        array = top * (1 - ramp) + bottom * ramp
        array = np.broadcast_to(array, (self.size, self.size, 3)).copy()
        return Image.fromarray(array.clip(0, 255).astype("uint8"))

    def render(self, pair: GlyphPair, out_dir: Path, image_format: str = "png") -> GlyphPair:
        import numpy as np
        from PIL import Image, ImageDraw, ImageFilter

        rng = random.Random(pair.seed)
        palette = PALETTES[rng.randrange(len(PALETTES))]
        background = self._background(rng)
        board_w = int(self.size * rng.uniform(0.72, 0.88))
        board_h = int(self.size * rng.uniform(0.26, 0.36))
        left = (self.size - board_w) // 2 + rng.randint(-15, 15)
        top = int(self.size * rng.uniform(0.18, 0.5))
        noise_seed = rng.randrange(2 ** 31)
        blur = rng.uniform(0.0, 0.8)
        # Mild perspective: push each image corner outwards by up to 7%, so the
        # warped image never shows empty borders. Same warp on both sides.
        outward = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
        corners = [(0, 0), (self.size, 0), (self.size, self.size), (0, self.size)]
        warped = [(x + dx * rng.uniform(0, 0.07) * self.size, y + dy * rng.uniform(0, 0.07) * self.size)
                  for (x, y), (dx, dy) in zip(corners, outward)] if self.perspective else corners

        clean_lang = SCRIPTS[pair.script].bcp47
        corrupt_lang = SCRIPTS[pair.script].bcp47 if pair.error_type not in ("script_swap", "transliterated") else (
            SCRIPTS[SCRIPTS[pair.script].swap_to].bcp47 if pair.error_type == "script_swap" else "en")
        sides = [(pair.target_text, pair.clean_font, True, clean_lang), (pair.corrupt_text, pair.corrupt_font, pair.corrupt_shaped, corrupt_lang)]

        # One font size for both sides, so text size never gives the answer away.
        # Text extent scales almost linearly with size: measure once, then step down.
        ref = 100
        boxes = [self._bbox(t, f, ref, s, l) for t, f, s, l in sides]
        ratio = min(min(board_w * 0.86 / max(1, b[2] - b[0]), board_h * 0.8 / max(1, b[3] - b[1])) for b in boxes)
        px = max(10, min(int(board_h * 0.62), int(ref * ratio)))
        while px > 10 and not all(
            b[2] - b[0] <= board_w * 0.86 and b[3] - b[1] <= board_h * 0.8
            for b in (self._bbox(t, f, px, s, l) for t, f, s, l in sides)
        ):
            px -= 2

        paths = []
        for role, (text, font_name, shaped, lang) in zip(("clean", "corrupt"), sides):
            image = background.copy()
            draw = ImageDraw.Draw(image)
            draw.rectangle((left - 6, top - 6, left + board_w + 6, top + board_h + 6), fill=palette[2])
            draw.rectangle((left, top, left + board_w, top + board_h), fill=palette[0])
            draw.text((left + board_w / 2, top + board_h / 2), text, font=self._font(font_name, px, shaped),
                      fill=palette[1], anchor="mm", language=lang if shaped else None)
            if warped != corners:
                image = image.transform(image.size, Image.Transform.PERSPECTIVE,
                                        perspective_coeffs(warped, corners), Image.Resampling.BICUBIC)
            if blur > 0.05:
                image = image.filter(ImageFilter.GaussianBlur(blur))
            noise = np.random.default_rng(noise_seed).normal(0, 4.0, (self.size, self.size, 3))
            image = Image.fromarray((np.asarray(image, dtype=float) + noise).clip(0, 255).astype("uint8"))
            path = Path(out_dir) / f"{pair.pair_id}_{role}.{image_format}"
            path.parent.mkdir(parents=True, exist_ok=True)
            image.save(path, quality=93) if image_format in ("jpg", "jpeg") else image.save(path)
            paths.append(path)

        a = np.asarray(Image.open(paths[0]), dtype=float)
        b = np.asarray(Image.open(paths[1]), dtype=float)
        pair.clean_image, pair.corrupt_image = paths[0].name, paths[1].name
        board = [(left - 6, top - 6), (left + board_w + 6, top - 6), (left + board_w + 6, top + board_h + 6), (left - 6, top + board_h + 6)]
        moved = [apply_perspective(perspective_coeffs(corners, warped), pt) for pt in board] if warped != corners else board
        pair.checks = {
            "font_px": px,
            "text_box": [max(0, round(min(x for x, _ in moved))), max(0, round(min(y for _, y in moved))),
                         min(self.size, round(max(x for x, _ in moved))), min(self.size, round(max(y for _, y in moved)))],
            "pixel_diff": round(float(np.abs(a - b).mean()), 4),
            "missing_glyphs_clean": missing_codepoints(pair.target_text, font_path(pair.clean_font, self.font_dir)),
            "missing_glyphs_corrupt": missing_codepoints(pair.corrupt_text, font_path(pair.corrupt_font, self.font_dir)),
        }
        return pair


def pair_is_valid(pair: GlyphPair, min_pixel_diff: float = 0.05) -> bool:
    checks = pair.checks
    if checks.get("missing_glyphs_clean") or checks.get("missing_glyphs_corrupt"):
        return False
    # An "unshaped" word that happens to render identically is not an error.
    return checks.get("pixel_diff", 0.0) >= min_pixel_diff


# ---------------------------------------------------------------- outputs

def candidate_rows(pairs: Sequence[GlyphPair], image_subdir: str = "images") -> List[Dict]:
    """Manifest rows in the ``ImageCandidate`` format used by ``scripts/score_images.py``."""
    rows = []
    for pair in pairs:
        for role, image, text in (("clean", pair.clean_image, pair.target_text), ("corrupt", pair.corrupt_image, pair.corrupt_text)):
            rows.append({
                "candidate_id": f"{pair.pair_id}:{role}",
                "prompt_id": pair.pair_id,
                "prompt": pair.prompt,
                "image_path": f"{image_subdir}/{image}",
                "generator": "glyph_pairs_renderer",
                "category": "text_rendering",
                "metadata": {
                    "pair_id": pair.pair_id, "role": role, "script": pair.script, "direction": pair.direction,
                    "error_type": pair.error_type, "expected_text": [pair.target_text],
                    "rendered_text": text, "text_box": pair.checks.get("text_box"),
                },
            })
    return rows


def contact_sheet(pairs: Sequence[GlyphPair], image_dir: Path, out_path: Path, thumb: int = 192, limit: int = 24) -> Path:
    from PIL import Image, ImageDraw, ImageFont

    chosen = list(pairs)[:limit]
    label_h = 18
    sheet = Image.new("RGB", (thumb * 2 + 12, (thumb + label_h) * len(chosen)), "white")
    draw = ImageDraw.Draw(sheet)
    label_font = ImageFont.load_default()
    for row, pair in enumerate(chosen):
        y = row * (thumb + label_h)
        draw.text((4, y + 2), f"{pair.pair_id}  (left: clean, right: corrupt)", fill="black", font=label_font)
        for col, name in enumerate((pair.clean_image, pair.corrupt_image)):
            image = Image.open(Path(image_dir) / name).convert("RGB").resize((thumb, thumb))
            sheet.paste(image, (col * (thumb + 12), y + label_h))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    return out_path


def build(out_dir: Path, scripts: Sequence[str] = STUDY_SCRIPTS, error_types: Sequence[str] = ERROR_TYPES,
          variants: int = 1, seed: int = 0, font_dir: Path = FONT_DIR, background_dir: Optional[Path] = None,
          image_format: str = "png", lexicon_path: Path = LEXICON_PATH) -> Dict:
    out_dir = Path(out_dir)
    renderer = PairRenderer(font_dir=font_dir, background_dir=background_dir)
    planned = plan_pairs(load_lexicon(lexicon_path), scripts, error_types, variants, seed)
    kept, dropped = [], []
    for pair in planned:
        renderer.render(pair, out_dir / "images", image_format)
        (kept if pair_is_valid(pair) else dropped).append(pair)
    for pair in dropped:  # do not leave invalid images next to valid ones
        for name in (pair.clean_image, pair.corrupt_image):
            (out_dir / "images" / name).unlink(missing_ok=True)
    kept = kept + [m for m in (mirror(p) for p in kept) if m is not None]
    (out_dir / "pairs.jsonl").write_text("".join(json.dumps(asdict(p), ensure_ascii=False) + "\n" for p in kept), encoding="utf-8")
    (out_dir / "candidates.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in candidate_rows(kept)), encoding="utf-8")
    (out_dir / "dropped.jsonl").write_text("".join(json.dumps(asdict(p), ensure_ascii=False) + "\n" for p in dropped), encoding="utf-8")
    counts: Dict[str, Dict[str, int]] = {}
    for pair in kept:
        if pair.direction == "forward":
            counts.setdefault(pair.script, {}).setdefault(pair.error_type, 0)
            counts[pair.script][pair.error_type] += 1
    forward = sum(1 for p in kept if p.direction == "forward")
    summary = {"planned": len(planned), "kept_forward": forward, "mirrored": len(kept) - forward,
               "dropped": len(dropped), "images": 2 * forward, "counts": counts}
    (out_dir / "build_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Render glyph minimal pairs (clean vs one controlled text error).")
    sub = parser.add_subparsers(dest="command", required=True)
    fonts = sub.add_parser("fetch-fonts", help="Download the Noto fonts (SIL OFL) used by the renderer.")
    fonts.add_argument("--font-dir", default=str(FONT_DIR))
    run = sub.add_parser("build", help="Plan, render and validate pairs.")
    run.add_argument("--out-dir", required=True)
    run.add_argument("--scripts", nargs="+", default=list(STUDY_SCRIPTS))
    run.add_argument("--error-types", nargs="+", default=list(ERROR_TYPES))
    run.add_argument("--variants", type=int, default=1, help="Renderings per (word, script, error type).")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--font-dir", default=str(FONT_DIR))
    run.add_argument("--background-dir", help="Optional folder of licensed photos to use as backgrounds.")
    run.add_argument("--format", default="png", choices=["png", "jpg"])
    run.add_argument("--contact-sheet", type=int, default=24, help="Pairs to include in contact_sheet.png (0 to skip).")
    args = parser.parse_args()
    if args.command == "fetch-fonts":
        for path in fetch_fonts(Path(args.font_dir)):
            print(f"[fonts] {path}")
        return
    summary = build(Path(args.out_dir), args.scripts, args.error_types, args.variants, args.seed,
                    Path(args.font_dir), Path(args.background_dir) if args.background_dir else None, args.format)
    print(json.dumps(summary, indent=2))
    if args.contact_sheet:
        pairs = [GlyphPair(**json.loads(line)) for line in (Path(args.out_dir) / "pairs.jsonl").read_text(encoding="utf-8").splitlines()]
        pairs = [p for p in pairs if p.direction == "forward"]
        rng = random.Random(args.seed)
        sample = rng.sample(pairs, min(args.contact_sheet, len(pairs)))
        print(f"[sheet] {contact_sheet(sample, Path(args.out_dir) / 'images', Path(args.out_dir) / 'contact_sheet.png', limit=args.contact_sheet)}")


if __name__ == "__main__":
    main()
