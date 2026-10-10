import random
import tempfile
import unittest
from pathlib import Path

from src.image_verification import glyph_eval
from src.image_verification.glyph_pairs import (
    FONT_DIR, SCRIPTS, SHARED_ERROR_TYPES, aksharas, candidate_rows, drop_akshara, drop_vowel_sign,
    font_path, load_lexicon, lookalike, mirror, plan_pairs, pseudo_word, script_swap, swap_vowel_sign,
    syllable_count,
)


def _fonts_available(names):
    try:
        from PIL import features
    except ImportError:
        return False
    return features.check("raqm") and all(font_path(n).exists() for n in names)


class TextCorruptionTest(unittest.TestCase):
    def test_akshara_segmentation(self):
        self.assertEqual(aksharas("विद्यालय"), ["वि", "द्या", "ल", "य"])
        self.assertEqual(aksharas("ಬ್ಯಾಂಕ್"), ["ಬ್ಯಾಂ", "ಕ್"])
        # Tamil does not form conjuncts: the pulli closes a syllable.
        self.assertEqual(aksharas("தண்ணீர்", join_virama=False), ["த", "ண்", "ணீ", "ர்"])
        self.assertEqual(syllable_count("बस स्टॉप", SCRIPTS["deva"]), 4)  # ब स स्टॉ प
        self.assertEqual(syllable_count("HOSPITAL", SCRIPTS["latn"]), 3)

    def test_corruptions_change_text_deterministically(self):
        cases = [("deva", "अस्पताल"), ("knda", "ಆಸ್ಪತ್ರೆ"), ("taml", "மருத்துவமனை"), ("latn", "HOSPITAL")]
        for code, word in cases:
            spec = SCRIPTS[code]
            for fn in (drop_vowel_sign, swap_vowel_sign, drop_akshara, pseudo_word):
                first = fn(word, spec, random.Random(7))
                self.assertIsNotNone(first, (code, fn.__name__))
                self.assertNotEqual(first, word, (code, fn.__name__))
                self.assertEqual(first, fn(word, spec, random.Random(7)))

    def test_specific_corruptions(self):
        self.assertEqual(drop_vowel_sign("दूध", SCRIPTS["deva"], random.Random(0)), "दध")
        self.assertEqual(lookalike("பால்", SCRIPTS["taml"], random.Random(0)), "பாள்")
        self.assertEqual(script_swap("दूध", SCRIPTS["deva"], random.Random(0)), "દૂધ")
        self.assertEqual(script_swap("ಹಾಲು", SCRIPTS["knda"], random.Random(0)), "హాలు")
        self.assertIsNone(script_swap("MILK", SCRIPTS["latn"], random.Random(0)))
        # The first Latin letter is never dropped.
        for seed in range(20):
            self.assertTrue(drop_vowel_sign("APPLE", SCRIPTS["latn"], random.Random(seed)).startswith("A"))

    def test_plan_and_mirror(self):
        pairs = plan_pairs(load_lexicon())
        self.assertGreater(len(pairs), 500)
        types = {(p.script, p.error_type) for p in pairs}
        self.assertNotIn(("latn", "unshaped"), types)
        self.assertIn(("knda", "unshaped"), types)
        null = [p for p in pairs if p.error_type == "null_font"]
        self.assertTrue(null and not any(p.is_error for p in null))
        self.assertTrue(all(p.clean_font != p.corrupt_font for p in null))
        self.assertTrue(all(p.corrupt_text != p.target_text for p in pairs if p.error_type in SHARED_ERROR_TYPES))
        pair = next(p for p in pairs if p.error_type == "drop_vowel_sign")
        pair.clean_image, pair.corrupt_image = "a.png", "b.png"
        flipped = mirror(pair)
        self.assertEqual((flipped.target_text, flipped.corrupt_text), (pair.corrupt_text, pair.target_text))
        self.assertEqual((flipped.clean_image, flipped.corrupt_image), ("b.png", "a.png"))
        self.assertIn(pair.corrupt_text, flipped.prompt)
        self.assertIsNone(mirror(next(p for p in pairs if p.error_type == "unshaped")))
        rows = candidate_rows([pair, flipped])
        self.assertEqual(len({r["candidate_id"] for r in rows}), 4)


@unittest.skipUnless(_fonts_available(["NotoSansKannada", "NotoSans"]), "fonts or libraqm missing; run build_glyph_pairs.py fetch-fonts")
class RenderTest(unittest.TestCase):
    def test_render_and_validate(self):
        from src.image_verification.glyph_pairs import PairRenderer, pair_is_valid

        pairs = [p for p in plan_pairs(load_lexicon(), scripts=["knda", "latn"], error_types=["unshaped", "drop_akshara", "null_font"])
                 if p.concept == "hospital"]
        renderer = PairRenderer(FONT_DIR)
        with tempfile.TemporaryDirectory() as tmp:
            for pair in pairs:
                renderer.render(pair, Path(tmp))
                self.assertTrue((Path(tmp) / pair.clean_image).exists())
                self.assertTrue(pair_is_valid(pair), pair.checks)


def _fake_pair(pid, script, concept, error_type, direction, target, corrupt):
    return {"pair_id": pid, "script": script, "concept": concept, "error_type": error_type,
            "is_error": error_type != "null_font", "direction": direction, "length_bin": "1-2",
            "target_text": target, "corrupt_text": corrupt}


class GlyphEvalTest(unittest.TestCase):
    def setUp(self):
        self.pairs = []
        for i in range(12):
            for script in ("latn", "deva"):
                base = f"c{i}-{script}"
                self.pairs.append(_fake_pair(base, script, f"c{i}", "drop_akshara", "forward", "LONGWORD", "SHORT"))
                self.pairs.append(_fake_pair(base + "~m", script, f"c{i}", "drop_akshara", "mirrored", "SHORT", "LONGWORD"))

    def _scores(self, fn):
        scores = {}
        for pair in self.pairs:
            scores[f"{pair['pair_id']}:clean"] = fn(pair, pair["target_text"])
            scores[f"{pair['pair_id']}:corrupt"] = fn(pair, pair["corrupt_text"])
        return scores

    def test_reader_vs_ink(self):
        # A perfect reader scores the image matching the prompt highest; it fails on Devanagari half the time.
        def reader(pair, shown):
            if pair["script"] == "deva" and int(pair["concept"][1:]) % 2:
                return 0.0 if shown == pair["target_text"] else 1.0
            return 1.0 if shown == pair["target_text"] else 0.0

        # An ink counter ignores the prompt and prefers the longer text.
        def ink(pair, shown):
            return float(len(shown))

        result = glyph_eval.analyse(self.pairs, {"reader": self._scores(reader), "ink": self._scores(ink)}, samples=200)
        reader_entry = result["evaluators"]["reader"]
        self.assertEqual(reader_entry["overall"]["latn"]["reading"]["accuracy"], 1.0)
        self.assertEqual(reader_entry["overall"]["deva"]["reading"]["accuracy"], 0.5)
        self.assertAlmostEqual(reader_entry["gap_vs_latin"]["deva"]["estimate"], 0.5)
        ink_entry = result["evaluators"]["ink"]
        self.assertEqual(ink_entry["overall"]["latn"]["forward"]["accuracy"], 1.0)
        self.assertEqual(ink_entry["overall"]["latn"]["reading"]["accuracy"], 0.5)
        self.assertEqual(ink_entry["overall"]["latn"]["prior"], 1.0)
        self.assertIn("Reading accuracy", glyph_eval.render_markdown(result))


if __name__ == "__main__":
    unittest.main()
