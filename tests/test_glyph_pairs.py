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


class ToolingTest(unittest.TestCase):
    def test_perspective_roundtrip(self):
        from src.image_verification.glyph_pairs import apply_perspective, perspective_coeffs

        src = [(0, 0), (512, 0), (512, 512), (0, 512)]
        dst = [(-10, -20), (530, -5), (520, 540), (-30, 515)]
        forward = perspective_coeffs(src, dst)
        back = perspective_coeffs(dst, src)
        for point in [(100, 200), (256, 256), (400, 50)]:
            x, y = apply_perspective(back, apply_perspective(forward, point))
            self.assertAlmostEqual(x, point[0], places=6)
            self.assertAlmostEqual(y, point[1], places=6)

    def test_option_parsing_and_language(self):
        from src.image_verification.evaluators import parse_options
        from src.image_verification.model_evaluators import tesseract_lang

        options = parse_options(["vlm_judge.model_name=Qwen/X", "vlm_judge.ocr_grounded=true", "ocr_tesseract.crop=false",
                                 "probe.rank=16"])
        self.assertEqual(options["vlm_judge"], {"model_name": "Qwen/X", "ocr_grounded": True})
        self.assertEqual(options["ocr_tesseract"]["crop"], False)
        self.assertEqual(options["probe"]["rank"], 16)
        self.assertEqual(tesseract_lang("ಹಾಲು"), "kan")
        self.assertEqual(tesseract_lang("दूध"), "hin")
        self.assertEqual(tesseract_lang("பால்"), "tam")
        self.assertEqual(tesseract_lang("દૂધ"), "guj")
        self.assertEqual(tesseract_lang("DOODH"), "eng")

    def test_exclusions_drop_mirrors(self):
        pairs = [_fake_pair("a", "deva", "c1", "lookalike", "forward", "x", "y"),
                 {**_fake_pair("a~m", "deva", "c1", "lookalike", "mirrored", "y", "x"), "base_pair_id": "a"},
                 _fake_pair("b", "knda", "c1", "lookalike", "forward", "x", "y"),
                 _fake_pair("c", "knda", "c2", "drop_akshara", "forward", "x", "y")]
        kept = glyph_eval.apply_exclusions(pairs, {"pairs": ["a"], "error_types": {"knda": ["lookalike"]}})
        self.assertEqual([p["pair_id"] for p in kept], ["c"])

    def test_probe_learns_to_read(self):
        import numpy as np

        from src.image_verification.glyph_probe import cross_fit_scores

        rng = np.random.default_rng(0)
        dim, words = 16, {}
        ids, cand_img, cand_txt, images, texts, pairs = [], [], [], [], [], []

        def word_vec(word):
            if word not in words:
                words[word] = rng.normal(size=dim)
            return words[word]

        def unit(v):
            return v / np.linalg.norm(v)

        for c in range(12):
            for script in ("latn", "deva"):
                clean_w, bad_w = f"w{c}{script}", f"w{c}{script}x"
                img_clean = unit(word_vec(clean_w) + rng.normal(scale=0.3, size=dim))
                img_bad = unit(word_vec(bad_w) + rng.normal(scale=0.3, size=dim))
                for direction, (want, good, bad) in (("forward", (clean_w, img_clean, img_bad)), ("mirrored", (bad_w, img_bad, img_clean))):
                    pid = f"{c}-{script}-{direction}"
                    texts.append(unit(word_vec(want)))
                    for role, img in (("clean", good), ("corrupt", bad)):
                        images.append(img)
                        ids.append(f"{pid}:{role}")
                        cand_img.append(len(images) - 1)
                        cand_txt.append(len(texts) - 1)
                    pairs.append({"pair_id": pid, "concept": f"c{c}", "script": script, "is_error": True,
                                  "error_type": "lookalike", "direction": direction, "length_bin": "1-2"})
        features = {"image_emb": np.array(images), "text_emb": np.array(texts), "candidate_ids": np.array(ids),
                    "candidate_image": np.array(cand_img), "candidate_text": np.array(cand_txt)}
        scores = cross_fit_scores(pairs, features, folds=4, rank=8, epochs=300)
        result = glyph_eval.analyse(pairs, {"probe": scores}, samples=100)
        self.assertGreater(result["evaluators"]["probe"]["overall"]["latn"]["reading"]["accuracy"], 0.8)
        held = cross_fit_scores(pairs, features, folds=4, rank=8, epochs=300, holdout_script="deva")
        self.assertTrue(all(cid.split("-")[1] == "deva" for cid in held))

    def test_validation_aggregate(self):
        import csv
        import json as _json

        from src.image_verification.glyph_validation import FIELDS, aggregate

        key = [{"item_id": "i1", "pair_id": "p1", "role": "clean", "script": "deva", "error_type": "lookalike"},
               {"item_id": "i2", "pair_id": "p1", "role": "corrupt", "script": "deva", "error_type": "lookalike"},
               {"item_id": "i3", "pair_id": "p2", "role": "clean", "script": "deva", "error_type": "lookalike"},
               {"item_id": "i4", "pair_id": "p2", "role": "corrupt", "script": "deva", "error_type": "lookalike"}]
        answers = {"i1": "yes", "i2": "yes", "i3": "no", "i4": "no"}  # i2: readers missed the error; i3: clean judged wrong
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "key.json").write_text(_json.dumps(key))
            with open(Path(tmp) / "sheet.csv", "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=FIELDS)
                writer.writeheader()
                for item, verdict in answers.items():
                    writer.writerow({"item_id": item, "matches_request": verdict, "legible": "yes"})
            result = aggregate([Path(tmp) / "sheet.csv"], Path(tmp) / "key.json")
        self.assertEqual(result["pairs"], ["p2"])
        self.assertEqual(result["error_types"], {"deva": ["lookalike"]})
        self.assertEqual(result["detection_rate"]["deva/lookalike"], 0.5)
