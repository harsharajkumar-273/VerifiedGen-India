import tempfile
import unittest
from pathlib import Path

try:
    import numpy  # noqa: F401
    from PIL import Image
except ImportError:  # pragma: no cover - image tests need Pillow and numpy
    Image = None


@unittest.skipIf(Image is None, "Pillow and numpy are required for image tests")
class ImageVerifierTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.blank = root / "blank.png"
        Image.new("RGB", (512, 512), (128, 128, 128)).save(self.blank)
        self.red = root / "red.png"
        image = Image.new("RGB", (512, 512), (220, 40, 40))
        for x in range(0, 512, 16):
            for y in range(512):
                image.putpixel((x, y), (20, 20, 20))
        image.save(self.red)

    def tearDown(self):
        self.tmp.cleanup()

    def test_features(self):
        from src.image_verification.image_features import color_coverage, compute_image_features

        _, blank = compute_image_features(self.blank)
        self.assertTrue(blank.is_blank)
        image, red = compute_image_features(self.red)
        self.assertFalse(red.is_blank)
        self.assertGreater(red.edge_density, 0.05)
        self.assertGreater(color_coverage(image, (220, 40, 40)), 0.8)

    def test_verifier_and_reward(self):
        from src.image_verification.evaluators import ImageStatsEvaluator, VerifierEvaluator
        from src.image_verification.reward import reward_from_report
        from src.image_verification.schema import ImageCandidate
        from src.image_verification.verifier import verify_candidate

        blank = ImageCandidate("b", "p", "a red wall", str(self.blank))
        red = ImageCandidate("r", "p", "a red wall", str(self.red))
        self.assertEqual(verify_candidate(blank).decision, "fail")
        red_report = verify_candidate(red)
        self.assertGreater(red_report.overall_score, verify_candidate(blank).overall_score)
        # No text/cultural axes: reward is renormalised rather than capped at 0.75.
        self.assertGreater(reward_from_report(red_report), 0.75)
        self.assertGreater(VerifierEvaluator().score(red), VerifierEvaluator().score(blank))
        self.assertGreater(ImageStatsEvaluator().score(red), ImageStatsEvaluator().score(blank))

    def test_indic_text_matching_is_normalised(self):
        from src.image_verification.schema import ImageCandidate
        from src.image_verification.verifier import verify_candidate

        # The same Bengali letter encoded two ways; NFC must make them compare equal.
        decomposed = "\u09af\u09bc"  # BENGALI LETTER YA + SIGN NUKTA
        precomposed = "\u09df"  # BENGALI LETTER YYA
        candidate = ImageCandidate("t", "p", "a sign", str(self.red),
                                   metadata={"expected_text": [precomposed], "ocr_text": decomposed})
        text_axis = next(a for a in verify_candidate(candidate).axis_scores if a.name == "text_rendering")
        self.assertEqual(text_axis.score, 1.0)


if __name__ == "__main__":
    unittest.main()
