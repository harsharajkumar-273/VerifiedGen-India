import tempfile, unittest
from pathlib import Path
from src.image_verification.prompt_bank import build_prompt_bank, save_prompt_bank

class SmokeTest(unittest.TestCase):
    def test_prompt_bank(self):
        prompts=build_prompt_bank(25,"india")
        self.assertEqual(len(prompts),25)
        self.assertTrue(any(p.metadata.get("expected_text") for p in prompts))
    def test_imports(self):
        from src.image_verification.verifier import ThresholdConfig
        self.assertGreater(ThresholdConfig().pass_threshold, 0)
    def test_safety_is_hard_block(self):
        from src.image_verification.safety import evaluate_safety
        result = evaluate_safety(["nsfw"])
        self.assertFalse(result.safe)
        self.assertEqual(result.score, 0.0)

if __name__ == "__main__": unittest.main()
