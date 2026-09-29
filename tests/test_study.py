import json
import random
import tempfile
import unittest
from pathlib import Path

from src.image_verification import stats
from src.image_verification.human_labels import (
    aggregate_labels, export_blinded_tasks, load_label_rows, verdict_agreement,
)
from src.image_verification.matched_pairs import build_matched_bank
from src.image_verification.prompt_parser import extract_prompt_requirements
from src.image_verification.schema import ImageCandidate
from src.image_verification.selection import capture
from src.image_verification.study import StudyConfig, build_rows, render_markdown, run_study


class StatsTest(unittest.TestCase):
    def test_auroc(self):
        self.assertEqual(stats.auroc([0.1, 0.2, 0.8, 0.9], [False, False, True, True]), 1.0)
        self.assertEqual(stats.auroc([0.9, 0.8, 0.2, 0.1], [False, False, True, True]), 0.0)
        self.assertEqual(stats.auroc([0.5, 0.5, 0.5, 0.5], [False, True, False, True]), 0.5)
        self.assertAlmostEqual(stats.auroc([0.1, 0.4, 0.35, 0.8], [False, False, True, True]), 0.75)
        self.assertIsNone(stats.auroc([0.1, 0.2], [True, True]))

    def test_rank_and_spearman(self):
        self.assertEqual(stats.rankdata([10, 20, 20, 30]), [1.0, 2.5, 2.5, 4.0])
        self.assertAlmostEqual(stats.spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)
        self.assertAlmostEqual(stats.spearman([1, 2, 3, 4], [4, 3, 2, 1]), -1.0)

    def test_threshold_and_false_accepts(self):
        scores = [0.9, 0.8, 0.7, 0.6, 0.5, 0.85, 0.2]
        labels = [True, True, True, True, True, False, False]
        cut = stats.threshold_at_true_accept_rate(scores, labels, 0.8)
        self.assertEqual(cut, 0.6)  # keeps 4 of 5 good images
        self.assertEqual(stats.false_accept_rate(scores, labels, cut), 0.5)
        self.assertEqual(stats.false_reject_rate(scores, labels, cut), 0.2)

    def test_krippendorff(self):
        self.assertEqual(stats.krippendorff_alpha_nominal([["a", "a"], ["b", "b"], ["a", "a"]]), 1.0)
        # Two coders, 4 items, one disagreement. Hand calculation: Do = 2/8, De = 16/28.
        alpha = stats.krippendorff_alpha_nominal([["a", "a"], ["a", "b"], ["b", "b"], ["a", "a"]])
        self.assertAlmostEqual(alpha, 1 - (2 / 8) / (2 * 5 * 3 / (8 * 7)), places=6)

    def test_holm(self):
        adjusted = stats.holm_adjust({"a": 0.01, "b": 0.04, "c": 0.03, "d": None})
        self.assertAlmostEqual(adjusted["a"], 0.03)
        self.assertAlmostEqual(adjusted["c"], 0.06)
        self.assertAlmostEqual(adjusted["b"], 0.06)
        self.assertIsNone(adjusted["d"])

    def test_cluster_bootstrap_is_deterministic(self):
        clusters = [i // 2 for i in range(20)]
        values = [float(i) for i in range(20)]
        stat = lambda idx: sum(values[i] for i in idx) / len(idx)
        a = stats.cluster_bootstrap(clusters, stat, samples=200, seed=3)
        b = stats.cluster_bootstrap(clusters, stat, samples=200, seed=3)
        self.assertEqual(a, b)
        self.assertLess(a["ci_low"], a["estimate"])
        self.assertGreater(a["ci_high"], a["estimate"])


class MatchedBankTest(unittest.TestCase):
    def setUp(self):
        self.bank = build_matched_bank()

    def test_counts_and_unique_ids(self):
        conditions = [s.metadata["condition"] for s in self.bank]
        self.assertEqual(conditions.count("india_indic"), 45)
        self.assertEqual(conditions.count("india_latin"), 45)
        self.assertEqual(conditions.count("global_latin"), 45)
        self.assertEqual(conditions.count("india"), 20)
        self.assertEqual(conditions.count("global"), 20)
        self.assertEqual(len({s.prompt_id for s in self.bank}), len(self.bank))

    def test_pairs_are_matched(self):
        pairs = {}
        for spec in self.bank:
            if spec.metadata["pair_id"]:
                pairs.setdefault(spec.metadata["pair_id"], []).append(spec)
        for pair_id, members in pairs.items():
            self.assertEqual(len({m.category for m in members}), 1, pair_id)
            counts = [sum(m.expected_objects.values()) for m in members]
            self.assertEqual(len(set(counts)), 1, f"{pair_id}: object requirements differ")
            if pair_id.startswith("script_"):
                self.assertEqual({m.metadata["condition"] for m in members}, {"india_indic", "india_latin", "global_latin"})
                latin = [m for m in members if m.metadata["script"] == "Latin"]
                self.assertEqual(len({tuple(m.metadata["expected_text"]) for m in latin}), 1)
            else:
                self.assertEqual({m.metadata["condition"] for m in members}, {"india", "global"})


class ParserTest(unittest.TestCase):
    def test_counts_skip_adjectives(self):
        self.assertEqual(extract_prompt_requirements("two red apples on a plate").objects, {"apple": 2})
        self.assertEqual(extract_prompt_requirements("three small boxes").objects, {"box": 3})
        self.assertEqual(extract_prompt_requirements("a dairy farm with two cows").objects, {"cow": 2})


class HumanLabelsTest(unittest.TestCase):
    def test_blinded_export_and_aggregation(self):
        candidates = [ImageCandidate(f"c{i}", "p", "prompt", f"/img/{i}.png", generator="sdxl", metadata={"condition": "india"}) for i in range(3)]
        tasks, key = export_blinded_tasks(candidates, salt="s3cret")
        self.assertEqual(len(tasks), 3)
        for task in tasks:
            self.assertNotIn("generator", task)
            self.assertNotIn("condition", json.dumps(task))
        task_for_c0 = next(t for t, c in key.items() if c == "c0")
        rows = [
            {"task_id": task_for_c0, "annotator_id": "a", "verdict": "accept", "alignment": 5},
            {"task_id": task_for_c0, "annotator_id": "b", "verdict": "accept", "alignment": 4},
            {"task_id": task_for_c0, "annotator_id": "c", "verdict": "reject", "alignment": 2},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "labels.jsonl"
            path.write_text("\n".join(json.dumps(r) for r in rows))
            loaded = load_label_rows(path, key)
            path.write_text(json.dumps({"task_id": task_for_c0, "verdict": "maybe"}))
            with self.assertRaises(ValueError):
                load_label_rows(path, key)
        label = aggregate_labels(loaded)["c0"]
        self.assertTrue(label.human_good)
        self.assertAlmostEqual(label.human_score, 11 / 3)

    def test_blinded_export_copies_images_under_opaque_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "script_001__india_indic" / "sdxl-01.png"
            source.parent.mkdir()
            source.write_bytes(b"fake-image")
            candidate = ImageCandidate("script_001__india_indic-sdxl-01", "script_001__india_indic", "prompt", str(source), generator="sdxl")
            tasks, _ = export_blinded_tasks([candidate], salt="s", image_dir=Path(tmp) / "blind")
            path = tasks[0]["image_path"]
            self.assertNotIn("india", path)
            self.assertNotIn("sdxl", path)
            self.assertEqual(Path(path).read_bytes(), b"fake-image")

    def test_tie_is_undecided(self):
        rows = [{"candidate_id": "x", "verdict": "accept"}, {"candidate_id": "x", "verdict": "reject"}]
        self.assertIsNone(aggregate_labels(rows)["x"].human_good)


def synthetic_study(seed=0, images_per_prompt=4, annotators=3):
    """Matched bank + simulated labels and three evaluators with known behaviour."""
    rng = random.Random(seed)
    candidates, label_rows = [], []
    fair, biased, noise = {}, {}, {}
    for spec in build_matched_bank(include_code_mixed=False):
        for k in range(images_per_prompt):
            cid = f"{spec.prompt_id}-{k}"
            candidates.append(ImageCandidate(cid, spec.prompt_id, spec.prompt, f"/nonexistent/{cid}.png",
                                             generator="sdxl" if k % 2 else "flux", category=spec.category, metadata=dict(spec.metadata)))
            quality = rng.random()
            for a in range(annotators):
                accept = quality + rng.gauss(0, 0.1) > 0.5
                label_rows.append({"candidate_id": cid, "annotator_id": f"a{a}", "verdict": "accept" if accept else "reject",
                                   "alignment": 1 + 4 * quality})
            informative = quality + rng.gauss(0, 0.15)
            fair[cid] = informative
            biased[cid] = rng.random() if spec.metadata["is_india_centric"] else informative
            noise[cid] = rng.random()
    return candidates, label_rows, {"fair": fair, "biased": biased, "noise": noise}


class StudyEndToEndTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        candidates, label_rows, scores = synthetic_study()
        cls.result = run_study(candidates, aggregate_labels(label_rows), scores, StudyConfig(bootstrap_samples=300), label_rows)

    def test_detects_gap_for_biased_evaluator(self):
        contrasts = self.result["evaluators"]["biased"]["contrasts"]
        for key in ("H1_auroc_gap_script", "H1_auroc_gap_cultural", "H1_auroc_gap_pooled"):
            self.assertGreater(contrasts[key]["ci_low"], 0.1, key)
        self.assertGreater(contrasts["H2_far_gap_script"]["ci_low"], 0.0)
        # The script track bias is in Indic script, not the Indian locale.
        self.assertGreater(contrasts["script_effect_auroc"]["ci_low"], 0.1)
        self.assertLess(contrasts["locale_effect_auroc"]["ci_low"], 0.0)
        self.assertGreater(contrasts["locale_effect_auroc"]["ci_high"], 0.0)

    def test_holm_family_excludes_controls(self):
        self.assertEqual(sorted(self.result["confirmatory_family"]),
                         ["biased:H1_auroc_gap_pooled", "fair:H1_auroc_gap_pooled", "noise:H1_auroc_gap_pooled"])
        self.assertIn("p_holm", self.result["evaluators"]["biased"]["contrasts"]["H1_auroc_gap_pooled"])

    def test_no_gap_for_fair_evaluator(self):
        gap = self.result["evaluators"]["fair"]["contrasts"]["H1_auroc_gap_pooled"]
        self.assertLess(gap["ci_low"], 0.0)
        self.assertGreater(gap["ci_high"], 0.0)
        self.assertGreater(self.result["evaluators"]["fair"]["by_condition"]["india"]["auroc"], 0.85)

    def test_noise_evaluator_near_chance(self):
        for entry in self.result["evaluators"]["noise"]["by_condition"].values():
            self.assertLess(abs(entry["auroc"] - 0.5), 0.15)

    def test_human_ceiling_reported(self):
        entry = self.result["evaluators"]["fair"]["by_condition"]["india_indic"]
        self.assertGreater(entry["human_ceiling_auroc"], 0.7)
        self.assertIsNotNone(entry["krippendorff_alpha"])

    def test_by_script_and_report(self):
        scripts = self.result["evaluators"]["fair"]["by_script"]["scripts"]
        self.assertIn("Devanagari", scripts)
        self.assertGreater(scripts["Devanagari"]["script_complexity"], 0)
        report = render_markdown(self.result)
        self.assertIn("H1_auroc_gap_pooled", report)
        json.dumps(self.result)  # must be serialisable

    def test_best_of_n(self):
        fair = self.result["evaluators"]["fair"]["best_of_n"]["global"]
        noise = self.result["evaluators"]["noise"]["best_of_n"]["global"]
        self.assertGreater(fair["capture"], 0.6)
        self.assertLess(abs(noise["capture"]), 0.4)
        self.assertGreater(self.result["evaluators"]["biased"]["best_of_n_capture_gap"]["cultural"]["ci_low"], 0.0)


class SelectionTest(unittest.TestCase):
    def test_capture_extremes(self):
        candidates, label_rows, _ = synthetic_study(images_per_prompt=4)
        rows = build_rows(candidates, aggregate_labels(label_rows))
        perfect = {r.candidate_id: r.human_score for r in rows}
        reverse = {cid: -v for cid, v in perfect.items()}
        self.assertAlmostEqual(capture(rows, perfect, "india"), 1.0)
        self.assertLess(capture(rows, reverse, "india"), 0.0)

    def test_agreement_without_labels_is_empty(self):
        self.assertIsNone(verdict_agreement([])["krippendorff_alpha"])


if __name__ == "__main__":
    unittest.main()
