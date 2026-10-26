from __future__ import annotations

import json
import re
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import make_comparison_table as table
from scripts import run_sweep
from scripts._runs import parse_runs


REPO_ROOT = Path(__file__).resolve().parents[1]


class RunSelectionTests(unittest.TestCase):
    def test_range_and_list_syntax(self):
        self.assertEqual(parse_runs("1:3"), ["1", "2", "3"])
        self.assertEqual(parse_runs("[1, 3, 5]"), ["1", "3", "5"])
        self.assertEqual(parse_runs("2-4,3"), ["2", "3", "4"])

    def test_rejects_descending_range(self):
        with self.assertRaisesRegex(ValueError, "descending"):
            parse_runs("3:1")

    def test_training_checkpoint_paths_are_isolated_by_run(self):
        methods = {
            "lcfx-train": "lcfx",
            "eiyw-train": "eiyw",
            "kvtok-train": "kvtok",
            "xkv-train": "xkv",
        }
        for target, checkpoint_method in methods.items():
            paths = []
            for run_id in ("1", "2"):
                proc = subprocess.run(
                    [
                        "make",
                        "-n",
                        target,
                        f"RUN={run_id}",
                        "DATASET=hotpotqa_bridge",
                        "SHARER=Qwen/Qwen3-0.6B",
                        "RECEIVER=Qwen/Qwen3-0.6B",
                    ],
                    cwd=REPO_ROOT,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                match = re.search(r"--checkpoint-out\s+(\S+)", proc.stdout)
                self.assertIsNotNone(match, proc.stdout)
                path = match.group(1)
                self.assertIn(
                    f"data/checkpoints/runs/run-{run_id}/{checkpoint_method}/",
                    path,
                )
                paths.append(path)
            self.assertNotEqual(paths[0], paths[1], target)

    def test_oom_retry_reduces_batch_repeatedly_to_one(self):
        overrides = {"BATCH_SIZE": "128", "GRAD_ACCUM": "2"}
        observed = []
        while run_sweep.reduce_batch_after_oom(overrides):
            observed.append((overrides["BATCH_SIZE"], overrides["GRAD_ACCUM"]))
        self.assertEqual(
            observed,
            [
                ("64", "4"),
                ("32", "8"),
                ("16", "16"),
                ("8", "32"),
                ("4", "64"),
                ("2", "128"),
                ("1", "256"),
            ],
        )

    def test_cuda_allocator_failure_is_classified_as_oom(self):
        self.assertEqual(
            run_sweep.classify_failure(
                "CUDA error: CUBLAS_STATUS_ALLOC_FAILED when calling cublasCreate"
            ),
            "oom",
        )


class AggregationTests(unittest.TestCase):
    def _write_result(
        self,
        root: Path,
        run_id: str,
        em: float,
        *,
        examples=10,
        extra_metadata=None,
    ):
        result_dir = (
            root
            / "runs"
            / f"run-{run_id}"
            / "t2t"
            / "dataset"
            / "sharer"
            / "receiver"
        )
        result_dir.mkdir(parents=True)
        (result_dir / "metrics.json").write_text(
            json.dumps({"scores": {"em": em, "f1": em + 0.1}}),
            encoding="utf-8",
        )
        metadata = {
            "method": "text_to_text",
            "created_at": f"2026-01-0{run_id}T00:00:00+00:00",
            "run_id": run_id,
            "seed": int(run_id) * 100,
            "dataset": "dataset.jsonl",
            "sharer_model": "sharer",
            "receiver_model": "receiver",
            "num_examples": examples,
            "scoring": "free_form",
        }
        metadata.update(extra_metadata or {})
        (result_dir / "run_metadata.json").write_text(
            json.dumps(metadata), encoding="utf-8"
        )

    def test_score_table_replaces_single_run_value_with_selected_run_mean(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_result(root, "1", 0.1)
            self._write_result(root, "2", 0.3)
            with (
                patch.object(table, "RESULTS_DIR", str(root)),
                patch.object(table, "RUN_IDS", ["1", "2"]),
                patch.object(table, "SELECTED_HP", {}),
            ):
                cell = table.cell_text("dataset", "sharer", "receiver")
            self.assertIn("T **20.0** / **30.0**", cell)
            self.assertNotIn("±", cell)
            self.assertNotIn("n=", cell)

    def test_run_zero_reads_the_legacy_result_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            legacy = root / "t2t" / "dataset" / "sharer" / "receiver"
            legacy.mkdir(parents=True)
            (legacy / "metrics.json").write_text(
                json.dumps({"scores": {"em": 0.4, "f1": 0.5}}),
                encoding="utf-8",
            )
            with (
                patch.object(table, "RESULTS_DIR", str(root)),
                patch.object(table, "RUN_IDS", ["0"]),
                patch.object(table, "SELECTED_HP", {}),
            ):
                paths = table.find_metrics_paths(
                    "t2t", "dataset", "sharer", "receiver"
                )
            self.assertEqual(paths, [str(legacy / "metrics.json")])

    def test_incompatible_metadata_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_result(root, "1", 0.1, examples=10)
            self._write_result(root, "2", 0.2, examples=11)
            with (
                patch.object(table, "RESULTS_DIR", str(root)),
                patch.object(table, "RUN_IDS", ["1", "2"]),
                patch.object(table, "SELECTED_HP", {}),
            ):
                with self.assertRaisesRegex(ValueError, "incompatible"):
                    table.find_metrics_paths(
                        "t2t", "dataset", "sharer", "receiver"
                    )

    def test_known_legacy_metadata_defaults_can_be_aggregated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_result(root, "1", 0.1)
            self._write_result(
                root,
                "2",
                0.2,
                extra_metadata={
                    "dropout": 0.1,
                    "receiver_pooling": True,
                    "pool_spans": 5,
                },
            )
            with (
                patch.object(table, "RESULTS_DIR", str(root)),
                patch.object(table, "RUN_IDS", ["1", "2"]),
                patch.object(table, "SELECTED_HP", {}),
            ):
                paths = table.find_metrics_paths(
                    "t2t", "dataset", "sharer", "receiver"
                )
            self.assertEqual(len(paths), 2)

    def test_explicit_nondefault_dropout_is_incompatible_with_legacy_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_result(root, "1", 0.1)
            self._write_result(
                root, "2", 0.2, extra_metadata={"dropout": 0.2}
            )
            with (
                patch.object(table, "RESULTS_DIR", str(root)),
                patch.object(table, "RUN_IDS", ["1", "2"]),
                patch.object(table, "SELECTED_HP", {}),
            ):
                with self.assertRaisesRegex(ValueError, "incompatible"):
                    table.find_metrics_paths(
                        "t2t", "dataset", "sharer", "receiver"
                    )

    def test_compare_uses_difference_of_nine_cell_method_averages(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models = ["model-a", "model-b", "model-c"]
            method_scores = {
                "lcfx": (0.20, 0.30),
                "xkv": (0.25, 0.37),
                "t2t": (0.10, 0.20),
            }
            for method, (em, f1) in method_scores.items():
                for sharer in models:
                    for receiver in models:
                        result_dir = (
                            root / method / "dataset" / sharer / receiver
                        )
                        result_dir.mkdir(parents=True)
                        (result_dir / "metrics.json").write_text(
                            json.dumps({"scores": {"em": em, "f1": f1}}),
                            encoding="utf-8",
                        )
            with (
                patch.object(table, "RESULTS_DIR", str(root)),
                patch.object(table, "RUN_IDS", []),
                patch.object(table, "SELECTED_HP", {}),
            ):
                report = table.build_method_comparison_table(["dataset"], models)
            self.assertIn("| xkv | 25.0±—/37.0±— (n=1) |", report)
            self.assertIn("| lcfx | 20.0±—/30.0±— (n=1) |", report)
            self.assertIn("| t2t | 10.0±—/20.0±— (n=1) |", report)
            self.assertIn("| xkv-lcfx | +5.0±—/+7.0±— (paired n=0/1) |", report)
            self.assertIn("| xkv-t2t | +15.0±—/+17.0±— (paired n=0/1) |", report)
            self.assertIn("| lcfx-t2t | +10.0±—/+10.0±— (paired n=0/1) |", report)

    def test_compare_uses_all_available_runs_with_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models = ["model-a", "model-b", "model-c"]
            for run_id, method_scores in {
                "1": {"lcfx": (0.20, 0.30), "xkv": (0.25, 0.37)},
                "2": {"xkv": (0.90, 0.90)},
            }.items():
                for method, (em, f1) in method_scores.items():
                    for sharer in models:
                        for receiver in models:
                            result_dir = (
                                root
                                / "runs"
                                / f"run-{run_id}"
                                / method
                                / "dataset"
                                / sharer
                                / receiver
                            )
                            result_dir.mkdir(parents=True)
                            (result_dir / "metrics.json").write_text(
                                json.dumps({"scores": {"em": em, "f1": f1}}),
                                encoding="utf-8",
                            )
            with (
                patch.object(table, "RESULTS_DIR", str(root)),
                patch.object(table, "RUN_IDS", ["1", "2"]),
                patch.object(table, "SELECTED_HP", {}),
            ):
                report = table.build_method_comparison_table(["dataset"], models)
            self.assertIn(
                "| xkv-lcfx | +37.5±—/+33.5±— "
                "(paired n=0/2; results=xkv 18/18, lcfx 9/18) |",
                report,
            )

    def test_compare_reports_sample_sd_and_paired_difference_sd(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models = ["model-a", "model-b", "model-c"]
            scores = {
                "1": {"lcfx": (0.20, 0.30), "xkv": (0.25, 0.37)},
                "2": {"lcfx": (0.22, 0.32), "xkv": (0.29, 0.41)},
                "3": {"lcfx": (0.24, 0.34), "xkv": (0.33, 0.45)},
            }
            for run_id, method_scores in scores.items():
                for method, (em, f1) in method_scores.items():
                    for sharer in models:
                        for receiver in models:
                            result_dir = (
                                root
                                / "runs"
                                / f"run-{run_id}"
                                / method
                                / "dataset"
                                / sharer
                                / receiver
                            )
                            result_dir.mkdir(parents=True)
                            (result_dir / "metrics.json").write_text(
                                json.dumps({"scores": {"em": em, "f1": f1}}),
                                encoding="utf-8",
                            )
                            (result_dir / "run_metadata.json").write_text(
                                json.dumps({"seed": int(run_id)}),
                                encoding="utf-8",
                            )
            with (
                patch.object(table, "RESULTS_DIR", str(root)),
                patch.object(table, "RUN_IDS", ["1", "2", "3"]),
                patch.object(table, "SELECTED_HP", {}),
            ):
                report = table.build_method_comparison_table(["dataset"], models)
            self.assertIn("| xkv | 29.0±4.0/41.0±4.0 (n=3) |", report)
            self.assertIn("| lcfx | 22.0±2.0/32.0±2.0 (n=3) |", report)
            self.assertIn(
                "| xkv-lcfx | +7.0±2.0/+9.0±2.0 (paired n=3) |",
                report,
            )


if __name__ == "__main__":
    unittest.main()
