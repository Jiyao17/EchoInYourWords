from __future__ import annotations

import io
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from methods.common.io import summarize_metrics
from methods.common.prompts import build_receiver_prompt
from methods.common.schema import Example
from methods.common.scoring import score_free_form
from scripts import make_comparison_table as tables
from scripts import run_single_model as single

ROOT = Path(__file__).resolve().parents[1]


class SingleBaselineTests(unittest.TestCase):
    def setUp(self):
        self.example = Example(
            id="sample", question="Which group?", answer="Group A",
            sharer_context="SOURCE_SECRET: the background rule.",
            receiver_context="RECEIVER_EVIDENCE: the situation.",
            metadata={"sharer_part": "background", "receiver_part": "situation"},
        )

    def test_half_is_exact_method_receiver_prompt_with_no_source_leakage(self):
        generate = Mock(return_value="Group A")
        result = single.run_single_model_example(
            self.example, generate, score_free_form, context_mode="half"
        )
        generate.assert_called_once_with(build_receiver_prompt(self.example))
        self.assertNotIn("SOURCE_SECRET", result.receiver_prompt)
        self.assertEqual(result.sharer_prompt, "")
        self.assertEqual(result.sharer_message, "")
        self.assertEqual(result.latency["sharer_ms"], 0)
        self.assertEqual(result.latency["fuser_ms"], 0)
        self.assertEqual(result.token_counts["sharer_input"], 0)
        self.assertEqual(result.metadata["context_mode"], "half")

    def test_full_changes_only_receiver_evidence_and_preserves_original(self):
        generate = Mock(return_value="Group A")
        result = single.run_single_model_example(
            self.example, generate, score_free_form, context_mode="full"
        )
        expected = replace(
            self.example,
            receiver_context=self.example.sharer_context + "\n\n" + self.example.receiver_context,
        )
        self.assertEqual(result.receiver_prompt, build_receiver_prompt(expected))
        self.assertIn("SOURCE_SECRET", result.receiver_prompt)
        self.assertIn("RECEIVER_EVIDENCE", result.receiver_prompt)
        self.assertEqual(self.example.receiver_context, "RECEIVER_EVIDENCE: the situation.")

    def test_invalid_outputs_remain_in_accuracy_and_f1_denominators(self):
        free_form = single.run_single_model_example(
            self.example, lambda _: "", score_free_form, context_mode="half"
        )
        valid = single.run_single_model_example(
            self.example, lambda _: "Group A", score_free_form, context_mode="half"
        )
        metrics = summarize_metrics([free_form, valid])
        self.assertIsNone(free_form.prediction)
        self.assertEqual(metrics["evaluated_examples"], 2)
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["scores"]["f1"], 0.5)

        choice = replace(self.example, choices=("Yes", "No"), answer="A")
        invalid = single.run_single_model_example(
            choice, lambda _: "???", context_mode="half"
        )
        self.assertIsNone(invalid.prediction)
        self.assertFalse(invalid.is_correct)
        self.assertEqual(summarize_metrics([invalid])["evaluated_examples"], 1)

    def test_loader_uses_existing_receiver_decoder_without_residual(self):
        backend = SimpleNamespace(
            decode_receiver=Mock(return_value=("Group A", 12, 2, 3.0, 4.0)),
            sharer_summary=Mock(side_effect=AssertionError("Sharer must not run")),
        )
        with patch("methods.xkv.model.load_xkv_pair", return_value=backend) as load:
            generate = single.load_receiver_generator("model", max_new_tokens=17, device="cpu")
            result = generate("the prompt")
        load.assert_called_once_with(
            "model", "model", device="cpu", dtype="auto", trust_remote_code=False
        )
        backend.decode_receiver.assert_called_once_with(
            "the prompt", residual_fn=None, max_new_tokens=17
        )
        backend.sharer_summary.assert_not_called()
        self.assertEqual(result.text, "Group A")
        self.assertEqual(result.output_tokens, 2)
        self.assertEqual(result.first_token_ms, 3.0)

    def test_both_conditions_share_one_model_and_write_isolated_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dataset = root / "examples.jsonl"
            examples = [self.example, replace(self.example, id="other", answer="Group B")]
            dataset.write_text("".join(json.dumps(e.to_dict()) + "\n" for e in examples))
            output = root / "results"
            output.mkdir()
            historical = output / "metrics.json"
            historical.write_text('{"historical": true}\n')
            generate = Mock(return_value=single.GenerationOutput(text="Group A", input_tokens=10, output_tokens=2))
            argv = [
                "run_single_model.py", "--dataset", str(dataset), "--model", "fake-model",
                "--output-dir", str(output), "--run-id", "7", "--seed", "123",
                "--print-every", "0",
            ]
            with patch.object(sys, "argv", argv), patch.object(
                single, "load_receiver_generator", return_value=generate
            ) as load, redirect_stdout(io.StringIO()):
                single.main()
            self.assertEqual(load.call_count, 1)
            self.assertEqual(generate.call_count, 4)
            self.assertEqual(json.loads(historical.read_text()), {"historical": True})
            for mode in ["half", "full"]:
                folder = output / mode
                metadata = json.loads((folder / "run_metadata.json").read_text())
                self.assertEqual(metadata["context_mode"], mode)
                self.assertEqual(metadata["run_id"], "7")
                self.assertEqual(metadata["seed"], 123)
                self.assertEqual(metadata["decoding"], "receiver_greedy_no_residual")
                metrics = json.loads((folder / "metrics.json").read_text())
                self.assertEqual(metrics["total_examples"], 2)
                records = [json.loads(line) for line in (folder / "predictions.jsonl").read_text().splitlines()]
                self.assertEqual([r["id"] for r in records], ["sample", "other"])
                self.assertEqual(records[0]["metadata"]["context_mode"], mode)
                self.assertEqual("SOURCE_SECRET" in records[0]["receiver_prompt"], mode == "full")
                logs = [json.loads(line) for line in (folder / "eval_log.jsonl").read_text().splitlines()]
                self.assertEqual(logs[-1]["event"], "done")

    def test_make_single_resolves_three_models_and_both_conditions(self):
        environment = dict(os.environ)
        for name in ["MODEL", "EXTRA_MODELS", "PRIMARY_MODELS"]:
            environment.pop(name, None)
        result = subprocess.run(
            ["make", "-n", "single", "DATASET=ropes", "RUN=7", "SEED=123"],
            cwd=ROOT, env=environment, check=True, capture_output=True, text=True,
        )
        models = re.findall(r"--model (\S+) --context-mode (\S+)", result.stdout)
        self.assertEqual(models, [
            ("Qwen/Qwen3-0.6B", "both"),
            ("unsloth/gemma-3-1b-it", "both"),
            ("unsloth/Llama-3.2-3B-Instruct", "both"),
        ])
        self.assertIn("results/runs/run-7/single/ropes/", result.stdout)
        self.assertNotIn("train_xkv.py", result.stdout)

    def test_make_can_select_one_model_and_half_context(self):
        result = subprocess.run(
            ["make", "-n", "single", "DATASET=ropes", "MODEL=Qwen/Qwen3-0.6B",
             "SINGLE_CONTEXT=half", "SEED=123"],
            cwd=ROOT, check=True, capture_output=True, text=True,
        )
        self.assertEqual(re.findall(r"--model (\S+) --context-mode (\S+)", result.stdout), [
            ("Qwen/Qwen3-0.6B", "half")
        ])

    def test_overview_never_mislabels_half_context_as_full(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            base = root / "single/ropes/model"
            (base / "half").mkdir(parents=True)
            (base / "half/metrics.json").write_text("{}")
            with patch.object(tables, "RESULTS_DIR", str(root)):
                self.assertIsNone(tables.find_single_metrics_path("ropes", "model"))
                (base / "metrics.json").write_text("{}")
                self.assertEqual(tables.find_single_metrics_path("ropes", "model"), str(base / "metrics.json"))
                (base / "full").mkdir()
                (base / "full/metrics.json").write_text("{}")
                self.assertEqual(tables.find_single_metrics_path("ropes", "model"), str(base / "full/metrics.json"))


if __name__ == "__main__":
    unittest.main()
