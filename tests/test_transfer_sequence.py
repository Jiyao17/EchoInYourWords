from __future__ import annotations

import argparse
import copy
import io
import json
import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from scripts import run_transfer_sequence as sequence
from scripts._transfer import (
    add_transfer_arguments, stamp_training_history, transfer_metadata,
    validate_checkpoint_history, validate_warm_start_paths,
)


class WarmStartTests(unittest.TestCase):
    def test_trainers_load_learned_weights_and_preserve_history(self):
        from methods import lcfx, xkv
        from scripts import train_lcfx_fuser, train_xkv

        spec = SimpleNamespace(name="tiny/model", layers=2, kv_heads=1, head_dim=4, hidden=32)
        pair = SimpleNamespace(sharer_spec=spec, receiver_spec=spec, device="cpu")
        for method in ("xkv", "lcfx"):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                source, output = root / "source.pt", root / "stage2.pt"
                data = root / "musique_train.jsonl"
                data.write_text(json.dumps({"id": "1", "question": "q", "answer": "a"}) + "\n")
                if method == "xkv":
                    module = xkv.XKVTranslator.from_specs(spec, spec, d_model=8, num_heads=2,
                                                          pool_slots=1, zero_init=False)
                    save = xkv.save_translator
                    runner = train_xkv
                    train_name, pair_name = "train_translator", "load_xkv_pair"
                    architecture = ["--d-model", "8", "--num-heads", "2", "--pool-k", "1"]
                else:
                    module = lcfx.LCFXFuser.from_specs(spec, spec, bottleneck_dim=8, zero_init=False)
                    save = lcfx.save_fuser
                    runner = train_lcfx_fuser
                    train_name, pair_name = "train_fuser", "load_lcfx_pair"
                    architecture = ["--bottleneck-dim", "8"]
                module.training_history = {"training_datasets": ["hotpotqa_bridge"]}
                with torch.no_grad():
                    next(module.parameters()).fill_(0.314)
                original_state = copy.deepcopy(module.state_dict())
                save(module, str(source), pair)
                source_hash = sequence.fingerprint(source)

                def train(_pair, loaded, rows, **kwargs):
                    for name, value in loaded.state_dict().items():
                        torch.testing.assert_close(value, original_state[name])
                    self.assertEqual(loaded.training_history["training_datasets"],
                                     ["hotpotqa_bridge", "musique"])
                    self.assertTrue(loaded.training_history["optimizer_reset"])
                    save(loaded, kwargs["checkpoint_path"], pair)

                argv = ["trainer", "--train-data", str(data), "--initial-checkpoint", str(source),
                        "--checkpoint-out", str(output), "--seed", "123", "--val-frac", "0",
                        "--training-datasets", "hotpotqa_bridge", "musique", *architecture]
                backend = xkv if method == "xkv" else lcfx
                with patch.object(backend, pair_name, return_value=pair), \
                        patch.object(backend, train_name, side_effect=train) as trained, \
                        patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()):
                    runner.main()
                trained.assert_called_once()
                saved = torch.load(output, map_location="cpu", weights_only=True)
                self.assertEqual(saved["training_history"]["transfer_step"], 2)
                self.assertEqual(sequence.fingerprint(source), source_hash)

    def test_history_and_checkpoint_overwrite_guards(self):
        with tempfile.TemporaryDirectory() as temp:
            checkpoint = Path(temp) / "source.pt"
            checkpoint.touch()
            args = SimpleNamespace(initial_checkpoint=checkpoint, checkpoint_out=checkpoint,
                                   training_datasets=["hotpotqa_bridge", "musique"])
            with self.assertRaisesRegex(ValueError, "must differ"):
                validate_warm_start_paths(args)
            module = SimpleNamespace(training_history={"training_datasets": ["ropes"]})
            with self.assertRaisesRegex(ValueError, "extend"):
                stamp_training_history(module, args)
            parser = argparse.ArgumentParser()
            add_transfer_arguments(parser)
            args = parser.parse_args([
                "--training-dataset", "hotpotqa_bridge+musique", "--inference-dataset", "musique",
                "--training-datasets", "hotpotqa_bridge", "musique",
            ])
            with self.assertRaisesRegex(ValueError, "already"):
                transfer_metadata(args, checkpoint)
            args.inference_dataset = "ropes"
            info = transfer_metadata(args, checkpoint)
            with self.assertRaisesRegex(ValueError, "does not match"):
                validate_checkpoint_history(module, info)


class SequenceTests(unittest.TestCase):
    def test_make_frontend_executes_sequence_without_target_training_data(self):
        # Exercise real recursive make/argv wiring, replacing only model runners.
        stub_code = '''import json, runpy, sys
from pathlib import Path
root = Path(__file__).parent
target = sys.argv.pop(1)
if target == "scripts/run_transfer_sequence.py":
    runpy.run_path(target, run_name="__main__")
    raise SystemExit
args = sys.argv[1:]
with (root / "calls.jsonl").open("a") as f:
    f.write(json.dumps([target, *args]) + "\\n")
def value(flag, default=None):
    return args[args.index(flag) + 1] if flag in args else default
if target.startswith("scripts/train_"):
    assert Path(value("--initial-checkpoint")).is_file()
    output = Path(value("--checkpoint-out"))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("continued weights")
else:
    baseline = target.endswith("run_single_model.py")
    output = Path(value("--output-dir"))
    if baseline:
        output = output / "half"
    output.mkdir(parents=True, exist_ok=True)
    history = []
    if "--training-datasets" in args:
        for item in args[args.index("--training-datasets") + 1:]:
            if item.startswith("--"):
                break
            history.append(item)
    model = value("--model", value("--receiver-model"))
    metadata = dict(scoring="free_form", context_mode="half", model=model,
                    receiver_model=model, max_new_tokens=int(value("--max-new-tokens")),
                    training_datasets=history, transfer_step=len(history),
                    inference_dataset=value("--inference-dataset"))
    (output / "run_metadata.json").write_text(json.dumps(metadata))
    (output / "metrics.json").write_text("{}")
    row = dict(id="1", answer="a", receiver_prompt="same", is_correct=True,
               scores=dict(em=1, f1=1))
    if (root / "win_step").is_file():
        win_step = int((root / "win_step").read_text())
        score = 0.75 if not baseline and len(history) >= win_step else 0.5
        row["scores"] = dict(em=score, f1=score)
    (output / "predictions.jsonl").write_text(json.dumps(row) + "\\n")
'''
        for method, winning_step in (("xkv", None), ("lcfx", None), ("xkv", 1), ("lcfx", 2)):
            with self.subTest(method=method, winning_step=winning_step), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                (root / "stub.py").write_text(stub_code)
                if winning_step:
                    (root / "win_step").write_text(str(winning_step))
                (root / "source.pt").write_text("source weights")
                tasks = root / "tasks"
                tasks.mkdir()
                for dataset, split in (("musique", "train"), ("musique", "val"), ("ropes", "val")):
                    (tasks / f"{dataset}_{split}.jsonl").write_text("{}\n")
                environment = dict(os.environ)
                for key in ("MAKEFLAGS", "MAKEOVERRIDES", "MFLAGS"):
                    environment.pop(key, None)
                subprocess.run([
                    "make", "transfer-sequence", f"METHOD={method}", "SEED=123", "RUN=7",
                    "SOURCE_RUN=5", "SHARER=tiny/model", "RECEIVER=tiny/model",
                    "TRANSFER_DATASETS=hotpotqa_bridge musique qasc", "TRANSFER_STEP=2",
                    "TRANSFER_TARGETS=ropes", f"SEQUENCE_SOURCE_CKPT={root / 'source.pt'}",
                    f"SEQUENCE_CKPT_ROOT={root / 'checkpoints'}", f"SEQUENCE_OUT={root / 'results'}",
                    f"TASKS={tasks}", "OFFLINE=", "EVAL_N=1",
                    f"PY={shlex.quote(sys.executable)} {shlex.quote(str(root / 'stub.py'))}",
                ], cwd=sequence.ROOT, env=environment, check=True, capture_output=True, text=True)
                calls = [json.loads(line) for line in (root / "calls.jsonl").read_text().splitlines()]
                trained = [call for call in calls if call[0].startswith("scripts/train_")]
                evaluated_steps = winning_step or 2
                self.assertEqual(len(trained), evaluated_steps - 1)
                if trained:
                    self.assertEqual(trained[0][trained[0].index("--initial-checkpoint") + 1], str(root / "source.pt"))
                    self.assertIn("--training-datasets", trained[0])
                self.assertEqual(sum(call[0].endswith("run_single_model.py") for call in calls), 1)
                self.assertEqual(sum(call[0] == f"scripts/run_{method}.py" for call in calls), evaluated_steps)
                self.assertFalse((tasks / "ropes_train.jsonl").exists())
                summary = sequence.read_json(root / "results" / "eval-1-tokens-64" / "summary.json")
                self.assertEqual([row["transfer_step"] for row in summary], list(range(1, evaluated_steps + 1)))
                report = sequence.read_json(root / "results" / "eval-1-tokens-64" / "stopping_result-ropes-em.json")
                self.assertEqual(report["status"], "positive_gain" if winning_step else "step_limit_reached")
                self.assertEqual(report["stopping_step"], winning_step)

    def test_make_defaults_fix_ropes_target_and_four_ordered_sources(self):
        result = subprocess.run(
            ["make", "transfer-sequence", "SEQUENCE_DRY_RUN=1", "SEED=123"],
            cwd=sequence.ROOT, check=True, capture_output=True, text=True,
        )
        self.assertIn("--datasets hotpotqa_bridge musique strategyqa qasc", result.stdout)
        self.assertIn("--targets ropes", result.stdout)
        self.assertIn("--stop-on-gain --stop-metric em", result.stdout)
        self.assertIn("DATASET=strategyqa", result.stdout)
        self.assertNotIn("2wikimultihop", result.stdout)
        self.assertNotIn("INFER_DATASET=strategyqa", result.stdout)
        self.assertNotIn("INFER_DATASET=musique", result.stdout)

    def arguments(self, root, method="xkv"):
        return ["--method", method, "--datasets", "hotpotqa_bridge", "musique", "qasc",
                "--source-checkpoint", str(root / "source.pt"), "--source-run-id", "5",
                "--checkpoint-root", str(root / "checkpoints"), "--output-root", str(root / "results"),
                "--sharer-model", "tiny/model", "--receiver-model", "tiny/model", "--seed", "123",
                "--make-var", f"TASKS={root / 'tasks'}"]

    def fake_execute(self, commands, *, stage_scores=None, baseline_scores=None):
        def execute(instance, command):
            commands.append(command)
            target = command[2]
            values = dict(item.split("=", 1) for item in command[3:])
            if target.endswith("-train"):
                output = Path(values.get("XKV_CKPT", values.get("CKPT")))
                self.assertTrue(Path(values["INIT_CKPT"]).is_file())
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text("continued " + values["TRAINING_DATASETS"])
            elif target in ("transfer", "_transfer-half"):
                baseline = target == "_transfer-half"
                output = Path(values["SINGLE_OUT"]) / "half" if baseline else Path(values["TRANSFER_OUT"])
                history = values.get("TRAINING_DATASETS", "").split()
                metadata = {"scoring": "free_form", "max_new_tokens": int(values["RECV_MAX"]),
                            "context_mode": "half", "model": "tiny/model", "receiver_model": "tiny/model",
                            "training_datasets": history, "transfer_step": len(history),
                            "inference_dataset": values["INFER_DATASET"]}
                sequence.write_json(output / "run_metadata.json", metadata)
                sequence.write_json(output / "metrics.json", {})
                scores = (baseline_scores if baseline else (stage_scores or {}).get(len(history)))
                rows = [dict(id="1", answer="a", receiver_prompt="same evidence", is_correct=True,
                             scores=scores if scores is not None else {"em": 1, "f1": 1})]
                (output / "predictions.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
        return execute

    def test_fixed_target_stops_at_first_strict_gain_and_reuses_that_result(self):
        cases = (
            ("em", {1: {"em": 0.75, "f1": 0.75}}, 1),
            # An EM tie continues even when F1 already beats the baseline.
            ("em", {1: {"em": 0.5, "f1": 0.9}, 2: {"em": 0.75, "f1": 0.9}}, 2),
            ("f1", {1: {"em": 0.5, "f1": 0.75}}, 1),
        )
        for metric, scores, stop_step in cases:
            for method in ("xkv", "lcfx"):
                with self.subTest(metric=metric, step=stop_step, method=method), tempfile.TemporaryDirectory() as temp:
                    root = Path(temp)
                    (root / "source.pt").write_text("original")
                    (root / "tasks").mkdir()
                    (root / "tasks" / "ropes_val.jsonl").write_text("{}\n")
                    arguments = [*self.arguments(root, method), "--targets", "ropes",
                                 "--stop-on-gain", "--stop-metric", metric]
                    commands = []
                    with patch.object(sequence.Sequence, "execute", self.fake_execute(
                            commands, stage_scores=scores, baseline_scores={"em": 0.5, "f1": 0.5})), \
                            redirect_stdout(io.StringIO()):
                        sequence.main(arguments)
                        trained = [c for c in commands if c[2].endswith("-train")]
                        self.assertEqual(len(trained), stop_step - 1)
                        evaluated = [c for c in commands if c[2] == "transfer"]
                        self.assertEqual(len(evaluated), stop_step)
                        self.assertTrue(all("INFER_DATASET=ropes" in c for c in evaluated))
                        commands.clear()
                        sequence.main(arguments)
                        self.assertFalse(any(c[2] in ("transfer", "_transfer-half", f"{method}-train")
                                             for c in commands))
                    budget = root / "results" / "eval-full-tokens-64"
                    report = sequence.read_json(budget / f"stopping_result-ropes-{metric}.json")
                    self.assertEqual(report["status"], "positive_gain")
                    self.assertEqual(report["stopping_step"], stop_step)
                    self.assertGreater(report["delta_pp"], 0)
                    self.assertEqual(len(sequence.read_json(budget / "summary.json")), stop_step)
                    self.assertFalse((root / "checkpoints" / "step-03-qasc").exists())

    def test_fixed_target_exhaustion_and_manual_step_limit_do_not_claim_a_gain(self):
        for limit, status in ((None, "sources_exhausted"), (2, "step_limit_reached")):
            with self.subTest(limit=limit), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                (root / "source.pt").write_text("original")
                (root / "tasks").mkdir()
                (root / "tasks" / "ropes_val.jsonl").write_text("{}\n")
                arguments = [*self.arguments(root), "--targets", "ropes", "--stop-on-gain"]
                if limit:
                    arguments += ["--step", str(limit)]
                commands = []
                with patch.object(sequence.Sequence, "execute", self.fake_execute(
                        commands, stage_scores={1: {"em": 0.5, "f1": 0.5},
                                                2: {"em": 0.25, "f1": 0.25},
                                                3: {"em": 0.5, "f1": 0.5}},
                        baseline_scores={"em": 0.5, "f1": 0.5})), redirect_stdout(io.StringIO()):
                    sequence.main(arguments)
                report = sequence.read_json(root / "results" / "eval-full-tokens-64" / "stopping_result-ropes-em.json")
                self.assertEqual(report["status"], status)
                self.assertIsNone(report["stopping_step"])
                self.assertLessEqual(report["delta_pp"], 0)

    def test_stopping_rule_rejects_multiple_targets_or_a_target_anywhere_in_source_plan(self):
        with tempfile.TemporaryDirectory() as temp:
            for targets in (("ropes", "strategyqa"), ("qasc",)):
                with self.subTest(targets=targets), self.assertRaises(SystemExit), \
                        patch("sys.stderr", new=io.StringIO()), patch.object(subprocess, "run") as run:
                    sequence.main([*self.arguments(Path(temp)), "--stop-on-gain", "--targets", *targets])
                run.assert_not_called()

    def test_stages_continue_previous_weights_and_only_evaluate_untrained_targets(self):
        for method in ("xkv", "lcfx"):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                (root / "source.pt").write_text("source weights")
                (root / "tasks").mkdir()
                for dataset in sequence.DATASETS:
                    (root / "tasks" / f"{dataset}_val.jsonl").write_text("{}\n")
                commands = []
                with patch.object(sequence.Sequence, "execute", self.fake_execute(commands)), \
                        redirect_stdout(io.StringIO()):
                    sequence.main(self.arguments(root, method))
                    first_commands = list(commands)
                    # Restart reuses every trained stage and completed evaluation.
                    commands.clear()
                    sequence.main(self.arguments(root, method))
                    self.assertFalse(any(c[2] in ("transfer", "_transfer-half", f"{method}-train")
                                         for c in commands))
                    commands.clear()
                    sequence.main([*self.arguments(root, method), "--evaluate-only", "--step", "2",
                                   "--targets", "ropes"])
                    self.assertFalse(any(c[2].endswith("-train") for c in commands))
                training = [dict(item.split("=", 1) for item in c[3:])
                            for c in first_commands if c[2] == f"{method}-train"]
                self.assertEqual([c["DATASET"] for c in training], ["musique", "qasc"])
                self.assertEqual(training[1]["INIT_CKPT"], training[0].get("XKV_CKPT", training[0].get("CKPT")))
                evaluations = [dict(item.split("=", 1) for item in c[3:])
                               for c in first_commands if c[2] == "transfer"]
                self.assertEqual(len(evaluations), 5 + 4 + 3)
                for evaluation in evaluations:
                    self.assertNotIn(evaluation["INFER_DATASET"], evaluation["TRAINING_DATASETS"].split())
                self.assertEqual(sum(c[2] == "_transfer-half" for c in first_commands), 5)
                self.assertEqual((root / "source.pt").read_text(), "source weights")

    def test_invalid_eval_request_and_dry_run_never_write_or_train(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with redirect_stdout(io.StringIO()), patch.object(subprocess, "run") as run:
                sequence.main([*self.arguments(root), "--dry-run", "--step", "2"])
                run.assert_not_called()
            self.assertFalse((root / "checkpoints").exists())
            with self.assertRaises(SystemExit), redirect_stdout(io.StringIO()), \
                    patch("sys.stderr", new=io.StringIO()), patch.object(subprocess, "run") as run:
                sequence.main([*self.arguments(root), "--evaluate-only", "--step", "2", "--targets", "musique"])
            run.assert_not_called()

    def test_changed_configuration_or_checkpoint_cannot_silently_replace_a_stage(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "source.pt").write_text("original")
            (root / "tasks").mkdir()
            (root / "tasks" / "ropes_val.jsonl").write_text("{}\n")
            arguments = [*self.arguments(root), "--step", "2", "--targets", "ropes"]
            commands = []
            with patch.object(sequence.Sequence, "execute", self.fake_execute(commands)), \
                    redirect_stdout(io.StringIO()), patch("sys.stderr", new=io.StringIO()):
                sequence.main(arguments)
                commands.clear()
                with self.assertRaises(SystemExit):
                    sequence.main([*arguments, "--seed", "456"])
                self.assertFalse(commands)
                stage_checkpoint = root / "checkpoints" / "step-02-musique" / "translator.pt"
                stage_checkpoint.write_text("modified weights")
                with self.assertRaises(SystemExit):
                    sequence.main(arguments)
                self.assertFalse(any(c[2].endswith("-train") for c in commands))
                self.assertEqual(stage_checkpoint.read_text(), "modified weights")

    def test_comparison_counts_invalid_outputs_and_rejects_mismatched_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            transfer, baseline = root / "transfer", root / "baseline"
            metadata = {"scoring": "free_form", "max_new_tokens": 64, "context_mode": "half",
                        "model": "tiny", "receiver_model": "tiny", "transfer_step": 1,
                        "training_datasets": ["hotpotqa_bridge"], "inference_dataset": "ropes"}
            for path in (transfer, baseline):
                sequence.write_json(path / "run_metadata.json", metadata)
            def rows(score):
                return [dict(id="1", answer="a", receiver_prompt="evidence", scores={"em": score, "f1": score}),
                        dict(id="2", answer="b", receiver_prompt="evidence2", scores={}, is_correct=None)]
            def write_rows(path, values):
                (path / "predictions.jsonl").write_text("\n".join(json.dumps(row) for row in values))
            write_rows(transfer, rows(1))
            write_rows(baseline, rows(0))
            result = sequence.compare_predictions(transfer, baseline)
            self.assertEqual(result["num_examples"], 2)
            self.assertEqual(result["metrics"]["em"]["delta_pp"], 50)
            self.assertEqual(result["metrics"]["em"]["ties"], 1)
            mismatched = rows(0)
            mismatched[0]["receiver_prompt"] = "different evidence"
            write_rows(baseline, mismatched)
            with self.assertRaisesRegex(ValueError, "different answer or receiver evidence"):
                sequence.compare_predictions(transfer, baseline)


if __name__ == "__main__":
    unittest.main()
