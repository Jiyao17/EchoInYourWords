from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts._transfer import add_transfer_arguments, transfer_metadata


ROOT = Path(__file__).resolve().parents[1]
MODEL = "unsloth/gemma-3-1b-it"
PAIR = "unsloth--gemma-3-1b-it/unsloth--gemma-3-1b-it"


class TransferTests(unittest.TestCase):
    def make(self, *arguments, check=True):
        environment = dict(os.environ)
        for key in ("METHOD", "RUN", "N", "SOURCE_RUN", "DATASET", "TRAIN_DATASET",
                    "INFER_DATASET", "TRANSFER_CKPT", "TRANSFER_OUT"):
            environment.pop(key, None)
        return subprocess.run(
            ["make", *arguments, "SEED=123"], cwd=ROOT, env=environment,
            capture_output=True, text=True, check=check,
        )

    def test_source_checkpoint_and_target_data_are_independent_for_both_methods(self):
        for method, shape, filename in (
            ("xkv", "dh8_n1_h8_do0p1_k2_v11", "translator.pt"),
            ("lcfx", "sp5", "fuser.pt"),
        ):
            with self.subTest(method=method):
                output = self.make(
                    "-n", "transfer", f"METHOD={method}", "TRAIN_DATASET=hotpotqa_bridge",
                    "INFER_DATASET=ropes", "SOURCE_RUN=5", "RUN=7",
                    f"SHARER={MODEL}", f"RECEIVER={MODEL}", "EVAL_N=12", "RECV_MAX=32",
                ).stdout
                checkpoint = f"data/checkpoints/runs/run-5/{method}/hotpotqa_bridge/{PAIR}/{shape}/{filename}"
                self.assertIn(f'"{checkpoint}"', output)
                self.assertIn("--dataset data/tasks/ropes_val.jsonl", output)
                self.assertIn("--training-dataset hotpotqa_bridge --inference-dataset ropes --source-run-id 5", output)
                self.assertIn(f"results/runs/run-7/transfer/hotpotqa_bridge_to_ropes/source-run-5/{method}/{PAIR}/{shape}", output)
                self.assertIn("--max-new-tokens 32 --num-examples 12", output)
                self.assertNotIn("scripts/train_", output)
                self.assertNotIn("--train-data", output)

    def test_other_dataset_direction_and_historical_source_run(self):
        output = self.make(
            "-n", "transfer", "TRAIN_DATASET=ropes", "INFER_DATASET=qasc",
            "DATASET=musique", "SOURCE_RUN=0", "RUN=8",
        ).stdout
        self.assertIn('"data/checkpoints/xkv/ropes/', output)
        self.assertIn("--dataset data/tasks/qasc_val.jsonl", output)
        self.assertIn("results/runs/run-8/transfer/ropes_to_qasc/source-run-0/", output)
        self.assertNotIn("musique/", output)

    def test_source_run_defaults_to_output_run(self):
        output = self.make("-n", "transfer", "RUN=3").stdout
        self.assertIn('"data/checkpoints/runs/run-3/xkv/', output)
        self.assertIn("--source-run-id 3", output)

    def test_missing_checkpoint_or_unsupported_method_stops_before_inference(self):
        with tempfile.TemporaryDirectory() as temp:
            for method, error in (("xkv", "Missing source checkpoint"), ("t2t", "supports METHOD")):
                with self.subTest(method=method):
                    result = self.make(
                        "transfer", f"METHOD={method}", f"TRANSFER_CKPT={temp}/absent.pt",
                        "PY=false", check=False,
                    )
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(error, result.stderr)
                    self.assertNotIn("scripts/run_", result.stdout)

    def test_make_executes_only_target_evaluation_and_preserves_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "ropes_val.jsonl").write_text("{}\n")
            checkpoint = root / "source.pt"
            checkpoint.write_text("source weights")
            capture = root / "arguments.json"
            stub = root / "capture.py"
            stub.write_text(
                "import json, sys\nfrom pathlib import Path\n"
                f"Path({str(capture)!r}).write_text(json.dumps(sys.argv[1:]))\n"
            )
            for method, flag in (("xkv", "--translator-checkpoint"), ("lcfx", "--fuser-checkpoint")):
                with self.subTest(method=method):
                    self.make(
                        "transfer", f"METHOD={method}", f"TASKS={root}",
                        f"TRANSFER_CKPT={checkpoint}", f"TRANSFER_OUT={root}/output",
                        f"PY={shlex.quote(sys.executable)} {shlex.quote(str(stub))}",
                        "SOURCE_RUN=5", "RUN=7", f"SHARER={MODEL}", f"RECEIVER={MODEL}",
                    )
                    argv = json.loads(capture.read_text())
                    self.assertEqual(argv[0], f"scripts/run_{method}.py")
                    self.assertEqual(argv[argv.index("--dataset") + 1], str(root / "ropes_val.jsonl"))
                    self.assertEqual(argv[argv.index(flag) + 1], str(checkpoint))
                    self.assertEqual(argv[argv.index("--run-id") + 1], "7")
                    self.assertEqual(argv[argv.index("--source-run-id") + 1], "5")
                    self.assertEqual(checkpoint.read_text(), "source weights")
                    self.assertFalse((root / "ropes_train.jsonl").exists())
                    self.assertFalse((root / "hotpotqa_bridge_train.jsonl").exists())

    def test_transfer_provenance_requires_weights_and_leaves_normal_evaluation_unchanged(self):
        parser = argparse.ArgumentParser()
        add_transfer_arguments(parser)
        self.assertEqual(transfer_metadata(parser.parse_args([]), None), {})
        with self.assertRaisesRegex(ValueError, "both"):
            transfer_metadata(parser.parse_args(["--training-dataset", "hotpotqa_bridge"]), None)
        args = parser.parse_args([
            "--training-dataset", "hotpotqa_bridge", "--inference-dataset", "ropes",
            "--source-run-id", "5",
        ])
        with self.assertRaisesRegex(ValueError, "existing trained checkpoint"):
            transfer_metadata(args, None)
        with tempfile.TemporaryDirectory() as temp:
            checkpoint = Path(temp) / "source.pt"
            with self.assertRaisesRegex(ValueError, "existing trained checkpoint"):
                transfer_metadata(args, checkpoint)
            checkpoint.touch()
            metadata = transfer_metadata(args, checkpoint)
            self.assertEqual(metadata["training_dataset"], "hotpotqa_bridge")
            self.assertEqual(metadata["inference_dataset"], "ropes")
            self.assertEqual(metadata["source_run_id"], "5")
            self.assertFalse(metadata["target_finetuning"])


if __name__ == "__main__":
    unittest.main()
