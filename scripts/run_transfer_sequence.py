"""Continue existing weights one dataset at a time; evaluate untrained targets.

Drive through make transfer-sequence / transfer-sequence-eval. This module uses
only the standard library; training and inference remain in the existing runners.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("hotpotqa_bridge", "musique", "qasc", "2wikimultihop", "ropes", "strategyqa")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compare_predictions(transfer: Path, baseline: Path) -> dict:
    """Compute paired gains, counting invalid predictions as errors."""
    def load(path):
        rows = [json.loads(line) for line in (path / "predictions.jsonl").read_text().splitlines()
                if line.strip()]
        by_id = {row["id"]: row for row in rows}
        if not rows or len(by_id) != len(rows):
            raise ValueError(f"empty predictions or duplicate example IDs: {path}")
        return by_id

    translated, half = load(transfer), load(baseline)
    if translated.keys() != half.keys():
        raise ValueError("transfer and half-context baseline must use the same example IDs")
    info, base_info = read_json(transfer / "run_metadata.json"), read_json(baseline / "run_metadata.json")
    if (base_info.get("context_mode") != "half"
            or info.get("receiver_model") != base_info.get("model")
            or info.get("max_new_tokens") != base_info.get("max_new_tokens")
            or info.get("scoring") != base_info.get("scoring")):
        raise ValueError("baseline must match the Receiver, scoring, decoding budget and half context")
    fields = ("em", "f1") if info["scoring"] == "free_form" else ("accuracy",)
    totals = {field: [0.0, 0.0, 0, 0, 0] for field in fields}
    for key, row in translated.items():
        other = half[key]
        if any(row.get(field) != other.get(field) for field in ("answer", "receiver_prompt")):
            raise ValueError(f"baseline has different answer or receiver evidence for example {key}")
        for field in fields:
            if field == "accuracy":
                left, right = float(bool(row.get("is_correct"))), float(bool(other.get("is_correct")))
            else:
                left = float(row.get("scores", {}).get(field, 0.0))
                right = float(other.get("scores", {}).get(field, 0.0))
            values = totals[field]
            values[0] += left
            values[1] += right
            values[2 if left > right else 3 if left < right else 4] += 1
    count = len(translated)
    return {
        "num_examples": count,
        "training_datasets": info["training_datasets"],
        "target_dataset": info["inference_dataset"],
        "transfer_step": info["transfer_step"],
        "metrics": {
            field: {"transfer": values[0] / count, "receiver_half": values[1] / count,
                    "delta_pp": 100 * (values[0] - values[1]) / count,
                    "wins": values[2], "losses": values[3], "ties": values[4]}
            for field, values in totals.items()
        },
        "interpretation": "Descriptive gains only; no automatic transfer-validation claim.",
    }


def write_summary(root: Path) -> None:
    comparisons = [read_json(path) for path in sorted(root.glob("step-*/*/comparison.json"))]
    write_json(root / "summary.json", comparisons)
    lines = [
        "# Sequential transfer against Receiver-only half context", "",
        "Descriptive results across evaluated stages and targets. A positive delta alone",
        "does not establish statistical reliability or validate a selected best result.", "",
        "| Step | Trained datasets | Target | N | Metric | Receiver half | Transfer | Delta (pp) |",
        "| --- | --- | --- | ---: | --- | ---: | ---: | ---: |",
    ]
    for comparison in comparisons:
        for metric, values in comparison["metrics"].items():
            lines.append(
                f"| {comparison['transfer_step']} | {', '.join(comparison['training_datasets'])} "
                f"| {comparison['target_dataset']} | {comparison['num_examples']} | {metric} "
                f"| {100 * values['receiver_half']:.2f} | {100 * values['transfer']:.2f} "
                f"| {values['delta_pp']:+.2f} |"
            )
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


class Sequence:
    def __init__(self, args):
        self.args = args
        self.options = dict(item.split("=", 1) for item in args.make_var)
        self.tasks = Path(self.options.get("TASKS", "data/tasks"))
        self.env = dict(os.environ)
        # Child make invocations get explicit argv, without inherited overrides,
        # dry-run flags or jobserver descriptors from the parent make process.
        for key in ("MAKEFLAGS", "MAKEOVERRIDES", "MFLAGS"):
            self.env.pop(key, None)
        self.common = {
            **self.options, "METHOD": args.method, "SHARER": args.sharer_model,
            "RECEIVER": args.receiver_model, "SEED": str(args.seed),
            "RUN": args.run_id or "", "SOURCE_RUN": args.source_run_id,
        }
        # A method-specific override from the parent environment must not replace
        # the explicitly selected pair or redirect a stage's data/checkpoints.
        for prefix in ("XKV", "LCFX"):
            self.common[f"{prefix}_SHARER"] = args.sharer_model
            self.common[f"{prefix}_RECEIVER"] = args.receiver_model

    def command(self, target: str, **overrides) -> list[str]:
        return ["make", "--no-print-directory", target,
                *(f"{key}={value}" for key, value in {**self.common, **overrides}.items())]

    def execute(self, command: list[str]) -> None:
        print(shlex.join(command), flush=True)
        if not self.args.dry_run:
            subprocess.run(command, cwd=ROOT, env=self.env, check=True)

    def identity(self) -> dict:
        args = self.args
        return {
            "method": args.method, "datasets": args.datasets,
            "sharer_model": args.sharer_model, "receiver_model": args.receiver_model,
            "seed": args.seed, "source_run_id": args.source_run_id,
            "source_checkpoint": str(args.source_checkpoint.resolve()),
            "source_sha256": fingerprint(args.source_checkpoint),
            "make_options": self.options,
        }

    def cached_evaluation(self, command: list[str], output: Path, inputs: dict) -> None:
        marker = output / "transfer_complete.json"
        required = ("metrics.json", "run_metadata.json", "predictions.jsonl")
        if not self.args.dry_run and marker.is_file():
            if read_json(marker) != inputs:
                raise ValueError(f"evaluation inputs changed; use a different SEQUENCE_OUT: {output}")
            if all((output / name).is_file() for name in required):
                print(f"Reusing completed evaluation: {output}", flush=True)
                return
        self.execute(command)
        if not self.args.dry_run:
            if not all((output / name).is_file() for name in required):
                raise ValueError(f"evaluation did not write complete outputs: {output}")
            write_json(marker, inputs)

    def evaluation_root(self) -> Path:
        args = self.args
        budget = f"eval-{args.num_examples or 'full'}-tokens-{args.max_new_tokens}"
        return args.output_root / budget

    def evaluate(self, step: int, checkpoint: Path) -> list[dict]:
        args = self.args
        history = args.datasets[:step]
        targets = [target for target in args.targets if target not in history]
        if args.evaluate_only and args.explicit_targets and len(targets) != len(args.targets):
            raise ValueError("requested evaluation target has already been used for training")
        root = self.evaluation_root()
        comparisons = []
        for target in targets:
            validation = self.tasks / f"{target}_val.jsonl"
            # Only prepare target validation, never use target data for selection.
            self.execute(self.command(str(validation)))
            signature = {} if args.dry_run else {
                "validation_sha256": fingerprint(validation),
                "seed": args.seed, "num_examples": args.num_examples,
                "max_new_tokens": args.max_new_tokens, "receiver": args.receiver_model,
            }
            baseline_root = root / "receiver-only" / target
            baseline = baseline_root / "half"
            shared = dict(DATASET=target, EVAL_N=args.num_examples or "",
                          RECV_MAX=args.max_new_tokens)
            self.cached_evaluation(
                self.command("_transfer-half", **shared, MODEL=args.receiver_model,
                             INFER_DATASET=target, SINGLE_OUT=baseline_root),
                baseline, signature,
            )
            output = root / f"step-{step:02d}-{history[-1]}" / target
            self.cached_evaluation(
                self.command("transfer", **shared, TRAIN_DATASET="+".join(history),
                             TRAINING_DATASETS=" ".join(history), INFER_DATASET=target,
                             TRANSFER_CKPT=checkpoint, TRANSFER_OUT=output),
                output, {**signature, "training_datasets": history,
                         "checkpoint_sha256": fingerprint(checkpoint) if not args.dry_run else None},
            )
            if not args.dry_run:
                comparison = compare_predictions(output, baseline)
                write_json(output / "comparison.json", comparison)
                write_summary(root)
                print(json.dumps(comparison, indent=2), flush=True)
                comparisons.append(comparison)
        return comparisons

    def check_stopping_rule(self, comparison: dict, checkpoint: Path, last: int) -> bool:
        args = self.args
        metric = args.stop_metric
        if metric not in comparison["metrics"]:
            raise ValueError(f"stopping metric {metric!r} is unavailable for this target")
        scores = comparison["metrics"][metric]
        improved = scores["delta_pp"] > 0
        step = comparison["transfer_step"]
        status = ("positive_gain" if improved else "continuing" if step < last
                  else "sources_exhausted" if last == len(args.datasets) else "step_limit_reached")
        report = {
            "status": status,
            "target_dataset": comparison["target_dataset"],
            "metric": metric,
            "rule": "transfer score strictly greater than Receiver-only half-context score",
            "planned_sources": args.datasets,
            "evaluated_through_step": step,
            "stopping_step": step if improved else None,
            "training_datasets": comparison["training_datasets"],
            "checkpoint": str(checkpoint.resolve()),
            "num_examples": comparison["num_examples"],
            **scores,
            "interpretation": "Exploratory target-based stage selection; not independent transfer validation.",
        }
        path = self.evaluation_root() / f"stopping_result-{args.targets[0]}-{metric}.json"
        write_json(path, report)
        print(f"Target {args.targets[0]} at step {step}: {metric} "
              f"{100 * scores['transfer']:.2f} vs half-context {100 * scores['receiver_half']:.2f} "
              f"({scores['delta_pp']:+.2f} pp); {status}. Result: {path}", flush=True)
        return improved

    def run(self) -> None:
        args = self.args
        last = args.step or len(args.datasets)
        if not args.dry_run:
            identity = self.identity()
            manifest = args.checkpoint_root / "sequence.json"
            if manifest.is_file() and read_json(manifest) != identity:
                raise ValueError("sequence configuration changed; reuse the original SEED/settings or a new checkpoint root")
            if not manifest.is_file():
                if args.evaluate_only:
                    raise ValueError("sequence has not been created; run transfer-sequence first")
                write_json(manifest, identity)
        checkpoint = args.source_checkpoint.resolve()
        for step in range(1, last + 1):
            history = args.datasets[:step]
            stage = args.checkpoint_root / f"step-{step:02d}-{history[-1]}"
            stage_file = stage / "stage.json"
            previous = checkpoint
            if step > 1:
                checkpoint = stage / ("translator.pt" if args.method == "xkv" else "fuser.pt")
            provenance = {
                "transfer_step": step, "training_datasets": history,
                "checkpoint": str(checkpoint.resolve()),
                "parent_checkpoint": str(previous.resolve()) if step > 1 else None,
                "optimizer_reset": step > 1,
            }
            if not args.dry_run and stage_file.is_file():
                saved = read_json(stage_file)
                if (any(saved.get(key) != value for key, value in provenance.items())
                        or saved.get("sha256") != fingerprint(checkpoint)):
                    raise ValueError(f"stage provenance or checkpoint changed: {stage}")
                print(f"Reusing stage {step}: {checkpoint}", flush=True)
            elif args.evaluate_only:
                if not args.dry_run:
                    raise ValueError(f"missing completed stage {step}; evaluation never trains")
            else:
                if step > 1:
                    if not args.dry_run and checkpoint.exists():
                        raise ValueError(f"untracked checkpoint exists; refusing to overwrite: {checkpoint}")
                    training_path = self.tasks / f"{history[-1]}_train.jsonl"
                    self.execute(self.command(
                        f"{args.method}-train", DATASET=history[-1], TRAIN=training_path,
                        VAL=self.tasks / f"{history[-1]}_val.jsonl",
                        INIT_CKPT=previous, TRAINING_DATASETS=" ".join(history),
                        **{"XKV_CKPT" if args.method == "xkv" else "CKPT": checkpoint},
                    ))
                if not args.dry_run:
                    write_json(stage_file, {**provenance, "sha256": fingerprint(checkpoint)})
            if not args.evaluate_only or step == last:
                comparisons = self.evaluate(step, checkpoint)
                if args.stop_on_gain and not args.evaluate_only:
                    if args.dry_run:
                        print(f"Stop here if {args.targets[0]} {args.stop_metric} beats Receiver-only half context; "
                              "otherwise continue to the next source.", flush=True)
                    elif self.check_stopping_rule(comparisons[0], checkpoint, last):
                        return


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("xkv", "lcfx"), default="xkv")
    parser.add_argument("--datasets", nargs="+", required=True, choices=DATASETS)
    parser.add_argument("--targets", nargs="+", default=None, choices=DATASETS)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--source-run-id", required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sharer-model", required=True)
    parser.add_argument("--receiver-model", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-gain", action="store_true",
                        help="Evaluate one fixed target and stop at the first strict improvement")
    parser.add_argument("--stop-metric", choices=("em", "f1", "accuracy"), default="em")
    parser.add_argument("--num-examples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--make-var", action="append", default=[])
    args = parser.parse_args(argv)
    args.explicit_targets = args.targets is not None
    args.targets = args.targets if args.explicit_targets else list(DATASETS)
    if len(set(args.datasets)) != len(args.datasets) or len(set(args.targets)) != len(args.targets):
        parser.error("datasets and targets must not repeat")
    if args.stop_on_gain:
        if len(args.targets) != 1:
            parser.error("--stop-on-gain requires exactly one fixed --targets dataset")
        if args.targets[0] in args.datasets:
            parser.error("the fixed target must be excluded from the entire source dataset plan")
    if args.step is not None and not 1 <= args.step <= len(args.datasets):
        parser.error("--step must select a stage in --datasets")
    if args.evaluate_only and args.step is None:
        parser.error("--evaluate-only requires --step")
    if args.evaluate_only and args.explicit_targets and set(args.targets) & set(args.datasets[:args.step]):
        parser.error("requested evaluation target has already been used for training")
    if args.num_examples is not None and args.num_examples < 1 or args.max_new_tokens < 1:
        parser.error("evaluation limits must be positive")
    allowed = re.compile(r"^[A-Z][A-Z0-9_]*=.*$", re.DOTALL)
    if any(not allowed.fullmatch(item) for item in args.make_var):
        parser.error("--make-var requires NAME=value")
    try:
        Sequence(args).run()
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"transfer sequence: {exc}\n")


if __name__ == "__main__":
    main()
