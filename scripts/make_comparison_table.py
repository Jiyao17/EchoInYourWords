#!/usr/bin/env python
"""Build comparison tables for the EIYW/LCF-X/T2T experiments.

ITEM=scores (default): Sharer x Receiver grid of lcfx / eiyw / t2t EM+F1 for
one DATASET, read from results/<method>/<dataset>/<sharer>/<receiver>/metrics.json
(eiyw has extra k*/b*/h*/d* subdirs).

ITEM=params: Sharer x Receiver grid of lcfx / eiyw trainable parameter counts
(t2t trains nothing) for one DATASET, read from the "start" event of
data/checkpoints/<method>/<dataset>/<sharer>/<receiver>/training_log.jsonl
(eiyw again has extra k*/b*/h*/d* subdirs).

ITEM=latency: Sharer x Receiver grid of lcfx / eiyw / t2t average per-example
total latency (ms) for one DATASET, read from the "average.total_ms" field of
results/<method>/<dataset>/<sharer>/<receiver>/latency.json (eiyw has extra
k*/b*/h*/d* subdirs; fastest of each cell is bolded).

ITEM=datasets: one overview table across every known dataset (DATASET is
ignored) with train/val split sizes (counted from data/tasks/*.jsonl) and the
single-model oracle EM+F1 (one model sees ALL evidence, no split) for a
representative 1B/4B/8B model, read from results/single/<dataset>/<model>/metrics.json.

In every mode, any missing method/checkpoint/result is left empty rather than
erroring.
"""
import argparse
import glob
import json
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
CHECKPOINTS_DIR = os.path.join(ROOT_DIR, "data", "checkpoints")
TASKS_DIR = os.path.join(ROOT_DIR, "data", "tasks")

# Mirrors PRIMARY_MODELS in the Makefile.
PRIMARY_MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen3-0.6B",
    "unsloth/Llama-3.2-1B-Instruct",
    "unsloth/gemma-3-4b-it",
    "Qwen/Qwen3-8B",
]

# Mirrors ALL_DATASETS in the Makefile.
ALL_DATASETS = ["hotpotqa_bridge", "musique", "qasc", "2wikimultihop", "ropes", "strategyqa"]

# Representative sizes drawn from PRIMARY_MODELS, used for the ITEM=datasets
# "typical score" columns (single-model oracle: one model, all evidence).
SIZE_MODELS = [
    ("1B", "unsloth/Llama-3.2-1B-Instruct"),
    ("4B", "unsloth/gemma-3-4b-it"),
    ("8B", "Qwen/Qwen3-8B"),
]

METHODS = [("lcfx", "L"), ("eiyw", "E"), ("t2t", "T")]

# EIYW's checkpoint/result dir nests an extra k<SOFT_TOKENS>/b<EIYW_BOTTLENECK>/
# s1h<STAGE1_NUM_HEADS>/h<STAGE2_NUM_HEADS>/d<STAGE2_HEAD_DIM>/sp<EIYW_SPANS>
# hyperparam path (and LCF-X nests sp<SPANS>); try the Makefile's defaults
# first for a stable pick when several hyperparam sweeps exist side by side.
EIYW_DEFAULT_HP = "k256/b256/s1h8/h8/d256/sp32"


def slug(model_id):
    return model_id.replace("/", "--")


def short_name(model_id):
    return model_id.split("/", 1)[1] if "/" in model_id else model_id


def find_nested_file(base, filename, preferred_hp=EIYW_DEFAULT_HP):
    direct = os.path.join(base, filename)
    if os.path.isfile(direct):
        return direct
    preferred = os.path.join(base, preferred_hp, filename)
    if os.path.isfile(preferred):
        return preferred
    # eiyw nests an extra k<K>/b<B>/h<H>/d<D> hyperparam path under the pair dir;
    # with no exact match to the Makefile defaults, fall back to any run found.
    matches = sorted(glob.glob(os.path.join(base, "**", filename), recursive=True))
    return matches[0] if matches else None


def find_metrics_path(method, dataset, sharer, receiver):
    base = os.path.join(RESULTS_DIR, method, dataset, slug(sharer), slug(receiver))
    return find_nested_file(base, "metrics.json")


def find_training_log_path(method, dataset, sharer, receiver):
    base = os.path.join(CHECKPOINTS_DIR, method, dataset, slug(sharer), slug(receiver))
    return find_nested_file(base, "training_log.jsonl")


def load_em_f1(path):
    if path is None:
        return None, None
    with open(path) as f:
        data = json.load(f)
    scores = data.get("scores")
    if scores:
        return scores.get("em"), scores.get("f1")
    # Classification-style datasets (e.g. qasc, strategyqa) report a bare
    # "accuracy" instead of EM/F1 — treat it as the EM slot, no F1.
    if "accuracy" in data:
        return data["accuracy"], None
    return None, None


def fmt_pct(value, bold):
    if value is None:
        return ""
    text = f"{value * 100:.1f}"
    return f"**{text}**" if bold else text


def fmt_pair(em, f1, bold_em, bold_f1):
    parts = [fmt_pct(v, b) for v, b in ((em, bold_em), (f1, bold_f1)) if v is not None]
    return "/".join(parts)


def cell_text(dataset, sharer, receiver):
    results = []
    for method, label in METHODS:
        path = find_metrics_path(method, dataset, sharer, receiver)
        em, f1 = load_em_f1(path)
        results.append((label, em, f1))

    best_em = max((em for _, em, _ in results if em is not None), default=None)
    best_f1 = max((f1 for _, _, f1 in results if f1 is not None), default=None)

    lines = []
    for label, em, f1 in results:
        if em is None and f1 is None:
            lines.append(label)
        else:
            pair = fmt_pair(em, f1, em is not None and em == best_em, f1 is not None and f1 == best_f1)
            lines.append(f"{label} {pair}")
    return "<br>".join(lines)


def find_latency_path(method, dataset, sharer, receiver):
    base = os.path.join(RESULTS_DIR, method, dataset, slug(sharer), slug(receiver))
    return find_nested_file(base, "latency.json")


def load_avg_total_ms(path):
    if path is None:
        return None
    with open(path) as f:
        data = json.load(f)
    return data.get("average", {}).get("total_ms")


def fmt_ms(value, bold):
    if value is None:
        return ""
    text = f"{value / 1000:.2f}s" if value >= 1000 else f"{value:.0f}ms"
    return f"**{text}**" if bold else text


def cell_latency_text(dataset, sharer, receiver):
    results = []
    for method, label in METHODS:
        path = find_latency_path(method, dataset, sharer, receiver)
        results.append((label, load_avg_total_ms(path)))

    best = min((ms for _, ms in results if ms is not None), default=None)

    lines = []
    for label, ms in results:
        if ms is None:
            lines.append(label)
        else:
            lines.append(f"{label} {fmt_ms(ms, ms == best)}")
    return "<br>".join(lines)


def load_trainable_params(path):
    if path is None:
        return None
    with open(path) as f:
        for line in f:
            record = json.loads(line)
            if "trainable_params" in record:
                return record["trainable_params"]
    return None


def fmt_count(n):
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    if n >= 1e6:
        return f"{n / 1e6:.1f}M"
    if n >= 1e3:
        return f"{n / 1e3:.1f}K"
    return str(n)


def cell_params_text(dataset, sharer, receiver):
    lines = []
    for method, label in [("lcfx", "L"), ("eiyw", "E")]:
        path = find_training_log_path(method, dataset, sharer, receiver)
        params = load_trainable_params(path)
        lines.append(label if params is None else f"{label} {fmt_count(params)}")
    return "<br>".join(lines)


def build_grid(models, cell_fn, title, caption):
    header = ["Sharer \\ Receiver"] + [short_name(m) for m in models]
    rows = [header, ["---"] * len(header)]
    for sharer in models:
        row = [short_name(sharer)]
        for receiver in models:
            row.append(cell_fn(sharer, receiver))
        rows.append(row)
    lines = [f"# {title}", "", caption, ""]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def build_scores_table(dataset, models):
    return build_grid(
        models,
        lambda sharer, receiver: cell_text(dataset, sharer, receiver),
        f"{dataset}: lcfx / eiyw / t2t (EM/F1, %)",
        "Each cell: L=lcfx, E=eiyw, T=t2t, values are EM/F1 (best of each in a cell is bolded). Missing results are left blank.",
    )


def build_params_table(dataset, models):
    return build_grid(
        models,
        lambda sharer, receiver: cell_params_text(dataset, sharer, receiver),
        f"{dataset}: trainable parameters (lcfx / eiyw)",
        "Each cell: L=lcfx, E=eiyw trainable parameter count (t2t trains nothing). Missing checkpoints are left blank.",
    )


def build_latency_table(dataset, models):
    return build_grid(
        models,
        lambda sharer, receiver: cell_latency_text(dataset, sharer, receiver),
        f"{dataset}: lcfx / eiyw / t2t average latency per example",
        "Each cell: L=lcfx, E=eiyw, T=t2t, average total latency per example (fastest of each in a cell is bolded). Missing results are left blank.",
    )


def count_lines(path):
    if not os.path.isfile(path):
        return None
    count = 0
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            count += chunk.count(b"\n")
    return count


def fmt_int(n):
    return "" if n is None else f"{n:,}"


def find_single_metrics_path(dataset, model):
    return find_nested_file(os.path.join(RESULTS_DIR, "single", dataset, slug(model)), "metrics.json")


def build_datasets_table(datasets):
    header = ["Dataset", "Train", "Val"] + [f"{size} oracle EM/F1" for size, _ in SIZE_MODELS]
    rows = [header, ["---"] * len(header)]
    for dataset in datasets:
        train_n = count_lines(os.path.join(TASKS_DIR, f"{dataset}_train.jsonl"))
        val_n = count_lines(os.path.join(TASKS_DIR, f"{dataset}_val.jsonl"))
        row = [dataset, fmt_int(train_n), fmt_int(val_n)]
        for _, model in SIZE_MODELS:
            em, f1 = load_em_f1(find_single_metrics_path(dataset, model))
            row.append(fmt_pair(em, f1, False, False))
        rows.append(row)
    lines = [
        "# Datasets: size and single-model oracle EM/F1 (%)",
        "",
        "Train/Val are example counts. Oracle = one model answers with ALL evidence combined "
        "(no Sharer/Receiver split) — a rough per-size ceiling for the split-input methods above. "
        "Classification-style datasets (qasc, strategyqa) report a single accuracy number instead of EM/F1. "
        f"Model per size: {', '.join(f'{size}={short_name(model)}' for size, model in SIZE_MODELS)}. "
        "Missing results are left blank.",
        "",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset")
    parser.add_argument("--item", choices=["scores", "params", "latency", "datasets"], default="scores")
    parser.add_argument("--models", nargs="+", default=PRIMARY_MODELS)
    parser.add_argument("--output")
    args = parser.parse_args()

    if args.item in ("scores", "params") and not args.dataset:
        parser.error(f"--dataset is required for --item {args.item}")

    if args.item == "datasets":
        output = args.output or os.path.join(RESULTS_DIR, "comparison", "datasets.md")
    else:
        suffix = "" if args.item == "scores" else f"_{args.item}"
        output = args.output or os.path.join(RESULTS_DIR, "comparison", f"{args.dataset}{suffix}.md")
    os.makedirs(os.path.dirname(output), exist_ok=True)

    if args.item == "scores":
        table = build_scores_table(args.dataset, args.models)
    elif args.item == "params":
        table = build_params_table(args.dataset, args.models)
    elif args.item == "latency":
        table = build_latency_table(args.dataset, args.models)
    else:
        table = build_datasets_table(ALL_DATASETS)

    with open(output, "w") as f:
        f.write(table)
    label = args.dataset if args.item != "datasets" else "all datasets"
    print(f"Wrote {args.item} comparison table for '{label}' -> {output}")


if __name__ == "__main__":
    main()
