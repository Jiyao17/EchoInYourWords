#!/usr/bin/env python
"""Build comparison tables for the LCF-X/XKV/T2T experiments.

``--study ablation`` keeps the same Sharer x Receiver Markdown layouts but
shows only Full XKV and the no-Receiver-pooling, no-Receiver-cross-attention,
and exact LCF-X-pooling variants.

ITEM=scores (default): Sharer x Receiver grid of lcfx / xkv / t2t
EM+F1 for one DATASET, read from
results/<method>/<dataset>/<sharer>/<receiver>/metrics.json
(lcfx/xkv have extra hyperparameter subdirs).

ITEM=params: Sharer x Receiver grid of lcfx / xkv trainable
parameter counts (t2t trains nothing) for one DATASET, read from the "start" event of
data/checkpoints/<method>/<dataset>/<sharer>/<receiver>/training_log.jsonl
(lcfx/xkv again have extra hyperparameter subdirs).

ITEM=latency: Sharer x Receiver grid of lcfx / xkv / t2t average
per-example total latency (ms) for one DATASET, read from the "average.total_ms" field of
results/<method>/<dataset>/<sharer>/<receiver>/latency.json
(lcfx/xkv have extra hyperparameter subdirs; fastest of each cell is
bolded). LCF-X and XKV entries use fuser-or-translator/total format and bold
the fastest value of each independently; the component comes from
"average.fuser_ms" when available.

ITEM=datasets: one overview table across every known dataset (DATASET is
ignored) with train/val split sizes (counted from data/tasks/*.jsonl) and the
single-model oracle EM+F1 (one model sees ALL evidence, no split) for a
representative 1B/3B/8B model, read from results/single/<dataset>/<model>/metrics.json.

In every mode, any missing method/checkpoint/result is left empty rather than
erroring. Every mode also appends a second table (same grid, same cell
labels) showing WHEN each result was produced -- the mtime of its result
file (metrics.json/latency.json), or the training run's "created_at" for
ITEM=params -- so stale results are easy to spot. All times are UTC.
"""

import argparse
import glob
import json
import os
from datetime import datetime, timezone

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
CHECKPOINTS_DIR = os.path.join(ROOT_DIR, "data", "checkpoints")
TASKS_DIR = os.path.join(ROOT_DIR, "data", "tasks")

# Mirrors PRIMARY_MODELS in the Makefile (3x3 grid by default).
PRIMARY_MODELS = [
    "Qwen/Qwen3-0.6B",
    "unsloth/gemma-3-1b-it",
    "unsloth/Llama-3.2-3B-Instruct",
]
# Opt-in large tier, appended to the grid by --extra-models (Makefile:
# EXTRA_MODELS=TRUE) -- mirrors EXTRA_MODELS in scripts/run_sweep.py.
EXTRA_MODELS = [
    "mistralai/Ministral-8B-Instruct-2410",
]

# Mirrors ALL_DATASETS in the Makefile.
ALL_DATASETS = [
    "hotpotqa_bridge",
    "musique",
    "qasc",
    "2wikimultihop",
    "ropes",
    "strategyqa",
]
CLASSIFICATION_DATASETS = {"qasc", "strategyqa"}

# Representative sizes drawn from PRIMARY_MODELS, used for the ITEM=datasets
# "typical score" columns (single-model oracle: one model, all evidence).
SIZE_MODELS = [
    ("1B", "unsloth/gemma-3-1b-it"),
    ("3B", "unsloth/Llama-3.2-3B-Instruct"),
    ("8B", "mistralai/Ministral-8B-Instruct-2410"),
]

METHODS = [
    ("lcfx", "L"),
    ("xkv", "X"),
    ("t2t", "T"),
]
TRAINED_METHODS = [
    ("lcfx", "L"),
    ("xkv", "X"),
]
ABLATION_METHODS = [
    ("xkv", "F"),
    ("xkv-no-receiver-pooling", "−RP"),
    ("xkv-no-receiver-xattn", "−RX"),
    ("xkv-lcfx-pooling", "LCFP"),
]
STUDY = "comparison"

# LCF-X and XKV checkpoint/result dirs nest a hyperparameter path under the
# pair directory. Candidates are tried in order — current defaults first, then
# legacy layouts — for a stable pick when several sweeps exist side by side.
PREFERRED_HP = {
    "lcfx": ["sp5"],
    "xkv": [
        "dh8_n1_h8_do0p1_k1_v11",
        "dh8_n1_h8_do0p1_k2_v11",
        "dh8_n1_h8_do0p1_k3_v11",
        "dh8_n2_h8_do0p1_k2_v11",
        "dh8_n3_h8_do0p1_k3_v11",
        "dh8_n1_h8_do0p1_k2_v10",
        "dh8_n1_h8_do0p1_k2_v9",
        "sp5_dh8_n1_h8_do0p1_p2_v8",
        "sp5_dh8_n1_h8_do0p1_p2_v7",
        "sp5_dh8_n1_h8_do0p1_p2_v6",
        "sp5_d256_n1_h8_do0p1_p2_v5",
        "sp5_d256_n1_h8_do0p1_v4",
        "sp5_d256_n1_h8_do0p1_v3",
        "sp5_d256_n1_h8_do0p1_v2",
        "sp5_d128_n1_h8_do0p1",
    ],
}

# Populated by the --*-hp arguments used by `make visual`.  An explicitly
# selected layout is strict: if that run is absent, leave the cell blank rather
# than silently displaying a result from a different experiment.
SELECTED_HP = {}


def slug(model_id):
    return model_id.replace("/", "--")


def short_name(model_id):
    return model_id.split("/", 1)[1] if "/" in model_id else model_id


def find_nested_file(base, filename, preferred_hp=None):
    # Check the preferred_hp candidates (the Makefile's current hyperparam
    # layout first, then legacy default layouts) before the bare/direct path:
    # some cells carry a leftover flat metrics.json/latency.json/
    # training_log.jsonl from before hyperparams were baked into the path,
    # which would otherwise permanently shadow the current result.
    if preferred_hp:
        candidates = [preferred_hp] if isinstance(preferred_hp, str) else preferred_hp
        for hp in candidates:
            preferred = os.path.join(base, hp, filename)
            if os.path.isfile(preferred):
                return preferred
    direct = os.path.join(base, filename)
    if os.path.isfile(direct):
        return direct
    # With no exact match to the Makefile defaults, fall back to any run found.
    # WARNING: this can pick up stale dirs from older hyperparam layouts; the
    # alphabetically-first match wins, so prefer keeping checkpoint trees clean.
    matches = sorted(glob.glob(os.path.join(base, "**", filename), recursive=True))
    return matches[0] if matches else None


def default_xkv_receiver_k(receiver):
    tokens = receiver.lower().replace("/", "-").replace("_", "-").split("-")
    for size, slots in (("0.6b", 1), ("1b", 2), ("3b", 3), ("4b", 3), ("8b", 4)):
        if size in tokens:
            return slots
    return 2


def find_method_file(method, base, filename, receiver):
    if method in SELECTED_HP:
        selected = SELECTED_HP[method].replace(
            "{receiver_k}", str(default_xkv_receiver_k(receiver))
        )
        path = os.path.join(base, selected, filename)
        return path if os.path.isfile(path) else None
    preferred = PREFERRED_HP.get(method)
    if method.startswith("xkv"):
        if method == "xkv-lcfx-pooling":
            receiver_default = "dh8_n1_h8_do0p1_k1_sp5_v12"
        elif method == "xkv":
            receiver_default = (
                f"dh8_n1_h8_do0p1_k{default_xkv_receiver_k(receiver)}_v11"
            )
        else:
            receiver_default = (
                f"dh8_n1_h8_do0p1_k{default_xkv_receiver_k(receiver)}_v11"
            )
        preferred = [receiver_default] + [
            candidate for candidate in preferred or [] if candidate != receiver_default
        ]
    return find_nested_file(base, filename, preferred_hp=preferred)


def find_metrics_path(method, dataset, sharer, receiver):
    base = os.path.join(RESULTS_DIR, method, dataset, slug(sharer), slug(receiver))
    return find_method_file(method, base, "metrics.json", receiver)


def find_training_log_path(method, dataset, sharer, receiver):
    base = os.path.join(CHECKPOINTS_DIR, method, dataset, slug(sharer), slug(receiver))
    return find_method_file(method, base, "training_log.jsonl", receiver)


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
            pair = fmt_pair(
                em,
                f1,
                em is not None and em == best_em,
                f1 is not None and f1 == best_f1,
            )
            lines.append(f"{label} {pair}")
    return "<br>".join(lines)


def find_latency_path(method, dataset, sharer, receiver):
    base = os.path.join(RESULTS_DIR, method, dataset, slug(sharer), slug(receiver))
    return find_method_file(method, base, "latency.json", receiver)


def load_avg_latency_ms(path):
    if path is None:
        return None, None
    with open(path) as f:
        data = json.load(f)
    average = data.get("average", {})
    return average.get("total_ms"), average.get("fuser_ms")


def load_avg_total_ms(path):
    """Return the average end-to-end latency (legacy helper)."""
    return load_avg_latency_ms(path)[0]


def fmt_ms(value, bold):
    if value is None:
        return ""
    text = f"{value / 1000:.2f}s" if value >= 1000 else f"{value:.0f}ms"
    return f"**{text}**" if bold else text


def cell_latency_text(dataset, sharer, receiver):
    results = []
    for method, label in METHODS:
        path = find_latency_path(method, dataset, sharer, receiver)
        total_ms, component_ms = load_avg_latency_ms(path)
        results.append((method, label, total_ms, component_ms))

    best_total = min(
        (total for _, _, total, _ in results if total is not None), default=None
    )
    best_component = min(
        (
            component
            for method, _, _, component in results
            if method in {name for name, _ in TRAINED_METHODS} and component is not None
        ),
        default=None,
    )

    lines = []
    for method, label, total_ms, component_ms in results:
        if total_ms is None:
            lines.append(label)
        else:
            total_text = fmt_ms(total_ms, total_ms == best_total)
            if (
                method in {name for name, _ in TRAINED_METHODS}
                and component_ms is not None
            ):
                component_text = fmt_ms(component_ms, component_ms == best_component)
                text = f"{label} {component_text}/{total_text}"
            else:
                text = f"{label} {total_text}"
            lines.append(text)
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
    for method, label in TRAINED_METHODS:
        path = find_training_log_path(method, dataset, sharer, receiver)
        params = load_trainable_params(path)
        lines.append(label if params is None else f"{label} {fmt_count(params)}")
    return "<br>".join(lines)


def file_mtime(path):
    if path is None or not os.path.isfile(path):
        return None
    return datetime.fromtimestamp(os.path.getmtime(path), tz=timezone.utc)


def load_training_start_time(path):
    if path is None:
        return None
    with open(path) as f:
        for line in f:
            record = json.loads(line)
            if "created_at" in record:
                return datetime.fromisoformat(record["created_at"])
    return None


def fmt_time(dt):
    return "" if dt is None else dt.strftime("%Y-%m-%d %H:%M")


def cell_scores_time_text(dataset, sharer, receiver):
    lines = []
    for method, label in METHODS:
        t = file_mtime(find_metrics_path(method, dataset, sharer, receiver))
        lines.append(f"{label} {fmt_time(t)}".rstrip())
    return "<br>".join(lines)


def cell_params_time_text(dataset, sharer, receiver):
    lines = []
    for method, label in TRAINED_METHODS:
        t = load_training_start_time(
            find_training_log_path(method, dataset, sharer, receiver)
        )
        lines.append(f"{label} {fmt_time(t)}".rstrip())
    return "<br>".join(lines)


def cell_latency_time_text(dataset, sharer, receiver):
    lines = []
    for method, label in METHODS:
        t = file_mtime(find_latency_path(method, dataset, sharer, receiver))
        lines.append(f"{label} {fmt_time(t)}".rstrip())
    return "<br>".join(lines)


def build_grid(models, cell_fn, title, caption):
    header = ["Sharer \\ Receiver"] + [short_name(m) for m in models]
    rows = [header, ["---"] * len(header)]
    for sharer in models:
        row = [short_name(sharer)]
        for receiver in models:
            row.append(cell_fn(sharer, receiver))
        rows.append(row)
    selected_configs = [("XKV", SELECTED_HP.get("xkv"))]
    if STUDY == "ablation":
        selected_configs = [
            ("Full XKV", SELECTED_HP.get("xkv")),
            (
                "Receiver-path ablations",
                SELECTED_HP.get("xkv-no-receiver-pooling"),
            ),
            (
                "exact LCF-X-pooling ablation",
                SELECTED_HP.get("xkv-lcfx-pooling"),
            ),
        ]
    for method_name, config in selected_configs:
        if config:
            caption += f" Selected {method_name} configuration: `{config}`."
    lines = [f"# {title}", "", caption, ""]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def build_scores_table(dataset, models):
    classification = dataset in CLASSIFICATION_DATASETS
    metric_label = "accuracy" if classification else "EM/F1"
    metric_caption = (
        "accuracy (best in each cell is bolded)"
        if classification
        else "EM/F1 (best of each in a cell is bolded)"
    )
    if STUDY == "ablation":
        return build_grid(
            models,
            lambda sharer, receiver: cell_text(dataset, sharer, receiver),
            f"{dataset}: XKV ablation study ({metric_label}, %)",
            "Each cell: F=Full XKV, −RP=no Receiver pooling, "
            "−RX=no raw Receiver-KV cross-attention, LCFP=original "
            f"non-vectorized LCF-X pooling. Values are {metric_caption}. Missing results "
            "are left blank.",
        )
    return build_grid(
        models,
        lambda sharer, receiver: cell_text(dataset, sharer, receiver),
        f"{dataset}: lcfx / xkv / t2t ({metric_label}, %)",
        "Each cell: L=lcfx, X=xkv, T=t2t, "
        f"values are {metric_caption}. Missing results are left blank.",
    )


def build_params_table(dataset, models):
    if STUDY == "ablation":
        return build_grid(
            models,
            lambda sharer, receiver: cell_params_text(dataset, sharer, receiver),
            f"{dataset}: XKV ablation trainable parameters",
            "Each cell: F=Full XKV, −RP=no Receiver pooling, "
            "−RX=no raw Receiver-KV cross-attention, LCFP=original "
            "non-vectorized LCF-X pooling. Missing checkpoints are left blank.",
        )
    return build_grid(
        models,
        lambda sharer, receiver: cell_params_text(dataset, sharer, receiver),
        f"{dataset}: trainable parameters (lcfx / xkv)",
        "Each cell: L=lcfx, X=xkv trainable parameter count (t2t trains nothing). Missing checkpoints are left blank.",
    )


def build_latency_table(dataset, models):
    if STUDY == "ablation":
        return build_grid(
            models,
            lambda sharer, receiver: cell_latency_text(dataset, sharer, receiver),
            f"{dataset}: XKV ablation average latency per example",
            "Each entry is translator/total latency. F=Full XKV, −RP=no "
            "Receiver pooling, −RX=no raw Receiver-KV cross-attention, "
            "LCFP=original non-vectorized LCF-X pooling. The fastest component and "
            "total in each cell are bolded independently. Missing results are "
            "left blank.",
        )
    return build_grid(
        models,
        lambda sharer, receiver: cell_latency_text(dataset, sharer, receiver),
        f"{dataset}: lcfx / xkv / t2t average latency per example",
        "Each cell: L=lcfx, X=xkv, T=t2t. L/X use fuser-or-translator/total latency; T shows total only. The fastest component and total in each cell are bolded independently. Missing results are left blank.",
    )


def build_scores_time_table(dataset, models):
    if STUDY == "ablation":
        return build_grid(
            models,
            lambda sharer, receiver: cell_scores_time_text(dataset, sharer, receiver),
            f"{dataset}: XKV ablation score result times",
            "Each cell uses F/−RP/−RX/LCFP labels and shows the UTC mtime of "
            "the metrics.json backing the score table above. Missing results "
            "are left blank.",
        )
    return build_grid(
        models,
        lambda sharer, receiver: cell_scores_time_text(dataset, sharer, receiver),
        f"{dataset}: time of result (lcfx / xkv / t2t)",
        "Each cell: L=lcfx, X=xkv, T=t2t, UTC mtime of the metrics.json backing the scores table above. Missing results are left blank.",
    )


def build_params_time_table(dataset, models):
    if STUDY == "ablation":
        return build_grid(
            models,
            lambda sharer, receiver: cell_params_time_text(dataset, sharer, receiver),
            f"{dataset}: XKV ablation parameter result times",
            "Each cell uses F/−RP/−RX/LCFP labels and shows the UTC training "
            "start time backing the parameter table above. Missing checkpoints "
            "are left blank.",
        )
    return build_grid(
        models,
        lambda sharer, receiver: cell_params_time_text(dataset, sharer, receiver),
        f"{dataset}: time of result (lcfx / xkv)",
        "Each cell: L=lcfx, X=xkv, UTC start time (created_at) of the training run backing the params table above. Missing checkpoints are left blank.",
    )


def build_latency_time_table(dataset, models):
    if STUDY == "ablation":
        return build_grid(
            models,
            lambda sharer, receiver: cell_latency_time_text(dataset, sharer, receiver),
            f"{dataset}: XKV ablation latency result times",
            "Each cell uses F/−RP/−RX/LCFP labels and shows the UTC mtime of "
            "the latency.json backing the latency table above. Missing results "
            "are left blank.",
        )
    return build_grid(
        models,
        lambda sharer, receiver: cell_latency_time_text(dataset, sharer, receiver),
        f"{dataset}: time of result (lcfx / xkv / t2t)",
        "Each cell: L=lcfx, X=xkv, T=t2t, UTC mtime of the latency.json backing the latency table above. Missing results are left blank.",
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
    return find_nested_file(
        os.path.join(RESULTS_DIR, "single", dataset, slug(model)), "metrics.json"
    )


def build_datasets_table(datasets):
    header = ["Dataset", "Train", "Val"] + [
        f"{size} oracle EM/F1" for size, _ in SIZE_MODELS
    ]
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

    time_header = ["Dataset"] + [
        f"{size} oracle result time" for size, _ in SIZE_MODELS
    ]
    time_rows = [time_header, ["---"] * len(time_header)]
    for dataset in datasets:
        row = [dataset]
        for _, model in SIZE_MODELS:
            row.append(fmt_time(file_mtime(find_single_metrics_path(dataset, model))))
        time_rows.append(row)
    lines += [
        "",
        "# Datasets: time of result (oracle EM/F1)",
        "",
        "UTC mtime of the metrics.json backing each oracle column above. Missing results are left blank.",
        "",
    ]
    for row in time_rows:
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines) + "\n"


def main():
    global METHODS, SELECTED_HP, STUDY, TRAINED_METHODS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset")
    parser.add_argument(
        "--study",
        choices=["comparison", "ablation"],
        default="comparison",
        help="Method set to render; ablation includes only Full XKV and its three variants.",
    )
    parser.add_argument(
        "--item", choices=["scores", "params", "latency", "datasets"], default="scores"
    )
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument(
        "--extra-models",
        action="store_true",
        help="Also include the opt-in large tier (Ministral-8B) in the grid, matching `make exp-all`",
    )
    parser.add_argument(
        "--lcfx-hp", help="Exact LCF-X hyperparameter subdirectory to display"
    )
    parser.add_argument(
        "--xkv-hp", help="Exact XKV hyperparameter subdirectory to display"
    )
    parser.add_argument(
        "--xkv-ablation-hp",
        help="Exact hyperparameter subdirectory for Receiver-path XKV ablations",
    )
    parser.add_argument(
        "--xkv-lcfx-pooling-hp",
        help="Exact hyperparameter subdirectory for the LCF-X-pooling ablation",
    )
    parser.add_argument("--output")
    args = parser.parse_args()
    STUDY = args.study
    if STUDY == "ablation":
        METHODS = list(ABLATION_METHODS)
        TRAINED_METHODS = list(ABLATION_METHODS)

    if args.models is None:
        args.models = PRIMARY_MODELS + (EXTRA_MODELS if args.extra_models else [])
    elif args.extra_models:
        args.models = args.models + [m for m in EXTRA_MODELS if m not in args.models]

    SELECTED_HP = {
        method: hp
        for method, hp in (
            ("lcfx", args.lcfx_hp),
            ("xkv", args.xkv_hp),
            (
                "xkv-no-receiver-pooling",
                args.xkv_ablation_hp if STUDY == "ablation" else None,
            ),
            (
                "xkv-no-receiver-xattn",
                args.xkv_ablation_hp if STUDY == "ablation" else None,
            ),
            ("xkv-lcfx-pooling", args.xkv_lcfx_pooling_hp),
        )
        if hp is not None
    }

    if args.item in ("scores", "params") and not args.dataset:
        parser.error(f"--dataset is required for --item {args.item}")

    if args.item == "datasets":
        output = args.output or os.path.join(RESULTS_DIR, "comparison", "datasets.md")
    else:
        suffix = "" if args.item == "scores" else f"_{args.item}"
        output_parts = [RESULTS_DIR, "comparison"]
        if STUDY == "ablation":
            output_parts.append("ablation")
        output = args.output or os.path.join(
            *output_parts, f"{args.dataset}{suffix}.md"
        )
    os.makedirs(os.path.dirname(output), exist_ok=True)

    if args.item == "scores":
        table = build_scores_table(args.dataset, args.models)
        table += "\n" + build_scores_time_table(args.dataset, args.models)
    elif args.item == "params":
        table = build_params_table(args.dataset, args.models)
        table += "\n" + build_params_time_table(args.dataset, args.models)
    elif args.item == "latency":
        table = build_latency_table(args.dataset, args.models)
        table += "\n" + build_latency_time_table(args.dataset, args.models)
    else:
        table = build_datasets_table(ALL_DATASETS)

    with open(output, "w") as f:
        f.write(table)
    label = args.dataset if args.item != "datasets" else "all datasets"
    print(f"Wrote {args.item} {STUDY} table for '{label}' -> {output}")


if __name__ == "__main__":
    main()
