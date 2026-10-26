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

ITEM=compare: three pairwise method-difference rows (xkv-lcfx, xkv-t2t,
lcfx-t2t) across all datasets. Each method score is averaged over the complete
primary-model 3x3 grid; selected runs are averaged within each grid cell first.

In every mode, any missing method/checkpoint/result is left empty rather than
erroring. Every mode also appends a second table (same grid, same cell
labels) showing WHEN each result was produced -- the mtime of its result
file (metrics.json/latency.json), or the training run's "created_at" for
ITEM=params -- so stale results are easy to spot. All times are UTC.

With ``--runs 1:5`` (also ``--range``), files are read from
``results/runs/run-<id>`` and ``data/checkpoints/runs/run-<id>``. Score and
latency cells show the run-level mean in the same format as a single run;
the second table reports completed/requested runs and missing ids. Parameter
cells verify that all available repetitions agree.
"""

import argparse
import glob
import json
import os
import statistics
from datetime import datetime, timezone

try:
    from scripts._runs import parse_runs
except ModuleNotFoundError:  # direct ``python scripts/...`` execution
    from _runs import parse_runs

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
METHOD_COMPARISONS = [
    ("xkv-lcfx", "xkv", "lcfx"),
    ("xkv-t2t", "xkv", "t2t"),
    ("lcfx-t2t", "lcfx", "t2t"),
]
STUDY = "comparison"
RUN_IDS = []

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


def result_roots():
    if not RUN_IDS:
        return [RESULTS_DIR]
    return [
        RESULTS_DIR
        if run_id == "0"
        else os.path.join(RESULTS_DIR, "runs", f"run-{run_id}")
        for run_id in RUN_IDS
    ]


def checkpoint_roots():
    if not RUN_IDS:
        return [CHECKPOINTS_DIR]
    return [
        CHECKPOINTS_DIR
        if run_id == "0"
        else os.path.join(CHECKPOINTS_DIR, "runs", f"run-{run_id}")
        for run_id in RUN_IDS
    ]


def find_method_paths(roots, method, dataset, sharer, receiver, filename):
    paths = []
    for root in roots:
        base = os.path.join(root, method, dataset, slug(sharer), slug(receiver))
        paths.append(find_method_file(method, base, filename, receiver))
    return paths


def _existing(paths):
    return [path for path in paths if path is not None]


def validate_run_metadata(paths):
    """Reject aggregation of runs whose result configurations differ."""

    ignored = {
        "created_at",
        "run_id",
        "seed",
        "training_seed",
        "fuser_checkpoint",
        "translator_checkpoint",
    }
    # Results written before these fields were added used these same defaults.
    # Normalize only known schema additions; an explicitly different value or
    # any other configuration mismatch must still fail validation.
    legacy_defaults = {
        "dropout": 0.1,
        "receiver_pooling": True,
        "pool_spans": 5,
    }
    configs = []
    for path in _existing(paths):
        metadata_path = os.path.join(os.path.dirname(path), "run_metadata.json")
        if not os.path.isfile(metadata_path):
            continue
        with open(metadata_path) as handle:
            metadata = json.load(handle)
        config = {key: value for key, value in metadata.items() if key not in ignored}
        for key, value in legacy_defaults.items():
            config.setdefault(key, value)
        configs.append((config, path))
    if len(configs) < 2:
        return
    expected, expected_path = configs[0]
    for actual, actual_path in configs[1:]:
        if actual != expected:
            raise ValueError(
                "Refusing to aggregate incompatible runs:\n"
                f"  reference: {expected_path}\n"
                f"  mismatch:  {actual_path}"
            )


def find_metrics_paths(method, dataset, sharer, receiver):
    paths = find_method_paths(
        result_roots(), method, dataset, sharer, receiver, "metrics.json"
    )
    validate_run_metadata(paths)
    return paths


def find_metrics_path(method, dataset, sharer, receiver):
    return find_metrics_paths(method, dataset, sharer, receiver)[0]


def find_training_log_paths(method, dataset, sharer, receiver):
    return find_method_paths(
        checkpoint_roots(),
        method,
        dataset,
        sharer,
        receiver,
        "training_log.jsonl",
    )


def find_training_log_path(method, dataset, sharer, receiver):
    return find_training_log_paths(method, dataset, sharer, receiver)[0]


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


def mean_value(values):
    present = [value for value in values if value is not None]
    if not present:
        return None
    return statistics.fmean(present)


def sample_sd(values):
    present = [value for value in values if value is not None]
    return statistics.stdev(present) if len(present) > 1 else None


def coverage(n):
    requested = len(RUN_IDS)
    return f"n={n}/{requested}" if requested and n != requested else f"n={n}"


def fmt_pct_mean(values, bold):
    present = [value for value in values if value is not None]
    avg = mean_value(present)
    if avg is None:
        return ""
    return fmt_pct(avg, bold)


def cell_text(dataset, sharer, receiver):
    if RUN_IDS:
        results = []
        for method, label in METHODS:
            values = [
                load_em_f1(path) if path is not None else (None, None)
                for path in find_metrics_paths(method, dataset, sharer, receiver)
            ]
            em_values = [value[0] for value in values]
            f1_values = [value[1] for value in values]
            em_mean = mean_value(em_values)
            f1_mean = mean_value(f1_values)
            results.append((label, em_values, f1_values, em_mean, f1_mean))

        best_em = max(
            (em_mean for _, _, _, em_mean, _ in results if em_mean is not None),
            default=None,
        )
        best_f1 = max(
            (f1_mean for _, _, _, _, f1_mean in results if f1_mean is not None),
            default=None,
        )
        lines = []
        for label, em_values, f1_values, em_mean, f1_mean in results:
            parts = []
            if em_mean is not None:
                parts.append(fmt_pct_mean(em_values, em_mean == best_em))
            if f1_mean is not None:
                parts.append(fmt_pct_mean(f1_values, f1_mean == best_f1))
            lines.append(label if not parts else f"{label} {' / '.join(parts)}")
        return "<br>".join(lines)

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
    return find_latency_paths(method, dataset, sharer, receiver)[0]


def find_latency_paths(method, dataset, sharer, receiver):
    paths = find_method_paths(
        result_roots(), method, dataset, sharer, receiver, "latency.json"
    )
    validate_run_metadata(paths)
    return paths


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


def fmt_ms_mean(values, bold):
    present = [value for value in values if value is not None]
    avg = mean_value(present)
    if avg is None:
        return ""
    return fmt_ms(avg, bold)


def cell_latency_text(dataset, sharer, receiver):
    if RUN_IDS:
        results = []
        for method, label in METHODS:
            values = [
                load_avg_latency_ms(path) if path is not None else (None, None)
                for path in find_latency_paths(method, dataset, sharer, receiver)
            ]
            totals = [value[0] for value in values]
            components = [value[1] for value in values]
            total_mean = mean_value(totals)
            component_mean = mean_value(components)
            results.append(
                (method, label, totals, components, total_mean, component_mean)
            )
        best_total = min(
            (item[4] for item in results if item[4] is not None), default=None
        )
        trained = {name for name, _ in TRAINED_METHODS}
        best_component = min(
            (
                item[5]
                for item in results
                if item[0] in trained and item[5] is not None
            ),
            default=None,
        )
        lines = []
        for method, label, totals, components, total_mean, component_mean in results:
            if total_mean is None:
                lines.append(label)
                continue
            total_text = fmt_ms_mean(totals, total_mean == best_total)
            if method in trained and component_mean is not None:
                component_text = fmt_ms_mean(
                    components, component_mean == best_component
                )
                lines.append(f"{label} {component_text}/{total_text}")
            else:
                lines.append(f"{label} {total_text}")
        return "<br>".join(lines)

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
        paths = find_training_log_paths(method, dataset, sharer, receiver)
        values = [
            load_trainable_params(path) if path is not None else None for path in paths
        ]
        present = [value for value in values if value is not None]
        if not present:
            lines.append(label)
        elif len(set(present)) != 1:
            lines.append(f"{label} VARIES")
        else:
            lines.append(f"{label} {fmt_count(present[0])}")
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
        paths = find_metrics_paths(method, dataset, sharer, receiver)
        if RUN_IDS:
            present = sum(path is not None for path in paths)
            missing = [
                run_id for run_id, path in zip(RUN_IDS, paths) if path is None
            ]
            text = coverage(present)
            if missing:
                text += f"; missing {','.join(missing)}"
            lines.append(f"{label} {text}")
        else:
            t = file_mtime(paths[0])
            lines.append(f"{label} {fmt_time(t)}".rstrip())
    return "<br>".join(lines)


def cell_params_time_text(dataset, sharer, receiver):
    lines = []
    for method, label in TRAINED_METHODS:
        paths = find_training_log_paths(method, dataset, sharer, receiver)
        if RUN_IDS:
            present = sum(path is not None for path in paths)
            missing = [
                run_id for run_id, path in zip(RUN_IDS, paths) if path is None
            ]
            text = coverage(present)
            if missing:
                text += f"; missing {','.join(missing)}"
            lines.append(f"{label} {text}")
        else:
            t = load_training_start_time(paths[0])
            lines.append(f"{label} {fmt_time(t)}".rstrip())
    return "<br>".join(lines)


def cell_latency_time_text(dataset, sharer, receiver):
    lines = []
    for method, label in METHODS:
        paths = find_latency_paths(method, dataset, sharer, receiver)
        if RUN_IDS:
            present = sum(path is not None for path in paths)
            missing = [
                run_id for run_id, path in zip(RUN_IDS, paths) if path is None
            ]
            text = coverage(present)
            if missing:
                text += f"; missing {','.join(missing)}"
            lines.append(f"{label} {text}")
        else:
            t = file_mtime(paths[0])
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
    if RUN_IDS:
        caption += f" Selected replicates: {', '.join(f'run-{run_id}' for run_id in RUN_IDS)}."
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
    if RUN_IDS:
        metric_caption += "; each value is the mean across the selected runs"
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
    aggregate_note = (
        " Values are means across the selected runs." if RUN_IDS else ""
    )
    if STUDY == "ablation":
        return build_grid(
            models,
            lambda sharer, receiver: cell_latency_text(dataset, sharer, receiver),
            f"{dataset}: XKV ablation average latency per example",
            "Each entry is translator/total latency. F=Full XKV, −RP=no "
            "Receiver pooling, −RX=no raw Receiver-KV cross-attention, "
            "LCFP=original non-vectorized LCF-X pooling. The fastest component and "
            "total in each cell are bolded independently. Missing results are "
            f"left blank.{aggregate_note}",
        )
    return build_grid(
        models,
        lambda sharer, receiver: cell_latency_text(dataset, sharer, receiver),
        f"{dataset}: lcfx / xkv / t2t average latency per example",
        "Each cell: L=lcfx, X=xkv, T=t2t. L/X use fuser-or-translator/total latency; T shows total only. The fastest component and total in each cell are bolded independently. Missing results are left blank."
        + aggregate_note,
    )


def build_scores_time_table(dataset, models):
    if RUN_IDS:
        title = (
            f"{dataset}: XKV ablation score run coverage"
            if STUDY == "ablation"
            else f"{dataset}: score run coverage (lcfx / xkv / t2t)"
        )
        caption = (
            "Each entry shows completed/requested replicates and explicitly lists missing run ids."
        )
        return build_grid(
            models,
            lambda sharer, receiver: cell_scores_time_text(
                dataset, sharer, receiver
            ),
            title,
            caption,
        )
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
    if RUN_IDS:
        title = (
            f"{dataset}: XKV ablation parameter run coverage"
            if STUDY == "ablation"
            else f"{dataset}: parameter run coverage (lcfx / xkv)"
        )
        return build_grid(
            models,
            lambda sharer, receiver: cell_params_time_text(
                dataset, sharer, receiver
            ),
            title,
            "Each entry shows completed/requested training replicates and explicitly lists missing run ids.",
        )
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
    if RUN_IDS:
        title = (
            f"{dataset}: XKV ablation latency run coverage"
            if STUDY == "ablation"
            else f"{dataset}: latency run coverage (lcfx / xkv / t2t)"
        )
        return build_grid(
            models,
            lambda sharer, receiver: cell_latency_time_text(
                dataset, sharer, receiver
            ),
            title,
            "Each entry shows completed/requested replicates and explicitly lists missing run ids.",
        )
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


def load_result_seed(path):
    if path is None:
        return None
    metadata_path = os.path.join(os.path.dirname(path), "run_metadata.json")
    if not os.path.isfile(metadata_path):
        return None
    with open(metadata_path) as handle:
        metadata = json.load(handle)
    return metadata.get("training_seed", metadata.get("seed"))


def method_grid_stats(method, dataset, models):
    """Return method mean/SD over run-level grid means plus result coverage."""

    run_count = len(result_roots())
    em_by_run = [[] for _ in range(run_count)]
    f1_by_run = [[] for _ in range(run_count)]
    seeds_by_run = [set() for _ in range(run_count)]
    available = 0
    expected = run_count * len(models) ** 2
    for sharer in models:
        for receiver in models:
            paths = find_metrics_paths(method, dataset, sharer, receiver)
            for index, path in enumerate(paths):
                em, f1 = load_em_f1(path)
                if em is None or (
                    dataset not in CLASSIFICATION_DATASETS and f1 is None
                ):
                    continue
                available += 1
                em_by_run[index].append(em)
                if f1 is not None:
                    f1_by_run[index].append(f1)
                seed = load_result_seed(path)
                if seed is not None:
                    seeds_by_run[index].add(seed)

    runs = []
    for index in range(run_count):
        em = mean_value(em_by_run[index])
        if em is None:
            runs.append(None)
            continue
        f1 = mean_value(f1_by_run[index])
        seeds = seeds_by_run[index]
        seed = next(iter(seeds)) if len(seeds) == 1 else None
        runs.append(
            {
                "em": em,
                "f1": f1,
                "seed": seed,
                "results": len(em_by_run[index]),
            }
        )

    em_values = [run["em"] for run in runs if run is not None]
    f1_values = [
        run["f1"] for run in runs if run is not None and run["f1"] is not None
    ]
    return {
        "em": mean_value(em_values),
        "em_sd": sample_sd(em_values),
        "f1": mean_value(f1_values),
        "f1_sd": sample_sd(f1_values),
        "runs": runs,
        "n": len(em_values),
        "available": available,
        "expected": expected,
    }


def fmt_score_stats(mean, deviation, *, signed=False):
    if mean is None:
        return ""
    mean_text = f"{mean * 100:+.1f}" if signed else f"{mean * 100:.1f}"
    sd_text = "—" if deviation is None else f"{deviation * 100:.1f}"
    return f"{mean_text}±{sd_text}"


def method_stats_text(stats, classification):
    if stats["em"] is None:
        return "—"
    text = fmt_score_stats(stats["em"], stats["em_sd"])
    if not classification:
        text += "/" + fmt_score_stats(stats["f1"], stats["f1_sd"])
    selected_runs = len(result_roots())
    n_text = str(stats["n"])
    if stats["n"] != selected_runs:
        n_text += f"/{selected_runs}"
    notes = [f"n={n_text}"]
    if stats["available"] != stats["expected"]:
        notes.append(f"results={stats['available']}/{stats['expected']}")
    return f"{text} ({'; '.join(notes)})"


def paired_differences(first, second, metric):
    differences = []
    for first_run, second_run in zip(first["runs"], second["runs"]):
        if first_run is None or second_run is None:
            continue
        first_seed = first_run["seed"]
        second_seed = second_run["seed"]
        if first_seed is None or second_seed is None or first_seed != second_seed:
            continue
        first_value = first_run[metric]
        second_value = second_run[metric]
        if first_value is not None and second_value is not None:
            differences.append(first_value - second_value)
    return differences


def build_method_comparison_table(datasets, models):
    scores = {
        (method, dataset): method_grid_stats(method, dataset, models)
        for method, _ in METHODS
        for dataset in datasets
    }
    method_rows = [["Method score"] + datasets, ["---"] * (len(datasets) + 1)]
    for method, _ in METHODS:
        method_rows.append(
            [method]
            + [
                method_stats_text(
                    scores[(method, dataset)],
                    dataset in CLASSIFICATION_DATASETS,
                )
                for dataset in datasets
            ]
        )

    rows = [["Method difference"] + datasets, ["---"] * (len(datasets) + 1)]
    for label, first, second in METHOD_COMPARISONS:
        row = [label]
        for dataset in datasets:
            first_stats = scores[(first, dataset)]
            second_stats = scores[(second, dataset)]
            expected = first_stats["expected"]
            if first_stats["em"] is None or second_stats["em"] is None:
                text = "—"
            else:
                em_delta = first_stats["em"] - second_stats["em"]
                em_differences = paired_differences(
                    first_stats, second_stats, "em"
                )
                text = fmt_score_stats(
                    em_delta, sample_sd(em_differences), signed=True
                )
                if dataset in CLASSIFICATION_DATASETS:
                    pass
                elif first_stats["f1"] is None or second_stats["f1"] is None:
                    text = "—"
                else:
                    f1_delta = first_stats["f1"] - second_stats["f1"]
                    f1_differences = paired_differences(
                        first_stats, second_stats, "f1"
                    )
                    text += "/" + fmt_score_stats(
                        f1_delta, sample_sd(f1_differences), signed=True
                    )

            paired_n = len(paired_differences(first_stats, second_stats, "em"))
            selected_runs = len(result_roots())
            notes = [f"paired n={paired_n}"]
            if paired_n != selected_runs:
                notes[0] += f"/{selected_runs}"
            if (
                first_stats["available"] != expected
                or second_stats["available"] != expected
            ):
                text += (
                    f" ({'; '.join(notes)}; results={first} "
                    f"{first_stats['available']}/{expected}, {second} "
                    f"{second_stats['available']}/{expected})"
                )
            else:
                text += f" ({'; '.join(notes)})"
            row.append(text)

        rows.append(row)

    run_note = ""
    if RUN_IDS:
        run_note = (
            " Selected repetitions: "
            + ", ".join(f"run-{run_id}" for run_id in RUN_IDS)
            + "."
        )
    lines = [
        "# Method score mean and standard deviation across runs",
        "",
        f"Each run contributes one score obtained by averaging its available {len(models)}×{len(models)} model-pair cells. Values are mean±sample SD in percent; free-form datasets show EM/F1, while qasc and strategyqa show accuracy. `n` is the number of selected runs with at least one result. Incomplete result-file coverage is shown separately.{run_note}",
        "",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in method_rows)
    lines.extend(
        [
            "",
            "# Pairwise method score differences across datasets",
            "",
            f"Each value is first method minus second method, shown as mean difference±sample SD in percentage points. The center is the difference between the method means above; SD is calculated from same-seed, per-run differences. Free-form datasets show ΔEM/ΔF1; qasc and strategyqa show Δaccuracy. Positive values favor the first method. `paired n` reports matching-seed runs used for SD; — means one method has no result.{run_note}",
            "",
        ]
    )
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines) + "\n"


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
    # A dataset overview is a FULL-context reference. Never recursively fall
    # back to half/ when only a private-context baseline has completed.
    base = os.path.join(RESULTS_DIR, "single", dataset, slug(model))
    for path in (os.path.join(base, "full", "metrics.json"), os.path.join(base, "metrics.json")):
        if os.path.isfile(path):
            return path
    return None


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
    global METHODS, RUN_IDS, SELECTED_HP, STUDY, TRAINED_METHODS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset")
    parser.add_argument(
        "--study",
        choices=["comparison", "ablation"],
        default="comparison",
        help="Method set to render; ablation includes only Full XKV and its three variants.",
    )
    parser.add_argument(
        "--item",
        choices=["scores", "params", "latency", "datasets", "compare"],
        default="scores",
    )
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument(
        "--runs",
        "--range",
        dest="runs",
        default=None,
        help="Replicates to aggregate: 1:5, 1-5, 1,3,5, or [1,2,3]",
    )
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
    if args.runs:
        try:
            RUN_IDS = parse_runs(args.runs)
        except ValueError as exc:
            parser.error(str(exc))
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

    if args.item in ("scores", "params", "latency") and not args.dataset:
        parser.error(f"--dataset is required for --item {args.item}")
    if args.item == "datasets" and RUN_IDS:
        parser.error("--runs/--range is not supported with --item datasets")

    if args.item == "datasets":
        output = args.output or os.path.join(RESULTS_DIR, "comparison", "datasets.md")
    elif args.item == "compare":
        output_parts = [RESULTS_DIR, "comparison"]
        if RUN_IDS:
            output_parts.append("runs-" + "_".join(RUN_IDS))
        output = args.output or os.path.join(*output_parts, "compare.md")
    else:
        suffix = "" if args.item == "scores" else f"_{args.item}"
        output_parts = [RESULTS_DIR, "comparison"]
        if RUN_IDS:
            output_parts.append("runs-" + "_".join(RUN_IDS))
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
    elif args.item == "compare":
        table = build_method_comparison_table(ALL_DATASETS, args.models)
    else:
        table = build_datasets_table(ALL_DATASETS)

    with open(output, "w") as f:
        f.write(table)
    label = (
        args.dataset
        if args.item not in ("datasets", "compare")
        else "all datasets"
    )
    print(f"Wrote {args.item} {STUDY} table for '{label}' -> {output}")


if __name__ == "__main__":
    main()
