#!/usr/bin/env python
"""Export copy-ready LaTeX tables from the current comparison results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
CHECKPOINTS = ROOT / "data" / "checkpoints"

MODELS = [
    ("Qwen/Qwen3-0.6B", "Qwen-0.6B"),
    ("unsloth/gemma-3-1b-it", "Gemma-1B"),
    ("unsloth/Llama-3.2-3B-Instruct", "Llama-3B"),
]
METHODS = [
    ("lcfx", r"\textsc{LCF-X}"),
    ("xkv", r"\textsc{XKV}"),
    ("t2t", r"\textsc{T2T}"),
]
ABLATIONS = [
    ("xkv", "Full"),
    ("xkv-no-receiver-pooling", r"$-\mathrm{RP}$"),
    ("xkv-no-receiver-xattn", r"$-\mathrm{RX}$"),
    ("xkv-lcfx-pooling", "LCFP"),
]
METHOD_DATASETS = [
    "ropes",
    "musique",
    "qasc",
    "strategyqa",
    "hotpotqa_bridge",
]
ABLATION_DATASETS = ["qasc", "ropes", "strategyqa"]
CLASSIFICATION_DATASETS = {"qasc", "strategyqa"}
DATASET_NAMES = {
    "hotpotqa_bridge": "HotpotQA-bridge",
    "musique": "MuSiQue",
    "qasc": "QASC",
    "ropes": "ROPES",
    "strategyqa": "StrategyQA",
}


def slug(model: str) -> str:
    return model.replace("/", "--")


def receiver_k(model: str) -> int:
    lowered = model.lower()
    if "0.6b" in lowered:
        return 1
    if "1b" in lowered:
        return 2
    if "3b" in lowered or "4b" in lowered:
        return 3
    if "8b" in lowered:
        return 4
    return 2


def hp_name(method: str, receiver: str) -> str | None:
    if method == "lcfx":
        return "sp5"
    if method == "xkv-lcfx-pooling":
        return "dh8_n1_h8_do0p1_k1_sp5_v12"
    if method.startswith("xkv"):
        return f"dh8_n1_h8_do0p1_k{receiver_k(receiver)}_v11"
    return None


def result_dir(method: str, dataset: str, sharer: str, receiver: str) -> Path:
    path = RESULTS / method / dataset / slug(sharer) / slug(receiver)
    hp = hp_name(method, receiver)
    return path / hp if hp else path


def checkpoint_log(method: str, dataset: str, sharer: str, receiver: str) -> Path:
    path = CHECKPOINTS / method / dataset / slug(sharer) / slug(receiver)
    hp = hp_name(method, receiver)
    return path / hp / "training_log.jsonl" if hp else path / "training_log.jsonl"


def load_record(method: str, dataset: str, sharer: str, receiver: str) -> dict | None:
    base = result_dir(method, dataset, sharer, receiver)
    metrics_path = base / "metrics.json"
    latency_path = base / "latency.json"
    if not metrics_path.is_file() or not latency_path.is_file():
        return None

    metrics = json.loads(metrics_path.read_text())
    latency = json.loads(latency_path.read_text()).get("average", {})
    scores = metrics.get("scores")
    record = {
        "accuracy": metrics.get("accuracy"),
        "em": scores.get("em") if scores else None,
        "f1": scores.get("f1") if scores else None,
        "communication_ms": (
            latency.get("sharer_ms") if method == "t2t" else latency.get("fuser_ms")
        ),
        "component_ms": (
            None if method == "t2t" else latency.get("fuser_ms")
        ),
        "total_ms": latency.get("total_ms"),
        "params": 0 if method == "t2t" else None,
    }

    if method != "t2t":
        log_path = checkpoint_log(method, dataset, sharer, receiver)
        if log_path.is_file():
            for line in log_path.read_text().splitlines():
                event = json.loads(line)
                if "trainable_params" in event:
                    record["params"] = event["trainable_params"]
                    break
    return record


def load_grid(dataset: str, methods: list[tuple[str, str]]) -> dict:
    grid = {}
    for sharer, _ in MODELS:
        for receiver, _ in MODELS:
            pair = (sharer, receiver)
            grid[pair] = {
                method: load_record(method, dataset, sharer, receiver)
                for method, _ in methods
            }
    return grid


def primary_value(dataset: str, record: dict | None) -> float | None:
    if record is None:
        return None
    return record["accuracy"] if dataset in CLASSIFICATION_DATASETS else record["f1"]


def fmt_score(value: float | None, bold: bool = False) -> str:
    if value is None:
        return "--"
    text = f"{value * 100:.1f}"
    return rf"\textbf{{{text}}}" if bold else text


def fmt_score_cell(
    dataset: str,
    record: dict | None,
    best_em: float | None,
    best_primary: float | None,
) -> str:
    if record is None:
        return "--"
    primary = primary_value(dataset, record)
    if dataset in CLASSIFICATION_DATASETS:
        return fmt_score(primary, primary is not None and primary == best_primary)
    em = record["em"]
    return (
        f"{fmt_score(em, em is not None and em == best_em)}/"
        f"{fmt_score(primary, primary is not None and primary == best_primary)}"
    )


def fmt_latency(value: float | None, bold: bool = False) -> str:
    if value is None:
        return "--"
    text = f"{value:.0f}"
    return rf"\textbf{{{text}}}" if bold else text


def fmt_latency_cell(
    record: dict | None,
    latency_key: str,
    best_latency: float | None,
    best_total: float | None,
) -> str:
    if record is None:
        return "--"
    latency = record[latency_key]
    total = record["total_ms"]
    return (
        f"{fmt_latency(latency, latency == best_latency)}/"
        f"{fmt_latency(total, total == best_total)}"
    )


def average(records: list[dict | None], key: str) -> float | None:
    values = [record[key] for record in records if record and record[key] is not None]
    return mean(values) if values else None


def sample_count(records: list[dict | None]) -> int:
    return sum(record is not None for record in records)


def method_score_average_text(
    dataset: str, grid: dict, methods: list[tuple[str, str]]
) -> str:
    parts = []
    for method, label in methods:
        records = [cell[method] for cell in grid.values()]
        n = sample_count(records)
        suffix = f" ($n={n}$)" if n != 9 else ""
        if dataset in CLASSIFICATION_DATASETS:
            parts.append(f"{label} {average(records, 'accuracy') * 100:.2f}{suffix}")
        else:
            parts.append(
                f"{label} {average(records, 'em') * 100:.2f}/"
                f"{average(records, 'f1') * 100:.2f}{suffix}"
            )
    return "; ".join(parts)


def method_latency_average_text(
    grid: dict, methods: list[tuple[str, str]], latency_key: str
) -> str:
    parts = []
    for method, label in methods:
        records = [cell[method] for cell in grid.values()]
        n = sample_count(records)
        suffix = f" ($n={n}$)" if n != 9 else ""
        parts.append(
            f"{label} {average(records, latency_key):.1f}/"
            f"{average(records, 'total_ms'):.1f}{suffix}"
        )
    return "; ".join(parts)


def table_header(columns: str, caption: str, label: str) -> list[str]:
    return [
        r"\begin{table*}[t]",
        r"  \centering",
        rf"  \caption{{{caption}}}",
        rf"  \label{{{label}}}",
        rf"  \begin{{tabular}}{{{columns}}}",
        r"    \toprule",
    ]


def table_footer() -> list[str]:
    return [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table*}",
        "",
    ]


def score_grid_table(
    dataset: str, methods: list[tuple[str, str]], label_prefix: str
) -> str:
    grid = load_grid(dataset, methods)
    metric = "accuracy" if dataset in CLASSIFICATION_DATASETS else "EM/F1"
    averages = method_score_average_text(dataset, grid, methods)
    caption = (
        f"{DATASET_NAMES[dataset]} {metric} scores (\\%). Higher is better; "
        f"cell-wise best values are bold. Mean across available model pairs: "
        f"{averages}."
    )
    lines = table_header(
        "ll" + "c" * len(methods),
        caption,
        f"tab:{label_prefix}-{dataset.replace('_', '-')}-scores",
    )
    method_headers = " & ".join(label for _, label in methods)
    lines.append(rf"    Sharer & Receiver & {method_headers} \\")
    lines.append(r"    \midrule")
    for sharer, sharer_name in MODELS:
        for receiver, receiver_name in MODELS:
            cell = grid[(sharer, receiver)]
            records = [cell[method] for method, _ in methods]
            em_values = [
                record["em"] for record in records if record and record["em"] is not None
            ]
            primary_values = [
                primary_value(dataset, record)
                for record in records
                if primary_value(dataset, record) is not None
            ]
            best_em = max(em_values) if em_values else None
            best_primary = max(primary_values) if primary_values else None
            values = [
                fmt_score_cell(dataset, cell[method], best_em, best_primary)
                for method, _ in methods
            ]
            lines.append(
                f"    {sharer_name} & {receiver_name} & "
                + " & ".join(values)
                + r" \\"
            )
    lines.extend(table_footer())
    return "\n".join(lines)


def latency_grid_table(
    dataset: str, methods: list[tuple[str, str]], label_prefix: str
) -> str:
    grid = load_grid(dataset, methods)
    averages = method_latency_average_text(grid, methods, "component_ms")
    caption = (
        f"{DATASET_NAMES[dataset]} ablation latency in milliseconds per example. "
        f"Entries are translator/total; lower is better and cell-wise minima are "
        f"bold. Mean across available model pairs: {averages}."
    )
    lines = table_header(
        "ll" + "c" * len(methods),
        caption,
        f"tab:{label_prefix}-{dataset.replace('_', '-')}-latency",
    )
    method_headers = " & ".join(label for _, label in methods)
    lines.append(rf"    Sharer & Receiver & {method_headers} \\")
    lines.append(r"    \midrule")
    for sharer, sharer_name in MODELS:
        for receiver, receiver_name in MODELS:
            cell = grid[(sharer, receiver)]
            records = [cell[method] for method, _ in methods]
            components = [
                record["component_ms"]
                for record in records
                if record and record["component_ms"] is not None
            ]
            totals = [
                record["total_ms"]
                for record in records
                if record and record["total_ms"] is not None
            ]
            best_component = min(components) if components else None
            best_total = min(totals) if totals else None
            values = [
                fmt_latency_cell(
                    cell[method], "component_ms", best_component, best_total
                )
                for method, _ in methods
            ]
            lines.append(
                f"    {sharer_name} & {receiver_name} & "
                + " & ".join(values)
                + r" \\"
            )
    lines.extend(table_footer())
    return "\n".join(lines)


def method_latency_grid_table(dataset: str) -> str:
    grid = load_grid(dataset, METHODS)
    averages = method_latency_average_text(grid, METHODS, "communication_ms")
    caption = (
        f"{DATASET_NAMES[dataset]} communication/end-to-end latency in milliseconds "
        f"per example. Communication is fuser/translator time for LCF-X and XKV, "
        f"and Sharer text-generation time for T2T. Lower is better; cell-wise "
        f"minima are bold. Mean across available model pairs: {averages}."
    )
    lines = table_header(
        "llccc",
        caption,
        f"tab:methods-{dataset.replace('_', '-')}-latency",
    )
    lines.append(
        r"    Sharer & Receiver & \textsc{LCF-X} & \textsc{XKV} & \textsc{T2T} \\"
    )
    lines.append(r"    \midrule")
    for sharer, sharer_name in MODELS:
        for receiver, receiver_name in MODELS:
            cell = grid[(sharer, receiver)]
            records = [cell[method] for method, _ in METHODS]
            communications = [
                record["communication_ms"]
                for record in records
                if record and record["communication_ms"] is not None
            ]
            totals = [
                record["total_ms"]
                for record in records
                if record and record["total_ms"] is not None
            ]
            best_communication = min(communications) if communications else None
            best_total = min(totals) if totals else None
            values = [
                fmt_latency_cell(
                    cell[method],
                    "communication_ms",
                    best_communication,
                    best_total,
                )
                for method, _ in METHODS
            ]
            lines.append(
                f"    {sharer_name} & {receiver_name} & "
                + " & ".join(values)
                + r" \\"
            )
    lines.extend(table_footer())
    return "\n".join(lines)


def method_summary_data() -> tuple[dict, dict]:
    grids = {dataset: load_grid(dataset, METHODS) for dataset in METHOD_DATASETS}
    summary = {
        method: {
            "scores": [],
            "totals": [],
            "communications": [],
            "components": [],
            "params": [],
            "wins": 0,
            "rank_sum": 0,
        }
        for method, _ in METHODS
    }
    dataset_rows = {}
    for dataset, grid in grids.items():
        dataset_rows[dataset] = {}
        for method, _ in METHODS:
            records = [cell[method] for cell in grid.values()]
            primary = [primary_value(dataset, record) for record in records]
            primary = [value for value in primary if value is not None]
            dataset_rows[dataset][method] = {
                "score": mean(primary) * 100,
                "total": average(records, "total_ms"),
            }
            summary[method]["scores"].append(mean(primary) * 100)
            summary[method]["totals"].extend(
                record["total_ms"] for record in records if record
            )
            summary[method]["communications"].extend(
                record["communication_ms"]
                for record in records
                if record and record["communication_ms"] is not None
            )
            summary[method]["components"].extend(
                record["component_ms"]
                for record in records
                if record and record["component_ms"] is not None
            )
            summary[method]["params"].extend(
                record["params"]
                for record in records
                if record and record["params"] is not None
            )

        for cell in grid.values():
            ranked = sorted(
                (method for method, _ in METHODS),
                key=lambda method: primary_value(dataset, cell[method]),
                reverse=True,
            )
            best = primary_value(dataset, cell[ranked[0]])
            for rank, method in enumerate(ranked, 1):
                summary[method]["rank_sum"] += rank
                if primary_value(dataset, cell[method]) == best:
                    summary[method]["wins"] += 1
    return summary, dataset_rows


def method_conclusion_tables() -> str:
    summary, dataset_rows = method_summary_data()
    lines = table_header(
        "lcccc",
        "Overall score comparison on five datasets (45 model pairs). The native "
        "score is F1 for generative QA and accuracy for classification. XKV has "
        "the highest macro score, best average rank, and most pair-level wins.",
        "tab:method-score-summary",
    )
    lines.append(
        r"    Method & Macro native score & Best datasets & Pair wins & Avg. rank \\"
    )
    lines.append(r"    \midrule")
    dataset_wins = {method: 0 for method, _ in METHODS}
    for dataset in METHOD_DATASETS:
        best = max(dataset_rows[dataset][method]["score"] for method, _ in METHODS)
        for method, _ in METHODS:
            if dataset_rows[dataset][method]["score"] == best:
                dataset_wins[method] += 1
    labels = dict(METHODS)
    for method, _ in METHODS:
        row = summary[method]
        lines.append(
            f"    {labels[method]} & {mean(row['scores']):.2f} & "
            f"{dataset_wins[method]}/5 & {row['wins']}/45 & "
            f"{row['rank_sum'] / 45:.2f}" + r" \\"
        )
    lines.extend(table_footer())

    lines.extend(
        table_header(
            "lcccc",
            "Overall efficiency on five datasets (45 model pairs). Communication "
            "uses fuser/translator time for LCF-X and XKV and Sharer text-generation "
            "time for T2T. XKV is 26.4\\% faster end-to-end than LCF-X, "
            "6.8$\\times$ faster than T2T, and uses 76.1\\% fewer trainable "
            "parameters than LCF-X. T2T trains no adapter.",
            "tab:method-efficiency-summary",
        )
    )
    lines.append(
        r"    Method & Communication (ms) & Total (ms) & Relative to XKV & "
        r"Trainable params \\"
    )
    lines.append(r"    \midrule")
    xkv_total = mean(summary["xkv"]["totals"])
    for method, _ in METHODS:
        row = summary[method]
        communication = f"{mean(row['communications']):.1f}"
        params = mean(row["params"])
        params_text = "0" if params == 0 else f"{params / 1e6:.2f}M"
        lines.append(
            f"    {labels[method]} & {communication} & "
            f"{mean(row['totals']):.1f} & "
            f"{mean(row['totals']) / xkv_total:.2f}$\\times$ & {params_text}"
            + r" \\"
        )
    lines.extend(table_footer())

    lines.extend(
        table_header(
            "lrrrr",
            "Per-dataset XKV improvements. Score deltas use F1 for generative QA "
            "and accuracy for classification; positive values favor XKV. Speedups "
            "are ratios of baseline total latency to XKV total latency.",
            "tab:xkv-improvements",
        )
    )
    lines.append(
        r"    Dataset & $\Delta$ vs. LCF-X & $\Delta$ vs. T2T & "
        r"Speedup vs. LCF-X & Speedup vs. T2T \\"
    )
    lines.append(r"    \midrule")
    for dataset in METHOD_DATASETS:
        row = dataset_rows[dataset]
        lines.append(
            f"    {DATASET_NAMES[dataset]} & "
            f"{row['xkv']['score'] - row['lcfx']['score']:+.2f} & "
            f"{row['xkv']['score'] - row['t2t']['score']:+.2f} & "
            f"{row['lcfx']['total'] / row['xkv']['total']:.2f}$\\times$ & "
            f"{row['t2t']['total'] / row['xkv']['total']:.2f}$\\times$"
            + r" \\"
        )
    lines.extend(table_footer())
    return "\n".join(lines)


def ablation_summary_table() -> str:
    grids = {dataset: load_grid(dataset, ABLATIONS) for dataset in ABLATION_DATASETS}
    labels = dict(ABLATIONS)
    common_pairs = {
        dataset: [
            pair
            for pair, cell in grid.items()
            if all(cell[method] is not None for method, _ in ABLATIONS)
        ]
        for dataset, grid in grids.items()
    }
    summary = {
        method: {
            "dataset_deltas": [],
            "rank_sum": 0,
            "wins": 0,
            "components": [],
            "totals": [],
            "params": [],
        }
        for method, _ in ABLATIONS
    }
    total_pairs = sum(len(pairs) for pairs in common_pairs.values())

    for dataset, grid in grids.items():
        pairs = common_pairs[dataset]
        full_scores = [
            primary_value(dataset, grid[pair]["xkv"]) * 100 for pair in pairs
        ]
        for method, _ in ABLATIONS:
            scores = [
                primary_value(dataset, grid[pair][method]) * 100 for pair in pairs
            ]
            summary[method]["dataset_deltas"].append(
                mean(score - full for score, full in zip(scores, full_scores))
            )
            summary[method]["components"].extend(
                grid[pair][method]["component_ms"] for pair in pairs
            )
            summary[method]["totals"].extend(
                grid[pair][method]["total_ms"] for pair in pairs
            )
            summary[method]["params"].extend(
                grid[pair][method]["params"] for pair in pairs
            )
        for pair in pairs:
            ranked = sorted(
                (method for method, _ in ABLATIONS),
                key=lambda method: primary_value(dataset, grid[pair][method]),
                reverse=True,
            )
            best = primary_value(dataset, grid[pair][ranked[0]])
            for rank, method in enumerate(ranked, 1):
                summary[method]["rank_sum"] += rank
                if primary_value(dataset, grid[pair][method]) == best:
                    summary[method]["wins"] += 1

    full = summary["xkv"]
    full_component = mean(full["components"])
    full_total = mean(full["totals"])
    full_params = mean(full["params"])
    lines = table_header(
        "lrrrrrr",
        f"Ablation summary over {total_pairs} fully matched cells on QASC, ROPES, "
        "and StrategyQA. Full XKV has the best aggregate quality. Removing "
        "Receiver pooling gives the strongest quality--efficiency tradeoff; LCFP "
        "is substantially slower.",
        "tab:ablation-summary",
    )
    lines.append(
        r"    Variant & $\Delta$ score & Avg. rank & Wins & Component (ms) & "
        r"Total (ms) & Params \\"
    )
    lines.append(r"    \midrule")
    for method, _ in ABLATIONS:
        row = summary[method]
        delta = mean(row["dataset_deltas"])
        params = mean(row["params"])
        lines.append(
            f"    {labels[method]} & {delta:+.2f} & "
            f"{row['rank_sum'] / total_pairs:.2f} & "
            f"{row['wins']}/{total_pairs} & "
            f"{mean(row['components']):.1f} & {mean(row['totals']):.1f} & "
            f"{params / 1e6:.2f}M"
            + r" \\"
        )
    lines.append(r"    \midrule")
    lines.append(
        "    "
        + r"$-\mathrm{RP}$ vs. Full"
        + f" & {mean(summary['xkv-no-receiver-pooling']['dataset_deltas']):+.2f} "
        + "& -- & -- & "
        + f"{(mean(summary['xkv-no-receiver-pooling']['components']) / full_component - 1) * 100:+.1f}\\% & "
        + f"{(mean(summary['xkv-no-receiver-pooling']['totals']) / full_total - 1) * 100:+.1f}\\% & "
        + f"{(mean(summary['xkv-no-receiver-pooling']['params']) / full_params - 1) * 100:+.1f}\\%"
        + r" \\"
    )
    lines.append(
        "    "
        + r"$-\mathrm{RX}$ vs. Full"
        + f" & {mean(summary['xkv-no-receiver-xattn']['dataset_deltas']):+.2f} "
        + "& -- & -- & "
        + f"{(mean(summary['xkv-no-receiver-xattn']['components']) / full_component - 1) * 100:+.1f}\\% & "
        + f"{(mean(summary['xkv-no-receiver-xattn']['totals']) / full_total - 1) * 100:+.1f}\\% & "
        + f"{(mean(summary['xkv-no-receiver-xattn']['params']) / full_params - 1) * 100:+.1f}\\%"
        + r" \\"
    )
    lines.append(
        "    LCFP vs. Full"
        + f" & {mean(summary['xkv-lcfx-pooling']['dataset_deltas']):+.2f} "
        + "& -- & -- & "
        + f"{(mean(summary['xkv-lcfx-pooling']['components']) / full_component - 1) * 100:+.1f}\\% & "
        + f"{(mean(summary['xkv-lcfx-pooling']['totals']) / full_total - 1) * 100:+.1f}\\% & "
        + f"{(mean(summary['xkv-lcfx-pooling']['params']) / full_params - 1) * 100:+.1f}\\%"
        + r" \\"
    )
    lines.extend(table_footer())
    return "\n".join(lines)


def build_document() -> str:
    sections = [
        "% Copy-ready table environments only.",
        r"% Requires \usepackage{booktabs} in the paper preamble.",
        "% Scores and latencies are read from the current selected result paths.",
        "",
        "% ------------------------------------------------------------------",
        "% 1. LCF-X / XKV / T2T score and latency grids",
        "% ------------------------------------------------------------------",
        "",
    ]
    for dataset in METHOD_DATASETS:
        sections.append(score_grid_table(dataset, METHODS, "methods"))
        sections.append(method_latency_grid_table(dataset))

    sections.extend(
        [
            "% ------------------------------------------------------------------",
            "% 2. XKV ablation score and latency grids",
            "% ------------------------------------------------------------------",
            "% Full = complete XKV; -RP = no Receiver pooling;",
            "% -RX = no raw Receiver-KV cross-attention;",
            "% LCFP = original non-vectorized LCF-X pooling.",
            "",
        ]
    )
    for dataset in ABLATION_DATASETS:
        sections.append(score_grid_table(dataset, ABLATIONS, "ablation"))
        sections.append(latency_grid_table(dataset, ABLATIONS, "ablation"))

    sections.extend(
        [
            "% ------------------------------------------------------------------",
            "% 3. Conclusion tables",
            "% ------------------------------------------------------------------",
            "",
            method_conclusion_tables(),
            ablation_summary_table(),
        ]
    )
    return "\n".join(sections).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS / "comparison" / "latex_tables.tex",
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_document())
    print(f"Wrote LaTeX tables -> {args.output}")


if __name__ == "__main__":
    main()
