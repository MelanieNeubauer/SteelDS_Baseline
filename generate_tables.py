#!/usr/bin/env python3
"""Generate four LaTeX tables from three model-evaluation CSV files."""

import argparse
import csv
import re
from pathlib import Path
from statistics import mean


# Output order matches the supplied LaTeX tables.
MODELS = [
    ("yolov8n-seg.yaml", "YOLOv8n"),
    ("yolo11n-seg.yaml", "YOLO11n"),
    ("yolo12n-seg.yaml", "YOLO12n"),
    ("yolo26n-seg.yaml", "YOLO26n"),
    ("maskrcnn-resnet50", "Mask R-CNN"),
]

DATASETS = ["a1", "a2", "a3"]

DETECTION_ROWS = [
    ("metrics/mAP50(B)", "mAP50 (Overall)"),
    ("metrics/mAP50-95(B)", "mAP50-95 (Overall)"),
    ("box_mAP50_Steel", "mAP50 (Steel)"),
    ("box_mAP50_Copper", "mAP50 (Copper)"),
    ("box_mAP50-95_Steel", "mAP50-95 (Steel)"),
    ("box_mAP50-95_Copper", "mAP50-95 (Copper)"),
    ("metrics/mAP_small(B)", "mAP Small"),
    ("metrics/mAP_medium(B)", "mAP Medium"),
    ("metrics/mAP_large(B)", "mAP Large"),
    ("metrics/recall_mar100(B)", "Recall"),
]

SEGMENTATION_ROWS = [
    ("metrics/mAP50(M)", "mAP50 (Overall)"),
    ("metrics/mAP50-95(M)", "mAP50-95 (Overall)"),
    ("seg_mAP50_Steel", "mAP50 (Steel)"),
    ("seg_mAP50_Copper", "mAP50 (Copper)"),
    ("seg_mAP50-95_Steel", "mAP50-95 (Steel)"),
    ("seg_mAP50-95_Copper", "mAP50-95 (Copper)"),
    ("metrics/mAP_small(M)", "mAP Small"),
    ("metrics/mAP_medium(M)", "mAP Medium"),
    ("metrics/mAP_large(M)", "mAP Large"),
    ("metrics/recall_mar100(M)", "Recall"),
]

FITNESS_ROWS = [
    ("fitness(B)", "Fitness (B)"),
    ("fitness(M)", "Fitness (M)"),
    ("fitness(Combined)", "Fitness (C)"),
]

AVERAGE_ROWS = [
    ("B", [
        ("metrics/mAP50-95(B)", "mAP50-95 (Overall)"),
        ("box_mAP50-95_Steel", "mAP50-95 (Steel)"),
        ("box_mAP50-95_Copper", "mAP50-95 (Copper)"),
        ("metrics/mAP_medium(B)", "mAP (Medium)"),
        ("metrics/mAP_large(B)", "mAP (Large)"),
        ("metrics/recall_mar100(B)", "Recall"),
    ]),
    ("M", [
        ("metrics/mAP50-95(M)", "mAP50-95 (Overall)"),
        ("seg_mAP50-95_Steel", "mAP50-95 (Steel)"),
        ("seg_mAP50-95_Copper", "mAP50-95 (Copper)"),
        ("metrics/mAP_medium(M)", "mAP (Medium)"),
        ("metrics/mAP_large(M)", "mAP (Large)"),
        ("metrics/recall_mar100(M)", "Recall"),
    ]),
]

NUMBER_RE = re.compile(
    r"^\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
    r"(?:\s*±\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?))?\s*$"
)


def load_csv(path):
    """Return rows indexed by model name."""
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "Model" not in reader.fieldnames:
            raise ValueError(f"{path}: CSV must contain a 'Model' column.")
        return {row["Model"].strip(): row for row in reader}


def parse_value(value):
    """Parse a CSV metric into (mean, standard deviation), or None."""
    if value is None:
        return None

    value = value.strip()
    if not value or value.upper() in {
        "N/A", "NA", "NAN", "NONE", "---", "-"
    }:
        return None

    match = NUMBER_RE.fullmatch(value)
    if not match:
        raise ValueError(f"Cannot parse metric value: {value!r}")

    mu = float(match.group(1))
    sd = float(match.group(2)) if match.group(2) is not None else None
    return mu, sd


def get_value(data, model, column):
    """Safely retrieve a metric for a model."""
    row = data.get(model)
    if row is None:
        return None
    return parse_value(row.get(column))


def latex_escape(text):
    """Escape special LaTeX characters in ordinary text."""
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(char, char) for char in text)


def format_metric(value, decimals=2, include_sd=True):
    """Format a metric as a percentage, or as --- when missing."""
    if value is None:
        return "---"

    mu, sd = value
    if include_sd and sd is not None:
        return (
            f"{mu * 100:.{decimals}f} "
            f"$\\pm$ {sd * 100:.{decimals}f}"
        )
    return f"{mu * 100:.{decimals}f}"


def bold(text):
    return rf"\textbf{{{text}}}"


def format_best_row(values, decimals=2, include_sd=True):
    """Format values and bold all entries tied for the largest mean."""
    available = [value for value in values if value is not None]
    if not available:
        return ["---" for _ in values]

    best = max(value[0] for value in available)
    result = []

    for value in values:
        formatted = format_metric(
            value, decimals=decimals, include_sd=include_sd
        )
        if value is not None and abs(value[0] - best) < 1e-12:
            formatted = bold(formatted)
        result.append(formatted)

    return result


def get_model_values(data, column):
    return [get_value(data, model, column) for model, _ in MODELS]


def generate_fitness(datasets):
    """Generate the fitness table."""
    out = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Fitness Results (Mean $\pm$ SD over 5 Seeds in \%). "
        r"Evaluated metrics: (C) Combined, (B) Bounding Box, "
        r"(M) Instance Segmentation.}",
        r"\label{tab:fitness_results}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{@{}lccccc@{}}",
        r"\toprule",
        " & " + " & ".join(bold(name) for _, name in MODELS) + r" \\",
        r"\midrule",
    ]

    for dataset in DATASETS:
        out.append(
            rf"\multicolumn{{6}}{{c}}{{\textbf{{Dataset {dataset}}}}} \\"
        )
        out.append(r"\midrule")

        for column, label in FITNESS_ROWS:
            values = get_model_values(datasets[dataset], column)
            cells = format_best_row(values)
            out.append(
                latex_escape(label) + " & " + " & ".join(cells) + r" \\"
            )

    out.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\end{table*}",
    ])
    return "\n".join(out) + "\n"


def generate_detection_or_segmentation(datasets, rows, caption, label):
    """Generate a bounding-box or instance-segmentation table."""
    out = [
        r"\begin{table*}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{@{}lccccc@{}}",
        r"\toprule",
        bold("Metric") + " & "
        + " & ".join(bold(name) for _, name in MODELS) + r" \\",
        r"\midrule",
    ]

    for dataset in DATASETS:
        out.append(
            rf"\multicolumn{{6}}{{c}}{{\textbf{{Dataset {dataset}}}}} \\"
        )
        out.append(r"\midrule")

        for index, (column, label) in enumerate(rows):
            values = get_model_values(datasets[dataset], column)
            cells = format_best_row(values)
            out.append(
                latex_escape(label) + " & " + " & ".join(cells) + r" \\"
            )

            if index in (1, 5, 8):
                out.append(r"\cmidrule{1-6}")

        if dataset != DATASETS[-1]:
            out.append(r"\midrule")

    out.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\end{table*}",
    ])
    return "\n".join(out) + "\n"


def generate_average(datasets):
    """
    Average each metric across available models for each dataset.

    Bold the best dataset separately for each metric within each task.
    """
    lines = [
        r"\begin{tabular}{@{}llcccccc@{}}",
        r"\toprule",
        r" & \textbf{Dataset} & "
        r"\textbf{mAP50-95} & \textbf{mAP50-95} & "
        r"\textbf{mAP50-95} & \textbf{mAP} & "
        r"\textbf{mAP} & \textbf{Recall} \\",
        r" & & \textbf{(Overall)} & \textbf{(Steel)} & "
        r"\textbf{(Copper)} & \textbf{(Medium)} & "
        r"\textbf{(Large)} & \textbf{(Overall)} \\",
        r"\midrule",
    ]

    for task_index, (task, metrics) in enumerate(AVERAGE_ROWS):
        if task_index > 0:
            lines.append(r"\midrule")

        # Matrix: dataset index -> metric index -> (mean, None).
        matrix = []

        for dataset in DATASETS:
            metric_values = []

            for column, _label in metrics:
                model_values = get_model_values(datasets[dataset], column)
                available = [
                    value[0]
                    for value in model_values
                    if value is not None
                ]

                metric_values.append(
                    (mean(available), None) if available else None
                )

            matrix.append(metric_values)

        # Format each metric column independently across datasets.
        formatted = [
            ["---"] * len(metrics)
            for _ in DATASETS
        ]

        for metric_index in range(len(metrics)):
            values = [
                matrix[dataset_index][metric_index]
                for dataset_index in range(len(DATASETS))
            ]

            cells = format_best_row(
                values,
                decimals=2,
                include_sd=False,
            )

            for dataset_index, cell in enumerate(cells):
                formatted[dataset_index][metric_index] = cell

        task_name = "Box (B)" if task == "B" else "Seg (M)"

        for dataset_index, dataset in enumerate(DATASETS):
            task_label = (
                r"\multirow{3}{*}{\rotatebox[origin=c]{90}{"
                + task_name
                + "}}"
                if dataset_index == 0
                else ""
            )

            row = [
                task_label,
                f"Dataset {latex_escape(dataset)}",
                *formatted[dataset_index],
            ]
            lines.append(" & ".join(row) + r" \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
    ])

    return "\n".join(lines) + "\n"


def save_tabular(path, content):
    """Save only the LaTeX tabular environment."""
    if isinstance(content, list):
        content = "\n".join(content)

    start = content.find(r"\begin{tabular}")
    end_marker = r"\end{tabular}"

    if start == -1:
        raise ValueError(f"No tabular environment found in {path}")

    end = content.find(end_marker, start)
    if end == -1:
        raise ValueError(f"No closing tabular environment found in {path}")

    end += len(end_marker)
    path.write_text(content[start:end] + "\n", encoding="utf-8")
    print(f"Wrote {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate four LaTeX tables from three CSV datasets."
    )
    parser.add_argument("--a1", type=Path, default=Path("a1.csv"))
    parser.add_argument("--a2", type=Path, default=Path("a2.csv"))
    parser.add_argument("--a3", type=Path, default=Path("a3.csv"))
    parser.add_argument(
        "--outdir", type=Path, default=Path("latex_tables")
    )
    args = parser.parse_args()

    paths = {
        "a1": args.a1,
        "a2": args.a2,
        "a3": args.a3,
    }
    datasets = {
        name: load_csv(path)
        for name, path in paths.items()
    }

    args.outdir.mkdir(parents=True, exist_ok=True)

    tables = {
        "Fitness_Results.tex": generate_fitness(datasets),
        "Average_Performance.tex": generate_average(datasets),
        "Bounding_Box_Results.tex": generate_detection_or_segmentation(
            datasets,
            DETECTION_ROWS,
            "Bounding Box Detection Results (B) "
            "(Mean $\\pm$ SD over 5 Seeds in \\%).",
            "tab:det_results",
        ),
        "Instance_Segmentation_Result.tex":
            generate_detection_or_segmentation(
                datasets,
                SEGMENTATION_ROWS,
                "Instance Segmentation Results (M) "
                "(Mean $\\pm$ SD over 5 Seeds in \\%).",
                "tab:seg_results",
            ),
    }

    for filename, content in tables.items():
        save_tabular(args.outdir / filename, content)


if __name__ == "__main__":
    main()
