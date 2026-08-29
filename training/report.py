"""
Renders the tracked runs as the tables the README carries.

Generating them from the tracking store rather than transcribing them is what
keeps the documented numbers and the recorded runs from drifting apart.

Usage: python -m training.report
"""

import mlflow

from training import config


def load_runs():
    mlflow.set_tracking_uri(config.MLFLOW_URI)
    runs = mlflow.search_runs(
        experiment_names=[config.MLFLOW_EXPERIMENT],
        # A run still in progress has no metrics yet and would render as blanks.
        filter_string="attributes.status = 'FINISHED'",
        order_by=["attributes.start_time ASC"],
    )
    if runs.empty:
        raise SystemExit("No finished runs recorded yet.")
    return runs


def stage(runs, column):
    """Each stage records a different family of metrics; this groups the runs by which."""
    if column not in runs.columns:
        return runs.iloc[0:0]
    return runs[runs[column].notna()]


def table(rows: list[list[str]], headers: list[str], alignment: str) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + alignment + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def quality_table(models) -> str:
    # Ordered weakest first, so each row reads as what it buys over the one below
    # it rather than as the order the runs happened to be launched in.
    rows = [
        [
            f"`{run['tags.mlflow.runName']}`",
            f"{run['metrics.test_accuracy']:.3f}",
            f"{run['metrics.test_f1_macro']:.3f}",
            f"**{run['metrics.test_f1_ironic']:.3f}**",
            f"{run['metrics.test_precision_ironic']:.3f}",
            f"{run['metrics.test_recall_ironic']:.3f}",
        ]
        for _, run in models.sort_values("metrics.test_f1_ironic").iterrows()
    ]
    return table(
        rows,
        ["Model", "Accuracy", "Macro F1", "F1 ironic", "Precision", "Recall"],
        "---|---:|---:|---:|---:|---:",
    )


def calibration_table(calibrations) -> str:
    run = calibrations.iloc[-1]
    rows = [
        [
            name,
            f"{threshold:.2f}",
            f"{run[f'metrics.{prefix}_accuracy']:.3f}",
            f"**{run[f'metrics.{prefix}_f1_macro']:.3f}**",
            f"{run[f'metrics.{prefix}_f1_ironic']:.3f}",
            f"{run[f'metrics.{prefix}_precision_ironic']:.3f}",
            f"{run[f'metrics.{prefix}_recall_ironic']:.3f}",
        ]
        for name, prefix, threshold in (
            ("Argmax", "argmax", 0.5),
            ("Selected", "validation", float(run["params.threshold"])),
        )
    ]
    return table(
        rows,
        [
            "Operating point",
            "Threshold",
            "Accuracy",
            "Macro F1",
            "F1 ironic",
            "Precision",
            "Recall",
        ],
        "---|---:|---:|---:|---:|---:|---:",
    )


def runtime_table(runtimes) -> str:
    rows = [
        [
            f"`{run['tags.mlflow.runName']}`",
            f"{run['metrics.size_mb']:.0f}",
            f"{run['metrics.latency_p50_ms']:.1f}",
            f"{run['metrics.latency_p95_ms']:.1f}",
            f"{run['metrics.accuracy']:.3f}",
            f"{run['metrics.f1_ironic']:.3f}",
        ]
        for _, run in runtimes.sort_values(
            "metrics.latency_p50_ms", ascending=False
        ).iterrows()
    ]
    return table(
        rows,
        ["Runtime", "Size MB", "p50 ms", "p95 ms", "Accuracy", "F1 ironic"],
        "---|---:|---:|---:|---:|---:",
    )


def main() -> None:
    runs = load_runs()
    models = stage(runs, "metrics.test_f1_ironic")
    calibrations = stage(runs, "params.criterion")
    runtimes = stage(runs, "metrics.size_mb")

    if not models.empty:
        print("## Models\n")
        print(quality_table(models))
    if not calibrations.empty:
        print("\n## Decision threshold, validation split\n")
        print(calibration_table(calibrations))
    if not runtimes.empty:
        print("\n## Runtimes\n")
        print(runtime_table(runtimes))

    noun = "run" if len(runs) == 1 else "runs"
    print(f"\n{len(runs)} finished {noun} in {config.MLFLOW_URI}")


if __name__ == "__main__":
    main()
