import argparse
import csv
from pathlib import Path


METRIC_COLUMNS = {
    "precision": "metrics/precision(B)",
    "recall": "metrics/recall(B)",
    "map50": "metrics/mAP50(B)",
    "map5095": "metrics/mAP50-95(B)",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize Ultralytics detection runs.")
    parser.add_argument(
        "--root",
        default="/mnt/disk/maoqijun/runs/detect/MLLM",
        help="Directory containing YOLO run folders.",
    )
    parser.add_argument(
        "--output-dir",
        default="/mnt/disk/maoqijun/MLLM/analysis",
        help="Directory to save summary csv/markdown/latex rows.",
    )
    return parser.parse_args()


def read_args_yaml(path):
    args = {}
    if not path.exists():
        return args
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        args[key.strip()] = value.strip()
    return args


def read_results_csv(path):
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            rows.append({k.strip(): v.strip() for k, v in row.items()})
    return rows


def to_float(value):
    return float(value) if value not in ("", None) else 0.0


def summarize_run(run_dir):
    rows = read_results_csv(run_dir / "results.csv")
    best = max(rows, key=lambda row: to_float(row[METRIC_COLUMNS["map5095"]]))
    args = read_args_yaml(run_dir / "args.yaml")
    weights = run_dir / "weights" / "best.pt"

    return {
        "run": run_dir.name,
        "best_epoch": str(int(float(best["epoch"]))),
        "precision": f"{to_float(best[METRIC_COLUMNS['precision']]):.4f}",
        "recall": f"{to_float(best[METRIC_COLUMNS['recall']]):.4f}",
        "map50": f"{to_float(best[METRIC_COLUMNS['map50']]):.4f}",
        "map5095": f"{to_float(best[METRIC_COLUMNS['map5095']]):.4f}",
        "epochs": args.get("epochs", ""),
        "batch": args.get("batch", ""),
        "imgsz": args.get("imgsz", ""),
        "optimizer": args.get("optimizer", ""),
        "lr0": args.get("lr0", ""),
        "best_pt": str(weights),
    }


def write_csv(path, summaries):
    fieldnames = [
        "run",
        "best_epoch",
        "precision",
        "recall",
        "map50",
        "map5095",
        "epochs",
        "batch",
        "imgsz",
        "optimizer",
        "lr0",
        "best_pt",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)


def write_markdown(path, summaries):
    lines = [
        "# Detection Results Summary",
        "",
        "| Run | Best Epoch | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in summaries:
        lines.append(
            f"| {item['run']} | {item['best_epoch']} | {item['precision']} | "
            f"{item['recall']} | {item['map50']} | {item['map5095']} |"
        )
    lines.extend(["", "## Best Checkpoints", ""])
    for item in summaries:
        lines.append(f"- `{item['run']}`: `{item['best_pt']}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex_rows(path, summaries):
    lines = []
    for item in summaries:
        lines.append(
            f"{item['run']} & {item['best_epoch']} & {item['precision']} & "
            f"{item['recall']} & {item['map50']} & {item['map5095']} \\\\"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    root = Path(args.root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = sorted(path.parent for path in root.glob("*/results.csv"))
    summaries = [summarize_run(run_dir) for run_dir in run_dirs]
    summaries.sort(key=lambda item: float(item["map5095"]), reverse=True)

    write_csv(output_dir / "detection_summary.csv", summaries)
    write_markdown(output_dir / "detection_summary.md", summaries)
    write_latex_rows(output_dir / "detection_latex_rows.txt", summaries)

    print((output_dir / "detection_summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
