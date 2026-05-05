"""
Fetch all run data from W&B and save locally for analysis/plotting.

Usage:
    python -m assignment_1_code.collect_results
    python -m assignment_1_code.collect_results --out results/

Output:
    summary.csv   - one row per run (hyperparams + final metrics)
    history.csv   - per-epoch metrics for all runs (for curve plots)
"""

import argparse
import os
from pathlib import Path

import pandas as pd
import wandb

ENTITY = os.getenv("WANDB_ENTITY", "ruslanbasyrov-tu-wien")
PROJECT = os.getenv("WANDB_PROJECT", "dlvs-assignment-1")
GROUPS = ["resnet18", "yourCNN", "yourViT"]

HISTORY_KEYS = [
    "train/loss",
    "train/accuracy",
    "train/per_class_accuracy",
    "val/loss",
    "val/accuracy",
    "val/per_class_accuracy",
    "test/loss",
    "test/accuracy",
    "test/per_class_accuracy",
]


def fetch_summary(runs) -> pd.DataFrame:
    records = []
    for run in runs:
        row = {
            "run_id": run.id,
            "name": run.name,
            "group": run.group,
            "state": run.state,
            **run.config,
            **{k: v for k, v in run.summary._json_dict.items() if not k.startswith("_")},
        }
        records.append(row)
    return pd.DataFrame(records)


def main(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    api = wandb.Api()
    runs = api.runs(
        f"{ENTITY}/{PROJECT}",
        filters={"group": {"$in": GROUPS}},
    )
    runs = list(runs)
    print(f"Found {len(runs)} runs across groups: {GROUPS}")

    summary_df = fetch_summary(runs)
    summary_path = out_dir / "summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary saved → {summary_path}  ({len(summary_df)} rows)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect W&B results locally")
    parser.add_argument("--out", default="results", help="Output directory for CSVs")
    args = parser.parse_args()
    main(Path(args.out))
