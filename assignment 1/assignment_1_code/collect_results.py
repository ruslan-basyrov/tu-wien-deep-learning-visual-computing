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
    "train/mean_loss",
    "train/accuracy",
    "train/mean_accuracy",
    "val/mean_loss",
    "val/accuracy",
    "val/mean_accuracy",
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


def fetch_history(runs) -> pd.DataFrame:
    frames = []
    for run in runs:
        hist = run.history()
        if hist.empty:
            continue
        keep = ["_step"] + [k for k in HISTORY_KEYS if k in hist.columns]
        hist = hist[keep].copy()
        hist.insert(0, "run_id", run.id)
        hist.insert(1, "name", run.name)
        hist.insert(2, "group", run.group)
        frames.append(hist)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


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

    history_df = fetch_history(runs)
    if not history_df.empty:
        history_path = out_dir / "history.csv"
        history_df.to_csv(history_path, index=False)
        print(f"History saved → {history_path}  ({len(history_df)} rows)")
    else:
        print("No history data found (runs may still be in progress or no metrics logged yet).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect W&B results locally")
    parser.add_argument("--out", default="results", help="Output directory for CSVs")
    args = parser.parse_args()
    main(Path(args.out))
