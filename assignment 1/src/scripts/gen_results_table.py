"""
Generate markdown results table from results/summary.csv.
Outputs build/tables/results_table.md.
"""

import csv
import math
import sys
from pathlib import Path


def fmt_float(val, digits=4):
    try:
        f = float(val)
        if math.isnan(f):
            return "—"
        return f"{f:.{digits}f}"
    except (TypeError, ValueError):
        return "—"


AUG_SHORT = {"augmentation": "aug", "basic": "basic", "strong": "strong", "none": "none"}
OPT_SHORT  = {"adamw": "AdamW", "sgd": "SGD"}
SCHED_SHORT = {"exponential": "exp", "cosine": "cos"}


def fmt_config(row):
    parts = []
    aug = row.get("augmentation", "")
    if aug:
        parts.append(AUG_SHORT.get(aug, aug))
    wd = row.get("weight_decay", "")
    if wd not in ("", None):
        try:
            parts.append(f"wd={float(wd):g}")
        except ValueError:
            parts.append(f"wd={wd}")
    dropout = row.get("dropout", "")
    if dropout not in ("", None):
        try:
            parts.append(f"d={float(dropout):g}")
        except ValueError:
            pass
    opt = row.get("optimizer", "")
    if opt:
        parts.append(OPT_SHORT.get(opt, opt))
    sched = row.get("scheduler", "")
    if sched:
        parts.append(SCHED_SHORT.get(sched, sched))
    return ", ".join(parts)


def md_table(headers, rows):
    col_widths = [max(len(h), max((len(r[i]) for r in rows), default=0)) for i, h in enumerate(headers)]
    def fmt_row(cells):
        return "| " + " | ".join(c.ljust(col_widths[i]) for i, c in enumerate(cells)) + " |"
    sep = "|" + "|".join("-" * (w + 2) for w in col_widths) + "|"
    lines = [fmt_row(headers), sep] + [fmt_row(r) for r in rows]
    return "\n".join(lines)


def main(summary_path: Path, out_path: Path):
    with open(summary_path, newline="") as f:
        rows = list(csv.DictReader(f))

    # best run per group = highest val/mean_accuracy
    best = {}
    for row in rows:
        g = row["group"]
        try:
            acc = float(row["val/mean_accuracy"])
        except (ValueError, KeyError):
            continue
        if math.isnan(acc):
            continue
        if g not in best or acc > float(best[g]["val/mean_accuracy"]):
            best[g] = row

    group_order = ["resnet18", "yourCNN", "yourViT"]
    display_names = {"resnet18": "ResNet18", "yourCNN": "CNN", "yourViT": "ViT"}

    headers = ["Model", "Best config", "Val acc", "Val p-c acc", "Test acc", "Test p-c acc"]
    table_rows = []
    for g in group_order:
        if g not in best:
            continue
        row = best[g]
        table_rows.append([
            display_names.get(g, g),
            fmt_config(row),
            fmt_float(row.get("val/accuracy")),
            fmt_float(row.get("val/mean_accuracy")),
            fmt_float(row.get("test/accuracy")),
            fmt_float(row.get("test/per_class_accuracy")),
        ])

    caption = ": Best configuration per model selected by highest validation mean per-class accuracy (p-c acc = per-class accuracy). {#tbl-best}"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md_table(headers, table_rows) + "\n" + caption + "\n")
    print(f"Table saved → {out_path}")


if __name__ == "__main__":
    root = Path(__file__).parent.parent.parent
    summary = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "results" / "summary.csv"
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else root / "build" / "tables" / "results_table.md"
    main(summary, out)
