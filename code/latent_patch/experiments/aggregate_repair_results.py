import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


MODULE_NAMES = ["base", "mlp", "attn"]
MODULE_CMAPS = {
    "base": "Purples",
    "mlp": "Greens",
    "attn": "Reds",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate last-hop repair results into module-wise accuracy tables."
    )
    parser.add_argument(
        "result_path",
        help="Path to results.jsonl or its parent run directory containing results.jsonl.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory to save aggregated tables. Defaults to <run_dir>/analysis.",
    )
    return parser.parse_args()


def resolve_paths(result_path: str):
    path = Path(result_path)
    if path.is_dir():
        run_dir = path
        jsonl_path = run_dir / "results.jsonl"
    else:
        jsonl_path = path
        run_dir = path.parent
    if not jsonl_path.is_file():
        raise FileNotFoundError(f"Could not find results.jsonl at: {jsonl_path}")
    return run_dir, jsonl_path


def load_rows(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def filter_valid_rows(rows: List[dict]) -> List[dict]:
    valid = []
    for row in rows:
        if not row.get("donor_is_correct", False):
            continue
        if row.get("baseline_is_correct", True):
            continue
        valid.append(row)
    return valid


def aggregate_module_tables(rows: List[dict]) -> Dict[str, dict]:
    grouped = {
        module: defaultdict(lambda: {"correct": 0, "total": 0}) for module in MODULE_NAMES
    }
    token_offsets = set()
    layers = set()

    for row in rows:
        module = row.get("module")
        if module not in MODULE_NAMES:
            continue
        token_offset = int(row["token_offset"])
        layer = int(row["layer"])
        token_offsets.add(token_offset)
        layers.add(layer)
        key = (token_offset, layer)
        grouped[module][key]["total"] += 1
        grouped[module][key]["correct"] += int(bool(row.get("patched_is_correct", False)))

    sorted_offsets = sorted(token_offsets, key=lambda x: abs(x))
    sorted_layers = sorted(layers)
    tables = {}

    for module in MODULE_NAMES:
        matrix = []
        for offset in sorted_offsets:
            row_values = []
            for layer in sorted_layers:
                stats = grouped[module].get((offset, layer))
                if not stats or stats["total"] == 0:
                    row_values.append(None)
                else:
                    row_values.append(stats["correct"] / stats["total"])
            matrix.append(row_values)
        tables[module] = {
            "token_offsets": sorted_offsets,
            "layers": sorted_layers,
            "values": matrix,
        }
    return tables


def offset_label(token_offset: int) -> str:
    return f"last_{abs(token_offset)}"


def write_module_csv(output_path: Path, table: dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["token_offset"] + [f"layer_{layer}" for layer in table["layers"]])
        for token_offset, values in zip(table["token_offsets"], table["values"]):
            writer.writerow(
                [offset_label(token_offset)]
                + ["" if value is None else f"{value:.6f}" for value in values]
            )


def write_summary(output_path: Path, total_rows: int, valid_rows: int, tables: Dict[str, dict]) -> None:
    summary = {
        "total_rows": total_rows,
        "valid_rows": valid_rows,
        "modules": {},
    }
    for module, table in tables.items():
        non_empty = 0
        for row in table["values"]:
            for value in row:
                if value is not None:
                    non_empty += 1
        summary["modules"][module] = {
            "num_token_offsets": len(table["token_offsets"]),
            "num_layers": len(table["layers"]),
            "num_filled_cells": non_empty,
        }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def write_tables_json(output_path: Path, tables: Dict[str, dict]) -> None:
    payload = {}
    for module, table in tables.items():
        payload[module] = {
            "token_offsets": table["token_offsets"],
            "token_labels": [offset_label(x) for x in table["token_offsets"]],
            "layers": table["layers"],
            "values": table["values"],
        }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def module_plot_color(module: str):
    cmap_name = MODULE_CMAPS[module]
    cmap = plt.get_cmap(cmap_name)
    return cmap(0.75)


def plot_per_offset_lines(output_dir: Path, tables: Dict[str, dict]) -> None:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    all_offsets = []
    for module in MODULE_NAMES:
        all_offsets.extend(tables.get(module, {}).get("token_offsets", []))
    unique_offsets = sorted(set(all_offsets), key=lambda x: abs(x))

    for token_offset in unique_offsets:
        fig, ax = plt.subplots(figsize=(5.0, 3.2), dpi=200)
        has_any_series = False
        all_values = []

        for module in MODULE_NAMES:
            table = tables.get(module)
            if not table:
                continue
            if token_offset not in table["token_offsets"]:
                continue

            offset_idx = table["token_offsets"].index(token_offset)
            layers = table["layers"]
            values = table["values"][offset_idx]

            filtered_layers = []
            filtered_values = []
            for layer, value in zip(layers, values):
                if value is None:
                    continue
                filtered_layers.append(layer)
                filtered_values.append(value)

            if not filtered_layers:
                continue

            has_any_series = True
            all_values.extend(filtered_values)
            ax.plot(
                filtered_layers,
                filtered_values,
                marker="o",
                linewidth=2,
                markersize=4,
                color=module_plot_color(module),
                label=module,
            )

        ax.set_xlabel("Layer")
        ax.set_ylabel("Repair success rate")
        ax.set_title(f"Repair accuracy by layer ({offset_label(token_offset)})")

        if all_values:
            y_min = min(all_values)
            y_max = max(all_values)
            if y_min == y_max:
                margin = max(0.02, abs(y_min) * 0.1, 0.01)
            else:
                margin = max((y_max - y_min) * 0.15, 0.01)
            lower = max(0.0, y_min - margin)
            upper = min(1.0, y_max + margin)
            if lower >= upper:
                lower = max(0.0, y_min - 0.05)
                upper = min(1.0, y_max + 0.05)
            ax.set_ylim(lower, upper)

        ax.grid(True, linestyle="--", alpha=0.3)

        if has_any_series:
            ax.legend()

        fig.tight_layout()
        fig.savefig(plots_dir / f"repair_rate_{offset_label(token_offset)}.png", bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    args = parse_args()
    run_dir, jsonl_path = resolve_paths(args.result_path)
    output_dir = Path(args.output_dir) if args.output_dir is not None else run_dir / "analysis"

    rows = load_rows(jsonl_path)
    valid_rows = filter_valid_rows(rows)
    tables = aggregate_module_tables(valid_rows)

    for module, table in tables.items():
        write_module_csv(output_dir / f"acc_table_{module}.csv", table)
    write_tables_json(output_dir / "acc_tables.json", tables)
    write_summary(output_dir / "summary.json", len(rows), len(valid_rows), tables)
    plot_per_offset_lines(output_dir, tables)
    print(f"Saved aggregated tables to: {output_dir}")


if __name__ == "__main__":
    main()