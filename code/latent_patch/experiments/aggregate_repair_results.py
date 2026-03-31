import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


MODULE_NAMES = ["base", "mlp", "attn"]


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
    print(f"Saved aggregated tables to: {output_dir}")


if __name__ == "__main__":
    main()