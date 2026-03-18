import argparse
import copy
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def as_case_list(data: Any) -> List[Dict[str, Any]]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    raise TypeError("Input JSON must be either a dict or a list of dicts.")


def extract_subject(case: Dict[str, Any]) -> Any:
    return case["orig"]["triples_labeled"][0][0]


def convert_cases(cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    converted: List[Dict[str, Any]] = []
    known_id = 0

    for case in cases:
        subject = extract_subject(case)
        questions = case.get("questions", [])
        answer = case.get("answer", "")

        for question_index, question in enumerate(questions):
            record = copy.deepcopy(case)
            record["known_id"] = known_id
            record["subject"] = subject
            record["prompt"] = question
            record["attribute"] = answer
            record["source_question_index"] = question_index
            converted.append(record)
            known_id += 1

    return converted


def split_cases_by_hop(cases: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    grouped: Dict[int, List[Dict[str, Any]]] = defaultdict(list)

    for case in cases:
        hop = len(case["orig"]["triples_labeled"])
        if hop in {2, 3, 4}:
            grouped[hop].append(case)

    return dict(grouped)


def build_output_paths(output_path: Path) -> Dict[int, Path]:
    suffix = output_path.suffix or ".json"
    prefix = output_path.stem if output_path.suffix else output_path.name
    parent = output_path.parent

    return {
        2: parent / f"{prefix}_2hop{suffix}",
        3: parent / f"{prefix}_3hop{suffix}",
        4: parent / f"{prefix}_4hop{suffix}",
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert custom causal-intervention data into a simplified question-based "
            "format for causal_trace.py, and split outputs by hop count."
        )
    )
    parser.add_argument("input_path", help="Path to the source JSON file.")
    parser.add_argument(
        "output_path",
        help=(
            "Output prefix/path. The script will generate three files named "
            "<prefix>_2hop.json, <prefix>_3hop.json, and <prefix>_4hop.json. "
            "If a suffix is provided, it will be preserved."
        ),
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    data = load_json(input_path)
    cases = as_case_list(data)
    cases_by_hop = split_cases_by_hop(cases)
    output_paths = build_output_paths(output_path)

    total_converted = 0

    for hop in (2, 3, 4):
        hop_cases = cases_by_hop.get(hop, [])
        converted = convert_cases(hop_cases)
        dump_json(converted, output_paths[hop])
        total_converted += len(converted)

        print(
            f"{hop}hop: Converted {len(hop_cases)} case(s) into {len(converted)} record(s)."
        )
        print(f"Saved to: {output_paths[hop]}")

    print(f"Processed {len(cases)} total case(s) into {total_converted} total record(s).")


if __name__ == "__main__":
    main()