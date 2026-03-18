import argparse
import copy
import json
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

        for question_index, question in enumerate(questions):
            record = copy.deepcopy(case)
            record["known_id"] = known_id
            record["subject"] = subject
            record["prompt"] = question
            record["source_question_index"] = question_index
            converted.append(record)
            known_id += 1

    return converted


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert custom causal-intervention data into a simplified question-based "
            "format for causal_trace.py."
        )
    )
    parser.add_argument("input_path", help="Path to the source JSON file.")
    parser.add_argument("output_path", help="Path to save the converted JSON file.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    data = load_json(input_path)
    cases = as_case_list(data)
    converted = convert_cases(cases)
    dump_json(converted, output_path)

    print(f"Converted {len(cases)} case(s) into {len(converted)} record(s).")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()