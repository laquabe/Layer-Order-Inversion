import argparse
import json
from pathlib import Path


def split_list(data, k):
    n = len(data)
    base = n // k
    remainder = n % k
    parts = []
    start = 0
    for i in range(k):
        size = base + (1 if i < remainder else 0)
        end = start + size
        parts.append(data[start:end])
        start = end
    return parts


def main():
    parser = argparse.ArgumentParser(description="Split a JSON dataset into k parts for parallel causal runs.")
    parser.add_argument("--input", required=True, help="Path to the input JSON file.")
    parser.add_argument("--k", required=True, type=int, help="Number of splits.")
    parser.add_argument("--output_dir", required=True, help="Directory to save split JSON files.")
    parser.add_argument(
        "--prefix",
        default=None,
        help="Optional output file prefix. Defaults to the input filename stem.",
    )
    args = parser.parse_args()

    if args.k <= 0:
        raise ValueError("--k must be a positive integer")

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(input_path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of records or a single record dict")

    total = len(data)
    parts = split_list(data, args.k)
    prefix = args.prefix or input_path.stem

    print(f"Total samples: {total}")
    print(f"Split into {args.k} parts")

    for idx, part in enumerate(parts):
        output_path = output_dir / f"{prefix}_part_{idx}.json"
        output_path.write_text(
            json.dumps(part, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Part {idx}: {len(part)} samples -> {output_path}")


if __name__ == "__main__":
    main()