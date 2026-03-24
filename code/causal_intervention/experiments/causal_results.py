import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm


KIND_LABELS = {
    None: "base",
    "mlp": "mlp",
    "attn": "attn",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Post-process causal tracing results and create aggregated plots."
    )
    parser.add_argument(
        "result_dir",
        help="Directory containing causal_trace outputs. Can be the run directory or its cases/ subdirectory.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory to save aggregated plots. Defaults to <result_dir>/analysis or <run_dir>/analysis.",
    )
    parser.add_argument(
        "--last_k",
        type=int,
        default=3,
        help="How many question-end tokens to include in the aggregated heatmap, counted backward from the last token without overlapping subject tokens.",
    )
    return parser.parse_args()


def resolve_cases_dir(result_dir: str) -> Tuple[Path, Path]:
    path = Path(result_dir)
    if path.name == "cases":
        cases_dir = path
        run_dir = path.parent
    else:
        cases_dir = path / "cases"
        run_dir = path

    if not cases_dir.is_dir():
        raise FileNotFoundError(f"Could not find cases directory at: {cases_dir}")
    return run_dir, cases_dir


def infer_kind_from_path(path: Path) -> Optional[str]:
    name = path.stem
    if name.endswith("_mlp"):
        return "mlp"
    if name.endswith("_attn"):
        return "attn"
    return None


def extract_known_id(path: Path) -> Optional[int]:
    match = re.match(r"knowledge_(\d+)(?:_(mlp|attn))?$", path.stem)
    if match is None:
        return None
    return int(match.group(1))


def is_valid_case(npz_data) -> bool:
    if "correct_prediction" not in npz_data:
        return False
    if not bool(npz_data["correct_prediction"]):
        return False
    required_keys = ["scores", "num_subject_tokens", "traced_labels"]
    return all(key in npz_data for key in required_keys)


def normalize_labels(raw_labels) -> List[str]:
    return [str(label) for label in raw_labels.tolist()]


def load_case_results(cases_dir: Path) -> Dict[Optional[str], List[dict]]:
    grouped_cases: Dict[Optional[str], List[dict]] = defaultdict(list)
    case_paths = sorted(cases_dir.glob("knowledge_*.npz"))

    for path in tqdm(case_paths, desc="Loading causal trace cases"):
        known_id = extract_known_id(path)
        if known_id is None:
            continue

        try:
            npz_data = np.load(path, allow_pickle=True)
        except Exception:
            continue

        if not is_valid_case(npz_data):
            continue

        scores = np.array(npz_data["scores"], dtype=float)
        if scores.ndim != 2 or scores.shape[0] == 0 or scores.shape[1] == 0:
            continue

        num_subject_tokens = int(np.array(npz_data["num_subject_tokens"]).item())
        if num_subject_tokens <= 0 or num_subject_tokens > scores.shape[0]:
            continue

        traced_labels = normalize_labels(npz_data["traced_labels"])
        if len(traced_labels) != scores.shape[0]:
            continue

        grouped_cases[infer_kind_from_path(path)].append(
            {
                "known_id": known_id,
                "path": path,
                "scores": scores,
                "num_subject_tokens": num_subject_tokens,
                "traced_labels": traced_labels,
            }
        )

    return grouped_cases


def aggregate_focus_token_heatmap(
    cases: List[dict], last_k: int
) -> Tuple[np.ndarray, List[str]]:
    if last_k <= 0:
        raise ValueError("last_k must be positive.")

    row_buckets: List[List[np.ndarray]] = [[] for _ in range(2 + last_k)]
    row_labels = ["subject_first", "subject_last"] + [f"last_{k}" for k in range(1, last_k + 1)]

    for case in cases:
        scores = case["scores"]
        num_subject_tokens = case["num_subject_tokens"]
        subject_last_idx = num_subject_tokens - 1
        available_tail_tokens = scores.shape[0] - num_subject_tokens

        row_buckets[0].append(scores[0, :])
        row_buckets[1].append(scores[subject_last_idx, :])

        usable_k = min(last_k, available_tail_tokens)
        for offset in range(1, usable_k + 1):
            token_idx = scores.shape[0] - offset
            row_buckets[1 + offset].append(scores[token_idx, :])

    mean_rows = []
    final_labels = []
    for label, bucket in zip(row_labels, row_buckets):
        if not bucket:
            continue
        mean_rows.append(np.mean(np.stack(bucket, axis=0), axis=0))
        final_labels.append(label)

    if not mean_rows:
        raise ValueError("No valid cases provided for heatmap aggregation.")

    mean_heatmap = np.stack(mean_rows, axis=0)
    return mean_heatmap, final_labels


def plot_focus_token_heatmap(
    heatmap: np.ndarray,
    y_labels: List[str],
    kind: Optional[str],
    output_path: Path,
    case_count: int,
) -> None:
    cmap = {None: "Purples", "mlp": "Greens", "attn": "Reds"}[kind]

    with plt.rc_context(rc={"font.family": "Liberation Serif"}):
        fig, ax = plt.subplots(figsize=(3.5, 2), dpi=200)
        image = ax.pcolor(heatmap, cmap=cmap, vmin=0)
        ax.invert_yaxis()
        ax.set_yticks([0.5 + i for i in range(len(y_labels))])
        ax.set_yticklabels(y_labels)
        xtick_positions = [0.5 + i for i in range(0, max(1, heatmap.shape[1] - 1), 5)]
        xtick_labels = list(range(0, max(1, heatmap.shape[1] - 1), 5))
        if not xtick_positions:
            xtick_positions = [0.5]
            xtick_labels = [0]
        ax.set_xticks(xtick_positions)
        ax.set_xticklabels(xtick_labels)
        if kind is None:
            ax.set_title("Impact of aggregated key-token states")
            ax.set_xlabel("single restored layer within GPT")
        else:
            kind_name = "MLP" if kind == "mlp" else "Attn"
            ax.set_title(f"Impact of aggregated key-token {kind_name} states")
            ax.set_xlabel(f"center of interval of patched {kind_name} layers")
        cbar = plt.colorbar(image)
        cbar.ax.set_title("mean Δp", y=-0.16, fontsize=10)
        fig.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, bbox_inches="tight")
        plt.close(fig)


def collect_peak_statistics(cases: List[dict]) -> dict:
    peak_layers = []
    dist_to_subject_last = []
    dist_to_last_token = []

    for case in cases:
        scores = case["scores"]
        num_subject_tokens = case["num_subject_tokens"]
        peak_token_idx, peak_layer_idx = np.unravel_index(np.argmax(scores), scores.shape)

        subject_last_idx = num_subject_tokens - 1
        last_token_idx = scores.shape[0] - 1

        peak_layers.append(int(peak_layer_idx))
        dist_to_subject_last.append(abs(int(peak_token_idx) - subject_last_idx))
        dist_to_last_token.append(abs(int(peak_token_idx) - last_token_idx))

    return {
        "peak_layers": peak_layers,
        "dist_to_subject_last": dist_to_subject_last,
        "dist_to_last_token": dist_to_last_token,
    }


def counts_from_values(values: List[int]) -> Tuple[List[int], List[int]]:
    counter = Counter(values)
    xs = sorted(counter)
    ys = [counter[x] for x in xs]
    return xs, ys


def plot_peak_layer_distribution(
    stats: dict, kind: Optional[str], output_path: Path, case_count: int
) -> None:
    xs, ys = counts_from_values(stats["peak_layers"])

    with plt.rc_context(rc={"font.family": "Liberation Serif"}):
        fig, ax = plt.subplots(figsize=(6.5, 3.0), dpi=200)
        ax.bar(xs, ys, color="#4C72B0")
        ax.set_xlabel("Layer of maximum Δp")
        ax.set_ylabel("Count")
        ax.set_title(
            f"Peak layer distribution ({KIND_LABELS[kind]}, n={case_count})"
        )
        fig.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, bbox_inches="tight")
        plt.close(fig)


def plot_peak_distance_distribution(
    stats: dict, kind: Optional[str], output_path: Path, case_count: int
) -> None:
    xs_subject, ys_subject = counts_from_values(stats["dist_to_subject_last"])
    xs_last, ys_last = counts_from_values(stats["dist_to_last_token"])

    with plt.rc_context(rc={"font.family": "Liberation Serif"}):
        fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.0), dpi=200)

        axes[0].bar(xs_subject, ys_subject, color="#55A868")
        axes[0].set_xlabel("|peak token - subject last|")
        axes[0].set_ylabel("Count")
        axes[0].set_title("Distance to subject last")

        axes[1].bar(xs_last, ys_last, color="#C44E52")
        axes[1].set_xlabel("|peak token - last token|")
        axes[1].set_ylabel("Count")
        axes[1].set_title("Distance to last token")

        fig.suptitle(f"Peak token distance distributions ({KIND_LABELS[kind]}, n={case_count})")
        fig.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, bbox_inches="tight")
        plt.close(fig)


def analyze_kind(
    cases: List[dict], kind: Optional[str], output_dir: Path, last_k: int
) -> None:
    if not cases:
        tqdm.write(f"Skipping kind={KIND_LABELS[kind]} because no valid cases were found.")
        return

    heatmap, labels = aggregate_focus_token_heatmap(cases, last_k=last_k)
    stats = collect_peak_statistics(cases)
    kind_name = KIND_LABELS[kind]

    plot_focus_token_heatmap(
        heatmap,
        labels,
        kind,
        output_dir / f"focus_heatmap_{kind_name}.pdf",
        len(cases),
    )
    plot_peak_layer_distribution(
        stats,
        kind,
        output_dir / f"peak_layer_distribution_{kind_name}.pdf",
        len(cases),
    )
    plot_peak_distance_distribution(
        stats,
        kind,
        output_dir / f"peak_distance_distribution_{kind_name}.pdf",
        len(cases),
    )


def main() -> None:
    args = parse_args()
    run_dir, cases_dir = resolve_cases_dir(args.result_dir)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else run_dir / "analysis"
    )

    grouped_cases = load_case_results(cases_dir)
    for kind in tqdm([None, "mlp", "attn"], desc="Analyzing kinds"):
        tqdm.write(
            f"Analyzing kind={KIND_LABELS[kind]} with {len(grouped_cases.get(kind, []))} valid cases"
        )
        analyze_kind(grouped_cases.get(kind, []), kind, output_dir, last_k=args.last_k)

    print(f"Saved analysis plots to: {output_dir}")


if __name__ == "__main__":
    main()