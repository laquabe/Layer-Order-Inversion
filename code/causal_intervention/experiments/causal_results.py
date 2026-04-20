import argparse
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter, PercentFormatter
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
        help="Directory containing causal_trace outputs. Can be a single run directory, its cases/ subdirectory, or an outer directory containing multiple run subdirectories.",
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


def resolve_run_dirs(result_dir: str) -> List[Path]:
    path = Path(result_dir)

    if path.name == "cases":
        return [path.parent]
    if (path / "cases").is_dir():
        return [path]

    run_dirs = sorted(
        child for child in path.iterdir() if child.is_dir() and (child / "cases").is_dir()
    )
    if not run_dirs:
        raise FileNotFoundError(
            f"Could not find a run directory or any child run directories under: {path}"
        )
    return run_dirs


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

        row_buckets[0].append(scores[0, :])
        row_buckets[1].append(scores[subject_last_idx, :])

        collected = 0
        for offset in range(1, scores.shape[0] + 1):
            token_idx = scores.shape[0] - offset
            if token_idx <= subject_last_idx:
                break
            collected += 1
            if collected > last_k:
                break
            row_buckets[1 + collected].append(scores[token_idx, :])

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

    fig, ax = plt.subplots(figsize=(5.6, 3.8), dpi=200)
    image = ax.pcolor(heatmap, cmap=cmap, vmin=0)
    ax.invert_yaxis()
    ax.set_yticks([0.5 + i for i in range(len(y_labels))])
    ax.set_yticklabels(y_labels, fontsize=9)
    ax.tick_params(axis="y", pad=6)
    xtick_positions = [0.5 + i for i in range(0, max(1, heatmap.shape[1] - 1), 5)]
    xtick_labels = list(range(0, max(1, heatmap.shape[1] - 1), 5))
    if not xtick_positions:
        xtick_positions = [0.5]
        xtick_labels = [0]
    ax.set_xticks(xtick_positions)
    ax.set_xticklabels(xtick_labels, fontsize=9)
    ax.tick_params(axis="x", pad=6)
    if kind is None:
        ax.set_title("Impact of aggregated key-token states", fontsize=12, pad=18)
        ax.set_xlabel("single restored layer within GPT", fontsize=10, labelpad=10)
    else:
        kind_name = "MLP" if kind == "mlp" else "Attn"
        ax.set_title(
            f"Impact of aggregated key-token {kind_name} states",
            fontsize=12,
            pad=18,
        )
        ax.set_xlabel(
            f"center of interval of patched {kind_name} layers",
            fontsize=10,
            labelpad=10,
        )
    max_abs_value = float(np.nanmax(np.abs(heatmap))) if heatmap.size else 0.0
    colorbar_exponent = int(np.floor(np.log10(max_abs_value))) if max_abs_value > 0 else 0
    colorbar_scale = 10 ** colorbar_exponent

    cbar = plt.colorbar(image, ax=ax, fraction=0.06, pad=0.08)
    cbar.formatter = FuncFormatter(
        lambda value, _pos: f"{(value / colorbar_scale):.1f}" if colorbar_scale != 0 else f"{value:.1f}"
    )
    cbar.update_ticks()
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label("mean Δp", fontsize=10, rotation=270, labelpad=16)
    cbar.ax.text(
        0.88,
        1.02,
        rf"$\times 10^{{{colorbar_exponent}}}$",
        transform=cbar.ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=8.5,
    )
    fig.subplots_adjust(left=0.24, right=0.88, bottom=0.28, top=0.76)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def collect_case_ratio_components(case: dict, last_k: int) -> Optional[Tuple[float, float]]:
    scores = case["scores"]
    num_subject_tokens = case["num_subject_tokens"]
    subject_last_idx = num_subject_tokens - 1

    subject_last_max = float(np.max(scores[subject_last_idx, :]))
    if subject_last_max <= 0:
        return None

    tail_rows = []
    collected = 0
    for offset in range(1, scores.shape[0] + 1):
        token_idx = scores.shape[0] - offset
        if token_idx <= subject_last_idx:
            break
        tail_rows.append(scores[token_idx, :])
        collected += 1
        if collected >= last_k:
            break

    if not tail_rows:
        return None

    last_k_max = float(np.max(np.stack(tail_rows, axis=0)))
    return last_k_max, subject_last_max


def summarize_subset_ratios(cases: List[dict], last_k: int) -> Optional[float]:
    """Return mean(last-k max) / mean(subject-last max) within one subset."""
    last_k_values = []
    subject_last_values = []
    for case in cases:
        components = collect_case_ratio_components(case, last_k=last_k)
        if components is not None:
            last_k_max, subject_last_max = components
            last_k_values.append(last_k_max)
            subject_last_values.append(subject_last_max)
    if not last_k_values or not subject_last_values:
        return None
    mean_last_k = float(np.mean(last_k_values))
    mean_subject_last = float(np.mean(subject_last_values))
    if mean_subject_last <= 0:
        return None
    return mean_last_k / mean_subject_last


def pretty_subset_name(run_dir: Path) -> str:
    name = run_dir.name
    for pattern in [r"(\d+hop)", r"([234]hop)"]:
        match = re.search(pattern, name)
        if match:
            return match.group(1)
    return name


def plot_subset_ratio_comparison(
    subset_scores: Dict[str, Dict[Optional[str], Optional[float]]],
    output_path: Path,
) -> None:
    subset_names = list(subset_scores.keys())
    kinds = [None, "mlp", "attn"]
    kind_display = {None: "base", "mlp": "mlp", "attn": "attn"}
    colors = {None: "#8172B2", "mlp": "#55A868", "attn": "#C44E52"}

    x = np.arange(len(subset_names))

    fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=200)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
    ax.set_axisbelow(True)

    markers = {None: "o", "mlp": "s", "attn": "^"}

    for kind in kinds:
        values = [subset_scores[name].get(kind) for name in subset_names]
        numeric_values = [np.nan if value is None else value for value in values]
        ax.plot(
            x,
            numeric_values,
            color=colors[kind],
            marker=markers[kind],
            linewidth=2.0,
            markersize=5.5,
            label=kind_display[kind],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(subset_names)
    ax.set_ylabel("Interference ratio (%)")
    ax.set_xlabel("Hop subset")
    ax.set_title("Tail-token interference across hop subsets")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.legend(frameon=False, ncol=3, loc="upper left")

    ymax = 0.0
    for subset in subset_scores.values():
        for value in subset.values():
            if value is not None:
                ymax = max(ymax, value)
    ax.set_ylim(0, max(0.1, ymax * 1.15))

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
    kind_name = KIND_LABELS[kind]

    plot_focus_token_heatmap(
        heatmap,
        labels,
        kind,
        output_dir / f"focus_heatmap_{kind_name}.pdf",
        len(cases),
    )


def analyze_run_dir(run_dir: Path, output_dir: Path, last_k: int) -> Dict[Optional[str], Optional[float]]:
    _, cases_dir = resolve_cases_dir(str(run_dir))
    grouped_cases = load_case_results(cases_dir)
    ratio_summary: Dict[Optional[str], Optional[float]] = {}

    for kind in tqdm([None, "mlp", "attn"], desc=f"Analyzing {run_dir.name}"):
        cases = grouped_cases.get(kind, [])
        tqdm.write(
            f"Analyzing run={run_dir.name}, kind={KIND_LABELS[kind]} with {len(cases)} valid cases"
        )
        analyze_kind(cases, kind, output_dir, last_k=last_k)
        ratio_summary[kind] = summarize_subset_ratios(cases, last_k=last_k)

    return ratio_summary


def main() -> None:
    args = parse_args()
    run_dirs = resolve_run_dirs(args.result_dir)
    input_root = Path(args.result_dir)
    is_outer_dir = len(run_dirs) > 1 or not (input_root / "cases").is_dir()

    if not is_outer_dir:
        run_dir = run_dirs[0]
        output_dir = (
            Path(args.output_dir) if args.output_dir is not None else run_dir / "analysis"
        )
        analyze_run_dir(run_dir, output_dir, last_k=args.last_k)
        print(f"Saved analysis plots to: {output_dir}")
        return

    root_output_dir = (
        Path(args.output_dir) if args.output_dir is not None else input_root / "analysis"
    )
    subset_scores: Dict[str, Dict[Optional[str], Optional[float]]] = {}

    for run_dir in tqdm(run_dirs, desc="Analyzing subset runs"):
        run_output_dir = root_output_dir / run_dir.name
        subset_scores[pretty_subset_name(run_dir)] = analyze_run_dir(
            run_dir, run_output_dir, last_k=args.last_k
        )

    plot_subset_ratio_comparison(
        subset_scores,
        root_output_dir / f"subset_ratio_comparison_lastk_{args.last_k}.pdf",
    )
    print(f"Saved analysis plots to: {root_output_dir}")


if __name__ == "__main__":
    main()