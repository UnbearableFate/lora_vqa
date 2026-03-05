from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict
from statistics import fmean
from typing import DefaultDict, Dict, Iterable, List, Set, Tuple

DEFAULT_GROUP_KEYS = [
    "base_model",
    "dataset_name",
    "init_lora_weights",
    "extra",
    "r",
    "lora_alpha",
]


def _parse_float(value: str) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _std(values: List[float], sample: bool = False) -> float:
    n = len(values)
    if n == 0:
        return float("nan")
    if n == 1:
        return 0.0
    mean = fmean(values)
    sq = sum((x - mean) ** 2 for x in values)
    denom = (n - 1) if sample else n
    return math.sqrt(sq / denom)


def _format_fixed_5(value: float) -> str:
    return f"{value:.5f}"


def _default_output_path(input_path: str) -> str:
    root, ext = os.path.splitext(input_path)
    ext = ext or ".csv"
    return f"{root}_seed_stats{ext}"


def _default_missing_seed_report_path(output_csv: str) -> str:
    root, _ = os.path.splitext(output_csv)
    return f"{root}_missing_seeds.txt"


def _sort_seed_strings(seeds: Iterable[str]) -> List[str]:
    def _seed_key(seed: str) -> Tuple[int, int | str]:
        try:
            return (0, int(seed))
        except ValueError:
            return (1, seed)

    return sorted(seeds, key=_seed_key)


def summarize_eval_results(
    input_csv: str,
    output_csv: str | None = None,
    value_column: str = "metric_exact_match",
    group_keys: Iterable[str] = DEFAULT_GROUP_KEYS,
    sample_std: bool = False,
) -> Tuple[str, str]:
    group_keys = list(group_keys)

    with open(input_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"No header found in CSV: {input_csv}")
        fieldnames = list(reader.fieldnames)

        missing = [k for k in [*group_keys, "seed", value_column] if k not in fieldnames]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # group -> seed -> values
        grouped: DefaultDict[
            Tuple[str, ...],
            DefaultDict[str, List[float]],
        ] = defaultdict(lambda: defaultdict(list))
        group_seed_seen: DefaultDict[Tuple[str, ...], Set[str]] = defaultdict(set)
        row_count_by_group: DefaultDict[Tuple[str, ...], int] = defaultdict(int)
        all_seeds: Set[str] = set()

        for row in reader:
            group = tuple((row.get(k) or "").strip() for k in group_keys)
            seed = (row.get("seed") or "").strip()
            if not seed:
                continue
            all_seeds.add(seed)
            group_seed_seen[group].add(seed)
            row_count_by_group[group] += 1
            value = _parse_float(row.get(value_column, ""))
            if value is not None:
                grouped[group][seed].append(value)

    output_csv = output_csv or _default_output_path(input_csv)
    missing_seed_report = _default_missing_seed_report_path(output_csv)

    out_fields = list(group_keys) #+ ["seed_count", "row_count"]
    out_fields.append(f"{value_column}_mean")
    out_fields.append(f"{value_column}_std")

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()

        output_rows: List[Dict[str, str | int | float]] = []
        for group in sorted(group_seed_seen.keys()):
            out_row: Dict[str, str | int | float] = {
                key: value for key, value in zip(group_keys, group)
            }
            seeds = group_seed_seen[group]
            #out_row["seed_count"] = len(seeds)
            #out_row["row_count"] = row_count_by_group[group]

            metric_values: List[float] = []
            for vals in grouped[group].values():
                metric_values.extend(vals)

            if metric_values:
                mean_value = fmean(metric_values)
                std_value = _std(metric_values, sample=sample_std)
                out_row[f"{value_column}_mean"] = _format_fixed_5(mean_value)
                out_row[f"{value_column}_std"] = _format_fixed_5(std_value)
            else:
                out_row[f"{value_column}_mean"] = ""
                out_row[f"{value_column}_std"] = ""

            output_rows.append(out_row)

        mean_col = f"{value_column}_mean"

        def _sort_key(row: Dict[str, str | int | float]) -> float:
            value = row.get(mean_col, "")
            if value == "":
                return float("-inf")
            return float(value)

        output_rows.sort(key=_sort_key, reverse=True)

        for row in output_rows:
            writer.writerow(row)

    sorted_all_seeds = _sort_seed_strings(all_seeds)
    with open(missing_seed_report, "w", encoding="utf-8") as f:
        f.write(f"input_csv: {input_csv}\n")
        f.write(f"output_csv: {output_csv}\n")
        f.write(f"value_column: {value_column}\n")
        f.write(f"all_seeds({len(sorted_all_seeds)}): {', '.join(sorted_all_seeds)}\n\n")

        missing_groups = 0
        for group in sorted(group_seed_seen.keys()):
            present = group_seed_seen[group]
            missing = [s for s in sorted_all_seeds if s not in present]
            if not missing:
                continue
            missing_groups += 1
            group_desc = ", ".join(f"{k}={v}" for k, v in zip(group_keys, group))
            f.write(f"{group_desc}\n")
            f.write(f"missing_seeds({len(missing)}): {', '.join(missing)}\n\n")

        if missing_groups == 0:
            f.write("All groups contain all observed seeds.\n")

    return output_csv, missing_seed_report


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate eval_results.csv by group keys and compute mean/standard deviation "
            "for one specified column across seeds."
        )
    )
    parser.add_argument("--input", required=True, help="Path to input eval_results.csv")
    parser.add_argument(
        "--output",
        default=None,
        help="Path to output summary CSV (default: <input>_seed_stats.csv).",
    )
    parser.add_argument(
        "--vc",
        default="metric_exact_match",
        help="Single numeric column to aggregate (for example: metric_exact_match).",
    )
    parser.add_argument(
        "--group-keys",
        nargs="+",
        default=DEFAULT_GROUP_KEYS,
        help=(
            "Columns used as the group key. "
            "Default: base_model dataset_name init_lora_weights extra r lora_alpha"
        ),
    )
    parser.add_argument(
        "--sample-std",
        "--sample-var",
        dest="sample_std",
        action="store_true",
        help="Use sample standard deviation (N-1). Default is population standard deviation (N).",
    )
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    output_path, missing_seed_report = summarize_eval_results(
        input_csv=args.input,
        output_csv=args.output,
        value_column=args.vc,
        group_keys=args.group_keys,
        sample_std=args.sample_std,
    )
    print(f"Saved summary to: {output_path}")
    print(f"Saved missing-seed report to: {missing_seed_report}")


if __name__ == "__main__":
    main()
