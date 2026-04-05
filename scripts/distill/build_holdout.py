from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.distill.common import DEFAULT_DISTILL_DIR, load_with_family


def stratified_holdout(
    dataframe: pd.DataFrame,
    eval_ratio: float,
    seed: int,
    min_eval_per_family: int,
    max_eval_per_family: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    eval_parts: list[pd.DataFrame] = []
    target_counts: dict[str, int] = {}

    for family, family_df in dataframe.groupby("family", sort=True):
        if len(family_df) <= 1:
            continue

        target = round(len(family_df) * eval_ratio)
        target = max(target, min_eval_per_family)
        target = min(target, max_eval_per_family)
        target = min(target, len(family_df) - 1)

        target_counts[family] = int(target)
        eval_parts.append(family_df.sample(n=target, random_state=seed))

    eval_df = pd.concat(eval_parts, ignore_index=False).sort_index()
    train_df = dataframe.loc[~dataframe.index.isin(eval_df.index)].copy()
    eval_df = eval_df.copy()
    return train_df, eval_df, target_counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a family-stratified local holdout split for distillation experiments."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("data/raw/train.csv"),
        help="Source CSV with prompt/answer pairs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_DISTILL_DIR / "holdout_v1",
        help="Directory to write the train/eval split artifacts.",
    )
    parser.add_argument("--eval-ratio", type=float, default=0.10)
    parser.add_argument("--min-eval-per-family", type=int, default=120)
    parser.add_argument("--max-eval-per-family", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_csv = args.input_csv.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataframe = load_with_family(input_csv)
    train_df, eval_df, target_counts = stratified_holdout(
        dataframe=dataframe,
        eval_ratio=args.eval_ratio,
        seed=args.seed,
        min_eval_per_family=args.min_eval_per_family,
        max_eval_per_family=args.max_eval_per_family,
    )

    train_path = output_dir / "train_split.csv"
    eval_path = output_dir / "eval_split.csv"
    report_path = output_dir / "split_report.json"

    train_df.to_csv(train_path, index=False)
    eval_df.to_csv(eval_path, index=False)

    report = {
        "input_csv": str(input_csv),
        "output_dir": str(output_dir),
        "seed": args.seed,
        "eval_ratio": args.eval_ratio,
        "min_eval_per_family": args.min_eval_per_family,
        "max_eval_per_family": args.max_eval_per_family,
        "rows_total": int(len(dataframe)),
        "rows_train": int(len(train_df)),
        "rows_eval": int(len(eval_df)),
        "family_counts_full": dataframe["family"].value_counts().sort_index().to_dict(),
        "family_counts_train": train_df["family"].value_counts().sort_index().to_dict(),
        "family_counts_eval": eval_df["family"].value_counts().sort_index().to_dict(),
        "family_target_eval": target_counts,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Wrote holdout train split: {train_path}")
    print(f"Wrote holdout eval split:  {eval_path}")
    print(f"Wrote split report:       {report_path}")


if __name__ == "__main__":
    main()
