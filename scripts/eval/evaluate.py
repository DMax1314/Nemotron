import argparse

import pandas as pd


def load_frame(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def resolve_truth_column(frame: pd.DataFrame, truth_column: str | None) -> str:
    if truth_column is not None:
        if truth_column not in frame.columns:
            raise ValueError(f"Truth column '{truth_column}' not found in {list(frame.columns)}")
        return truth_column

    for candidate in ("answer", "prediction"):
        if candidate in frame.columns:
            return candidate
    raise ValueError(
        "Could not infer the ground-truth column. Pass --truth-column explicitly."
    )


def score_submission(
    submission: pd.DataFrame,
    ground_truth: pd.DataFrame,
    prediction_column: str,
    truth_column: str,
    case_insensitive: bool,
) -> float:
    merged = submission.merge(ground_truth[["id", truth_column]], on="id", how="inner")
    predictions = merged[prediction_column].astype(str).str.strip()
    targets = merged[truth_column].astype(str).str.strip()
    if case_insensitive:
        predictions = predictions.str.lower()
        targets = targets.str.lower()
    return float((predictions == targets).mean())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Local evaluation for the Nemotron reasoning challenge"
    )
    parser.add_argument("--submission", required=True, help="Path to submission CSV")
    parser.add_argument(
        "--ground-truth", required=True, help="Path to a CSV with id and answer columns"
    )
    parser.add_argument(
        "--prediction-column",
        default="prediction",
        help="Prediction column name in the submission file",
    )
    parser.add_argument(
        "--truth-column",
        default=None,
        help="Ground-truth column name. Defaults to 'answer' or 'prediction' if present.",
    )
    parser.add_argument(
        "--metric",
        default="exact_match",
        choices=["exact_match", "case_insensitive"],
    )
    args = parser.parse_args()

    submission = load_frame(args.submission)
    ground_truth = load_frame(args.ground_truth)
    truth_column = resolve_truth_column(ground_truth, args.truth_column)

    score = score_submission(
        submission=submission,
        ground_truth=ground_truth,
        prediction_column=args.prediction_column,
        truth_column=truth_column,
        case_insensitive=args.metric == "case_insensitive",
    )

    print(f"Metric: {args.metric}")
    print(f"Truth column: {truth_column}")
    print(f"Score: {score:.4f}")


if __name__ == "__main__":
    main()
