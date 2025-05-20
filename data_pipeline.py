from __future__ import annotations

"""Generate engineered features for call volume forecasting.

This command-line utility merges the raw calls, visitors and chatbot query
CSV files into a single business-day indexed dataset with dummy variables for
holidays, notice-of-value mail-outs and other campaign periods. The result is
written to a CSV file that can be used for modeling or further analysis.
"""

from pathlib import Path
import argparse

from data_preparation import prepare_data


def main(calls: Path, visitors: Path, queries: Path, out_path: Path) -> None:
    """Create features from raw input files and save them to ``out_path``."""
    df, _ = prepare_data(calls, visitors, queries, scale_features=True)
    df.to_csv(out_path, index_label="date")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate engineered feature CSV for forecasting"
    )
    parser.add_argument("calls", type=Path, help="Path to calls CSV file")
    parser.add_argument("visitors", type=Path, help="Path to visitors CSV file")
    parser.add_argument("queries", type=Path, help="Path to chatbot queries CSV file")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("engineered_features.csv"),
        help="Output CSV file path",
    )
    args = parser.parse_args()
    main(args.calls, args.visitors, args.queries, args.out)
