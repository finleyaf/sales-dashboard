from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def repo_root() -> Path:
    # src/ -> repo root
    return Path(__file__).resolve().parents[1]


def load_raw(input_path: Path | None = None) -> pd.DataFrame:
    if input_path is None:
        input_path = repo_root() / "data" / "online_retail_raw.csv"
    return pd.read_csv(input_path, encoding="ISO-8859-1")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    # Drop rows where Description is missing
    df = df.dropna(subset=["Description"])

    # Convert invoice date to datetime
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    # Filter non-positive quantities and prices
    df = df[df["Quantity"] > 0]
    df = df[df["Price"] > 0]

    # Handle missing Customer ID
    df = df.dropna(subset=["Customer ID"])
    df["Customer ID"] = df["Customer ID"].astype(int)

    # Feature engineering
    df["TotalPrice"] = df["Quantity"] * df["Price"]
    df["InvoiceYear"] = df["InvoiceDate"].dt.year
    df["InvoiceMonth"] = df["InvoiceDate"].dt.month
    df["YearMonth"] = df["InvoiceDate"].dt.to_period("M")
    df["InvoiceWeek"] = df["InvoiceDate"].dt.isocalendar().week
    df["InvoiceDay"] = df["InvoiceDate"].dt.day_name()

    # Tidy text
    df["Description"] = df["Description"].str.strip()

    # Finalize
    df = df.reset_index(drop=True)
    return df


def save_clean(df: pd.DataFrame, output_path: Path | None = None) -> Path:
    if output_path is None:
        output_path = repo_root() / "data" / "online_retail_clean.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prepare and clean online retail data.")
    parser.add_argument(
        "-i", "--input",
        type=Path,
        default=repo_root() / "data" / "online_retail_raw.csv",
        help="Path to raw CSV (default: data/online_retail_raw.csv)",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=repo_root() / "data" / "online_retail_clean.csv",
        help="Path to write cleaned CSV (default: data/online_retail_clean.csv)",
    )
    args = parser.parse_args(argv)

    df = load_raw(args.input)
    df = clean(df)
    out = save_clean(df, args.output)
    print(f"Wrote cleaned data to: {out}")


if __name__ == "__main__":
    main()