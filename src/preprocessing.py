"""
preprocessing.py
----------------
Load raw STRING TSV files, clean the data, and filter edges by combined score.
"""

import pandas as pd


def load_string_tsv(filepath: str) -> pd.DataFrame:
    """Load a STRING database TSV file into a DataFrame."""
    df = pd.read_csv(filepath, sep="\t")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate edges and rows with missing values."""
    df = df.dropna()
    df = df.drop_duplicates()
    return df


def filter_edges(df: pd.DataFrame, score_col: str = "combined_score", threshold: int = 700) -> pd.DataFrame:
    """Keep only edges whose combined score meets the threshold (default 700/1000)."""
    df = df[df[score_col] >= threshold].reset_index(drop=True)
    return df


def preprocess(filepath: str, score_col: str = "combined_score", threshold: int = 700) -> pd.DataFrame:
    """Full preprocessing pipeline: load, clean, and filter a STRING TSV file."""
    df = load_string_tsv(filepath)
    df = clean_data(df)
    df = filter_edges(df, score_col=score_col, threshold=threshold)
    return df


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python preprocessing.py <input_tsv> <output_csv>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    result = preprocess(input_path)
    result.to_csv(output_path, index=False)
    print(f"Preprocessed {len(result)} edges saved to {output_path}")
