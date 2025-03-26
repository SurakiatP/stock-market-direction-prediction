# generate_eda_summary.py
"""
generate_eda_summary.py

This script reads a processed dataset (e.g. sp500_cleaned.csv),
performs basic EDA, and generates a markdown summary in reports/eda_summary.md
"""

import pandas as pd
import os

def generate_eda_summary(input_path="data/processed/sp500_cleaned.csv", output_path="reports/eda_summary.md"):
    # Load data
    df = pd.read_csv(input_path, index_col=0, parse_dates=True)
    
    # Basic info
    row_count = len(df)
    col_count = df.shape[1]
    start_date = df.index.min()
    end_date = df.index.max()

    # Check missing values
    missing_summary = df.isna().sum().sort_values(ascending=False).head(10)

    # Simple stats
    stats = df.describe().T

    # Generate markdown
    md_content = []
    md_content.append("# EDA Summary\n")
    md_content.append(f"- **Dataset shape**: {row_count} rows, {col_count} columns\n")
    md_content.append(f"- **Date range**: {start_date} to {end_date}\n")
    md_content.append("\n## Missing Values (Top 10)\n")
    md_content.append("```\n" + missing_summary.to_string() + "\n```\n")
    md_content.append("\n## Basic Statistics\n")
    md_content.append("```\n" + stats.to_string() + "\n```\n")

    # Ensure reports folder
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Write to markdown file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_content))
    print(f" EDA summary generated at {output_path}")

def main():
    generate_eda_summary()

if __name__ == "__main__":
    main()
