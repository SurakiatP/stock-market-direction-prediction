# generate_feature_insights.py
"""
generate_feature_insights.py

Reads a model-ready dataset (sp500_cleaned_model_ready.csv),
analyzes feature distributions, correlations, etc.,
and saves a markdown file reports/feature_insights.md
"""

import pandas as pd
import os

def generate_feature_insights(input_path="data/processed/sp500_cleaned_model_ready.csv", output_path="reports/feature_insights.md"):
    df = pd.read_csv(input_path, index_col=0, parse_dates=True)
    
    # Exclude Target for feature-only analysis
    feature_cols = [c for c in df.columns if c != "Target"]
    df_features = df[feature_cols]

    # Correlation with Target (if it still exists in the DataFrame)
    if "Target" in df.columns:
        corr_with_target = df.corr(numeric_only=True)["Target"].sort_values(ascending=False)
    else:
        corr_with_target = pd.Series([])  # Empty if no Target

    # Distribution info (mean, std, min, max)
    distribution_stats = df_features.describe().T

    # Generate markdown
    md_content = []
    md_content.append("# Feature Insights\n")
    md_content.append(f"- **Number of features**: {len(feature_cols)}\n")
    md_content.append("\n## Distribution Stats\n")
    md_content.append("```\n" + distribution_stats.to_string() + "\n```\n")

    if not corr_with_target.empty:
        md_content.append("\n## Correlation with Target\n")
        md_content.append("```\n" + corr_with_target.to_string() + "\n```\n")

    # Ensure folder
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Write markdown
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_content))
    print(f" Feature insights generated at {output_path}")

def main():
    generate_feature_insights()

if __name__ == "__main__":
    main()
