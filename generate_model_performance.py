# generate_model_performance.py
"""
generate_model_performance.py

This script reads the model_training_results.csv (or any CSV that stores
model performance) and generates a markdown report summarizing the best model,
precision scores, etc.
"""

import pandas as pd
import os

def generate_model_performance(input_path="reports/model_training_results.csv", output_path="reports/model_performance.md"):
    if not os.path.exists(input_path):
        print(f" Model training results not found at {input_path}")
        return
    
    df = pd.read_csv(input_path)
    df_sorted = df.sort_values(by="Best Precision", ascending=False)

    best_model = df_sorted.iloc[0]["Model"]
    best_precision = df_sorted.iloc[0]["Best Precision"]
    best_params = df_sorted.iloc[0]["Best Params"]

    # Generate markdown
    md_content = []
    md_content.append("# Model Performance\n")
    md_content.append("## Overall Results\n")
    md_content.append("```\n" + df_sorted.to_string(index=False) + "\n```\n")
    md_content.append(f"\n## Best Model\n- **Name**: {best_model}\n- **Precision**: {best_precision:.4f}\n- **Params**: {best_params}\n")

    # Ensure folder
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_content))
    print(f" Model performance report generated at {output_path}")

def main():
    generate_model_performance()

if __name__ == "__main__":
    main()
