import json
import matplotlib.pyplot as plt
import numpy as np
import os

# paths
multi_path = "results_multi/metrics.json"
single_path = "results_single/metrics.json"

# load both
multi = json.load(open(multi_path))
single = json.load(open(single_path))

def extract_summary(metrics, name):
    return {
        "Name": name,
        "Val AUC (subject)": metrics["val"]["subject"]["auc"],
        "Val Acc (subject)": metrics["val"]["subject"]["accuracy"],
        "Test AUC (subject)": metrics["test"]["subject"]["auc"],
        "Test Acc (subject)": metrics["test"]["subject"]["accuracy"],
    }

multi_summary = extract_summary(multi, "Multi-slice")
single_summary = extract_summary(single, "Single-slice")

# print table
print("\n📊 Summary Comparison:")
for key in multi_summary.keys():
    if key != "Name":
        print(f"{key:20s}: {single_summary['Name']:>12s}={single_summary[key]:.3f} | {multi_summary['Name']:>12s}={multi_summary[key]:.3f}")

# plot
labels = ["Val AUC", "Val Acc", "Test AUC", "Test Acc"]
single_vals = [
    single_summary["Val AUC (subject)"],
    single_summary["Val Acc (subject)"],
    single_summary["Test AUC (subject)"],
    single_summary["Test Acc (subject)"],
]
multi_vals = [
    multi_summary["Val AUC (subject)"],
    multi_summary["Val Acc (subject)"],
    multi_summary["Test AUC (subject)"],
    multi_summary["Test Acc (subject)"],
]

x = np.arange(len(labels))
width = 0.35
fig, ax = plt.subplots(figsize=(6,4))
ax.bar(x - width/2, single_vals, width, label="Single-slice")
ax.bar(x + width/2, multi_vals, width, label="Multi-slice")

ax.set_ylabel("Score")
ax.set_title("Comparison: Single vs Multi-slice Model")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
plt.tight_layout()

os.makedirs("results_compare", exist_ok=True)
plt.savefig("results_compare/comparison_barplot.png", dpi=300)
print("\n Saved comparison bar plot to results_compare/comparison_barplot.png")
