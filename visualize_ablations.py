import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Exact result values from the ablation test run
data = {
    "Run": [
        "Ours (RGAT + Focal)", 
        "GCN + Focal", 
        "RGAT + CE", 
        "GCN + CE", 
        "RGAT + DQN", 
        "No Semantic", 
        "All Severity"
    ],
    "Accuracy": [0.375, 0.250, 0.250, 0.000, 0.125, 0.000, 0.250],
    "Macro_F1": [0.2667, 0.1467, 0.2133, 0.0000, 0.0444, 0.0000, 0.0800],
    "F1_GodClass": [0.000, 0.400, 0.000, 0.000, 0.000, 0.000, 0.000],
    "F1_FeatureEnvy": [0.6667, 0.3333, 0.6667, 0.000, 0.000, 0.000, 0.400],
    "F1_LongMethod": [0.6667, 0.000, 0.400, 0.000, 0.000, 0.000, 0.000],
    "F1_DataClass": [0.000, 0.000, 0.000, 0.000, 0.2222, 0.000, 0.000]
}

df = pd.DataFrame(data)

# Set the style
sns.set_theme(style="whitegrid")
output_dir = "data/ablations/visualizations"
os.makedirs(output_dir, exist_ok=True)

# 1. Overall Performance (Accuracy & Macro-F1)
plt.figure(figsize=(12, 6))
x = np.arange(len(df))
width = 0.35

plt.bar(x - width/2, df["Accuracy"], width, label='Accuracy', color='#3498db')
plt.bar(x + width/2, df["Macro_F1"], width, label='Macro F1', color='#e74c3c')

plt.ylabel('Score')
plt.title('Ablation Study: Overall Accuracy and Macro F1-Score')
plt.xticks(x, df["Run"], rotation=45, ha="right")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'ablation_overall_performance.png'), dpi=300)
plt.close()

# 2. Component Analysis (Encoder & Loss)
components = pd.DataFrame({
    "Configuration": ["GCN + CE", "GCN + Focal", "RGAT + CE", "Ours (RGAT + Focal)"],
    "Macro_F1": [0.0000, 0.1467, 0.2133, 0.2667]
})

plt.figure(figsize=(8, 5))
sns.barplot(x="Configuration", y="Macro_F1", hue="Configuration", data=components, palette="viridis", legend=False)
plt.title('Impact of R-GAT and Focal Loss on Macro-F1')
plt.ylabel('Macro F1-Score')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'ablation_component_impact.png'), dpi=300)
plt.close()

# 3. Per-Class F1 Scores Heatmap
heatmap_data = df.set_index("Run")[["F1_GodClass", "F1_FeatureEnvy", "F1_LongMethod", "F1_DataClass"]]
plt.figure(figsize=(10, 6))
sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", fmt=".3f", linewidths=.5)
plt.title('Per-Class F1 Scores Across Ablations')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'ablation_per_class_f1.png'), dpi=300)
plt.close()

print(f"Visualizations successfully generated in {output_dir}")
