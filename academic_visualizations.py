import matplotlib.pyplot as plt
import numpy as np
import os

# Academic publication settings (IEEE/Scopus standards)
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.dpi': 600,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight'
})

output_dir = "data/ablations/visualizations"
os.makedirs(output_dir, exist_ok=True)

# Data: Semantics vs Non-Semantics
categories = ['Macro-Average', 'Feature Envy', 'Long Method']
with_semantics = [0.2667, 0.6667, 0.6667]  # full_rgat_focal
without_semantics = [0.0000, 0.0000, 0.0000] # rgat_focal_no_semantic

x = np.arange(len(categories))
width = 0.35

fig, ax = plt.subplots(figsize=(7, 5))

# Plot bars with distinctive academic hatch patterns for B&W printing compatibility
rects1 = ax.bar(x - width/2, with_semantics, width, label='With Semantic Features (GraphCodeBERT)', 
                color='#34495e', edgecolor='black', hatch='//')
rects2 = ax.bar(x + width/2, without_semantics, width, label='Structural Features Only (Halstead)', 
                color='#ecf0f1', edgecolor='black', hatch='..')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('F1-Score')
ax.set_title('Impact of Semantic Embeddings on Code Smell Detection')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.set_ylim(0, 1.0)
ax.legend(loc='upper right', framealpha=1, edgecolor='black')

# Function to auto-label bars
def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11)

autolabel(rects1)
autolabel(rects2)

fig.tight_layout()

# Save in high-quality formats for Scopus-indexed publications
pdf_path = os.path.join(output_dir, 'fig_semantics_vs_structural.pdf')
png_path = os.path.join(output_dir, 'fig_semantics_vs_structural.png')
svg_path = os.path.join(output_dir, 'fig_semantics_vs_structural.svg')

plt.savefig(pdf_path, format='pdf')
plt.savefig(png_path, format='png')
plt.savefig(svg_path, format='svg')
plt.close()

print(f"Publication-ready visualizations saved to {output_dir}")
