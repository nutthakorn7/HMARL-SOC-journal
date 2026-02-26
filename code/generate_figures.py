#!/usr/bin/env python3
"""
Generate all figures for HMARL-SOC IEEE Access paper.
Reads CSV data from checkpoints/ and outputs PDF figures to ../paper/figures/
"""

import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch

# ── Config ──────────────────────────────────────────────────────────
SEEDS = [42, 123, 456, 789, 1024]
CKPT = "checkpoints"
OUT = "../paper/figures"
EVAL_WINDOW = 2000

METHODS = {
    "Rule-SOAR":   "train_rule_soar_seed",
    "Single-DRL":  "train_single_drl_seed",
    "IQL":         "train_iql_seed",
    "MAPPO":       "train_mappo_seed",
    "QMIX":        "train_qmix_seed",
    "HMARL-SOC":   "train_seed",
}

ABLATION = {
    "Full HMARL-SOC": "train_seed",
    "w/o SC":          "train_wo_sc_seed",
    "w/o Buffer":      "train_wo_buf_seed",
    "Flat MARL":       "train_flat_marl_seed",
}

# Professional color palette
COLORS = {
    "Rule-SOAR":   "#95a5a6",
    "Single-DRL":  "#e67e22",
    "IQL":         "#9b59b6",
    "MAPPO":       "#3498db",
    "QMIX":        "#e74c3c",
    "HMARL-SOC":   "#2ecc71",
}

ABL_COLORS = {
    "Full HMARL-SOC": "#2ecc71",
    "w/o SC":          "#3498db",
    "w/o Buffer":      "#e74c3c",
    "Flat MARL":       "#e67e22",
}

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

os.makedirs(OUT, exist_ok=True)


# ── Data Loading ────────────────────────────────────────────────────
def load_all(methods_dict, window=EVAL_WINDOW):
    """Load tail-window averages per seed for each method."""
    results = {}
    for name, prefix in methods_dict.items():
        seeds_data = {"reward": [], "mttd": [], "mttr": [], "fpr": [], "csr": []}
        for seed in SEEDS:
            csv_path = os.path.join(CKPT, f"{prefix}{seed}.csv")
            if not os.path.exists(csv_path):
                print(f"  WARNING: {csv_path} not found, skipping")
                continue
            data = np.genfromtxt(csv_path, delimiter=",", names=True)
            tail = data[-min(len(data), window):]
            seeds_data["reward"].append(np.mean(tail["reward"]))
            seeds_data["mttd"].append(np.mean(tail["mttd"]))
            seeds_data["mttr"].append(np.mean(tail["mttr"]))
            seeds_data["fpr"].append(np.mean(tail["fpr"]) * 100)
            seeds_data["csr"].append(np.mean(tail["csr"]) * 100)
        results[name] = {k: np.array(v) for k, v in seeds_data.items()}
    return results


def load_timeseries(methods_dict, metric="reward", smooth=200):
    """Load full episode timeseries for learning curves."""
    ts = {}
    for name, prefix in methods_dict.items():
        all_series = []
        min_len = float('inf')
        for seed in SEEDS:
            csv_path = os.path.join(CKPT, f"{prefix}{seed}.csv")
            if not os.path.exists(csv_path):
                continue
            data = np.genfromtxt(csv_path, delimiter=",", names=True)
            all_series.append(data[metric])
            min_len = min(min_len, len(data[metric]))
        # Truncate to min length
        truncated = [s[:min_len] for s in all_series]
        arr = np.array(truncated)
        # Smooth with rolling mean
        if smooth > 0 and min_len > smooth:
            kernel = np.ones(smooth) / smooth
            smoothed = np.array([np.convolve(row, kernel, mode='valid') for row in arr])
            ts[name] = smoothed
        else:
            ts[name] = arr
    return ts


# ── Load Data ───────────────────────────────────────────────────────
print("Loading data...")
main_data = load_all(METHODS)
abl_data = load_all(ABLATION)


# ── Fig 3: Radar Chart ──────────────────────────────────────────────
print("Generating Fig 3: Radar Chart...")

metrics = ["Reward", "MTTD", "MTTR", "FPR", "CSR"]
# For radar: higher is better, so invert MTTD, MTTR, FPR
raw_vals = {}
for name, d in main_data.items():
    raw_vals[name] = [
        d["reward"].mean(),
        -d["mttd"].mean(),   # invert: lower is better
        -d["mttr"].mean(),   # invert: lower is better
        -d["fpr"].mean(),    # invert: lower is better
        d["csr"].mean(),
    ]

# Normalize to 0-1
all_vals = np.array(list(raw_vals.values()))
mins = all_vals.min(axis=0)
maxs = all_vals.max(axis=0)
ranges = maxs - mins
ranges[ranges == 0] = 1

normalized = {}
for name, vals in raw_vals.items():
    normalized[name] = [(v - mi) / r for v, mi, r in zip(vals, mins, ranges)]

# Radar plot
angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
angles += angles[:1]  # close the polygon

fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

for name in ["HMARL-SOC", "QMIX", "MAPPO", "IQL", "Single-DRL", "Rule-SOAR"]:
    vals = normalized[name] + normalized[name][:1]
    lw = 2.5 if name == "HMARL-SOC" else 1.2
    alpha_fill = 0.15 if name == "HMARL-SOC" else 0.05
    ax.plot(angles, vals, 'o-', linewidth=lw, label=name, color=COLORS[name], markersize=4)
    ax.fill(angles, vals, alpha=alpha_fill, color=COLORS[name])

display_metrics = ["Reward↑", "MTTD↓", "MTTR↓", "FPR↓", "CSR↑"]
ax.set_xticks(angles[:-1])
ax.set_xticklabels(display_metrics, fontsize=10, fontweight='bold')
ax.set_ylim(0, 1.05)
ax.set_yticks([0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=7)
ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.12), frameon=True, 
          fancybox=True, shadow=True, fontsize=8)
ax.grid(True, alpha=0.3)
plt.savefig(os.path.join(OUT, "radar_comparison.pdf"))
plt.close()
print("  -> radar_comparison.pdf")


# ── Fig 4: Ablation Bar Chart ───────────────────────────────────────
print("Generating Fig 4: Ablation Bar Chart...")

fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
abl_names = list(ABLATION.keys())
x = np.arange(len(abl_names))
width = 0.6

# Panel (a): Reward
ax = axes[0]
means = [abl_data[n]["reward"].mean() for n in abl_names]
stds = [abl_data[n]["reward"].std() for n in abl_names]
bars = ax.bar(x, means, width, yerr=stds, capsize=4,
              color=[ABL_COLORS[n] for n in abl_names], edgecolor='black', linewidth=0.5)
ax.set_ylabel("Cumulative Reward")
ax.set_title("(a) Reward", fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(abl_names, rotation=20, ha='right', fontsize=8)
ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
ax.grid(axis='y', alpha=0.3)

# Panel (b): FPR
ax = axes[1]
means = [abl_data[n]["fpr"].mean() for n in abl_names]
stds = [abl_data[n]["fpr"].std() for n in abl_names]
bars = ax.bar(x, means, width, yerr=stds, capsize=4,
              color=[ABL_COLORS[n] for n in abl_names], edgecolor='black', linewidth=0.5)
ax.set_ylabel("FPR (%)")
ax.set_title("(b) False Positive Rate", fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(abl_names, rotation=20, ha='right', fontsize=8)
ax.grid(axis='y', alpha=0.3)

# Panel (c): CSR
ax = axes[2]
means = [abl_data[n]["csr"].mean() for n in abl_names]
stds = [abl_data[n]["csr"].std() for n in abl_names]
bars = ax.bar(x, means, width, yerr=stds, capsize=4,
              color=[ABL_COLORS[n] for n in abl_names], edgecolor='black', linewidth=0.5)
ax.set_ylabel("CSR (%)")
ax.set_title("(c) Containment Success", fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(abl_names, rotation=20, ha='right', fontsize=8)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUT, "ablation_bars.pdf"))
plt.close()
print("  -> ablation_bars.pdf")


# ── Fig 5: Box Plots ────────────────────────────────────────────────
print("Generating Fig 5: Box Plots...")

fig, axes = plt.subplots(2, 2, figsize=(9, 6))
metric_labels = [
    ("reward", "Cumulative Reward", "(a) Reward"),
    ("fpr", "FPR (%)", "(b) False Positive Rate"),
    ("csr", "CSR (%)", "(c) Containment Success Rate"),
    ("mttd", "MTTD (steps)", "(d) Mean Time to Detect"),
]

method_order = ["Rule-SOAR", "Single-DRL", "IQL", "MAPPO", "QMIX", "HMARL-SOC"]

for idx, (metric, ylabel, title) in enumerate(metric_labels):
    ax = axes[idx // 2][idx % 2]
    data_list = [main_data[m][metric] for m in method_order]
    bp = ax.boxplot(data_list, patch_artist=True, widths=0.5,
                    medianprops=dict(color='black', linewidth=1.5))
    for patch, name in zip(bp['boxes'], method_order):
        patch.set_facecolor(COLORS[name])
        patch.set_alpha(0.7)
    ax.set_xticklabels([n.replace("-", "-\n") if len(n) > 8 else n 
                         for n in method_order], fontsize=7, rotation=0)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUT, "boxplots_comparison.pdf"))
plt.close()
print("  -> boxplots_comparison.pdf")


# ── Fig 6: Ablation Learning Curves ─────────────────────────────────
print("Generating Fig 6: Ablation Learning Curves...")

abl_ts = load_timeseries(ABLATION, metric="reward", smooth=100)

fig, ax = plt.subplots(figsize=(7, 4))
for name in ["Full HMARL-SOC", "w/o SC", "w/o Buffer", "Flat MARL"]:
    arr = abl_ts[name]
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    episodes = np.arange(len(mean))
    lw = 2.0 if name == "Full HMARL-SOC" else 1.2
    ax.plot(episodes, mean, label=name, color=ABL_COLORS[name], linewidth=lw)
    ax.fill_between(episodes, mean - std, mean + std, 
                    alpha=0.15, color=ABL_COLORS[name])

ax.set_xlabel("Episode")
ax.set_ylabel("Cumulative Reward (smoothed)")
ax.set_title("Ablation Study: Training Convergence", fontweight='bold')
ax.legend(frameon=True, fancybox=True, shadow=True)
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "ablation_learning_curves.pdf"))
plt.close()
print("  -> ablation_learning_curves.pdf")


# ── Fig 7: FPR vs CSR Scatter ───────────────────────────────────────
print("Generating Fig 7: FPR vs CSR Scatter...")

fig, ax = plt.subplots(figsize=(6, 4.5))

for name in method_order:
    d = main_data[name]
    fpr_vals = d["fpr"]
    csr_vals = d["csr"]
    
    # Plot individual seeds as small dots
    ax.scatter(fpr_vals, csr_vals, c=COLORS[name], alpha=0.4, s=30, zorder=2)
    
    # Plot mean as large marker
    ax.scatter(fpr_vals.mean(), csr_vals.mean(), c=COLORS[name], 
               s=150, edgecolors='black', linewidth=1.2, zorder=3,
               label=f"{name} ({fpr_vals.mean():.2f}%, {csr_vals.mean():.1f}%)",
               marker='o')
    
    # Error bars for mean
    ax.errorbar(fpr_vals.mean(), csr_vals.mean(),
                xerr=fpr_vals.std(), yerr=csr_vals.std(),
                c=COLORS[name], capsize=3, linewidth=1, zorder=2, fmt='none')

# Mark ideal quadrant
ax.axvspan(-0.5, 1.0, alpha=0.05, color='green')
ax.axhspan(65, 105, alpha=0.05, color='green')
ax.text(0.3, 100, "Ideal\nQuadrant", fontsize=8, color='green', 
        alpha=0.6, ha='center', fontweight='bold')

ax.set_xlabel("False Positive Rate (%)")
ax.set_ylabel("Containment Success Rate (%)")
ax.set_title("FPR vs CSR Trade-off", fontweight='bold')
ax.legend(loc='lower right', fontsize=7, frameon=True, fancybox=True)
ax.grid(True, alpha=0.3)
ax.set_xlim(-0.3, 6.0)
ax.set_ylim(25, 105)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fpr_csr_scatter.pdf"))
plt.close()
print("  -> fpr_csr_scatter.pdf")


# ── Done ────────────────────────────────────────────────────────────
print("\n✅ All 5 figures generated successfully!")
print(f"Output directory: {OUT}")
for f in sorted(os.listdir(OUT)):
    if f.endswith('.pdf') and f != 'learning_curves.pdf':
        size = os.path.getsize(os.path.join(OUT, f))
        print(f"  {f} ({size:,} bytes)")
