# HMARL-SOC Journal Extension

## Project Overview
This is a journal extension of the HMARL-SOC conference paper (ITC-CSCC 2026) targeting **IEEE Access** (Q1).

HMARL-SOC is a **Hierarchical Multi-Agent Reinforcement Learning** framework for autonomous SOC (Security Operations Center) coordination. It uses a 3-tier architecture: Strategic Coordinator (PPO) → Operational Agents (SAC/DQN/MADDPG) → Shared Communication Layer.

## Project Structure
```
Journal_HMARL-SOC/
├── paper/                    # LaTeX paper (IEEE Access format)
│   ├── main.tex              # Main manuscript (11 pages)
│   ├── main.pdf              # Compiled PDF (11 pages, no blank pages)
│   ├── ieeeaccess.cls        # IEEE Access document class (modified)
│   ├── spotcolor.sty         # Required style
│   └── figures/
│       ├── author_photo.jpg
│       ├── learning_curves.pdf      # Fig 2: Learning curves
│       ├── radar_comparison.pdf     # Fig 3: Radar chart
│       ├── boxplots_comparison.pdf  # Fig 4: Box plots
│       ├── fpr_csr_scatter.pdf      # Fig 5: FPR vs CSR
│       ├── ablation_bars.pdf        # Fig 6: Ablation bars
│       ├── ablation_learning_curves.pdf  # Fig 7: Ablation curves
│       ├── graphical_abstract.png   # For submission
│       └── graphical_abstract.pdf
├── code/                     # Experiment code
│   ├── train.py              # Main training script
│   ├── train_fast.py         # Optimized training (MPS/CUDA)
│   ├── train_baselines.py    # Baseline methods
│   ├── train_qmix.py         # QMIX baseline
│   ├── evaluate.py           # Evaluation script
│   ├── evaluate_cyborg.py    # CybORG CAGE-4 transfer
│   ├── compute_table2.py     # Compute Table II results
│   ├── plot_results.py       # Generate learning curve figures
│   ├── generate_figures.py   # Generate all 5 new figures
│   ├── run_ablation.sh       # Ablation experiment runner
│   ├── configs/default.yaml  # Hyperparameter config
│   ├── hmarl_soc/            # Core framework code
│   └── checkpoints/          # Trained models + CSV logs (55 files)
└── HMARL-SOC_IEEE_Access_Submission.zip  # Ready-to-submit package
```

## Commands

### Training
```bash
# Full HMARL-SOC (5 seeds)
bash train_multi_seed_fast.sh

# Ablation experiments (w/o SC, w/o Buffer, Flat MARL)
bash run_ablation.sh

# Single run
python3 train_fast.py --episodes 10000 --seed 42
```

### Evaluation
```bash
# Compute Table II
python3 compute_table2.py

# Generate all figures
python3 generate_figures.py

# CybORG CAGE-4 transfer
python3 evaluate_cyborg.py
```

### Paper
```bash
# Compile LaTeX (two passes for cross-refs)
cd paper && pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex
```

## Key Hyperparameters
- γ = 0.99, lr = 3e-4, batch = 256, replay buffer = 200K
- K = 5 (SC temporal abstraction)
- Reward weights: α=1.0, β=1.5, δ=-0.3, λ=-2.0
- Networks: 3-layer MLP, 256 hidden units, ReLU
- Training: 10K episodes × 5 seeds

## Modifications to ieeeaccess.cls
The following changes were made to `ieeeaccess.cls` for correct compilation:
- **Line 274:** `\RequirePackage{color}` → `\RequirePackage{xcolor}` (fix TikZ conflict)
- **Lines 278-281:** Replaced PANTONE spot colors with CMYK equivalents for `accessblue` and `greycolor`
- **Lines 506-507:** Updated `\thevol` to 14 and `\theyear` to 2026
- **Lines 555-571:** Emptied footer definitions (`\@oddfoot{}`, `\@evenfoot{}`) to prevent VOLUME text overlapping with body content — IEEE adds proper footers during production

## Current Status
- Conference paper: ✅ Submitted to ITC-CSCC 2026
- Journal extension: ✅ Complete (IEEE Access)
  - 11 pages, 7 figures, 4 tables, 35 references
  - Sections: Introduction, Related Work, System Model, Framework, Convergence Analysis, Experiments, Ablation Study, Cross-Environment Transfer, Limitations, Conclusion
  - Graphical abstract ready
  - Content reviewed and all issues fixed
