# HMARL-SOC Journal Extension

## Project Overview
This is a journal extension of the HMARL-SOC conference paper (ITC-CSCC 2026) targeting **IEEE Access** (Q1).

HMARL-SOC is a **Hierarchical Multi-Agent Reinforcement Learning** framework for autonomous SOC (Security Operations Center) coordination. It uses a 3-tier architecture: Strategic Coordinator (PPO) → Operational Agents (SAC/DQN/MADDPG) → Shared Communication Layer.

**GitHub**: https://github.com/nutthakorn7/HMARL-SOC-journal

### Journal Extension Contributions (vs. Conference)
The conference paper (ITC-CSCC 2026) covered the basic 3-tier architecture and 4-baseline comparison. The journal extension adds:
1. **Ablation study** — component-level analysis (w/o SC, w/o Buffer, Flat MARL)
2. **Scalability analysis** — wall-clock benchmarks 50–500 hosts
3. **CybORG CAGE-4 zero-shot transfer** — cross-environment generalization
4. **Formal convergence analysis** — two-timescale stochastic approximation proofs
5. **K-sensitivity analysis** — temporal abstraction factor sweep (K=1,3,5,10,20)
6. **QMIX baseline** — added as 5th baseline (monotonic value factorization)
7. **Extended related work** — sim-to-real transfer, LLM-augmented SOC sections

## Project Structure
```
Journal_HMARL-SOC/
├── paper/                    # LaTeX paper (IEEE Access format)
│   ├── main.tex              # Main manuscript
│   ├── main.pdf              # Compiled PDF
│   ├── ieeeaccess.cls        # IEEE Access document class (modified)
│   ├── IEEEtran.cls          # IEEEtran document class
│   ├── IEEEtran.bst          # BibTeX style
│   ├── spotcolor.sty         # Required style
│   ├── cover_letter.tex      # IEEE Access cover letter
│   ├── cover_letter.pdf
│   ├── ai_check.py           # AI content check script
│   └── figures/
│       ├── author_photo.jpg
│       ├── learning_curves.pdf      # Fig 2: Learning curves (6 methods)
│       ├── learning_curves.png      # PNG version
│       ├── radar_comparison.pdf     # Fig 3: Radar chart
│       ├── boxplots_comparison.pdf  # Fig 4: Box plots
│       ├── fig_k_sensitivity.pdf    # Fig 5: K sensitivity analysis
│       ├── fpr_csr_scatter.pdf      # Fig 6: FPR vs CSR
│       ├── ablation_bars.pdf        # Fig 7: Ablation bars
│       ├── ablation_learning_curves.pdf  # Fig 8: Ablation curves
│       ├── graphical_abstract.png   # For submission
│       └── graphical_abstract.pdf
├── code/                     # Experiment code
│   ├── train.py              # Main training script
│   ├── train_fast.py         # Optimized training (MPS/CUDA)
│   ├── train_baselines.py    # Baseline methods (IQL, MAPPO, Single-DRL, Rule-SOAR)
│   ├── train_qmix.py         # QMIX baseline
│   ├── train_multi_seed.sh   # Multi-seed runner (original)
│   ├── train_multi_seed_fast.sh  # Multi-seed runner (optimized)
│   ├── evaluate.py           # Evaluation script
│   ├── evaluate_cyborg.py    # CybORG CAGE-4 transfer evaluation
│   ├── compute_table2.py     # Compute Table II results
│   ├── plot_results.py       # Generate learning curve figures
│   ├── generate_figures.py   # Generate all new figures (radar, boxplot, scatter, ablation)
│   ├── run_ablation.sh       # Ablation experiment runner
│   ├── run_k_sweep.sh        # K sensitivity sweep (K=1,3,5,10,20)
│   ├── train_colab.ipynb     # Google Colab notebook
│   ├── requirements.txt      # Python dependencies
│   ├── cyborg_results.json   # CybORG CAGE-4 transfer results
│   ├── configs/
│   │   ├── default.yaml      # Default hyperparameters (K=5)
│   │   ├── k1_config.yaml    # K=1 config
│   │   ├── k3_config.yaml    # K=3 config
│   │   ├── k10_config.yaml   # K=10 config
│   │   └── k20_config.yaml   # K=20 config
│   ├── hmarl_soc/            # Core framework code
│   │   ├── __init__.py
│   │   ├── agents/
│   │   │   ├── strategic_coordinator.py  # Tier 1: PPO-based SC
│   │   │   ├── threat_hunter.py          # Tier 2: SAC agent
│   │   │   ├── alert_triage.py           # Tier 2: DQN agent
│   │   │   └── response_orchestrator.py  # Tier 2: MADDPG agent
│   │   ├── core/
│   │   │   ├── attention.py              # Multi-head attention (explainability)
│   │   │   └── replay_buffer.py          # Shared experience buffer
│   │   ├── env/
│   │   │   ├── soc_env.py                # Main SOC environment (200 hosts)
│   │   │   ├── network.py                # Network topology
│   │   │   └── attacker.py               # MITRE ATT&CK adversary
│   │   ├── models/
│   │   │   └── networks.py               # MLP / Actor-Critic architectures
│   │   └── utils/
│   ├── cyborg_cage4/         # CybORG CAGE-4 environment (external)
│   ├── checkpoints/          # Trained models (7 .pt) + CSV logs (61) + figures (3)
│   └── .agent/workflows/     # Agent workflow definitions
│       └── training.md       # Training workflow
├── README.md                 # GitHub README with results & architecture
├── CLAUDE.md                 # This file — project overview for AI assistants
├── .gitignore
└── HMARL-SOC_IEEE_Access_Submission.zip  # Ready-to-submit package
```

## Setup
```bash
# Python 3.12 venv (recommended)
source ~/.venvs/hmarl/bin/activate

# Install dependencies
pip install -r code/requirements.txt
# torch>=2.0, gymnasium>=0.29, numpy>=1.24, pyyaml>=6.0, matplotlib>=3.7, tensorboard>=2.14
```

> **Note:** Always use `train_fast.py` (7× faster than legacy `train.py`) — supports MPS/CUDA, vectorized envs, `torch.compile`, and mixed precision.

## Commands

### Training
```bash
# Full HMARL-SOC (5 seeds, optimized)
bash train_multi_seed_fast.sh

# Ablation experiments (w/o SC, w/o Buffer, Flat MARL)
bash run_ablation.sh

# K sensitivity sweep (K=1,3,5,10,20)
bash run_k_sweep.sh

# Single run
python3 train_fast.py --episodes 10000 --seed 42
```

### Evaluation
```bash
# Compute Table II
python3 compute_table2.py

# Generate all figures (radar, boxplot, scatter, ablation)
python3 generate_figures.py

# CybORG CAGE-4 transfer
python3 evaluate_cyborg.py
```

### Paper
```bash
# Compile LaTeX (two passes for cross-refs)
cd paper && pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex
```

## Code Architecture

**3-tier hierarchy** (Strategic Coordinator → Operational Agents → Shared Communication):

| Tier | Agent | Algorithm | Role |
|------|-------|-----------|------|
| 1 | Strategic Coordinator | PPO | High-level directives every K steps |
| 2 | Threat Hunter | SAC | Continuous threat detection |
| 2 | Alert Triage | DQN | Discrete: escalate/suppress/correlate/enrich |
| 2 | Response Orchestrator | MADDPG | Multi-agent response coordination |

**Environment**: 5 network segments (DMZ, Corporate, Development, DataCenter, Cloud) × 40 hosts = 200 hosts. Adversary follows MITRE ATT&CK kill chain (T1595→T1190→T1059→T1021→T1048). Max 200 steps/episode.

**Communication**: Multi-head attention layer (4 heads, d_k=64) enables inter-agent message passing and provides attention-weight explainability.

## Key Hyperparameters
- γ = 0.99, lr = 3e-4, batch = 256, replay buffer = 200K
- K = 5 (SC temporal abstraction)
- Reward weights: α=1.0, β=1.5, δ=-0.3, λ=-2.0
- Networks: 3-layer MLP, 256 hidden units, ReLU
- Training: 10K episodes × 5 seeds

## Baselines (6 methods in Table II)
- **HMARL-SOC**: Full hierarchical framework (K=5)
- **IQL**: Independent Q-Learning (per-agent DQN)
- **QMIX**: Monotonic value factorization
- **MAPPO**: Multi-Agent PPO with centralized critic
- **Single-DRL**: Monolithic PPO (no hierarchy)
- **Rule-SOAR**: Rule-based SOAR playbook

## Evaluation Metrics
- **Reward** (↑): Composite reward = α·R_det + β·R_resp + δ·R_cost + λ·R_fp
- **MTTD** (↓): Mean Time to Detect (steps)
- **MTTR** (↓): Mean Time to Respond (steps)
- **FPR** (↓): False Positive Rate (%)
- **CSR** (↑): Containment Success Rate (%)

Evaluation window: last 2,000 episodes, averaged over 5 seeds.

## Key Results (Table II)

| Method | Reward | MTTD | MTTR | FPR (%) | CSR (%) |
|--------|--------|------|------|---------|---------|
| Rule-SOAR | −1238.2±26.6 | **8.0**±0.1 | 136.9±3.1 | 5.14±0.01 | 35.2±1.8 |
| Single-DRL | −336.5±34.0 | 10.9±1.4 | 95.1±12.3 | 2.97±0.28 | 65.0±7.3 |
| IQL | +1.8±4.7 | 22.8±18.3 | 91.9±56.7 | 0.22±0.23 | 67.8±29.8 |
| MAPPO | −292.2±12.6 | 8.8±0.1 | 78.6±0.9 | 2.52±0.04 | 69.7±0.6 |
| QMIX | −99.3±66.3 | 8.2±0.2 | **63.4**±32.4 | 1.03±0.02 | **77.5**±18.6 |
| **HMARL-SOC** | **+6.9**±1.0 | 16.8±5.3 | 93.2±21.3 | **0.17**±0.08 | 71.0±10.4 |

**HMARL-SOC wins on:** Highest reward (+6.9), lowest FPR (0.17%, 15× lower than MAPPO), best overall balance across all objectives. QMIX wins on CSR (77.5%) and MTTR (63.4) but with 6× higher FPR and high inter-seed variance.

### CybORG CAGE-4 Transfer
Zero-shot transfer (no fine-tuning): mean reward −30,018 ± 7,665 → **12.2% improvement** over no-defense baseline. The hierarchical coordination structure transfers, not the low-level features.

### Ablation Study (Table IV)

| Variant | Reward | MTTD | FPR (%) | CSR (%) |
|---------|--------|------|---------|---------|
| **Full HMARL-SOC** | **+6.9**±1.0 | 16.8±5.3 | **0.17**±0.08 | **71.0**±10.4 |
| w/o SC | +6.0±0.9 | 26.3±2.9 | 0.24±0.06 | 66.3±9.5 |
| w/o Shared Buffer | −124.5±28.0 | **12.5**±0.6 | 1.69±0.05 | 74.4±8.5 |
| Flat MARL | −127.6±24.7 | 13.1±0.7 | 1.61±0.01 | 73.9±7.4 |

- **Shared Buffer removal**: Reward collapses (→ −124.5), FPR rises 10× (→ 1.69%). Most impactful component.
- **SC removal**: CSR drops 4.7pp, MTTD rises 56%. Primary contribution is coordination efficiency.
- **Flat MARL**: Homogeneous PPO for all agents matches buffer-removal degradation — confirms need for heterogeneous algorithms.

### K Sensitivity (Temporal Abstraction)

| K | Reward | FPR (%) | CSR (%) |
|---|--------|---------|--------|
| 1 | +0.9 | 0.06 | 26.4 |
| 3 | +3.5 | 0.12 | 44.2 |
| **5** | **+5.1** | **0.15** | **54.4** |
| 10 | −71.5 | 11.52 | 49.3 |
| 20 | +4.1 | 0.09 | 52.2 |

K=5 is optimal: balances SC reaction speed vs. operational agent autonomy. K=1 overwhelms agents with rapidly changing directives; K=10 causes FPR spike due to delayed campaign context.

### Scalability

| Hosts | Inference (ms/step) |
|-------|--------------------|
| 50 | 2.1 |
| 200 | 4.2 |
| 500 | 8.3 |

Linear scaling: O(N·d²) per step. Real-time capable at all tested scales (Apple M2).

### Emergent Agent Behaviors
Interesting learned policies discovered after convergence:
- **Threat Hunter**: Self-learned to focus on DataCenter/Cloud during off-peak hours (highest signal-to-noise)
- **Alert Triage**: Invented temporal windowing — suppresses duplicate alerts within 5 steps, escalates on 3+ correlated items
- **Response Orchestrator**: Learned graduated containment — throttle traffic first, isolate only if throttling fails (34% less unnecessary disruption)

### Reward Weight Failure Modes
- λ > −1.0: Triage becomes too conservative, suppresses real threats to avoid false alarms
- δ ≈ 0: Response Orchestrator isolates hosts aggressively, including clean ones
- Default (α=1.0, β=1.5, δ=−0.3, λ=−2.0) chosen via grid sweep on held-out validation campaign

## Future Work
1. **Live SOC pilot** — transfer sim-trained policies to real SIEM/EDR telemetry
2. **Adversarial robustness** — stress-test against observation poisoning attacks
3. **LLM explainability** — pipe attention maps through LLM for plain-English explanations
4. **CybORG fine-tuning** — close domain gap with target-environment adaptation
5. **Human-in-the-loop** — deploy as recommendation layer before full automation

## Modifications to ieeeaccess.cls
The following changes were made to `ieeeaccess.cls` for correct compilation:
- **Line 274:** `\RequirePackage{color}` → `\RequirePackage{xcolor}` (fix TikZ conflict)
- **Lines 278-281:** Replaced PANTONE spot colors with CMYK equivalents for `accessblue` and `greycolor`
- **Lines 506-507:** Updated `\thevol` to 14 and `\theyear` to 2026
- **Lines 555-571:** Emptied footer definitions (`\@oddfoot{}`, `\@evenfoot{}`) to prevent VOLUME text overlapping with body content — IEEE adds proper footers during production

## Paper Structure
| # | Section | Key Content |
|---|---------|-------------|
| I | Introduction | Motivation, contributions |
| II | Related Work | RL/MARL for cybersecurity, SOC automation, sim-to-real, LLM-augmented |
| III | System Model | Dec-POMDP formulation, reward function |
| IV | Proposed Framework | 3-tier architecture, training procedure, attention explainability |
| V | Convergence Analysis | Theoretical convergence guarantees |
| VI | Experimental Evaluation | 6 baselines, K sensitivity analysis, multi-stage attack |
| VII | Ablation Study | Component-level analysis (w/o SC, w/o Buffer, Flat MARL) |
| VIII | Cross-Environment Transfer | CybORG CAGE-4 zero-shot transfer |
| IX | Limitations & Future Work | Threats to validity |
| X | Conclusion | Summary of contributions |

- **8 figures**, **5 tables**, **34 references**
- 1 architecture diagram (TikZ), 1 graphical abstract

## Important Notes
- `paper/` is **gitignored** — the paper directory is not pushed to GitHub (only `code/` and `README.md` are public)
- `train.py` is **deprecated** — always use `train_fast.py` (7× faster)
- The `w/o Buffer` and `Flat MARL` ablation variants diverged early (500–1,000 episodes) — results at divergence point, not full 10K
- K-sensitivity analysis uses **single seed** (42); multi-seed sweep is noted as future work
- IQL shows **bimodal convergence**: 2 seeds reach CSR>85%, 2 fall below 45%

## Current Status
- Conference paper: ✅ Submitted to ITC-CSCC 2026
- Journal extension: ✅ Complete (IEEE Access)
  - 8 figures, 5 tables, 34 references
  - Includes: K sensitivity analysis, ablation study, cross-environment transfer
  - Graphical abstract and cover letter ready
  - Content reviewed and all compilation issues fixed
  - Submission package zipped
