# HMARL-SOC

**Hierarchical Multi-Agent Reinforcement Learning for Autonomous SOC Coordination**

> Accompanying code for the IEEE Access journal paper (extended from ITC-CSCC 2026).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nutthakorn7/HMARL-SOC-journal/blob/main/code/train_colab.ipynb)

## Architecture

```
┌─────────────────────────────────────────────┐
│      Strategic Coordinator (PPO)            │  Tier 1
│      Campaign Decomposition & Allocation    │
├──────────┬──────────────┬───────────────────┤
│  Threat  │  Alert       │  Response         │  Tier 2
│  Hunter  │  Triage      │  Orchestrator     │
│  (SAC)   │  (DQN)       │  (MADDPG)         │
├──────────┴──────────────┴───────────────────┤
│  Shared Replay Buffer + Attention Explainer │  Tier 3
└─────────────────────────────────────────────┘
```

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Train (optimized — 7× faster than legacy train.py)
python train_fast.py --config configs/default.yaml --episodes 10000 --seed 42

# Multi-seed training (5 seeds, 2 parallel)
bash train_multi_seed_fast.sh

# Evaluate
python evaluate.py --checkpoint checkpoints/checkpoint_best.pt --episodes 2000

# Compute Table II
python compute_table2.py

# Generate all figures
python generate_figures.py
```

Or use **Google Colab** (free GPU): click the badge above ☝️

## Project Structure

```
├── configs/default.yaml          # Hyperparameters (K=5)
├── hmarl_soc/
│   ├── env/
│   │   ├── soc_env.py            # Gymnasium SOC environment (Dec-POMDP)
│   │   ├── network.py            # Enterprise network (5 segments × 40 hosts)
│   │   └── attacker.py           # MITRE ATT&CK multi-stage attacker
│   ├── agents/
│   │   ├── strategic_coordinator.py  # PPO (Tier 1)
│   │   ├── threat_hunter.py          # SAC (Tier 2)
│   │   ├── alert_triage.py           # DQN (Tier 2)
│   │   └── response_orchestrator.py  # MADDPG (Tier 2)
│   ├── models/networks.py        # 3-layer MLP (256 hidden, ReLU)
│   └── core/
│       ├── replay_buffer.py      # Prioritized shared replay (200K)
│       └── attention.py          # Multi-head attention explainer
├── train_fast.py                 # Optimized training (MPS/CUDA, torch.compile)
├── train.py                      # Legacy training (deprecated)
├── train_baselines.py            # All baselines (Rule-SOAR → QMIX)
├── train_qmix.py                 # QMIX baseline
├── evaluate.py                   # Evaluation & metrics
├── evaluate_cyborg.py            # CybORG CAGE-4 transfer
├── compute_table2.py             # Compute Table II results
├── generate_figures.py           # Reproduce all paper figures
├── run_ablation.sh               # Ablation experiments
├── run_k_sweep.sh                # K sensitivity sweep
├── train_colab.ipynb             # Google Colab notebook
└── checkpoints/                  # Trained models & CSV logs
```

## Key Results (Table II)

| Method | Reward ↑ | MTTD ↓ | MTTR ↓ | FPR % ↓ | CSR % ↑ |
|--------|:---:|:---:|:---:|:---:|:---:|
| Rule-SOAR | −1238.2±26.6 | **8.0**±0.1 | 136.9±3.1 | 5.14±0.01 | 35.2±1.8 |
| Single-DRL | −336.5±34.0 | 10.9±1.4 | 95.1±12.3 | 2.97±0.28 | 65.0±7.3 |
| IQL | +1.8±4.7 | 22.8±18.3 | 91.9±56.7 | 0.22±0.23 | 67.8±29.8 |
| MAPPO | −292.2±12.6 | 8.8±0.1 | 78.6±0.9 | 2.52±0.04 | 69.7±0.6 |
| QMIX | −99.3±66.3 | 8.2±0.2 | **63.4**±32.4 | 1.03±0.02 | **77.5**±18.6 |
| **HMARL-SOC** | **+6.9**±1.0 | 16.8±5.3 | 93.2±21.3 | **0.17**±0.08 | 71.0±10.4 |

*Mean ± std over 5 seeds, last 2,000 episodes.*

## Hyperparameters

- γ = 0.99, lr = 3×10⁻⁴, buffer = 200K, batch = 256
- K = 5 (SC temporal abstraction)
- Reward: α=1.0, β=1.5, δ=−0.3, λ=−2.0
- Networks: 3-layer MLP, 256 hidden, ReLU
- Training: 10K episodes × 5 seeds
- Complexity: O(N·d²) per step

## Citation

```bibtex
@article{chalaemwongwan2026hmarl,
  title   = {{HMARL-SOC}: Hierarchical Multi-Agent Reinforcement Learning
             for Autonomous Security Operations Center Coordination},
  author  = {Chalaemwongwan, Nutthakorn},
  journal = {IEEE Access},
  year    = {2026},
  note    = {Under review}
}
```

## License

MIT
