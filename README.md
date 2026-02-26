<p align="center">
  <img src="code/checkpoints/fig2_learning_curves.png" width="700" alt="Learning Curves">
</p>

<h1 align="center">🛡️ HMARL-SOC</h1>

<p align="center">
  <strong>Hierarchical Multi-Agent Reinforcement Learning for Autonomous Security Operations Center Coordination</strong>
</p>

<p align="center">
  <a href="https://colab.research.google.com/github/nutthakorn7/HMARL-SOC-journal/blob/main/code/train_colab.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
  <img src="https://img.shields.io/badge/python-3.8%2B-blue" alt="Python">
  <img src="https://img.shields.io/badge/framework-PyTorch-EE4C2C?logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/env-Gymnasium-0081A5" alt="Gymnasium">
  <img src="https://img.shields.io/badge/paper-IEEE%20Access-00629B?logo=ieee" alt="IEEE Access">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
</p>

<p align="center">
  <em>Extended version of our <a href="https://github.com/nutthakorn7/HMARL-SOC">ITC-CSCC 2026 paper</a> — submitted to IEEE Access</em>
</p>

---

## 🎯 What is HMARL-SOC?

Enterprise SOCs face **thousands of security events per hour**, yet analysts can resolve fewer than half within a shift. HMARL-SOC is a **three-tier hierarchical multi-agent RL architecture** that mirrors the real division of labor in SOC teams:

```
                    ┌──────────────────────────────────────┐
                    │    🎖️  Strategic Coordinator (PPO)    │  Tier 1
                    │    Campaign decomposition & goals     │
                    └──────┬───────────┬───────────┬───────┘
                           │ d_t       │ d_t       │ d_t
                    ┌──────▼──┐  ┌─────▼─────┐  ┌──▼──────────┐
                    │ 🔍 Hunt │  │ 📊 Triage │  │ 🚨 Response │  Tier 2
                    │  (SAC)  │  │   (DQN)   │  │  (MADDPG)   │
                    └────┬────┘  └─────┬─────┘  └──────┬──────┘
                         │  τ_i        │               │  τ_i
                    ┌────▼─────────────▼───────────────▼──────┐
                    │ 💾 Shared Replay Buffer + 🔎 Attention  │  Tier 3
                    └──────────────────────────────────────────┘
```

Each agent uses the **RL algorithm best suited to its action space**: SAC for continuous threat hunting, DQN for discrete alert classification, and MADDPG for coordinated incident response.

---

## 📊 Key Results

Performance comparison on a **200-host, 5-segment MITRE ATT&CK simulator** (mean ± std, 5 seeds):

| Method | Reward (↑) | MTTD (↓) | MTTR (↓) | FPR % (↓) | CSR % (↑) |
|:-------|:---:|:---:|:---:|:---:|:---:|
| Rule-SOAR | −1238.2±26.6 | **8.0**±0.1 | 136.9±3.1 | 5.14±0.01 | 35.2±1.8 |
| Single-DRL | −336.5±34.0 | 10.9±1.4 | 95.1±12.3 | 2.97±0.28 | 65.0±7.3 |
| IQL | +1.8±4.7 | 22.8±18.3 | 91.9±56.7 | 0.22±0.23 | 67.8±29.8 |
| MAPPO | −292.2±12.6 | 8.8±0.1 | 78.6±0.9 | 2.52±0.04 | 69.7±0.6 |
| QMIX | −99.3±66.3 | 8.2±0.2 | **63.4**±32.4 | 1.03±0.02 | **77.5**±18.6 |
| **HMARL-SOC** | **+6.9**±1.0 | 16.8±5.3 | 93.2±21.3 | **0.17**±0.08 | 71.0±10.4 |

> **HMARL-SOC achieves the lowest false positive rate (0.17%) — a 6× reduction vs QMIX and 15× vs MAPPO — and the highest cumulative reward, reflecting the best overall balance across detection, response speed, disruption cost, and false alarm minimization.**

### K-Sensitivity Analysis

<p align="center">
  <img src="code/checkpoints/fig_k_sensitivity.png" width="550" alt="K-Sensitivity">
</p>

The Strategic Coordinator's temporal abstraction factor **K=5** yields optimal performance. At K=1, rapidly changing directives overwhelm operational agents (reward +0.9). At K=10, the SC reacts too slowly (FPR spikes to 11.5%).

---

## 🚀 Quick Start

```bash
# Clone
git clone https://github.com/nutthakorn7/HMARL-SOC-journal.git
cd HMARL-SOC-journal/code

# Install
pip install -r requirements.txt

# Train HMARL-SOC (10K episodes)
python train.py --config configs/default.yaml --episodes 10000 --seed 42

# Train all baselines
python train_baselines.py --episodes 10000 --seeds 42 123 456 789 1024

# Evaluate
python evaluate.py --checkpoint checkpoints/checkpoint_best.pt --episodes 2000

# Generate paper figures
python generate_figures.py
```

**Google Colab** (free GPU): click the Colab badge above ☝️

---

## 📁 Project Structure

```
code/
├── configs/
│   └── default.yaml              # Hyperparameters (Table II in paper)
├── hmarl_soc/
│   ├── env/
│   │   ├── soc_env.py            # Gymnasium SOC environment (Dec-POMDP)
│   │   ├── network.py            # Enterprise network (5 segments × 40 hosts)
│   │   └── attacker.py           # MITRE ATT&CK 5-phase campaign
│   ├── agents/
│   │   ├── strategic_coordinator.py  # PPO + GAE (Tier 1, updates every K steps)
│   │   ├── threat_hunter.py          # SAC with auto-entropy (Tier 2)
│   │   ├── alert_triage.py           # Dueling Double DQN (Tier 2)
│   │   └── response_orchestrator.py  # MADDPG (Tier 2)
│   ├── models/
│   │   └── networks.py           # 3-layer MLP (256 hidden, ReLU)
│   └── core/
│       ├── replay_buffer.py      # Prioritized shared replay buffer (200K)
│       └── attention.py          # Multi-head attention explainer
├── train.py                      # Main training loop (Algorithm 1)
├── train_baselines.py            # Rule-SOAR, Single-DRL, IQL, QMIX, MAPPO
├── train_qmix.py                 # QMIX with per-segment action targeting
├── evaluate.py                   # Evaluation & metric computation
├── evaluate_cyborg.py            # CybORG CAGE-4 zero-shot transfer
├── generate_figures.py           # Reproduce all paper figures
├── checkpoints/                  # Trained models & training CSVs
└── requirements.txt
```

---

## ⚙️ Hyperparameters

| Category | Parameter | Value |
|----------|-----------|-------|
| **Environment** | Segments / Hosts | 5 / 200 |
| **Training** | Episodes / Seeds | 10,000 / 5 |
| **SC (PPO)** | LR / Clip ε / K | 3×10⁻⁴ / 0.2 / 5 |
| **TH (SAC)** | LR / α / τ | 3×10⁻⁴ / 0.2 / 0.005 |
| **AT (DQN)** | ε-greedy decay | 1.0 → 0.05 over 50K |
| **RO (MADDPG)** | LR / τ | 3×10⁻⁴ / 0.005 |
| **Reward** | α, β, δ, λ | 1.0, 1.5, −0.3, −2.0 |
| **Complexity** | Per-step cost | O(N·d²) |

---

## 📄 Citation

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

Conference version:

```bibtex
@inproceedings{chalaemwongwan2026hmarl_conf,
  title     = {{HMARL-SOC}: Hierarchical Multi-Agent Reinforcement Learning
               for Autonomous {SOC} Operations},
  author    = {Chalaemwongwan, Nutthakorn},
  booktitle = {Proc. ITC-CSCC},
  year      = {2026}
}
```

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙏 Acknowledgment

During manuscript preparation the author used Claude (Anthropic) for English grammar checking and sentence-level editing. All research design, algorithm development, implementation, experimentation, and interpretation of results were performed solely by the author.
