# MCL775 — Model-Free RL Algorithm Comparison
### Stochastic Gridworld | IIT Delhi 

> **Reproducing and extending:** Fox, R., Pakman, A., & Tishby, N. (2016).  
> *Taming the Noise in Reinforcement Learning via Soft Updates.* UAI 2016. [arXiv:1512.08562](https://arxiv.org/abs/1512.08562)

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Algorithms Compared](#2-algorithms-compared)
3. [Environment](#3-environment)
4. [Results Summary](#4-results-summary)
5. [Repository Structure](#5-repository-structure)
6. [Setup — Method A: VSCode + Conda (Recommended)](#6-setup--method-a-vscode--conda-recommended)
7. [Setup — Method B: Classic Jupyter Notebook](#7-setup--method-b-classic-jupyter-notebook)
8. [Running the Experiments](#8-running-the-experiments)
9. [Key Implementation Decisions](#9-key-implementation-decisions)
10. [LLM Usage Disclosure](#10-llm-usage-disclosure)
11. [References](#11-references)

---

## 1. Project Overview

This project implements, trains, and compares **eight model-free control algorithms** on the Fox (2016) 8×8 stochastic Gridworld environment as part of the MCL775 Reinforcement Learning course project at IIT Delhi.

The central question studied is: **how do different algorithms handle the optimistic bias problem in noisy environments?** Standard Q-learning suffers from a systematic negative bias early in training — a consequence of Jensen's inequality applied to the min operator over noisy Q-value estimates. The primary paper (Fox et al., 2016) proposes G-Learning as a solution via entropy regularisation. This project reproduces that comparison and extends it to six additional algorithms across four algorithm families.

**Key findings:**
- Expected SARSA achieved the best value estimation accuracy (final MAE = 0.023)
- TRPO achieved the best deployed-policy quality (policy error = 0.075) — better than Q-learning
- G-Learning's entropy regularisation successfully eliminated the optimistic bias (late bias = +0.097 vs Q-learning's −0.011)
- The distinction between value accuracy (MAE) and policy quality (policy error) is a central empirical finding — TRPO ranks 7th on MAE but 2nd on policy error

**Group Members:**
- Shreenath
- Kirtan
- Swapnil
- Inderpal

---

## 2. Algorithms Compared

| # | Algorithm | Family | Key Property |
|---|---|---|---|
| 1 | **Q-Learning** | TD Value-Based | Off-policy, hard min, baseline |
| 2 | **SARSA** | TD Value-Based | On-policy, learns exploration policy value |
| 3 | **Expected SARSA** | TD Value-Based | On-policy, expected value over actions |
| 4 | **Double Q-Learning** | TD Value-Based | Decoupled selection/evaluation, bias reduction |
| 5 | **G-Learning** | Information-Theoretic | Entropy-regularised soft-min, the paper's main algorithm |
| 6 | **REINFORCE** | Policy Gradient | Monte Carlo, unbiased but high variance |
| 7 | **Actor-Critic (A2C)** | Actor-Critic | TD advantage, online policy gradient |
| 8 | **TRPO (Tabular)** | KL-Constrained | Natural gradient, hard KL constraint per update |

### Algorithm Update Rules

**Q-Learning:**
```
Q(s,a) ← Q(s,a) + α[c + γ·min_{a'} Q(s',a') − Q(s,a)]
```

**SARSA:**
```
Q(s,a) ← Q(s,a) + α[c + γ·Q(s',a') − Q(s,a)]   # a' is actual next action
```

**Expected SARSA:**
```
Q(s,a) ← Q(s,a) + α[c + γ·Σ_{a'} π(a'|s')Q(s',a') − Q(s,a)]
```

**Double Q-Learning:**
```
a* = argmin Q_A(s',·);   Q_B(s,a) ← Q_B(s,a) + α[c + γ·Q_B(s',a*) − Q_B(s,a)]
```

**G-Learning (soft-min update):**
```
G(s,a) ← G(s,a) + α[c − (γ/β)·log Σ_{a'} exp(−βG(s',a')) − G(s,a)]
```
with β scheduled linearly from 0.1 → 15.0 over 3000 episodes.

**REINFORCE:**
```
θ(s_t,a_t) ← θ(s_t,a_t) − α·Ĝ_t·∇_θ log π(a_t|s_t)
```
Returns normalised to zero mean/unit variance before update.

**Actor-Critic:**
```
δ_t = c + γ·V(s') − V(s)
V(s) ← V(s) + α_c·δ_t
θ(s,·) ← θ(s,·) − α_a·δ_t·∇_θ log π(a|s)
```

**TRPO (Tabular):**
```
F(s) = diag(π_s) − π_s·π_s^T          # Fisher information matrix
natural_grad = F^{-1}·g                 # g = advantage × grad_log_pi
θ_new = θ_old + step·natural_grad       # step found via backtracking line search
subject to: D_KL(π_old ‖ π_new) ≤ 0.01
```

---

## 3. Environment

The **8×8 stochastic Gridworld** from Fox et al. (2016):

```
. . . . . . . .
. X . . X . . .
. X . . X . . .
. X . . X X X .
. X X . X . X .
. . X . . . X .
. . X . . . X .
. . . . . . . .
            ^-- Terminal state at [4,4]
X = blocked wall cell
```

| Property | Value |
|---|---|
| Grid size | 8×8 = 64 cells |
| Blocked states | 15 wall cells |
| Terminal state | [4,4] — ends episode |
| Valid states (for metrics) | 48 non-terminal, non-wall |
| Actions | 9 (8 directions + stay) |
| Transition model | 70% intended, 30% random drift |
| Step cost | 1 + N(0, 0.04) |
| Wall collision cost | 10 |
| Discount factor γ | 0.85 |

**Stochastic drift probabilities:**
- Cardinal directions (N/E/S/W): 5% each
- Diagonal directions (NE/SE/SW/NW): 2.5% each
- No drift: 70%

---

## 4. Results Summary

All results from **3000 episodes × 10 independent runs** (seed=42).

### Final Performance Table

| Algorithm | Final MAE | Final Bias | Policy Error | Ep→MAE<20% | Bellman↓ |
|---|---|---|---|---|---|
| Q-Learning | 0.0176 | −0.011 | 0.0846 | 114 | 64% |
| Expected SARSA | 0.0234 | +0.018 | 0.0668 | 81 | 65% |
| SARSA | 0.4319 | +0.429 | 0.0962 | 63 | 5% |
| **G-Learning** | **0.1805** | **+0.097** | 0.2497 | 532 | 62% |
| Double Q | 1.2716 | +1.280 | 0.1827 | >3000 | 72% |
| TRPO (Tabular) | 1.0461 | +1.038 | **0.0748** | >3000 | 39% |
| REINFORCE | 1.0690 | −1.069 | 0.5606 | >3000 | 53% |
| Actor-Critic | 3.1254 | +3.147 | 0.5696 | >3000 | 25% |

**Policy Error Ranking (best → worst):**
`Exp.SARSA (0.067) > TRPO (0.075) > Q-Learning (0.085) > SARSA (0.096) > Double Q (0.183) > G-Learning (0.250) > REINFORCE (0.561) > Actor-Critic (0.570)`

### Metric Definitions
- **Final MAE**: Mean absolute relative error `|V_est(s) − V*(s)| / V*(s)` averaged over 48 valid states, last 100 episodes
- **Final Bias**: Signed relative error — negative = optimistic (underestimates costs)
- **Policy Error**: True value of greedy policy vs V*, computed analytically every 50 episodes
- **Ep→MAE<20%**: First episode where MAE crosses below 20% threshold
- **Bellman↓**: Percentage reduction in mean |TD error| from first to last 100 episodes

---

## 5. Repository Structure

```
MCL775_RL_Project/
│
├── notebooks/
│   └── main_experiments.ipynb    ← Single notebook, all code
│
├── results/                      ← Auto-created at runtime (gitignored)
│   ├── q_learning.npy
│   ├── sarsa.npy
│   ├── expected_sarsa.npy
│   ├── double_q.npy
│   ├── g_learning.npy
│   ├── reinforce.npy
│   ├── actor_critic.npy
│   ├── trpo.npy
│   ├── fig_all_metrics.png
│   ├── fig_compare_*.png         ← Pairwise comparison plots
│   ├── fig_convergence_speed.png
│   ├── fig_final_mae.png
│   ├── viz_v_star.png            ← V* heatmap
│   ├── viz_V_*.png               ← Learned V per algorithm
│   ├── viz_policy_*.png          ← Policy arrow plots per algorithm
│   └── diag_convergence_check.png
│
├── report/
│   └── MCL775_Final_Report.docx
│
├── requirements.txt
├── README.md                     ← this file
└── .gitignore
```

### Notebook Cell Structure

| Cell | Purpose |
|---|---|
| Cell 0 | CONFIG — all hyperparameters in one place |
| Cell 1 | Imports |
| Cell 2 | GridWorldEnv class (sampled environment) |
| Cell 3 | Analytical model: P, C matrices, V*, policy_eval() |
| Cell 4 | Utilities: softmax, boltzmann, learning rate, action selection |
| Cell 5 | Algorithm state initialisation functions |
| Cell 6 | Algorithm update functions — one per algorithm |
| Cell 7 | Episode runners: run_td_episode(), run_reinforce_episode() |
| Cell 8 | Metric extraction functions |
| Cell 9 | run_experiment() — master training loop, auto-saves .npy |
| Cell 9b | Load saved results (skip retraining) |
| Cell 9c | Post-training diagnostics — NaN audit, convergence check |
| Cell 10 | Plotting functions |
| Cell 11 | Generate all comparison figures |
| Cell 12 | Gridworld visualisations: heatmaps, arrow plots |

---

## 6. Setup — Method A: VSCode + Conda (Recommended)

This is the method used by the project team. Requires **Anaconda** and **VSCode** installed.

### Step 1 — Clone the Repository

```bash
git clone https://github.com/Zeleckto/RL-Biased-Estimator-Noise-Taming-Implementation.git
cd RL-Biased-Estimator-Noise-Taming-Implementation
```

### Step 2 — Create the Conda Environment

Open **Anaconda Prompt** (Windows) or Terminal (Mac/Linux):

```bash
conda create -n mcl775 python=3.10 -y
conda activate mcl775
pip install -r requirements.txt
```

> **Windows note:** If your Anaconda is installed in a path with spaces (e.g. `D:\Ai ML\Anaconda`), always use Anaconda Prompt for package management — not the VSCode terminal — to avoid activation issues.

### Step 3 — Register the Kernel

```bash
python -m ipykernel install --user --name mcl775 --display-name "MCL775 RL (Python 3.10)"
```

### Step 4 — Open in VSCode

1. Open VSCode → **File → Open Folder** → select project root
2. Open `notebooks/main_experiments.ipynb`
3. Top-right kernel selector → **MCL775 RL (Python 3.10)**
4. Done — kernel auto-selects on every subsequent open

### Step 5 — Set VSCode Python Interpreter (Optional)

This makes VSCode terminals auto-activate the environment:

1. `Ctrl+Shift+P` → **Python: Select Interpreter**
2. Choose the `mcl775` environment (shows the Anaconda path)
3. New terminals will now auto-activate `(mcl775)`

### Step 6 — Create Results Folder

The notebook auto-creates this, but you can create it manually:

```bash
mkdir results
```

### Verify Installation

Run this in a VSCode terminal or Anaconda Prompt:

```bash
conda activate mcl775
python -c "import numpy, scipy, matplotlib, tqdm; print('All dependencies OK')"
```

---

## 7. Setup — Method B: Classic Jupyter Notebook

Alternative if you prefer the browser-based interface without VSCode.

### Step 1 — Clone and Install

```bash
git clone https://github.com/Zeleckto/RL-Biased-Estimator-Noise-Taming-Implementation.git
cd RL-Biased-Estimator-Noise-Taming-Implementation
pip install -r requirements.txt
```

Or with conda:

```bash
conda create -n mcl775 python=3.10 -y
conda activate mcl775
pip install -r requirements.txt
```

### Step 2 — Launch Jupyter

```bash
# If using conda:
conda activate mcl775

# Launch from project root:
jupyter notebook
```

This opens a browser tab at `http://localhost:8888`.

### Step 3 — Open the Notebook

In the browser: navigate to `notebooks/` → click `main_experiments.ipynb`

### Step 4 — Select Kernel

In the notebook: **Kernel → Change Kernel → mcl775** (or Python 3 if you used pip without conda)

### Step 5 — Run All Cells

**Kernel → Restart & Run All**

Or run cells manually top to bottom with `Shift+Enter`.

> **Important:** Always run from Cell 0 downward in order. Each cell depends on variables from cells above. If you restart the kernel, start from Cell 0 again.

---

## 8. Running the Experiments

### Quick Test (2-3 minutes)

Before the full run, verify everything works:

```python
# In Cell 0, temporarily change:
CONFIG["num_runs"]    = 2
CONFIG["num_episodes"]= 200
CONFIG["algos"]       = ["q_learning", "g_learning"]
```

Run Cells 0 → 9. Should complete in under 3 minutes and produce two plots.

### Full Training Run (~40 minutes)

Restore full config in Cell 0:

```python
CONFIG["num_runs"]    = 10
CONFIG["num_episodes"]= 3000
CONFIG["algos"] = [
    "q_learning", "sarsa", "expected_sarsa", "double_q",
    "g_learning", "reinforce", "actor_critic", "trpo"
]
```

Run Cell 9. Results save automatically to `results/` after each algorithm completes.

### Load Saved Results (Skip Retraining)

If results are already saved, use Cell 9b:

```python
# Cell 9b — uncomment and run:
ALL_RESULTS = {}
for algo in CONFIG["algos"]:
    path = f"{CONFIG['save_path']}{algo}.npy"
    ALL_RESULTS[algo] = np.load(path, allow_pickle=True).item()
    print(f"✓ Loaded {LABELS[algo]}")
```

### Generate Plots Only

After loading results, run Cells 10 → 11 → 12. All figures save to `results/`.

### Run a Single Algorithm

```python
# In Cell 0:
CONFIG["algos"] = ["trpo"]   # any single algorithm name

# In Cell 9, add at the top to preserve other results:
for algo in ["q_learning","sarsa","expected_sarsa","double_q",
             "g_learning","reinforce","actor_critic"]:
    path = f"{CONFIG['save_path']}{algo}.npy"
    if os.path.exists(path):
        ALL_RESULTS[algo] = np.load(path, allow_pickle=True).item()
```

### Switching Algorithms (Modular Design)

All algorithm logic lives in Cell 6 as independent functions. The episode runner in Cell 7 dispatches via algorithm name — no if-elif chains in the training loop. To compare a subset:

```python
CONFIG["algos"] = ["q_learning", "g_learning", "trpo"]
```

To add a new algorithm: implement `init_*()` and `update_*()` functions in Cell 6, add to `INIT_FNS` dict and the if-elif block in Cell 7.

---

## 9. Key Implementation Decisions

### Wall Penalty: 10 (not 1000)

Initial experiments used a wall penalty of 1000. This caused catastrophic value inflation in all algorithms that bootstrap from V(s) as a weighted average (SARSA, REINFORCE, Actor-Critic, TRPO, G-Learning). With uniform policy over 9 actions and 2 wall actions at cost 1000, early V(s) ≈ 224 vs V* ≈ 3. MAE = |224-3|/3 ≈ 73 — meaningless. A penalty of 10 is sufficient to discourage wall movement while keeping estimates in the same order of magnitude as V*.

### G-Learning Visit Count Reset

G-Learning resets its visit count array at the start of each episode, matching the Fox (2016) original implementation. Without this, globally accumulated counts cause the learning rate to decay to ~0.003 by episode 500. The initial soft-value bias (V_G ≈ −22 at β=0.1) then takes thousands of episodes to correct. With per-episode reset, effective learning rate ≈ 1.0 at each episode start, correcting the initialisation bias efficiently.

### SARSA ε Consistency

SARSA's on-policy requirement means the pre-selected next action a' must use the same decayed ε as the current step. Using a fixed ε=0.4 for next-action selection creates an inconsistency: the algorithm evaluates a permanently-40%-random policy but follows a decaying-ε policy. This caused SARSA MAE to climb to 13.1 in initial runs. Fixed by passing the current ε to both selection calls.

### REINFORCE Return Normalisation

Returns normalised to zero mean / unit variance before gradient computation. Early episodes (300 random steps, many wall hits) produce returns ~1000. Late episodes (10 optimal steps) produce returns ~10. Without normalisation, gradient steps of magnitude α×1000 saturate the softmax logits immediately and the policy never recovers. Normalisation is mathematically equivalent to an adaptive learning rate and does not change the algorithm's theoretical properties.

### TRPO and Actor-Critic Critic Learning Rate

Critic learning rates reduced from 0.1/0.05 to 0.01. At 0.1, V(s) oscillates by ~90 per step in early training due to noisy TD targets, producing an unstable advantage signal. At 0.01, V(s) moves by ~9 per step — responsive but not chaotic. Actor learning rate set 10× smaller than critic since the actor depends on critic signal quality.

---

## 10. LLM Usage Disclosure

As required by course guidelines: Claude (Anthropic) was used for the following purposes:
- Explaining mathematical derivations from Fox et al. (2016)
- Structuring the tabular TRPO natural gradient implementation
- Identifying implementation bugs (visit count issue, wall penalty inflation, SARSA ε inconsistency)
- Generating boilerplate code structure and plot functions
- Drafting report sections

All experimental design, hyperparameter choices, training runs, result interpretation, and conclusions are entirely the group's own work. All equations were independently verified against original papers. The group takes full responsibility for all code and report content.

---

## 11. References

1. Fox, R., Pakman, A., & Tishby, N. (2017). Taming the noise in reinforcement learning via soft updates. *UAI 2016*. arXiv:1512.08562
2. Sutton, R.S. & Barto, A.G. (2018). *Reinforcement Learning: An Introduction*, 2nd ed. MIT Press.
3. Schulman, J. et al. (2015). Trust Region Policy Optimization. *ICML 2015*. arXiv:1502.05477
4. van Hasselt, H. (2010). Double Q-Learning. *NeurIPS 2010*.
5. van Seijen, H. et al. (2009). A theoretical and empirical analysis of Expected SARSA. *IEEE ADPRL*.
6. Williams, R.J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. *Machine Learning*, 8(3–4).
7. Watkins, C.J.C.H. & Dayan, P. (1992). Q-Learning. *Machine Learning*, 8(3–4).

---

*MCL775 Course Project — Department of Mechanical Engineering, IIT Delhi, 2025*
