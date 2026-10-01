# WACV 2027 RWS Workshop — Track 2: Attribute-based Person Retrieval (ABPR)
## Target: 1st Place Worldwide on the UPAR Benchmark ($\ge 0.8000\text{ mADM}$)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![Codabench #18196](https://img.shields.io/badge/Codabench-%2318196-success.svg)](https://www.codabench.org/competitions/18196/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Competition:** IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) 2027 Workshop on Real-World Surveillance (RWS)  
> **Challenge:** Track 2 — Attribute-based Person Retrieval on the Unified Pedestrian Attribute Recognition (UPAR) Benchmark  
> **Platform:** [Codabench Competition #18196](https://www.codabench.org/competitions/18196/)  
> **Official Team / Participant:** Jyothiradithya Lingam  
> **Current Official Standing:** **`#9 Worldwide`** (`mADM: 0.4041`, `Rank-1: 13.58%`, `Rank-5: 29.35%`, `mAP: 0.1273`)  
> **Target Score:** **`>= 0.8000 mADM`** (Surpassing SOTA ceiling and winning 1st Place)

---

## 1. Problem Formulation & Challenge Overview

Attribute-based Person Retrieval (ABPR) requires searching a massive surveillance gallery ($\mathcal{G} > 33,000$ images) for target individuals based exclusively on a human-annotated 40-dimensional binary semantic attribute query:
$$\mathbf{q} \in \{0, 1\}^{40}$$

The primary evaluation metric is **Mean Average Degree of Match (mADM)**:
$$\text{dom}(g, \mathbf{q}) = \frac{1}{40} \sum_{a=1}^{40} \mathbb{I}(\mathbf{y}_{g, a} == \mathbf{q}_a) = 1.0 - \frac{1}{40} \|\mathbf{y}_g - \mathbf{q}\|_1$$
$$\text{ndom}(g, \mathbf{q}) = \max\left(0, \frac{\text{dom}(g, \mathbf{q}) - \mu_{\text{dom}}(\mathbf{q})}{1.0 - \mu_{\text{dom}}(\mathbf{q})}\right)$$
$$\text{ADM}(\pi, \mathbf{q}) = \frac{1}{R} \sum_{k=1}^R \left( \frac{1}{r_k} \sum_{i=1}^{r_k} \text{ndom}(g_{(i)}, \mathbf{q}) \right)$$
$$\text{mADM} = \frac{1}{Q} \sum_{q=1}^Q \text{ADM}(\pi_q, \mathbf{q})$$

Where $r_k$ is the sorted rank of the $k$-th ground-truth exact match in the returned gallery permutation $\pi$.

---

## 2. Repository Architecture & Directory Layout

```
.
├── configs/                             # Experiment configuration files
│   └── exp003_convnext_base_kaggle.yaml # Kaggle Dual-T4 production config
├── docs/                                # Publication-grade research blueprints & math specs
│   ├── 01_challenge_overview_and_live_intelligence.md
│   ├── 02_mathematics_of_evaluation_metrics_and_madm.md
│   ├── 03_dataset_forensics_and_imbalance_analysis.md
│   ├── 04_state_of_the_art_architectures_c2tnet_clear_attrivision.md
│   ├── 05_the_1st_place_winning_system_blueprint.md
│   ├── 06_training_pipeline_loss_functions_and_domain_generalization.md
│   ├── 07_codabench_container_engineering_and_submission_contract.md
│   ├── 08_hardware_optimization_m1_pro_16gb_and_cloud_strategy.md
│   ├── 09_the_21_day_winning_roadmap_and_wacv_paper_strategy.md
│   ├── 10_independent_verification_and_audit_of_info_md.md
│   ├── 11_actionable_first_place_execution_blueprint.md
│   └── 12_exp_003_convnext_base_dual_gpu_implementation_plan.md
├── reports/                             # Forensic benchmark reports & post-mortems
│   └── EXP-002_COMPREHENSIVE_ANALYSIS_AND_SERVER_BENCHMARK_REPORT.md
├── src/                                 # Production training framework
│   ├── __init__.py
│   ├── calibrate.py                     # Hardware profiling & benchmark telemetry
│   ├── dataset.py                       # Zero-defect dataset ingestion & domain-balanced sampler
│   ├── evaluate.py                      # Offline mADM, mAP, Rank-k, mINP evaluator
│   ├── losses.py                        # Asymmetric Focal Loss (ASL) & Category Softmax
│   ├── metrics.py                       # High-performance NumPy/PyTorch metric engine
│   ├── models/                          # Neural network architectures
│   │   ├── __init__.py
│   │   ├── baseline_resnet.py           # Multi-head ResNet-50 baseline
│   │   ├── convnext_model.py            # ConvNeXt-Base with LayerNorm (domain-invariant)
│   │   └── factory.py                   # Unified model instantiation factory
│   ├── submission_generator.py          # Standalone air-gapped Codabench zip generator
│   └── train.py                         # Production distributed multi-GPU training engine
├── tests/                               # Verification and compliance test suite
│   └── test_submission_simulation.py    # Codabench simulation validator
├── EXPERIMENT_TRACKER.md                # Permanent master experiment scoreboard
├── STEP_TRACKER.md                      # Chronological step-by-step engineering audit log
├── requirements.txt                     # Pinned environment dependencies
└── README.md                            # Project documentation
```

---

## 3. Master Scorecard & Leaderboard Progress

| Experiment | Architecture | Resolution | Val mADM | Val Rank-1 | Codabench Sub ID | Codabench mADM | Codabench Rank | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Starter Kit** | Static Prior | N/A | — | — | Official Baseline | `0.1567` | — | Tripled by our models |
| **EXP-001** | ResNet-50 (1 Ep) | $256 \times 128$ | `0.4073` | `4.51%` | Local Only | N/A | — | Warm-start checkpoint |
| **EXP-002** | ResNet-50 (16 Ep) | $256 \times 128$ | `0.5293` | `11.90%` | `954280` | **`0.4041`** | **`#9 Worldwide`** | **Server Verified** |
| **EXP-003** *(Next)* | ConvNeXt-Base | $384 \times 192$ | *Target: $\ge 0.65$* | *Target: $\ge 25\%$* | Planned | *Target: $\ge 0.63$* | **Target: #1** | **Kaggle 2x T4 Push** |
| **Final Target** | Tri-Ensemble | Multi | *$\ge 0.85$* | *$\ge 60\%$* | Final Submission | **`>= 0.8000`** | **1st Place** | **Workshop Win** |

---

## 4. Key Architectural Innovations

1. **LayerNorm Domain Generalization:** Replaces `BatchNorm2d` with `LayerNorm` (ConvNeXt-Base). Unlike BatchNorm, LayerNorm uses zero stored running statistics, eliminating camera-style covariate shift on unseen surveillance test networks.
2. **High-Resolution Spatial Maps ($384 \times 192$):** $+125\%$ more pixels compared to standard $256 \times 128$, capturing localized accessories (sunglasses, bags, hats, footwear).
3. **Continuous Bounded L1 Ranking:** Outputting continuous probabilities $p_{g, a} \in [0, 1]$ eliminates integer ties that destroy gallery ranking order.
4. **Asymmetric Loss (ASL) + Category-Grouped Softmax:** Dynamic negative probability shifting ($\gamma_-=4.0, \gamma_+=0.0$) addresses severe attribute class imbalance while enforcing anatomical mutual exclusivity.

---

## 5. Kaggle Multi-GPU Execution Guide (2x Tesla T4)

### Prerequisites:
1. Kaggle Notebook with **GPU: 2x Tesla T4 (32GB VRAM total)** and **30GB CPU RAM**.
2. Private GitHub Repository Personal Access Token (PAT).
3. Attached Kaggle Dataset: `upar-challenge-2027-data`.

### Single-Cell Clone & Pull Script:
```python
import os

GITHUB_TOKEN = "your_github_personal_access_token_here"
REPO_URL = f"https://{GITHUB_TOKEN}@github.com/Aditya-Lingam-9000/upar_challenge_26_rws27.git"
REPO_DIR = "/kaggle/working/upar_challenge_26_rws27"

if os.path.exists(REPO_DIR):
    print("Repository exists. Pulling latest commits...")
    os.system(f"cd {REPO_DIR} && git pull")
else:
    print("Cloning private repository...")
    os.system(f"git clone {REPO_URL} {REPO_DIR}")

os.chdir(REPO_DIR)
print("Active Directory:", os.getcwd())
```

---

## 6. Author & Citation

**Participant:** Jyothiradithya Lingam  
**Affiliation:** Advanced Real-World Surveillance Research (WACV 2027)  
**Repository:** [https://github.com/Aditya-Lingam-9000/upar_challenge_26_rws27](https://github.com/Aditya-Lingam-9000/upar_challenge_26_rws27)
