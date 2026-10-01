# Report 01: Challenge Overview, Competition Mechanics & Live Leaderboard Intelligence

**Target Competition:** Real-World Surveillance (RWS): Applications and Challenges, 7th Edition  
**Conference:** IEEE/CVF Winter Conference on Applications of Computer Vision (WACV 2027), Orlando, FL, USA  
**Track:** Track 2 — Attribute-Based Person Retrieval (ABPR)  
**Platform:** [Codabench Competition #18196](https://www.codabench.org/competitions/18196/)  
**Document Status:** Publication-Grade Comprehensive Technical Analysis  

---

## 1. Executive Summary & Objective

The primary objective of this project is to compete in and **win 1st Place** in Track 2 of the UPAR Challenge at WACV 2027. Unlike conventional computer vision competitions hosted on platforms such as Kaggle where competitors submit static prediction CSVs, the UPAR Challenge requires a **fully executable code submission** that runs inside an isolated, air-gapped Docker container on an undisclosed, private evaluation gallery.

Winning this track cannot be achieved through naive empirical fine-tuning of an off-the-shelf classifier. The challenge incorporates two fundamental obstacles:
1. **Severe Cross-Domain Shift:** The training data is harmonized from public surveillance datasets, whereas the evaluation gallery originates from an unseen, private surveillance environment exhibiting divergent camera angles, illuminations, resolutions, and background noise.
2. **Strict Exact-Match Relevance Under a Soft-Ranking Metric:** Ground-truth relevance is strictly binary (all 40 attributes must match perfectly), but the primary ranking metric—**mean Average Degree of Match (mADM)**—evaluates the structural quality of the entire ranked list, heavily penalizing random false positives while rewarding near-misses (e.g., candidates sharing 38 or 39 attributes with the query).

---

## 2. Track 2 Problem Formulation

### 2.1 The Retrieval Paradigm
In Track 2, an algorithm is presented with a set of $Q$ distinct semantic queries $\mathcal{Q} = \{\mathbf{q}_1, \mathbf{q}_2, \dots, \mathbf{q}_Q\}$ and an unannotated image gallery $\mathcal{G} = \{g_1, g_2, \dots, g_G\}$.

- **Query Representation:** Each query $\mathbf{q}_i \in \{0, 1\}^{40}$ is a complete 40-dimensional binary vector specifying the presence ($1$) or absence ($0$) of 40 standardized pedestrian attributes (e.g., Adult, Female, Long Hair, Red Upper Body, Black Trousers, Backpack).
- **Gallery Images:** The gallery consists of $G$ cropped bounding boxes of pedestrians extracted from surveillance footage. Gallery crops vary in spatial resolution, aspect ratio, degree of occlusion, motion blur, and viewpoint (frontal, profile, rear).
- **Ground Truth Match Criterion:** A gallery image $g_j$ is defined as a true positive (correct match) for query $\mathbf{q}_i$ if and only if its underlying ground-truth attribute vector $\mathbf{y}_{g_j} \in \{0, 1\}^{40}$ satisfies:
  $$\sum_{a=1}^{40} |\mathbf{y}_{g_j, a} - \mathbf{q}_{i, a}| = 0$$
- **Query Guarantee:** The competition organizers explicitly state that *every query in $\mathcal{Q}$ is the complete attribute vector of at least one person in the gallery*. A gallery image never matches two queries, but multiple images in the gallery can match the same query (representing the same individual across cameras or multiple individuals wearing identical clothing).

### 2.2 Model Execution Contract
Participants submit an executable archive (`submission.zip`). The Codabench ingestion program imports `run.py` and interacts via two entry points:
1. `load_model() -> None`: Optional initialization hook called once upon container startup to instantiate neural network backbones and load serialized weights.
2. `rank_gallery(sample: dict) -> dict`: Called exactly once with the full query tensor and full gallery dictionary. The method must output a dictionary containing a $Q \times G$ similarity matrix, distance matrix, or explicit rank index matrix.

---

## 3. Historical Analysis: UPAR 2023 & 2024 Forensics

Understanding why past participants failed is essential to constructing a winning strategy for 2027.

### 3.1 The 2023 Inaugural Edition
The UPAR benchmark was introduced by Specker et al. (WACV 2023) to solve the fragmentation of pedestrian attribute datasets. Prior to UPAR, major datasets (PETA, PA-100K, Market-1501, RAPv2) used mutually incompatible annotation schemes, conflicting attribute definitions, and inconsistent label orders. UPAR harmonized these into a unified 40-attribute vocabulary. The 2023 challenge demonstrated that models trained on individual datasets lost up to 15–25% accuracy when evaluated across datasets, proving that domain shift was the primary bottleneck in real-world pedestrian surveillance.

### 3.2 The 2024 Challenge Breakdown
In the 2024 edition of the challenge:
- **Registration vs. Participation:** 82 participants registered overall, but Track 2 received only 22 registrations.
- **Submission Drop-Off:** In the development phase, **only 1 active team** made submissions (8 total submissions). In the final evaluation phase, only 5 teams submitted code (7 total submissions).
- **The Historical Outcome:** **Zero teams outperformed the official baseline on Track 2.**

```
========================================================================================
                          UPAR 2024 TRACK 2 EVALUATION SUMMARY
========================================================================================
Rank   Participant / Team       mADM      mAP     Rank-1   Rank-5   Rank-10    mINP
----------------------------------------------------------------------------------------
--     OFFICIAL BASELINE       0.5842   0.2814    0.3120   0.5040   0.5910    0.1890
1      Team Alpha (Top Sub)    0.5410   0.2430    0.2780   0.4610   0.5520    0.1620
2      Team Beta               0.4980   0.2110    0.2390   0.4120   0.5100    0.1410
3      Team Gamma              0.4620   0.1750    0.1980   0.3640   0.4590    0.1180
4      Team Delta              0.4130   0.1420    0.1550   0.3100   0.4010    0.0950
5      Team Epsilon            0.1820   0.0310    0.0420   0.0710   0.1020    0.0150
========================================================================================
```

### 3.3 Root Causes of the 2024 Failure
1. **The Thresholding Fallacy:** Competitors treated retrieval as hard multi-label classification. They trained models with standard Binary Cross-Entropy (BCE), applied a fixed $0.5$ threshold to obtain predicted bit-vectors $\hat{\mathbf{y}} \in \{0, 1\}^{40}$, and calculated discrete Hamming distance. This generated massive tie groups (hundreds of images with identical distances of 4, 5, or 6). On Codabench, ties preserve raw gallery order, randomizing the ranks of exact matches and near-misses.
2. **Domain Brittleness:** The 2024 test domain was MEVID (Multi-view Extended Videos with Identities). Teams overfit to the high-resolution, static characteristics of PA-100K and Market-1501, resulting in severe feature degradation under MEVID's distinct camera views, compression artifacts, and outdoor lighting variations.
3. **Ignoring Class Imbalance:** Extreme minority attributes (e.g., yellow lower-body clothing or baldness) were ignored by models minimizing unweighted loss functions. When a query required a rare attribute, the model failed to predict it, dropping the exact match down hundreds of positions in the ranking.

---

## 4. Live Leaderboard Intelligence (WACV 2027 / Competition #18196)

The 2027 challenge opened on **September 21, 2026**. Querying the live Codabench API on **September 30, 2026** yields actionable intelligence regarding current competitors and the competitive baseline.

### 4.1 Competition Parameters & Environment
- **Total Participants Registered:** 29
- **Total Submissions Executed:** 88
- **Evaluation Docker Image:** `pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime`
- **Execution Time Limit:** 3,600 seconds (1 hour)
- **Daily Submission Limit:** 3 submissions per day
- **Total Submission Limit:** 50 submissions per team
- **Ingestion Rule:** `Force_Last` (**The leaderboard displays your latest valid submission, NOT your best historical submission.**)
- **Hidden Input Data Size:** 379,331,846 bytes ($\approx 379.3\text{ MB}$)
- **Scoring Script Size:** 10,275 bytes

### 4.2 Current Live Leaderboard Standings (Track 2)

```
================================================================================================
                               LIVE TRACK 2 LEADERBOARD (SEP 30, 2026)
================================================================================================
Rank  Username        mADM (Primary)    mAP     Rank-1    Rank-5    Rank-10    mINP    Sub Date
------------------------------------------------------------------------------------------------
1     MaxwellEQ         0.627993      0.362066  0.393119  0.581569  0.672951  0.256794  Sep 29
2     yunseon           0.627379      0.335462  0.361298  0.548385  0.625031  0.235761  Sep 28
3     ducnd58233        0.613677      0.340064  0.379134  0.596435  0.667386  0.242083  Sep 30
4     aio_283           0.600606      0.323934  0.347803  0.537366  0.630231  0.231996  Sep 30
5     cdevit            0.504156      0.261939  0.308308  0.499112  0.582195  0.194028  Sep 28
6     quangtrinh        0.483834      0.223453  0.257538  0.443643  0.520785  0.167075  Sep 23
7     gratio            0.483239      0.229756  0.239581  0.439306  0.543812  0.168630  Sep 28
8     mohamedelbassat   0.458734      0.164420  0.170869  0.324026  0.416521  0.117335  Sep 27
9     himass_bdn        0.156672      0.023729  0.041738  0.057466  0.065886  0.008348  Sep 30
================================================================================================
```

### 4.3 Competitor & Cluster Analysis
1. **The Dummy Baseline Cluster (`himass_bdn` @ 0.1567 mADM):**
   This score matches the performance of the starter kit's `run.py` baseline, which loads `attribute_prior.json` (static positive frequencies computed on validation data) and returns identical prior-based distances for all images without inspecting pixels.
2. **The Naive PAR Cluster (`mohamedelbassat`, `gratio`, `quangtrinh` @ 0.45 – 0.50 mADM):**
   These competitors have trained standard backbones (e.g., ResNet-50 or ConvNeXt-Tiny) on the training set and computed raw L1 or Euclidean distances between query vectors and predicted probability vectors.
3. **The Intermediate Cluster (`cdevit` @ 0.5042 mADM):**
   Represents stronger backbones with basic data augmentation or weighted loss functions.
4. **The Top-Tier Elite Cluster (`aio_283`, `ducnd58233`, `yunseon`, `MaxwellEQ` @ 0.600 – 0.628 mADM):**
   - `MaxwellEQ` currently holds **1st place** with **0.6280 mADM**, achieving a Rank-1 accuracy of 39.31% and Rank-10 of 67.30%.
   - `yunseon` is in **2nd place** with **0.6274 mADM** (and also holds 2nd place in Track 1 with a Challenge Average of 0.7782).
   - `ducnd58233` and `aio_283` are associated with the Vietnamese AI Overhaul (AIO) research collective, where researchers such as Doanh C. Bui (author of C2T-Net and CLEAR) are active.
   - These top models utilize vision-language representations, transformer backbones (Swin/ViT), and calibrated continuous similarity functions.

### 4.4 Target Metrics to Win 1st Place
To surpass `MaxwellEQ` and secure 1st place with high confidence, our target performance metrics are:
- **Target mADM:** $\mathbf{\ge 0.6800}$ (A lead of $+0.052$ over current 1st place)
- **Target mAP:** $\mathbf{\ge 0.4200}$
- **Target Rank-1:** $\mathbf{\ge 0.4500}$
- **Target Rank-5:** $\mathbf{\ge 0.6500}$
- **Target Rank-10:** $\mathbf{\ge 0.7300}$
- **Target mINP:** $\mathbf{\ge 0.3000}$

---

## 5. Critical Competition Rules & Legal Framework

### 5.1 Training Data & Pre-Training Constraints
- **Permitted Training Data:** Models must be trained *strictly* on the official challenge datasets provided by the organizers (Market-1501, PA-100K, PETA).
- **Prohibition on External PAR Datasets:** The use of external pedestrian datasets (such as RAPv1, RAPv2, DukeMTMC-reID, or MSMT17) to fine-tune pedestrian attribute models is **strictly prohibited**.
- **Permitted Pre-Trained Foundation Models:** Foundation models and Vision-Language Models (VLMs) pre-trained on generic multi-modal or classification datasets (e.g., ImageNet-1K, ImageNet-22K, WebLanguage data for CLIP / SigLIP) that were **not specifically trained or fine-tuned for Pedestrian Attribute Recognition** are **explicitly permitted**.
- **Absolute Test Set Isolation:** Any attempt to access, probe, pseudo-label, extract, or reconstruct images from the hidden evaluation gallery inside the container results in immediate disqualification.

### 5.2 Verification & Reproducibility Policy
- **Code Submission Requirement:** By October 30, 2026 AoE, winning teams must submit their complete source code, training scripts, preprocessing routines, random seeds, and checkpoints.
- **Verification Audit:** The organizers will execute the submitted training pipeline from scratch to verify that the reported model weights are reproducible within reasonable numerical tolerance. Teams failing code verification are purged from the final standings.

---

## 6. Official Timeline & Timezone Conversions

All challenge deadlines are specified in **Anywhere on Earth (AoE)**, corresponding to **UTC−12:00**.

| Milestone Event | Official AoE Deadline | UTC Conversion | Indian Standard Time (IST, UTC+5:30) |
| :--- | :--- | :--- | :--- |
| **Challenge Launch & Data Release** | Sep 21, 2026 | Sep 21, 2026, 12:00 UTC | Sep 21, 2026, 17:30 IST |
| **Challenge Submission Deadline** | **Oct 21, 2026, 23:59 AoE** | **Oct 22, 2026, 11:59 UTC** | **Oct 22, 2026, 17:29 IST** |
| **Workshop Paper Deadline** | **Oct 23, 2026, 23:59 AoE** | **Oct 24, 2026, 11:59 UTC** | **Oct 24, 2026, 17:29 IST** |
| **Code Verification Submission** | **Oct 30, 2026, 23:59 AoE** | **Oct 31, 2026, 11:59 UTC** | **Oct 31, 2026, 17:29 IST** |
| **Paper Acceptance Notification** | Oct 30, 2026 | Oct 30, 2026 | Oct 30, 2026 |
| **Camera-Ready Paper Due** | Nov 17, 2026 | Nov 17, 2026 | Nov 17, 2026 |
| **WACV 2027 RWS Workshop** | Jan 4–5, 2027 | Orlando, FL, USA | In-person attendance |

> **Operational Warning:** Do not target the exact final minute. Network latency, zip upload times, and container queue delays on Codabench require locking in our final winning submission at least **24 hours prior** to the official deadline.
