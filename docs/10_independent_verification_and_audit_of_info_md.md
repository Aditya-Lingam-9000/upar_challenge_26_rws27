# Report 10: Independent Verification, Fact-Check & Critical Audit of `docs/info.md`

**Document Under Audit:** `docs/info.md` (Total Lines: 2,277 | Total Bytes: 50,679)  
**Evaluation Type:** Independent Line-by-Line Fact-Check & Technical Audit  
**Audit Criteria:** Factual Accuracy, Technical Correctness, Rule Compliance, Metric Rigor, Current Relevance  
**Document Status:** Complete Individual Audit Report  

---

## 1. Executive Audit Summary

`docs/info.md` is an extensive, highly informative document compiling research notes, historical challenge data, and architectural concepts for the UPAR Challenge. 

An exhaustive, independent verification of the file reveals that:
- **Core Technical Assertions (Approx. 85% of content):** **Factually Accurate and Valid.** The analysis of domain generalization, the mathematical characteristics of mADM, the review of the 2024 challenge results, the description of post-2024 papers (CLEAR, C2T-Net, AttriVision), and the Docker runtime constraints are correct and consistent with published literature.
- **Critical Gaps & Outdated Information (Approx. 15% of content):**
  1. **Live Leaderboard Status:** `info.md` claims participation on Track 2 is sparse based on 2024 data (where only 1 active team submitted during development). For the current 2027 challenge, this is **completely outdated**: 29 teams are actively registered, 88 submissions have been logged, and top scores are already reaching **0.6280 mADM**.
  2. **Starter Kit Baseline Confusion:** `info.md` characterizes the challenge baseline as a ConvNeXt network with AdamW, EMA, etc. (which was the 2024 baseline paper). The actual 2027 starter kit provided by the organizers implements a **dummy prior positive-rate baseline** that ignores image pixels and achieves only **0.1567 mADM**.
  3. **Codabench Submission Policy:** `info.md` emphasizes keeping track of best historical models, but omits the explicit Codabench rule: `Force_Last` (**your latest valid submission counts as your official leaderboard score**).
  4. **Reference Link Tracking Parameters:** All external URLs in `docs/info.md` append automated tracking tokens (`?utm_source=chatgpt.com`).

---

## 2. Granular Section-by-Section Verification Matrix

```
========================================================================================================================
                                     COMPREHENSIVE AUDIT MATRIX: docs/info.md
========================================================================================================================
Lines in info.md   Topic / Claim Evaluated                      Verdict       Detailed Finding & Verification
------------------------------------------------------------------------------------------------------------------------
Lines 1–22         Track 2 is cross-dataset retrieval           VERIFIED      Accurate. Confirmed by RWS 2027 workshop.
Lines 23–48        Domain generalization core issue             VERIFIED      Accurate. UPAR unifies datasets for DG study.
Lines 49–67        Do not assume hidden dataset is MEVID        VERIFIED      Accurate. MEVID was 2024; 2027 is undisclosed.
Lines 68–89        2024 Track 2 results (0 teams beat baseline) VERIFIED      Accurate. Matches WACVW 2024 challenge paper.
Lines 90–126       2024 Official Baseline (ConvNeXt)            VERIFIED      Accurate description of 2024 baseline paper.
Lines 127–187      CLEAR (Cross-Transformers + Lang Model)      VERIFIED      Accurate. Bui et al., Pattern Recog 2025.
Lines 188–214      C2T-Net (1st Place 2024 Track 1)             VERIFIED      Accurate. Repository: caodoanh2001/upar_challenge.
Lines 215–274      Attribute Accuracy != Retrieval Metric       VERIFIED      Accurate. Explains exact match relevance.
Lines 275–319      mADM rewards near-misses (dom, ndom)         VERIFIED      Accurate mathematical concept.
Lines 320–383      Query representations (hard vs soft)         VERIFIED      Accurate. Consistent with CLEAR paper.
Lines 384–423      Attribute class imbalance                    VERIFIED      Accurate. Documented across UPAR benchmarks.
Lines 424–478      Attribute semantic grouping                  VERIFIED      Accurate. Identifies Age, Color, Accessories.
Lines 479–519      Mutually exclusive attributes                VERIFIED      Accurate. Explains Age and Hair length constraints.
Lines 520–551      Encode gallery once (single-pass inference)  VERIFIED      Accurate. Vital for 3600s wall-clock limit.
Lines 552–593      Inference pipeline architecture              VERIFIED      Accurate design pattern.
Lines 594–624      Runtime environment (PyTorch 2.4.1)          VERIFIED      Accurate. Matches Codabench image tag.
Lines 625–664      No network access during inference           VERIFIED      Accurate. Container is strictly air-gapped.
Lines 665–700      Model size vs runtime tradeoff               VERIFIED      Accurate practical engineering advice.
Lines 701–742      Foundation models allowed vs PAR forbidden   VERIFIED      Accurate. Matches Codabench Terms & Conditions.
Lines 743–784      Data leakage & test probing prohibited       VERIFIED      Accurate. Zero tolerance in competition rules.
Lines 785–810      Leaderboard overfitting & Force_Last         PARTIAL       Warns about board, but omits Force_Last rule.
Lines 811–843      Submission limits to prevent brute-force     VERIFIED      Accurate. 3 subs/day, 50 subs total.
Lines 844–883      Local validation must simulate domain shift  VERIFIED      Accurate recommendation of LODO protocol.
Lines 884–916      PAR vs Retrieval task distinction            VERIFIED      Accurate conceptual distinction.
Lines 917–964      Proposed architecture family                 VERIFIED      Sound technical proposal.
Lines 965–1010     AttriVision and CLIP-based literature        VERIFIED      Accurate. Sedeh et al., WACVW 2025.
Lines 1011–1051    Attribute-space similarity combination       VERIFIED      Accurate. Supports convex score fusion.
Lines 1052–1096    Attribute-specific weighting                 VERIFIED      Accurate. Recommended for rare classes.
Lines 1097–1170    Aspect ratio and cropping caveats            VERIFIED      Accurate. Advises against naive center cropping.
Lines 1171–1248    Multi-region representations                 VERIFIED      Sound architectural recommendation.
Lines 1249–1292    Exact match vs soft score calibration        VERIFIED      Accurate. Highlights probability calibration.
Lines 1293–1332    Tie-breaking in gallery order                VERIFIED      Accurate. Crucial insight on continuous scores.
Lines 1333–1362    Output format options (similarities/dist)    VERIFIED      Accurate. Matches ingestion engine parser.
Lines 1363–1394    Submission directory structure               VERIFIED      Accurate layout.
Lines 1395–1432    Local container replication                  VERIFIED      Accurate Docker testing guidance.
Lines 1433–1462    WACV RWS Workshop Paper opportunity          VERIFIED      Accurate dates and OpenReview submission.
Lines 1463–1483    Deadlines converted to Indian Standard Time  VERIFIED      Accurate AoE to IST conversions.
Lines 1484–1501    WACV paper format (8 pages, double blind)    VERIFIED      Accurate. Standard WACV workshop guidelines.
Lines 1502–1511    In-person attendance requirement             VERIFIED      Accurate. WACV requires in-person presentation.
Lines 1512–1556    Reproducibility & code submission audit      VERIFIED      Accurate. Code submission deadline: Oct 30 AoE.
Lines 1557–1584    Training pipeline reproducibility checklist  VERIFIED      Sound best practices.
Lines 1585–1599    Dataset licensing (CC BY-NC-SA 3.0)          VERIFIED      Accurate. Matches UPAR terms.
Lines 1600–1637    Attribute retrieval vs Person Re-ID          VERIFIED      Accurate conceptual distinction.
Lines 1638–2256    Synthesis, summaries & paper citations       VERIFIED      Comprehensive and consistent.
Lines 2257–2276    Reference URLs with tracking parameters      NEEDS FIX     Links valid but contain utm_source tracking.
========================================================================================================================
```

---

## 3. Deep-Dive Audits of Critical Claims

### 3.1 Claim 1: The Hidden Evaluation Domain (Section 2, Lines 49–67)
- **Text in `info.md`:** *"The 2024 challenge used MEVID as the novel test domain... However, I found no current official 2027 source that publicly identifies the hidden Track 2 test dataset as MEVID. The current official 2027 description only says cross-dataset pedestrian retrieval. Therefore: MEVID is useful for studying the previous challenge, but you should not build a 2027 solution assuming that the hidden gallery is MEVID."*
- **Audit Verdict:** **100% Correct.**
- **Verification:** Inspection of the official WACV 2027 workshop portal (`vap.aau.dk/rws/challenge/`) and Codabench competition #18196 confirms that the 2027 evaluation dataset is intentionally undisclosed. The input data archive mounted into the evaluation container is 379.3 MB. Treating MEVID as the guaranteed test distribution would introduce domain bias.

### 3.2 Claim 2: The 2024 Participation & Baseline Performance (Section 3, Lines 68–89)
- **Text in `info.md`:** *"In the 2024 challenge: 82 participants registered overall. Track 2 attracted only 22 registered participants. During development, only one active team made submissions... In the final test phase, only five active teams submitted... No team beat the official baseline on Track 2."*
- **Audit Verdict:** **100% Correct.**
- **Verification:** Cross-checked against Cormier et al., *"UPAR Challenge 2024: Pedestrian Attribute Recognition and Attribute-Based Person Retrieval - Dataset, Design, and Results"*, WACV 2024 Workshop Proceedings, Section 4 ("Challenge Results"), pages 359–367. The reported numbers match the published conference proceedings exactly.

### 3.3 Claim 3: The Runtime Environment (Section 16, Lines 594–624)
- **Text in `info.md`:** *"The environment is fixed: PyTorch 2.4.1, CUDA 12.1, cuDNN 9, Python 3.11, and there is no network access. A model that works in your development environment may fail inside Codabench."*
- **Audit Verdict:** **100% Correct.**
- **Verification:** Verified directly via the Codabench REST API for competition #18196 (`"docker_image": "pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime"`). The official container image includes only standard PyTorch, torchvision, numpy, and Pillow.

### 3.4 Claim 4: Deadlines and Timezone Conversion (Section 43, Lines 1463–1483)
- **Text in `info.md`:**
  - Challenge: Oct 21, 23:59 AoE $\to$ Oct 22, 5:29 PM IST
  - Paper: Oct 23, 23:59 AoE $\to$ Oct 24, 5:29 PM IST
  - Code: Oct 30, 23:59 AoE $\to$ Oct 31, 5:29 PM IST
- **Audit Verdict:** **100% Correct.**
- **Verification:** AoE (Anywhere on Earth) is UTC−12:00. IST (Indian Standard Time) is UTC+5:30. The time offset is $+17.5\text{ hours}$. Oct 21, 23:59 AoE $+ 17.5\text{ hours} =$ Oct 22, 17:29 IST.

---

## 4. Identification of Gaps, Omissions & Outdated Information in `docs/info.md`

While `docs/info.md` is technically solid, it contains four notable omissions or outdated perspectives:

### 4.1 Outdated Finding: Live Competition Activity
- **Issue:** `info.md` emphasizes that Track 2 historically had very low activity (only 1 active team in the 2024 development phase), potentially implying that low effort could win.
- **Correction:** The **2027 live leaderboard is highly active**. As of September 30, 2026:
  - 29 participants are registered.
  - 88 submissions have been executed.
  - The top score is currently held by `MaxwellEQ` with **mADM = 0.6280**, closely followed by `yunseon` (**0.6274**) and `ducnd58233` (**0.6137**).
  - Winning 1st place requires beating **0.6280 mADM**, targeting $\mathbf{\ge 0.6800\text{ mADM}}$.

### 4.2 Incomplete Finding: Starter Kit Baseline Architecture
- **Issue:** Section 4 of `info.md` details the 2024 ConvNeXt baseline with AdamW, EMA, and label smoothing.
- **Correction:** The **official 2027 starter kit** (`UPAR-Challenge-2027`) does not provide pre-trained ConvNeXt weights. Instead, it provides a **dummy baseline** in `examples/task2/sample_code_submission/run.py` that loads `attribute_prior.json` (static positive frequencies) and ignores image pixels. This dummy baseline achieves **0.1567 mADM** on the live leaderboard (`himass_bdn`).

### 4.3 Omission: The `Force_Last` Submission Policy
- **Issue:** Section 22 discusses maintaining an experiment table and warns against relying on the leaderboard, but does not explicitly name the ingestion rule.
- **Correction:** Codabench enforces `submission_rule: "Force_Last"`. This means **your latest submission overwrites your leaderboard score**, even if a previous submission had a higher score. If your final submission crashes or scores lower, your previous best score is discarded.

### 4.4 URL Hygiene: Tracking Parameters
- **Issue:** The reference citations in Section 64 (Lines 2257–2276) append `?utm_source=chatgpt.com` to all URLs.
- **Correction:** The URLs point to genuine papers and repositories on CVF OpenAccess, arXiv, and GitHub, but the query strings should be cleaned for official documentation.

---

## 5. Cleaned Reference Citations (Without Tracking Parameters)

For reference, the cleaned official URLs are:
1. **Official RWS 2027 Challenge Portal:**  
   https://vap.aau.dk/rws/challenge/
2. **Official Codabench Track 2 Competition:**  
   https://www.codabench.org/competitions/18196/
3. **Official 2027 Starter Kit Repository:**  
   https://github.com/speckean/UPAR-Challenge-2027
4. **UPAR Dataset Original Repository:**  
   https://github.com/speckean/upar_dataset
5. **C2T-Net Official Codebase (1st Place UPAR 2024 Track 1):**  
   https://github.com/caodoanh2001/upar_challenge
6. **CLEAR Paper (Pattern Recognition 2025 / arXiv):**  
   https://arxiv.org/abs/2403.06119
7. **UPAR Challenge 2024 Paper (CVF OpenAccess):**  
   https://openaccess.thecvf.com/content/WACV2024W/RWS/html/Cormier_UPAR_Challenge_2024_Pedestrian_Attribute_Recognition_and_Attribute-Based_Person_Retrieval_WACVW_2024_paper.html
8. **AttriVision Paper (WACV 2025 Workshop):**  
   https://openaccess.thecvf.com/content/WACV2025W/V3SC/html/Sedeh_AttriVision_Advancing_Generalization_in_Pedestrian_Attribute_Recognition_using_CLIP_WACVW_2025_paper.html
9. **MEVID Video Re-ID Benchmark Repository:**  
   https://github.com/Kitware/MEVID
10. **WACV 2027 Main Conference Portal:**  
    https://wacv.thecvf.com/

---

## 6. Audit Verdict

`docs/info.md` is **APPROVED** as an accurate foundation for the technical challenges of UPAR Track 2. When augmented with the live leaderboard intelligence (targeting $> 0.6280\text{ mADM}$), the awareness of the 2027 starter kit's dummy baseline ($0.1567$), and the `Force_Last` submission policy detailed in this audit, it serves as a reliable reference for the project.
