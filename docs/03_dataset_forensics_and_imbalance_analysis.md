# Report 03: Deep Dataset Forensics, Domain Asymmetry & Imbalance Spectrum Analysis

**Document:** Dataset Analysis & Statistical Profiling
**Datasets Analyzed:** Market-1501, PA-100K, PETA, UPAR Harmonized Annotations
**Total Images:** 131,076 (Train: 97,669 | Validation: 33,407)
**Total Query Vocabularies:** Train: 7,204 | Validation: 3,462
**Document Status:** Publication-Grade Statistical Profile

---

## 1. Composition of the Official 2027 UPAR Dataset

The challenge training and development data is constructed by harmonizing three widely studied public pedestrian benchmarks into a standardized 40-attribute vocabulary.

```
+---------------------------------------------------------------------------------------------------+
|                                  OVERALL DATASET DISTRIBUTION TABLE                                |
+------------------+-------------------+---------------------+--------------------+-----------------+
| Sub-Dataset      | Train Images (%)  | Val Images (%)      | Total Images (%)   | Primary Origin  |
+------------------+-------------------+---------------------+--------------------+-----------------+
| PA-100K          |  79,001 (80.89%)  |   9,986 (29.89%)    |  88,987 (67.89%)   | Outdoor Traffic |
| Market-1501      |  10,000 (10.24%)  |  16,458 (49.27%)    |  26,458 (20.19%)   | Campus (Low Res)|
| PETA             |   8,668  (8.87%)  |   6,963 (20.84%)    |  15,631 (11.92%)   | Multi-Source    |
+------------------+-------------------+---------------------+--------------------+-----------------+
| TOTALS           |  97,669 (100.0%)  |  33,407 (100.0%)    | 131,076 (100.0%)   | Unified UPAR    |
+------------------+-------------------+---------------------+--------------------+-----------------+
| Unique Queries   |   7,204 queries   |   3,462 queries     |  10,666 unique     | Semantic IDs    |
+------------------+-------------------+---------------------+--------------------+-----------------+
```

---

## 2. The Structural Asymmetry Dilemma: A Trap for Naive Models

### 2.1 The Training vs. Validation Imbalance Shift

A critical observation emerges when comparing the dataset distribution between Train and Validation:

- In **Train**, PA-100K dominates, comprising **80.89%** of all training samples. Market-1501 contributes only **10.24%**, and PETA contributes **8.87%**.
- In **Validation**, Market-1501 becomes the dominant domain, comprising **49.27%** of samples, while PA-100K drops to **29.89%** and PETA rises to **20.84%**.

```
    TRAINING SPLIT (97,669 Images)              VALIDATION SPLIT (33,407 Images)
+------------------------------------+        +------------------------------------+
| PA-100K: 80.89%                    |        | Market-1501: 49.27%                |
| (Outdoor traffic, high resolution) |        | (Campus surveillance, low-res)     |
+------------------------------------+        +------------------------------------+
| Market: 10.24% | PETA: 8.87%       |        | PA-100K: 29.89% | PETA: 20.84%     |
+------------------------------------+        +------------------------------------+
```

### 2.2 Why Empirical Risk Minimization (ERM) Collapses

Under standard ERM (sampling mini-batches uniformly from `train.csv`):

1. **Domain Overfitting:** Over $8$ out of every $10$ images sampled by the dataloader originate from PA-100K. The model learns visual features specifically tuned to high-angle Chinese intersection cameras, specific lighting conditions, and road surface backgrounds.
2. **Sensor Domain Mismatch:** When evaluated on Market-1501 (which forms half of the validation set) or the hidden evaluation container domain, the model encounters lower spatial resolution ($128 \times 64$ original native resolution), different aspect ratios, and campus background clutter. Feature norms drop, leading to noisy predictions.
3. **The Solution:** Training must implement **domain-balanced sampling** (enforcing equal probability of drawing batches from PA-100K, Market-1501, and PETA) or use **Domain-Adversarial Training (DANN)** to align feature distributions across source domains.

---

## 3. The 40 UPAR Attributes: Vocabulary, Ordering & Imbalance Spectrum

The challenge specification mandates that queries and predictions adhere to a strict, ordered list of 40 binary attributes (indices 0 to 39).

### 3.1 Complete 40-Attribute Reference & Empirical Positive Rates

Empirical positive frequencies $f_a$ computed across the 33,407 validation images (from `attribute_prior.json`):

```
Idx  Attribute Name                  Category        Positive Rate  Imbalance Ratio
-----------------------------------------------------------------------------------
 0   Age-Young                       Age                 0.01102        1 : 90.7
 1   Age-Adult                       Age                 0.96498        1 : 1.04 (Dom)
 2   Age-Old                         Age                 0.02368        1 : 42.2
 3   Gender-Female                   Gender              0.42916        1 : 2.33
 4   Hair-Length-Short               Hair                0.65459        1 : 1.53
 5   Hair-Length-Long                Hair                0.31140        1 : 3.21
 6   Hair-Length-Bald                Hair                0.01260        1 : 79.4
 7   UpperBody-Length-Short          Upper Length        0.62876        1 : 1.59
 8   UpperBody-Color-Black           Upper Color         0.25564        1 : 3.91
 9   UpperBody-Color-Blue            Upper Color         0.08214        1 : 12.2
10   UpperBody-Color-Brown           Upper Color         0.01988        1 : 50.3
11   UpperBody-Color-Green           Upper Color         0.04346        1 : 23.0
12   UpperBody-Color-Grey            Upper Color         0.16209        1 : 6.17
13   UpperBody-Color-Orange          Upper Color         0.00581        1 : 172.1
14   UpperBody-Color-Pink            Upper Color         0.01051        1 : 95.1
15   UpperBody-Color-Purple          Upper Color         0.02532        1 : 39.5
16   UpperBody-Color-Red             Upper Color         0.06942        1 : 14.4
17   UpperBody-Color-White           Upper Color         0.22501        1 : 4.44
18   UpperBody-Color-Yellow          Upper Color         0.04972        1 : 20.1
19   UpperBody-Color-Other           Upper Color         0.07531        1 : 13.3
20   LowerBody-Length-Short          Lower Length        0.37507        1 : 2.67
21   LowerBody-Color-Black           Lower Color         0.48774        1 : 2.05
22   LowerBody-Color-Blue            Lower Color         0.17879        1 : 5.59
23   LowerBody-Color-Brown           Lower Color         0.04481        1 : 22.3
24   LowerBody-Color-Green           Lower Color         0.01497        1 : 66.8
25   LowerBody-Color-Grey            Lower Color         0.17194        1 : 5.82
26   LowerBody-Color-Orange          Lower Color         0.00210        1 : 476.2
27   LowerBody-Color-Pink            Lower Color         0.01548        1 : 64.6
28   LowerBody-Color-Purple          Lower Color         0.00078        1 : 1282.1
29   LowerBody-Color-Red             Lower Color         0.00389        1 : 257.1
30   LowerBody-Color-White           Lower Color         0.04469        1 : 22.4
31   LowerBody-Color-Yellow          Lower Color         0.00168        1 : 595.2
32   LowerBody-Color-Other           Lower Color         0.03724        1 : 26.9
33   LowerBody-Type-Trousers&Shorts  Lower Type          0.90382        1 : 1.11 (Dom)
34   LowerBody-Type-Skirt&Dress      Lower Type          0.09579        1 : 10.4
35   Accessory-Backpack              Accessory           0.20855        1 : 4.79
36   Accessory-Bag                   Accessory           0.29314        1 : 3.41
37   Accessory-Glasses-Normal        Accessory           0.13937        1 : 7.18
38   Accessory-Glasses-Sun           Accessory           0.02176        1 : 46.0
39   Accessory-Hat                   Accessory           0.04248        1 : 23.5
-----------------------------------------------------------------------------------
```

### 3.2 Four Tiers of Imbalance Severity

1. **Tier 1 — Extreme Minority ($< 0.5\%$ Positive Rate):**
   - `LowerBody-Color-Purple`: $0.078\%$ ($1$ in $1,282$ images)
   - `LowerBody-Color-Yellow`: $0.168\%$ ($1$ in $595$ images)
   - `LowerBody-Color-Orange`: $0.210\%$ ($1$ in $476$ images)
   - `LowerBody-Color-Red`: $0.389\%$ ($1$ in $257$ images)
   - *Impact on Model:* A model predicting strictly $0$ for these attributes achieves $>99.6\%$ accuracy on them, but will permanently fail any query requiring them.
2. **Tier 2 — Critical Minority ($0.5\% - 2.5\%$ Positive Rate):**
   - `UpperBody-Color-Orange`: $0.58\%$
   - `UpperBody-Color-Pink`: $1.05\%$
   - `Age-Young`: $1.10\%$
   - `Hair-Length-Bald`: $1.26\%$
   - `LowerBody-Color-Green`: $1.50\%$
   - `LowerBody-Color-Pink`: $1.55\%$
   - `UpperBody-Color-Brown`: $1.99\%$
   - `Accessory-Glasses-Sun`: $2.18\%$
   - `Age-Old`: $2.37\%$
3. **Tier 3 — Moderate Balance ($2.5\% - 45\%$ Positive Rate):**
   - Color classes (Black, Blue, Grey, White), Accessories (Backpack, Bag, Glasses), Gender (Female: $42.9\%$), Hair Length (Long: $31.1\%$).
4. **Tier 4 — Dominant Majority ($> 60\%$ Positive Rate):**
   - `UpperBody-Length-Short`: $62.9\%$
   - `Hair-Length-Short`: $65.5\%$
   - `LowerBody-Type-Trousers&Shorts`: $90.4\%$
   - `Age-Adult`: $96.5\%$

---

## 4. Semantic Hierarchy & Mutual Exclusivity Constraints

Although all 40 attributes are represented as binary $0/1$ flags, the human body imposes strict mutual exclusivity within specific sub-groups:

### 4.1 Mutually Exclusive Subsets

- **Age Subset:** An individual cannot simultaneously be `Age-Young`, `Age-Adult`, and `Age-Old`. Exactly one of these three attributes is $1$:
  $$
  \sum_{a \in \{0, 1, 2\}} y_a = 1
  $$
- **Hair Length Subset:** An individual cannot be simultaneously `Hair-Length-Short`, `Hair-Length-Long`, and `Hair-Length-Bald`:
  $$
  \sum_{a \in \{4, 5, 6\}} y_a = 1
  $$
- **Lower Body Type Subset:** `LowerBody-Type-Trousers&Shorts` and `LowerBody-Type-Skirt&Dress` are mutually exclusive:
  $$
  y_{33} + y_{34} \le 1
  $$

### 4.2 Multi-Label Color Subsets

Pedestrian clothing frequently exhibits patterns, graphics, or stripes. Therefore, color categories (UpperBody Colors: indices 8–19; LowerBody Colors: indices 21–32) are **not strictly mutually exclusive**, but are sparse:

$$
1 \le \sum_{a \in \text{UpperColors}} y_a \le 3
$$

A person wearing a white shirt with red stripes has both `UpperBody-Color-White=1` and `UpperBody-Color-Red=1`.

---

## 5. Query Vocabulary & Mapping Forensics

### 5.1 The `queries.csv` and `ids.csv` Structure

In `UPAR-Challenge-2027/data/annotations/task2/`:

- **Training Set:** Contains **7,204 distinct query vectors** mapped across 97,669 images ($\approx 13.56$ images per query on average).
- **Validation Set:** Contains **3,462 distinct query vectors** mapped across 33,407 images ($\approx 9.65$ images per query on average).
- **Disjoint Query Vocabularies:** The organizers intentionally separated the query sets: **no query in `val/queries.csv` appears in `train/queries.csv`**. This tests generalization to novel attribute combinations that were never seen as a paired query during training.

### 5.2 Multi-Instance Distribution

For validation:

- The maximum number of images matching a single query is **1,421 images** (corresponding to the dominant default query: Adult, Male, Short Hair, Short Upper, Black Upper, Long Lower, Black Trousers, No Accessories).
- Over **820 queries** match exactly $1$ image in the gallery ($R=1$).
- **Strategic Impact:** For queries with $R=1$, the metric evaluates a single rank $r_1$. If the model ranks the true image at rank 1, $AP = 1.0$. If it ranks it at rank 10, $AP = 0.10$. However, $ADM$ will remain high ($\approx 0.70 - 0.85$) if ranks 1–9 are near-misses that match on 38 or 39 attributes.
