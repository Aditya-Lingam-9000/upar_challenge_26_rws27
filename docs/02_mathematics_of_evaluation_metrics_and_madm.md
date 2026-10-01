# Report 02: Mathematical Foundations of Evaluation Metrics: Decoupling mADM, mAP, Rank-k & Tie-Breaking Dynamics

**Document:** Mathematical Analysis & Metric Optimization  
**Target Metric:** mean Average Degree of Match (mADM)  
**Associated Metrics:** mAP, Rank-1, Rank-5, Rank-10, mINP  
**Document Status:** Publication-Grade Mathematical Specification  

---

## 1. Mathematical Formulation of the Retrieval Metrics

In Track 2, for every semantic query $\mathbf{q} \in \{0, 1\}^{40}$, an algorithm returns a permutation of the gallery indices:
$$\pi = [g_{(1)}, g_{(2)}, \dots, g_{(G)}]$$
where $g_{(1)}$ denotes the top-ranked candidate and $g_{(G)}$ denotes the lowest-ranked candidate.

Let $R \ge 1$ denote the number of ground-truth exact matches for query $\mathbf{q}$ in the gallery $\mathcal{G}$. An image $g$ is an exact match if and only if $\mathbf{y}_g = \mathbf{q}$, where $\mathbf{y}_g \in \{0, 1\}^{40}$ is the ground-truth attribute vector of $g$.

Let the sorted ranks of these $R$ correct matches in permutation $\pi$ be:
$$1 \le r_1 < r_2 < \dots < r_R \le G$$

---

## 2. Standard Information Retrieval Metrics: mAP, Rank-k, and mINP

### 2.1 Average Precision (AP)
Average Precision measures the area under the Precision-Recall curve for query $\mathbf{q}$:
$$AP = \frac{1}{R} \sum_{k=1}^R \text{Precision}@(r_k) = \frac{1}{R} \sum_{k=1}^R \frac{k}{r_k}$$
where $\text{Precision}@(r_k) = \frac{k}{r_k}$ is the fraction of true matches among the top $r_k$ retrieved images.

- **Mean Average Precision (mAP):**
  $$mAP = \frac{1}{Q} \sum_{i=1}^Q AP_i$$
- **Limitation of AP in Attribute Retrieval:** AP is an indicator-based, binary metric. If rank 1 contains an image differing from query $\mathbf{q}$ in only 1 attribute ($39/40$ correct), its precision contribution is strictly **0**. It is penalized identically to an image differing in all 40 attributes.

### 2.2 Rank-k (Cumulative Match Characteristic at Rank k)
Rank-$k$ represents the probability that at least one correct match appears within the top $k$ positions:
$$\text{Rank-}k = \frac{1}{Q} \sum_{i=1}^Q \mathbb{I}(r_{i, 1} \le k)$$
Track 2 specifically logs:
- **Rank-1:** Share of queries where the top retrieved image is an exact match.
- **Rank-5:** Share of queries with at least one exact match in the top 5 results.
- **Rank-10:** Share of queries with at least one exact match in the top 10 results.

### 2.3 Inverse Negative Penalty (INP) & mINP
Introduced in person re-identification to penalize hard-to-find matches, INP evaluates the rank of the *last* retrieved correct match ($r_R$):
$$INP = \frac{R}{r_R}$$
- If all $R$ matches appear at the very top of the ranking ($r_1=1, r_2=2, \dots, r_R=R$), then $INP = \frac{R}{R} = 1.0$.
- If the hardest match is pushed deep into the gallery (e.g., $R=2$, but $r_2=500$), then $INP = \frac{2}{500} = 0.004$.
- **Mean Inverse Negative Penalty (mINP):**
  $$mINP = \frac{1}{Q} \sum_{i=1}^Q INP_i$$

---

## 3. The Decisive Challenge Metric: Average Degree of Match (ADM) & mADM

The official ranking metric for Track 2 is **mADM (mean Average Degree of Match)**, formalized by Specker & Beyerer (2023).

### 3.1 Step 1: Degree of Match ($\text{dom}$)
For query $\mathbf{q} \in \{0, 1\}^{40}$ and any arbitrary gallery image $g \in \mathcal{G}$, the raw degree of match is the proportion of the 40 binary attributes on which $g$ and $\mathbf{q}$ agree:
$$\text{dom}(g, \mathbf{q}) = \frac{1}{40} \sum_{a=1}^{40} \mathbb{I}(\mathbf{y}_{g, a} == \mathbf{q}_a) = 1.0 - \frac{1}{40} \|\mathbf{y}_g - \mathbf{q}\|_1$$
- For an exact match $g^*$: $\text{dom}(g^*, \mathbf{q}) = 1.0$.
- For an image agreeing on 38 attributes: $\text{dom}(g, \mathbf{q}) = \frac{38}{40} = 0.95$.
- For an image agreeing on 20 attributes: $\text{dom}(g, \mathbf{q}) = \frac{20}{40} = 0.50$.

### 3.2 Step 2: Gallery-Normalized Degree of Match ($\text{ndom}$)
Because random gallery candidates naturally share common attributes (e.g., Adult, Trousers), raw $\text{dom}$ has an empirical baseline mean across the gallery. ADM normalizes $\text{dom}$ per query such that an average gallery image earns zero credit, while a perfect match earns 1.0:

$$\mu_{\text{dom}}(\mathbf{q}) = \frac{1}{G} \sum_{g' \in \mathcal{G}} \text{dom}(g', \mathbf{q})$$
$$M_{\text{dom}}(\mathbf{q}) = \max_{g' \in \mathcal{G}} \text{dom}(g', \mathbf{q})$$

Since every query $\mathbf{q}$ is guaranteed to match at least one image in the gallery, $M_{\text{dom}}(\mathbf{q}) \equiv 1.0$. The normalized degree of match is defined as:
$$\text{ndom}(g, \mathbf{q}) = \max\left(0, \frac{\text{dom}(g, \mathbf{q}) - \mu_{\text{dom}}(\mathbf{q})}{1.0 - \mu_{\text{dom}}(\mathbf{q})}\right)$$
- If $\text{dom}(g, \mathbf{q}) \le \mu_{\text{dom}}(\mathbf{q})$, then $\text{ndom}(g, \mathbf{q}) = 0.0$.
- If $\text{dom}(g, \mathbf{q}) = 1.0$, then $\text{ndom}(g, \mathbf{q}) = 1.0$.

### 3.3 Step 3: Average Degree of Match (ADM)
For the ordered permutation $\pi = [g_{(1)}, g_{(2)}, \dots, g_{(G)}]$ with exact match ranks $r_1 < r_2 < \dots < r_R$:
$$ADM(\pi, \mathbf{q}) = \frac{1}{R} \sum_{k=1}^R \left( \frac{1}{r_k} \sum_{i=1}^{r_k} \text{ndom}(g_{(i)}, \mathbf{q}) \right)$$

Finally, the challenge score is the mean across all queries:
$$mADM = \frac{1}{Q} \sum_{q=1}^Q ADM(\pi_q, \mathbf{q})$$

---

## 4. Mathematical Comparison: Why mADM Fundamentally Changes Ranking Dynamics

To illustrate the mathematical distinction between AP and ADM, consider an empirical query where $\mu_{\text{dom}} = 0.65$ (average gallery image agrees on 26/40 attributes). 
Let $R=1$, meaning there is exactly 1 correct match $g^*$ in the gallery. Suppose our model places $g^*$ at rank $r_1 = 5$.

```
+-------------------------------------------------------------------------------------------------+
|                             CASE STUDY: EXACT MATCH LOCATED AT RANK 5                            |
+-------------------------------------------------------------------------------------------------+
| Scenario A (Random False Positives):                                                           |
| Ranks 1-4 are filled with random gallery images having dom = 0.60 (ndom = 0).                  |
| Rank 5 is the exact match (dom = 1.0, ndom = 1.0).                                              |
|                                                                                                 |
| AP  = 1/5 = 0.2000                                                                             |
| ADM = (ndom(g1) + ndom(g2) + ndom(g3) + ndom(g4) + ndom(g5)) / 5                                |
|     = (0.0 + 0.0 + 0.0 + 0.0 + 1.0) / 5 = 1.0 / 5 = 0.2000                                     |
+-------------------------------------------------------------------------------------------------+
| Scenario B (Near-Misses Placed Ahead of Target):                                                |
| Ranks 1-4 are filled with near-misses agreeing on 38/40 attributes (dom = 0.95).               |
| ndom = (0.95 - 0.65) / (1.0 - 0.65) = 0.30 / 0.35 = 0.8571                                    |
| Rank 5 is the exact match (dom = 1.0, ndom = 1.0).                                              |
|                                                                                                 |
| AP  = 1/5 = 0.2000  (Notice: AP is completely unchanged!)                                       |
| ADM = (0.8571 + 0.8571 + 0.8571 + 0.8571 + 1.0) / 5                                            |
|     = (3.4284 + 1.0) / 5 = 4.4284 / 5 = 0.8857                                                 |
+-------------------------------------------------------------------------------------------------+
| Gain in ADM from placing near-misses ahead of random clutter: +0.6857 (+342% improvement!)      |
+-------------------------------------------------------------------------------------------------+
```

### Mathematical Insights:
1. **AP is insensitive to ranking quality prior to $r_k$:** AP only cares whether preceding images are exact matches.
2. **ADM rewards semantically structured rankings:** Even if a model fails to place the exact match at rank 1, ranking plausible candidates with 37–39 matching attributes preserves high ADM.
3. **The Penalty of Random Clutter:** If unrelated images pollute the top ranks, the numerator sum in ADM collapses to 1.0, causing the score to decay as $\frac{1}{r_k}$.

---

## 5. The Tie-Breaking Catastrophe: Analysis of Codabench's Gallery Rule

### 5.1 The Competition Rule
The Codabench submission documentation explicitly states:
> *"If you return distances or similarities, equal values are ranked in gallery order. Avoid large groups of ties, they make your result depend on that order."*

### 5.2 Why Integer Hamming Distances Destroy Performance
Suppose a model produces binary hard predictions $\hat{\mathbf{y}} \in \{0, 1\}^{40}$ and outputs the L1 / Hamming distance:
$$D_{\text{Hamming}}(g, \mathbf{q}) = \sum_{a=1}^{40} |\hat{\mathbf{y}}_{g, a} - \mathbf{q}_a| \in \{0, 1, 2, \dots, 40\}$$

In a gallery of $G = 30,000$ images:
- Suppose for a query, the model correctly predicts 39 attributes for an exact match, yielding distance $D = 1$.
- However, there are 85 other gallery candidates that also have distance $D = 1$.
- Because all 86 candidates share the identical distance $D = 1.000$, Python / NumPy stable sorting preserves their original input index in `images.csv`.
- If the exact match happens to appear near the end of `images.csv` (e.g., at position 80 within the tied group), its rank drops from $r_1 = 1$ to $r_1 = 80$.
- **Score Impact:**
  $$\frac{1}{r_1} \text{ collapses from } \frac{1}{1} = 1.0 \longrightarrow \frac{1}{80} = 0.0125$$
  This single tie-breaking artifact reduces the ADM contribution of that query by **98.7%**.

### 5.3 The Continuous Probability Solution
To guarantee monotonic, strictly unique sorting without ties, models must output continuous distances derived from unthresholded probabilities $p_{g, a} \in [0, 1]$:
$$D_{\text{cont}}(g, \mathbf{q}) = \sum_{a=1}^{40} \left( q_a (1 - p_{g, a}) + (1 - q_a) p_{g, a} \right)$$
Because probabilities $p_{g, a}$ are 32-bit floating point numbers, the sum produces fine-grained distances:
$$\{0.142857, 0.142891, 0.143012, \dots\}$$
Ties are mathematically eliminated ($P(\text{tie}) \approx 0$), ensuring candidate ranking is dictated strictly by model confidence rather than arbitrary CSV file order.

---

## 6. Metric Surrogate Objectives for Loss Function Formulation

Because mADM involves non-differentiable rank sorting operations, it cannot be optimized directly with gradient descent. We construct smooth, continuous surrogates:

### 6.1 Soft-Rank Formulation
The discrete rank of an exact match $g^*$ can be approximated by a soft ranking operator over similarity scores $s_g = -D(g, \mathbf{q})$:
$$\hat{r}(g^*) = 1 + \sum_{g \neq g^*} \sigma\left(\frac{s_g - s_{g^*}}{\tau}\right)$$
where $\sigma$ is the standard sigmoid function and $\tau > 0$ is a temperature hyperparameter. As $\tau \to 0$, $\hat{r}(g^*)$ converges to the discrete rank.

### 6.2 Weighted Attribute Distance with Class Importance
Not all attributes are equally informative. An agreement on a rare attribute (e.g., `LowerBody-Color-Purple`, positive rate $0.08\%$) provides vastly stronger discriminative evidence of identity than agreement on a ubiquitous attribute (e.g., `Age-Adult`, positive rate $96.5\%$).

We define the information-weighted attribute distance:
$$D_w(g, \mathbf{q}) = \sum_{a=1}^{40} w_a \cdot \left| p_{g, a} - \mathbf{q}_a \right|$$
where weights $w_a$ are inverse-frequency scaled:
$$w_a = -\log_2(f_a + \epsilon)$$
where $f_a$ is the empirical training prior for attribute $a$. Agreeing on rare attributes heavily compresses the predicted distance, pushing true matches to the top ranks and directly maximizing mADM.
