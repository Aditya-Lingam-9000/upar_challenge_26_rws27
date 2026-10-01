I researched this substantially beyond the supplied PDF and Codabench page. I ran well over the requested 30–50 targeted web searches across the official RWS/WACV pages, UPAR/UPAR-Challenge repositories, the 2023/2024 challenge papers, the original UPAR paper, MEVID documentation, recent 2025 research, GitHub implementations, and WACV 2027 submission information.

The most important finding is that **UPAR 2027 Track 2 is not simply “run an attribute classifier and submit rankings.”** The difficult part is the combination of **cross-dataset generalization + exact 40-attribute matching + retrieval ranking + hidden-domain evaluation + strict code-only execution**.

---

# UPAR Challenge 2027 — Track 2

## What you need to know before participating that is not obvious from the supplied PDF

### Current official status

The official RWS 2027 website confirms that Track 2 is specifically being extended toward **cross-dataset pedestrian retrieval**, rather than merely repeating the previous challenge. ([Visual Lab][1])

[Official RWS 2027 Challenge Page](https://vap.aau.dk/rws/challenge/?utm_source=chatgpt.com)

The supplied PDF describes the mechanics, but the external research reveals much more about **why the task is difficult, what has already been attempted, where previous approaches failed, and what kind of system is actually required**.

The supplied document itself confirms that the gallery is hidden and that your code is executed against it rather than you submitting precomputed rankings. 

---

# 1. The biggest thing to understand: this is a domain-generalization challenge

This is not ordinary person retrieval.

The training/development data and the hidden evaluation gallery are deliberately separated by domain.

The original UPAR work was created specifically because pedestrian-attribute models can perform well on one dataset but degrade substantially when attributes, camera viewpoints, illumination, image resolution, pose and occlusion change. UPAR unified PA100K, PETA, RAPv2 and Market-1501 with 40 attributes specifically to study this generalization problem. ([arXiv][2])

The 2024 challenge made this even more explicit: training/development data came from PA100K, PETA and Market1501-Attributes, while the official test domain was MEVID, a different surveillance environment. ([ChaLearn][3])

For 2027, the official wording now explicitly says:

> Track 2 extends ABPR to cross-dataset pedestrian retrieval.

([Visual Lab][1])

### Important consequence

Do **not** design the system around memorizing the visual characteristics of PA100K/PETA/Market1501.

The real target is:

**learn attribute semantics that survive a dataset/domain change.**

That changes the architecture and training strategy considerably.

---

# 2. Do not assume the 2027 hidden dataset is MEVID

This is one of the most important things I found.

The 2024 challenge used MEVID as the novel test domain. MEVID contains 158 unique people, 598 outfits, 8,092 tracklets and 33 camera views, with substantial variation in environment, viewpoint, dates and clothing. ([GitHub][4])

However, **I found no current official 2027 source that publicly identifies the hidden Track 2 test dataset as MEVID.**

The current official 2027 description only says **cross-dataset pedestrian retrieval**. ([Visual Lab][1])

Therefore:

**MEVID is useful for studying the previous challenge, but you should not build a 2027 solution assuming that the hidden gallery is MEVID.**

That would create exactly the type of domain-specific bias the challenge is designed to expose.

---

# 3. The 2024 Track 2 results reveal how difficult this problem actually is

This is probably the most useful historical information for deciding how serious your approach needs to be.

In the 2024 challenge:

* 82 participants registered overall.
* Track 2 attracted only 22 registered participants.
* During development, only **one active team** made submissions.
* That team made 8 development submissions.
* During the final test phase, only five active teams submitted.
* There were only 7 test-phase submissions.
* **No team beat the official baseline on Track 2.**

([CVF Open Access][5])

That means Track 2 historically had a very high technical barrier.

This is not a typical Kaggle-style competition where a large number of teams repeatedly tune models against a public leaderboard.

---

# 4. The challenge has historically defeated strong models

The 2024 official challenge baseline was already strong.

The baseline used:

* ConvNeXt backbone
* fully connected attribute head
* sigmoid outputs
* weighted cross-entropy
* AdamW
* weight decay
* learning-rate scheduling
* label smoothing
* dropout
* data augmentation
* exponential moving averages
* mixed precision

The authors specifically designed it for **cross-domain attribute-based person retrieval** and generalization. ([CVF Open Access][5])

So a naive:

```text
ResNet
    ↓
40 sigmoid outputs
    ↓
Hamming distance
    ↓
ranking
```

system is unlikely to exploit the full problem.

---

# 5. But the research landscape changed after the 2024 competition

This is where the external research becomes especially important.

A major post-challenge development is **CLEAR**, published first as an arXiv paper and later in *Pattern Recognition* in 2025.

CLEAR is explicitly designed for both:

1. Pedestrian Attribute Recognition
2. Attribute-Based Person Retrieval

It builds on C2T-Net and introduces language-based representations for attribute queries. ([arXiv][6])

The core idea is important:

```text
Pedestrian image
       ↓
strong visual encoder
       ↓
visual embedding
       ↓
       similarity
       ↑
attribute query
       ↓
pseudo-text / language representation
       ↓
language embedding
```

Instead of treating the query merely as a 40-dimensional binary vector, the method constructs pseudo-textual descriptions and uses pretrained language representations to bridge the query/image modality gap. ([ScienceDirect][7])

---

# 6. CLEAR is particularly relevant to this competition

CLEAR reports experiments on:

* PETA
* PA100K
* Market-1501
* RAPv2
* UPAR2024

and reports strong retrieval results, including substantial improvements on UPAR2024 over earlier approaches. ([arXiv][6])

The published method uses:

* C2T-Net visual features
* pseudo-textual attribute descriptions
* GPT-derived language embeddings
* adapters
* a shared retrieval space

([ScienceDirect][7])

This is significant because it demonstrates that **attribute retrieval can benefit from treating the query as semantic information rather than merely a binary vector.**

---

# 7. C2T-Net is another important system to study

C2T-Net was the first-ranking solution for UPAR2024 Track 1 according to its public repository. ([GitHub][8])

Its architecture combines:

* Swin Transformer
* EVA/ViT
* channel-aware self-attention
* cross-fusion

The objective is to capture both local body-region information and global pedestrian information. ([CVF Open Access][9])

The public implementation provides an unusually useful resource because it contains the training/inference setup used for the UPAR challenge. ([GitHub][8])

[C2T-Net UPAR Challenge implementation](https://github.com/caodoanh2001/upar_challenge?utm_source=chatgpt.com)

For Track 2, the important lesson is not simply "use C2T-Net."

The more important lesson is:

**a strong PAR representation can become the visual backbone of an attribute-retrieval system.**

That is exactly the direction CLEAR subsequently explored. ([ScienceDirect][7])

---

# 8. Attribute classification accuracy is not the actual objective

This distinction is critical.

You are not being evaluated primarily on:

```text
How accurately can I predict the 40 attributes?
```

You are evaluated on:

```text
Given a 40-attribute query,
how well can I rank the entire hidden gallery?
```

The supplied challenge specification defines a correct image as one whose complete 40-attribute vector exactly equals the query. 

Therefore, even if your model obtains excellent average attribute recognition, it can still produce poor retrieval.

Example:

Query:

```text
Adult
Female
Long hair
Red upper body
Black trousers
Backpack
...
```

Candidate A:

```text
39/40 attributes correct
```

Candidate B:

```text
40/40 correct
```

Candidate C:

```text
35/40 correct
```

The official relevance definition makes B the exact match. But the **mADM metric** also gives useful credit to candidates that are visually/semantically close. 

This means the ranking function itself matters enormously.

---

# 9. mADM changes how you should think about ranking

The challenge's mADM is especially interesting.

Traditional AP essentially treats:

```text
correct = 1
incorrect = 0
```

mADM additionally considers **how many attributes the candidate agrees with**.

The challenge paper explains that mADM rewards rankings where visually similar people appear before unrelated people, even if the exact-match positions are the same. ([CVF Open Access][5])

So your ranking function should ideally create this ordering:

```text
Exact match
    ↓
39/40 attributes
    ↓
38/40
    ↓
37/40
    ↓
...
    ↓
unrelated person
```

rather than:

```text
Exact match
    ↓
random candidates
    ↓
near matches
```

This makes a sophisticated continuous similarity function much more valuable than a simple binary match test.

---

# 10. The query is actually unusually structured

The challenge does not give natural-language descriptions such as:

> "Find a woman with long hair wearing a red shirt."

Instead, it gives a 40-dimensional binary vector.

The current format is effectively:

```text
q = [0,1,0,1,0,...,1]
```

with the fixed 40-attribute ordering.

The supplied document confirms the exact ordering. 

That means you have an opportunity to build multiple representations of the same query.

### Representation 1 — raw binary

```text
40-dimensional vector
```

### Representation 2 — weighted attribute vector

```text
attribute-specific importance weights
```

### Representation 3 — learned query embedding

```text
40 attributes
      ↓
embedding layer
      ↓
transformer / MLP
      ↓
query embedding
```

### Representation 4 — pseudo-language representation

For example:

```text
adult female person,
long hair,
short upper clothing,
red upper body,
black lower body,
trousers,
backpack,
...
```

Then encode it with a pretrained language model.

CLEAR provides research evidence supporting this general direction. ([arXiv][6])

---

# 11. The biggest technical problem is attribute imbalance

The 40 attributes are not equally common.

Some attributes occur frequently:

```text
adult
black clothing
trousers
```

while others are comparatively rare:

```text
hat
sun glasses
pink
purple
green
bag
etc.
```

This is a known problem in pedestrian attribute recognition.

Recent PAR research continues to address imbalance explicitly, and AttriVision specifically reports the difficulty of the UPAR setting under imbalanced attributes. ([CVF Open Access][10])

Therefore, simply optimizing unweighted BCE can make the model strong on common attributes while weak on rare attributes.

That is especially dangerous here because:

```text
one wrong attribute
```

can destroy an exact 40/40 match.

---

# 12. Attribute groups should not necessarily be treated independently

The 40 attributes are organized into semantic groups:

```text
Age
Gender
Hair
Upper-body length
Upper-body colors
Lower-body length
Lower-body colors
Lower-body type
Accessories
```

The attributes have relationships.

For example:

```text
LowerBody-Type = Skirt&Dress
```

has strong semantic interaction with:

```text
LowerBody-Type = Trousers&Shorts
```

Likewise:

```text
UpperBody-Color-Red
UpperBody-Color-Blue
UpperBody-Color-Black
...
```

represent a structured category rather than 11 completely independent concepts.

This suggests that a strong system should model:

```text
attribute-level features
        +
category-level structure
        +
global pedestrian representation
```

rather than treating the 40 outputs as unrelated binary classifiers.

---

# 13. There is a subtle issue with mutually exclusive attributes

The 40 labels are binary, but some categories are effectively mutually exclusive.

For example:

```text
Age-Young
Age-Adult
Age-Old
```

and:

```text
Hair-Length-Short
Hair-Length-Long
Hair-Length-Bald
```

and multiple clothing colors.

The UPAR dataset documentation explains that all attributes are represented as binary labels even though some categories represent mutually exclusive alternatives. ([KITopen Publications][11])

That means your loss function can potentially exploit category structure.

Instead of independently learning:

```text
P(red)
P(blue)
P(green)
...
```

you can also construct a category-aware representation.

This is a research direction rather than an official competition requirement.

---

# 14. The hidden gallery is not just a few images

The submission API gives your model the **whole gallery and all queries in one call**. 

This creates an important optimization opportunity.

You do not need to do:

```text
for query:
    for image:
        run model(image)
```

Instead:

```text
for every gallery image:
    extract visual attributes/features once

for every query:
    calculate query-gallery similarity
```

The challenge document itself points toward this approach. 

This is a major architectural distinction.

---

# 15. You should design the inference pipeline around gallery caching

A good inference architecture is conceptually:

```text
Hidden Gallery
      │
      ▼
Image preprocessing
      │
      ▼
Visual backbone
      │
      ├───────────────┐
      ▼               ▼
40 attribute probs   visual embedding
      │               │
      └───────┬───────┘
              │
              ▼
        gallery cache
              
Queries
   │
   ▼
40-D query vector
   │
   ├── weighted attribute similarity
   ├── semantic embedding similarity
   └── category-aware similarity
              │
              ▼
        fusion / ranking
              │
              ▼
          Q × G ranking
```

This is much more realistic than repeatedly executing the complete network for every query.

---

# 16. Runtime is a real engineering constraint

The environment is fixed:

```text
PyTorch 2.4.1
CUDA 12.1
cuDNN 9
Python 3.11
```

and there is no network access. 

The important implication is:

**a model that works in your development environment may fail inside Codabench.**

You must test:

* package availability
* CUDA behavior
* model loading
* checkpoint size
* memory consumption
* inference speed
* image decoding
* preprocessing
* output tensor dimensions

before submission.

---

# 17. You cannot depend on downloading anything during inference

This is more restrictive than many competitions.

You cannot have:

```python
from transformers import ...
model = load_from_huggingface(...)
```

and expect the container to download the weights.

Everything must be packaged.

The challenge explicitly states:

```text
No network access.
Ship your weights inside the zip.
```



Therefore, if you use:

* CLIP
* EVA
* ViT
* Swin
* language model
* tokenizer
* custom package

the necessary weights and code need to be available locally inside the submission.

---

# 18. This creates a major model-size tradeoff

A huge foundation model is not automatically advantageous.

You need:

```text
accuracy
+
domain generalization
+
memory
+
inference time
+
submission size
```

The competition evaluates the executable system.

Therefore:

```text
SOTA model on paper
```

is less useful than:

```text
strong model that reliably executes inside the official container
```

This is particularly relevant if you plan to use large VLMs.

---

# 19. Foundation models are allowed, but there is an important distinction

The 2027 rules permit pretrained models that were not specifically trained for pedestrian attribute recognition.

The supplied rules explicitly permit things such as ImageNet-pretrained models and foundation/VLM models not specifically trained for PAR. 

The prohibited category is external data used to **fine-tune PAR**.

That distinction matters.

Conceptually:

### Allowed

```text
ImageNet-pretrained backbone
        ↓
train on UPAR challenge data
```

### Allowed

```text
general-purpose VLM
        ↓
use pretrained weights
        ↓
fine-tune using permitted challenge data
```

### Not allowed

```text
external pedestrian dataset
        ↓
fine-tune PAR model
        ↓
UPAR challenge
```

The 2027 terms also prohibit use of test images for training, pseudo-labeling, manual annotation or hyperparameter selection. 

---

# 20. This means external-data leakage is a serious danger

Be extremely careful with datasets such as:

* Market-1501
* PETA
* PA100K
* RAPv2
* MEVID
* other pedestrian datasets

The question is not merely:

> "Is this dataset publicly available?"

The relevant question is:

> "Am I using it in a way allowed by the 2027 rules?"

For 2027, you should treat the official challenge training data as the authoritative training source unless a particular pretrained model falls under the explicitly allowed pretrained-model category.

---

# 21. Do not accidentally use hidden-test-derived information

The rules prohibit:

* test-image training
* pseudo-labeling
* manual annotation
* hyperparameter selection using test images
* attempts to extract test images

and violations can cause disqualification. 

This matters because domain-generalization experiments often tempt researchers to inspect the target domain.

For this challenge, that is exactly what you cannot do.

---

# 22. The leaderboard itself can encourage overfitting

The official document says the **latest valid submission** is the one shown on the leaderboard, rather than your best historical submission. 

This is important.

You should maintain your own experiment table:

```text
Experiment ID
Checkpoint
Architecture
Training seed
Loss
Augmentation
Validation mAP
Validation Rank-1
Validation mADM
Submission score
Notes
```

Do not rely on the leaderboard as your experiment database.

---

# 23. Submission limits are deliberately intended to prevent brute-force leaderboard tuning

The previous challenge explicitly limited final-phase submissions.

In 2024, the final test phase allowed only 3 submissions per participant, specifically to prevent trial-and-error optimization against the hidden test set. ([CVF Open Access][5])

The 2027 rules say submission limits are shown for the relevant phase. 

So your development methodology should be:

```text
local validation
      ↓
ablation
      ↓
cross-domain testing
      ↓
robustness testing
      ↓
only then leaderboard submission
```

not:

```text
submit
change one thing
submit
change one thing
submit
```

---

# 24. Your local validation must simulate domain shift

This is one of the strongest practical conclusions from the research.

The original UPAR dataset contains multiple domains and provides protocols for cross-validation and leave-one-out evaluation. ([GitHub][12])

You should therefore build validation such as:

```text
Train:
PA100K + PETA

Validate:
Market1501
```

then:

```text
Train:
PETA + Market1501

Validate:
PA100K
```

and:

```text
Train:
PA100K + Market1501

Validate:
PETA
```

The exact official 2027 split must be followed for competition compliance, but these experiments are extremely useful for understanding whether your model is actually learning transferable attribute semantics.

---

# 25. The biggest weakness of ordinary PAR models

A standard PAR model learns:

```text
image → attribute labels
```

But Track 2 requires:

```text
attribute description → image ranking
```

These are related but not identical problems.

CLEAR's work is particularly relevant because it explicitly bridges the two modalities. ([arXiv][6])

This is why I would treat the following as separate components:

```text
PAR representation learning
+
query representation learning
+
cross-modal alignment
+
retrieval ranking
```

rather than one classification head.

---

# 26. A particularly promising architecture family

Based on the published research after UPAR2024, a technically reasonable architecture family is:

```text
                    ┌────────────────────┐
                    │  Pedestrian Image  │
                    └─────────┬──────────┘
                              │
                              ▼
                    Visual Transformer
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
             Attribute Head       Visual Embedding
                    │                   │
                    │                   │
                    ▼                   │
              40 attributes             │
                                        │
                                        ▼
                               Retrieval Projection
                                        │
                                        │
Query 40-D vector                       │
       │                                │
       ▼                                │
Attribute Embedding                     │
       │                                │
       ▼                                │
Semantic / Text Encoder ────────────────┘
                │
                ▼
       Cross-modal similarity
                │
                ▼
       Attribute-aware fusion
                │
                ▼
             Ranking
```

This is broadly consistent with the direction demonstrated by CLEAR, rather than merely copying a 2024 baseline. ([ScienceDirect][7])

---

# 27. But do not blindly copy CLEAR

There is an important reason.

CLEAR was evaluated on UPAR2024, whose official test domain was MEVID. ([ScienceDirect][7])

UPAR2027 is described as a new cross-dataset Track 2 setting. ([Visual Lab][1])

Therefore, CLEAR should be treated as:

```text
important prior research
```

not:

```text
guaranteed winning architecture
```

The hidden 2027 distribution could expose different weaknesses.

---

# 28. Recent research confirms that language/vision fusion is becoming important

AttriVision uses CLIP for pedestrian attribute recognition and explicitly evaluates generalization on UPAR2024. ([CVF Open Access][10])

CLEAR goes further and directly applies language representations to attribute retrieval. ([ScienceDirect][7])

This gives you a useful research progression:

```text
UPAR baseline
      ↓
C2T-Net
      ↓
CLIP-based PAR
      ↓
CLEAR / language-assisted retrieval
      ↓
your 2027 cross-dataset method
```

---

# 29. Do not overlook simple attribute-space retrieval

Even if you use a sophisticated multimodal model, retain a direct attribute-space similarity.

For a predicted probability vector:

```text
p ∈ R^40
```

and query:

```text
q ∈ {0,1}^40
```

you can calculate:

```text
weighted L1
weighted L2
binary cross-entropy distance
Hamming-like distance
cosine similarity
```

and combine it with embedding similarity.

For example conceptually:

```text
Score =
    α × attribute_similarity
  + β × embedding_similarity
  + γ × category_similarity
```

This is particularly attractive because the official evaluation itself is based on the 40 attributes.

---

# 30. Attribute-specific weighting is potentially important

A generic distance assumes:

```text
every attribute = equally reliable
```

That is unlikely to be optimal.

For example:

```text
Gender
```

may be relatively easier than:

```text
Hair length
```

under severe occlusion.

Similarly:

```text
shirt color
```

may be affected by illumination.

And:

```text
glasses
```

may occupy very few pixels.

A learned or validation-derived weighting scheme can therefore be investigated.

The critical constraint is that weights must be determined without using the hidden test set.

---

# 31. Image quality matters disproportionately

The gallery consists of pedestrian crops with varying:

* size
* aspect ratio
* resolution
* occlusion
* viewpoint

The supplied specification explicitly states that gallery images vary in size and aspect ratio. 

Therefore preprocessing should not assume:

```text
person = clean 224×224 object
```

You should test:

* aspect-ratio-preserving resize
* padding
* crop strategies
* horizontal flip
* multi-scale inference
* image normalization
* low-resolution robustness

---

# 32. Do not use aggressive cropping blindly

This is a subtle issue.

Attributes include:

```text
hair
hat
glasses
backpack
bag
upper clothing
lower clothing
```

A crop that improves global body recognition can remove:

```text
head
feet
bag
backpack
```

Therefore, a global pedestrian crop may be more appropriate than object-detection-style center cropping.

Potentially useful:

```text
global crop
+
upper-body crop
+
lower-body crop
+
head/face region
```

with feature fusion.

---

# 33. Multi-region representations are particularly logical for this task

Because the attributes are spatially distributed:

```text
head
  ├── hair
  ├── glasses
  └── hat

upper body
  ├── length
  └── color

lower body
  ├── length
  ├── color
  └── type

global
  ├── age
  ├── gender
  └── accessories
```

A model that explicitly learns region-aware features could have a strong advantage over a single global feature.

This also aligns conceptually with why transformer-based pedestrian models such as C2T-Net use both local and global information. ([CVF Open Access][9])

---

# 34. The query itself contains information about which visual regions matter

For example:

```text
Query:
Female + long hair + red upper body + backpack
```

does not require equal attention to every body region.

You could condition image-query interaction on the query:

```text
query → attribute importance
          ↓
image feature attention
          ↓
query-conditioned representation
```

This is more powerful than calculating one generic embedding for every person.

---

# 35. Another important opportunity: query-conditioned region weighting

Conceptually:

```text
query:
hair = long
upper color = red
backpack = yes

             ↓

head feature       → high weight
upper-body feature → high weight
backpack region    → high weight
lower body         → lower weight
```

This can potentially improve ranking when the target attributes are sparse.

---

# 36. Do not forget the exact-match problem

Suppose:

```text
Query = 40 attributes
```

and your model predicts:

```text
39 correct
1 wrong
```

The candidate is not an exact match.

Therefore, the system needs high confidence in **rare and difficult attributes**, not merely good average accuracy.

This is why class-balanced losses, category-aware losses and calibrated probabilities deserve serious experimentation.

---

# 37. Calibration may matter more than you initially expect

Suppose two gallery images have:

```text
Image A:
attribute agreement = 37/40
but uncertain predictions

Image B:
attribute agreement = 36/40
but very high-confidence predictions
```

A raw distance may rank them incorrectly.

Probability calibration or uncertainty-aware scoring could potentially improve ranking.

This is a research direction worth testing locally, particularly because the final ranking is based on continuous similarity values.

---

# 38. Tie handling is a real implementation issue

The challenge explicitly says that equal similarity/distance values are resolved according to gallery order. 

Therefore:

```text
score = integer number of matching attributes
```

is dangerous.

For example:

```text
38
38
38
38
38
```

creates huge tie groups.

The final ordering then depends on gallery order.

Instead use continuous scores such as:

```text
0.9283
0.9271
0.9259
...
```

This is a small implementation detail with potentially significant leaderboard consequences.

---

# 39. The output format gives you three choices

Your `rank_gallery()` can return:

### Distances

```text
smaller = better
```

### Similarities

```text
larger = better
```

### Explicit ranking

```text
gallery indices
best → worst
```

The challenge validates shape and ranking correctness. Invalid matrices, NaN/Inf, incorrect shapes or invalid permutations can cause submission failure. 

For development, I would use similarity matrices because they make score fusion and debugging easier.

---

# 40. Your submission should be engineered like a production package

A robust submission should roughly look like:

```text
submission/
│
├── run.py
├── metadata.yaml
│
├── models/
│   ├── backbone.py
│   ├── retrieval.py
│   └── ...
│
├── weights/
│   └── model.pth
│
├── configs/
│   └── config.yaml
│
├── tokenizer/
│   └── ...
│
└── utils/
    └── ...
```

The supplied rules require `run.py` at the archive root and `metadata.yaml` copied unchanged from the example submission. 

---

# 41. You need a local replica of the Codabench execution environment

Do not wait until the final day.

The official environment is:

```text
pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime
Python 3.11
Torch 2.4.1
Torchvision
NumPy
Pillow
```



Build your local Docker test around that environment.

Then test:

```text
unzip
↓
import run.py
↓
load_model()
↓
rank_gallery(sample)
↓
validate output
↓
measure runtime
↓
measure RAM/VRAM
```

---

# 42. The paper opportunity is significant

The RWS workshop is not merely a competition leaderboard.

The official workshop CFP says accepted challenge papers will be included in the WACV Workshop proceedings and published through CVF/IEEE. ([Visual Lab][13])

Challenge participants have a later paper deadline:

**October 23, 2026, 11:59 PM AoE.** ([Visual Lab][13])

The challenge itself closes:

**October 21, 2026 AoE.**

So you effectively have:

```text
Oct 21
challenge submission

Oct 23
challenge paper

Oct 30
code verification
```

The supplied document confirms the same sequence. 

---

# 43. Convert the deadlines into Indian Standard Time

This is easy to overlook.

"Anywhere on Earth" is UTC−12.

Therefore approximately:

| Deadline  |           AoE |                   India |
| --------- | ------------: | ----------------------: |
| Challenge | Oct 21, 23:59 | **Oct 22, 5:29 PM IST** |
| Paper     | Oct 23, 23:59 | **Oct 24, 5:29 PM IST** |
| Code      | Oct 30, 23:59 | **Oct 31, 5:29 PM IST** |

Do not plan around the converted final minute. Treat the previous Indian calendar day as your internal deadline.

As of **September 30, 2026**, there are only about **21 days until the official challenge deadline**.

---

# 44. There is an important WACV/RWS publication detail

The RWS workshop page says accepted papers will be included in the WACV Workshop proceedings and published by CVF/IEEE. ([Visual Lab][13])

The general WACV 2027 author guide specifies the WACV format as:

* maximum 8 pages excluding references
* double-blind
* PDF
* maximum 50 MB
* supplementary material up to 200 MB

([WACV][14])

The RWS challenge deadline is specifically later than the normal workshop paper deadline because challenge participants get the special October 23 deadline. ([Visual Lab][13])

---

# 45. In-person participation matters

WACV 2027 is scheduled at Disney Springs, Florida, January 4–8, 2027, with workshops January 4–5. ([WACV][15])

The WACV workshop framework specifies that workshops are in person and remote presentations are not accommodated. ([WACV][16])

For the RWS workshop, this matters primarily if your challenge paper is accepted and your team is expected to present.

---

# 46. The competition has an unusually strong reproducibility requirement

This is not just:

```text
submit model
```

The organizers can require:

```text
trained models
training code
parameters
method description
```

for verification.

The supplied terms explicitly say that teams that do not comply can be removed from the final ranking. 

Therefore maintain:

```text
Git repository
+
training configuration
+
dataset preprocessing
+
checkpoint
+
exact commit
+
random seeds
+
environment file
+
experiment logs
```

from the beginning.

---

# 47. Keep the training pipeline reproducible

Your experiment record should contain:

```text
Git commit
Dataset version
Backbone
Pretraining source
Input size
Batch size
Optimizer
Learning rate
Scheduler
Loss
Class weights
Augmentation
Seed
Epochs
Checkpoint
Validation metrics
```

This is particularly important because code verification happens after the leaderboard phase.

---

# 48. The licensing situation needs attention

The UPAR annotations are released under CC BY-NC-SA 3.0, while the underlying datasets retain their own licenses. 

The 2024 challenge repository documents, for example:

* PA100K — CC BY 4.0
* PETA — research-only/non-commercial
* Market1501 — no license listed in that repository

([GitHub][17])

Therefore, do not assume that because the challenge is hosted publicly, all underlying images are unrestricted.

---

# 49. One major strategic mistake would be treating this as ordinary person Re-ID

Traditional person Re-ID generally asks:

```text
image of person A
      ↓
find same identity
```

This challenge asks:

```text
attribute description
      ↓
find people matching those attributes
```

The query does not provide an identity image.

That changes the representation problem fundamentally.

You need:

```text
semantic appearance matching
```

rather than only:

```text
identity matching
```

CLEAR's work explicitly frames the relationship between PAR and attribute-based retrieval this way. ([arXiv][6])

---

# 50. The most important research gap for 2027

After examining the previous challenge, recent methods and the new 2027 description, the central research problem can be summarized as:

```text
How can a model learn a domain-invariant representation of
fine-grained pedestrian attributes such that a binary semantic
query can retrieve the correct people in an unseen surveillance domain?
```

That is much closer to the real research problem than:

> "Build a person retrieval model."

---

# 51. What I would consider the major technical components

A serious Track 2 system should be investigated around these components:

### A. Strong visual backbone

Potential families:

```text
ConvNeXt
Swin Transformer
ViT / EVA
C2T-Net
CLIP-style visual encoder
```

### B. Multi-label attribute head

```text
40 attributes
```

### C. Attribute-aware feature representation

```text
attribute tokens
+
category tokens
```

### D. Query encoder

```text
40-D binary query
        ↓
learned embedding
```

### E. Optional language branch

```text
40-D query
   ↓
pseudo-description
   ↓
language encoder
```

### F. Cross-modal alignment

```text
image ↔ query
```

### G. Attribute-space similarity

```text
predicted attributes ↔ query
```

### H. Embedding-space similarity

```text
visual embedding ↔ query embedding
```

### I. Score fusion

```text
S = αS_attribute + βS_embedding + γS_category
```

### J. Efficient gallery inference

```text
encode gallery once
```

---

# 52. What should not be your first approach

Avoid starting with:

```text
YOLO
+
person detection
+
ResNet
+
40 independent classifiers
+
Hamming distance
```

The gallery is already pedestrian crops, so detection adds little value unless your local experiments demonstrate a specific benefit.

Similarly, do not spend the majority of the remaining time building a giant general-purpose VLM before establishing a strong UPAR baseline.

The highest-value progression is:

```text
official baseline
       ↓
strong visual PAR
       ↓
attribute-space retrieval
       ↓
query-conditioned retrieval
       ↓
language-assisted retrieval
       ↓
score fusion
       ↓
domain-generalization experiments
```

---

# 53. Your first benchmark should be brutally simple

Before experimenting with sophisticated architecture, establish:

```text
Model:
ConvNeXt / strong pretrained backbone

Output:
40 attribute probabilities

Ranking:
weighted BCE / attribute distance
```

Then record:

```text
mADM
mAP
Rank-1
Rank-5
Rank-10
mINP
```

Only after that should you add:

```text
CLIP
language
cross-attention
query conditioning
multi-region features
ensemble
```

Otherwise you will not know which component actually improves retrieval.

---

# 54. Ensemble methods are particularly interesting here

You can potentially combine:

```text
Model A:
attribute probability distance

Model B:
visual embedding similarity

Model C:
language-query similarity

Model D:
region-aware similarity
```

Then:

```text
final_score =
    w1 * S_attribute
  + w2 * S_visual
  + w3 * S_language
  + w4 * S_region
```

The weights must be determined using permitted development/validation data, not the hidden test gallery.

This approach also reduces dependence on a single representation.

---

# 55. But ensemble size must be balanced against runtime

Because all gallery images must be processed inside the evaluation environment, a four-model ensemble could become computationally expensive.

A practical architecture might instead use:

```text
one shared backbone
        ↓
multiple lightweight heads
```

rather than:

```text
four independent giant backbones
```

This is particularly important because `rank_gallery()` is a single evaluation call and the official environment imposes a wall-clock limit. 

---

# 56. Your evaluation code should reproduce the official metrics locally

Do not only track:

```text
accuracy
F1
mA
```

Track the actual retrieval metrics:

```text
mADM
mAP
Rank-1
Rank-5
Rank-10
mINP
```

The challenge explicitly uses these retrieval metrics. 

This matters because a change that improves classification F1 may decrease retrieval performance.

---

# 57. The most informative metric combination

Think of them as measuring different failure modes:

| Metric  | What it tells you                             |
| ------- | --------------------------------------------- |
| Rank-1  | Can you put an exact match first?             |
| Rank-5  | Can you retrieve one very early?              |
| Rank-10 | Does retrieval remain useful slightly deeper? |
| mAP     | Are all exact matches ranked well?            |
| mINP    | Can you retrieve the hardest exact match?     |
| mADM    | Are near-matches also sensibly ranked?        |

The challenge's mADM is particularly useful because it evaluates the quality of the ranking even when exact matches are sparse. 

---

# 58. A hidden danger: optimizing only Rank-1

A model can improve Rank-1 while becoming worse at retrieving the remaining correct gallery images.

mAP and mINP expose this.

For example:

```text
Model A:
Rank-1 = high
but second/third/etc. exact matches are scattered

Model B:
Rank-1 = slightly lower
but all correct matches are tightly grouped
```

The official evaluation considers multiple retrieval properties.

Therefore, optimize the whole ranking, not one number.

---

# 59. Another hidden danger: visually similar false positives

Suppose the query is:

```text
adult
female
long hair
red upper body
black trousers
backpack
```

A candidate with:

```text
adult
female
long hair
red upper body
black trousers
NO backpack
```

is visually extremely close.

That candidate can be useful for mADM but is not an exact match.

This is exactly why a continuous similarity representation is valuable.

---

# 60. The strongest research direction emerging from all of this

The external literature suggests the problem is moving toward:

```text
PAR
+
vision-language representation
+
attribute-aware retrieval
+
domain generalization
```

rather than conventional classification.

CLEAR is particularly important because it demonstrates a unified PAR + retrieval architecture using a pretrained language model and adapters. ([arXiv][6])

---

# 61. Practical competition strategy for the remaining time

Given the current date, a realistic sequence is:

### Phase 1 — immediately

Implement the official starter kit and reproduce:

```text
data loader
query loader
gallery loader
submission API
local evaluation
```

### Phase 2

Build:

```text
strong pretrained backbone
+
40-attribute head
```

### Phase 3

Implement:

```text
query ↔ predicted attribute distance
```

### Phase 4

Introduce:

```text
query-conditioned embedding
```

### Phase 5

Experiment with:

```text
CLIP / language embeddings
```

### Phase 6

Test:

```text
cross-domain validation
```

### Phase 7

Add:

```text
score fusion
```

### Phase 8

Optimize:

```text
runtime
memory
zip size
```

### Phase 9

Run the official container locally.

### Phase 10

Submit only stable candidates.

---

# 62. The research papers you should actually study

The most relevant reading stack from the research I found is:

### 1. UPAR — foundational paper

[UPAR: Unified Pedestrian Attribute Recognition and Person Retrieval](https://openaccess.thecvf.com/content/WACV2023/html/Specker_UPAR_Unified_Pedestrian_Attribute_Recognition_and_Person_Retrieval_WACV2023_paper.html?utm_source=chatgpt.com)

Understand:

* 40 attributes
* dataset harmonization
* domain generalization
* original retrieval formulation
* baseline architecture

([CVF Open Access][18])

### 2. UPAR Challenge 2024

[UPAR Challenge 2024 paper](https://openaccess.thecvf.com/content/WACV2024W/RWS/html/Cormier_UPAR_Challenge_2024_Pedestrian_Attribute_Recognition_and_Attribute-Based_Person_Retrieval_WACVW_2024_paper.html?utm_source=chatgpt.com)

Understand:

* actual challenge difficulty
* MEVID domain shift
* baseline
* failed Track 2 approaches
* submission behavior

([CVF Open Access][19])

### 3. C2T-Net

[C2T-Net paper](https://openaccess.thecvf.com/content/WACV2024W/RWS/html/Bui_C2T-Net_Channel-Aware_Cross-Fused_Transformer-Style_Networks_for_Pedestrian_Attribute_Recognition_WACVW_2024_paper.html?utm_source=chatgpt.com)

Understand:

* local/global feature modeling
* Swin + EVA
* cross-fusion
* strong PAR representation

([CVF Open Access][9])

### 4. CLEAR

[CLEAR arXiv paper](https://arxiv.org/abs/2403.06119?utm_source=chatgpt.com)

This is probably the most relevant post-2024 paper for Track 2.

Understand:

* pseudo-textual queries
* GPT embeddings
* adapters
* cross-modal retrieval
* unified PAR/AR

([arXiv][6])

### 5. AttriVision

[AttriVision paper](https://openaccess.thecvf.com/content/WACV2025W/V3SC/html/Sedeh_AttriVision_Advancing_Generalization_in_Pedestrian_Attribute_Recognition_using_CLIP_WACVW_2025_paper.html?utm_source=chatgpt.com)

Understand:

* CLIP-based PAR
* generalization
* attribute imbalance
* UPAR2024 behavior

([CVF Open Access][20])

---

# 63. Useful code repositories

The official UPAR challenge repository:

[UPAR Challenge repository](https://github.com/speckean/upar_challenge?utm_source=chatgpt.com)

The original UPAR dataset repository:

[UPAR dataset repository](https://github.com/speckean/upar_dataset?utm_source=chatgpt.com)

The C2T-Net implementation:

[C2T-Net implementation](https://github.com/caodoanh2001/upar_challenge?utm_source=chatgpt.com)

MEVID:

[MEVID official repository](https://github.com/Kitware/MEVID?utm_source=chatgpt.com)

The MEVID repository is useful for understanding the kind of environmental/domain variation used in the previous challenge, but it should **not** be assumed to be the hidden 2027 test set. ([GitHub][4])

---

# 64. The most important conclusions

After combining the current 2027 information with the previous challenge results and post-2024 research, the situation is:

### What the competition really tests

```text
Fine-grained attribute understanding
             +
Cross-dataset generalization
             +
Attribute-query representation
             +
Image-query matching
             +
Large-gallery ranking
             +
Efficient code execution
```

### What historically failed

```text
ordinary PAR
+
simple attribute distance
```

was not enough to beat the 2024 baseline; in fact, no Track 2 team surpassed the baseline in that edition. ([CVF Open Access][5])

### What the newer research indicates

```text
strong visual backbone
+
semantic attribute representation
+
language/query representation
+
cross-modal alignment
```

is a more promising direction, with CLEAR providing direct evidence on UPAR2024. ([ScienceDirect][7])

### What remains unknown for 2027

The **exact hidden test domain/dataset composition and gallery statistics are not publicly disclosed in the sources I found**. The official 2027 description only establishes the cross-dataset retrieval objective. ([Visual Lab][1])

That uncertainty is deliberate and should shape the solution: **optimize for domain invariance rather than for one known target dataset.**

---

## Bottom line

The most important information not obvious from the supplied competition document is this:

**UPAR 2027 Track 2 should be treated as a domain-generalized, multimodal attribute-retrieval research problem, not as a conventional 40-label classification competition.**

The strongest evidence from the previous edition is unusually clear: **Track 2 was difficult enough that no 2024 participant surpassed the official baseline**, despite strong PAR systems and only a small number of active teams. ([CVF Open Access][5])

Since then, research such as **C2T-Net, AttriVision and especially CLEAR** has changed the technical landscape by showing that stronger visual representations and language-assisted query representations can substantially improve attribute-based retrieval. ([CVF Open Access][9])

The 2027 challenge now explicitly pushes Track 2 toward **cross-dataset retrieval**, while keeping the evaluation gallery hidden. ([Visual Lab][1])

Therefore the technically relevant target is:

```text
DOMAIN-INVARIANT PEDESTRIAN REPRESENTATION
                    +
ATTRIBUTE-AWARE QUERY ENCODING
                    +
VISION-LANGUAGE ALIGNMENT
                    +
EXACT + SOFT ATTRIBUTE MATCHING
                    +
EFFICIENT FULL-GALLERY RANKING
                    +
STRICT REPRODUCIBLE CODE SUBMISSION
```

The supplied PDF confirms the official mechanics, but the external evidence shows that **this underlying research problem—not the submission ZIP itself—is what will determine whether a solution is merely functional or genuinely competitive.** 

[1]: https://vap.aau.dk/rws/challenge/ "Challenge – Real World Surveillance: Applications and Challenges, 7th"
[2]: https://arxiv.org/abs/2209.02522?utm_source=chatgpt.com "UPAR: Unified Pedestrian Attribute Recognition and Person Retrieval"
[3]: https://chalearnlap.cvc.uab.cat/dataset/45/description/?utm_source=chatgpt.com "ChaLearn"
[4]: https://github.com/Kitware/MEVID?utm_source=chatgpt.com "GitHub - Kitware/MEVID: Official Code for WACV 2023 paper MEVID: Multi-view Extended Videos with Identities for Video Person Re-Identification · GitHub"
[5]: https://openaccess.thecvf.com/content/WACV2024W/RWS/papers/Cormier_UPAR_Challenge_2024_Pedestrian_Attribute_Recognition_and_Attribute-Based_Person_Retrieval_WACVW_2024_paper.pdf?utm_source=chatgpt.com "UPAR Challenge 2024: Pedestrian Attribute Recognition and Attribute-Based Person Retrieval - Dataset, Design, and Results"
[6]: https://arxiv.org/abs/2403.06119?utm_source=chatgpt.com "CLEAR: Cross-Transformers with Pre-trained Language Model is All you need for Person Attribute Recognition and Retrieval"
[7]: https://www.sciencedirect.com/science/article/pii/S0031320325001463?utm_source=chatgpt.com "CLEAR: Cross-Transformers With Pre-Trained Language Model for Person Attribute Recognition and Retrieval - ScienceDirect"
[8]: https://github.com/caodoanh2001/upar_challenge?utm_source=chatgpt.com "GitHub - caodoanh2001/upar_challenge: Official repository of the first-ranking solution for the UPAR2024 Challenge - Track 1. · GitHub"
[9]: https://openaccess.thecvf.com/content/WACV2024W/RWS/html/Bui_C2T-Net_Channel-Aware_Cross-Fused_Transformer-Style_Networks_for_Pedestrian_Attribute_Recognition_WACVW_2024_paper.html?utm_source=chatgpt.com "WACV 2024 Open Access Repository"
[10]: https://openaccess.thecvf.com/content/WACV2025W/V3SC/papers/Sedeh_AttriVision_Advancing_Generalization_in_Pedestrian_Attribute_Recognition_using_CLIP_WACVW_2025_paper.pdf?utm_source=chatgpt.com "AttriVision: Advancing Generalization in Pedestrian Attribute Recognition using CLIP"
[11]: https://publikationen.bibliothek.kit.edu/1000191274/176817090?utm_source=chatgpt.com "Privacy Motivated Models for Deep Learning-based Human Action Recognition in Intelligent Video Surveillance"
[12]: https://github.com/speckean/upar_dataset?utm_source=chatgpt.com "GitHub - speckean/upar_dataset: Official repository of the UPAR dataset for pedestrian attribute recognition and attribute-based person retrieval · GitHub"
[13]: https://vap.aau.dk/rws/call-for-papers/ "Call for Papers – Real World Surveillance: Applications and Challenges, 7th"
[14]: https://wacv.thecvf.com/Conferences/2027/AuthorGuides?utm_source=chatgpt.com "WACV 2027 Author Guides"
[15]: https://wacv.thecvf.com/Conferences/2027/Dates?utm_source=chatgpt.com "2027 Dates and Deadlines"
[16]: https://wacv.thecvf.com/Conferences/2027/CallForWorkshops?utm_source=chatgpt.com "2027 Workshop Information"
[17]: https://github.com/speckean/upar_challenge/blob/main/README.md "upar_challenge/README.md at main · speckean/upar_challenge · GitHub"
[18]: https://openaccess.thecvf.com/content/WACV2023/html/Specker_UPAR_Unified_Pedestrian_Attribute_Recognition_and_Person_Retrieval_WACV_2023_paper.html?utm_source=chatgpt.com "WACV 2023 Open Access Repository"
[19]: https://openaccess.thecvf.com/content/WACV2024W/RWS/html/Cormier_UPAR_Challenge_2024_Pedestrian_Attribute_Recognition_and_Attribute-Based_Person_Retrieval_WACVW_2024_paper.html?utm_source=chatgpt.com "WACV 2024 Open Access Repository"
[20]: https://openaccess.thecvf.com/content/WACV2025W/V3SC/html/Sedeh_AttriVision_Advancing_Generalization_in_Pedestrian_Attribute_Recognition_using_CLIP_WACVW_2025_paper.html?utm_source=chatgpt.com "WACV 2025 Open Access Repository"
