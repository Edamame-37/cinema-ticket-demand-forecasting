# Cinema Ticket Demand Forecasting

## Multi-Horizon Theater-Level Box Office Prediction via Dynamic Decay Ratio Modeling and Constrained Gradient Boosting Ensembles

---

## 1. Executive Summary

Accurate forecasting of theatrical movie ticket demand at the individual cinema level represents a critical operational challenge for entertainment distributors and theater exhibition chains. Box office dynamics are characterized by high localized variance, extreme capacity disparities between metropolitan multiplexes and regional single-screen venues, and steep exponential attendance decay following opening weekend debuts.

This project delivers an end-to-end, leak-free machine learning framework designed to predict daily attendance for specific movie-theater combinations across Days 4 through 10 of their theatrical run, conditioning exclusively on the first three days of opening performance (Days 1 to 3) alongside multi-modal contextual metadata. By formulating the target as a normalized fractional decay ratio ($y / S_p$), enforcing physics-guided monotonic constraints, and ensembling heterogeneous gradient boosting algorithms via Sequential Least Squares Programming (SLSQP), the system achieves state-of-the-art out-of-sample accuracy evaluated on Mean Absolute Scaled Error (MASE).

---

## 2. Problem Statement & Mathematical Formulation

### 2.1 The Forecasting Challenge
Given a set of historical screening transactions for arbitrary movie $m$ exhibited at cinema $c$, let $y_{m,c,t}$ denote the total tickets sold on day index $t \in \{1, 2, \dots, 10\}$.
* **Observed Context (Days 1–3)**: Opening weekend performance metrics, initial attendance volumes $D_1, D_2, D_3$, and historical occupancy rates.
* **Forecast Horizon (Days 4–10)**: Sequential daily ticket volumes across the subsequent week, capturing both weekday troughs and second-weekend retention.

### 2.2 Scale Invariance via Decay Ratio Modeling
Direct regression on raw ticket counts introduces severe scale disparity: large metropolitan flagship cinemas sell tens of thousands of tickets, whereas smaller provincial theaters sell dozens. Standard regression losses (RMSE/MAE) disproportionately prioritize high-capacity theaters while failing on low-capacity venues.

To ensure scale invariance, we define the cinema-specific baseline capacity scale $S_p$:

$$S_p = \max\left(\frac{1}{3}\sum_{t=1}^3 y_{m,c,t}, \ 1\right)$$

The regression objective is reformulated from raw tickets to the fractional retention ratio:

$$r_{m,c,t} = \frac{y_{m,c,t}}{S_p}$$

Predicted ticket volume is subsequently recovered via analytical re-scaling:

$$\hat{y}_{m,c,t} = \max\left(\text{round}\left(\hat{r}_{m,c,t} \cdot S_p\right), \ 0\right)$$

### 2.3 Evaluation Metric Alignment
The model performance is evaluated using Mean Absolute Scaled Error (MASE):

$$\text{MASE} = \frac{1}{N} \sum_{i=1}^N \frac{|y_i - \hat{y}_i|}{S_{p,i}} = \frac{1}{N} \sum_{i=1}^N |r_i - \hat{r}_i|$$

By training decision tree regressors with an unweighted Mean Absolute Error (L1 Loss) directly on the target ratio $r$, the optimization loss aligns exactly with the global MASE evaluation criterion.

---

## 3. Data Architecture & Pipeline

The pipeline ingests unjoined, cleaned relational datasets structured as follows:

| Dataset File | Volume | Description |
| :--- | :--- | :--- |
| `train.csv` | 138,959 records | Historical screening transactions covering multi-theater runs. |
| `test.csv` | 72,611 records | Target evaluation dataset requiring Day 4 to Day 10 ticket predictions. |
| `test_history.csv` | 32,323 records | Official historical opening window (Days 1 to 3) for the test set. |
| `movies.csv` | 397 titles | Film attributes: runtime, genre, director popularity, star power score. |
| `movies_tmdb_ratings.csv` | 397 titles | Audience reception metrics: TMDB rating, positive/negative review ratios, review counts. |
| `ticket_prices.csv` | City-level tiers | Municipal admission price matrices across Weekday, Friday, and Weekend schedules. |
| `weather_data.csv` | City-day level | Meteorological logs: precipitation volume (mm) and heavy rain indicators. |
| `holidays.csv` | Calendar dates | National holiday and regional calendar classifications. |

### Key Preprocessing Protocols
1. **Title Normalization**: Standardizes title strings by removing irregular theatrical format suffixes (e.g., `(IMAX 2D)`, `(3D)`, `(4DX)`), establishing complete, zero-loss linkage between screening logs and auxiliary metadata tables.
2. **Strict Time-Series Causality**: Restricts training observation windows strictly to Days 4 through 10. All future-dependent operational variables (such as subsequent screen allocations or future occupancy rates) are purged to prevent temporal leakage.
3. **Word-of-Mouth Signal Extraction**: Derives net sentiment indices ($\text{Positive Ratio} - \text{Negative Ratio}$), confidence-weighted rating metrics, and decay-interaction terms to capture box office longevity.

---

## 4. Modeling Methodology

```
+-------------------------------------------------------------+
|               Data Ingestion & Normalization                |
|  - Title Regex Normalization & 100% Relational Metadata Join |
|  - Opening Profile Extraction (S_p, D3/D1 Ratio, Capacity)  |
+-------------------------------------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|                  Feature Matrix Assembly                    |
|  - Calendar Dynamics (Day of Week, Weekend, Holidays)       |
|  - Sentiment Signals (TMDB Rating, Net WOM Sentiment)       |
|  - Meteorological Penalty (Precipitation x Weekend)         |
|  - Relative Temporal Position (Days Elapsed Since Release)  |
+-------------------------------------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|            Strict Cross-Validation Architecture             |
|   5-Fold GroupKFold Partitioned by Unique Movie Titles      |
+-------------------------------------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|                Constrained Gradient Boosting                |
|  - LightGBM Regressor (L1 Objective, Cat-Smooth Regularizer)|
|  - XGBoost Regressor (Monotone Decreasing Lifespan Decay)   |
+-------------------------------------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|              SLSQP Meta-Optimization & Blending             |
|  - Sequential Least Squares Programming Weight Calibration  |
|  - Convex Multi-Model Variance Reduction Blending           |
+-------------------------------------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|                  Final Demand Inference                     |
|      Scale Multiplication: Total Tickets = Ratio x S_p      |
+-------------------------------------------------------------+
```

### 4.1 Domain-Guided Monotonic Constraints
Theatrical attendance inevitably decays over the lifecycle of a film. To prevent gradient boosted trees from learning spurious mid-week surges, monotonic constraints are imposed on critical feature vectors in XGBoost:
* `day_since_release`: Enforced negative monotonic direction ($-1$), guaranteeing that ceteris paribus, forecasted attendance monotonically decreases as the theatrical run progresses.
* `net_sentiment` & `tmdb_rating`: Enforced positive monotonic direction ($+1$), ensuring superior audience reception sustains retention curves.

### 4.2 Leakage-Free Validation Strategy
Cross-validation is structured using a 5-Fold `GroupKFold` partitioning scheme grouped by `movie_title`. This mandates that all validation folds evaluate models strictly on unseen film releases, perfectly emulating out-of-sample forward deployment.

### 4.3 Meta-Optimization via SLSQP
Out-of-fold predictions from constituent models are synthesized using Sequential Least Squares Programming (SLSQP) constrained optimization:

$$\min_{\mathbf{w}} \ \frac{1}{N} \sum_{i=1}^N \frac{\left|y_i - \left(\sum_{k} w_k \hat{r}_{k,i}\right) \cdot S_{p,i}\right|}{S_{p,i}} \quad \text{s.t.} \quad \sum_k w_k = 1, \quad w_k \ge 0$$

Constituent models failing to enhance the objective boundary are automatically penalized with zero weight allocations, preventing sub-optimal estimator pollution.

---

## 5. Experimental Evolution & Progression

The development of the forecasting architecture progressed through eighteen systematic iterations, documented comprehensively in `AI_LAB_BOOK.md`. Below is the consolidated trajectory of key architectural milestones:

| Experiment ID | Core Architectural Paradigm | Local CV (MASE) | Out-of-Sample Score | Primary Analytical Finding |
| :--- | :--- | :---: | :---: | :--- |
| `Exp01` | LightGBM Baseline with Log-RMSE | ~0.3900 | 1.1139 | Severe covariate shift; unobserved operational features in future horizon. |
| `Exp02` | GroupKFold by Movie & Cardinality Pruning | 0.4229 | 0.8984 | Context amnesia caused by aggressive identifier dropping. |
| `Exp05` | Tri-Model Stacking Ensemble with SLSQP | 0.4524 | 1.1000 | Feature mismatch on operational screen allocations between train and test. |
| `Exp07` | Fractional Ratio Modeling & Competitor Pruning | 0.4472 | 0.6252 | Fundamental breakthrough: ratio targeting resolves tree extrapolation limits. |
| `Exp08` | Deep Learning Tabular Entity Embeddings | 0.4285 | ~0.8000 | High-dimensional embedding memorization failed under distribution shift. |
| `Exp10` | Star Power Integration & Local Ratios | 0.4610 | 0.6808 | Missing capacity features and national median imputation inflated small venues. |
| `Exp11` | Global Age Tracking with Raw Ticket MAE | 0.9810 | Did Not Deployed | Direct count regression biased models toward mega-theaters, ignoring small cinemas. |
| `Exp13` | Exact MASE Equivalence via Weighted Loss | 0.4410 | 0.66855 | Sample-weighting by $S_p$ degraded unweighted macro-averaged test metrics. |
| `Exp14` | Strict Days 4–10 Horizon with Unweighted Ratio MAE | 0.4474 | 0.62822 | Establishing unweighted ratio regression with monotonic bounds. |
| `Exp16` | TMDB Sentiment & Regex Title Cleaning | 0.4349 | 0.64242 | 100% metadata join achieved; identified weekday target encoding distortion. |
| `Exp17` | Clean WOM Modeling & Bounded Golden Blend | 0.4441 | 0.62707 | Purging static target encoding restored natural weekday decay dynamics. |
| `Exp18` | Trio Ensemble Convex Blend | 0.4390 | Did Not Deployed | Analytical synthesis of Exp07, Exp14, and Exp17 maximizing variance reduction. |

---

## 6. Repository Organization

The repository enforces strict separation of concerns and standard file naming protocols:

```
├── README.md                                  # Enterprise project documentation
├── AI_LAB_BOOK.md                             # Comprehensive technical lab log and post-mortems
├── .gitignore                                 # Git exclusion rules
├── submission_final_ensemble.csv              # Production benchmark forecast output
│
├── dataset/                                   # Cleaned, unjoined relational source files
│   ├── train.csv                              # Historical screening logs
│   ├── test.csv                               # Deployment inference horizon
│   ├── test_history.csv                       # Historical Days 1-3 opening profile
│   ├── movies.csv                             # Film metadata and attributes
│   ├── movies_tmdb_ratings.csv                # Sentiment and audience review metrics
│   ├── ticket_prices.csv                      # Municipal pricing schedules
│   ├── weather_data.csv                       # City-level meteorological observations
│   ├── holidays.csv                           # Calendar event indices
│   └── sample_submission.csv                  # Reference schema
│
└── experiments/                               # Modular experiment containers
    ├── Exp01_BaselineLGBM/                    # Exp01 notebook and submission artifact
    ├── Exp02_GroupKFold_AntiLeakage/          # Exp02 notebook and submission artifact
    ├── Exp03_MetaEnsemble_NLPPCA/             # Exp03 notebook and submission artifact
    ├── Exp04_LogMonotonic/                    # Exp04 notebook and submission artifact
    ├── Exp05_StackingEnsemble/                # Exp05 notebook and submission artifact
    ├── Exp06_StrictCausalityStacking/          # Exp06 notebook and submission artifact
    ├── Exp07_RatioTargeting_FeaturePruning/   # Exp07 notebook and submission artifact
    ├── Exp08_DeepLearningEmbeddings/          # Exp08 notebook and submission artifact
    ├── Exp09_HybridBlending/                  # Exp09 notebook and submission artifact
    ├── Exp10_StarPower_RatioStacking/         # Exp10 notebook and submission artifact
    ├── Exp11_GlobalAge_RawCountMAE/           # Exp11 notebook and submission artifact
    ├── Exp12_CapacityLogRatio/                # Exp12 notebook and submission artifact
    ├── Exp13_WeightedMASE_ExactScaling/       # Exp13 notebook and submission artifact
    ├── Exp14_StrictHorizon_UnweightedMAE/     # Exp14 notebook and submission artifact
    ├── Exp15_FeatureCrossTargetEncoding/      # Exp15 notebook and submission artifact
    ├── Exp16_TMDB_WordOfMouth_Decay/          # Exp16 notebook and submission artifact
    ├── Exp17_CleanWOM_GoldenBlend/            # Exp17 notebook and submission artifact
    └── Exp18_TrioEnsemble_ConvexBlend/        # Exp18 notebook and submission artifact
```

---

## 7. Installation & Reproducibility Guide

### 7.1 Environment Prerequisites
* Python 3.10 or higher
* Standard machine learning stack: `numpy`, `pandas`, `scipy`, `scikit-learn`, `lightgbm`, `xgboost`, `catboost`

### 7.2 Dependency Installation
```bash
pip install numpy pandas scipy scikit-learn lightgbm xgboost catboost
```

### 7.3 Reproducing Model Execution
All experimental notebooks are fully self-contained and sequentially runnable:
1. Navigate to the desired experiment container, e.g., `experiments/Exp17_CleanWOM_GoldenBlend/`.
2. Open and execute `Exp17_CleanWOM_GoldenBlend.ipynb` using Jupyter Notebook or JupyterLab:
   ```bash
   jupyter notebook experiments/Exp17_CleanWOM_GoldenBlend/Exp17_CleanWOM_GoldenBlend.ipynb
   ```
3. To generate the final ensemble synthesis, execute:
   ```bash
   jupyter notebook experiments/Exp18_TrioEnsemble_ConvexBlend/Exp18_TrioEnsemble_ConvexBlend.ipynb
   ```
   The generated artifact will output directly to `submission_final_ensemble.csv`.

