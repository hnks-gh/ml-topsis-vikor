# ML-MCDM Methods Documentation v2.0

This document describes the refactored ML-MCDM framework architecture that separates:
- **Traditional MCDM**: Applied to the last year of panel data
- **Fuzzy MCDM**: Applied to the last year with uncertainty from temporal variance
- **ML Forecasting**: Uses all historical data to predict next year values

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Weight Calculation Methods](#2-weight-calculation-methods)
3. [Traditional MCDM Methods](#3-traditional-mcdm-methods)
4. [Fuzzy MCDM Methods](#4-fuzzy-mcdm-methods)
5. [Advanced ML Forecasting](#5-advanced-ml-forecasting)
6. [Pipeline Integration](#6-pipeline-integration)

---

## 1. Architecture Overview

### 1.1 Design Philosophy

The v2 architecture follows a clear separation of concerns:

```
Panel Data (Multiple Years)
    │
    ├── Last Year Data ─────────┬──► Traditional MCDM (5 methods)
    │                           │
    │   Temporal Variance ──────┴──► Fuzzy MCDM (5 methods)
    │
    └── All Historical Data ───────► ML Forecasting ──► Next Year Predictions
                                            │
                                            └──► Future MCDM Application
```

### 1.2 MCDM Methods Implemented

| Method | Traditional | Fuzzy | Description |
|--------|------------|-------|-------------|
| TOPSIS | ✓ | ✓ | Technique for Order Preference by Similarity to Ideal Solution |
| VIKOR | ✓ | ✓ | Multi-criteria Optimization and Compromise Solution |
| PROMETHEE | ✓ | ✓ | Preference Ranking Organization Method for Enrichment Evaluations |
| COPRAS | ✓ | ✓ | Complex Proportional Assessment |
| EDAS | ✓ | ✓ | Evaluation based on Distance from Average Solution |

### 1.3 ML Forecasting Methods

- **Gradient Boosting Ensemble**: XGBoost-style with Huber loss
- **Random Forest**: With time-series cross-validation
- **Extra Trees**: For ensemble diversity
- **Bayesian Ridge**: For uncertainty quantification
- **Neural Networks**: MLP with residual connections and attention

---

## 2. Weight Calculation Methods

### 2.1 Entropy Weights

Assigns higher weights to criteria with more variation (information content).

$$w_j = \frac{1 - E_j}{\sum_{k=1}^{n} (1 - E_k)}$$

where $E_j = -\frac{1}{\ln(m)} \sum_{i=1}^{m} p_{ij} \ln(p_{ij})$

### 2.2 CRITIC Weights

Considers both contrast intensity and inter-criteria correlation.

$$w_j = \frac{\sigma_j \cdot \sum_{k}(1-r_{jk})}{\sum_{j} \sigma_j \cdot \sum_{k}(1-r_{jk})}$$

---

## 3. Traditional MCDM Methods

All traditional methods operate on the **last year** of panel data only.

### 3.1 TOPSIS

**Technique for Order Preference by Similarity to Ideal Solution**

**Steps:**
1. Normalize the decision matrix
2. Calculate weighted normalized matrix
3. Determine ideal ($A^+$) and anti-ideal ($A^-$) solutions
4. Calculate distances to ideal and anti-ideal
5. Calculate relative closeness: $C_i = \frac{D_i^-}{D_i^+ + D_i^-}$

**Code:** `src/mcdm/topsis.py`

### 3.2 VIKOR

**Multi-criteria Optimization and Compromise Solution**

**Steps:**
1. Determine best ($f_j^*$) and worst ($f_j^-$) values
2. Calculate $S_i$ (maximum group utility) and $R_i$ (minimum regret)
3. Calculate $Q_i = v \cdot \frac{S_i - S^*}{S^- - S^*} + (1-v) \cdot \frac{R_i - R^*}{R^- - R^*}$

**Code:** `src/mcdm/vikor.py`

### 3.3 PROMETHEE

**Preference Ranking Organization Method**

**Steps:**
1. Calculate pairwise preference functions
2. Calculate aggregated preference indices
3. Compute positive flow $\Phi^+(a)$ and negative flow $\Phi^-(a)$
4. Calculate net flow: $\Phi(a) = \Phi^+(a) - \Phi^-(a)$

**Preference Functions:**
- Usual: $P(d) = 0$ if $d \leq 0$, else $1$
- V-shape: $P(d) = \min(1, d/p)$
- Gaussian: $P(d) = 1 - e^{-d^2/2\sigma^2}$

**Code:** `src/mcdm/promethee.py`

### 3.4 COPRAS

**Complex Proportional Assessment**

**Steps:**
1. Calculate normalized weighted matrix
2. Sum maximizing criteria: $S_i^+ = \sum_{j \in J_{max}} w_j \cdot r_{ij}$
3. Sum minimizing criteria: $S_i^- = \sum_{j \in J_{min}} w_j \cdot r_{ij}$
4. Calculate relative significance: $Q_i = S_i^+ + \frac{\sum S_i^-}{S_i^- \cdot \sum(1/S_i^-)}$
5. Utility degree: $N_i = \frac{Q_i}{Q_{max}} \times 100\%$

**Code:** `src/mcdm/copras.py`

### 3.5 EDAS

**Evaluation based on Distance from Average Solution**

**Steps:**
1. Calculate average solution (AV)
2. Calculate Positive Distance from Average (PDA)
3. Calculate Negative Distance from Average (NDA)
4. Weighted sum: $SP_i = \sum w_j \cdot PDA_{ij}$, $SN_i = \sum w_j \cdot NDA_{ij}$
5. Normalize: $NSP_i$, $NSN_i$
6. Appraisal Score: $AS_i = \frac{NSP_i + (1-NSN_i)}{2}$

**Code:** `src/mcdm/edas.py`

---

## 4. Fuzzy MCDM Methods

All fuzzy methods use **Triangular Fuzzy Numbers (TFN)** to represent uncertainty.

### 4.1 Triangular Fuzzy Numbers

A TFN is represented as $\tilde{A} = (l, m, u)$ where:
- $l$: Lower bound (pessimistic)
- $m$: Modal value (most likely)
- $u$: Upper bound (optimistic)

**Arithmetic Operations:**
- Addition: $(l_1+l_2, m_1+m_2, u_1+u_2)$
- Multiplication: $(l_1 \cdot l_2, m_1 \cdot m_2, u_1 \cdot u_2)$
- Division: $(l_1/u_2, m_1/m_2, u_1/l_2)$

**Defuzzification (Centroid):**
$$\text{defuzz}(\tilde{A}) = \frac{l + m + u}{3}$$

### 4.2 Uncertainty from Panel Data

Fuzzy numbers are created from panel data using temporal variance:

```python
# For each entity-criterion pair:
m = last_year_value
spread = std_over_years * spread_factor
l = m - spread
u = m + spread
```

**Code:** `src/mcdm/fuzzy_base.py`

### 4.3 Fuzzy TOPSIS

Extends TOPSIS with fuzzy arithmetic:
1. Create fuzzy decision matrix from panel data
2. Calculate fuzzy weighted matrix
3. Determine fuzzy ideal and anti-ideal
4. Calculate vertex distance to ideal/anti-ideal
5. Compute closeness coefficient

**Code:** `src/mcdm/fuzzy_topsis.py`

### 4.4 Fuzzy VIKOR

Extends VIKOR with fuzzy numbers:
1. Create fuzzy decision matrix
2. Calculate fuzzy S (group utility) and R (regret)
3. Defuzzify and calculate Q
4. Rank based on Q values

**Code:** `src/mcdm/fuzzy_vikor.py`

### 4.5 Fuzzy PROMETHEE

Extends PROMETHEE with fuzzy preferences:
1. Create fuzzy decision matrix
2. Calculate fuzzy preference degrees
3. Aggregate to fuzzy preference indices
4. Calculate fuzzy positive/negative flows
5. Defuzzify for final ranking

**Code:** `src/mcdm/fuzzy_promethee.py`

### 4.6 Fuzzy COPRAS

Extends COPRAS with fuzzy arithmetic:
1. Create fuzzy decision matrix
2. Calculate fuzzy S+ and S-
3. Calculate fuzzy Q values
4. Defuzzify for utility degree

**Code:** `src/mcdm/fuzzy_copras.py`

### 4.7 Fuzzy EDAS

Extends EDAS with fuzzy numbers:
1. Create fuzzy decision matrix
2. Calculate fuzzy average solution
3. Calculate fuzzy PDA and NDA
4. Compute fuzzy SP and SN
5. Calculate appraisal score

**Code:** `src/mcdm/fuzzy_edas.py`

---

## 5. Advanced ML Forecasting

The ML forecasting system predicts **next year values** using all historical data.

### 5.1 Feature Engineering

**Temporal Features Created:**
- Current values (t)
- Lag features (t-1, t-2)
- Rolling statistics (mean, std, min, max)
- Momentum (rate of change)
- Acceleration (change in momentum)
- Trend (linear slope)
- Cross-entity features (percentile rank, z-score)

**Code:** `src/ml/advanced_forecasting.py` → `TemporalFeatureEngineer`

### 5.2 Model Ensemble

The system combines multiple models:

| Model | Type | Strength |
|-------|------|----------|
| Gradient Boosting | Tree | Handles non-linearity, robust to outliers |
| Random Forest | Tree | Stable, interpretable |
| Extra Trees | Tree | Diversity, fast |
| Bayesian Ridge | Linear | Uncertainty quantification |
| Huber Regression | Linear | Outlier robust |
| Neural MLP | NN | Complex patterns |
| Attention Model | NN | Temporal weighting |

### 5.3 Ensemble Weighting

Optimal weights are calculated based on cross-validation performance:

$$w_m = \frac{\text{CV\_Score}_m^2}{\sum_{k} \text{CV\_Score}_k^2}$$

### 5.4 Uncertainty Quantification

Prediction intervals are calculated using:
1. Bayesian Ridge posterior (if available)
2. Model disagreement (ensemble spread)

$$\text{CI}_{95\%} = \hat{y} \pm 1.96 \cdot \sigma_{ensemble}$$

### 5.5 Usage Modes

| Mode | Models | Speed | Accuracy |
|------|--------|-------|----------|
| fast | GB + Huber | ⚡⚡⚡ | ★★ |
| balanced | GB + RF + BR + Huber + NN | ⚡⚡ | ★★★ |
| accurate | All models | ⚡ | ★★★★ |
| neural | RF + NN | ⚡⚡ | ★★★ |
| ensemble | All models | ⚡ | ★★★★★ |

**Code:** `src/ml/unified_forecasting.py`

---

## 6. Pipeline Integration

### 6.1 Pipeline v2 Workflow

```python
from src.pipeline_v2 import PipelineV2

pipeline = PipelineV2(
    output_dir="outputs",
    ml_mode="balanced",
    verbose=True
)

result = pipeline.run(panel_data)

# Access results
print(result.summary())

# Traditional MCDM results
for method, score in result.traditional_mcdm.items():
    print(f"{method}: Top entity = {result.entities[score.rankings.argmin()]}")

# Fuzzy MCDM results
for method, score in result.fuzzy_mcdm.items():
    print(f"{method}: Top entity = {result.entities[score.rankings.argmin()]}")

# ML predictions
if result.predicted_next_year is not None:
    print(f"Predicted next year: {result.predicted_next_year}")

# Final consensus ranking
print(result.final_rankings.head(10))
```

### 6.2 Result Structure

```
PipelineResultV2
├── panel_data          # Original data
├── traditional_mcdm    # Dict[method_name -> MCDMScore]
├── fuzzy_mcdm          # Dict[method_name -> MCDMScore]
├── weights             # Dict[method -> array]
├── ml_forecast         # UnifiedForecastResult
├── predicted_next_year # DataFrame
├── prediction_uncertainty # DataFrame
├── final_rankings      # DataFrame with consensus
└── consensus_ranking   # Array of final ranks
```

---

## 7. References

1. Hwang, C.L., & Yoon, K. (1981). Multiple Attribute Decision Making. Springer.
2. Opricovic, S., & Tzeng, G.H. (2004). Compromise solution by MCDM methods. EJOR.
3. Brans, J.P., & Vincke, P. (1985). A preference ranking organisation method. Management Science.
4. Zavadskas, E.K., & Kaklauskas, A. (1996). Determination of a rational complex assessment. Statyba.
5. Ghorabaee, M.K., et al. (2015). Multi-criteria inventory classification using EDAS method. JIEC.
6. Chen, C.T. (2000). Extensions of the TOPSIS for group decision-making under fuzzy environment. FSS.
