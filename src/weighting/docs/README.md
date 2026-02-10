# Weighting Methods — Robust Global Hybrid

## Overview

This module provides **objective weight calculation** for Multi-Criteria Decision Making (MCDM) using a **Robust Global Hybrid Weighting** pipeline that operates on the entire panel dataset (all entities × all time periods) simultaneously.

## Key Changes (February 2026)

### New: Robust Global Hybrid Weighting Pipeline

Replaced all previous weighting methods (panel-aware, ensemble, game theory, 
Bayesian bootstrap, integrated hybrid) with a single, unified 7-step pipeline 
that is **more rigorous, scientifically grounded, and better suited for 
correlated panel data**.

| Before (Removed) | After (New) |
|---|---|
| Panel-aware Entropy/CRITIC/PCA with spatial-temporal split | Global panel operation (preserves temporal trends) |
| Ensemble: arithmetic, geometric, integrated hybrid | KL-Divergence fusion (geometric mean, information-theoretic) |
| Game theory: entropy-confidence heuristic | True 3-vector fusion (Entropy + CRITIC + PCA) |
| Bayesian bootstrap: row resampling | Bayesian bootstrap: Dirichlet observation weights (Rubin, 1981) |
| No stability verification | Split-half cosine similarity + Spearman correlation |

---

## Architecture

```
src/weighting/
├── robust_global.py    # RobustGlobalWeighting — PRIMARY (7-step pipeline)
├── base.py             # WeightResult dataclass + calculate_weights() convenience
├── entropy.py          # EntropyWeightCalculator (standalone utility)
├── critic.py           # CRITICWeightCalculator (standalone utility)
└── pca.py              # PCAWeightCalculator (standalone utility)
```

**Primary class**: `RobustGlobalWeighting` — the only class used in the pipeline.

**Standalone utilities**: `EntropyWeightCalculator`, `CRITICWeightCalculator`, 
`PCAWeightCalculator` are retained for testing and ad-hoc analysis but are 
**not called by the pipeline**.

---

## The 7-Step Pipeline

### Step 1: Global Min-Max Normalization

Normalize across the **entire panel** (not year-by-year) to preserve temporal 
trends over all years.

```
x_norm = (x - min_global) / (max_global - min_global) + ε
```

The ε-shift (default 1e-10) ensures no exact zeros for entropy calculation.

### Step 2: PCA Structural Decomposition & Residualization

Remove dominant common trends via PCA, yielding residuals that capture each 
criterion's **unique information**.

- Component selection: **cumulative variance threshold** (default 0.80)
- Not Kaiser rule (eigenvalue > 1), which over-extracts for p > 20 (Zwick & Velicer, 1986)

```
X_hat = Z @ V_K^T @ V_K
R = Z - X_hat
```

### Step 3: PCA-Residualized CRITIC Weights

CRITIC with a deliberate hybrid: **σ from the global matrix** (absolute contrast 
intensity) and **Pearson r from the residual matrix** (unique conflict).

```
C_j = σ_j(global) × Σ_k(1 - r_jk(residual))
```

Rationale: σ captures how much a criterion varies *overall*, while residualized 
correlation captures how *uniquely* it varies relative to dominant common trends.

### Step 4: Global Entropy Weights

Standard Shannon entropy on the full N-row panel:

```
p_ij = x_ij / Σ_i(x_ij)
e_j = -(1/ln(N)) × Σ_i(p_ij × ln(p_ij))
w_j = (1 - e_j) / Σ_k(1 - e_k)
```

### Step 5: PCA Loadings-based Weights

```
w_j = Σ_k (λ_k / Σλ) × v_jk²
```

Retaining components up to the variance threshold.

### Step 6: KL-Divergence Fusion (Geometric Mean)

Fuse the three weight vectors by minimizing total KL-divergence (Genest & Zidek, 1986):

```
w_j* ∝ Π_k (w_j^(k))^(α_k)
```

With equal coefficients α = [1/3, 1/3, 1/3] by default.

**Why KL-divergence over Game Theory?**
- With m=2 vectors, Game Theory (Gram matrix) yields trivially equal weights
- With m=3, KL-divergence is information-theoretically optimal
- Conservative at the tails: requires consensus among all methods
- Robust with correlated indicators (dampens single-method outliers)

### Step 7: Bayesian Bootstrap (Dirichlet-Weighted)

For each of B=999 iterations:
1. Draw observation weights from Dirichlet(1, ..., 1)
2. Re-run Steps 2–6 with weighted statistics
3. Collect the fused weight vector

Output: posterior **mean** (final weights), **std**, **95% credible intervals**.

B=999 is standard for percentile-based credible intervals (Davison & Hinkley, 1997). 
Odd numbers avoid interpolation at the 2.5th/97.5th percentiles.

### Stability Verification

Split-half check: compute weights on the first half vs. second half of time periods.

- **Cosine similarity** (vector agreement): target ≥ 0.95
- **Spearman rank correlation** (ordinal agreement): measures whether the 
  ranking of criteria importance is preserved

High values confirm the weights are **structural** (not dependent on which 
years are included), making them valid for future predictive data.

---

## Configuration

```python
@dataclass
class WeightingConfig:
    pca_variance_threshold: float = 0.80     # PCA component retention
    bootstrap_iterations: int = 999          # Bayesian bootstrap B
    fusion_alphas: List[float] = [1/3, 1/3, 1/3]  # [entropy, critic, pca]
    stability_threshold: float = 0.95        # Split-half cosine target
    epsilon: float = 1e-10                   # Numerical stability
```

---

## Usage

```python
from src.weighting import RobustGlobalWeighting

calc = RobustGlobalWeighting(
    pca_variance_threshold=0.80,
    bootstrap_iterations=999,
)

result = calc.calculate(
    panel_df,               # Long-format: Year, Province, C01..C29
    entity_col='Province',
    time_col='Year',
    criteria_cols=['C01', 'C02', ..., 'C29']
)

# Final weights (posterior mean from Bayesian Bootstrap)
print(result.weights)

# Individual weight vectors
print(result.details['individual_weights']['entropy'])
print(result.details['individual_weights']['critic'])
print(result.details['individual_weights']['pca'])

# Bootstrap uncertainty
print(result.details['bootstrap']['std_weights'])
print(result.details['bootstrap']['ci_lower_2_5'])
print(result.details['bootstrap']['ci_upper_97_5'])

# Stability check
print(result.details['stability']['cosine_similarity'])
print(result.details['stability']['spearman_correlation'])
print(result.details['stability']['is_stable'])
```

---

## Scientific Corrections Applied

The original strategy document ([new-strategy.txt](new-strategy.txt)) proposed 
a 7-step pipeline. The following corrections were applied during implementation:

| Issue | Correction | Reference |
|---|---|---|
| Kaiser rule for PCA (eigenvalue > 1) | Cumulative variance threshold (0.80) — Kaiser over-extracts for p > 20 | Zwick & Velicer (1986) |
| Game Theory with m=2 is degenerate | KL-Divergence fusion with m=3 vectors (Entropy + CRITIC + PCA) | Genest & Zidek (1986) |
| Bootstrap B=1000 (even number) | B=999 (odd, conventional for percentile CIs) | Davison & Hinkley (1997) |
| Cosine similarity only for stability | Added Spearman rank correlation alongside | — |
| Zero values in entropy (0·ln0 undefined) | ε-shift in normalization step | Standard practice |
| No justification for σ/r mixing in CRITIC | Documented: σ = absolute contrast, r = unique conflict | Diakoulaki et al. (1995) |

---

## References

1. Diakoulaki, D., Mavrotas, G., & Papayannakis, L. (1995). Determining 
   objective weights in multiple criteria problems: The CRITIC method. 
   *Computers & Operations Research.*

2. Shannon, C.E. (1948). A Mathematical Theory of Communication. 
   *Bell System Technical Journal.*

3. Deng, H., Yeh, C.H., & Willis, R.J. (2000). Inter-company comparison 
   using modified TOPSIS with objective weights. *Computers & Ops Research.*

4. Genest, C. & Zidek, J.V. (1986). Combining probability distributions: 
   A critique and annotated bibliography. *Statistical Science.*

5. Abbas, A.E. (2009). A Kullback-Leibler view of linear and log-linear pools. 
   *Decision Analysis.*

6. Rubin, D.B. (1981). The Bayesian Bootstrap. *Annals of Statistics.*

7. Davison, A.C. & Hinkley, D.V. (1997). Bootstrap Methods and Their 
   Application. Cambridge University Press.

8. Zwick, W.R. & Velicer, W.F. (1986). Comparison of five rules for 
   determining the number of components to retain. *Psychological Bulletin.*
