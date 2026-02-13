# Future Directions: Deep Ensemble with Attention Mechanisms

**Status:** Proposed enhancement to the current IFS+ER framework  
**Current Architecture:** Intuitionistic Fuzzy Sets (IFS) + Evidential Reasoning (ER)  
**Proposed Extension:** Deep learning ensemble with self-attention for method aggregation

---

## Background

The current ML-MCDM framework uses a **hierarchical two-stage aggregation**:

1. **Stage 1 (Within-Criterion):** Combine 12 MCDM methods (6 traditional + 6 IFS) via Evidential Reasoning
2. **Stage 2 (Global):** Aggregate 8 criterion beliefs using weighted ER with GTWC weights

This approach provides:
- ✅ Strong theoretical foundation (Dempster-Shafer Theory)
- ✅ Transparent belief structures
- ✅ Uncertainty quantification through hesitancy degree (π)
- ✅ Robust to missing data and conflicting evidence

However, it has inherent limitations:
- ⚠️ Fixed aggregation rules (not data-adaptive)
- ⚠️ Linear weight combination (GTWC coefficients)
- ⚠️ No learned method-specific reliability
- ⚠️ Cannot capture complex method interactions

---

## Proposed Architecture: Deep Ensemble with Attention

### Core Concept

Replace the **fixed ER aggregation** with a **learned attention-based ensemble** that:

1. **Learns method importance** from historical data
2. **Captures nonlinear interactions** between MCDM methods
3. **Adapts weights dynamically** based on input characteristics
4. **Maintains uncertainty quantification** through ensemble variance

### Technical Design

#### 1. Input Representation

For each entity *i* at time *t*, construct feature vector:

```
X_it = [
    # Raw MCDM scores (12 methods × 8 criteria = 96 features)
    TOPSIS_C01, ..., TOPSIS_C08,
    VIKOR_C01, ..., VIKOR_C08,
    ...,
    IFS_SAW_C01, ..., IFS_SAW_C08,
    
    # IFS uncertainty components (24 features: μ, ν, π for 8 criteria)
    μ_C01, ν_C01, π_C01, ..., μ_C08, ν_C08, π_C08,
    
    # Historical context (lag features)
    Score_t-1, Rank_t-1, Δ_Score, Δ_Rank,
    
    # Criterion weights (8 features)
    w_C01, ..., w_C08
]
```

**Total:** ~130 features

---

#### 2. Multi-Head Self-Attention Layer

**Purpose:** Learn which MCDM methods are most reliable under different conditions

```python
class MCDMAttention(nn.Module):
    def __init__(self, d_model=128, n_heads=4, n_methods=12):
        super().__init__()
        self.method_embed = nn.Linear(8, d_model)  # 8 criterion scores → embedding
        self.multihead_attn = nn.MultiheadAttention(d_model, n_heads)
        self.output = nn.Linear(d_model, 1)  # → aggregated score
        
    def forward(self, method_scores):
        """
        method_scores: [N, 12 methods, 8 criteria]
        returns: [N, 1] aggregated score + [N, 12] attention weights
        """
        # Embed each method's criterion scores
        x = self.method_embed(method_scores)  # [N, 12, d_model]
        
        # Self-attention: methods attend to each other
        x = x.transpose(0, 1)  # [12, N, d_model] (seq_len first)
        attn_output, attn_weights = self.multihead_attn(x, x, x)
        
        # Aggregate
        score = self.output(attn_output).mean(dim=0)  # [N, 1]
        return score, attn_weights
```

**Key idea:** The attention mechanism learns:
- Which methods are complementary (high cross-attention)
- Which methods are redundant (low attention)
- Context-dependent reliability (attention varies by input)

---

#### 3. Ensemble Architecture

Combine **multiple attention heads** with **traditional ML models** for robustness:

```
┌─────────────────────────────────────────────────────┐
│  Input: 12 MCDM method scores × 8 criteria         │
└───────────────┬─────────────────────────────────────┘
                │
    ┌───────────┼───────────┬───────────┐
    ▼           ▼           ▼           ▼
┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
│Attn-1  │ │Attn-2  │ │  RF    │ │  GB    │
│4 heads │ │8 heads │ │        │ │        │
└───┬────┘ └───┬────┘ └───┬────┘ └───┬────┘
    │          │          │          │
    └──────────┴──────────┴──────────┘
                │
                ▼
        ┌───────────────┐
        │ Meta-learner  │ (weighted avg by CV R²)
        └───────┬───────┘
                │
                ▼
        ┌───────────────┐
        │ Final Score + │
        │ Uncertainty   │
        └───────────────┘
```

**Uncertainty quantification:**
$$
\sigma^2_{\text{pred}} = \underbrace{\text{Var}[\hat{y}_i]}_{\text{model disagreement}} + \underbrace{\mathbb{E}[\sigma^2_i]}_{\text{epistemic}}
$$

---

#### 4. Training Procedure

**Objective:** Predict next-year ranking score from current-year MCDM outputs

```python
# Dataset construction
for t in years[:-1]:
    X_train.append({
        'mcdm_scores': scores[t],      # [12 methods, 8 criteria]
        'ifs_uncertainty': ifs[t],     # [μ, ν, π] for each criterion
        'weights': weights[t],         # GTWC criterion weights
        'context': lags[t]             # historical features
    })
    y_train.append(final_score[t+1])   # Target: next year's ER score

# Training
model = DeepEnsemble([
    MCDMAttention(n_heads=4),
    MCDMAttention(n_heads=8),
    RandomForest(),
    GradientBoosting()
])

model.fit(X_train, y_train, 
          validation='time-series-cv',
          n_splits=5,
          optimize='ranking-loss')  # NDCG or Kendall's tau
```

**Loss function:** Ranking-aware (not just MSE)

$$
\mathcal{L} = \alpha \cdot \text{MSE}(\hat{y}, y) + (1-\alpha) \cdot (1 - \tau(\text{rank}(\hat{y}), \text{rank}(y)))
$$

Where τ is Kendall's tau correlation.

---

### 5. Comparison with Current IFS+ER

| Aspect | IFS+ER (Current) | Deep Attention (Proposed) |
|--------|------------------|---------------------------|
| **Theoretical Foundation** | Dempster-Shafer Theory | Deep learning + ensemble |
| **Interpretability** | High (belief structures) | Medium (attention weights) |
| **Adaptivity** | Fixed rules | Learned from data |
| **Method Weighting** | Equal (within ER) | Context-dependent attention |
| **Nonlinear Interactions** | No | Yes (multi-head attention) |
| **Data Requirements** | Low (works with 14 years) | High (50+ years ideal) |
| **Uncertainty Quantification** | Via π (hesitancy) | Via ensemble variance |
| **Computational Cost** | Low (closed-form ER) | High (gradient descent) |
| **Robustness to Outliers** | High (ER conflict handling) | Medium (depends on training) |

---

## Expected Benefits

### 1. **Performance Improvement**

- **Hypothesis:** Attention can learn that certain method combinations are more reliable
  - Example: TOPSIS + VIKOR might agree on "clear winners"
  - Example: PROMETHEE might excel at "close contests"
- **Expected gain:** 5-15% improvement in ranking accuracy (Kendall's tau)

### 2. **Interpretable Method Selection**

Attention weights reveal:
- Which methods contribute most to final ranking
- When certain methods should be down-weighted (e.g., EDAS in low-variance scenarios)
- Method redundancy (high correlation → low unique contribution)

**Example visualization:**
```
Attention Matrix (Criterion C01):
           TOPSIS  VIKOR  PROMETHEE  COPRAS  ...
TOPSIS      0.12    0.18      0.08    0.15
VIKOR       0.18    0.14      0.09    0.12
PROMETHEE   0.08    0.09      0.11    0.22
...
```

### 3. **Dynamic Adaptation**

Unlike fixed ER, attention can adjust to:
- **Temporal shifts:** Methods may drift in reliability over time
- **Regional patterns:** Certain provinces may benefit from specific methods
- **Data quality:** Down-weight methods in high-uncertainty scenarios

---

## Implementation Roadmap

### Phase 1: Baseline Enhancement (2-3 months)

1. **Data Preparation**
   - Extend historical data to 20+ years (if available)
   - Construct feature matrix with all 130 features
   - Split into train (70%), validation (15%), test (15%)

2. **Simple Attention Model**
   - Implement single-head attention over 12 methods
   - Train with MSE loss
   - Compare CV R² vs current ER

3. **Ablation Studies**
   - Attention vs fixed weights
   - Multi-head vs single-head
   - With/without IFS uncertainty features

### Phase 2: Full Ensemble (3-4 months)

1. **Multi-Model Architecture**
   - 2 attention variants + 2 tree models
   - Meta-learner for ensemble combination
   - Uncertainty quantification module

2. **Ranking-Aware Training**
   - Implement Kendall's tau loss
   - Test NDCG vs pairwise ranking loss
   - Optimize for top-k accuracy (k=5, 10, 20)

3. **Production Integration**
   - PyTorch model serving
   - Fallback to ER if deep model fails
   - A/B testing framework

### Phase 3: Advanced Features (4-6 months)

1. **Temporal Attention**
   - Attend over time series (not just methods)
   - Capture trend patterns in governance

2. **Hierarchical Attention**
   - Criterion-level attention
   - Method-level attention
   - Two-stage like current ER, but learned

3. **Transfer Learning**
   - Pre-train on similar indices (HDI, SDG indicators)
   - Fine-tune on PAPI data

---

## Challenges & Mitigations

### 1. **Limited Data (14 years)**

**Problem:** Deep learning needs >50 observations per parameter  
**Mitigations:**
- Use pre-trained embeddings from larger MCDM datasets
- Strong regularization (dropout, weight decay)
- Bayesian neural networks for uncertainty
- Augment with synthetic bootstrap samples

### 2. **Interpretability Loss**

**Problem:** Stakeholders trust ER's transparent belief structures  
**Mitigations:**
- Provide attention weight visualizations
- SHAP values for feature importance
- Keep ER as interpretable baseline
- Use attention as "advisory" second opinion

### 3. **Overfitting Risk**

**Problem:** Complex model on small dataset  
**Mitigations:**
- Cross-validation with strict temporal ordering
- Ensemble prevents single-model overfitting
- Monitor validation performance closely
- Early stopping with patience

### 4. **Computational Cost**

**Problem:** Deep models slower than closed-form ER  
**Mitigations:**
- Cache trained models (retrain yearly, not daily)
- GPU acceleration for batch inference
- Quantize model for production (FP16)
- Hybrid: ER for real-time, deep for annual analysis

---

## Integration with Current System

### Mode 1: Parallel Evaluation (Recommended Start)

Run **both** IFS+ER and Deep Attention, compare results:

```python
# In pipeline.py
result_er = hierarchical_ranking_pipeline.run()  # Current
result_deep = deep_attention_ensemble.predict()  # New

comparison = {
    'kendall_w_agreement': kendalltau(result_er.ranks, result_deep.ranks),
    'top10_overlap': len(set(result_er.top10) & set(result_deep.top10)),
    'rank_differences': result_er.ranks - result_deep.ranks
}

# Save both for validation
output_manager.save_comparison(comparison)
```

### Mode 2: Ensemble of Ensembles (Long-term)

Meta-ensemble combining ER and Deep Attention:

$$
\text{Score}_{\text{final}} = \beta \cdot \text{Score}_{\text{ER}} + (1-\beta) \cdot \text{Score}_{\text{Deep}}
$$

Where β is learned from validation performance.

### Mode 3: Conditional Selection (Production)

```python
if data_quality < threshold or uncertainty > threshold:
    use_er()  # Safe, interpretable baseline
else:
    use_deep_attention()  # Higher accuracy when confident
```

---

## Research Questions to Investigate

1. **Does attention learn meaningful MCDM method patterns?**
   - Analyze attention weight correlations with method properties
   - Check if attention aligns with expert knowledge

2. **How much data is "enough"?**
   - Learning curves: performance vs training set size
   - Bootstrap confidence intervals

3. **Can we transfer learn from other indices?**
   - Pre-train on HDI, GII, SEDA → fine-tune on PAPI
   - Domain adaptation techniques

4. **What is the bias-variance tradeoff?**
   - Simple ER: high bias, low variance
   - Deep attention: low bias, high variance (maybe overfit)
   - Optimal point?

5. **Is ranking-aware loss better than MSE?**
   - Compare Kendall's tau loss vs NDCG vs pairwise ranking
   - Top-k accuracy vs full ranking accuracy

---

## References

### Deep Ensemble Methods

1. **Lakshminarayanan, B., Pritzel, A., & Blundell, C.** (2017). Simple and scalable predictive uncertainty estimation using deep ensembles. *NeurIPS*.

2. **Vaswani, A., et al.** (2017). Attention is all you need. *NeurIPS*.

### MCDM + Machine Learning

3. **Hafezalkotob, A., & Hafezalkotob, A.** (2017). Interval MULTIMOORA method integrating interval Borda rule and interval best–worst multi-criteria decision making. *Journal of Intelligent & Fuzzy Systems*, 32(6), 4249-4261.

4. **Jain, V., et al.** (2020). Ensemble of machine learning algorithms for MCDM: A review. *Expert Systems with Applications*.

### Ranking Loss Functions

5. **Burges, C., et al.** (2005). Learning to rank using gradient descent. *ICML*.

6. **Cao, Z., et al.** (2007). Learning to rank: From pairwise approach to listwise approach. *ICML*.

### Uncertainty Quantification

7. **Kendall, A., & Gal, Y.** (2017). What uncertainties do we need in Bayesian deep learning for computer vision?. *NeurIPS*.

8. **Malinin, A., & Gales, M.** (2018). Predictive uncertainty estimation via prior networks. *NeurIPS*.

---

## Conclusion

The **Deep Ensemble with Attention Mechanisms** represents a natural evolution of the current IFS+ER framework:

- **Short-term:** Run in parallel for validation
- **Medium-term:** Ensemble both approaches for robustness
- **Long-term:** Transition to primary method if consistently superior

**Key advantage:** Learns data-adaptive method combination while maintaining uncertainty quantification.

**Key risk:** Requires more data and careful validation to avoid overfitting.

**Recommendation:** Start with Phase 1 (simple attention baseline) once 20+ years of data are available. Until then, the current IFS+ER system provides a theoretically grounded, interpretable, and robust solution.

---

**Document Status:** Proposal  
**Target Implementation:** 2026-2027 (pending data availability)  
**Contact:** ML-MCDM Development Team
