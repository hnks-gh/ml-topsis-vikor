# ML Forecasting Module

## ⚠️ Experimental Status

This module is currently **experimental** and **not integrated** into the main ML-MCDM pipeline.

### Current State

- **Status**: Fully implemented but isolated from main workflow
- **Stability**: API may change in future releases
- **Testing**: Limited integration testing with main pipeline
- **Documentation**: Comprehensive in [docs/forecast.md](../docs/forecast.md)

### Capabilities

The forecasting module provides 7 model types for time-series prediction:

#### Tree-Based Ensemble
- **Gradient Boosting** (GB): Boosted decision trees with adaptive learning
- **Random Forest** (RF): Ensemble of independent decision trees  
- **Extra Trees** (ET): Randomized tree ensemble

#### Linear Models
- **Bayesian Ridge**: Probabilistic linear regression with uncertainty
- **Huber**: Robust linear regression (outlier-resistant)
- **Ridge**: L2-regularized linear regression

#### Neural Networks
- **MLP**: Multi-layer perceptron with dropout
- **Attention**: Attention-based neural architecture (planned)

### Architecture

```
UnifiedForecaster
├── Model Training: Cross-validated training on historical panel data
├── Ensemble Weighting: Performance-based model combination
├── Uncertainty Quantification: Prediction intervals
└── Feature Engineering: Temporal feature extraction
```

### Future Roadmap

#### Near Term
1. **Integration with main pipeline**: Expose forecasting through main workflow
2. **Enhanced uncertainty**: Conformal prediction, quantile regression
3. **Multi-step forecasting**: Predict multiple years ahead

#### Long Term
4. **Hybrid MCDM-ML**: Integrate rankings into forecasting features
5. **Online learning**: Incremental model updates
6. **Explainability**: SHAP values for feature importance

### Usage (When Integrated)

```python
from forecasting import UnifiedForecaster, ForecastMode

# Create forecaster
forecaster = UnifiedForecaster(mode=ForecastMode.BALANCED)

# Fit and predict
result = forecaster.fit_predict(panel_data, target_year=2025)

# Get predictions
predictions = result.predictions  # Point predictions
uncertainty = result.uncertainty  # Prediction intervals
```

### Development Status

| Component | Status | Notes |
|-----------|--------|-------|
| Tree Ensemble | ✅ Complete | GB, RF, ET implemented |
| Linear Models | ✅ Complete | Bayesian, Huber, Ridge |
| Neural Models | 🚧 Partial | MLP complete, Attention planned |
| Feature Engineering | ✅ Complete | Temporal features, lags, rolling stats |
| Unified Interface | ✅ Complete | Model orchestration ready |
| Pipeline Integration | ❌ Pending | Main integration not yet implemented |
| Testing | ⚠️ Limited | Standalone tests exist, integration tests needed |

### Why Isolated?

The forecasting module was developed as part of the ML-MCDM framework but remains
isolated for the following reasons:

1. **Core Focus**: Current pipeline focuses on ranking with historical data
2. **Validation Complexity**: Forecasting requires different validation approaches
3. **Use Case Clarity**: Need to define clear forecasting use cases for MCDM context
4. **Performance Considerations**: Model training can be computationally expensive

### Contributing

If you want to help integrate this module:

1. Review [docs/forecast.md](../docs/forecast.md) for technical details
2. Examine [tests/](../tests/) for existing test patterns
3. Consider integration points in [pipeline.py](../pipeline.py)
4. Propose integration design via GitHub Issues

---

**Last Updated**: 2026-02-14  
**Maintainer**: Son Hoang  
**Status**: Experimental - Use with caution
