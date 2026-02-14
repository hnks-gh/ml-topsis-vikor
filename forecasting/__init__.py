# -*- coding: utf-8 -*-
"""
ML Forecasting Module (EXPERIMENTAL)
=====================================

⚠️ **EXPERIMENTAL STATUS**: This module is currently under active development
and is not integrated into the main ML-MCDM pipeline. It is provided for
future enhancement and research purposes. The API may change without notice.

**Current Status**: Isolated from main workflow, planned for future integration.

**Planned Enhancements**:
- Integration with main pipeline for temporal forecasting
- Enhanced uncertainty quantification
- Multi-step ahead forecasting capabilities
- Model selection optimization

This module provides machine learning forecasting methods for multi-criteria
decision making with temporal/panel data.

Submodules:
    - tree_ensemble: Gradient Boosting, Random Forest, Extra Trees
    - linear: Bayesian Ridge, Huber, Ridge regression
    - neural: MLP, Attention-based networks
    - features: Temporal feature engineering
    - unified: Orchestrated ensemble forecasting

Available Classes:
    - UnifiedForecaster: Main orchestrator for ensemble forecasting
    - GradientBoostingForecaster: GB-based forecasting
    - RandomForestForecaster: RF-based forecasting
    - ExtraTreesForecaster: ET-based forecasting
    - BayesianForecaster: Bayesian linear regression
    - HuberForecaster: Robust linear regression
    - NeuralForecaster: MLP with modern architecture
    - AttentionForecaster: Attention-based neural network
    - TemporalFeatureEngineer: Feature engineering for time series

Example Usage:
    >>> from forecasting import UnifiedForecaster, ForecastMode
    >>> 
    >>> # Create forecaster
    >>> forecaster = UnifiedForecaster(mode=ForecastMode.BALANCED)
    >>> 
    >>> # Fit and predict
    >>> result = forecaster.fit_predict(panel_data, target_year=2025)
    >>> 
    >>> # Get predictions
    >>> predictions = result.predictions
    >>> uncertainty = result.uncertainty
"""

# Feature engineering
from .features import TemporalFeatureEngineer

# Tree-based ensemble methods
from .tree_ensemble import (
    GradientBoostingForecaster,
    RandomForestForecaster,
    ExtraTreesForecaster,
)

# Linear methods
from .linear import (
    BayesianForecaster,
    HuberForecaster,
    RidgeForecaster,
)

# Neural network methods
from .neural import (
    NeuralForecaster,
    AttentionForecaster,
    DenseLayer,
    AttentionLayer,
)

# Unified orchestrator
from .unified import (
    UnifiedForecaster,
    UnifiedForecastResult,
    ForecastMode,
)

# Base classes and results
from .base import (
    BaseForecaster,
    ForecastResult,
)

# Time-series specific Random Forest (for panel data)
from .random_forest_ts import (
    RandomForestTS,
    RandomForestTSResult,
    TimeSeriesSplit,
    calculate_shap_importance,
)

__all__ = [
    # Feature engineering
    'TemporalFeatureEngineer',
    # Tree ensemble
    'GradientBoostingForecaster',
    'RandomForestForecaster',
    'ExtraTreesForecaster',
    # Linear
    'BayesianForecaster',
    'HuberForecaster',
    'RidgeForecaster',
    # Neural
    'NeuralForecaster',
    'AttentionForecaster',
    'DenseLayer',
    'AttentionLayer',
    # Unified
    'UnifiedForecaster',
    'UnifiedForecastResult',
    'ForecastMode',
    # Base
    'BaseForecaster',
    'ForecastResult',
    # Time-series RF (panel data)
    'RandomForestTS',
    'RandomForestTSResult',
    'TimeSeriesSplit',
    'calculate_shap_importance',
]
