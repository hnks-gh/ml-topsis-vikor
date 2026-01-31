# -*- coding: utf-8 -*-
"""
Machine Learning Module
========================

Comprehensive ML toolkit for panel data analysis and forecasting:
- Panel regression for fixed/random effects
- Random Forest with Time-Series CV
- LSTM neural network forecasting
- Rough Sets for feature reduction
- Advanced ML Forecasting (Gradient Boosting Ensemble)
- Neural Network Forecasting (MLP, Attention)
- Unified Forecasting Orchestrator
"""

# Legacy modules
from .panel_regression import PanelRegression, PanelRegressionResult
from .random_forest_ts import RandomForestTS, RandomForestTSResult
from .lstm_forecast import LSTMForecaster, LSTMResult
from .rough_sets import RoughSetReducer, RoughSetResult

# Advanced forecasting
from .advanced_forecasting import (
    AdvancedMLForecaster,
    TemporalFeatureEngineer,
    ForecastResult,
    GradientBoostingForecaster,
    RandomForestForecaster,
    ExtraTreesForecaster,
    BayesianRidgeForecaster,
    HuberForecaster,
    ComponentForecaster
)

# Neural network forecasting
from .neural_forecasting import (
    NeuralForecaster,
    AttentionTemporalForecaster,
    NeuralEnsembleForecaster,
    DenseLayer,
    AttentionLayer,
    ResidualBlock
)

# Unified forecasting system
from .unified_forecasting import (
    UnifiedForecaster,
    UnifiedForecastResult,
    ForecastMode,
    forecast_next_year
)

__all__ = [
    # Legacy
    'PanelRegression', 'PanelRegressionResult',
    'RandomForestTS', 'RandomForestTSResult',
    'LSTMForecaster', 'LSTMResult',
    'RoughSetReducer', 'RoughSetResult',
    
    # Advanced Forecasting
    'AdvancedMLForecaster',
    'TemporalFeatureEngineer',
    'ForecastResult',
    'GradientBoostingForecaster',
    'RandomForestForecaster',
    'ExtraTreesForecaster',
    'BayesianRidgeForecaster',
    'HuberForecaster',
    'ComponentForecaster',
    
    # Neural Forecasting
    'NeuralForecaster',
    'AttentionTemporalForecaster',
    'NeuralEnsembleForecaster',
    'DenseLayer',
    'AttentionLayer',
    'ResidualBlock',
    
    # Unified System
    'UnifiedForecaster',
    'UnifiedForecastResult',
    'ForecastMode',
    'forecast_next_year'
]


def get_forecaster(mode: str = 'balanced'):
    """
    Get a configured forecaster instance.
    
    Args:
        mode: 'fast', 'balanced', 'accurate', 'neural', or 'ensemble'
    
    Returns:
        Configured UnifiedForecaster instance
    """
    return UnifiedForecaster(mode=ForecastMode(mode.lower()))
