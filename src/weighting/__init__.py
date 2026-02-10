# -*- coding: utf-8 -*-
"""
Weighting Methods Module

Objective weight calculation methods for MCDM:
- Entropy Weights: Information theory-based weighting
- CRITIC Weights: Criteria Importance Through Inter-criteria Correlation
- PCA Weights: Principal Component Analysis-based multivariate weighting
- Ensemble Weights: Advanced combined weighting (hybrid, game theory, Bayesian)
- Panel-Aware Weights: Enhanced methods utilizing panel data structure (time + cross-section)
"""

from .entropy import EntropyWeightCalculator
from .critic import CRITICWeightCalculator
from .pca import PCAWeightCalculator
from .ensemble import EnsembleWeightCalculator
from .panel_weighting import (
    PanelEntropyCalculator,
    PanelCRITICCalculator,
    PanelPCACalculator,
    PanelEnsembleCalculator
)
from .base import WeightResult, calculate_weights

__all__ = [
    'WeightResult',
    'EntropyWeightCalculator',
    'CRITICWeightCalculator',
    'PCAWeightCalculator',
    'EnsembleWeightCalculator',
    'PanelEntropyCalculator',
    'PanelCRITICCalculator',
    'PanelPCACalculator',
    'PanelEnsembleCalculator',
    'calculate_weights'
]
