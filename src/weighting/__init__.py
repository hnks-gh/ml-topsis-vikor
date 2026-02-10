# -*- coding: utf-8 -*-
"""
Weighting Methods Module

Objective weight calculation methods for MCDM:
- RobustGlobalWeighting: 7-step global hybrid pipeline (PCA-CRITIC-Entropy
    + KL-Divergence Fusion + Bayesian Bootstrap) — PRIMARY METHOD
- Entropy Weights: Information theory-based weighting (standalone utility)
- CRITIC Weights: Criteria Importance Through Inter-criteria Correlation (standalone utility)
- PCA Weights: Principal Component Analysis-based multivariate weighting (standalone utility)
"""

from .robust_global import RobustGlobalWeighting
from .entropy import EntropyWeightCalculator
from .critic import CRITICWeightCalculator
from .pca import PCAWeightCalculator
from .base import WeightResult, calculate_weights

__all__ = [
    'WeightResult',
    'RobustGlobalWeighting',
    'EntropyWeightCalculator',
    'CRITICWeightCalculator',
    'PCAWeightCalculator',
    'calculate_weights'
]
