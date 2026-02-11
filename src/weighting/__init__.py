# -*- coding: utf-8 -*-
"""
Weighting Methods Module

Objective weight calculation methods for MCDM:
- RobustGlobalWeighting: 4-method hybrid pipeline (Entropy + CRITIC + MEREC + 
    Standard Deviation → Reliability-Weighted Fusion + Bayesian Bootstrap) — PRIMARY
- Entropy: Information theory-based weighting (standalone)
- CRITIC: Criteria Importance Through Inter-criteria Correlation (standalone)
- MEREC: Method based on Removal Effects of Criteria (standalone)
- Standard Deviation: Variance-based weighting (standalone)
- AdvancedWeightFusion: State-of-the-art fusion methods for combining weights
"""

from .robust_global import RobustGlobalWeighting
from .entropy import EntropyWeightCalculator
from .critic import CRITICWeightCalculator
from .merec import MERECWeightCalculator
from .standard_deviation import StandardDeviationWeightCalculator
from .fusion import AdvancedWeightFusion
from .base import WeightResult, calculate_weights

__all__ = [
    'WeightResult',
    'RobustGlobalWeighting',
    'EntropyWeightCalculator',
    'CRITICWeightCalculator',
    'MERECWeightCalculator',
    'StandardDeviationWeightCalculator',
    'AdvancedWeightFusion',
    'calculate_weights'
]
