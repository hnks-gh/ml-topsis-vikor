# -*- coding: utf-8 -*-
"""
MCDM Module - TOPSIS, VIKOR, Fuzzy TOPSIS, and Weights
=======================================================

Production-ready implementations of Multi-Criteria Decision Making methods.
"""

from .topsis import TOPSISCalculator, DynamicTOPSIS, TOPSISResult
from .vikor import VIKORCalculator, VIKORResult, MultiPeriodVIKOR
from .fuzzy_topsis import FuzzyTOPSIS, FuzzyTOPSISResult
from .weights import EntropyWeightCalculator, CRITICWeightCalculator, EnsembleWeightCalculator

__all__ = [
    'TOPSISCalculator', 'DynamicTOPSIS', 'TOPSISResult',
    'VIKORCalculator', 'VIKORResult', 'MultiPeriodVIKOR',
    'FuzzyTOPSIS', 'FuzzyTOPSISResult',
    'EntropyWeightCalculator', 'CRITICWeightCalculator', 'EnsembleWeightCalculator'
]
