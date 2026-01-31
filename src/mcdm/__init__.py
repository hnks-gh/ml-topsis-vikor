# -*- coding: utf-8 -*-
"""Multi-Criteria Decision Making methods (traditional and fuzzy variants)."""

# Fuzzy base classes
from .fuzzy_base import (
    TriangularFuzzyNumber,
    FuzzyDecisionMatrix,
    LINGUISTIC_SCALE_5,
    LINGUISTIC_SCALE_7,
    IMPORTANCE_SCALE
)

# Weight calculation methods
from .weights import (
    EntropyWeightCalculator,
    CRITICWeightCalculator,
    EnsembleWeightCalculator,
    WeightResult
)

# TOPSIS - Traditional, Dynamic (legacy), and Fuzzy
from .topsis import TOPSISCalculator, TOPSISResult, DynamicTOPSIS, DynamicTOPSISResult
from .fuzzy_topsis import FuzzyTOPSIS, FuzzyTOPSISResult

# VIKOR - Traditional, MultiPeriod (legacy), and Fuzzy
from .vikor import VIKORCalculator, VIKORResult, MultiPeriodVIKOR
from .fuzzy_vikor import FuzzyVIKOR, FuzzyVIKORResult

# PROMETHEE - Traditional and Fuzzy
from .promethee import PROMETHEECalculator, PROMETHEEResult
from .fuzzy_promethee import FuzzyPROMETHEE, FuzzyPROMETHEEResult

# COPRAS - Traditional and Fuzzy
from .copras import COPRASCalculator, COPRASResult
from .fuzzy_copras import FuzzyCOPRAS, FuzzyCOPRASResult

# EDAS - Traditional and Fuzzy
from .edas import EDASCalculator, EDASResult
from .fuzzy_edas import FuzzyEDAS, FuzzyEDASResult


__all__ = [
    # Fuzzy base
    'TriangularFuzzyNumber',
    'FuzzyDecisionMatrix',
    'LINGUISTIC_SCALE_5',
    'LINGUISTIC_SCALE_7',
    'IMPORTANCE_SCALE',
    
    # Weights
    'EntropyWeightCalculator',
    'CRITICWeightCalculator',
    'EnsembleWeightCalculator',
    'WeightResult',
    
    # TOPSIS (including legacy Dynamic)
    'TOPSISCalculator', 'TOPSISResult',
    'DynamicTOPSIS', 'DynamicTOPSISResult',
    'FuzzyTOPSIS', 'FuzzyTOPSISResult',
    
    # VIKOR (including legacy MultiPeriod)
    'VIKORCalculator', 'VIKORResult',
    'MultiPeriodVIKOR',
    'FuzzyVIKOR', 'FuzzyVIKORResult',
    
    # PROMETHEE
    'PROMETHEECalculator', 'PROMETHEEResult',
    'FuzzyPROMETHEE', 'FuzzyPROMETHEEResult',
    
    # COPRAS
    'COPRASCalculator', 'COPRASResult',
    'FuzzyCOPRAS', 'FuzzyCOPRASResult',
    
    # EDAS
    'EDASCalculator', 'EDASResult',
    'FuzzyEDAS', 'FuzzyEDASResult',
]


# Convenience function to get all MCDM calculators
def get_all_calculators(fuzzy: bool = False):
    """
    Get dictionary of all MCDM calculators.
    
    Parameters:
        fuzzy: If True, returns fuzzy versions; if False, returns traditional
    
    Returns:
        Dict[str, class]: Dictionary of calculator classes
    """
    if fuzzy:
        return {
            'topsis': FuzzyTOPSIS,
            'vikor': FuzzyVIKOR,
            'promethee': FuzzyPROMETHEE,
            'copras': FuzzyCOPRAS,
            'edas': FuzzyEDAS,
        }
    else:
        return {
            'topsis': TOPSISCalculator,
            'vikor': VIKORCalculator,
            'promethee': PROMETHEECalculator,
            'copras': COPRASCalculator,
            'edas': EDASCalculator,
        }
