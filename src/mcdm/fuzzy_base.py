# -*- coding: utf-8 -*-
"""
Fuzzy Number Base Classes
==========================

Common fuzzy number types and operations for all Fuzzy MCDM methods.
Supports triangular and trapezoidal fuzzy numbers.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class TriangularFuzzyNumber:
    """
    Triangular fuzzy number (l, m, u).
    
    Attributes:
        l: Lower bound (minimum possible value)
        m: Modal value (most likely value)
        u: Upper bound (maximum possible value)
    """
    l: float  # Lower bound
    m: float  # Modal value (most likely)
    u: float  # Upper bound
    
    def __post_init__(self):
        if not (self.l <= self.m <= self.u):
            # Auto-correct if out of order
            self.l, self.m, self.u = sorted([self.l, self.m, self.u])
    
    def defuzzify(self, method: str = "centroid") -> float:
        """
        Convert fuzzy number to crisp value.
        
        Methods:
            - centroid: (l + m + u) / 3
            - mom: Mean of maximum = m
            - bisector: (l + 2m + u) / 4
            - graded_mean: (l + 4m + u) / 6
        """
        if method == "centroid":
            return (self.l + self.m + self.u) / 3
        elif method == "mom":
            return self.m
        elif method == "bisector":
            return (self.l + 2*self.m + self.u) / 4
        elif method == "graded_mean":
            return (self.l + 4*self.m + self.u) / 6
        else:
            raise ValueError(f"Unknown defuzzification method: {method}")
    
    def __add__(self, other: 'TriangularFuzzyNumber') -> 'TriangularFuzzyNumber':
        return TriangularFuzzyNumber(
            self.l + other.l,
            self.m + other.m,
            self.u + other.u
        )
    
    def __sub__(self, other: 'TriangularFuzzyNumber') -> 'TriangularFuzzyNumber':
        return TriangularFuzzyNumber(
            self.l - other.u,
            self.m - other.m,
            self.u - other.l
        )
    
    def __mul__(self, other: Union['TriangularFuzzyNumber', float]) -> 'TriangularFuzzyNumber':
        if isinstance(other, TriangularFuzzyNumber):
            # Fuzzy multiplication (approximate)
            products = [
                self.l * other.l, self.l * other.u,
                self.u * other.l, self.u * other.u
            ]
            return TriangularFuzzyNumber(
                min(products),
                self.m * other.m,
                max(products)
            )
        else:
            # Scalar multiplication
            scalar = float(other)
            if scalar >= 0:
                return TriangularFuzzyNumber(
                    self.l * scalar, self.m * scalar, self.u * scalar
                )
            else:
                return TriangularFuzzyNumber(
                    self.u * scalar, self.m * scalar, self.l * scalar
                )
    
    def __rmul__(self, scalar: float) -> 'TriangularFuzzyNumber':
        return self.__mul__(scalar)
    
    def __truediv__(self, other: Union['TriangularFuzzyNumber', float]) -> 'TriangularFuzzyNumber':
        if isinstance(other, TriangularFuzzyNumber):
            # Avoid division by zero
            l_safe = other.l if other.l != 0 else 1e-10
            m_safe = other.m if other.m != 0 else 1e-10
            u_safe = other.u if other.u != 0 else 1e-10
            
            quotients = [
                self.l / l_safe, self.l / u_safe,
                self.u / l_safe, self.u / u_safe
            ]
            return TriangularFuzzyNumber(
                min(quotients),
                self.m / m_safe,
                max(quotients)
            )
        else:
            scalar = float(other) if other != 0 else 1e-10
            if scalar >= 0:
                return TriangularFuzzyNumber(
                    self.l / scalar, self.m / scalar, self.u / scalar
                )
            else:
                return TriangularFuzzyNumber(
                    self.u / scalar, self.m / scalar, self.l / scalar
                )
    
    def distance(self, other: 'TriangularFuzzyNumber') -> float:
        """Vertex distance between two fuzzy numbers."""
        return np.sqrt(
            ((self.l - other.l) ** 2 + 
             (self.m - other.m) ** 2 + 
             (self.u - other.u) ** 2) / 3
        )
    
    def euclidean_distance(self, other: 'TriangularFuzzyNumber') -> float:
        """Euclidean distance between two fuzzy numbers."""
        return np.sqrt(
            (self.l - other.l) ** 2 + 
            (self.m - other.m) ** 2 + 
            (self.u - other.u) ** 2
        )
    
    @staticmethod
    def from_crisp(value: float, spread: float = 0.0) -> 'TriangularFuzzyNumber':
        """Create TFN from crisp value with given spread."""
        return TriangularFuzzyNumber(
            value - spread,
            value,
            value + spread
        )
    
    @staticmethod
    def from_interval(low: float, high: float) -> 'TriangularFuzzyNumber':
        """Create TFN from interval [low, high] with modal at center."""
        return TriangularFuzzyNumber(
            low,
            (low + high) / 2,
            high
        )
    
    def normalize(self, max_val: float) -> 'TriangularFuzzyNumber':
        """Normalize fuzzy number by dividing by max value."""
        if max_val == 0:
            return TriangularFuzzyNumber(0, 0, 0)
        return TriangularFuzzyNumber(
            self.l / max_val,
            self.m / max_val,
            self.u / max_val
        )
    
    def __repr__(self) -> str:
        return f"TFN({self.l:.4f}, {self.m:.4f}, {self.u:.4f})"


class FuzzyDecisionMatrix:
    """
    Container for fuzzy decision matrix operations.
    
    Stores alternatives × criteria matrix of triangular fuzzy numbers.
    """
    
    def __init__(self, 
                 matrix: Dict[str, Dict[str, TriangularFuzzyNumber]],
                 alternatives: List[str],
                 criteria: List[str]):
        self.matrix = matrix
        self.alternatives = alternatives
        self.criteria = criteria
    
    def get(self, alternative: str, criterion: str) -> TriangularFuzzyNumber:
        """Get fuzzy value for alternative and criterion."""
        return self.matrix[alternative][criterion]
    
    def set(self, alternative: str, criterion: str, value: TriangularFuzzyNumber):
        """Set fuzzy value for alternative and criterion."""
        if alternative not in self.matrix:
            self.matrix[alternative] = {}
        self.matrix[alternative][criterion] = value
    
    def to_crisp(self, method: str = "centroid") -> pd.DataFrame:
        """Convert to crisp decision matrix using defuzzification."""
        data = {}
        for alt in self.alternatives:
            data[alt] = {}
            for crit in self.criteria:
                data[alt][crit] = self.matrix[alt][crit].defuzzify(method)
        return pd.DataFrame(data).T
    
    @staticmethod
    def from_crisp_with_uncertainty(
        data: pd.DataFrame,
        uncertainty: Optional[pd.DataFrame] = None,
        default_spread_ratio: float = 0.1
    ) -> 'FuzzyDecisionMatrix':
        """
        Create fuzzy matrix from crisp data with uncertainty.
        
        Parameters:
            data: Crisp decision matrix
            uncertainty: Standard deviation or spread for each cell
            default_spread_ratio: Default spread as ratio of value
        """
        alternatives = list(data.index)
        criteria = list(data.columns)
        matrix = {}
        
        for alt in alternatives:
            matrix[alt] = {}
            for crit in criteria:
                value = data.loc[alt, crit]
                if uncertainty is not None and alt in uncertainty.index and crit in uncertainty.columns:
                    spread = uncertainty.loc[alt, crit]
                else:
                    spread = abs(value) * default_spread_ratio
                
                matrix[alt][crit] = TriangularFuzzyNumber(
                    value - spread,
                    value,
                    value + spread
                )
        
        return FuzzyDecisionMatrix(matrix, alternatives, criteria)
    
    @staticmethod
    def from_panel_temporal_variance(
        panel_data,
        spread_factor: float = 1.0
    ) -> 'FuzzyDecisionMatrix':
        """
        Create fuzzy matrix from panel data using temporal variance.
        
        Uses (mean - std, mean, mean + std) for each entity-criterion pair.
        """
        alternatives = panel_data.provinces
        criteria = panel_data.components
        matrix = {}
        
        for alt in alternatives:
            matrix[alt] = {}
            province_data = panel_data.get_province(alt)
            
            for crit in criteria:
                if crit in province_data.columns:
                    values = province_data[crit].values
                    mean_val = np.mean(values)
                    std_val = np.std(values) * spread_factor
                    
                    matrix[alt][crit] = TriangularFuzzyNumber(
                        mean_val - std_val,
                        mean_val,
                        mean_val + std_val
                    )
                else:
                    matrix[alt][crit] = TriangularFuzzyNumber(0, 0, 0)
        
        return FuzzyDecisionMatrix(matrix, alternatives, criteria)


# Linguistic scales for qualitative fuzzy assessments
LINGUISTIC_SCALE_5 = {
    "very_low": TriangularFuzzyNumber(0.0, 0.0, 0.25),
    "low": TriangularFuzzyNumber(0.0, 0.25, 0.5),
    "medium": TriangularFuzzyNumber(0.25, 0.5, 0.75),
    "high": TriangularFuzzyNumber(0.5, 0.75, 1.0),
    "very_high": TriangularFuzzyNumber(0.75, 1.0, 1.0),
}

LINGUISTIC_SCALE_7 = {
    "very_poor": TriangularFuzzyNumber(0.0, 0.0, 0.167),
    "poor": TriangularFuzzyNumber(0.0, 0.167, 0.333),
    "moderately_poor": TriangularFuzzyNumber(0.167, 0.333, 0.5),
    "fair": TriangularFuzzyNumber(0.333, 0.5, 0.667),
    "moderately_good": TriangularFuzzyNumber(0.5, 0.667, 0.833),
    "good": TriangularFuzzyNumber(0.667, 0.833, 1.0),
    "very_good": TriangularFuzzyNumber(0.833, 1.0, 1.0),
}

# Linguistic scale for importance weights
IMPORTANCE_SCALE = {
    "very_low": TriangularFuzzyNumber(0.0, 0.0, 0.1),
    "low": TriangularFuzzyNumber(0.0, 0.1, 0.3),
    "medium_low": TriangularFuzzyNumber(0.1, 0.3, 0.5),
    "medium": TriangularFuzzyNumber(0.3, 0.5, 0.7),
    "medium_high": TriangularFuzzyNumber(0.5, 0.7, 0.9),
    "high": TriangularFuzzyNumber(0.7, 0.9, 1.0),
    "very_high": TriangularFuzzyNumber(0.9, 1.0, 1.0),
}
