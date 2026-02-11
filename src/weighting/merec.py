# -*- coding: utf-8 -*-
"""
MEREC (Method based on Removal Effects of Criteria) weight calculator.

Measures criterion importance by impact of removal on overall performance.
Reference: Keshavarz-Ghorabaee et al. (2021), Symmetry, 13(4), 525.
"""

import numpy as np
import pandas as pd
from .base import WeightResult


class MERECWeightCalculator:
    """
    MEREC weight calculator.
    
    Parameters
    ----------
    epsilon : float, default=1e-10
        Numerical stability constant.
    """
    
    def __init__(self, epsilon: float = 1e-10):
        self.epsilon = epsilon
    
    def calculate(self, data: pd.DataFrame) -> WeightResult:
        """Calculate MEREC weights from decision matrix."""
        n_alternatives, n_criteria = data.shape
        columns = data.columns.tolist()
        
        # Step 1: Min-Max Normalization to [0, 1]
        X_norm = self._normalize(data.values)
        
        # Step 2: Calculate overall performance with all criteria
        # S_i = ln(1 + (1/n)Σ|ln(x_ij)|)
        S_overall = self._calculate_performance(X_norm)
        
        # Step 3-4: Calculate removal effect for each criterion
        removal_effects = np.zeros(n_criteria)
        S_without = {}  # Store for details
        
        for j in range(n_criteria):
            # Remove criterion j (set to 1 in normalized space = neutral)
            X_removed = X_norm.copy()
            X_removed[:, j] = 1.0
            
            # Calculate performance without criterion j
            S_j = self._calculate_performance(X_removed)
            S_without[columns[j]] = S_j
            
            # Removal effect: sum of absolute log differences
            # E_j = Σ|ln(S_i^j) - ln(S_i)|
            removal_effects[j] = np.sum(np.abs(
                np.log(S_j + self.epsilon) - np.log(S_overall + self.epsilon)
            ))
        
        # Step 5: Normalize to weights
        # Ensure no zero removal effects
        removal_effects = np.clip(removal_effects, self.epsilon, None)
        weights = removal_effects / removal_effects.sum()
        
        return WeightResult(
            weights={col: float(weights[j]) for j, col in enumerate(columns)},
            method="merec",
            details={
                "removal_effects": {col: float(removal_effects[j]) 
                                   for j, col in enumerate(columns)},
                "overall_performance": S_overall.tolist(),
                "n_alternatives": n_alternatives,
                "n_criteria": n_criteria,
                "interpretation": "Higher weights indicate criteria whose removal "
                                "significantly impacts alternative performance rankings."
            }
        )
    
    def _normalize(self, X: np.ndarray) -> np.ndarray:
        """Min-max normalization to [epsilon, 1]."""
        X_min = X.min(axis=0)
        X_max = X.max(axis=0)
        
        denominator = X_max - X_min
        denominator[denominator < self.epsilon] = self.epsilon
        
        # Normalize to [0, 1] then shift to [epsilon, 1]
        X_norm = (X - X_min) / denominator
        X_norm = X_norm * (1 - self.epsilon) + self.epsilon
        
        return X_norm
    
    def _calculate_performance(self, X_norm: np.ndarray) -> np.ndarray:
        """Calculate overall performance: S_i = ln(1 + (1/n)Σ|ln(x_ij)|)."""
        n_alternatives, n_criteria = X_norm.shape
        
        # Ensure all values are positive for logarithm
        X_safe = np.clip(X_norm, self.epsilon, None)
        
        # Calculate S_i for each alternative
        log_sum = np.sum(np.abs(np.log(X_safe)), axis=1) / n_criteria
        S = np.log(1 + log_sum)
        
        return S
