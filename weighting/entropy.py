# -*- coding: utf-8 -*-
"""
Entropy Weight Calculator

Shannon entropy-based objective weight calculation method.
Assigns higher weights to criteria with more variation (information content).

Mathematical Formula:
    w_j = (1 - E_j) / Σ(1 - E_k)
    
where:
    E_j = -k × Σ(p_ij × ln(p_ij))  [entropy of criterion j]
    k = 1 / ln(m)  [normalization constant]
    p_ij = x_ij / Σx_ij  [proportion of value]
"""

import numpy as np
import pandas as pd
from .base import WeightResult


class EntropyWeightCalculator:
    """
    Shannon entropy-based objective weight calculation.
    
    The entropy method assigns higher weights to criteria that have 
    more variation across alternatives, as these provide more 
    information for distinguishing between options.
    
    Parameters
    ----------
    epsilon : float
        Small constant to avoid log(0) and division by zero
    
    Attributes
    ----------
    epsilon : float
        Numerical stability constant
    
    Examples
    --------
    >>> import pandas as pd
    >>> from weighting import EntropyWeightCalculator
    >>> 
    >>> data = pd.DataFrame({
    ...     'C1': [0.8, 0.6, 0.9, 0.7],
    ...     'C2': [0.5, 0.5, 0.5, 0.5],  # No variation - low weight
    ...     'C3': [0.3, 0.9, 0.1, 0.7]   # High variation - high weight
    ... })
    >>> 
    >>> calc = EntropyWeightCalculator()
    >>> result = calc.calculate(data)
    >>> print(result.weights)
    
    References
    ----------
    Shannon, C.E. (1948). A Mathematical Theory of Communication. 
    Bell System Technical Journal.
    """
    
    def __init__(self, epsilon: float = 1e-10):
        self.epsilon = epsilon
    
    def calculate(self, data: pd.DataFrame) -> WeightResult:
        """
        Calculate entropy weights.
        
        Parameters
        ----------
        data : pd.DataFrame
            Decision matrix (alternatives × criteria)
            All values should be positive
        
        Returns
        -------
        WeightResult
            Calculated weights with entropy and divergence details
            
        Raises
        ------
        ValueError
            If data is empty or has less than 2 observations
        TypeError
            If data contains non-numeric columns
        """
        # Input validation
        if data.empty:
            raise ValueError("Input DataFrame is empty")
        if len(data) < 2:
            raise ValueError("Entropy calculation requires at least 2 observations")
        
        # Check for non-numeric columns
        non_numeric = data.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric:
            raise TypeError(f"Non-numeric columns found: {non_numeric}")
        
        # Validate input: entropy requires non-negative values
        # If negative values exist, shift data to positive range
        data_valid = data.copy()
        for col in data_valid.columns:
            if data_valid[col].min() < 0:
                # Shift column to positive range: x' = x - min(x) + epsilon
                data_valid[col] = data_valid[col] - data_valid[col].min() + self.epsilon
        
        # Normalize to proportions
        col_sums = data_valid.sum(axis=0)
        # Handle edge case of zero or near-zero sums
        col_sums = col_sums.clip(lower=self.epsilon)
        
        P = data_valid / col_sums
        P = P.clip(lower=self.epsilon)  # Ensure all proportions are positive
        
        # Calculate entropy
        n = len(data)
        if n < 2:
            raise ValueError("Entropy calculation requires at least 2 observations")
        
        k = 1 / np.log(n)
        # P is already >= epsilon from clipping, so log(P) is safe
        E = -k * (P * np.log(P)).sum(axis=0)
        
        # Calculate divergence (information content)
        D = 1 - E
        D = D.clip(lower=self.epsilon)
        
        # Normalize to weights
        weights = D / D.sum()
        
        return WeightResult(
            weights=weights.to_dict(),
            method="entropy",
            details={
                "entropy_values": E.to_dict(),
                "divergence_values": D.to_dict(),
                "n_samples": n
            }
        )
