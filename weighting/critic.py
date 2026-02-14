# -*- coding: utf-8 -*-
"""
CRITIC Weight Calculator

Criteria Importance Through Inter-criteria Correlation method.
Considers both contrast intensity (standard deviation) and 
inter-criteria correlation to determine weights.

Mathematical Formula:
    w_j = C_j / Σ(C_k)
    
where:
    C_j = σ_j × Σ(1 - r_jk)  [information content]
    σ_j = standard deviation of criterion j
    r_jk = correlation between criteria j and k
"""

import numpy as np
import pandas as pd
from .base import WeightResult


class CRITICWeightCalculator:
    """
    CRITIC (Criteria Importance Through Inter-criteria Correlation) weights.
    
    The CRITIC method considers both:
    1. Contrast Intensity: Standard deviation of criterion values
    2. Conflicting Character: Correlation with other criteria
    
    Criteria with high variation AND low correlation with others
    receive higher weights as they provide unique information.
    
    Parameters
    ----------
    epsilon : float
        Small constant to avoid division by zero
    
    Attributes
    ----------
    epsilon : float
        Numerical stability constant
    
    Examples
    --------
    >>> import pandas as pd
    >>> from weighting import CRITICWeightCalculator
    >>> 
    >>> data = pd.DataFrame({
    ...     'C1': [0.8, 0.6, 0.9, 0.7],
    ...     'C2': [0.75, 0.55, 0.85, 0.65],  # Highly correlated with C1
    ...     'C3': [0.3, 0.9, 0.1, 0.7]       # Uncorrelated - higher weight
    ... })
    >>> 
    >>> calc = CRITICWeightCalculator()
    >>> result = calc.calculate(data)
    >>> print(result.weights)
    
    References
    ----------
    Diakoulaki, D., Mavrotas, G., & Papayannakis, L. (1995).
    Determining objective weights in multiple criteria problems: 
    The CRITIC method. Computers & Operations Research.
    """
    
    def __init__(self, epsilon: float = 1e-10):
        self.epsilon = epsilon
    
    def calculate(self, data: pd.DataFrame) -> WeightResult:
        """
        Calculate CRITIC weights.
        
        Parameters
        ----------
        data : pd.DataFrame
            Decision matrix (alternatives × criteria)
        
        Returns
        -------
        WeightResult
            Calculated weights with standard deviation, conflict, 
            and correlation details
            
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
            raise ValueError("CRITIC calculation requires at least 2 observations")
        
        non_numeric = data.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric:
            raise TypeError(f"Non-numeric columns found: {non_numeric}")
        
        # Standard deviation (contrast intensity)
        std = data.std(axis=0)
        std = std.replace(0, self.epsilon)
        
        # Correlation matrix
        corr_matrix = data.corr()
        
        # Handle NaN values in correlation matrix (occurs with constant columns)
        # Replace NaN with 1.0 (assume perfect self-correlation, zero conflict)
        corr_matrix = corr_matrix.fillna(1.0)
        
        # Conflict measure (sum of 1 - r_jk for all k)
        # Higher values indicate more independence from other criteria
        conflict = (1 - corr_matrix).sum(axis=0)
        
        # Ensure conflict values are non-negative and handle edge cases
        conflict = conflict.clip(lower=self.epsilon)
        
        # Information content (variance × conflict)
        C = std * conflict
        
        # Handle edge case where all C values might be near zero
        if C.sum() < self.epsilon:
            # Fallback to equal weights if no distinguishable information
            n_criteria = len(data.columns)
            weights = pd.Series(1.0 / n_criteria, index=data.columns)
        else:
            # Normalize to weights
            weights = C / C.sum()
        
        return WeightResult(
            weights=weights.to_dict(),
            method="critic",
            details={
                "std_values": std.to_dict(),
                "conflict_values": conflict.to_dict(),
                "information_content": C.to_dict(),
                "correlation_matrix": corr_matrix.to_dict()
            }
        )
