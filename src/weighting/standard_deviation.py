# -*- coding: utf-8 -*-
"""
Standard Deviation weight calculator.

Simple variance-based weighting: w_j = σ_j / Σσ_k
Reference: Wang & Luo (2010), Math & Computer Modelling, 51(1-2), 1-12.
"""

import numpy as np
import pandas as pd
from .base import WeightResult


class StandardDeviationWeightCalculator:
    """
    Standard Deviation weight calculator.
    
    Parameters
    ----------
    epsilon : float, default=1e-10
        Numerical stability constant.
    ddof : int, default=1
        Degrees of freedom for std calculation.
    """
    
    def __init__(self, epsilon: float = 1e-10, ddof: int = 1):
        self.epsilon = epsilon
        self.ddof = ddof
    
    def calculate(self, data: pd.DataFrame) -> WeightResult:
        """
        Calculate standard deviation weights from decision matrix.
        
        Parameters
        ----------
        data : pd.DataFrame
            Decision matrix (alternatives × criteria)
            
        Returns
        -------
        WeightResult
            Calculated weights with std and CV details
            
        Raises
        ------
        ValueError
            If data is empty or has insufficient observations for ddof
        TypeError
            If data contains non-numeric columns
        """
        # Input validation
        if data.empty:
            raise ValueError("Input DataFrame is empty")
        if len(data) <= self.ddof:
            raise ValueError(f"Need more than {self.ddof} observations for ddof={self.ddof}")
        
        non_numeric = data.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric:
            raise TypeError(f"Non-numeric columns found: {non_numeric}")
        
        columns = data.columns.tolist()
        
        # Calculate standard deviation for each criterion
        std = data.std(axis=0, ddof=self.ddof)
        std = std.replace(0, self.epsilon)
        
        # Normalize to weights
        weights = std / std.sum()
        
        # Also compute coefficient of variation (CV) for reference
        # CV = σ/μ measures relative variability
        mean = data.mean(axis=0)
        mean = mean.replace(0, self.epsilon)
        cv = std / mean
        
        # Compute range (max - min) as additional dispersion measure
        data_range = data.max(axis=0) - data.min(axis=0)
        
        return WeightResult(
            weights=weights.to_dict(),
            method="standard_deviation",
            details={
                "std_values": std.to_dict(),
                "coefficient_of_variation": cv.to_dict(),
                "range_values": data_range.to_dict(),
                "mean_values": mean.to_dict(),
                "n_samples": len(data),
                "ddof": self.ddof,
                "interpretation": "Higher weights indicate criteria with more "
                                "variation (dispersion) across alternatives."
            }
        )
