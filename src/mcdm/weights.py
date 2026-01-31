# -*- coding: utf-8 -*-
"""
Weight Calculation Methods: Entropy, CRITIC, and Ensemble
==========================================================
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from scipy.stats import spearmanr

import sys
sys.path.append('..')
from ..config import get_config


@dataclass
class WeightResult:
    """Result container for weight calculations."""
    weights: Dict[str, float]
    method: str
    details: Dict
    
    @property
    def as_array(self) -> np.ndarray:
        return np.array(list(self.weights.values()))
    
    @property
    def as_series(self) -> pd.Series:
        return pd.Series(self.weights)


class EntropyWeightCalculator:
    """Shannon entropy-based objective weight calculation."""
    
    def __init__(self, epsilon: float = 1e-10):
        self.epsilon = epsilon
    
    def calculate(self, data: pd.DataFrame) -> WeightResult:
        """Calculate entropy weights."""
        # Normalize to proportions
        data_norm = data.copy()
        col_sums = data_norm.sum(axis=0)
        col_sums = col_sums.replace(0, self.epsilon)
        
        P = data_norm / col_sums
        P = P.replace(0, self.epsilon)
        
        # Calculate entropy
        n = len(data)
        k = 1 / np.log(n + self.epsilon)
        E = -k * (P * np.log(P + self.epsilon)).sum(axis=0)
        
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


class CRITICWeightCalculator:
    """CRITIC (Criteria Importance Through Inter-criteria Correlation) weights."""
    
    def __init__(self, epsilon: float = 1e-10):
        self.epsilon = epsilon
    
    def calculate(self, data: pd.DataFrame) -> WeightResult:
        """Calculate CRITIC weights."""
        # Standard deviation (contrast intensity)
        std = data.std(axis=0)
        std = std.replace(0, self.epsilon)
        
        # Correlation matrix
        corr_matrix = data.corr()
        
        # Conflict measure
        conflict = (1 - corr_matrix).sum(axis=0)
        
        # Information content
        C = std * conflict
        
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


class EnsembleWeightCalculator:
    """Ensemble weight calculation combining multiple methods."""
    
    def __init__(self, methods: Optional[List[str]] = None, 
                 aggregation: str = "geometric"):
        self.methods = methods or ["entropy", "critic"]
        self.aggregation = aggregation
        self.entropy_calc = EntropyWeightCalculator()
        self.critic_calc = CRITICWeightCalculator()
    
    def calculate(self, data: pd.DataFrame, 
                 method_weights: Optional[Dict[str, float]] = None) -> WeightResult:
        """Calculate ensemble weights."""
        # Get individual weights
        weight_results = {}
        
        if "entropy" in self.methods:
            weight_results["entropy"] = self.entropy_calc.calculate(data)
        
        if "critic" in self.methods:
            weight_results["critic"] = self.critic_calc.calculate(data)
        
        # Default method weights
        if method_weights is None:
            method_weights = {m: 1.0 / len(self.methods) for m in self.methods}
        
        # Aggregate weights
        columns = data.columns.tolist()
        
        if self.aggregation == "arithmetic":
            ensemble_weights = self._arithmetic_mean(weight_results, method_weights, columns)
        elif self.aggregation == "geometric":
            ensemble_weights = self._geometric_mean(weight_results, columns)
        elif self.aggregation == "harmonic":
            ensemble_weights = self._harmonic_mean(weight_results, columns)
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")
        
        # Calculate correlation between methods
        weight_correlation = {}
        methods_list = list(weight_results.keys())
        for i, m1 in enumerate(methods_list):
            for m2 in methods_list[i+1:]:
                w1 = np.array([weight_results[m1].weights[c] for c in columns])
                w2 = np.array([weight_results[m2].weights[c] for c in columns])
                corr, _ = spearmanr(w1, w2)
                weight_correlation[f"{m1}_vs_{m2}"] = corr
        
        return WeightResult(
            weights=ensemble_weights,
            method=f"ensemble_{self.aggregation}",
            details={
                "individual_weights": {m: r.weights for m, r in weight_results.items()},
                "method_weights": method_weights,
                "aggregation": self.aggregation,
                "weight_correlation": weight_correlation
            }
        )
    
    def _arithmetic_mean(self, results: Dict, method_weights: Dict, 
                        columns: List[str]) -> Dict[str, float]:
        """Weighted arithmetic mean."""
        ensemble = {}
        for col in columns:
            weighted_sum = sum(
                method_weights.get(m, 1/len(results)) * r.weights[col]
                for m, r in results.items()
            )
            ensemble[col] = weighted_sum
        
        # Normalize
        total = sum(ensemble.values())
        return {k: v/total for k, v in ensemble.items()}
    
    def _geometric_mean(self, results: Dict, columns: List[str]) -> Dict[str, float]:
        """Geometric mean (equal importance)."""
        ensemble = {}
        n_methods = len(results)
        
        for col in columns:
            product = 1.0
            for r in results.values():
                product *= r.weights[col] ** (1/n_methods)
            ensemble[col] = product
        
        total = sum(ensemble.values())
        return {k: v/total for k, v in ensemble.items()}
    
    def _harmonic_mean(self, results: Dict, columns: List[str]) -> Dict[str, float]:
        """Harmonic mean."""
        ensemble = {}
        n_methods = len(results)
        epsilon = 1e-10
        
        for col in columns:
            harmonic_sum = sum(1/(r.weights[col] + epsilon) for r in results.values())
            ensemble[col] = n_methods / harmonic_sum
        
        total = sum(ensemble.values())
        return {k: v/total for k, v in ensemble.items()}


def calculate_weights(data: pd.DataFrame, method: str = "ensemble") -> WeightResult:
    """Convenience function to calculate weights."""
    if method == "entropy":
        return EntropyWeightCalculator().calculate(data)
    elif method == "critic":
        return CRITICWeightCalculator().calculate(data)
    elif method == "ensemble":
        return EnsembleWeightCalculator().calculate(data)
    elif method == "equal":
        cols = data.columns.tolist()
        w = 1.0 / len(cols)
        return WeightResult(
            weights={c: w for c in cols},
            method="equal",
            details={}
        )
    else:
        raise ValueError(f"Unknown method: {method}")
