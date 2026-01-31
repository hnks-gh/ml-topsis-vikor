# -*- coding: utf-8 -*-
"""
Fuzzy TOPSIS Implementation
============================

Type-1 Fuzzy TOPSIS with triangular fuzzy numbers.
Supports automatic fuzzy number generation from temporal variance.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

from .weights import WeightResult, EnsembleWeightCalculator


@dataclass
class TriangularFuzzyNumber:
    """Triangular fuzzy number (l, m, u)."""
    l: float  # Lower bound
    m: float  # Modal value (most likely)
    u: float  # Upper bound
    
    def __post_init__(self):
        if not (self.l <= self.m <= self.u):
            # Auto-correct if out of order
            self.l, self.m, self.u = sorted([self.l, self.m, self.u])
    
    def defuzzify(self, method: str = "centroid") -> float:
        """Convert fuzzy number to crisp value."""
        if method == "centroid":
            return (self.l + self.m + self.u) / 3
        elif method == "mom":  # Mean of maximum
            return self.m
        elif method == "bisector":
            return (self.l + 2*self.m + self.u) / 4
        else:
            raise ValueError(f"Unknown method: {method}")
    
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
    
    def __mul__(self, scalar: float) -> 'TriangularFuzzyNumber':
        if scalar >= 0:
            return TriangularFuzzyNumber(
                self.l * scalar, self.m * scalar, self.u * scalar
            )
        else:
            return TriangularFuzzyNumber(
                self.u * scalar, self.m * scalar, self.l * scalar
            )
    
    def distance(self, other: 'TriangularFuzzyNumber') -> float:
        """Vertex distance between two fuzzy numbers."""
        return np.sqrt(
            ((self.l - other.l) ** 2 + 
             (self.m - other.m) ** 2 + 
             (self.u - other.u) ** 2) / 3
        )


@dataclass
class FuzzyTOPSISResult:
    """Result container for Fuzzy TOPSIS calculation."""
    scores: pd.Series                    # Closeness coefficients
    ranks: pd.Series                     # Final rankings
    d_positive: pd.Series                # Distance to fuzzy ideal
    d_negative: pd.Series                # Distance to fuzzy anti-ideal
    crisp_equivalent: pd.DataFrame       # Defuzzified decision matrix
    fuzzy_matrix: Dict                   # Original fuzzy decision matrix
    fuzzy_weights: Dict[str, TriangularFuzzyNumber]
    
    def top_n(self, n: int = 10) -> pd.DataFrame:
        """Get top n alternatives."""
        return pd.DataFrame({
            'Score': self.scores,
            'Rank': self.ranks
        }).nsmallest(n, 'Rank')


class FuzzyTOPSIS:
    """
    Fuzzy TOPSIS with triangular fuzzy numbers.
    
    Supports automatic generation of fuzzy numbers from:
    - Temporal variance (for panel data)
    - Linguistic scales
    - Custom bounds
    """
    
    def __init__(self,
                 use_temporal_variance: bool = True,
                 spread_factor: float = 1.0,
                 defuzzification: str = "centroid",
                 benefit_criteria: Optional[List[str]] = None,
                 cost_criteria: Optional[List[str]] = None):
        self.use_temporal_variance = use_temporal_variance
        self.spread_factor = spread_factor
        self.defuzzification = defuzzification
        self.benefit_criteria = benefit_criteria
        self.cost_criteria = cost_criteria or []
    
    def calculate_from_panel(self,
                            panel_data,
                            weights: Union[Dict[str, float], WeightResult, None] = None
                            ) -> FuzzyTOPSISResult:
        """
        Calculate Fuzzy TOPSIS using panel data temporal variance.
        
        Fuzzy numbers generated as: (mean - std, mean, mean + std)
        """
        # Generate fuzzy decision matrix from temporal variance
        fuzzy_matrix = self._generate_fuzzy_from_panel(panel_data)
        
        # Get weights
        if weights is None:
            latest_data = panel_data.get_latest()
            weight_calc = EnsembleWeightCalculator()
            weight_result = weight_calc.calculate(latest_data)
            weights = weight_result.weights
        elif isinstance(weights, WeightResult):
            weights = weights.weights
        
        # Convert to fuzzy weights (crisp weights as TFN with zero spread)
        fuzzy_weights = {
            col: TriangularFuzzyNumber(w, w, w)
            for col, w in weights.items()
        }
        
        return self._calculate_fuzzy_topsis(
            fuzzy_matrix, fuzzy_weights, panel_data.provinces
        )
    
    def calculate_from_crisp(self,
                            data: pd.DataFrame,
                            uncertainty: Optional[pd.DataFrame] = None,
                            weights: Union[Dict[str, float], WeightResult, None] = None
                            ) -> FuzzyTOPSISResult:
        """
        Calculate Fuzzy TOPSIS from crisp data with optional uncertainty.
        
        If uncertainty not provided, uses 10% of value as spread.
        """
        # Generate fuzzy matrix
        fuzzy_matrix = {}
        
        for alt in data.index:
            fuzzy_matrix[alt] = {}
            for col in data.columns:
                value = data.loc[alt, col]
                
                if uncertainty is not None and col in uncertainty.columns:
                    spread = uncertainty.loc[alt, col] * self.spread_factor
                else:
                    spread = abs(value) * 0.1 * self.spread_factor
                
                fuzzy_matrix[alt][col] = TriangularFuzzyNumber(
                    max(0, value - spread),
                    value,
                    min(1, value + spread) if value <= 1 else value + spread
                )
        
        # Get weights
        if weights is None:
            weight_calc = EnsembleWeightCalculator()
            weight_result = weight_calc.calculate(data)
            weights = weight_result.weights
        elif isinstance(weights, WeightResult):
            weights = weights.weights
        
        fuzzy_weights = {
            col: TriangularFuzzyNumber(w, w, w)
            for col, w in weights.items()
        }
        
        return self._calculate_fuzzy_topsis(
            fuzzy_matrix, fuzzy_weights, data.index.tolist()
        )
    
    def _generate_fuzzy_from_panel(self, panel_data) -> Dict:
        """Generate fuzzy numbers from panel data temporal statistics."""
        fuzzy_matrix = {}
        
        for province in panel_data.provinces:
            province_data = panel_data.get_province(province)
            fuzzy_matrix[province] = {}
            
            for col in panel_data.components:
                values = province_data[col].values
                mean_val = np.mean(values)
                std_val = np.std(values) * self.spread_factor
                
                # Clip to [0, 1] for normalized data
                l = max(0, mean_val - std_val)
                u = min(1, mean_val + std_val)
                
                fuzzy_matrix[province][col] = TriangularFuzzyNumber(l, mean_val, u)
        
        return fuzzy_matrix
    
    def _calculate_fuzzy_topsis(self,
                               fuzzy_matrix: Dict,
                               fuzzy_weights: Dict[str, TriangularFuzzyNumber],
                               alternatives: List[str]
                               ) -> FuzzyTOPSISResult:
        """Core Fuzzy TOPSIS calculation."""
        columns = list(fuzzy_weights.keys())
        
        # Step 1: Normalize fuzzy decision matrix
        normalized = self._normalize_fuzzy(fuzzy_matrix, alternatives, columns)
        
        # Step 2: Calculate weighted normalized fuzzy matrix
        weighted = self._apply_fuzzy_weights(normalized, fuzzy_weights, 
                                            alternatives, columns)
        
        # Step 3: Determine fuzzy ideal solutions
        f_ideal, f_anti_ideal = self._get_fuzzy_ideals(weighted, alternatives, columns)
        
        # Step 4: Calculate distances
        d_positive = {}
        d_negative = {}
        
        for alt in alternatives:
            d_pos = sum(
                weighted[alt][col].distance(f_ideal[col])
                for col in columns
            )
            d_neg = sum(
                weighted[alt][col].distance(f_anti_ideal[col])
                for col in columns
            )
            d_positive[alt] = d_pos
            d_negative[alt] = d_neg
        
        # Step 5: Calculate closeness coefficient
        scores = {}
        for alt in alternatives:
            d_neg = d_negative[alt]
            d_pos = d_positive[alt]
            scores[alt] = d_neg / (d_pos + d_neg + 1e-10)
        
        scores_series = pd.Series(scores, name='Fuzzy_TOPSIS_Score')
        ranks = scores_series.rank(ascending=False).astype(int)
        ranks.name = 'Fuzzy_TOPSIS_Rank'
        
        # Create crisp equivalent for reference
        crisp_data = {}
        for alt in alternatives:
            crisp_data[alt] = {
                col: fuzzy_matrix[alt][col].defuzzify(self.defuzzification)
                for col in columns
            }
        crisp_df = pd.DataFrame(crisp_data).T
        
        # Convert crisp weights for output
        crisp_weights = {col: w.m for col, w in fuzzy_weights.items()}
        
        return FuzzyTOPSISResult(
            scores=scores_series,
            ranks=ranks,
            d_positive=pd.Series(d_positive),
            d_negative=pd.Series(d_negative),
            crisp_equivalent=crisp_df,
            fuzzy_matrix=fuzzy_matrix,
            fuzzy_weights=fuzzy_weights
        )
    
    def _normalize_fuzzy(self, fuzzy_matrix: Dict, 
                        alternatives: List[str],
                        columns: List[str]) -> Dict:
        """Normalize fuzzy decision matrix."""
        normalized = {alt: {} for alt in alternatives}
        
        for col in columns:
            # Get max upper bound for normalization
            if col in self.cost_criteria:
                # For cost: normalize by min lower bound
                min_l = min(fuzzy_matrix[alt][col].l for alt in alternatives)
                min_l = max(min_l, 1e-10)
                
                for alt in alternatives:
                    tfn = fuzzy_matrix[alt][col]
                    normalized[alt][col] = TriangularFuzzyNumber(
                        min_l / tfn.u if tfn.u > 0 else 0,
                        min_l / tfn.m if tfn.m > 0 else 0,
                        min_l / tfn.l if tfn.l > 0 else 1
                    )
            else:
                # For benefit: normalize by max upper bound
                max_u = max(fuzzy_matrix[alt][col].u for alt in alternatives)
                max_u = max(max_u, 1e-10)
                
                for alt in alternatives:
                    tfn = fuzzy_matrix[alt][col]
                    normalized[alt][col] = TriangularFuzzyNumber(
                        tfn.l / max_u,
                        tfn.m / max_u,
                        tfn.u / max_u
                    )
        
        return normalized
    
    def _apply_fuzzy_weights(self, normalized: Dict,
                            fuzzy_weights: Dict[str, TriangularFuzzyNumber],
                            alternatives: List[str],
                            columns: List[str]) -> Dict:
        """Apply fuzzy weights to normalized matrix."""
        weighted = {alt: {} for alt in alternatives}
        
        for alt in alternatives:
            for col in columns:
                tfn = normalized[alt][col]
                w = fuzzy_weights[col]
                
                # Fuzzy multiplication
                weighted[alt][col] = TriangularFuzzyNumber(
                    tfn.l * w.l,
                    tfn.m * w.m,
                    tfn.u * w.u
                )
        
        return weighted
    
    def _get_fuzzy_ideals(self, weighted: Dict,
                         alternatives: List[str],
                         columns: List[str]
                         ) -> Tuple[Dict, Dict]:
        """Determine fuzzy ideal and anti-ideal solutions."""
        f_ideal = {}
        f_anti_ideal = {}
        
        for col in columns:
            all_tfns = [weighted[alt][col] for alt in alternatives]
            
            if col in self.cost_criteria:
                f_ideal[col] = TriangularFuzzyNumber(
                    min(t.l for t in all_tfns),
                    min(t.m for t in all_tfns),
                    min(t.u for t in all_tfns)
                )
                f_anti_ideal[col] = TriangularFuzzyNumber(
                    max(t.l for t in all_tfns),
                    max(t.m for t in all_tfns),
                    max(t.u for t in all_tfns)
                )
            else:
                f_ideal[col] = TriangularFuzzyNumber(
                    max(t.l for t in all_tfns),
                    max(t.m for t in all_tfns),
                    max(t.u for t in all_tfns)
                )
                f_anti_ideal[col] = TriangularFuzzyNumber(
                    min(t.l for t in all_tfns),
                    min(t.m for t in all_tfns),
                    min(t.u for t in all_tfns)
                )
        
        return f_ideal, f_anti_ideal
