# -*- coding: utf-8 -*-
"""
EDAS Implementation
====================

EDAS (Evaluation based on Distance from Average Solution)
MCDM method based on distance from average solution rather than ideal solutions.

Advantages over TOPSIS:
- Uses average solution (more robust to outliers)
- Separate positive/negative distances for comprehensive evaluation
- Better handling of criteria with different scales
- Simpler computation while maintaining accuracy

Features:
- Standard EDAS for cross-sectional data
- Multi-Period EDAS for panel data
- EDAS with interval/uncertain data support
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from scipy.stats import spearmanr

from .weights import WeightResult, EnsembleWeightCalculator


@dataclass
class EDASResult:
    """Result container for EDAS calculation."""
    PDA: pd.DataFrame              # Positive Distance from Average
    NDA: pd.DataFrame              # Negative Distance from Average
    SP: pd.Series                  # Weighted sum of PDA
    SN: pd.Series                  # Weighted sum of NDA
    NSP: pd.Series                 # Normalized SP
    NSN: pd.Series                 # Normalized SN
    AS: pd.Series                  # Appraisal Score
    ranks: pd.Series               # Final rankings
    average_solution: pd.Series    # Average solution values
    weights: Dict[str, float]
    
    @property
    def final_ranks(self) -> pd.Series:
        """Get final rankings."""
        return self.ranks
    
    def top_n(self, n: int = 10) -> pd.DataFrame:
        """Get top n alternatives."""
        return pd.DataFrame({
            'SP': self.SP,
            'SN': self.SN,
            'NSP': self.NSP,
            'NSN': self.NSN,
            'AS': self.AS,
            'Rank': self.ranks
        }).nsmallest(n, 'Rank')
    
    def summary(self) -> str:
        lines = [
            f"\n{'='*60}",
            "EDAS RESULTS",
            f"{'='*60}",
            f"\nAlternatives: {len(self.ranks)}",
            f"Criteria: {len(self.average_solution)}",
            f"\nAverage Solution:",
        ]
        for crit, val in self.average_solution.items():
            lines.append(f"  {crit}: {val:.4f}")
        lines.append(f"\nTop 10 Alternatives:")
        top10 = self.top_n(10)
        for i, (idx, row) in enumerate(top10.iterrows(), 1):
            lines.append(f"  {i}. {idx}: AS={row['AS']:.4f}")
        lines.append("=" * 60)
        return "\n".join(lines)


class EDASCalculator:
    """
    EDAS (Evaluation based on Distance from Average Solution) calculator.
    
    Unlike TOPSIS which uses ideal solutions, EDAS uses the average solution
    as a reference point. This makes it more robust to outliers and extreme values.
    
    Parameters
    ----------
    benefit_criteria : List[str]
        List of benefit criteria (higher is better)
    cost_criteria : List[str]
        List of cost criteria (lower is better)
    """
    
    def __init__(self,
                 benefit_criteria: Optional[List[str]] = None,
                 cost_criteria: Optional[List[str]] = None):
        self.benefit_criteria = benefit_criteria
        self.cost_criteria = cost_criteria or []
    
    def calculate(self,
                 data: pd.DataFrame,
                 weights: Union[Dict[str, float], WeightResult, None] = None
                 ) -> EDASResult:
        """
        Calculate EDAS scores and rankings.
        
        Parameters
        ----------
        data : pd.DataFrame
            Decision matrix (alternatives × criteria)
        weights : Dict or WeightResult
            Criteria weights
        
        Returns
        -------
        EDASResult
            Complete EDAS results
        """
        # Get weights
        if weights is None:
            weight_calc = EnsembleWeightCalculator()
            weight_result = weight_calc.calculate(data)
            weights = weight_result.weights
        elif isinstance(weights, WeightResult):
            weights = weights.weights
        
        weights = {col: weights.get(col, 1/len(data.columns)) 
                  for col in data.columns}
        
        # Determine benefit and cost criteria
        if self.benefit_criteria is None:
            self.benefit_criteria = [col for col in data.columns 
                                     if col not in self.cost_criteria]
        
        # Step 1: Calculate Average Solution (AV)
        average_solution = data.mean()
        
        # Step 2: Calculate PDA (Positive Distance from Average) 
        #         and NDA (Negative Distance from Average)
        PDA = pd.DataFrame(index=data.index, columns=data.columns, dtype=float)
        NDA = pd.DataFrame(index=data.index, columns=data.columns, dtype=float)
        
        for col in data.columns:
            av = average_solution[col]
            
            if col in self.benefit_criteria:
                # For benefit criteria:
                # PDA = max(0, x_ij - AV_j) / AV_j
                # NDA = max(0, AV_j - x_ij) / AV_j
                if av > 0:
                    PDA[col] = np.maximum(0, data[col] - av) / av
                    NDA[col] = np.maximum(0, av - data[col]) / av
                else:
                    PDA[col] = np.maximum(0, data[col] - av)
                    NDA[col] = np.maximum(0, av - data[col])
            else:
                # For cost criteria:
                # PDA = max(0, AV_j - x_ij) / AV_j
                # NDA = max(0, x_ij - AV_j) / AV_j
                if av > 0:
                    PDA[col] = np.maximum(0, av - data[col]) / av
                    NDA[col] = np.maximum(0, data[col] - av) / av
                else:
                    PDA[col] = np.maximum(0, av - data[col])
                    NDA[col] = np.maximum(0, data[col] - av)
        
        # Step 3: Calculate weighted sum of PDA and NDA
        SP = pd.Series(0.0, index=data.index)
        SN = pd.Series(0.0, index=data.index)
        
        for col in data.columns:
            w = weights[col]
            SP += w * PDA[col]
            SN += w * NDA[col]
        
        SP.name = 'SP'
        SN.name = 'SN'
        
        # Step 4: Normalize SP and SN
        SP_max = SP.max()
        SN_max = SN.max()
        
        NSP = SP / SP_max if SP_max > 0 else pd.Series(0, index=data.index)
        NSN = 1 - (SN / SN_max) if SN_max > 0 else pd.Series(1, index=data.index)
        
        NSP.name = 'NSP'
        NSN.name = 'NSN'
        
        # Step 5: Calculate Appraisal Score (AS)
        AS = (NSP + NSN) / 2
        AS.name = 'AS'
        
        # Step 6: Rank alternatives
        ranks = AS.rank(ascending=False).astype(int)
        ranks.name = 'Rank'
        
        return EDASResult(
            PDA=PDA,
            NDA=NDA,
            SP=SP,
            SN=SN,
            NSP=NSP,
            NSN=NSN,
            AS=AS,
            ranks=ranks,
            average_solution=average_solution,
            weights=weights
        )


@dataclass
class MultiPeriodEDASResult:
    """Result container for Multi-Period EDAS."""
    yearly_results: Dict[int, EDASResult]
    aggregated_AS: pd.DataFrame        # Entity × Year matrix
    average_evolution: pd.DataFrame    # How average solution changes over time
    trend_scores: pd.Series            # Improvement trends
    stability_scores: pd.Series        # Score stability over time
    final_ranking: pd.Series
    composite_scores: pd.Series
    distance_from_avg_trend: pd.DataFrame  # Are entities moving toward/away from avg?
    
    def summary(self) -> str:
        lines = [
            f"\n{'='*60}",
            "MULTI-PERIOD EDAS RESULTS",
            f"{'='*60}",
            f"\nYears analyzed: {list(self.yearly_results.keys())}",
            f"Entities: {len(self.final_ranking)}",
            f"\nTop 10 Final Ranking:"
        ]
        top10 = self.final_ranking.head(10)
        for i, (entity, rank) in enumerate(top10.items(), 1):
            score = self.composite_scores.get(entity, 0)
            lines.append(f"  {i}. {entity} (Rank: {rank}, Score: {score:.4f})")
        lines.append("=" * 60)
        return "\n".join(lines)


class MultiPeriodEDAS:
    """
    Multi-Period EDAS for panel data analysis.
    
    Extends EDAS to handle temporal dynamics with:
    - Yearly EDAS analysis with evolving average solutions
    - Temporal aggregation of appraisal scores
    - Trend analysis (improving vs declining vs average)
    - Stability analysis
    - Tracking distance from temporal average
    """
    
    def __init__(self,
                 temporal_discount: float = 0.9,
                 trend_weight: float = 0.3,
                 stability_weight: float = 0.2,
                 use_global_average: bool = False,
                 benefit_criteria: Optional[List[str]] = None,
                 cost_criteria: Optional[List[str]] = None):
        """
        Parameters
        ----------
        temporal_discount : float
            Discount factor for older years (0.9 means recent years weighted more)
        trend_weight : float
            Weight given to improvement trend in composite score
        stability_weight : float
            Weight given to stability in composite score
        use_global_average : bool
            If True, uses global average across all years; if False, uses yearly average
        """
        self.temporal_discount = temporal_discount
        self.trend_weight = trend_weight
        self.stability_weight = stability_weight
        self.use_global_average = use_global_average
        self.benefit_criteria = benefit_criteria
        self.cost_criteria = cost_criteria
        self.calculator = EDASCalculator(
            benefit_criteria=benefit_criteria,
            cost_criteria=cost_criteria
        )
    
    def calculate(self,
                 panel_data,
                 weights: Union[Dict[str, float], WeightResult, None] = None
                 ) -> MultiPeriodEDASResult:
        """
        Calculate Multi-Period EDAS rankings.
        
        Parameters
        ----------
        panel_data : PanelData
            Panel data object with cross_section dictionary
        weights : Dict or WeightResult
            Criteria weights
        """
        yearly_results = {}
        AS_matrix = {}
        avg_matrix = {}
        
        years = sorted(panel_data.years)
        
        # If using global average, compute it first
        global_avg = None
        if self.use_global_average:
            all_data = []
            for year in years:
                year_data = panel_data.cross_section[year]
                year_data = year_data.set_index('Province') if 'Province' in year_data.columns else year_data
                numeric_cols = [col for col in year_data.columns 
                               if col in panel_data.components]
                all_data.append(year_data[numeric_cols])
            global_avg = pd.concat(all_data).mean()
        
        # Calculate EDAS for each year
        for year in years:
            year_data = panel_data.cross_section[year]
            year_data = year_data.set_index('Province') if 'Province' in year_data.columns else year_data
            
            # Keep only numeric columns (components)
            numeric_cols = [col for col in year_data.columns 
                          if col in panel_data.components]
            year_data = year_data[numeric_cols]
            
            # Update benefit criteria if not set
            if self.benefit_criteria is None:
                self.calculator.benefit_criteria = [col for col in numeric_cols 
                                                   if col not in (self.cost_criteria or [])]
            
            result = self.calculator.calculate(year_data, weights)
            yearly_results[year] = result
            
            AS_matrix[year] = result.AS
            avg_matrix[year] = result.average_solution
        
        # Create DataFrames
        AS_df = pd.DataFrame(AS_matrix)
        avg_evolution = pd.DataFrame(avg_matrix)
        
        # Calculate distance from average trend
        distance_trend = self._calculate_distance_trend(panel_data, avg_evolution, years)
        
        # Calculate trend scores
        trend_scores = self._calculate_trend_scores(AS_df, years)
        
        # Calculate stability scores
        stability_scores = self._calculate_stability_scores(AS_df, years)
        
        # Calculate composite scores
        composite_scores = self._calculate_composite_scores(
            AS_df, trend_scores, stability_scores, years
        )
        
        # Final ranking
        final_ranking = composite_scores.rank(ascending=False).astype(int)
        final_ranking = final_ranking.sort_values()
        
        return MultiPeriodEDASResult(
            yearly_results=yearly_results,
            aggregated_AS=AS_df,
            average_evolution=avg_evolution,
            trend_scores=trend_scores,
            stability_scores=stability_scores,
            final_ranking=final_ranking,
            composite_scores=composite_scores,
            distance_from_avg_trend=distance_trend
        )
    
    def _calculate_distance_trend(self,
                                   panel_data,
                                   avg_evolution: pd.DataFrame,
                                   years: List[int]) -> pd.DataFrame:
        """Calculate how each entity's distance from average changes over time."""
        distance_data = {}
        
        for year in years:
            year_data = panel_data.cross_section[year]
            year_data = year_data.set_index('Province') if 'Province' in year_data.columns else year_data
            
            numeric_cols = [col for col in year_data.columns 
                          if col in panel_data.components]
            year_data = year_data[numeric_cols]
            
            avg = avg_evolution[year]
            
            # Calculate Euclidean distance from average for each entity
            distances = {}
            for entity in year_data.index:
                entity_values = year_data.loc[entity, numeric_cols]
                dist = np.sqrt(((entity_values - avg[numeric_cols]) ** 2).sum())
                distances[entity] = dist
            
            distance_data[year] = pd.Series(distances)
        
        return pd.DataFrame(distance_data)
    
    def _calculate_trend_scores(self,
                                AS_df: pd.DataFrame,
                                years: List[int]) -> pd.Series:
        """Calculate improvement trends for each entity."""
        trends = {}
        for entity in AS_df.index:
            values = AS_df.loc[entity, years].values
            if len(values) > 1:
                x = np.arange(len(values))
                slope = np.polyfit(x, values, 1)[0]
                val_range = values.max() - values.min()
                trends[entity] = slope / (val_range + 1e-10)
            else:
                trends[entity] = 0.0
        return pd.Series(trends, name='Trend')
    
    def _calculate_stability_scores(self,
                                    AS_df: pd.DataFrame,
                                    years: List[int]) -> pd.Series:
        """Calculate appraisal score stability over time."""
        AS_std = AS_df[years].std(axis=1)
        max_std = AS_std.max() if AS_std.max() > 0 else 1
        stability = 1 - (AS_std / max_std)
        stability.name = 'Stability'
        return stability
    
    def _calculate_composite_scores(self,
                                    AS_df: pd.DataFrame,
                                    trend_scores: pd.Series,
                                    stability_scores: pd.Series,
                                    years: List[int]) -> pd.Series:
        """Calculate composite scores with temporal weighting."""
        n_years = len(years)
        temporal_weights = np.array([self.temporal_discount ** (n_years - 1 - i) 
                                     for i in range(n_years)])
        temporal_weights /= temporal_weights.sum()
        
        weighted_AS = AS_df[years].values @ temporal_weights
        weighted_AS_series = pd.Series(weighted_AS, index=AS_df.index)
        
        # Normalize
        def normalize(s):
            min_val, max_val = s.min(), s.max()
            if max_val - min_val > 0:
                return (s - min_val) / (max_val - min_val)
            return pd.Series(0.5, index=s.index)
        
        AS_norm = normalize(weighted_AS_series)
        trend_norm = normalize(trend_scores)
        
        base_weight = 1 - self.trend_weight - self.stability_weight
        composite = (base_weight * AS_norm + 
                    self.trend_weight * trend_norm + 
                    self.stability_weight * stability_scores)
        composite.name = 'Composite_Score'
        
        return composite


class ModifiedEDAS:
    """
    Modified EDAS with additional features for robust analysis.
    
    Extensions:
    - Interval-valued EDAS for uncertain data
    - Robust EDAS using trimmed mean as reference
    - Weighted average solution options
    """
    
    def __init__(self,
                 benefit_criteria: Optional[List[str]] = None,
                 cost_criteria: Optional[List[str]] = None,
                 use_trimmed_mean: bool = False,
                 trim_percentage: float = 0.1,
                 use_weighted_average: bool = False):
        self.benefit_criteria = benefit_criteria
        self.cost_criteria = cost_criteria or []
        self.use_trimmed_mean = use_trimmed_mean
        self.trim_percentage = trim_percentage
        self.use_weighted_average = use_weighted_average
    
    def calculate(self,
                 data: pd.DataFrame,
                 weights: Union[Dict[str, float], WeightResult, None] = None
                 ) -> EDASResult:
        """
        Calculate Modified EDAS scores and rankings.
        
        Uses trimmed mean or weighted average as the reference solution
        for more robust results.
        """
        # Get weights
        if weights is None:
            weight_calc = EnsembleWeightCalculator()
            weight_result = weight_calc.calculate(data)
            weights = weight_result.weights
        elif isinstance(weights, WeightResult):
            weights = weights.weights
        
        weights = {col: weights.get(col, 1/len(data.columns)) 
                  for col in data.columns}
        
        if self.benefit_criteria is None:
            self.benefit_criteria = [col for col in data.columns 
                                     if col not in self.cost_criteria]
        
        # Calculate reference solution based on method
        if self.use_trimmed_mean:
            average_solution = self._trimmed_mean(data)
        elif self.use_weighted_average:
            average_solution = self._weighted_average(data, weights)
        else:
            average_solution = data.mean()
        
        # Continue with standard EDAS calculation using modified average
        PDA = pd.DataFrame(index=data.index, columns=data.columns, dtype=float)
        NDA = pd.DataFrame(index=data.index, columns=data.columns, dtype=float)
        
        for col in data.columns:
            av = average_solution[col]
            
            if col in self.benefit_criteria:
                if av > 0:
                    PDA[col] = np.maximum(0, data[col] - av) / av
                    NDA[col] = np.maximum(0, av - data[col]) / av
                else:
                    PDA[col] = np.maximum(0, data[col] - av)
                    NDA[col] = np.maximum(0, av - data[col])
            else:
                if av > 0:
                    PDA[col] = np.maximum(0, av - data[col]) / av
                    NDA[col] = np.maximum(0, data[col] - av) / av
                else:
                    PDA[col] = np.maximum(0, av - data[col])
                    NDA[col] = np.maximum(0, data[col] - av)
        
        SP = pd.Series(0.0, index=data.index)
        SN = pd.Series(0.0, index=data.index)
        
        for col in data.columns:
            w = weights[col]
            SP += w * PDA[col]
            SN += w * NDA[col]
        
        SP_max = SP.max()
        SN_max = SN.max()
        
        NSP = SP / SP_max if SP_max > 0 else pd.Series(0, index=data.index)
        NSN = 1 - (SN / SN_max) if SN_max > 0 else pd.Series(1, index=data.index)
        
        AS = (NSP + NSN) / 2
        ranks = AS.rank(ascending=False).astype(int)
        
        return EDASResult(
            PDA=PDA,
            NDA=NDA,
            SP=SP,
            SN=SN,
            NSP=NSP,
            NSN=NSN,
            AS=AS,
            ranks=ranks,
            average_solution=average_solution,
            weights=weights
        )
    
    def _trimmed_mean(self, data: pd.DataFrame) -> pd.Series:
        """Calculate trimmed mean for each criterion."""
        from scipy import stats
        result = {}
        for col in data.columns:
            result[col] = stats.trim_mean(data[col].values, self.trim_percentage)
        return pd.Series(result)
    
    def _weighted_average(self, 
                          data: pd.DataFrame,
                          weights: Dict[str, float]) -> pd.Series:
        """Calculate importance-weighted average."""
        # Weight each criterion's contribution to the average
        result = {}
        for col in data.columns:
            # Higher weight criteria have their mean pulled toward better values
            w = weights[col]
            if col in self.benefit_criteria:
                # Pull toward higher values for benefit criteria
                result[col] = data[col].mean() + w * data[col].std()
            else:
                # Pull toward lower values for cost criteria
                result[col] = data[col].mean() - w * data[col].std()
        return pd.Series(result)
