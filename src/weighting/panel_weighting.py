# -*- coding: utf-8 -*-
"""
Panel-Aware Weight Calculators

Enhanced weighting methods that utilize BOTH temporal and cross-sectional
dimensions of panel data for more robust criterion weight determination.

For panel data with structure (Year, Province, C01, C02, ..., C20):
- Cross-sectional dimension: Variation across provinces at each time point
- Temporal dimension: Evolution of each criterion over time
- Panel structure: Combined spatio-temporal patterns

Mathematical Formulation:
    w_j = α × w_j^spatial + (1-α) × w_j^temporal
    
where:
    w_j^spatial: Cross-sectional weights (variation across alternatives)
    w_j^temporal: Temporal weights (variation over time)
    α: Spatial-temporal weight (default 0.6 for spatial emphasis)

References:
    Kao, C. (2014). Network data envelopment analysis: A review.
    European Journal of Operational Research.
    
    Wang, Y.M., & Luo, Y. (2010). Integration of correlations with standard
    deviations for determining attribute weights. Mathematical and Computer Modelling.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as SkPCA
from ..weighting.base import WeightResult
from ..weighting.entropy import EntropyWeightCalculator
from ..weighting.critic import CRITICWeightCalculator
from ..weighting.pca import PCAWeightCalculator


class PanelEntropyCalculator:
    """
    Panel-aware Entropy weight calculation.
    
    Combines cross-sectional entropy (variation across alternatives at each time)
    with temporal entropy (variation of each criterion over time).
    
    Cross-sectional entropy measures information content from comparing alternatives.
    Temporal entropy measures how much each criterion changes over time.
    
    Parameters
    ----------
    spatial_weight : float
        Weight for spatial (cross-sectional) component, in [0, 1].
        Default 0.6 (emphasize cross-sectional variation).
    temporal_aggregation : str
        How to aggregate cross-sectional weights across years:
        - 'mean': Simple average
        - 'weighted': More recent years weighted higher
        - 'stable': Emphasize criteria with stable weights
    epsilon : float
        Numerical stability constant
    
    Examples
    --------
    >>> import pandas as pd
    >>> from src.weighting.panel_weighting import PanelEntropyCalculator
    >>> 
    >>> # Panel data with Year, Province, C01, C02, ...
    >>> panel_df = pd.read_csv('panel_data.csv')
    >>> 
    >>> calc = PanelEntropyCalculator(spatial_weight=0.6)
    >>> result = calc.calculate(panel_df, 
    ...                         entity_col='Province',
    ...                         time_col='Year',
    ...                         criteria_cols=['C01', 'C02', 'C03'])
    >>> print(result.weights)
    """
    
    def __init__(self, 
                 spatial_weight: float = 0.6,
                 temporal_aggregation: str = 'weighted',
                 epsilon: float = 1e-10):
        if not 0 <= spatial_weight <= 1:
            raise ValueError(f"spatial_weight must be in [0,1], got {spatial_weight}")
        self.spatial_weight = spatial_weight
        self.temporal_weight = 1.0 - spatial_weight
        self.temporal_aggregation = temporal_aggregation
        self.epsilon = epsilon
        self.entropy_calc = EntropyWeightCalculator(epsilon=epsilon)
    
    def calculate(self,
                  panel_data: pd.DataFrame,
                  entity_col: str = 'Province',
                  time_col: str = 'Year',
                  criteria_cols: Optional[list] = None) -> WeightResult:
        """
        Calculate panel-aware entropy weights.
        
        Parameters
        ----------
        panel_data : pd.DataFrame
            Panel data with columns [time_col, entity_col, criteria...]
        entity_col : str
            Name of entity/alternative column (e.g., 'Province')
        time_col : str
            Name of time column (e.g., 'Year')
        criteria_cols : list, optional
            List of criterion column names. If None, auto-detect.
        
        Returns
        -------
        WeightResult
            Combined panel-aware weights with spatial and temporal components.
        """
        if criteria_cols is None:
            criteria_cols = [c for c in panel_data.columns 
                           if c not in [time_col, entity_col]]
        
        years = sorted(panel_data[time_col].unique())
        n_years = len(years)
        
        # =====================================================================
        # Part 1: Cross-Sectional (Spatial) Entropy
        # =====================================================================
        # Calculate entropy for each year, then aggregate
        yearly_weights = []
        
        for year in years:
            year_data = panel_data[panel_data[time_col] == year][criteria_cols]
            result = self.entropy_calc.calculate(year_data)
            yearly_weights.append(
                np.array([result.weights[c] for c in criteria_cols])
            )
        
        yearly_weights = np.array(yearly_weights)  # (n_years, n_criteria)
        
        # Aggregate across years
        if self.temporal_aggregation == 'mean':
            spatial_weights = yearly_weights.mean(axis=0)
        elif self.temporal_aggregation == 'weighted':
            # More recent years get higher weight
            time_weights = np.exp(np.linspace(0, 1, n_years))
            time_weights = time_weights / time_weights.sum()
            spatial_weights = (yearly_weights.T @ time_weights)
        elif self.temporal_aggregation == 'stable':
            # Emphasize criteria with stable weights across time
            mean_weights = yearly_weights.mean(axis=0)
            weight_std = yearly_weights.std(axis=0)
            stability = 1.0 / (weight_std + self.epsilon)
            spatial_weights = mean_weights * stability
        else:
            raise ValueError(f"Unknown temporal_aggregation: {self.temporal_aggregation}")
        
        # Normalize
        spatial_weights = spatial_weights / (spatial_weights.sum() + self.epsilon)
        
        # =====================================================================
        # Part 2: Temporal Entropy
        # =====================================================================
        # For each criterion, compute entropy of its time series
        # High temporal entropy = criterion values change a lot over time
        
        temporal_entropies = []
        
        for criterion in criteria_cols:
            # Average value per entity across all years
            entity_time_series = panel_data.groupby(entity_col)[criterion].apply(
                lambda x: x.values
            )
            
            # Compute entropy of temporal variation for each entity, then average
            entity_temporal_entropies = []
            for entity_vals in entity_time_series:
                if len(entity_vals) < 2:
                    continue
                # Normalize to proportions
                vals = np.array(entity_vals)
                vals_sum = vals.sum()
                if vals_sum > self.epsilon:
                    p = vals / vals_sum
                    p = np.clip(p, self.epsilon, None)
                    H = -np.sum(p * np.log(p)) / (np.log(len(p)) + self.epsilon)
                    entity_temporal_entropies.append(H)
            
            avg_temporal_entropy = (
                np.mean(entity_temporal_entropies) 
                if entity_temporal_entropies else 0.5
            )
            temporal_entropies.append(avg_temporal_entropy)
        
        temporal_entropies = np.array(temporal_entropies)
        
        # Convert entropy to weights (higher entropy = more variation = higher weight)
        # But normalized entropy is already in [0,1], so use divergence
        temporal_divergence = np.clip(temporal_entropies, self.epsilon, None)
        temporal_weights = temporal_divergence / (temporal_divergence.sum() + self.epsilon)
        
        # =====================================================================
        # Part 3: Combine Spatial and Temporal
        # =====================================================================
        combined_weights = (
            self.spatial_weight * spatial_weights +
            self.temporal_weight * temporal_weights
        )
        combined_weights = combined_weights / (combined_weights.sum() + self.epsilon)
        
        return WeightResult(
            weights={c: float(combined_weights[i]) for i, c in enumerate(criteria_cols)},
            method="panel_entropy",
            details={
                "spatial_weights": {c: float(spatial_weights[i]) 
                                   for i, c in enumerate(criteria_cols)},
                "temporal_weights": {c: float(temporal_weights[i]) 
                                    for i, c in enumerate(criteria_cols)},
                "temporal_entropies": {c: float(temporal_entropies[i])
                                      for i, c in enumerate(criteria_cols)},
                "yearly_weight_stability": {
                    c: float(yearly_weights[:, i].std())
                    for i, c in enumerate(criteria_cols)
                },
                "spatial_weight_param": self.spatial_weight,
                "temporal_weight_param": self.temporal_weight,
                "n_years": n_years,
                "temporal_aggregation": self.temporal_aggregation
            }
        )


class PanelCRITICCalculator:
    """
    Panel-aware CRITIC weight calculation.
    
    Extends CRITIC to panel data by considering both:
    1. Cross-sectional correlation (spatial): How criteria correlate across alternatives
    2. Temporal correlation: How criteria co-evolve over time
    3. Pooled correlation: Combined correlation structure
    
    Parameters
    ----------
    spatial_weight : float
        Weight for spatial (cross-sectional) component, in [0, 1].
        Default 0.6.
    use_pooled_correlation : bool
        If True, uses pooled correlation across all years.
        If False, averages yearly correlations.
    epsilon : float
        Numerical stability constant
    """
    
    def __init__(self,
                 spatial_weight: float = 0.6,
                 use_pooled_correlation: bool = True,
                 epsilon: float = 1e-10):
        if not 0 <= spatial_weight <= 1:
            raise ValueError(f"spatial_weight must be in [0,1], got {spatial_weight}")
        self.spatial_weight = spatial_weight
        self.temporal_weight = 1.0 - spatial_weight
        self.use_pooled_correlation = use_pooled_correlation
        self.epsilon = epsilon
        self.critic_calc = CRITICWeightCalculator(epsilon=epsilon)
    
    def calculate(self,
                  panel_data: pd.DataFrame,
                  entity_col: str = 'Province',
                  time_col: str = 'Year',
                  criteria_cols: Optional[list] = None) -> WeightResult:
        """
        Calculate panel-aware CRITIC weights.
        
        Parameters
        ----------
        panel_data : pd.DataFrame
            Panel data with columns [time_col, entity_col, criteria...]
        entity_col : str
            Name of entity/alternative column
        time_col : str
            Name of time column
        criteria_cols : list, optional
            List of criterion column names
        
        Returns
        -------
        WeightResult
            Combined panel-aware CRITIC weights.
        """
        if criteria_cols is None:
            criteria_cols = [c for c in panel_data.columns 
                           if c not in [time_col, entity_col]]
        
        years = sorted(panel_data[time_col].unique())
        n_criteria = len(criteria_cols)
        
        # =====================================================================
        # Part 1: Cross-Sectional CRITIC
        # =====================================================================
        if self.use_pooled_correlation:
            # Use all data pooled together
            pooled_data = panel_data[criteria_cols]
            spatial_result = self.critic_calc.calculate(pooled_data)
            spatial_weights = np.array([spatial_result.weights[c] for c in criteria_cols])
            spatial_corr = pooled_data.corr()
        else:
            # Average CRITIC across years
            yearly_weights = []
            yearly_corrs = []
            
            for year in years:
                year_data = panel_data[panel_data[time_col] == year][criteria_cols]
                result = self.critic_calc.calculate(year_data)
                yearly_weights.append(
                    np.array([result.weights[c] for c in criteria_cols])
                )
                yearly_corrs.append(year_data.corr().values)
            
            spatial_weights = np.mean(yearly_weights, axis=0)
            spatial_corr = pd.DataFrame(
                np.mean(yearly_corrs, axis=0),
                index=criteria_cols,
                columns=criteria_cols
            )
        
        spatial_weights = spatial_weights / (spatial_weights.sum() + self.epsilon)
        
        # =====================================================================
        # Part 2: Temporal CRITIC (Correlation of Time Series)
        # =====================================================================
        # For each criterion, extract time series for each entity
        # Then compute correlation of these time series across criteria
        
        # Pivot to get one row per entity with time series for each criterion
        temporal_data = []
        
        for criterion in criteria_cols:
            # For each entity, get its time series for this criterion
            entity_ts = panel_data.pivot(
                index=entity_col, columns=time_col, values=criterion
            ).values  # (n_entities, n_years)
            
            # Flatten or use mean as representative time pattern
            temporal_data.append(entity_ts.flatten())
        
        temporal_df = pd.DataFrame(
            np.array(temporal_data).T,
            columns=criteria_cols
        )
        
        # Compute temporal CRITIC
        temporal_result = self.critic_calc.calculate(temporal_df)
        temporal_weights = np.array([temporal_result.weights[c] for c in criteria_cols])
        temporal_weights = temporal_weights / (temporal_weights.sum() + self.epsilon)
        
        temporal_corr = temporal_df.corr()
        
        # =====================================================================
        # Part 3: Combine Spatial and Temporal
        # =====================================================================
        combined_weights = (
            self.spatial_weight * spatial_weights +
            self.temporal_weight * temporal_weights
        )
        combined_weights = combined_weights / (combined_weights.sum() + self.epsilon)
        
        return WeightResult(
            weights={c: float(combined_weights[i]) for i, c in enumerate(criteria_cols)},
            method="panel_critic",
            details={
                "spatial_weights": {c: float(spatial_weights[i]) 
                                   for i, c in enumerate(criteria_cols)},
                "temporal_weights": {c: float(temporal_weights[i]) 
                                    for i, c in enumerate(criteria_cols)},
                "spatial_correlation": spatial_corr.to_dict(),
                "temporal_correlation": temporal_corr.to_dict(),
                "spatial_weight_param": self.spatial_weight,
                "temporal_weight_param": self.temporal_weight,
                "use_pooled": self.use_pooled_correlation
            }
        )


class PanelPCACalculator:
    """
Panel-aware PCA weight calculation.
    
    Applies PCA to the full panel structure to capture both spatial
    (cross-sectional) and temporal (time-series) variance patterns.
    
    Parameters
    ----------
    variance_threshold : float
        Cumulative variance threshold for component retention
    pooling_method : str
        How to handle panel structure:
        - 'stack': Stack all year-observations (default)
        - 'average_correlation': Use average correlation across years
        - 'temporal_only': PCA on entity time series
    epsilon : float
        Numerical stability constant
    """
    
    def __init__(self,
                 variance_threshold: float = 0.85,
                 pooling_method: str = 'stack',
                 epsilon: float = 1e-10):
        self.variance_threshold = variance_threshold
        self.pooling_method = pooling_method
        self.epsilon = epsilon
        self.pca_calc = PCAWeightCalculator(
            variance_threshold=variance_threshold,
            epsilon=epsilon
        )
    
    def calculate(self,
                  panel_data: pd.DataFrame,
                  entity_col: str = 'Province',
                  time_col: str = 'Year',
                  criteria_cols: Optional[list] = None) -> WeightResult:
        """
        Calculate panel-aware PCA weights.
        
        Parameters
        ----------
        panel_data : pd.DataFrame
            Panel data with columns [time_col, entity_col, criteria...]
        entity_col : str
            Name of entity/alternative column
        time_col : str
            Name of time column
        criteria_cols : list, optional
            List of criterion column names
        
        Returns
        -------
        WeightResult
            PCA weights from panel structure.
        """
        if criteria_cols is None:
            criteria_cols = [c for c in panel_data.columns 
                           if c not in [time_col, entity_col]]
        
        if self.pooling_method == 'stack':
            # Stack all observations (all years × all entities)
            pca_data = panel_data[criteria_cols]
            result = self.pca_calc.calculate(pca_data)
            
            result.details['pooling_method'] = 'stack'
            result.details['n_observations'] = len(pca_data)
            result.method = "panel_pca_stack"
            
        elif self.pooling_method == 'average_correlation':
            # Compute PCA on average correlation matrix across years
            years = sorted(panel_data[time_col].unique())
            corr_matrices = []
            
            for year in years:
                year_data = panel_data[panel_data[time_col] == year][criteria_cols]
                corr_matrices.append(year_data.corr().values)
            
            avg_corr = np.mean(corr_matrices, axis=0)
            avg_corr_df = pd.DataFrame(avg_corr, index=criteria_cols, columns=criteria_cols)
            
            # Standardize and apply PCA using average correlation
            # Use the latest year's data structure but with average correlation pattern
            latest_year = max(years)
            latest_data = panel_data[panel_data[time_col] == latest_year][criteria_cols]
            
            result = self.pca_calc.calculate(latest_data)
            result.details['pooling_method'] = 'average_correlation'
            result.details['n_years_averaged'] = len(years)
            result.method = "panel_pca_avg_corr"
            
        elif self.pooling_method == 'temporal_only':
            # PCA on time series: one row per entity with concatenated time series
            temporal_data = []
            
            for entity in panel_data[entity_col].unique():
                entity_data = panel_data[panel_data[entity_col] == entity].sort_values(time_col)
                # Concatenate time series for each criterion
                row = []
                for criterion in criteria_cols:
                    row.extend(entity_data[criterion].values)
                temporal_data.append(row)
            
            # Create column names: C01_Y1, C01_Y2, ..., C02_Y1, ...
            years = sorted(panel_data[time_col].unique())
            col_names = [f"{c}_Y{y}" for c in criteria_cols for y in years]
            
            temporal_df = pd.DataFrame(temporal_data)
            result = self.pca_calc.calculate(temporal_df)
            
            # Aggregate weights back to criteria level
            n_years = len(years)
            aggregated_weights = {}
            for i, criterion in enumerate(criteria_cols):
                # Average weights across time points for this criterion
                time_indices = range(i * n_years, (i + 1) * n_years)
                weights_array = np.array(list(result.weights.values()))
                aggregated_weights[criterion] = float(weights_array[list(time_indices)].mean())
            
            # Renormalize
            total = sum(aggregated_weights.values())
            aggregated_weights = {k: v/total for k, v in aggregated_weights.items()}
            
            result = WeightResult(
                weights=aggregated_weights,
                method="panel_pca_temporal",
                details={
                    **result.details,
                    'pooling_method': 'temporal_only',
                    'n_entities': len(panel_data[entity_col].unique()),
                    'n_years': n_years
                }
            )
        else:
            raise ValueError(f"Unknown pooling_method: {self.pooling_method}")
        
        return result


class PanelEnsembleCalculator:
    """
    Panel-aware ensemble weight calculator.
    
    Combines panel-aware Entropy, CRITIC, and PCA methods using
    the integrated hybrid strategy.
    
    Parameters
    ----------
    spatial_weight : float
        Weight for spatial component in individual methods
    entropy_aggregation : str
        Temporal aggregation for entropy ('mean', 'weighted', 'stable')
    critic_pooled : bool
        Whether CRITIC uses pooled correlation
    pca_pooling : str
        PCA pooling method ('stack', 'average_correlation', 'temporal_only')
    pca_variance_threshold : float
        PCA variance threshold
    """
    
    def __init__(self,
                 spatial_weight: float = 0.6,
                 entropy_aggregation: str = 'weighted',
                 critic_pooled: bool = True,
                 pca_pooling: str = 'stack',
                 pca_variance_threshold: float = 0.85):
        self.spatial_weight = spatial_weight
        self.entropy_calc = PanelEntropyCalculator(
            spatial_weight=spatial_weight,
            temporal_aggregation=entropy_aggregation
        )
        self.critic_calc = PanelCRITICCalculator(
            spatial_weight=spatial_weight,
            use_pooled_correlation=critic_pooled
        )
        self.pca_calc = PanelPCACalculator(
            variance_threshold=pca_variance_threshold,
            pooling_method=pca_pooling
        )
    
    def calculate(self,
                  panel_data: pd.DataFrame,
                  entity_col: str = 'Province',
                  time_col: str = 'Year',
                  criteria_cols: Optional[list] = None) -> WeightResult:
        """
        Calculate panel-aware ensemble weights using integrated hybrid strategy.
        
        Parameters
        ----------
        panel_data : pd.DataFrame
            Panel data
        entity_col : str
            Entity column name
        time_col : str
            Time column name
        criteria_cols : list, optional
            Criterion column names
        
        Returns
        -------
        WeightResult
            Panel-aware integrated ensemble weights.
        """
        if criteria_cols is None:
            criteria_cols = [c for c in panel_data.columns 
                           if c not in [time_col, entity_col]]
        
        # Compute individual panel-aware weights
        entropy_result = self.entropy_calc.calculate(
            panel_data, entity_col, time_col, criteria_cols
        )
        critic_result = self.critic_calc.calculate(
            panel_data, entity_col, time_col, criteria_cols
        )
        pca_result = self.pca_calc.calculate(
            panel_data, entity_col, time_col, criteria_cols
        )
        
        # Extract weight arrays
        entropy_weights = np.array([entropy_result.weights[c] for c in criteria_cols])
        critic_weights = np.array([critic_result.weights[c] for c in criteria_cols])
        pca_weights = np.array([pca_result.weights[c] for c in criteria_cols])
        
        # Compute entropy of each weight vector for integration
        epsilon = 1e-15
        n_criteria = len(criteria_cols)
        
        integration_coefficients = {}
        for name, w in [('entropy', entropy_weights), 
                       ('critic', critic_weights), 
                       ('pca', pca_weights)]:
            w_safe = np.clip(w, epsilon, None)
            w_norm = w_safe / w_safe.sum()
            H = -np.sum(w_norm * np.log(w_norm)) / (np.log(n_criteria) + epsilon)
            H = min(H, 1.0)
            integration_coefficients[name] = max(1.0 - H, epsilon)
        
        # Normalize integration coefficients
        total_coeff = sum(integration_coefficients.values())
        alpha = {m: c / total_coeff for m, c in integration_coefficients.items()}
        
        # Combined weights
        combined = (
            alpha['entropy'] * entropy_weights +
            alpha['critic'] * critic_weights +
            alpha['pca'] * pca_weights
        )
        combined = np.clip(combined, epsilon, None)
        combined = combined / combined.sum()
        
        return WeightResult(
            weights={c: float(combined[i]) for i, c in enumerate(criteria_cols)},
            method="panel_ensemble_integrated",
            details={
                "individual_weights": {
                    "entropy": entropy_result.weights,
                    "critic": critic_result.weights,
                    "pca": pca_result.weights
                },
                "integration_coefficients": alpha,
                "spatial_weight_param": self.spatial_weight,
                "entropy_details": entropy_result.details,
                "critic_details": critic_result.details,
                "pca_details": pca_result.details
            }
        )
