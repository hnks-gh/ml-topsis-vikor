# -*- coding: utf-8 -*-
"""
Main Pipeline Orchestrator
==========================

Comprehensive ML-MCDM panel data analysis pipeline.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import warnings
import time

warnings.filterwarnings('ignore')

# Internal imports
from .config import Config, get_default_config
from .logger import setup_logger, ProgressLogger
from .data_loader import PanelDataLoader, PanelData, TemporalFeatureEngineer

from .mcdm import (
    EntropyWeightCalculator, CRITICWeightCalculator, EnsembleWeightCalculator,
    TOPSISCalculator, DynamicTOPSIS, VIKORCalculator, MultiPeriodVIKOR,
    FuzzyTOPSIS
)

from .ml import (
    PanelRegression, RandomForestTS, LSTMForecaster, RoughSetReducer
)

from .ensemble import (
    StackingEnsemble, BordaCount, CopelandMethod, aggregate_rankings
)

from .analysis import (
    ConvergenceAnalysis, SensitivityAnalysis, CrossValidator, BootstrapValidator
)

from .visualization import PanelVisualizer


@dataclass
class PipelineResult:
    """Container for all pipeline results."""
    # Data
    panel_data: PanelData
    decision_matrix: np.ndarray
    
    # Weights
    entropy_weights: np.ndarray
    critic_weights: np.ndarray
    ensemble_weights: np.ndarray
    
    # MCDM Results
    topsis_scores: np.ndarray
    topsis_rankings: np.ndarray
    dynamic_topsis_scores: np.ndarray
    vikor_results: Dict
    fuzzy_topsis_scores: np.ndarray
    
    # ML Results
    panel_regression_result: Any
    rf_feature_importance: Dict[str, float]
    lstm_forecasts: Optional[np.ndarray]
    rough_set_reduction: Any
    
    # Ensemble
    stacking_result: Any
    aggregated_ranking: Any
    
    # Analysis
    convergence_result: Any
    sensitivity_result: Any
    
    # Meta
    execution_time: float
    config: Config
    
    def get_final_ranking_df(self) -> pd.DataFrame:
        """Get final aggregated ranking as DataFrame."""
        entities = self.panel_data.entities
        return pd.DataFrame({
            'province': entities,
            'final_rank': self.aggregated_ranking.final_ranking,
            'final_score': self.aggregated_ranking.final_scores,
            'topsis_rank': self.topsis_rankings,
            'kendall_w': self.aggregated_ranking.kendall_w
        }).sort_values('final_rank')


class MLTOPSISPipeline:
    """
    Production-grade ML-MCDM pipeline for panel data analysis.
    
    Integrates:
    - Multiple weighting methods (Entropy, CRITIC, Ensemble)
    - Multiple MCDM methods (TOPSIS, Dynamic TOPSIS, VIKOR, Fuzzy TOPSIS)
    - Multiple ML methods (Panel Regression, Random Forest, LSTM, Rough Sets)
    - Ensemble methods (Stacking, Rank Aggregation)
    - Comprehensive analysis (Convergence, Sensitivity, Validation)
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize pipeline.
        
        Parameters
        ----------
        config : Config, optional
            Pipeline configuration
        """
        self.config = config or get_default_config()
        log_file = Path(self.config.output_dir) / 'logs' / 'pipeline.log'
        self.logger = setup_logger('ml_topsis', log_file=log_file)
        self.visualizer = PanelVisualizer(
            output_dir=str(Path(self.config.output_dir) / 'figures')
        )
    
    def run(self, data_path: Optional[str] = None) -> PipelineResult:
        """
        Execute full analysis pipeline.
        
        Parameters
        ----------
        data_path : str, optional
            Path to panel data CSV
        
        Returns
        -------
        PipelineResult
            Comprehensive results object
        """
        start_time = time.time()
        
        self.logger.info("=" * 60)
        self.logger.info("ML-MCDM PANEL DATA ANALYSIS PIPELINE")
        self.logger.info("=" * 60)
        
        # Phase 1: Data Loading
        with ProgressLogger(self.logger, "Phase 1: Data Loading"):
            panel_data = self._load_data(data_path)
        
        # Phase 2: Weight Calculation
        with ProgressLogger(self.logger, "Phase 2: Weight Calculation"):
            weights = self._calculate_weights(panel_data)
        
        # Phase 3: MCDM Analysis
        with ProgressLogger(self.logger, "Phase 3: MCDM Analysis"):
            mcdm_results = self._run_mcdm(panel_data, weights)
        
        # Phase 4: ML Analysis
        with ProgressLogger(self.logger, "Phase 4: ML Analysis"):
            ml_results = self._run_ml(panel_data, mcdm_results)
        
        # Phase 5: Ensemble Integration
        with ProgressLogger(self.logger, "Phase 5: Ensemble Integration"):
            ensemble_results = self._run_ensemble(mcdm_results, ml_results, panel_data)
        
        # Phase 6: Advanced Analysis
        with ProgressLogger(self.logger, "Phase 6: Advanced Analysis"):
            analysis_results = self._run_analysis(panel_data, mcdm_results, weights)
        
        # Phase 7: Visualization
        with ProgressLogger(self.logger, "Phase 7: Visualization"):
            self._generate_visualizations(panel_data, mcdm_results, 
                                         ensemble_results, analysis_results)
        
        execution_time = time.time() - start_time
        
        self.logger.info("=" * 60)
        self.logger.info(f"Pipeline completed in {execution_time:.2f} seconds")
        self.logger.info("=" * 60)
        
        # Compile results
        return PipelineResult(
            panel_data=panel_data,
            decision_matrix=panel_data.cross_section.values,
            entropy_weights=weights['entropy'],
            critic_weights=weights['critic'],
            ensemble_weights=weights['ensemble'],
            topsis_scores=mcdm_results['topsis_scores'],
            topsis_rankings=mcdm_results['topsis_rankings'],
            dynamic_topsis_scores=mcdm_results['dynamic_topsis_scores'],
            vikor_results=mcdm_results['vikor'],
            fuzzy_topsis_scores=mcdm_results['fuzzy_scores'],
            panel_regression_result=ml_results['panel_regression'],
            rf_feature_importance=ml_results['rf_importance'],
            lstm_forecasts=ml_results.get('lstm_forecasts'),
            rough_set_reduction=ml_results.get('rough_set'),
            stacking_result=ensemble_results['stacking'],
            aggregated_ranking=ensemble_results['aggregated'],
            convergence_result=analysis_results['convergence'],
            sensitivity_result=analysis_results['sensitivity'],
            execution_time=execution_time,
            config=self.config
        )
    
    def _load_data(self, data_path: Optional[str]) -> PanelData:
        """Load and prepare panel data."""
        loader = PanelDataLoader(self.config)
        
        if data_path:
            panel_data = loader.load(data_path)
        else:
            # Auto-generate if no data provided
            self.logger.info("No data path provided, generating synthetic panel data")
            panel_data = loader.generate_synthetic(
                n_provinces=self.config.panel.n_provinces,
                n_years=self.config.panel.n_years,
                n_components=self.config.panel.n_components
            )
        
        self.logger.info(f"Panel data loaded: {len(panel_data.provinces)} entities, "
                        f"{len(panel_data.years)} periods, "
                        f"{len(panel_data.components)} components")
        
        return panel_data
    
    def _calculate_weights(self, panel_data: PanelData) -> Dict[str, np.ndarray]:
        """Calculate weights using multiple methods."""
        # Get latest cross-section as DataFrame
        latest_year = max(panel_data.years)
        df = panel_data.cross_section[latest_year][panel_data.components]
        
        # Entropy weights
        entropy_calc = EntropyWeightCalculator()
        entropy_result = entropy_calc.calculate(df)
        
        # CRITIC weights
        critic_calc = CRITICWeightCalculator()
        critic_result = critic_calc.calculate(df)
        
        # Ensemble weights (this will calculate both again, but that's fine)
        ensemble_calc = EnsembleWeightCalculator()
        ensemble_result = ensemble_calc.calculate(df)
        
        # Convert weight dicts to arrays
        components = panel_data.components
        entropy_weights = np.array([entropy_result.weights[c] for c in components])
        critic_weights = np.array([critic_result.weights[c] for c in components])
        ensemble_weights = np.array([ensemble_result.weights[c] for c in components])
        
        self.logger.info(f"Entropy weights range: [{entropy_weights.min():.4f}, "
                        f"{entropy_weights.max():.4f}]")
        self.logger.info(f"CRITIC weights range: [{critic_weights.min():.4f}, "
                        f"{critic_weights.max():.4f}]")
        
        return {
            'entropy': entropy_weights,
            'critic': critic_weights,
            'ensemble': ensemble_weights
        }
    
    def _run_mcdm(self, panel_data: PanelData, 
                  weights: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Run MCDM analysis methods."""
        results = {}
        
        # Use ensemble weights for main analysis
        w = weights['ensemble']
        # Get latest cross-section as DataFrame
        latest_year = max(panel_data.years)
        df = panel_data.cross_section[latest_year][panel_data.components]
        
        # Convert numpy weights to dict format for TOPSIS
        weights_dict = {c: w[i] for i, c in enumerate(panel_data.components)}
        
        # 1. Static TOPSIS
        topsis = TOPSISCalculator(normalization=self.config.topsis.normalization.value)
        topsis_result = topsis.calculate(df, weights_dict)
        results['topsis_scores'] = topsis_result.scores.values
        results['topsis_rankings'] = topsis_result.ranks.values
        results['topsis_result'] = topsis_result
        
        self.logger.info(f"TOPSIS: Top performer score = {topsis_result.scores.max():.4f}")
        
        # 2. Dynamic TOPSIS (panel)
        dynamic_topsis = DynamicTOPSIS(
            temporal_discount=self.config.topsis.temporal_discount,
            trajectory_weight=self.config.topsis.trajectory_weight,
            stability_weight=self.config.topsis.stability_weight,
            normalization=self.config.topsis.normalization.value
        )
        dynamic_result = dynamic_topsis.calculate(panel_data, weights_dict)
        results['dynamic_topsis_scores'] = dynamic_result.scores.values
        results['dynamic_topsis'] = dynamic_result
        
        # 3. VIKOR
        vikor = VIKORCalculator(v=self.config.vikor.v)
        vikor_result = vikor.calculate(df, weights_dict)
        results['vikor'] = {
            'Q': vikor_result.Q,
            'S': vikor_result.S,
            'R': vikor_result.R,
            'rankings': vikor_result.final_ranks
        }
        
        self.logger.info(f"VIKOR: Best alternative Q = {vikor_result.Q.min():.4f}")
        
        # 4. Fuzzy TOPSIS
        fuzzy = FuzzyTOPSIS()
        fuzzy_result = fuzzy.calculate_from_panel(panel_data, weights_dict)
        results['fuzzy_scores'] = fuzzy_result.scores
        results['fuzzy_result'] = fuzzy_result
        self.logger.info(f"Fuzzy TOPSIS: Top performer = {fuzzy_result.scores.max():.4f}")
        
        return results
    
    def _run_ml(self, panel_data: PanelData, 
                mcdm_results: Dict[str, Any]) -> Dict[str, Any]:
        """Run ML analysis methods."""
        results = {}
        
        # Prepare features
        feature_eng = TemporalFeatureEngineer()
        features_df = feature_eng.create_all_features(panel_data)
        
        # 1. Panel Regression
        # Add TOPSIS scores to panel data for regression
        topsis_scores = mcdm_results['topsis_scores']
        
        # Create a DataFrame with scores for panel regression
        try:
            # Create panel format data with target column
            reg_df = panel_data.long.copy()
            # Map TOPSIS scores to each province
            score_map = dict(zip(panel_data.entities, topsis_scores))
            reg_df['topsis_score'] = reg_df['Province'].map(score_map)
            
            panel_regression = PanelRegression()
            reg_result = panel_regression.fit(
                reg_df, 
                y_col='topsis_score',
                X_cols=panel_data.components,
                province_col='Province',
                year_col='Year'
            )
            results['panel_regression'] = reg_result
            self.logger.info(f"Panel Regression R²: {reg_result.r_squared:.4f}")
        except Exception as e:
            self.logger.warning(f"Panel regression failed: {e}")
            results['panel_regression'] = None
        
        # 2. Random Forest with Time-Series CV
        rf = RandomForestTS(
            n_estimators=self.config.random_forest.n_estimators,
            max_depth=self.config.random_forest.max_depth,
            min_samples_split=self.config.random_forest.min_samples_split,
            min_samples_leaf=self.config.random_forest.min_samples_leaf,
            max_features=self.config.random_forest.max_features,
            n_splits=self.config.random_forest.n_splits
        )
        
        # Create target column in long-form data
        # Map TOPSIS scores to each province-year observation
        df_with_target = features_df.copy()
        topsis_map = dict(zip(panel_data.get_latest().index, 
                              mcdm_results['topsis_scores']))
        df_with_target['topsis_score'] = df_with_target['Province'].map(topsis_map)
        
        # Filter valid observations
        df_rf = df_with_target.dropna(subset=['topsis_score'])
        
        # Check actual unique years in the filtered data
        unique_years = df_rf['Year'].nunique() if 'Year' in df_rf.columns else 0
        rf_splits = self.config.random_forest.n_splits
        
        if len(df_rf) >= 10 and unique_years >= rf_splits + 1:
            try:
                rf_result = rf.fit_predict(df_rf, 'topsis_score', panel_data.components)
                results['rf_result'] = rf_result
                results['rf_importance'] = rf_result.feature_importance.to_dict()
                cv_r2_mean = np.mean(rf_result.cv_scores['r2'])
                self.logger.info(f"Random Forest CV R²: {cv_r2_mean:.4f}")
            except Exception as e:
                self.logger.warning(f"Random Forest failed: {e}")
                results['rf_result'] = None
                results['rf_importance'] = {}
        else:
            self.logger.warning(f"Skipping RF: need >= {rf_splits + 1} years ({unique_years} available in data)")
            results['rf_result'] = None
            results['rf_importance'] = {}
        
        # 3. LSTM Forecasting (optional)
        if self.config.lstm.enabled:
            try:
                lstm = LSTMForecaster(
                    sequence_length=self.config.lstm.sequence_length,
                    hidden_units=self.config.lstm.hidden_units,
                    n_layers=self.config.lstm.n_layers,
                    dropout=self.config.lstm.dropout,
                    epochs=self.config.lstm.epochs,
                    batch_size=self.config.lstm.batch_size,
                    learning_rate=self.config.lstm.learning_rate,
                    patience=self.config.lstm.patience
                )
                lstm_result = lstm.fit_predict(panel_data, panel_data.components[:5])
                results['lstm_forecasts'] = lstm_result.predictions
                self.logger.info(f"LSTM Test Metrics: {lstm_result.test_metrics}")
            except Exception as e:
                self.logger.warning(f"LSTM forecasting skipped: {e}")
                results['lstm_forecasts'] = None
        
        # 4. Rough Set Reduction
        try:
            rough_set = RoughSetReducer(
                quality_threshold=self.config.rough_sets.quality_threshold,
                n_bins=self.config.rough_sets.n_bins
            )
            # Use latest cross-section DataFrame
            rs_result = rough_set.reduce(panel_data.get_latest()[panel_data.components])
            results['rough_set'] = rs_result
            
            self.logger.info(f"Rough Set: Reduced from {rs_result.original_n_attributes} "
                           f"to {rs_result.reduced_n_attributes} attributes")
        except Exception as e:
            self.logger.warning(f"Rough set reduction failed: {e}")
            results['rough_set'] = None
        
        return results
    
    def _run_ensemble(self, mcdm_results: Dict[str, Any],
                      ml_results: Dict[str, Any],
                      panel_data: PanelData) -> Dict[str, Any]:
        """Run ensemble methods."""
        results = {}
        
        # 1. Stacking Ensemble for score prediction
        base_predictions = {
            'TOPSIS': mcdm_results['topsis_scores'],
            'VIKOR_Q': 1 - mcdm_results['vikor']['Q'],  # Invert so higher is better
            'Dynamic_TOPSIS': mcdm_results['dynamic_topsis_scores']
        }
        
        # Use TOPSIS as pseudo-target for meta-learning
        target = mcdm_results['topsis_scores']
        
        stacking = StackingEnsemble(
            meta_learner=self.config.ensemble.meta_learner,
            alpha=self.config.ensemble.alpha
        )
        stacking_result = stacking.fit_predict(base_predictions, target)
        results['stacking'] = stacking_result
        
        self.logger.info(f"Stacking Meta-Model R²: {stacking_result.meta_model_r2:.4f}")
        
        # 2. Rank Aggregation
        rankings = {
            'TOPSIS': mcdm_results['topsis_rankings'],
            'VIKOR': mcdm_results['vikor']['rankings'],
            'Fuzzy_TOPSIS': len(mcdm_results['fuzzy_scores']) - 
                           np.argsort(np.argsort(mcdm_results['fuzzy_scores']))
        }
        
        # Borda count
        borda = BordaCount()
        aggregated = borda.aggregate(rankings)
        results['aggregated'] = aggregated
        
        self.logger.info(f"Rank Aggregation Kendall's W: {aggregated.kendall_w:.4f}")
        
        return results
    
    def _run_analysis(self, panel_data: PanelData,
                      mcdm_results: Dict[str, Any],
                      weights: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Run advanced analysis."""
        results = {}
        
        # 1. Convergence Analysis
        # Create scores DataFrame
        scores_df = panel_data.long.copy()
        scores_df['score'] = np.tile(
            mcdm_results['topsis_scores'], 
            len(panel_data.time_periods)
        )[:len(scores_df)]
        
        try:
            convergence = ConvergenceAnalysis()
            conv_result = convergence.analyze(
                scores_df, 
                entity_col='Province',
                time_col='Year',
                score_col='score'
            )
            results['convergence'] = conv_result
            
            self.logger.info(f"Convergence: β = {conv_result.beta_coefficient:.4f}, "
                           f"Half-life = {conv_result.half_life:.1f} years")
        except Exception as e:
            self.logger.warning(f"Convergence analysis failed: {e}")
            results['convergence'] = None
        
        # 2. Sensitivity Analysis
        def topsis_ranking_func(matrix, w):
            # Convert numpy arrays back to DataFrame for TOPSIS
            df = pd.DataFrame(matrix, columns=panel_data.components)
            weights_dict = dict(zip(panel_data.components, w))
            calc = TOPSISCalculator()
            result = calc.calculate(df, weights_dict)
            return result.ranks.values
        
        sensitivity = SensitivityAnalysis(
            n_simulations=self.config.validation.n_simulations
        )
        
        sens_result = sensitivity.analyze(
            panel_data.get_latest()[panel_data.components].values,
            weights['ensemble'],
            topsis_ranking_func,
            criteria_names=panel_data.components,
            alternative_names=panel_data.entities
        )
        results['sensitivity'] = sens_result
        
        self.logger.info(f"Sensitivity: Overall robustness = {sens_result.overall_robustness:.4f}")
        
        return results
    
    def _generate_visualizations(self, panel_data: PanelData,
                                mcdm_results: Dict[str, Any],
                                ensemble_results: Dict[str, Any],
                                analysis_results: Dict[str, Any]) -> None:
        """Generate all visualizations."""
        try:
            # Score evolution
            scores_df = panel_data.long.copy()
            scores_df['score'] = np.tile(
                mcdm_results['topsis_scores'],
                len(panel_data.time_periods)
            )[:len(scores_df)]
            
            self.visualizer.plot_score_evolution(
                scores_df, 'Province', 'Year', 'score',
                title='Sustainability Score Evolution'
            )
            
            # Method comparison
            rankings = {
                'TOPSIS': mcdm_results['topsis_rankings'],
                'VIKOR': mcdm_results['vikor']['rankings'],
            }
            self.visualizer.plot_method_comparison(
                rankings, panel_data.entities,
                title='MCDM Method Comparison'
            )
            
            # Convergence
            if analysis_results['convergence']:
                conv = analysis_results['convergence']
                self.visualizer.plot_convergence(
                    conv.sigma_by_year,
                    conv.beta_coefficient,
                    conv.half_life
                )
            
            # Weight sensitivity
            if analysis_results['sensitivity']:
                self.visualizer.plot_weight_sensitivity(
                    analysis_results['sensitivity'].weight_sensitivity
                )
            
            # Ensemble weights
            if ensemble_results['stacking']:
                method_weights = dict(zip(
                    ensemble_results['stacking'].base_model_predictions.keys(),
                    ensemble_results['stacking'].meta_model_weights
                ))
                self.visualizer.plot_ensemble_weights(method_weights)
            
            self.logger.info("Visualizations saved to outputs/figures/")
            
        except Exception as e:
            self.logger.warning(f"Visualization generation failed: {e}")


def run_pipeline(data_path: Optional[str] = None,
                config: Optional[Config] = None) -> PipelineResult:
    """
    Convenience function to run the full pipeline.
    
    Parameters
    ----------
    data_path : str, optional
        Path to panel data CSV
    config : Config, optional
        Pipeline configuration
    
    Returns
    -------
    PipelineResult
        Comprehensive results
    """
    pipeline = MLTOPSISPipeline(config)
    return pipeline.run(data_path)
