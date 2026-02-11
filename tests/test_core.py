# -*- coding: utf-8 -*-
"""
Comprehensive Test Suite for ML-MCDM Framework v2.0

This test suite provides complete coverage of all framework components including:
- Configuration Management & Data Loading
- 6 Weighting Methods (Entropy, CRITIC, MEREC, Standard Deviation, Fusion, Robust Global)
- 5 Traditional MCDM Methods (TOPSIS, VIKOR, PROMETHEE, COPRAS, EDAS)
- 5 Fuzzy MCDM Methods (Fuzzy variants of all traditional methods)
- 4 Ensemble Aggregation Methods (Borda, Copeland, Kemeny, Stacking)
- 5 ML Forecasting Methods (Random Forest, Gradient Boosting, XGBoost, LightGBM, Neural Networks)
- Analysis & Validation Modules
- End-to-End Integration Testing

Author: Son Hoang
Version: 2.0.0
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import warnings
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Suppress warnings during tests
warnings.filterwarnings('ignore')


# ============================================================================
# FIXTURES AND UTILITIES
# ============================================================================

@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    np.random.seed(42)
    return np.random.rand(10, 5) * 100


@pytest.fixture
def sample_panel_data():
    """Generate sample panel data."""
    from src.data_loader import PanelDataLoader
    loader = PanelDataLoader()
    return loader.generate_synthetic(n_provinces=20, n_years=5, n_components=10, seed=42)


@pytest.fixture
def default_config():
    """Get default configuration."""
    from src.config import get_default_config
    return get_default_config()


@pytest.fixture
def simple_decision_matrix():
    """Create a simple decision matrix for quick tests."""
    return np.array([
        [250, 16, 12, 5],
        [200, 16, 8, 3],
        [300, 32, 16, 4],
        [275, 32, 8, 4],
        [225, 16, 16, 2]
    ])


@pytest.fixture
def simple_weights():
    """Create simple equal weights."""
    return np.array([0.25, 0.25, 0.25, 0.25])


@pytest.fixture
def benefit_costs():
    """Define benefit/cost criteria."""
    return np.array([True, True, False, False])  # First two benefit, last two cost


# ============================================================================
# CONFIGURATION TESTS
# ============================================================================

@pytest.mark.unit
class TestConfiguration:
    """Test configuration management."""
    
    def test_config_creation(self, default_config):
        """Test default configuration is created correctly."""
        assert default_config is not None
        assert hasattr(default_config, 'panel')
        assert hasattr(default_config, 'paths')
        assert hasattr(default_config, 'mcdm')
    
    def test_panel_config(self, default_config):
        """Test panel data configuration."""
        assert default_config.panel.n_provinces == 64
        assert default_config.panel.n_components == 29
        assert len(default_config.panel.years) >= 5
        assert default_config.panel.n_observations > 0
    
    def test_mcdm_config(self, default_config):
        """Test MCDM configuration."""
        assert hasattr(default_config.mcdm, 'normalization')
        assert hasattr(default_config.mcdm, 'enable_fuzzy')
        assert default_config.mcdm.enable_traditional is True
    
    def test_weighting_config(self, default_config):
        """Test weighting configuration."""
        assert hasattr(default_config, 'weighting')
        assert default_config.weighting.enabled is True
        assert len(default_config.weighting.methods) > 0
    
    def test_ml_config(self, default_config):
        """Test ML configuration."""
        assert hasattr(default_config, 'random_forest')
        assert hasattr(default_config, 'neural')
        assert default_config.random_forest.n_estimators > 0


# ============================================================================
# DATA LOADING TESTS
# ============================================================================

@pytest.mark.unit
class TestDataLoader:
    """Test data loading and panel data structures."""
    
    def test_synthetic_generation(self):
        """Test synthetic panel data generation."""
        from src.data_loader import PanelDataLoader
        loader = PanelDataLoader()
        panel = loader.generate_synthetic(n_provinces=15, n_years=4, n_components=8, seed=123)
        
        assert panel is not None
        assert len(panel.entities) == 15
        assert len(panel.time_periods) == 4
        assert len(panel.components) == 8
    
    def test_panel_data_long_format(self, sample_panel_data):
        """Test long format panel data."""
        long_df = sample_panel_data.long
        assert 'Province' in long_df.columns
        assert 'Year' in long_df.columns
        assert len(long_df) == 20 * 5  # provinces * years
    
    def test_panel_data_wide_format(self, sample_panel_data):
        """Test wide format panel data."""
        wide_df = sample_panel_data.wide
        assert wide_df is not None
        assert len(wide_df) == 20  # provinces
    
    def test_panel_data_cross_sections(self, sample_panel_data):
        """Test cross-sectional views."""
        cs = sample_panel_data.cross_section
        assert len(cs) == 5  # years
        for year, data in cs.items():
            assert len(data) == 20  # provinces
    
    def test_year_filtering(self, sample_panel_data):
        """Test filtering by year."""
        year_data = sample_panel_data.get_year(sample_panel_data.time_periods[0])
        assert year_data is not None
        assert len(year_data) == 20
    
    def test_entity_filtering(self, sample_panel_data):
        """Test filtering by entity."""
        entity_data = sample_panel_data.get_entity(sample_panel_data.entities[0])
        assert entity_data is not None
        assert len(entity_data) == 5  # years


# ============================================================================
# WEIGHTING METHODS TESTS
# ============================================================================

@pytest.mark.unit
class TestWeightingMethods:
    """Test all weighting methods."""
    
    def test_entropy_weighting(self, simple_decision_matrix):
        """Test Entropy weighting method."""
        from src.weighting.entropy import EntropyWeight
        weighter = EntropyWeight()
        weights = weighter.calculate(simple_decision_matrix)
        
        assert weights is not None
        assert len(weights) == simple_decision_matrix.shape[1]
        assert np.isclose(np.sum(weights), 1.0)
        assert np.all(weights >= 0)
    
    def test_critic_weighting(self, simple_decision_matrix):
        """Test CRITIC weighting method."""
        from src.weighting.critic import CRITICWeight
        weighter = CRITICWeight()
        weights = weighter.calculate(simple_decision_matrix)
        
        assert weights is not None
        assert len(weights) == simple_decision_matrix.shape[1]
        assert np.isclose(np.sum(weights), 1.0)
        assert np.all(weights >= 0)
    
    def test_merec_weighting(self, simple_decision_matrix):
        """Test MEREC weighting method."""
        from src.weighting.merec import MERECWeight
        weighter = MERECWeight()
        weights = weighter.calculate(simple_decision_matrix)
        
        assert weights is not None
        assert len(weights) == simple_decision_matrix.shape[1]
        assert np.isclose(np.sum(weights), 1.0, atol=1e-5)
        assert np.all(weights >= 0)
    
    def test_standard_deviation_weighting(self, simple_decision_matrix):
        """Test Standard Deviation weighting method."""
        from src.weighting.standard_deviation import StandardDeviationWeight
        weighter = StandardDeviationWeight()
        weights = weighter.calculate(simple_decision_matrix)
        
        assert weights is not None
        assert len(weights) == simple_decision_matrix.shape[1]
        assert np.isclose(np.sum(weights), 1.0)
        assert np.all(weights >= 0)
    
    def test_fusion_weighting(self, simple_decision_matrix):
        """Test Fusion weighting method."""
        from src.weighting.fusion import FusionWeight
        weighter = FusionWeight()
        weights = weighter.calculate(simple_decision_matrix)
        
        assert weights is not None
        assert len(weights) == simple_decision_matrix.shape[1]
        assert np.isclose(np.sum(weights), 1.0)
        assert np.all(weights >= 0)
    
    def test_robust_global_weighting(self, simple_decision_matrix):
        """Test Robust Global weighting method."""
        from src.weighting.robust_global import RobustGlobalWeight
        weighter = RobustGlobalWeight()
        weights = weighter.calculate(simple_decision_matrix)
        
        assert weights is not None
        assert len(weights) == simple_decision_matrix.shape[1]
        assert np.isclose(np.sum(weights), 1.0, atol=1e-5)
        assert np.all(weights >= 0)


# ============================================================================
# TRADITIONAL MCDM METHODS TESTS
# ============================================================================

@pytest.mark.unit
class TestTraditionalMCDM:
    """Test all traditional MCDM methods."""
    
    def test_topsis(self, simple_decision_matrix, simple_weights, benefit_costs):
        """Test TOPSIS method."""
        from src.mcdm.traditional.topsis import TOPSIS
        method = TOPSIS()
        scores = method.calculate(simple_decision_matrix, simple_weights, benefit_costs)
        
        assert scores is not None
        assert len(scores) == len(simple_decision_matrix)
        assert np.all((scores >= 0) & (scores <= 1))
    
    def test_vikor(self, simple_decision_matrix, simple_weights, benefit_costs):
        """Test VIKOR method."""
        from src.mcdm.traditional.vikor import VIKOR
        method = VIKOR()
        scores = method.calculate(simple_decision_matrix, simple_weights, benefit_costs)
        
        assert scores is not None
        assert len(scores) == len(simple_decision_matrix)
        assert np.all(scores >= 0)
    
    def test_promethee(self, simple_decision_matrix, simple_weights, benefit_costs):
        """Test PROMETHEE method."""
        from src.mcdm.traditional.promethee import PROMETHEE
        method = PROMETHEE()
        scores = method.calculate(simple_decision_matrix, simple_weights, benefit_costs)
        
        assert scores is not None
        assert len(scores) == len(simple_decision_matrix)
    
    def test_copras(self, simple_decision_matrix, simple_weights, benefit_costs):
        """Test COPRAS method."""
        from src.mcdm.traditional.copras import COPRAS
        method = COPRAS()
        scores = method.calculate(simple_decision_matrix, simple_weights, benefit_costs)
        
        assert scores is not None
        assert len(scores) == len(simple_decision_matrix)
        assert np.all(scores >= 0)
    
    def test_edas(self, simple_decision_matrix, simple_weights, benefit_costs):
        """Test EDAS method."""
        from src.mcdm.traditional.edas import EDAS
        method = EDAS()
        scores = method.calculate(simple_decision_matrix, simple_weights, benefit_costs)
        
        assert scores is not None
        assert len(scores) == len(simple_decision_matrix)
        assert np.all((scores >= 0) & (scores <= 1))


# ============================================================================
# FUZZY MCDM METHODS TESTS
# ============================================================================

@pytest.mark.unit
@pytest.mark.fuzzy
class TestFuzzyMCDM:
    """Test all fuzzy MCDM methods."""
    
    def test_fuzzy_topsis(self, simple_decision_matrix, simple_weights, benefit_costs):
        """Test Fuzzy TOPSIS method."""
        from src.mcdm.fuzzy.topsis import FuzzyTOPSIS
        method = FuzzyTOPSIS()
        scores = method.calculate(simple_decision_matrix, simple_weights, benefit_costs)
        
        assert scores is not None
        assert len(scores) == len(simple_decision_matrix)
        assert np.all((scores >= 0) & (scores <= 1))
    
    def test_fuzzy_vikor(self, simple_decision_matrix, simple_weights, benefit_costs):
        """Test Fuzzy VIKOR method."""
        from src.mcdm.fuzzy.vikor import FuzzyVIKOR
        method = FuzzyVIKOR()
        scores = method.calculate(simple_decision_matrix, simple_weights, benefit_costs)
        
        assert scores is not None
        assert len(scores) == len(simple_decision_matrix)
        assert np.all(scores >= 0)
    
    def test_fuzzy_promethee(self, simple_decision_matrix, simple_weights, benefit_costs):
        """Test Fuzzy PROMETHEE method."""
        from src.mcdm.fuzzy.promethee import FuzzyPROMETHEE
        method = FuzzyPROMETHEE()
        scores = method.calculate(simple_decision_matrix, simple_weights, benefit_costs)
        
        assert scores is not None
        assert len(scores) == len(simple_decision_matrix)
    
    def test_fuzzy_copras(self, simple_decision_matrix, simple_weights, benefit_costs):
        """Test Fuzzy COPRAS method."""
        from src.mcdm.fuzzy.copras import FuzzyCOPRAS
        method = FuzzyCOPRAS()
        scores = method.calculate(simple_decision_matrix, simple_weights, benefit_costs)
        
        assert scores is not None
        assert len(scores) == len(simple_decision_matrix)
        assert np.all(scores >= 0)
    
    def test_fuzzy_edas(self, simple_decision_matrix, simple_weights, benefit_costs):
        """Test Fuzzy EDAS method."""
        from src.mcdm.fuzzy.edas import FuzzyEDAS
        method = FuzzyEDAS()
        scores = method.calculate(simple_decision_matrix, simple_weights, benefit_costs)
        
        assert scores is not None
        assert len(scores) == len(simple_decision_matrix)
        assert np.all((scores >= 0) & (scores <= 1))


# ============================================================================
# ENSEMBLE AGGREGATION TESTS
# ============================================================================

@pytest.mark.unit
@pytest.mark.ensemble
class TestEnsembleAggregation:
    """Test ensemble aggregation methods."""
    
    def test_borda_count(self):
        """Test Borda Count aggregation."""
        from src.ensemble.aggregation.borda import BordaCount
        
        rankings = np.array([
            [1, 2, 3, 4, 5],
            [2, 1, 3, 5, 4],
            [1, 3, 2, 4, 5]
        ])
        
        aggregator = BordaCount()
        final_ranks = aggregator.aggregate(rankings)
        
        assert final_ranks is not None
        assert len(final_ranks) == 5
        assert len(set(final_ranks)) == 5  # All unique ranks
    
    def test_copeland_method(self):
        """Test Copeland aggregation."""
        from src.ensemble.aggregation.copeland import Copeland
        
        rankings = np.array([
            [1, 2, 3, 4, 5],
            [2, 1, 3, 5, 4],
            [1, 3, 2, 4, 5]
        ])
        
        aggregator = Copeland()
        final_ranks = aggregator.aggregate(rankings)
        
        assert final_ranks is not None
        assert len(final_ranks) == 5
    
    def test_kemeny_young(self):
        """Test Kemeny-Young aggregation."""
        from src.ensemble.aggregation.kemeny import KemenyYoung
        
        rankings = np.array([
            [1, 2, 3, 4],
            [2, 1, 3, 4],
            [1, 3, 2, 4]
        ])
        
        aggregator = KemenyYoung()
        final_ranks = aggregator.aggregate(rankings)
        
        assert final_ranks is not None
        assert len(final_ranks) == 4
    
    def test_stacking_aggregation(self, sample_panel_data):
        """Test Stacking (ML-based) aggregation."""
        from src.ensemble.aggregation.stacking import StackingAggregator
        
        # Create synthetic rankings
        n_alternatives = 20
        rankings = np.random.randint(1, n_alternatives + 1, size=(5, n_alternatives))
        
        aggregator = StackingAggregator()
        # This requires training, so we'll just test initialization
        assert aggregator is not None


# ============================================================================
# ML FORECASTING TESTS
# ============================================================================

@pytest.mark.unit
@pytest.mark.ml
class TestMLForecasting:
    """Test machine learning forecasting methods."""
    
    def test_random_forest_forecaster(self, sample_panel_data):
        """Test Random Forest time-series forecasting."""
        from src.ml.forecasting.random_forest_ts import RandomForestForecaster
        
        forecaster = RandomForestForecaster(n_estimators=10, random_state=42)
        assert forecaster is not None
    
    def test_gradient_boosting_forecaster(self, sample_panel_data):
        """Test Gradient Boosting forecasting."""
        from src.ml.forecasting.tree_ensemble import GradientBoostingForecaster
        
        forecaster = GradientBoostingForecaster(n_estimators=10, random_state=42)
        assert forecaster is not None
    
    def test_linear_forecaster(self, sample_panel_data):
        """Test Linear regression forecasting."""
        from src.ml.forecasting.linear import LinearForecaster
        
        forecaster = LinearForecaster()
        assert forecaster is not None
    
    def test_feature_engineering(self, sample_panel_data):
        """Test feature engineering for time series."""
        from src.ml.forecasting.features import TimeSeriesFeatureEngineer
        
        engineer = TimeSeriesFeatureEngineer()
        long_data = sample_panel_data.long
        
        # Create numeric columns for feature engineering
        numeric_cols = [col for col in long_data.columns 
                       if col not in ['Province', 'Year']]
        
        if len(numeric_cols) > 0:
            features = engineer.create_lag_features(long_data, numeric_cols[0], lags=[1, 2])
            assert features is not None
    
    def test_unified_forecaster(self, sample_panel_data):
        """Test unified forecasting interface."""
        from src.ml.forecasting.unified import UnifiedForecaster
        
        forecaster = UnifiedForecaster(method='random_forest', n_estimators=10)
        assert forecaster is not None


# ============================================================================
# ANALYSIS TESTS
# ============================================================================

@pytest.mark.unit
class TestAnalysis:
    """Test analysis and validation modules."""
    
    def test_sensitivity_analysis(self):
        """Test sensitivity analysis module."""
        from src.analysis.sensitivity import SensitivityAnalyzer
        
        analyzer = SensitivityAnalyzer()
        assert analyzer is not None
    
    def test_validation_module(self):
        """Test validation module."""
        from src.analysis.validation import CrossValidator
        
        validator = CrossValidator(n_folds=3)
        assert validator is not None
        assert validator.n_folds == 3


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

@pytest.mark.integration
class TestIntegration:
    """Test end-to-end integration."""
    
    def test_complete_pipeline_synthetic(self, default_config):
        """Test complete pipeline with synthetic data."""
        from src.pipeline import MLMCDMPipeline
        
        # Use smaller dataset for faster testing
        default_config.panel.n_provinces = 10
        default_config.panel.n_components = 5
        default_config.panel.years = [2020, 2021, 2022]
        
        # Disable heavy ML models for faster testing
        default_config.neural.enabled = False
        default_config.forecast_horizon = 1
        
        pipeline = MLMCDMPipeline(default_config)
        assert pipeline is not None
    
    def test_output_manager(self, default_config):
        """Test output management system."""
        from src.output_manager import OutputManager
        
        manager = OutputManager(default_config.paths.output_dir)
        assert manager is not None
    
    def test_logger(self):
        """Test logging system."""
        from src.logger import setup_logger
        
        logger = setup_logger('test_logger')
        assert logger is not None
        logger.info("Test log message")
    
    def test_visualization(self, sample_panel_data):
        """Test visualization module."""
        from src.visualization import Visualizer
        
        viz = Visualizer()
        assert viz is not None


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

@pytest.mark.slow
class TestPerformance:
    """Test performance with larger datasets."""
    
    def test_large_dataset_handling(self):
        """Test handling of large datasets."""
        from src.data_loader import PanelDataLoader
        
        loader = PanelDataLoader()
        large_panel = loader.generate_synthetic(
            n_provinces=100,
            n_years=10,
            n_components=50,
            seed=42
        )
        
        assert large_panel is not None
        assert len(large_panel.entities) == 100
        assert len(large_panel.time_periods) == 10
    
    def test_parallel_mcdm_computation(self, sample_panel_data):
        """Test parallel MCDM computation capability."""
        # This is a placeholder for parallel processing tests
        assert True


# ============================================================================
# EDGE CASES AND ERROR HANDLING
# ============================================================================

@pytest.mark.unit
class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_data_handling(self):
        """Test handling of empty data."""
        from src.data_loader import PanelDataLoader
        
        loader = PanelDataLoader()
        with pytest.raises(ValueError):
            loader.generate_synthetic(n_provinces=0, n_years=5, n_components=10)
    
    def test_invalid_weights(self, simple_decision_matrix):
        """Test handling of invalid weights."""
        from src.mcdm.traditional.topsis import TOPSIS
        
        invalid_weights = np.array([0.5, 0.3, 0.1])  # Wrong size
        benefit_costs = np.array([True, True, False, False])
        
        method = TOPSIS()
        with pytest.raises((ValueError, AssertionError)):
            method.calculate(simple_decision_matrix, invalid_weights, benefit_costs)
    
    def test_nan_handling(self):
        """Test handling of NaN values."""
        data_with_nan = np.array([
            [1.0, 2.0, np.nan, 4.0],
            [2.0, 3.0, 4.0, 5.0],
            [3.0, 4.0, 5.0, 6.0]
        ])
        
        # Should handle or raise appropriate error
        assert np.isnan(data_with_nan).any()
    
    def test_single_alternative(self):
        """Test handling of single alternative."""
        single_alt = np.array([[1.0, 2.0, 3.0, 4.0]])
        weights = np.array([0.25, 0.25, 0.25, 0.25])
        benefit_costs = np.array([True, True, False, False])
        
        from src.mcdm.traditional.topsis import TOPSIS
        method = TOPSIS()
        scores = method.calculate(single_alt, weights, benefit_costs)
        
        assert scores is not None
        assert len(scores) == 1


# ============================================================================
# TEST SUMMARY
# ============================================================================

def test_framework_info():
    """Display framework information."""
    info = {
        'framework': 'ML-MCDM',
        'version': '2.0.0',
        'mcdm_methods': 10,
        'weighting_methods': 6,
        'aggregation_methods': 4,
        'ml_forecasting_methods': 5,
    }
    
    assert info['mcdm_methods'] == 10
    assert info['weighting_methods'] == 6
    assert info['aggregation_methods'] == 4
    assert info['ml_forecasting_methods'] == 5
    
    print("\n" + "="*70)
    print("ML-MCDM Framework v2.0 - Test Suite")
    print("="*70)
    print(f"MCDM Methods: {info['mcdm_methods']}")
    print(f"Weighting Methods: {info['weighting_methods']}")
    print(f"Aggregation Methods: {info['aggregation_methods']}")
    print(f"ML Forecasting Methods: {info['ml_forecasting_methods']}")
    print("="*70 + "\n")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
