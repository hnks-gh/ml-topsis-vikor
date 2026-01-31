# -*- coding: utf-8 -*-
"""Core unit tests for ML-MCDM framework."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestConfig:
    """Test configuration module."""
    
    def test_default_config_creation(self):
        from src.config import get_default_config
        config = get_default_config()
        assert config is not None
        assert config.panel.n_provinces == 64
        assert config.panel.n_components == 20
    
    def test_config_panel_settings(self):
        from src.config import get_default_config
        config = get_default_config()
        assert len(config.panel.years) == 5
        assert config.panel.n_observations == 64 * 5


class TestDataLoader:
    """Test data loading functionality."""
    
    def test_synthetic_data_generation(self):
        from src.data_loader import PanelDataLoader
        loader = PanelDataLoader()
        panel_data = loader.generate_synthetic(
            n_provinces=10,
            n_years=3,
            n_components=5
        )
        
        assert panel_data is not None
        assert len(panel_data.entities) == 10
        assert len(panel_data.time_periods) == 3
        assert len(panel_data.components) == 5
    
    def test_panel_data_views(self):
        from src.data_loader import PanelDataLoader
        loader = PanelDataLoader()
        panel_data = loader.generate_synthetic(
            n_provinces=5,
            n_years=2,
            n_components=3
        )
        
        # Check long format
        assert 'Province' in panel_data.long.columns
        assert 'Year' in panel_data.long.columns
        
        # Check wide format
        assert panel_data.wide is not None
        
        # Check cross-sections
        assert len(panel_data.cross_section) == 2


class TestMCDM:
    """Test MCDM methods."""
    
    def test_entropy_weights(self):
        from src.mcdm import EntropyWeightCalculator
        
        # Create test data
        data = pd.DataFrame({
            'A': [0.1, 0.2, 0.3, 0.4, 0.5],
            'B': [0.5, 0.4, 0.3, 0.2, 0.1],
            'C': [0.3, 0.3, 0.3, 0.3, 0.3]
        })
        
        calc = EntropyWeightCalculator()
        weights = calc.calculate(data)
        
        assert len(weights) == 3
        assert abs(sum(weights.values()) - 1.0) < 0.001
    
    def test_topsis_calculation(self):
        from src.mcdm import TOPSISCalculator
        
        data = pd.DataFrame({
            'A': [0.8, 0.6, 0.4, 0.2],
            'B': [0.2, 0.4, 0.6, 0.8],
        }, index=['P1', 'P2', 'P3', 'P4'])
        
        weights = {'A': 0.5, 'B': 0.5}
        
        calc = TOPSISCalculator()
        result = calc.calculate(data, weights)
        
        assert len(result.scores) == 4
        assert all(0 <= s <= 1 for s in result.scores)
    
    def test_vikor_calculation(self):
        from src.mcdm import VIKORCalculator
        
        data = pd.DataFrame({
            'A': [0.8, 0.6, 0.4, 0.2],
            'B': [0.2, 0.4, 0.6, 0.8],
        }, index=['P1', 'P2', 'P3', 'P4'])
        
        weights = {'A': 0.5, 'B': 0.5}
        
        calc = VIKORCalculator()
        result = calc.calculate(data, weights)
        
        assert hasattr(result, 'Q')
        assert hasattr(result, 'S')
        assert hasattr(result, 'R')
    
    def test_promethee_calculation(self):
        """Test PROMETHEE method."""
        from src.mcdm import PROMETHEECalculator
        
        data = pd.DataFrame({
            'A': [0.8, 0.6, 0.4, 0.2, 0.5],
            'B': [0.2, 0.4, 0.6, 0.8, 0.5],
            'C': [0.5, 0.3, 0.7, 0.4, 0.6],
        }, index=['P1', 'P2', 'P3', 'P4', 'P5'])
        
        weights = {'A': 0.4, 'B': 0.3, 'C': 0.3}
        
        calc = PROMETHEECalculator(preference_function="vshape")
        result = calc.calculate(data, weights)
        
        assert hasattr(result, 'phi_positive')
        assert hasattr(result, 'phi_negative')
        assert hasattr(result, 'phi_net')
        assert hasattr(result, 'ranks_promethee_ii')
        assert len(result.phi_net) == 5
    
    def test_copras_calculation(self):
        """Test COPRAS method."""
        from src.mcdm import COPRASCalculator
        
        data = pd.DataFrame({
            'A': [0.8, 0.6, 0.4, 0.2],
            'B': [0.2, 0.4, 0.6, 0.8],
            'C': [0.5, 0.3, 0.7, 0.4],
        }, index=['P1', 'P2', 'P3', 'P4'])
        
        weights = {'A': 0.4, 'B': 0.3, 'C': 0.3}
        
        calc = COPRASCalculator(cost_criteria=['B'])
        result = calc.calculate(data, weights)
        
        assert hasattr(result, 'S_plus')
        assert hasattr(result, 'S_minus')
        assert hasattr(result, 'Q')
        assert hasattr(result, 'utility_degree')
        assert hasattr(result, 'ranks')
        assert len(result.ranks) == 4
        # Utility degree should be percentage (0-100)
        assert result.utility_degree.max() == 100
    
    def test_edas_calculation(self):
        """Test EDAS method."""
        from src.mcdm import EDASCalculator
        
        data = pd.DataFrame({
            'A': [0.8, 0.6, 0.4, 0.2],
            'B': [0.2, 0.4, 0.6, 0.8],
            'C': [0.5, 0.3, 0.7, 0.4],
        }, index=['P1', 'P2', 'P3', 'P4'])
        
        weights = {'A': 0.4, 'B': 0.3, 'C': 0.3}
        
        calc = EDASCalculator(cost_criteria=['B'])
        result = calc.calculate(data, weights)
        
        assert hasattr(result, 'PDA')
        assert hasattr(result, 'NDA')
        assert hasattr(result, 'AS')
        assert hasattr(result, 'average_solution')
        assert hasattr(result, 'ranks')
        assert len(result.ranks) == 4
        # Appraisal score should be between 0 and 1
        assert all(0 <= s <= 1 for s in result.AS)


class TestPipeline:
    """Test main pipeline."""
    
    def test_pipeline_initialization(self):
        from src.main import MLTOPSISPipeline
        from src.config import get_default_config
        
        config = get_default_config()
        pipeline = MLTOPSISPipeline(config)
        
        assert pipeline is not None
        assert pipeline.config is not None
    
    def test_full_pipeline_small(self):
        from src.main import MLTOPSISPipeline
        from src.config import get_default_config
        
        config = get_default_config()
        config.panel.n_provinces = 10
        config.panel.years = [2020, 2021, 2022]
        config.panel.n_components = 5
        config.lstm.enabled = False  # Faster test
        
        pipeline = MLTOPSISPipeline(config)
        result = pipeline.run()
        
        assert result is not None
        assert len(result.panel_data.entities) == 10
        assert result.topsis_scores is not None
        assert result.aggregated_ranking is not None


class TestAdvancedMLForecasting:
    """Test advanced ML forecasting modules."""
    
    def test_temporal_feature_engineer(self):
        """Test feature engineering for time series."""
        from src.ml.advanced_forecasting import TemporalFeatureEngineer
        from src.data_loader import PanelDataLoader
        
        loader = PanelDataLoader()
        panel_data = loader.generate_synthetic(
            n_provinces=5,
            n_years=4,
            n_components=3
        )
        
        engineer = TemporalFeatureEngineer(
            lag_periods=[1],
            rolling_windows=[2],
            include_momentum=True,
            include_cross_entity=True
        )
        
        X_train, y_train, X_pred, _ = engineer.fit_transform(
            panel_data, panel_data.years[-1]
        )
        
        assert X_train is not None
        assert y_train is not None
        assert X_pred is not None
        assert len(X_pred) == 5  # One prediction per entity
    
    def test_gradient_boosting_forecaster(self):
        """Test gradient boosting forecaster."""
        from src.ml.advanced_forecasting import GradientBoostingForecaster
        
        np.random.seed(42)
        X = np.random.rand(50, 10)
        y = np.random.rand(50)
        
        forecaster = GradientBoostingForecaster(
            n_estimators=50,
            max_depth=3
        )
        forecaster.fit(X, y)
        
        predictions = forecaster.predict(X[:5])
        importance = forecaster.get_feature_importance()
        
        assert len(predictions) == 5
        assert len(importance) == 10
    
    def test_advanced_ml_forecaster(self):
        """Test the full advanced ML forecaster."""
        from src.ml.advanced_forecasting import AdvancedMLForecaster
        from src.data_loader import PanelDataLoader
        
        loader = PanelDataLoader()
        panel_data = loader.generate_synthetic(
            n_provinces=8,
            n_years=4,
            n_components=3
        )
        
        forecaster = AdvancedMLForecaster(
            include_gb=True,
            include_rf=True,
            include_et=False,
            include_bayesian=True,
            include_huber=False,
            cv_splits=2
        )
        
        result = forecaster.fit_predict(panel_data, panel_data.components[:2])
        
        assert result.predictions is not None
        assert len(result.predictions) == 8
        assert len(result.ensemble_weights) > 0


class TestNeuralForecasting:
    """Test neural network forecasting."""
    
    def test_dense_layer(self):
        """Test dense layer operations."""
        from src.ml.neural_forecasting import DenseLayer
        
        layer = DenseLayer(10, 5, activation='relu')
        X = np.random.rand(4, 10)
        
        output = layer.forward(X)
        
        assert output.shape == (4, 5)
        assert np.all(output >= 0)  # ReLU output
    
    def test_neural_forecaster(self):
        """Test neural network forecaster."""
        from src.ml.neural_forecasting import NeuralForecaster
        
        np.random.seed(42)
        X = np.random.rand(30, 10)
        y = np.random.rand(30)
        
        forecaster = NeuralForecaster(
            hidden_dims=[16, 8],
            n_epochs=20,
            batch_size=8
        )
        forecaster.fit(X, y)
        
        predictions = forecaster.predict(X[:5])
        importance = forecaster.get_feature_importance()
        
        assert len(predictions) == 5
        assert importance is not None
    
    def test_neural_ensemble(self):
        """Test neural ensemble forecaster."""
        from src.ml.neural_forecasting import NeuralEnsembleForecaster
        
        np.random.seed(42)
        X = np.random.rand(30, 10)
        y = np.random.rand(30)
        
        ensemble = NeuralEnsembleForecaster(
            n_mlp_models=2,
            include_attention=False
        )
        ensemble.fit(X, y)
        
        predictions = ensemble.predict(X[:5])
        pred_with_unc = ensemble.predict_with_uncertainty(X[:5])
        
        assert len(predictions) == 5
        assert len(pred_with_unc) == 2  # (predictions, uncertainty)


class TestUnifiedForecasting:
    """Test unified forecasting system."""
    
    def test_forecast_mode_enum(self):
        """Test forecast mode enumeration."""
        from src.ml.unified_forecasting import ForecastMode
        
        assert ForecastMode.FAST.value == "fast"
        assert ForecastMode.BALANCED.value == "balanced"
        assert ForecastMode.ACCURATE.value == "accurate"
    
    def test_unified_forecaster(self):
        """Test unified forecaster with synthetic data."""
        from src.ml.unified_forecasting import UnifiedForecaster, ForecastMode
        from src.data_loader import PanelDataLoader
        
        loader = PanelDataLoader()
        panel_data = loader.generate_synthetic(
            n_provinces=6,
            n_years=4,
            n_components=3
        )
        
        forecaster = UnifiedForecaster(
            mode=ForecastMode.FAST,
            include_neural=False,  # Faster test
            cv_folds=2,
            verbose=False
        )
        
        result = forecaster.forecast(panel_data, panel_data.components[:2])
        
        assert result.predictions is not None
        assert len(result.predictions) == 6
        assert result.uncertainty is not None
        assert len(result.model_contributions) > 0
    
    def test_convenience_function(self):
        """Test forecast_next_year convenience function."""
        from src.ml.unified_forecasting import forecast_next_year
        from src.data_loader import PanelDataLoader
        
        loader = PanelDataLoader()
        panel_data = loader.generate_synthetic(
            n_provinces=5,
            n_years=4,
            n_components=2
        )
        
        result = forecast_next_year(panel_data, mode="fast", verbose=False)
        
        assert result is not None
        assert result.predictions is not None


class TestPipelineV2:
    """Test the new v2 pipeline."""
    
    def test_pipeline_v2_initialization(self):
        """Test v2 pipeline initialization."""
        from src.pipeline_v2 import PipelineV2
        
        pipeline = PipelineV2(
            output_dir="outputs_test",
            ml_mode="fast",
            verbose=False
        )
        
        assert pipeline is not None
        assert pipeline.ml_mode == "fast"
    
    def test_pipeline_v2_full_run(self):
        """Test full v2 pipeline execution."""
        from src.pipeline_v2 import PipelineV2
        from src.data_loader import PanelDataLoader
        
        loader = PanelDataLoader()
        panel_data = loader.generate_synthetic(
            n_provinces=6,
            n_years=4,
            n_components=4
        )
        
        pipeline = PipelineV2(
            output_dir="outputs_test",
            ml_mode="fast",
            verbose=False
        )
        
        result = pipeline.run(panel_data)
        
        assert result is not None
        assert len(result.traditional_mcdm) >= 3  # At least TOPSIS, VIKOR, PROMETHEE
        assert len(result.fuzzy_mcdm) >= 3
        assert result.final_rankings is not None
        assert result.execution_time > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
