# -*- coding: utf-8 -*-
"""
Basic tests for ML-MCDM framework.
"""

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


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
