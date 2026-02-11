# -*- coding: utf-8 -*-
"""
Comprehensive Test Suite for Weighting Module

Tests all components of the hybrid weighting pipeline:
- Individual methods (Entropy, CRITIC, MEREC, SD)
- Normalization
- Fusion
- Bootstrap
- Validation
- Full pipeline integration
"""

import pytest
import numpy as np
import pandas as pd
from typing import Dict

# Import all weighting components
from src.weighting import (
    HybridWeightingPipeline,
    EntropyWeightCalculator,
    CRITICWeightCalculator,
    MERECWeightCalculator,
    StandardDeviationWeightCalculator,
    GameTheoryWeightCombination,
    global_min_max_normalize,
    GlobalNormalizer,
    bayesian_bootstrap_weights,
    temporal_stability_verification,
    WeightResult,
    calculate_weights,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def simple_data():
    """Simple 5x3 decision matrix for basic tests."""
    return pd.DataFrame({
        'C1': [1.0, 2.0, 3.0, 4.0, 5.0],
        'C2': [10.0, 20.0, 30.0, 40.0, 50.0],
        'C3': [100.0, 200.0, 300.0, 400.0, 500.0],
    })


@pytest.fixture
def panel_data():
    """Panel data (long format) for pipeline tests."""
    np.random.seed(42)
    n_entities = 5
    n_years = 3
    n_criteria = 4
    
    data = {
        'Year': np.repeat(range(2020, 2020 + n_years), n_entities),
        'Entity': np.tile([f'E{i}' for i in range(n_entities)], n_years),
    }
    
    # Add criteria with some structure
    for i in range(n_criteria):
        values = np.random.rand(n_entities * n_years) * (i + 1) * 10
        data[f'C{i+1}'] = values
    
    return pd.DataFrame(data)


@pytest.fixture
def uniform_data():
    """Data with identical values (edge case)."""
    return pd.DataFrame({
        'C1': [5.0] * 10,
        'C2': [10.0] * 10,
        'C3': [15.0] * 10,
    })


# ============================================================================
# TEST INDIVIDUAL CALCULATORS
# ============================================================================

class TestIndividualCalculators:
    """Test each weighting method independently."""
    
    def test_entropy_weights(self, simple_data):
        """Test Entropy weight calculator."""
        calc = EntropyWeightCalculator()
        result = calc.calculate(simple_data)
        
        # Validate result structure
        assert isinstance(result, WeightResult)
        assert result.method == 'entropy'
        assert len(result.weights) == 3
        
        # Validate weights sum to 1
        weight_sum = sum(result.weights.values())
        assert abs(weight_sum - 1.0) < 1e-6
        
        # All weights should be positive
        assert all(w > 0 for w in result.weights.values())
    
    def test_critic_weights(self, simple_data):
        """Test CRITIC weight calculator."""
        calc = CRITICWeightCalculator()
        result = calc.calculate(simple_data)
        
        assert isinstance(result, WeightResult)
        assert result.method == 'critic'
        assert len(result.weights) == 3
        assert abs(sum(result.weights.values()) - 1.0) < 1e-6
        assert all(w > 0 for w in result.weights.values())
        
        # CRITIC should have correlation info in details
        assert 'correlation_matrix' in result.details
    
    def test_merec_weights(self, simple_data):
        """Test MEREC weight calculator."""
        calc = MERECWeightCalculator()
        result = calc.calculate(simple_data)
        
        assert isinstance(result, WeightResult)
        assert result.method == 'merec'
        assert len(result.weights) == 3
        assert abs(sum(result.weights.values()) - 1.0) < 1e-6
        assert all(w > 0 for w in result.weights.values())
        
        # MEREC should have removal effects
        assert 'removal_effects' in result.details
    
    def test_standard_deviation_weights(self, simple_data):
        """Test Standard Deviation weight calculator."""
        calc = StandardDeviationWeightCalculator()
        result = calc.calculate(simple_data)
        
        assert isinstance(result, WeightResult)
        assert result.method == 'std_dev'
        assert len(result.weights) == 3
        assert abs(sum(result.weights.values()) - 1.0) < 1e-6
        assert all(w > 0 for w in result.weights.values())
        
        # SD should have coefficient of variation
        assert 'coefficient_of_variation' in result.details
    
    def test_uniform_data_handling(self, uniform_data):
        """Test that calculators handle constant columns gracefully."""
        # Entropy should handle uniform distribution
        entropy_result = EntropyWeightCalculator().calculate(uniform_data)
        assert all(abs(w - 1/3) < 0.05 for w in entropy_result.weights.values())
        
        # CRITIC should handle zero std
        critic_result = CRITICWeightCalculator().calculate(uniform_data)
        assert abs(sum(critic_result.weights.values()) - 1.0) < 1e-6


# ============================================================================
# TEST NORMALIZATION
# ============================================================================

class TestNormalization:
    """Test normalization functions and classes."""
    
    def test_global_min_max_normalize_function(self):
        """Test functional normalization."""
        X = np.array([[1, 10], [2, 20], [3, 30]])
        X_norm = global_min_max_normalize(X, epsilon=1e-10)
        
        # Check shape preserved
        assert X_norm.shape == X.shape
        
        # Check min/max of each column
        assert np.allclose(X_norm.min(axis=0), 1e-10, atol=1e-9)
        assert np.allclose(X_norm.max(axis=0), 1.0, atol=1e-9)
        
        # Check monotonicity preserved
        for col in range(X.shape[1]):
            assert np.all(np.diff(X_norm[:, col]) > 0)
    
    def test_global_normalizer_class(self):
        """Test stateful GlobalNormalizer."""
        normalizer = GlobalNormalizer(epsilon=1e-10)
        
        # Training data
        X_train = np.array([[1, 10], [2, 20], [3, 30]])
        X_train_norm = normalizer.fit_transform(X_train)
        
        assert normalizer.is_fitted_
        assert X_train_norm.shape == X_train.shape
        
        # Test data (using same transformation)
        X_test = np.array([[4, 40]])
        X_test_norm = normalizer.transform(X_test)
        
        # Test data normalized with training min/max
        # Should be > 1.0 since 4 > max(1,2,3)=3
        assert X_test_norm[0, 0] > 1.0
        assert X_test_norm[0, 1] > 1.0
    
    def test_normalizer_inverse_transform(self):
        """Test inverse normalization."""
        normalizer = GlobalNormalizer(epsilon=1e-10)
        X_original = np.array([[1, 10], [2, 20], [3, 30]])
        
        X_norm = normalizer.fit_transform(X_original)
        X_recovered = normalizer.inverse_transform(X_norm)
        
        # Should recover original values (within epsilon tolerance)
        assert np.allclose(X_recovered, X_original, atol=1e-8)


# ============================================================================
# TEST FUSION
# ============================================================================

class TestGameTheoryWeightCombination:
    """Test Game Theory Weight Combination (GTWC) fusion."""
    
    def test_gtwc_basic(self):
        """Test basic GTWC fusion with 4 weight vectors."""
        gtwc = GameTheoryWeightCombination()
        
        weight_vectors = {
            'entropy': np.array([0.20, 0.30, 0.50]),
            'std_dev': np.array([0.25, 0.35, 0.40]),
            'critic':  np.array([0.30, 0.40, 0.30]),
            'merec':   np.array([0.40, 0.30, 0.30]),
        }
        
        W_final, details = gtwc.combine(weight_vectors)
        
        # Weights must sum to 1 and be positive
        assert W_final.shape == (3,)
        assert abs(W_final.sum() - 1.0) < 1e-6
        assert np.all(W_final > 0)
        
        # Details structure
        assert details['method'] == 'game_theory_weight_combination'
        assert 'phase_2' in details
        assert 'phase_3' in details
        assert 'phase_4' in details
    
    def test_gtwc_alpha_coefficients(self):
        """Test that GTWC alpha coefficients are valid."""
        gtwc = GameTheoryWeightCombination()
        
        weight_vectors = {
            'entropy': np.array([0.10, 0.20, 0.30, 0.40]),
            'std_dev': np.array([0.15, 0.25, 0.25, 0.35]),
            'critic':  np.array([0.30, 0.30, 0.20, 0.20]),
            'merec':   np.array([0.25, 0.25, 0.25, 0.25]),
        }
        
        W_final, details = gtwc.combine(weight_vectors)
        
        alpha_d = details['phase_3']['alpha_dispersion']
        alpha_i = details['phase_3']['alpha_interaction']
        
        # Alphas should be non-negative and sum to 1
        assert alpha_d >= 0
        assert alpha_i >= 0
        assert abs(alpha_d + alpha_i - 1.0) < 1e-6
    
    def test_gtwc_group_a_geometric_mean(self):
        """Test Group A uses geometric mean (not raw product)."""
        gtwc = GameTheoryWeightCombination()
        
        # If entropy gives near-zero to criterion 1, SD gives high
        # Geometric mean should preserve some importance
        weight_vectors = {
            'entropy': np.array([0.01, 0.49, 0.50]),
            'std_dev': np.array([0.50, 0.30, 0.20]),
            'critic':  np.array([0.33, 0.33, 0.34]),
            'merec':   np.array([0.33, 0.34, 0.33]),
        }
        
        W_final, details = gtwc.combine(weight_vectors)
        W_GroupA = np.array(details['phase_2']['W_GroupA'])
        
        # Group A criterion 1 should NOT be near-zero (geometric mean protects it)
        # Raw product would give ~0.005, but sqrt gives ~0.07 before normalization
        assert W_GroupA[0] > 0.03, (
            f"Geometric mean should prevent zero-dominance, got {W_GroupA[0]:.4f}"
        )
    
    def test_gtwc_group_b_harmonic_mean(self):
        """Test Group B uses harmonic mean (conservative fusion)."""
        gtwc = GameTheoryWeightCombination()
        
        # CRITIC rates criterion 1 high, MEREC rates it low
        # Harmonic mean should pull toward the lower value
        weight_vectors = {
            'entropy': np.array([0.33, 0.33, 0.34]),
            'std_dev': np.array([0.33, 0.34, 0.33]),
            'critic':  np.array([0.60, 0.20, 0.20]),
            'merec':   np.array([0.10, 0.45, 0.45]),
        }
        
        W_final, details = gtwc.combine(weight_vectors)
        W_GroupB = np.array(details['phase_2']['W_GroupB'])
        
        # Harmonic mean of 0.60 and 0.10 is much closer to 0.10
        # Should be conservative (lower than arithmetic mean of 0.35)
        arithmetic_mean = (0.60 + 0.10) / 2.0
        assert W_GroupB[0] < arithmetic_mean * 1.1, (
            f"Harmonic mean should be conservative, got {W_GroupB[0]:.4f}"
        )
    
    def test_gtwc_identical_groups(self):
        """Test GTWC when both groups produce identical weights."""
        gtwc = GameTheoryWeightCombination()
        
        # All methods agree → both groups should be similar → equal alphas
        weight_vectors = {
            'entropy': np.array([0.25, 0.25, 0.25, 0.25]),
            'std_dev': np.array([0.25, 0.25, 0.25, 0.25]),
            'critic':  np.array([0.25, 0.25, 0.25, 0.25]),
            'merec':   np.array([0.25, 0.25, 0.25, 0.25]),
        }
        
        W_final, details = gtwc.combine(weight_vectors)
        
        # Final weights should be approximately uniform
        assert np.allclose(W_final, 0.25, atol=0.01)
    
    def test_gtwc_missing_key_raises(self):
        """Test that missing required weight vector raises KeyError."""
        gtwc = GameTheoryWeightCombination()
        
        weight_vectors = {
            'entropy': np.array([0.5, 0.5]),
            'std_dev': np.array([0.5, 0.5]),
            # Missing 'critic' and 'merec'
        }
        
        with pytest.raises(KeyError):
            gtwc.combine(weight_vectors)
    
    def test_gtwc_many_criteria(self):
        """Test GTWC with many criteria (realistic scenario)."""
        gtwc = GameTheoryWeightCombination()
        np.random.seed(42)
        
        n = 20  # 20 criteria
        weight_vectors = {
            'entropy': np.random.dirichlet(np.ones(n)),
            'std_dev': np.random.dirichlet(np.ones(n)),
            'critic':  np.random.dirichlet(np.ones(n)),
            'merec':   np.random.dirichlet(np.ones(n)),
        }
        
        W_final, details = gtwc.combine(weight_vectors)
        
        assert W_final.shape == (n,)
        assert abs(W_final.sum() - 1.0) < 1e-6
        assert np.all(W_final > 0)
        assert details['phase_3']['condition_number'] > 0


# ============================================================================
# TEST BOOTSTRAP
# ============================================================================

class TestBootstrap:
    """Test Bayesian Bootstrap functionality."""
    
    def test_bayesian_bootstrap_weights(self, simple_data):
        """Test bootstrap uncertainty quantification."""
        X_norm = global_min_max_normalize(simple_data.values)
        criteria_cols = simple_data.columns.tolist()
        
        def simple_calculator(X_df, cols):
            # Simple: just use entropy
            calc = EntropyWeightCalculator()
            result = calc.calculate(X_df)
            return np.array([result.weights[c] for c in cols])
        
        # Run with few iterations for speed
        results = bayesian_bootstrap_weights(
            X_norm=X_norm,
            criteria_cols=criteria_cols,
            weight_calculator=simple_calculator,
            n_iterations=50,
            seed=42
        )
        
        # Check output structure
        assert 'mean_weights' in results
        assert 'std_weights' in results
        assert 'ci_lower' in results
        assert 'ci_upper' in results
        assert 'all_weights' in results
        assert 'convergence_rate' in results
        
        # Check shapes
        assert results['mean_weights'].shape == (3,)
        assert results['std_weights'].shape == (3,)
        assert results['all_weights'].shape == (50, 3)
        
        # Check convergence
        assert results['convergence_rate'] > 0.9  # Most iterations should succeed
        
        # Check CI bounds
        assert np.all(results['ci_lower'] <= results['mean_weights'])
        assert np.all(results['mean_weights'] <= results['ci_upper'])


# ============================================================================
# TEST VALIDATION
# ============================================================================

class TestValidation:
    """Test temporal stability verification."""
    
    def test_temporal_stability_verification(self, panel_data):
        """Test split-half stability check."""
        X_raw = panel_data[['C1', 'C2', 'C3', 'C4']].values
        time_values = panel_data['Year'].values
        criteria_cols = ['C1', 'C2', 'C3', 'C4']
        
        def simple_calculator(X, cols):
            X_norm = global_min_max_normalize(X)
            X_df = pd.DataFrame(X_norm, columns=cols)
            calc = EntropyWeightCalculator()
            result = calc.calculate(X_df)
            return np.array([result.weights[c] for c in cols])
        
        results = temporal_stability_verification(
            X_raw=X_raw,
            time_values=time_values,
            criteria_cols=criteria_cols,
            weight_calculator=simple_calculator,
            stability_threshold=0.95
        )
        
        # Check output structure
        assert 'cosine_similarity' in results
        assert 'spearman_correlation' in results
        assert 'spearman_pvalue' in results
        assert 'is_stable' in results
        assert 'split_point' in results
        
        # Check value ranges
        assert -1 <= results['cosine_similarity'] <= 1 or np.isnan(results['cosine_similarity'])
        assert -1 <= results['spearman_correlation'] <= 1 or np.isnan(results['spearman_correlation'])
        assert isinstance(results['is_stable'], bool)


# ============================================================================
# TEST FULL PIPELINE
# ============================================================================

class TestHybridWeightingPipeline:
    """Test complete hybrid weighting pipeline."""
    
    def test_full_pipeline_execution(self, panel_data):
        """Test complete pipeline with GTWC (default) and all steps."""
        pipeline = HybridWeightingPipeline(
            bootstrap_iterations=50,  # Reduced for speed
            stability_threshold=0.95,
            seed=42
        )
        
        result = pipeline.calculate(
            panel_data,
            entity_col='Entity',
            time_col='Year',
            criteria_cols=['C1', 'C2', 'C3', 'C4']
        )
        
        # Validate result structure
        assert isinstance(result, WeightResult)
        assert result.method == 'hybrid_weighting_pipeline'
        assert len(result.weights) == 4
        
        # Validate weights
        weight_sum = sum(result.weights.values())
        assert abs(weight_sum - 1.0) < 1e-6
        assert all(w > 0 for w in result.weights.values())
        
        # Validate details structure
        assert 'individual_weights' in result.details
        assert 'fusion' in result.details
        assert 'fusion_method' in result.details
        assert 'bootstrap' in result.details
        assert 'stability' in result.details
        
        # Check individual methods present
        assert 'entropy' in result.details['individual_weights']
        assert 'critic' in result.details['individual_weights']
        assert 'merec' in result.details['individual_weights']
        assert 'std_dev' in result.details['individual_weights']
        assert 'fused' in result.details['individual_weights']
        
        # Check GTWC-specific fusion details
        assert result.details['fusion_method'] == 'gtwc'
        fusion = result.details['fusion']
        assert fusion['method'] == 'game_theory_weight_combination'
        assert 'phase_2' in fusion
        assert 'phase_3' in fusion
        assert 'phase_4' in fusion
        assert 'alpha_dispersion' in fusion['phase_3']
        assert 'alpha_interaction' in fusion['phase_3']
        
        # Check bootstrap results
        assert 'mean_weights' in result.details['bootstrap']
        assert 'std_weights' in result.details['bootstrap']
        assert 'ci_lower_2_5' in result.details['bootstrap']
        assert 'ci_upper_97_5' in result.details['bootstrap']
        assert 'convergence_rate' in result.details['bootstrap']
        
        # Check stability results
        assert 'cosine_similarity' in result.details['stability']
        assert 'is_stable' in result.details['stability']
    
    def test_pipeline_auto_detect_criteria(self, panel_data):
        """Test automatic criteria detection."""
        pipeline = HybridWeightingPipeline(bootstrap_iterations=10)
        
        # Don't specify criteria_cols
        result = pipeline.calculate(
            panel_data,
            entity_col='Entity',
            time_col='Year'
        )
        
        # Should auto-detect C1, C2, C3, C4
        assert len(result.weights) == 4
        assert all(f'C{i}' in result.weights for i in range(1, 5))
    
    def test_pipeline_reproducibility(self, panel_data):
        """Test that results are reproducible with same seed."""
        pipeline1 = HybridWeightingPipeline(bootstrap_iterations=50, seed=42)
        pipeline2 = HybridWeightingPipeline(bootstrap_iterations=50, seed=42)
        
        result1 = pipeline1.calculate(
            panel_data, entity_col='Entity', time_col='Year',
            criteria_cols=['C1', 'C2', 'C3', 'C4']
        )
        result2 = pipeline2.calculate(
            panel_data, entity_col='Entity', time_col='Year',
            criteria_cols=['C1', 'C2', 'C3', 'C4']
        )
        
        # Weights should be identical
        for criterion in result1.weights:
            assert abs(result1.weights[criterion] - result2.weights[criterion]) < 1e-10


# ============================================================================
# TEST CONVENIENCE FUNCTIONS
# ============================================================================

class TestConvenienceFunctions:
    """Test high-level convenience functions."""
    
    def test_calculate_weights_function(self, simple_data):
        """Test calculate_weights() convenience function."""
        # Test different methods
        methods = ['entropy', 'critic', 'merec', 'std_dev', 'equal']
        
        for method in methods:
            result = calculate_weights(simple_data, method=method)
            assert isinstance(result, WeightResult)
            assert result.method == method
            assert abs(sum(result.weights.values()) - 1.0) < 1e-6
    
    def test_calculate_weights_hybrid(self, panel_data):
        """Test calculate_weights with hybrid method."""
        # Note: calculate_weights expects simple matrix, not panel format
        # Extract just criteria for this test
        simple_matrix = panel_data[['C1', 'C2', 'C3', 'C4']]
        
        result = calculate_weights(simple_matrix, method='hybrid')
        assert isinstance(result, WeightResult)
        assert len(result.weights) == 4


# ============================================================================
# TEST EDGE CASES
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_single_criterion(self):
        """Test with single criterion (should return weight=1.0)."""
        data = pd.DataFrame({'C1': [1, 2, 3, 4, 5]})
        
        result = EntropyWeightCalculator().calculate(data)
        assert abs(result.weights['C1'] - 1.0) < 1e-6
    
    def test_two_criteria(self):
        """Test with two criteria."""
        data = pd.DataFrame({
            'C1': [1, 2, 3, 4, 5],
            'C2': [10, 20, 30, 40, 50]
        })
        
        result = EntropyWeightCalculator().calculate(data)
        assert len(result.weights) == 2
        assert abs(sum(result.weights.values()) - 1.0) < 1e-6
    
    def test_minimal_panel_data(self):
        """Test pipeline with minimal panel data."""
        data = pd.DataFrame({
            'Year': [2020, 2020, 2021, 2021],
            'Entity': ['A', 'B', 'A', 'B'],
            'C1': [1.0, 2.0, 1.5, 2.5],
            'C2': [10.0, 20.0, 15.0, 25.0],
        })
        
        pipeline = HybridWeightingPipeline(bootstrap_iterations=10)
        result = pipeline.calculate(
            data, entity_col='Entity', time_col='Year',
            criteria_cols=['C1', 'C2']
        )
        
        assert len(result.weights) == 2
        assert abs(sum(result.weights.values()) - 1.0) < 1e-6


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests with realistic scenarios."""
    
    def test_realistic_mcdm_scenario(self):
        """Test with realistic MCDM panel data."""
        np.random.seed(42)
        
        # Simulate 10 provinces over 5 years with 6 criteria
        n_provinces = 10
        n_years = 5
        n_criteria = 6
        
        data = {
            'Year': np.repeat(range(2019, 2019 + n_years), n_provinces),
            'Province': np.tile([f'P{i:02d}' for i in range(n_provinces)], n_years),
        }
        
        # Add correlated criteria (realistic MCDM setting)
        for i in range(n_criteria):
            base = np.random.randn(n_provinces * n_years)
            trend = np.repeat(np.arange(n_years), n_provinces) * 0.5
            data[f'C{i+1:02d}'] = 50 + base * 10 + trend
        
        df = pd.DataFrame(data)
        
        # Run pipeline
        pipeline = HybridWeightingPipeline(
            bootstrap_iterations=100,
            stability_threshold=0.90
        )
        
        result = pipeline.calculate(
            df, entity_col='Province', time_col='Year'
        )
        
        # Comprehensive validation
        assert len(result.weights) == n_criteria
        assert abs(sum(result.weights.values()) - 1.0) < 1e-6
        assert all(w > 0 for w in result.weights.values())
        
        # Bootstrap should have good convergence
        assert result.details['bootstrap']['convergence_rate'] > 0.95
        
        # Should have meaningful uncertainty bounds
        for criterion in result.weights:
            mean_w = result.details['bootstrap']['mean_weights'][criterion]
            std_w = result.details['bootstrap']['std_weights'][criterion]
            ci_low = result.details['bootstrap']['ci_lower_2_5'][criterion]
            ci_high = result.details['bootstrap']['ci_upper_97_5'][criterion]
            
            # CI should bracket mean
            assert ci_low <= mean_w <= ci_high
            
            # CI width should be roughly proportional to std
            ci_width = ci_high - ci_low
            assert ci_width > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
