"""
Pytest configuration and fixtures for ML-MCDM tests.
"""
import pytest
import numpy as np
import pandas as pd


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    return pd.DataFrame(
        np.random.rand(10, 5),
        columns=[f'C{i+1:02d}' for i in range(5)]
    )


@pytest.fixture
def sample_weights():
    """Create sample weights for testing."""
    return np.array([0.2, 0.2, 0.2, 0.2, 0.2])


@pytest.fixture
def sample_criteria_types():
    """Create sample criteria types for testing."""
    return ['benefit'] * 5
