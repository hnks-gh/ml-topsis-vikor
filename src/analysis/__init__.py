# -*- coding: utf-8 -*-
"""
Analysis Module
===============

Convergence testing, sensitivity analysis, and validation.
"""

from .convergence import ConvergenceAnalysis, ConvergenceResult
from .sensitivity import SensitivityAnalysis, SensitivityResult
from .validation import (
    CrossValidator, ValidationResult, BootstrapValidator,
    bootstrap_validation, r2_score, mse_score, mae_score
)

__all__ = [
    'ConvergenceAnalysis', 'ConvergenceResult',
    'SensitivityAnalysis', 'SensitivityResult',
    'CrossValidator', 'ValidationResult', 'BootstrapValidator',
    'bootstrap_validation', 'r2_score', 'mse_score', 'mae_score'
]
