# -*- coding: utf-8 -*-
"""
Cross-Validation and Bootstrap Validation
==========================================

Model validation for robust ML and MCDM results.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass


@dataclass
class ValidationResult:
    """Result container for validation analysis."""
    cv_scores: Dict[str, float]          # Mean CV scores
    cv_std: Dict[str, float]             # CV score standard deviations
    bootstrap_ci: Dict[str, Tuple[float, float]]  # 95% confidence intervals
    fold_scores: Dict[str, List[float]]  # Scores per fold
    n_folds: int
    n_bootstrap: int
    validation_type: str
    
    def summary(self) -> str:
        lines = [
            f"\n{'='*60}",
            "VALIDATION RESULTS",
            f"{'='*60}",
            f"\nValidation type: {self.validation_type}",
            f"Number of folds: {self.n_folds}",
            f"Bootstrap samples: {self.n_bootstrap}",
            f"\n{'─'*30}",
            "CROSS-VALIDATION SCORES",
            f"{'─'*30}"
        ]
        
        for metric in self.cv_scores.keys():
            mean = self.cv_scores[metric]
            std = self.cv_std[metric]
            lines.append(f"  {metric}: {mean:.4f} ± {std:.4f}")
        
        if self.bootstrap_ci:
            lines.extend([
                f"\n{'─'*30}",
                "BOOTSTRAP 95% CONFIDENCE INTERVALS",
                f"{'─'*30}"
            ])
            for metric, (low, high) in self.bootstrap_ci.items():
                lines.append(f"  {metric}: [{low:.4f}, {high:.4f}]")
        
        lines.append("=" * 60)
        return "\n".join(lines)


class CrossValidator:
    """
    Cross-validation for panel data with temporal awareness.
    """
    
    def __init__(self,
                 n_folds: int = 5,
                 time_series_split: bool = True,
                 shuffle: bool = False,
                 seed: int = 42):
        """
        Initialize cross-validator.
        
        Parameters
        ----------
        n_folds : int
            Number of CV folds
        time_series_split : bool
            Use time-series aware splitting
        shuffle : bool
            Shuffle data before splitting
        seed : int
            Random seed
        """
        self.n_folds = n_folds
        self.time_series_split = time_series_split
        self.shuffle = shuffle
        self.seed = seed
    
    def validate(self,
                X: np.ndarray,
                y: np.ndarray,
                model_func: Callable,
                metrics: Dict[str, Callable],
                time_indices: Optional[np.ndarray] = None) -> ValidationResult:
        """
        Perform cross-validation.
        
        Parameters
        ----------
        X : np.ndarray
            Features
        y : np.ndarray
            Target
        model_func : Callable
            Function that trains and returns predictions: f(X_train, y_train, X_test) -> y_pred
        metrics : Dict[str, Callable]
            Dictionary of metric functions {name: func(y_true, y_pred)}
        time_indices : np.ndarray, optional
            Time indices for time-series split
        """
        np.random.seed(self.seed)
        
        n_samples = len(y)
        
        # Generate fold indices
        if self.time_series_split and time_indices is not None:
            folds = self._time_series_folds(time_indices)
        else:
            folds = self._kfold_indices(n_samples)
        
        # Score storage
        fold_scores = {metric: [] for metric in metrics.keys()}
        
        for train_idx, test_idx in folds:
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Get predictions
            y_pred = model_func(X_train, y_train, X_test)
            
            # Calculate metrics
            for metric_name, metric_func in metrics.items():
                score = metric_func(y_test, y_pred)
                fold_scores[metric_name].append(score)
        
        # Aggregate results
        cv_scores = {k: np.mean(v) for k, v in fold_scores.items()}
        cv_std = {k: np.std(v) for k, v in fold_scores.items()}
        
        return ValidationResult(
            cv_scores=cv_scores,
            cv_std=cv_std,
            bootstrap_ci={},
            fold_scores=fold_scores,
            n_folds=len(folds),
            n_bootstrap=0,
            validation_type='time_series' if self.time_series_split else 'k_fold'
        )
    
    def _kfold_indices(self, n_samples: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate k-fold cross-validation indices."""
        indices = np.arange(n_samples)
        
        if self.shuffle:
            np.random.shuffle(indices)
        
        fold_sizes = np.full(self.n_folds, n_samples // self.n_folds)
        fold_sizes[:n_samples % self.n_folds] += 1
        
        folds = []
        current = 0
        
        for fold_size in fold_sizes:
            test_idx = indices[current:current + fold_size]
            train_idx = np.concatenate([indices[:current], indices[current + fold_size:]])
            folds.append((train_idx, test_idx))
            current += fold_size
        
        return folds
    
    def _time_series_folds(self, time_indices: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate time-series cross-validation folds."""
        unique_times = np.unique(time_indices)
        n_times = len(unique_times)
        
        if n_times < self.n_folds + 1:
            # Not enough time periods, use regular k-fold
            return self._kfold_indices(len(time_indices))
        
        folds = []
        
        # Expanding window
        min_train_periods = max(2, n_times // 2)
        
        for i in range(min_train_periods, n_times):
            train_times = unique_times[:i]
            test_times = unique_times[i:i+1]
            
            train_idx = np.where(np.isin(time_indices, train_times))[0]
            test_idx = np.where(np.isin(time_indices, test_times))[0]
            
            if len(test_idx) > 0:
                folds.append((train_idx, test_idx))
        
        # Limit to n_folds
        if len(folds) > self.n_folds:
            step = len(folds) // self.n_folds
            folds = folds[::step][:self.n_folds]
        
        return folds


class BootstrapValidator:
    """
    Bootstrap validation for confidence intervals.
    """
    
    def __init__(self,
                 n_bootstrap: int = 1000,
                 confidence_level: float = 0.95,
                 seed: int = 42):
        """
        Initialize bootstrap validator.
        
        Parameters
        ----------
        n_bootstrap : int
            Number of bootstrap samples
        confidence_level : float
            Confidence level for intervals
        seed : int
            Random seed
        """
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self.seed = seed
    
    def validate(self,
                X: np.ndarray,
                y: np.ndarray,
                model_func: Callable,
                metrics: Dict[str, Callable]) -> ValidationResult:
        """
        Perform bootstrap validation.
        
        Parameters
        ----------
        X : np.ndarray
            Features
        y : np.ndarray
            Target
        model_func : Callable
            Function: f(X_train, y_train, X_test) -> y_pred
        metrics : Dict[str, Callable]
            Metric functions
        """
        np.random.seed(self.seed)
        
        n_samples = len(y)
        bootstrap_scores = {metric: [] for metric in metrics.keys()}
        
        for _ in range(self.n_bootstrap):
            # Bootstrap sample (with replacement)
            boot_idx = np.random.choice(n_samples, n_samples, replace=True)
            oob_idx = np.setdiff1d(np.arange(n_samples), boot_idx)
            
            if len(oob_idx) < 5:
                continue
            
            X_train, X_test = X[boot_idx], X[oob_idx]
            y_train, y_test = y[boot_idx], y[oob_idx]
            
            # Get predictions
            y_pred = model_func(X_train, y_train, X_test)
            
            # Calculate metrics
            for metric_name, metric_func in metrics.items():
                score = metric_func(y_test, y_pred)
                bootstrap_scores[metric_name].append(score)
        
        # Calculate confidence intervals
        alpha = 1 - self.confidence_level
        ci = {}
        
        for metric, scores in bootstrap_scores.items():
            scores = np.array(scores)
            ci[metric] = (
                np.percentile(scores, 100 * alpha / 2),
                np.percentile(scores, 100 * (1 - alpha / 2))
            )
        
        cv_scores = {k: np.mean(v) for k, v in bootstrap_scores.items()}
        cv_std = {k: np.std(v) for k, v in bootstrap_scores.items()}
        
        return ValidationResult(
            cv_scores=cv_scores,
            cv_std=cv_std,
            bootstrap_ci=ci,
            fold_scores=bootstrap_scores,
            n_folds=0,
            n_bootstrap=self.n_bootstrap,
            validation_type='bootstrap'
        )


class RankingValidator:
    """
    Specialized validation for MCDM rankings.
    """
    
    def __init__(self,
                 n_bootstrap: int = 500,
                 perturbation_std: float = 0.05,
                 seed: int = 42):
        """
        Initialize ranking validator.
        
        Parameters
        ----------
        n_bootstrap : int
            Number of bootstrap samples
        perturbation_std : float
            Standard deviation for data perturbation
        seed : int
            Random seed
        """
        self.n_bootstrap = n_bootstrap
        self.perturbation_std = perturbation_std
        self.seed = seed
    
    def validate_ranking(self,
                        decision_matrix: np.ndarray,
                        weights: np.ndarray,
                        ranking_func: Callable) -> Dict:
        """
        Validate ranking stability using bootstrap perturbation.
        
        Parameters
        ----------
        decision_matrix : np.ndarray
            Decision matrix
        weights : np.ndarray
            Criterion weights
        ranking_func : Callable
            Ranking function
        """
        np.random.seed(self.seed)
        
        n_alternatives = decision_matrix.shape[0]
        base_ranking = ranking_func(decision_matrix, weights)
        
        bootstrap_rankings = np.zeros((self.n_bootstrap, n_alternatives))
        
        for i in range(self.n_bootstrap):
            # Perturb decision matrix
            noise = np.random.normal(0, self.perturbation_std, decision_matrix.shape)
            perturbed_matrix = decision_matrix * (1 + noise)
            perturbed_matrix = np.clip(perturbed_matrix, 0.001, None)
            
            bootstrap_rankings[i] = ranking_func(perturbed_matrix, weights)
        
        # Calculate stability metrics
        mean_rankings = bootstrap_rankings.mean(axis=0)
        std_rankings = bootstrap_rankings.std(axis=0)
        
        # Rank correlation with base
        from scipy import stats
        correlations = []
        for i in range(self.n_bootstrap):
            corr, _ = stats.spearmanr(base_ranking, bootstrap_rankings[i])
            correlations.append(corr)
        
        # Confidence intervals for rankings
        rank_ci = {}
        for j in range(n_alternatives):
            ranks_j = bootstrap_rankings[:, j]
            rank_ci[j] = (
                np.percentile(ranks_j, 2.5),
                np.percentile(ranks_j, 97.5)
            )
        
        return {
            'base_ranking': base_ranking,
            'mean_ranking': mean_rankings,
            'std_ranking': std_rankings,
            'mean_correlation': np.mean(correlations),
            'std_correlation': np.std(correlations),
            'rank_ci': rank_ci,
            'bootstrap_rankings': bootstrap_rankings
        }


# Convenience functions
def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R² score."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1 - ss_res / (ss_tot + 1e-10)


def mse_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean squared error."""
    return np.mean((y_true - y_pred) ** 2)


def mae_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error."""
    return np.mean(np.abs(y_true - y_pred))


def bootstrap_validation(X: np.ndarray,
                        y: np.ndarray,
                        model_func: Callable,
                        n_bootstrap: int = 1000) -> ValidationResult:
    """Convenience function for bootstrap validation."""
    validator = BootstrapValidator(n_bootstrap=n_bootstrap)
    metrics = {
        'R2': r2_score,
        'MSE': mse_score,
        'MAE': mae_score
    }
    return validator.validate(X, y, model_func, metrics)
