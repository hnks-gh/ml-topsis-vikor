# -*- coding: utf-8 -*-
"""Advanced ensemble forecasting with gradient boosting models."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import warnings
from scipy import stats
from sklearn.ensemble import (
    RandomForestRegressor, 
    GradientBoostingRegressor,
    ExtraTreesRegressor,
    AdaBoostRegressor
)
from sklearn.linear_model import (
    Ridge, 
    Lasso, 
    ElasticNet,
    BayesianRidge,
    HuberRegressor
)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error, 
    r2_score,
    mean_absolute_percentage_error
)
from sklearn.model_selection import TimeSeriesSplit
import copy

warnings.filterwarnings('ignore')


@dataclass
class ForecastResult:
    """Result container for ML forecasting."""
    predictions: pd.DataFrame          # Entity × Component predictions
    prediction_intervals: Dict[str, pd.DataFrame]  # Lower/upper bounds
    feature_importance: pd.DataFrame   # Feature importance scores
    model_performance: Dict[str, Dict[str, float]]  # Model-wise metrics
    ensemble_weights: Dict[str, float]  # Optimal model weights
    cv_scores: Dict[str, List[float]]  # Cross-validation scores
    training_history: Dict[str, Any]   # Training details
    metadata: Dict[str, Any]           # Additional metadata
    
    def summary(self) -> str:
        lines = [
            f"\n{'='*70}",
            "ADVANCED ML FORECASTING RESULTS",
            f"{'='*70}",
            f"\nPredicted entities: {len(self.predictions)}",
            f"Predicted components: {len(self.predictions.columns)}",
            f"\nEnsemble Model Weights:",
        ]
        for model, weight in sorted(self.ensemble_weights.items(), 
                                   key=lambda x: x[1], reverse=True):
            bar = '█' * int(weight * 30)
            lines.append(f"  {model:20s}: {weight:.3f} {bar}")
        
        lines.append(f"\nCross-Validation Performance:")
        for model, scores in self.cv_scores.items():
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            lines.append(f"  {model:20s}: R² = {mean_score:.4f} ± {std_score:.4f}")
        
        lines.append(f"\nTop 10 Important Features:")
        top_features = self.feature_importance.mean(axis=1).nlargest(10)
        for feat, imp in top_features.items():
            lines.append(f"  {feat}: {imp:.4f}")
        
        lines.append("=" * 70)
        return "\n".join(lines)


class TemporalFeatureEngineer:
    """
    Advanced feature engineering for time series panel data.
    
    Creates rich feature set including:
    - Lag features (t-1, t-2, ...)
    - Rolling statistics (mean, std, min, max, trend)
    - Seasonal features
    - Cross-entity features (relative position)
    - Momentum and acceleration features
    """
    
    def __init__(self,
                 lag_periods: List[int] = [1, 2],
                 rolling_windows: List[int] = [2, 3],
                 include_momentum: bool = True,
                 include_cross_entity: bool = True):
        self.lag_periods = lag_periods
        self.rolling_windows = rolling_windows
        self.include_momentum = include_momentum
        self.include_cross_entity = include_cross_entity
        self.feature_names_ = []
    
    def fit_transform(self,
                      panel_data,
                      target_year: int
                      ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create feature matrix for training and prediction.
        
        Returns:
            X_train: Features for training
            y_train: Targets for training
            X_pred: Features for prediction (next year)
            entity_index: Entity identifiers
        """
        entities = panel_data.provinces
        components = panel_data.components
        years = sorted(panel_data.years)
        
        # Determine train years (all except last) and test year (last)
        train_years = years[:-1]
        last_year = years[-1]
        
        X_train_list = []
        y_train_list = []
        X_pred_list = []
        
        for entity in entities:
            entity_data = panel_data.get_province(entity)
            
            # Create features for each training year (predicting next year)
            for i, year in enumerate(train_years[:-1]):  # Need at least one future year
                target_year_idx = i + 1
                if target_year_idx < len(train_years):
                    features = self._create_features(
                        entity_data, entity, years, year, entities, panel_data
                    )
                    target = entity_data.loc[train_years[target_year_idx], components].values
                    
                    X_train_list.append(features)
                    y_train_list.append(target)
            
            # Create features for prediction (using last year to predict next)
            pred_features = self._create_features(
                entity_data, entity, years, last_year, entities, panel_data
            )
            X_pred_list.append(pred_features)
        
        # Stack into arrays
        X_train = np.vstack(X_train_list)
        y_train = np.vstack(y_train_list)
        X_pred = np.vstack(X_pred_list)
        
        # Create DataFrames
        X_train_df = pd.DataFrame(X_train, columns=self.feature_names_)
        y_train_df = pd.DataFrame(y_train, columns=components)
        X_pred_df = pd.DataFrame(X_pred, columns=self.feature_names_, index=entities)
        
        return X_train_df, y_train_df, X_pred_df, pd.DataFrame(index=entities)
    
    def _create_features(self,
                         entity_data: pd.DataFrame,
                         entity: str,
                         years: List[int],
                         current_year: int,
                         all_entities: List[str],
                         panel_data) -> np.ndarray:
        """Create feature vector for a single entity-year combination."""
        components = list(entity_data.columns)
        features = []
        feature_names = []
        
        # Current values
        current_values = entity_data.loc[current_year, components].values
        features.extend(current_values)
        feature_names.extend([f"{c}_current" for c in components])
        
        # Lag features
        year_idx = years.index(current_year)
        for lag in self.lag_periods:
            if year_idx - lag >= 0:
                lag_year = years[year_idx - lag]
                lag_values = entity_data.loc[lag_year, components].values
            else:
                lag_values = current_values  # Use current if not enough history
            features.extend(lag_values)
            feature_names.extend([f"{c}_lag{lag}" for c in components])
        
        # Rolling statistics
        available_years = [y for y in years if y <= current_year]
        for window in self.rolling_windows:
            if len(available_years) >= window:
                window_years = available_years[-window:]
                window_data = entity_data.loc[window_years, components]
                
                # Mean
                features.extend(window_data.mean().values)
                feature_names.extend([f"{c}_roll{window}_mean" for c in components])
                
                # Std
                std_vals = window_data.std().fillna(0).values
                features.extend(std_vals)
                feature_names.extend([f"{c}_roll{window}_std" for c in components])
                
                # Min/Max
                features.extend(window_data.min().values)
                feature_names.extend([f"{c}_roll{window}_min" for c in components])
                features.extend(window_data.max().values)
                feature_names.extend([f"{c}_roll{window}_max" for c in components])
            else:
                # Pad with current values
                features.extend(current_values)
                feature_names.extend([f"{c}_roll{window}_mean" for c in components])
                features.extend(np.zeros(len(components)))
                feature_names.extend([f"{c}_roll{window}_std" for c in components])
                features.extend(current_values)
                feature_names.extend([f"{c}_roll{window}_min" for c in components])
                features.extend(current_values)
                feature_names.extend([f"{c}_roll{window}_max" for c in components])
        
        # Momentum features (rate of change)
        if self.include_momentum and year_idx > 0:
            prev_year = years[year_idx - 1]
            prev_values = entity_data.loc[prev_year, components].values
            momentum = current_values - prev_values
            features.extend(momentum)
            feature_names.extend([f"{c}_momentum" for c in components])
            
            # Acceleration (change in momentum)
            if year_idx > 1:
                prev_prev_year = years[year_idx - 2]
                prev_prev_values = entity_data.loc[prev_prev_year, components].values
                prev_momentum = prev_values - prev_prev_values
                acceleration = momentum - prev_momentum
                features.extend(acceleration)
                feature_names.extend([f"{c}_acceleration" for c in components])
            else:
                features.extend(np.zeros(len(components)))
                feature_names.extend([f"{c}_acceleration" for c in components])
        else:
            features.extend(np.zeros(len(components)))
            feature_names.extend([f"{c}_momentum" for c in components])
            features.extend(np.zeros(len(components)))
            feature_names.extend([f"{c}_acceleration" for c in components])
        
        # Trend feature (slope of linear fit)
        if len(available_years) >= 2:
            for c in components:
                y_vals = entity_data.loc[available_years, c].values
                x_vals = np.arange(len(y_vals))
                if len(y_vals) > 1:
                    slope = np.polyfit(x_vals, y_vals, 1)[0]
                else:
                    slope = 0
                features.append(slope)
                feature_names.append(f"{c}_trend")
        else:
            features.extend(np.zeros(len(components)))
            feature_names.extend([f"{c}_trend" for c in components])
        
        # Cross-entity features (relative position)
        if self.include_cross_entity:
            year_cross_section = panel_data.cross_section[current_year]
            if 'Province' in year_cross_section.columns:
                year_cross_section = year_cross_section.set_index('Province')
            
            for c in components:
                if c in year_cross_section.columns:
                    col_values = year_cross_section[c]
                    entity_value = current_values[components.index(c)]
                    
                    # Percentile rank
                    percentile = (col_values < entity_value).mean()
                    features.append(percentile)
                    feature_names.append(f"{c}_percentile")
                    
                    # Z-score
                    mean_val = col_values.mean()
                    std_val = col_values.std()
                    z_score = (entity_value - mean_val) / (std_val + 1e-10)
                    features.append(z_score)
                    feature_names.append(f"{c}_zscore")
                else:
                    features.extend([0.5, 0])
                    feature_names.extend([f"{c}_percentile", f"{c}_zscore"])
        
        self.feature_names_ = feature_names
        return np.array(features)


class BaseForecaster(ABC):
    """Abstract base class for forecasting models."""
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseForecaster':
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass
    
    @abstractmethod
    def get_feature_importance(self) -> np.ndarray:
        pass


class GradientBoostingForecaster(BaseForecaster):
    """
    Advanced Gradient Boosting forecaster.
    
    Combines multiple gradient boosting variants for robust predictions.
    """
    
    def __init__(self,
                 n_estimators: int = 200,
                 max_depth: int = 6,
                 learning_rate: float = 0.1,
                 subsample: float = 0.8,
                 random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.random_state = random_state
        
        self.model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            random_state=random_state,
            loss='huber',  # Robust to outliers
            validation_fraction=0.1,
            n_iter_no_change=20,
            tol=1e-4
        )
        self.scaler = RobustScaler()
        self.feature_importance_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GradientBoostingForecaster':
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y.ravel() if y.ndim > 1 else y)
        self.feature_importance_ = self.model.feature_importances_
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def get_feature_importance(self) -> np.ndarray:
        return self.feature_importance_


class RandomForestForecaster(BaseForecaster):
    """Random Forest forecaster with advanced settings."""
    
    def __init__(self,
                 n_estimators: int = 300,
                 max_depth: int = 12,
                 min_samples_split: int = 5,
                 min_samples_leaf: int = 2,
                 random_state: int = 42):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=-1,
            oob_score=True
        )
        self.scaler = RobustScaler()
        self.feature_importance_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RandomForestForecaster':
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y.ravel() if y.ndim > 1 else y)
        self.feature_importance_ = self.model.feature_importances_
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def get_feature_importance(self) -> np.ndarray:
        return self.feature_importance_


class ExtraTreesForecaster(BaseForecaster):
    """Extra Trees forecaster for diversity."""
    
    def __init__(self,
                 n_estimators: int = 300,
                 max_depth: int = 15,
                 random_state: int = 42):
        self.model = ExtraTreesRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )
        self.scaler = RobustScaler()
        self.feature_importance_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ExtraTreesForecaster':
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y.ravel() if y.ndim > 1 else y)
        self.feature_importance_ = self.model.feature_importances_
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def get_feature_importance(self) -> np.ndarray:
        return self.feature_importance_


class BayesianRidgeForecaster(BaseForecaster):
    """Bayesian Ridge regression with uncertainty quantification."""
    
    def __init__(self):
        self.model = BayesianRidge(
            compute_score=True,
            fit_intercept=True
        )
        self.scaler = StandardScaler()
        self.feature_importance_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BayesianRidgeForecaster':
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y.ravel() if y.ndim > 1 else y)
        self.feature_importance_ = np.abs(self.model.coef_)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_with_std(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled, return_std=True)
    
    def get_feature_importance(self) -> np.ndarray:
        return self.feature_importance_


class HuberForecaster(BaseForecaster):
    """Huber regression - robust to outliers."""
    
    def __init__(self, epsilon: float = 1.35):
        self.model = HuberRegressor(epsilon=epsilon, max_iter=1000)
        self.scaler = RobustScaler()
        self.feature_importance_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'HuberForecaster':
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y.ravel() if y.ndim > 1 else y)
        self.feature_importance_ = np.abs(self.model.coef_)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def get_feature_importance(self) -> np.ndarray:
        return self.feature_importance_


class AdvancedMLForecaster:
    """
    State-of-the-art ensemble ML forecaster for panel data.
    
    Combines multiple models using stacking with optimal weights:
    - Gradient Boosting (robust to outliers)
    - Random Forest (stable, interpretable)
    - Extra Trees (diversity)
    - Bayesian Ridge (uncertainty quantification)
    - Huber Regression (outlier robust linear)
    
    Features:
    - Automatic model selection and weighting
    - Time-series cross-validation
    - Prediction intervals
    - Feature importance aggregation
    """
    
    def __init__(self,
                 include_gb: bool = True,
                 include_rf: bool = True,
                 include_et: bool = True,
                 include_bayesian: bool = True,
                 include_huber: bool = True,
                 cv_splits: int = 3,
                 random_state: int = 42):
        self.include_gb = include_gb
        self.include_rf = include_rf
        self.include_et = include_et
        self.include_bayesian = include_bayesian
        self.include_huber = include_huber
        self.cv_splits = cv_splits
        self.random_state = random_state
        
        self.models: Dict[str, List[BaseForecaster]] = {}
        self.weights: Dict[str, float] = {}
        self.feature_engineer = TemporalFeatureEngineer()
        self.cv_scores: Dict[str, List[float]] = {}
        self.feature_importance_: Optional[pd.DataFrame] = None
    
    def fit_predict(self,
                   panel_data,
                   target_components: Optional[List[str]] = None
                   ) -> ForecastResult:
        """
        Fit models and predict next year values.
        
        Parameters:
            panel_data: PanelData object with historical data
            target_components: Components to predict (default: all)
        
        Returns:
            ForecastResult with predictions and analysis
        """
        if target_components is None:
            target_components = panel_data.components
        
        # Feature engineering
        X_train, y_train, X_pred, entities = self.feature_engineer.fit_transform(
            panel_data, panel_data.years[-1]
        )
        
        # Initialize models
        self._initialize_models()
        
        # Store predictions for each component
        all_predictions = {}
        all_lower = {}
        all_upper = {}
        all_importance = {}
        model_performance = {}
        
        for comp_idx, component in enumerate(target_components):
            y_comp = y_train.iloc[:, comp_idx].values if comp_idx < y_train.shape[1] else y_train.iloc[:, 0].values
            
            # Train and evaluate each model
            comp_predictions = {}
            comp_cv_scores = {}
            comp_importance = {}
            
            for model_name, model_class in self._get_model_classes().items():
                # Time-series cross-validation
                cv_scores, trained_models = self._cross_validate(
                    X_train.values, y_comp, model_class
                )
                comp_cv_scores[model_name] = cv_scores
                
                # Train final model on all data
                final_model = model_class()
                final_model.fit(X_train.values, y_comp)
                
                # Predict
                predictions = final_model.predict(X_pred.values)
                comp_predictions[model_name] = predictions
                
                # Feature importance
                importance = final_model.get_feature_importance()
                if importance is not None:
                    comp_importance[model_name] = importance
                
                # Store models
                if model_name not in self.models:
                    self.models[model_name] = []
                self.models[model_name].append(final_model)
            
            # Calculate optimal ensemble weights
            weights = self._calculate_optimal_weights(comp_cv_scores)
            
            # Ensemble prediction
            ensemble_pred = np.zeros(len(X_pred))
            for model_name, preds in comp_predictions.items():
                ensemble_pred += weights.get(model_name, 0) * preds
            
            all_predictions[component] = ensemble_pred
            
            # Prediction intervals (using Bayesian Ridge if available)
            if self.include_bayesian and 'bayesian_ridge' in comp_predictions:
                bayesian_model = self.models['bayesian_ridge'][-1]
                _, std = bayesian_model.predict_with_std(X_pred.values)
                all_lower[component] = ensemble_pred - 1.96 * std
                all_upper[component] = ensemble_pred + 1.96 * std
            else:
                # Estimate uncertainty from model disagreement
                pred_matrix = np.column_stack(list(comp_predictions.values()))
                std = pred_matrix.std(axis=1)
                all_lower[component] = ensemble_pred - 1.96 * std
                all_upper[component] = ensemble_pred + 1.96 * std
            
            # Aggregate feature importance
            if comp_importance:
                avg_importance = np.mean(
                    [imp for imp in comp_importance.values()], axis=0
                )
                all_importance[component] = avg_importance
            
            # Store CV scores
            for model_name, scores in comp_cv_scores.items():
                if model_name not in self.cv_scores:
                    self.cv_scores[model_name] = []
                self.cv_scores[model_name].extend(scores)
        
        # Create result DataFrames
        predictions_df = pd.DataFrame(all_predictions, index=X_pred.index)
        lower_df = pd.DataFrame(all_lower, index=X_pred.index)
        upper_df = pd.DataFrame(all_upper, index=X_pred.index)
        
        # Feature importance DataFrame
        if all_importance:
            importance_df = pd.DataFrame(
                all_importance, 
                index=self.feature_engineer.feature_names_
            )
        else:
            importance_df = pd.DataFrame()
        
        self.feature_importance_ = importance_df
        self.weights = weights
        
        return ForecastResult(
            predictions=predictions_df,
            prediction_intervals={'lower': lower_df, 'upper': upper_df},
            feature_importance=importance_df,
            model_performance=model_performance,
            ensemble_weights=weights,
            cv_scores=self.cv_scores,
            training_history={'n_samples': len(X_train), 'n_features': X_train.shape[1]},
            metadata={
                'target_components': target_components,
                'entities': list(X_pred.index),
                'feature_names': self.feature_engineer.feature_names_
            }
        )
    
    def _initialize_models(self):
        """Initialize model dictionary."""
        self.models = {}
    
    def _get_model_classes(self) -> Dict[str, type]:
        """Get dictionary of model classes to use."""
        classes = {}
        if self.include_gb:
            classes['gradient_boosting'] = GradientBoostingForecaster
        if self.include_rf:
            classes['random_forest'] = RandomForestForecaster
        if self.include_et:
            classes['extra_trees'] = ExtraTreesForecaster
        if self.include_bayesian:
            classes['bayesian_ridge'] = BayesianRidgeForecaster
        if self.include_huber:
            classes['huber'] = HuberForecaster
        return classes
    
    def _cross_validate(self,
                        X: np.ndarray,
                        y: np.ndarray,
                        model_class: type
                        ) -> Tuple[List[float], List[BaseForecaster]]:
        """Perform time-series cross-validation."""
        n_samples = len(X)
        n_splits = min(self.cv_splits, max(2, n_samples // 10))
        
        if n_samples < 10:
            # Not enough data for CV, just fit on all
            model = model_class()
            model.fit(X, y)
            return [1.0], [model]
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        scores = []
        models = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train_cv, X_val_cv = X[train_idx], X[val_idx]
            y_train_cv, y_val_cv = y[train_idx], y[val_idx]
            
            model = model_class()
            model.fit(X_train_cv, y_train_cv)
            
            y_pred = model.predict(X_val_cv)
            
            # Calculate R² score (handle edge cases)
            if len(np.unique(y_val_cv)) > 1:
                score = r2_score(y_val_cv, y_pred)
            else:
                score = 0.0
            
            scores.append(max(0, score))  # Clip negative R²
            models.append(model)
        
        return scores, models
    
    def _calculate_optimal_weights(self,
                                   cv_scores: Dict[str, List[float]]
                                   ) -> Dict[str, float]:
        """Calculate optimal ensemble weights based on CV performance."""
        # Use mean CV score as weight basis
        mean_scores = {}
        for model_name, scores in cv_scores.items():
            mean_scores[model_name] = max(0.01, np.mean(scores))  # Min weight
        
        # Normalize to sum to 1
        total = sum(mean_scores.values())
        weights = {name: score / total for name, score in mean_scores.items()}
        
        return weights


class ComponentForecaster:
    """
    Forecaster that predicts each MCDM score component separately.
    
    Optimized for predicting criteria values that will be used
    in MCDM methods for the next year.
    """
    
    def __init__(self,
                 forecaster: Optional[AdvancedMLForecaster] = None):
        self.forecaster = forecaster or AdvancedMLForecaster()
    
    def forecast_next_year(self,
                          panel_data,
                          components: Optional[List[str]] = None
                          ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Forecast component values for next year.
        
        Parameters:
            panel_data: Historical panel data
            components: Components to forecast (default: all)
        
        Returns:
            Tuple of (predictions_df, uncertainty_df)
        """
        if components is None:
            components = panel_data.components
        
        result = self.forecaster.fit_predict(panel_data, components)
        
        # Calculate uncertainty as (upper - lower) / 2
        uncertainty = (
            result.prediction_intervals['upper'] - 
            result.prediction_intervals['lower']
        ) / 2
        
        return result.predictions, uncertainty
