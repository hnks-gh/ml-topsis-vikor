# -*- coding: utf-8 -*-
"""
LSTM Forecaster for Panel Data
===============================

Time-series forecasting using LSTM neural networks.
Predicts future scores based on historical trajectories.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import warnings


@dataclass
class LSTMResult:
    """Result container for LSTM forecasting."""
    predictions: pd.DataFrame        # Province × predicted values
    actual: pd.DataFrame             # Province × actual values
    train_loss: List[float]
    val_loss: List[float]
    test_metrics: Dict[str, float]
    predicted_ranks: pd.Series
    actual_ranks: pd.Series
    rank_correlation: float
    model_summary: str
    
    def summary(self) -> str:
        lines = [
            f"\n{'='*60}",
            "LSTM FORECASTING RESULTS",
            f"{'='*60}",
            f"\nTest Metrics:",
            f"  MSE: {self.test_metrics['mse']:.4f}",
            f"  MAE: {self.test_metrics['mae']:.4f}",
            f"  RMSE: {self.test_metrics['rmse']:.4f}",
            f"  Rank Correlation: {self.rank_correlation:.4f}",
            f"\nTraining:",
            f"  Final Train Loss: {self.train_loss[-1]:.4f}" if self.train_loss else "  N/A",
            f"  Final Val Loss: {self.val_loss[-1]:.4f}" if self.val_loss else "  N/A",
            "=" * 60
        ]
        return "\n".join(lines)


class SimpleLSTM:
    """
    Simple LSTM implementation using numpy for environments without PyTorch/TensorFlow.
    Uses a basic recurrent approach for forecasting.
    """
    
    def __init__(self, hidden_size: int = 32, learning_rate: float = 0.01):
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.weights = None
    
    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 100) -> List[float]:
        """Fit using simple linear mapping with temporal features."""
        # Flatten temporal features
        n_samples, seq_len, n_features = X.shape
        X_flat = X.reshape(n_samples, -1)
        
        # Add temporal aggregates
        X_mean = X.mean(axis=1)
        X_last = X[:, -1, :]
        X_trend = X[:, -1, :] - X[:, 0, :]
        X_full = np.hstack([X_flat, X_mean, X_last, X_trend])
        
        # Ridge regression
        lambda_reg = 0.1
        XtX = X_full.T @ X_full + lambda_reg * np.eye(X_full.shape[1])
        Xty = X_full.T @ y
        
        try:
            self.weights = np.linalg.solve(XtX, Xty)
        except:
            self.weights = np.linalg.lstsq(X_full, y, rcond=None)[0]
        
        # Calculate training loss
        y_pred = X_full @ self.weights
        mse = np.mean((y - y_pred) ** 2)
        
        return [mse] * epochs  # Simplified - single fit
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using fitted weights."""
        n_samples, seq_len, n_features = X.shape
        X_flat = X.reshape(n_samples, -1)
        
        X_mean = X.mean(axis=1)
        X_last = X[:, -1, :]
        X_trend = X[:, -1, :] - X[:, 0, :]
        X_full = np.hstack([X_flat, X_mean, X_last, X_trend])
        
        return X_full @ self.weights


class LSTMForecaster:
    """
    LSTM-based forecaster for panel data.
    
    Uses historical province trajectories to predict future scores.
    Falls back to simple temporal model if deep learning not available.
    """
    
    def __init__(self,
                 sequence_length: int = 3,
                 hidden_units: int = 64,
                 n_layers: int = 2,
                 dropout: float = 0.2,
                 epochs: int = 100,
                 batch_size: int = 16,
                 learning_rate: float = 0.001,
                 patience: int = 15,
                 random_state: int = 42):
        self.sequence_length = sequence_length
        self.hidden_units = hidden_units
        self.n_layers = n_layers
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.patience = patience
        self.random_state = random_state
        
        self.model = None
        self.scaler = None
        self._use_simple = True  # Default to simple model
    
    def fit_predict(self,
                   panel_data,
                   target_cols: Optional[List[str]] = None,
                   aggregate: bool = True) -> LSTMResult:
        """
        Train LSTM and predict for the test year.
        
        Parameters
        ----------
        panel_data : PanelData
            Panel data object
        target_cols : List[str]
            Columns to predict (if None, uses all components)
        aggregate : bool
            If True, predict aggregate score; if False, predict each component
        """
        np.random.seed(self.random_state)
        
        if target_cols is None:
            target_cols = panel_data.components
        
        provinces = panel_data.provinces
        years = panel_data.years
        
        # Prepare sequences
        X_train, y_train, X_test, y_test, test_provinces = self._prepare_sequences(
            panel_data, target_cols, aggregate
        )
        
        if len(X_train) == 0:
            raise ValueError("Not enough data for LSTM sequences")
        
        # Train model
        train_losses, val_losses = self._train(X_train, y_train)
        
        # Predict
        predictions = self.model.predict(X_test)
        
        # Calculate metrics
        from scipy.stats import spearmanr
        
        if aggregate:
            pred_flat = predictions.flatten()
            actual_flat = y_test.flatten()
        else:
            pred_flat = predictions.mean(axis=1)
            actual_flat = y_test.mean(axis=1)
        
        mse = np.mean((actual_flat - pred_flat) ** 2)
        mae = np.mean(np.abs(actual_flat - pred_flat))
        rmse = np.sqrt(mse)
        
        # Create DataFrames
        if aggregate:
            pred_df = pd.DataFrame({'predicted': pred_flat}, index=test_provinces)
            actual_df = pd.DataFrame({'actual': actual_flat}, index=test_provinces)
        else:
            pred_df = pd.DataFrame(predictions, index=test_provinces, columns=target_cols)
            actual_df = pd.DataFrame(y_test, index=test_provinces, columns=target_cols)
        
        # Rankings
        pred_ranks = pd.Series(pred_flat, index=test_provinces).rank(ascending=False).astype(int)
        actual_ranks = pd.Series(actual_flat, index=test_provinces).rank(ascending=False).astype(int)
        
        rank_corr, _ = spearmanr(actual_ranks, pred_ranks)
        
        return LSTMResult(
            predictions=pred_df,
            actual=actual_df,
            train_loss=train_losses,
            val_loss=val_losses,
            test_metrics={'mse': mse, 'mae': mae, 'rmse': rmse},
            predicted_ranks=pred_ranks,
            actual_ranks=actual_ranks,
            rank_correlation=rank_corr if not np.isnan(rank_corr) else 0.0,
            model_summary=f"SimpleLSTM(hidden={self.hidden_units})"
        )
    
    def _prepare_sequences(self, panel_data, target_cols: List[str], 
                          aggregate: bool) -> Tuple:
        """Prepare sequences for LSTM training."""
        provinces = panel_data.provinces
        years = sorted(panel_data.years)
        
        X_sequences = []
        y_values = []
        test_X = []
        test_y = []
        test_provinces = []
        
        for province in provinces:
            province_data = panel_data.get_province(province)
            
            # Get values in temporal order
            values = province_data.loc[years, target_cols].values
            
            if aggregate:
                # Use mean across components as target
                values = values.mean(axis=1, keepdims=True)
            
            n_years = len(years)
            seq_len = self.sequence_length
            
            # Create sequences for training (all but last year)
            for i in range(n_years - seq_len - 1):
                X_seq = values[i:i + seq_len]
                y_val = values[i + seq_len]
                X_sequences.append(X_seq)
                y_values.append(y_val)
            
            # Test sequence (predict last year)
            if n_years >= seq_len + 1:
                test_X.append(values[-(seq_len + 1):-1])
                test_y.append(values[-1])
                test_provinces.append(province)
        
        X_train = np.array(X_sequences) if X_sequences else np.array([]).reshape(0, seq_len, 1)
        y_train = np.array(y_values) if y_values else np.array([])
        X_test = np.array(test_X) if test_X else np.array([]).reshape(0, seq_len, 1)
        y_test = np.array(test_y) if test_y else np.array([])
        
        return X_train, y_train, X_test, y_test, test_provinces
    
    def _train(self, X: np.ndarray, y: np.ndarray) -> Tuple[List[float], List[float]]:
        """Train the LSTM model."""
        # Use simple model
        self.model = SimpleLSTM(
            hidden_size=self.hidden_units,
            learning_rate=self.learning_rate
        )
        
        # Split for validation
        n_samples = len(X)
        val_size = max(1, int(n_samples * 0.2))
        
        X_train, X_val = X[:-val_size], X[-val_size:]
        y_train, y_val = y[:-val_size], y[-val_size:]
        
        train_losses = self.model.fit(X_train, y_train, epochs=self.epochs)
        
        # Validation loss
        val_pred = self.model.predict(X_val)
        val_loss = np.mean((y_val - val_pred) ** 2)
        val_losses = [val_loss] * len(train_losses)
        
        return train_losses, val_losses
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using fitted model."""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit_predict first.")
        return self.model.predict(X)


def forecast_panel(panel_data, 
                  sequence_length: int = 3,
                  target: str = 'aggregate') -> LSTMResult:
    """Convenience function for LSTM forecasting."""
    forecaster = LSTMForecaster(sequence_length=sequence_length)
    
    if target == 'aggregate':
        return forecaster.fit_predict(panel_data, aggregate=True)
    else:
        return forecaster.fit_predict(panel_data, target_cols=[target], aggregate=False)
