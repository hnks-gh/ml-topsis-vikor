# -*- coding: utf-8 -*-
"""Panel regression: Fixed Effects, Random Effects, and Pooled OLS."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from scipy import stats


@dataclass
class PanelRegressionResult:
    """Result container for panel regression."""
    model_type: str
    coefficients: pd.Series
    std_errors: pd.Series
    t_stats: pd.Series
    p_values: pd.Series
    r_squared: float
    r_squared_adj: float
    f_stat: float
    f_pvalue: float
    n_obs: int
    n_groups: int
    predictions: pd.Series
    residuals: pd.Series
    fixed_effects: Optional[pd.Series] = None  # Province effects
    time_effects: Optional[pd.Series] = None   # Year effects
    
    def summary(self) -> str:
        """Return regression summary string."""
        lines = [
            f"\n{'='*60}",
            f"PANEL REGRESSION RESULTS ({self.model_type.upper()})",
            f"{'='*60}",
            f"Observations: {self.n_obs}",
            f"Groups (provinces): {self.n_groups}",
            f"R-squared: {self.r_squared:.4f}",
            f"Adjusted R-squared: {self.r_squared_adj:.4f}",
            f"F-statistic: {self.f_stat:.4f} (p={self.f_pvalue:.4e})",
            f"\n{'Coefficient':<15} {'Estimate':<12} {'Std.Err':<12} {'t-stat':<10} {'p-value':<10}",
            "-" * 60
        ]
        
        for var in self.coefficients.index:
            lines.append(
                f"{var:<15} {self.coefficients[var]:>11.4f} {self.std_errors[var]:>11.4f} "
                f"{self.t_stats[var]:>9.3f} {self.p_values[var]:>9.4f}"
            )
        
        lines.append("=" * 60)
        return "\n".join(lines)


class PanelRegression:
    """
    Panel data regression with Fixed Effects, Random Effects, or Pooled OLS.
    
    Implements within-transformation for FE and GLS for RE.
    """
    
    def __init__(self, 
                 model_type: str = "fe",
                 time_effects: bool = True,
                 robust_se: bool = True):
        """
        Initialize panel regression.
        
        Parameters
        ----------
        model_type : str
            'fe' (fixed effects), 're' (random effects), or 'pooled'
        time_effects : bool
            Include year fixed effects
        robust_se : bool
            Use cluster-robust standard errors
        """
        if model_type not in ['fe', 're', 'pooled']:
            raise ValueError("model_type must be 'fe', 're', or 'pooled'")
        
        self.model_type = model_type
        self.time_effects = time_effects
        self.robust_se = robust_se
    
    def fit(self, 
            panel_data,
            y_col: str,
            X_cols: Optional[List[str]] = None,
            province_col: str = 'Province',
            year_col: str = 'Year') -> PanelRegressionResult:
        """
        Fit panel regression model.
        
        Parameters
        ----------
        panel_data : PanelData or DataFrame
            Panel data with Province, Year, and component columns
        y_col : str
            Dependent variable column name
        X_cols : List[str]
            Independent variable columns (if None, use all except y)
        """
        # Get DataFrame
        if hasattr(panel_data, 'long'):
            df = panel_data.long.copy()
        else:
            df = panel_data.copy()
        
        # Set up variables
        if X_cols is None:
            exclude_cols = [province_col, year_col, y_col]
            X_cols = [c for c in df.columns if c not in exclude_cols]
        
        # Ensure numeric
        for col in [y_col] + X_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna(subset=[y_col] + X_cols)
        
        y = df[y_col].values
        X = df[X_cols].values
        provinces = df[province_col].values
        years = df[year_col].values
        
        n_obs = len(y)
        n_groups = len(np.unique(provinces))
        n_years = len(np.unique(years))
        
        # Add time dummies if requested
        if self.time_effects:
            year_dummies = pd.get_dummies(years, prefix='year', drop_first=True)
            X = np.hstack([X, year_dummies.values])
            X_cols = X_cols + list(year_dummies.columns)
        
        # Add constant for pooled
        if self.model_type == 'pooled':
            X = np.column_stack([np.ones(n_obs), X])
            X_cols = ['const'] + X_cols
        
        # Fit model
        if self.model_type == 'fe':
            result = self._fit_fixed_effects(y, X, provinces, X_cols, n_obs, n_groups)
        elif self.model_type == 're':
            result = self._fit_random_effects(y, X, provinces, X_cols, n_obs, n_groups)
        else:
            result = self._fit_pooled_ols(y, X, X_cols, n_obs, n_groups)
        
        return result
    
    def _fit_fixed_effects(self, y: np.ndarray, X: np.ndarray,
                          provinces: np.ndarray, X_cols: List[str],
                          n_obs: int, n_groups: int) -> PanelRegressionResult:
        """Fit Fixed Effects model using within transformation."""
        # Within transformation: demean within each province
        y_demean = np.zeros_like(y)
        X_demean = np.zeros_like(X)
        
        unique_provinces = np.unique(provinces)
        province_means_y = {}
        province_means_X = {}
        
        for p in unique_provinces:
            mask = provinces == p
            province_means_y[p] = y[mask].mean()
            province_means_X[p] = X[mask].mean(axis=0)
            
            y_demean[mask] = y[mask] - province_means_y[p]
            X_demean[mask] = X[mask] - province_means_X[p]
        
        # OLS on demeaned data
        try:
            beta = np.linalg.lstsq(X_demean, y_demean, rcond=None)[0]
        except:
            beta = np.linalg.pinv(X_demean) @ y_demean
        
        # Predictions and residuals
        y_pred_demean = X_demean @ beta
        residuals = y_demean - y_pred_demean
        
        # Full predictions (add back means)
        y_pred = np.zeros_like(y)
        for p in unique_provinces:
            mask = provinces == p
            y_pred[mask] = y_pred_demean[mask] + province_means_y[p]
        
        # Fixed effects (province-specific intercepts)
        fe = {}
        for p in unique_provinces:
            fe[p] = province_means_y[p] - province_means_X[p] @ beta
        
        # Statistics
        k = len(beta)
        dof = n_obs - n_groups - k
        
        # R-squared (within)
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y_demean - y_demean.mean()) ** 2)
        r_sq = 1 - ss_res / (ss_tot + 1e-10)
        r_sq_adj = 1 - (1 - r_sq) * (n_obs - 1) / (dof + 1e-10)
        
        # Standard errors
        mse = ss_res / (dof + 1e-10)
        
        if self.robust_se:
            se = self._cluster_robust_se(X_demean, residuals, provinces, beta)
        else:
            try:
                var_beta = mse * np.linalg.inv(X_demean.T @ X_demean)
                se = np.sqrt(np.diag(var_beta))
            except:
                se = np.ones(k) * np.nan
        
        t_stats = beta / (se + 1e-10)
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), dof))
        
        # F-statistic
        f_stat = (r_sq / k) / ((1 - r_sq) / (dof + 1e-10)) if k > 0 else 0
        f_pvalue = 1 - stats.f.cdf(f_stat, k, dof)
        
        return PanelRegressionResult(
            model_type='fe',
            coefficients=pd.Series(beta, index=X_cols),
            std_errors=pd.Series(se, index=X_cols),
            t_stats=pd.Series(t_stats, index=X_cols),
            p_values=pd.Series(p_values, index=X_cols),
            r_squared=r_sq,
            r_squared_adj=r_sq_adj,
            f_stat=f_stat,
            f_pvalue=f_pvalue,
            n_obs=n_obs,
            n_groups=n_groups,
            predictions=pd.Series(y_pred),
            residuals=pd.Series(residuals),
            fixed_effects=pd.Series(fe)
        )
    
    def _fit_random_effects(self, y: np.ndarray, X: np.ndarray,
                           provinces: np.ndarray, X_cols: List[str],
                           n_obs: int, n_groups: int) -> PanelRegressionResult:
        """Fit Random Effects model using GLS."""
        # First get between and within variances via FE
        unique_provinces = np.unique(provinces)
        T_i = {p: np.sum(provinces == p) for p in unique_provinces}
        
        # Within transformation for variance estimation
        y_demean = np.zeros_like(y)
        X_demean = np.zeros_like(X)
        
        for p in unique_provinces:
            mask = provinces == p
            y_demean[mask] = y[mask] - y[mask].mean()
            X_demean[mask] = X[mask] - X[mask].mean(axis=0)
        
        # Within variance
        try:
            beta_within = np.linalg.lstsq(X_demean, y_demean, rcond=None)[0]
        except:
            beta_within = np.linalg.pinv(X_demean) @ y_demean
        
        resid_within = y_demean - X_demean @ beta_within
        sigma_e_sq = np.sum(resid_within ** 2) / (n_obs - n_groups - len(beta_within))
        
        # Between variance (simplified estimation)
        y_means = np.array([y[provinces == p].mean() for p in unique_provinces])
        X_means = np.array([X[provinces == p].mean(axis=0) for p in unique_provinces])
        
        try:
            X_means_const = np.column_stack([np.ones(n_groups), X_means])
            beta_between = np.linalg.lstsq(X_means_const, y_means, rcond=None)[0]
        except:
            beta_between = np.zeros(X_means.shape[1] + 1)
        
        resid_between = y_means - X_means_const @ beta_between
        sigma_u_sq = max(0, np.var(resid_between) - sigma_e_sq / np.mean(list(T_i.values())))
        
        # GLS transformation
        theta = {}
        for p in unique_provinces:
            theta[p] = 1 - np.sqrt(sigma_e_sq / (T_i[p] * sigma_u_sq + sigma_e_sq + 1e-10))
        
        y_gls = np.zeros_like(y)
        X_gls = np.zeros_like(X)
        
        for p in unique_provinces:
            mask = provinces == p
            y_mean = y[mask].mean()
            X_mean = X[mask].mean(axis=0)
            
            y_gls[mask] = y[mask] - theta[p] * y_mean
            X_gls[mask] = X[mask] - theta[p] * X_mean
        
        # Add constant
        X_gls = np.column_stack([np.ones(n_obs) * (1 - np.mean(list(theta.values()))), X_gls])
        X_cols_re = ['const'] + X_cols
        
        # GLS estimation
        try:
            beta = np.linalg.lstsq(X_gls, y_gls, rcond=None)[0]
        except:
            beta = np.linalg.pinv(X_gls) @ y_gls
        
        y_pred = X_gls @ beta
        residuals = y_gls - y_pred
        
        # Statistics
        k = len(beta)
        dof = n_obs - k
        
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y_gls - y_gls.mean()) ** 2)
        r_sq = 1 - ss_res / (ss_tot + 1e-10)
        r_sq_adj = 1 - (1 - r_sq) * (n_obs - 1) / (dof + 1e-10)
        
        mse = ss_res / (dof + 1e-10)
        
        if self.robust_se:
            se = self._cluster_robust_se(X_gls, residuals, provinces, beta)
        else:
            try:
                var_beta = mse * np.linalg.inv(X_gls.T @ X_gls)
                se = np.sqrt(np.diag(var_beta))
            except:
                se = np.ones(k) * np.nan
        
        t_stats = beta / (se + 1e-10)
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), dof))
        
        f_stat = (r_sq / k) / ((1 - r_sq) / (dof + 1e-10)) if k > 0 else 0
        f_pvalue = 1 - stats.f.cdf(f_stat, k, dof)
        
        return PanelRegressionResult(
            model_type='re',
            coefficients=pd.Series(beta, index=X_cols_re),
            std_errors=pd.Series(se, index=X_cols_re),
            t_stats=pd.Series(t_stats, index=X_cols_re),
            p_values=pd.Series(p_values, index=X_cols_re),
            r_squared=r_sq,
            r_squared_adj=r_sq_adj,
            f_stat=f_stat,
            f_pvalue=f_pvalue,
            n_obs=n_obs,
            n_groups=n_groups,
            predictions=pd.Series(y_pred),
            residuals=pd.Series(residuals)
        )
    
    def _fit_pooled_ols(self, y: np.ndarray, X: np.ndarray,
                       X_cols: List[str], n_obs: int, 
                       n_groups: int) -> PanelRegressionResult:
        """Fit Pooled OLS model."""
        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
        except:
            beta = np.linalg.pinv(X) @ y
        
        y_pred = X @ beta
        residuals = y - y_pred
        
        k = len(beta)
        dof = n_obs - k
        
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r_sq = 1 - ss_res / (ss_tot + 1e-10)
        r_sq_adj = 1 - (1 - r_sq) * (n_obs - 1) / (dof + 1e-10)
        
        mse = ss_res / (dof + 1e-10)
        
        try:
            var_beta = mse * np.linalg.inv(X.T @ X)
            se = np.sqrt(np.diag(var_beta))
        except:
            se = np.ones(k) * np.nan
        
        t_stats = beta / (se + 1e-10)
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), dof))
        
        f_stat = (r_sq / k) / ((1 - r_sq) / (dof + 1e-10)) if k > 0 else 0
        f_pvalue = 1 - stats.f.cdf(f_stat, k, dof)
        
        return PanelRegressionResult(
            model_type='pooled',
            coefficients=pd.Series(beta, index=X_cols),
            std_errors=pd.Series(se, index=X_cols),
            t_stats=pd.Series(t_stats, index=X_cols),
            p_values=pd.Series(p_values, index=X_cols),
            r_squared=r_sq,
            r_squared_adj=r_sq_adj,
            f_stat=f_stat,
            f_pvalue=f_pvalue,
            n_obs=n_obs,
            n_groups=n_groups,
            predictions=pd.Series(y_pred),
            residuals=pd.Series(residuals)
        )
    
    def _cluster_robust_se(self, X: np.ndarray, residuals: np.ndarray,
                          clusters: np.ndarray, beta: np.ndarray) -> np.ndarray:
        """Calculate cluster-robust standard errors."""
        n, k = X.shape
        unique_clusters = np.unique(clusters)
        n_clusters = len(unique_clusters)
        
        try:
            XtX_inv = np.linalg.inv(X.T @ X)
        except:
            XtX_inv = np.linalg.pinv(X.T @ X)
        
        # Cluster sum of score vectors
        meat = np.zeros((k, k))
        for c in unique_clusters:
            mask = clusters == c
            X_c = X[mask]
            e_c = residuals[mask]
            score_c = X_c.T @ e_c
            meat += np.outer(score_c, score_c)
        
        # Sandwich estimator with small-sample correction
        correction = n_clusters / (n_clusters - 1) * (n - 1) / (n - k)
        var_robust = correction * XtX_inv @ meat @ XtX_inv
        
        return np.sqrt(np.diag(var_robust))


def hausman_test(fe_result: PanelRegressionResult, 
                 re_result: PanelRegressionResult) -> Dict:
    """
    Hausman specification test for FE vs RE.
    
    H0: RE is consistent and efficient
    H1: FE is consistent, RE is inconsistent
    """
    # Get common coefficients (exclude const in RE)
    fe_coef = fe_result.coefficients
    re_coef = re_result.coefficients.drop('const', errors='ignore')
    
    common_vars = fe_coef.index.intersection(re_coef.index)
    
    b_fe = fe_coef[common_vars].values
    b_re = re_coef[common_vars].values
    
    # Difference
    diff = b_fe - b_re
    
    # Variance of difference (simplified)
    var_diff = np.diag(fe_result.std_errors[common_vars].values ** 2)
    
    try:
        chi2_stat = diff @ np.linalg.inv(var_diff) @ diff
    except:
        chi2_stat = np.sum(diff ** 2 / (fe_result.std_errors[common_vars].values ** 2 + 1e-10))
    
    dof = len(common_vars)
    p_value = 1 - stats.chi2.cdf(chi2_stat, dof)
    
    return {
        'chi2_statistic': chi2_stat,
        'p_value': p_value,
        'dof': dof,
        'preferred_model': 'fe' if p_value < 0.05 else 're',
        'interpretation': (
            "Reject H0: Use Fixed Effects" if p_value < 0.05 
            else "Cannot reject H0: Random Effects is consistent"
        )
    }
