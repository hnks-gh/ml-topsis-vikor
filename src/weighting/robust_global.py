# -*- coding: utf-8 -*-
"""
Robust Global Hybrid Weighting (PCA-CRITIC-Entropy + KL-Divergence Fusion)

A 7-step pipeline for determining objective criteria weights from panel data,
operating on the full panel (all entities × all time periods) simultaneously
to preserve temporal trends and maximize information utilization.

Steps:
    1. Global Min-Max Normalization (preserves N-year temporal trend)
    2. PCA Structural Decomposition & Residualization
    3. PCA-Residualized CRITIC Weights (σ from global, r from residual)
    4. Global Entropy Weights (Shannon entropy on full panel)
    5. PCA Loadings-based Weights (eigenstructure-derived)
    6. KL-Divergence Fusion (geometric mean of 3 weight vectors)
    7. Bayesian Bootstrap validation (Dirichlet-weighted, B=999 iterations)
    + Split-half stability verification (cosine similarity + Spearman)

Mathematical Foundation:
    - CRITIC: Diakoulaki, Mavrotas & Papayannakis (1995)
    - Entropy: Shannon (1948)
    - PCA Weights: Deng, Yeh & Willis (2000)
    - KL-Divergence Fusion: Genest & Zidek (1986), Abbas (2009)
    - Bayesian Bootstrap: Rubin (1981)
    - Bootstrap Intervals: Davison & Hinkley (1997)

References:
    Diakoulaki, D. et al. (1995). Determining objective weights in multiple
        criteria problems: The CRITIC method. Computers & Ops Research.
    Genest, C. & Zidek, J.V. (1986). Combining probability distributions:
        A critique and annotated bibliography. Statistical Science.
    Abbas, A.E. (2009). A Kullback-Leibler view of linear and log-linear
        pools. Decision Analysis.
    Rubin, D.B. (1981). The Bayesian Bootstrap. Annals of Statistics.
    Davison, A.C. & Hinkley, D.V. (1997). Bootstrap Methods and Their
        Application. Cambridge University Press.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy import stats
import logging

from .base import WeightResult

logger = logging.getLogger(__name__)


class RobustGlobalWeighting:
    """
    Robust Global Hybrid Weighting for panel MCDM data.

    Computes objective criteria weights via a 7-step pipeline that operates
    on the full panel matrix (all entities × all time periods), preserving
    temporal dynamics and extracting unique criterion information through
    PCA residualization.

    Three weight vectors (Entropy, PCA-Residualized CRITIC, PCA Loadings)
    are fused via KL-Divergence minimization (geometric mean), then
    validated through Bayesian Bootstrap with Dirichlet observation weights.

    Parameters
    ----------
    pca_variance_threshold : float
        Cumulative variance threshold for PCA component retention.
        Default 0.80 (appropriate for p≥20 correlated indicators;
        avoids over-extraction per Zwick & Velicer, 1986).
    bootstrap_iterations : int
        Number of Bayesian Bootstrap iterations. Use odd numbers for
        clean percentile-based credible intervals (Davison & Hinkley, 1997).
        Default 999.
    fusion_alphas : list of float
        KL-divergence fusion coefficients for [entropy, critic, pca].
        Default [1/3, 1/3, 1/3] (equal information-theoretic weighting).
    stability_threshold : float
        Minimum cosine similarity for split-half weight stability.
        Default 0.95.
    epsilon : float
        Numerical stability constant for log/division operations.
    seed : int
        Random seed for reproducibility of bootstrap.

    Examples
    --------
    >>> from src.weighting import RobustGlobalWeighting
    >>> calc = RobustGlobalWeighting()
    >>> result = calc.calculate(panel_df, 'Province', 'Year',
    ...                         ['C01', 'C02', ..., 'C29'])
    >>> print(result.weights)  # dict of criterion → weight
    >>> print(result.details['stability'])  # split-half verification
    """

    def __init__(
        self,
        pca_variance_threshold: float = 0.80,
        bootstrap_iterations: int = 999,
        fusion_alphas: Optional[List[float]] = None,
        stability_threshold: float = 0.95,
        epsilon: float = 1e-10,
        seed: int = 42,
    ):
        self.pca_variance_threshold = pca_variance_threshold
        self.bootstrap_iterations = bootstrap_iterations
        self.fusion_alphas = fusion_alphas or [1/3, 1/3, 1/3]
        self.stability_threshold = stability_threshold
        self.epsilon = epsilon
        self.seed = seed

        # Validate fusion alphas
        if len(self.fusion_alphas) != 3:
            raise ValueError("fusion_alphas must have exactly 3 elements "
                             "[entropy, critic, pca]")
        alpha_sum = sum(self.fusion_alphas)
        self.fusion_alphas = [a / alpha_sum for a in self.fusion_alphas]

    # =====================================================================
    # PUBLIC API
    # =====================================================================

    def calculate(
        self,
        panel_df: pd.DataFrame,
        entity_col: str = "Province",
        time_col: str = "Year",
        criteria_cols: Optional[List[str]] = None,
    ) -> WeightResult:
        """
        Execute the full 7-step Robust Global Hybrid Weighting pipeline.

        Parameters
        ----------
        panel_df : pd.DataFrame
            Long-format panel data with columns for entity, time, and criteria.
        entity_col : str
            Column name identifying entities (e.g., 'Province').
        time_col : str
            Column name identifying time periods (e.g., 'Year').
        criteria_cols : list of str, optional
            Criterion column names. If None, inferred as all numeric columns
            excluding entity_col and time_col.

        Returns
        -------
        WeightResult
            Final weights with comprehensive details including:
            - Individual weight vectors (entropy, critic, pca)
            - Bootstrap statistics (mean, std, 95% credible intervals)
            - Stability verification (cosine similarity, Spearman correlation)
            - PCA decomposition details
        """
        # Validate and prepare data
        panel_df = panel_df.copy()
        if criteria_cols is None:
            criteria_cols = [c for c in panel_df.columns
                            if c not in (entity_col, time_col)
                            and pd.api.types.is_numeric_dtype(panel_df[c])]

        n_obs = len(panel_df)
        n_criteria = len(criteria_cols)
        logger.info(f"Robust Global Weighting: {n_obs} observations × "
                    f"{n_criteria} criteria")

        # Extract criteria matrix
        X_raw = panel_df[criteria_cols].values.astype(np.float64)

        # ── Step 1: Global Min-Max Normalization ──
        X_norm = self._global_min_max_normalize(X_raw)
        logger.info("Step 1: Global Min-Max normalization complete")

        # ── Step 2: PCA Structural Decomposition ──
        pca_info = self._pca_residualize(X_norm)
        R = pca_info["residual"]
        logger.info(f"Step 2: PCA residualization — {pca_info['n_components']} "
                    f"components retained ({pca_info['variance_explained']:.1%} "
                    f"variance)")

        # ── Step 3: PCA-Residualized CRITIC Weights ──
        W_c = self._critic_weights(X_norm, R)
        logger.info(f"Step 3: CRITIC weights — range [{W_c.min():.4f}, "
                    f"{W_c.max():.4f}]")

        # ── Step 4: Global Entropy Weights ──
        W_e = self._entropy_weights(X_norm)
        logger.info(f"Step 4: Entropy weights — range [{W_e.min():.4f}, "
                    f"{W_e.max():.4f}]")

        # ── Step 5: PCA Loadings-based Weights ──
        W_p = self._pca_weights(X_norm)
        logger.info(f"Step 5: PCA weights — range [{W_p.min():.4f}, "
                    f"{W_p.max():.4f}]")

        # ── Step 6: KL-Divergence Fusion ──
        W_fused = self._kl_divergence_fusion(W_e, W_c, W_p)
        logger.info(f"Step 6: KL-Divergence fusion — range [{W_fused.min():.4f}, "
                    f"{W_fused.max():.4f}]")

        # ── Step 7: Bayesian Bootstrap ──
        bootstrap_results = self._bayesian_bootstrap(X_norm)
        W_final = bootstrap_results["mean_weights"]
        logger.info(f"Step 7: Bayesian Bootstrap ({self.bootstrap_iterations} "
                    f"iterations) — mean weight std: "
                    f"{bootstrap_results['std_weights'].mean():.6f}")

        # ── Stability Verification ──
        time_values = panel_df[time_col].values
        stability = self._stability_verification(X_raw, time_values)
        logger.info(f"Stability: cosine={stability['cosine_similarity']:.4f}, "
                    f"spearman={stability['spearman_correlation']:.4f}")

        # Build result
        weights_dict = {col: float(W_final[j])
                        for j, col in enumerate(criteria_cols)}

        details = {
            # Individual weight vectors
            "individual_weights": {
                "entropy": {col: float(W_e[j])
                            for j, col in enumerate(criteria_cols)},
                "critic": {col: float(W_c[j])
                           for j, col in enumerate(criteria_cols)},
                "pca": {col: float(W_p[j])
                        for j, col in enumerate(criteria_cols)},
                "kl_fused": {col: float(W_fused[j])
                             for j, col in enumerate(criteria_cols)},
            },
            # Fusion
            "fusion_alphas": {
                "entropy": self.fusion_alphas[0],
                "critic": self.fusion_alphas[1],
                "pca": self.fusion_alphas[2],
            },
            # Bootstrap statistics
            "bootstrap": {
                "iterations": self.bootstrap_iterations,
                "mean_weights": {col: float(W_final[j])
                                 for j, col in enumerate(criteria_cols)},
                "std_weights": {col: float(bootstrap_results["std_weights"][j])
                                for j, col in enumerate(criteria_cols)},
                "ci_lower_2_5": {col: float(bootstrap_results["ci_lower"][j])
                                 for j, col in enumerate(criteria_cols)},
                "ci_upper_97_5": {col: float(bootstrap_results["ci_upper"][j])
                                  for j, col in enumerate(criteria_cols)},
            },
            # PCA decomposition
            "pca": {
                "n_components": pca_info["n_components"],
                "variance_explained": pca_info["variance_explained"],
                "eigenvalues": pca_info["eigenvalues"].tolist(),
                "explained_ratio": pca_info["explained_ratio"].tolist(),
            },
            # Stability verification
            "stability": stability,
            # Dimensions
            "n_observations": n_obs,
            "n_criteria": n_criteria,
        }

        return WeightResult(
            weights=weights_dict,
            method="robust_global_hybrid",
            details=details,
        )

    # =====================================================================
    # STEP 1: GLOBAL MIN-MAX NORMALIZATION
    # =====================================================================

    def _global_min_max_normalize(self, X: np.ndarray) -> np.ndarray:
        """
        Normalize across the entire panel to preserve temporal trends.

        Formula: x_norm = (x - min_global) / (max_global - min_global) + ε

        The epsilon shift ensures no exact zeros, which would cause
        problems in entropy calculation (0·ln(0) is undefined).

        Parameters
        ----------
        X : np.ndarray, shape (N, p)
            Raw criteria matrix (all entities × all time periods).

        Returns
        -------
        np.ndarray, shape (N, p)
            Globally normalized matrix with values in (ε, 1+ε].
        """
        col_min = X.min(axis=0)
        col_max = X.max(axis=0)
        denom = col_max - col_min
        denom[denom < self.epsilon] = self.epsilon  # avoid division by zero

        X_norm = (X - col_min) / denom + self.epsilon
        return X_norm

    # =====================================================================
    # STEP 2: PCA STRUCTURAL DECOMPOSITION & RESIDUALIZATION
    # =====================================================================

    def _pca_residualize(self, X_norm: np.ndarray) -> Dict:
        """
        Remove dominant common trends via PCA, yielding residuals that
        capture each criterion's unique information.

        Uses cumulative variance threshold (default 0.80) rather than
        the Kaiser rule (eigenvalue > 1), which over-extracts when
        p > 20 variables (Zwick & Velicer, 1986).

        Formula:
            X_hat = Z @ V_K^T @ V_K  (reconstruction from top-K PCs)
            R = Z - X_hat             (residual matrix)

        where Z is the standardized X_norm, and V_K contains the top-K
        eigenvectors.

        Parameters
        ----------
        X_norm : np.ndarray, shape (N, p)
            Globally normalized criteria matrix.

        Returns
        -------
        dict
            Keys: 'residual' (N×p), 'reconstructed' (N×p), 'n_components',
                  'variance_explained', 'eigenvalues', 'explained_ratio',
                  'components' (K×p), 'scaler', 'pca_model'.
        """
        N, p = X_norm.shape

        # Standardize for PCA
        scaler = StandardScaler()
        Z = scaler.fit_transform(X_norm)

        # Fit PCA with all components
        pca = PCA(n_components=min(N, p))
        pca.fit(Z)

        eigenvalues = pca.explained_variance_
        explained_ratio = pca.explained_variance_ratio_
        cumulative = np.cumsum(explained_ratio)

        # Select components via cumulative variance threshold
        n_components = max(1, int(np.searchsorted(
            cumulative, self.pca_variance_threshold) + 1))
        n_components = min(n_components, p - 1)  # keep at least 1 residual dim

        # Reconstruct and residualize
        V_K = pca.components_[:n_components]  # (K, p)
        Z_hat = Z @ V_K.T @ V_K              # (N, p)
        Z_residual = Z - Z_hat                # (N, p)

        return {
            "residual": Z_residual,
            "reconstructed": Z_hat,
            "n_components": n_components,
            "variance_explained": float(cumulative[n_components - 1]),
            "eigenvalues": eigenvalues,
            "explained_ratio": explained_ratio,
            "components": V_K,
            "scaler": scaler,
            "pca_model": pca,
        }

    # =====================================================================
    # STEP 3: PCA-RESIDUALIZED CRITIC WEIGHTS
    # =====================================================================

    def _critic_weights(
        self, X_norm: np.ndarray, R: np.ndarray
    ) -> np.ndarray:
        """
        CRITIC weights using σ from the global matrix and Pearson r from
        the PCA-residualized matrix.

        Rationale: σ captures absolute contrast intensity (how much a
        criterion varies), while the residualized r captures unique
        conflict (how differently a criterion behaves relative to the
        dominant common trends, per Diakoulaki et al. 1995).

        Formula:
            f_j = Σ_k (1 - r_jk^residual)   (conflict from residual)
            C_j = σ_j^global × f_j            (information content)
            W_cj = C_j / Σ C_k               (normalized weight)

        Parameters
        ----------
        X_norm : np.ndarray, shape (N, p)
            Globally normalized criteria matrix (for σ).
        R : np.ndarray, shape (N, p)
            PCA-residualized matrix (for correlation).

        Returns
        -------
        np.ndarray, shape (p,)
            Normalized CRITIC weights.
        """
        p = X_norm.shape[1]

        # Standard deviation from global normalized matrix
        sigma = np.std(X_norm, axis=0, ddof=1)
        sigma[sigma < self.epsilon] = self.epsilon

        # Correlation from residual matrix
        R_df = pd.DataFrame(R)
        # Handle near-zero variance columns in residuals
        r_std = R_df.std()
        for col in r_std[r_std < self.epsilon].index:
            R_df[col] += np.random.RandomState(self.seed).normal(
                0, self.epsilon, size=len(R_df))
        corr = R_df.corr().fillna(0.0).values

        # Conflict measure
        conflict = np.sum(1 - corr, axis=1)

        # Information content
        C = sigma * conflict
        C[C < self.epsilon] = self.epsilon

        # Normalize
        weights = C / C.sum()
        return weights

    # =====================================================================
    # STEP 4: GLOBAL ENTROPY WEIGHTS
    # =====================================================================

    def _entropy_weights(self, X_norm: np.ndarray) -> np.ndarray:
        """
        Shannon entropy weights on the full panel.

        Formula:
            p_ij = x_ij / Σ_i x_ij          (column proportions)
            e_j  = -(1/ln(N)) Σ p_ij ln(p_ij) (normalized entropy)
            d_j  = 1 - e_j                     (divergence coefficient)
            W_ej = d_j / Σ d_k                 (normalized weight)

        Parameters
        ----------
        X_norm : np.ndarray, shape (N, p)
            Globally normalized criteria matrix (all values > 0 due to
            epsilon shift in Step 1).

        Returns
        -------
        np.ndarray, shape (p,)
            Normalized entropy weights.
        """
        N, p = X_norm.shape

        # Column proportions
        col_sums = X_norm.sum(axis=0)
        col_sums[col_sums < self.epsilon] = self.epsilon
        P = X_norm / col_sums  # (N, p)

        # Clamp for numerical safety
        P = np.clip(P, self.epsilon, None)

        # Entropy
        k = 1.0 / np.log(N)
        E = -k * np.sum(P * np.log(P), axis=0)  # (p,)

        # Divergence
        D = 1.0 - E
        D = np.clip(D, self.epsilon, None)

        # Normalize
        weights = D / D.sum()
        return weights

    # =====================================================================
    # STEP 5: PCA LOADINGS-BASED WEIGHTS
    # =====================================================================

    def _pca_weights(self, X_norm: np.ndarray) -> np.ndarray:
        """
        Derive weights from PCA eigenstructure (loading-squared method).

        Formula:
            w_j = Σ_k (λ_k / Σλ) × v_jk²

        where λ_k are eigenvalues and v_jk are PC loadings, retaining
        components up to the variance threshold.

        Parameters
        ----------
        X_norm : np.ndarray, shape (N, p)
            Globally normalized criteria matrix.

        Returns
        -------
        np.ndarray, shape (p,)
            Normalized PCA weights.
        """
        N, p = X_norm.shape

        scaler = StandardScaler()
        Z = scaler.fit_transform(X_norm)

        pca = PCA(n_components=min(N, p))
        pca.fit(Z)

        eigenvalues = pca.explained_variance_
        explained_ratio = pca.explained_variance_ratio_
        cumulative = np.cumsum(explained_ratio)

        n_retained = max(1, int(np.searchsorted(
            cumulative, self.pca_variance_threshold) + 1))
        n_retained = min(n_retained, len(eigenvalues))

        # Variance proportion for retained components
        retained_ev = eigenvalues[:n_retained]
        proportion = retained_ev / (retained_ev.sum() + self.epsilon)

        # Loading-squared weights
        components = pca.components_[:n_retained]  # (K, p)
        raw_weights = np.zeros(p)
        for k in range(n_retained):
            raw_weights += proportion[k] * (components[k, :] ** 2)

        raw_weights = np.clip(raw_weights, self.epsilon, None)
        weights = raw_weights / raw_weights.sum()
        return weights

    # =====================================================================
    # STEP 6: KL-DIVERGENCE FUSION (GEOMETRIC MEAN)
    # =====================================================================

    def _kl_divergence_fusion(
        self,
        W_e: np.ndarray,
        W_c: np.ndarray,
        W_p: np.ndarray,
        alphas: Optional[List[float]] = None,
    ) -> np.ndarray:
        """
        Fuse weight vectors via KL-divergence minimization.

        The geometric mean of probability distributions minimizes the
        total KL-divergence from all source distributions (Genest & Zidek,
        1986). This is information-theoretically optimal and conservative
        at the tails — requiring consensus among methods.

        Formula:
            w_j* ∝ Π_k (w_j^(k))^(α_k)

        which is equivalent to:
            log(w_j*) = Σ_k α_k log(w_j^(k))  + const

        Parameters
        ----------
        W_e : np.ndarray, shape (p,)
            Entropy weights.
        W_c : np.ndarray, shape (p,)
            CRITIC weights.
        W_p : np.ndarray, shape (p,)
            PCA weights.
        alphas : list of float, optional
            Fusion coefficients. If None, uses self.fusion_alphas.

        Returns
        -------
        np.ndarray, shape (p,)
            Fused weights (sum to 1).
        """
        alphas = alphas or self.fusion_alphas

        # Ensure no zeros (for log safety)
        W_e = np.clip(W_e, self.epsilon, None)
        W_c = np.clip(W_c, self.epsilon, None)
        W_p = np.clip(W_p, self.epsilon, None)

        # Geometric mean in log space
        log_w = (alphas[0] * np.log(W_e) +
                 alphas[1] * np.log(W_c) +
                 alphas[2] * np.log(W_p))

        w_star = np.exp(log_w)
        w_star = np.clip(w_star, self.epsilon, None)
        w_star /= w_star.sum()

        return w_star

    # =====================================================================
    # STEP 7: BAYESIAN BOOTSTRAP (DIRICHLET-WEIGHTED)
    # =====================================================================

    def _bayesian_bootstrap(self, X_norm: np.ndarray) -> Dict:
        """
        Bayesian Bootstrap validation of the full weighting pipeline.

        Unlike standard bootstrapping (row resampling), the Bayesian
        Bootstrap (Rubin, 1981) assigns random Dirichlet weights to
        observations, preserving the full dataset while varying the
        emphasis on each observation.

        For each of B iterations:
          1. Draw observation weights from Dirichlet(1, ..., 1)
          2. Compute weighted PCA residualization
          3. Compute weighted CRITIC, Entropy, and PCA weights
          4. Fuse via KL-divergence

        Output: posterior mean (final weights), std, 95% credible intervals.

        Parameters
        ----------
        X_norm : np.ndarray, shape (N, p)
            Globally normalized criteria matrix.

        Returns
        -------
        dict
            Keys: 'mean_weights', 'std_weights', 'ci_lower', 'ci_upper',
                  'all_weights' (B×p matrix of bootstrap weight vectors).
        """
        N, p = X_norm.shape
        B = self.bootstrap_iterations
        rng = np.random.RandomState(self.seed)

        all_weights = np.zeros((B, p))

        for b in range(B):
            # Draw Dirichlet(1,...,1) weights (equivalent to Exp(1) normalized)
            g = rng.exponential(1.0, size=N)
            obs_weights = g / g.sum()  # (N,)

            try:
                # Weighted pipeline
                W_e = self._weighted_entropy(X_norm, obs_weights)
                R_w = self._weighted_pca_residualize(X_norm, obs_weights)
                W_c = self._weighted_critic(X_norm, R_w, obs_weights)
                W_p = self._weighted_pca_weights(X_norm, obs_weights)
                W_fused = self._kl_divergence_fusion(W_e, W_c, W_p)
                all_weights[b, :] = W_fused
            except Exception:
                # Fallback: use unweighted if numerical issues with this draw
                all_weights[b, :] = all_weights[max(0, b-1), :]

        # Posterior statistics
        mean_weights = all_weights.mean(axis=0)
        mean_weights /= mean_weights.sum()  # ensure sum-to-one

        std_weights = all_weights.std(axis=0)

        # 95% credible intervals (2.5th and 97.5th percentiles)
        ci_lower = np.percentile(all_weights, 2.5, axis=0)
        ci_upper = np.percentile(all_weights, 97.5, axis=0)

        return {
            "mean_weights": mean_weights,
            "std_weights": std_weights,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "all_weights": all_weights,
        }

    # =====================================================================
    # STABILITY VERIFICATION
    # =====================================================================

    def _stability_verification(
        self,
        X_raw: np.ndarray,
        time_values: np.ndarray,
    ) -> Dict:
        """
        Split-half stability check: compare weights from the first half
        of time periods vs. the second half.

        Uses both cosine similarity (vector magnitude agreement) and
        Spearman rank correlation (ordinal agreement) for robustness.

        Parameters
        ----------
        X_raw : np.ndarray, shape (N, p)
            Raw (unnormalized) criteria matrix.
        time_values : np.ndarray, shape (N,)
            Time period for each observation.

        Returns
        -------
        dict
            Keys: 'cosine_similarity', 'spearman_correlation',
                  'spearman_pvalue', 'is_stable', 'split_point'.
        """
        unique_times = np.sort(np.unique(time_values))
        mid = len(unique_times) // 2
        split_point = unique_times[mid]

        mask_first = time_values < split_point
        mask_second = time_values >= split_point

        if mask_first.sum() < 2 or mask_second.sum() < 2:
            return {
                "cosine_similarity": np.nan,
                "spearman_correlation": np.nan,
                "spearman_pvalue": np.nan,
                "is_stable": False,
                "split_point": int(split_point),
                "note": "Insufficient data for split-half verification",
            }

        # Compute weights on each half independently
        X_first = X_raw[mask_first]
        X_second = X_raw[mask_second]

        X_first_norm = self._global_min_max_normalize(X_first)
        X_second_norm = self._global_min_max_normalize(X_second)

        W_first = self._compute_fused_weights(X_first_norm)
        W_second = self._compute_fused_weights(X_second_norm)

        # Cosine similarity
        cos_sim = float(np.dot(W_first, W_second) /
                        (np.linalg.norm(W_first) * np.linalg.norm(W_second)
                         + self.epsilon))

        # Spearman rank correlation
        spearman_r, spearman_p = stats.spearmanr(W_first, W_second)

        return {
            "cosine_similarity": cos_sim,
            "spearman_correlation": float(spearman_r),
            "spearman_pvalue": float(spearman_p),
            "is_stable": cos_sim >= self.stability_threshold,
            "split_point": int(split_point),
        }

    # =====================================================================
    # WEIGHTED STATISTIC HELPERS (for Bayesian Bootstrap)
    # =====================================================================

    def _weighted_mean(self, X: np.ndarray, w: np.ndarray) -> np.ndarray:
        """Weighted column means. w shape (N,), X shape (N, p)."""
        return np.dot(w, X)  # (p,)

    def _weighted_cov(self, X: np.ndarray, w: np.ndarray) -> np.ndarray:
        """
        Weighted covariance matrix.

        Uses the reliability-weighted formula:
            Σ = Σ_i w_i (x_i - μ_w)(x_i - μ_w)^T / (1 - Σ w_i²)
        """
        mu = self._weighted_mean(X, w)
        X_centered = X - mu  # (N, p)
        # Weighted cross-product
        cov = (X_centered * w[:, np.newaxis]).T @ X_centered
        # Bessel correction for weighted data
        correction = 1.0 - np.sum(w ** 2)
        if correction < self.epsilon:
            correction = self.epsilon
        cov /= correction
        return cov  # (p, p)

    def _weighted_std(self, X: np.ndarray, w: np.ndarray) -> np.ndarray:
        """Weighted standard deviations from weighted covariance diagonal."""
        cov = self._weighted_cov(X, w)
        var = np.diag(cov)
        var = np.clip(var, 0, None)
        return np.sqrt(var)

    def _weighted_corr(self, X: np.ndarray, w: np.ndarray) -> np.ndarray:
        """Weighted Pearson correlation matrix."""
        cov = self._weighted_cov(X, w)
        std = np.sqrt(np.clip(np.diag(cov), self.epsilon, None))
        corr = cov / np.outer(std, std)
        # Clamp to [-1, 1]
        corr = np.clip(corr, -1, 1)
        np.fill_diagonal(corr, 1.0)
        return corr

    def _weighted_entropy(
        self, X_norm: np.ndarray, w: np.ndarray
    ) -> np.ndarray:
        """
        Entropy weights with observation weights.

        p_ij = w_i * x_ij / Σ_i(w_i * x_ij) for each column j.
        """
        N, p = X_norm.shape
        W_X = X_norm * w[:, np.newaxis]  # (N, p), observation-weighted values
        col_sums = W_X.sum(axis=0)
        col_sums[col_sums < self.epsilon] = self.epsilon
        P = W_X / col_sums
        P = np.clip(P, self.epsilon, None)

        # Effective sample size for normalization
        # Use log(N) as conventional constant (observations are weighted,
        # not reduced)
        k = 1.0 / np.log(N)
        E = -k * np.sum(P * np.log(P), axis=0)

        D = 1.0 - E
        D = np.clip(D, self.epsilon, None)
        weights = D / D.sum()
        return weights

    def _weighted_pca_residualize(
        self, X_norm: np.ndarray, w: np.ndarray
    ) -> np.ndarray:
        """
        PCA residualization using weighted covariance.

        Instead of sklearn's PCA (which uses uniform weights), we:
        1. Compute the weighted covariance matrix
        2. Eigen-decompose it
        3. Select top-K components by cumulative variance
        4. Project and subtract to get residuals
        """
        N, p = X_norm.shape

        mu_w = self._weighted_mean(X_norm, w)
        X_centered = X_norm - mu_w  # (N, p)

        # Weighted covariance
        cov_w = self._weighted_cov(X_norm, w)  # (p, p)

        # Standardize using weighted std
        std_w = np.sqrt(np.clip(np.diag(cov_w), self.epsilon, None))
        Z = X_centered / std_w  # (N, p)

        # Correlation matrix from weighted cov
        corr_w = cov_w / np.outer(std_w, std_w)
        corr_w = np.clip(corr_w, -1, 1)
        np.fill_diagonal(corr_w, 1.0)

        # Eigen-decomposition of weighted correlation
        eigenvalues, eigenvectors = np.linalg.eigh(corr_w)
        # Sort descending
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Clip negative eigenvalues (numerical artifact)
        eigenvalues = np.clip(eigenvalues, 0, None)
        total_var = eigenvalues.sum()
        if total_var < self.epsilon:
            return Z  # degenerate case

        explained_ratio = eigenvalues / total_var
        cumulative = np.cumsum(explained_ratio)

        n_components = max(1, int(np.searchsorted(
            cumulative, self.pca_variance_threshold) + 1))
        n_components = min(n_components, p - 1)

        # Residualize
        V_K = eigenvectors[:, :n_components]  # (p, K)
        Z_hat = Z @ V_K @ V_K.T             # (N, p)
        Z_residual = Z - Z_hat               # (N, p)

        return Z_residual

    def _weighted_critic(
        self,
        X_norm: np.ndarray,
        R: np.ndarray,
        w: np.ndarray,
    ) -> np.ndarray:
        """CRITIC with weighted σ (from X_norm) and weighted r (from R)."""
        sigma = self._weighted_std(X_norm, w)
        sigma[sigma < self.epsilon] = self.epsilon

        corr = self._weighted_corr(R, w)
        conflict = np.sum(1 - corr, axis=1)

        C = sigma * conflict
        C[C < self.epsilon] = self.epsilon
        weights = C / C.sum()
        return weights

    def _weighted_pca_weights(
        self, X_norm: np.ndarray, w: np.ndarray
    ) -> np.ndarray:
        """PCA loadings-based weights using weighted covariance."""
        N, p = X_norm.shape

        cov_w = self._weighted_cov(X_norm, w)
        std_w = np.sqrt(np.clip(np.diag(cov_w), self.epsilon, None))
        corr_w = cov_w / np.outer(std_w, std_w)
        corr_w = np.clip(corr_w, -1, 1)
        np.fill_diagonal(corr_w, 1.0)

        eigenvalues, eigenvectors = np.linalg.eigh(corr_w)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        eigenvalues = np.clip(eigenvalues, 0, None)

        total_var = eigenvalues.sum()
        if total_var < self.epsilon:
            return np.ones(p) / p

        explained_ratio = eigenvalues / total_var
        cumulative = np.cumsum(explained_ratio)

        n_retained = max(1, int(np.searchsorted(
            cumulative, self.pca_variance_threshold) + 1))
        n_retained = min(n_retained, len(eigenvalues))

        retained_ev = eigenvalues[:n_retained]
        proportion = retained_ev / (retained_ev.sum() + self.epsilon)

        components = eigenvectors[:, :n_retained].T  # (K, p)
        raw_weights = np.zeros(p)
        for k in range(n_retained):
            raw_weights += proportion[k] * (components[k, :] ** 2)

        raw_weights = np.clip(raw_weights, self.epsilon, None)
        weights = raw_weights / raw_weights.sum()
        return weights

    # =====================================================================
    # HELPER: Single-pass fused weights (for stability verification)
    # =====================================================================

    def _compute_fused_weights(self, X_norm: np.ndarray) -> np.ndarray:
        """Compute fused weights from a normalized matrix (no bootstrap)."""
        pca_info = self._pca_residualize(X_norm)
        R = pca_info["residual"]
        W_c = self._critic_weights(X_norm, R)
        W_e = self._entropy_weights(X_norm)
        W_p = self._pca_weights(X_norm)
        return self._kl_divergence_fusion(W_e, W_c, W_p)
