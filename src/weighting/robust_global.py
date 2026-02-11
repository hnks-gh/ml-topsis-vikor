# -*- coding: utf-8 -*-
"""
Robust Global Hybrid Weighting for panel MCDM data.

Combines four objective methods (Entropy, CRITIC, MEREC, SD) via reliability-weighted
fusion, validated through Bayesian Bootstrap.

See technical documentation for detailed methodology and references.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from scipy import stats
import logging

from .base import WeightResult
from .entropy import EntropyWeightCalculator
from .critic import CRITICWeightCalculator
from .merec import MERECWeightCalculator
from .standard_deviation import StandardDeviationWeightCalculator
from .fusion import AdvancedWeightFusion

logger = logging.getLogger(__name__)


class RobustGlobalWeighting:
    """
    Combines Entropy, CRITIC, MEREC, and SD weights via adaptive fusion.
    
    Parameters
    ----------
    bootstrap_iterations : int, default=999
        Number of Bayesian Bootstrap iterations.
    stability_threshold : float, default=0.95
        Minimum cosine similarity for split-half stability.
    epsilon : float, default=1e-10
        Numerical stability constant.
    seed : int, default=42
        Random seed for reproducibility.
    """

    def __init__(
        self,
        bootstrap_iterations: int = 999,
        stability_threshold: float = 0.95,
        epsilon: float = 1e-10,
        seed: int = 42,
    ):
        self.bootstrap_iterations = bootstrap_iterations
        self.stability_threshold = stability_threshold
        self.epsilon = epsilon
        self.seed = seed
        
        # Initialize calculators
        self.entropy_calc = EntropyWeightCalculator(epsilon=epsilon)
        self.critic_calc = CRITICWeightCalculator(epsilon=epsilon)
        self.merec_calc = MERECWeightCalculator(epsilon=epsilon)
        self.sd_calc = StandardDeviationWeightCalculator(epsilon=epsilon)
        self.fusion = AdvancedWeightFusion(epsilon=epsilon)

    def calculate(
        self,
        panel_df: pd.DataFrame,
        entity_col: str = "Province",
        time_col: str = "Year",
        criteria_cols: Optional[List[str]] = None,
    ) -> WeightResult:
        """
        Execute weighting pipeline on panel data.
        
        Returns WeightResult with final weights and comprehensive statistics.
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

        # ── Step 2: Compute Individual Objective Weights ──
        X_df = pd.DataFrame(X_norm, columns=criteria_cols)
        
        # 2a. Entropy weights
        entropy_result = self.entropy_calc.calculate(X_df)
        W_e = np.array([entropy_result.weights[c] for c in criteria_cols])
        logger.info(f"Step 2a: Entropy weights — range [{W_e.min():.4f}, {W_e.max():.4f}]")
        
        # 2b. CRITIC weights
        critic_result = self.critic_calc.calculate(X_df)
        W_c = np.array([critic_result.weights[c] for c in criteria_cols])
        logger.info(f"Step 2b: CRITIC weights — range [{W_c.min():.4f}, {W_c.max():.4f}]")
        
        # 2c. MEREC weights
        merec_result = self.merec_calc.calculate(X_df)
        W_m = np.array([merec_result.weights[c] for c in criteria_cols])
        logger.info(f"Step 2c: MEREC weights — range [{W_m.min():.4f}, {W_m.max():.4f}]")
        
        # 2d. Standard Deviation weights
        sd_result = self.sd_calc.calculate(X_df)
        W_s = np.array([sd_result.weights[c] for c in criteria_cols])
        logger.info(f"Step 2d: Std Dev weights — range [{W_s.min():.4f}, {W_s.max():.4f}]")

        # ── Step 3: Reliability-Weighted Adaptive Fusion ──
        weight_vectors = {
            'entropy': W_e,
            'critic': W_c,
            'merec': W_m,
            'std_dev': W_s
        }
        W_fused, reliability_scores, fusion_details = self.fusion.reliability_weighted_fusion(
            weight_vectors
        )
        logger.info(f"Step 3: Reliability-weighted fusion — range [{W_fused.min():.4f}, "
                    f"{W_fused.max():.4f}]")
        logger.info(f"  Method reliabilities: {', '.join([f'{k}={v:.3f}' for k, v in reliability_scores.items()])}")

        # ── Step 4: Bayesian Bootstrap Validation ──
        bootstrap_results = self._bayesian_bootstrap(X_norm, criteria_cols)
        W_final = bootstrap_results["mean_weights"]
        logger.info(f"Step 4: Bayesian Bootstrap ({self.bootstrap_iterations} "
                    f"iterations) — mean weight std: "
                    f"{bootstrap_results['std_weights'].mean():.6f}")

        # ── Stability Verification ──
        time_values = panel_df[time_col].values
        stability = self._stability_verification(X_raw, time_values, criteria_cols)
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
                "merec": {col: float(W_m[j])
                          for j, col in enumerate(criteria_cols)},
                "std_dev": {col: float(W_s[j])
                            for j, col in enumerate(criteria_cols)},
                "fused": {col: float(W_fused[j])
                          for j, col in enumerate(criteria_cols)},
            },
            # Fusion details
            "fusion": fusion_details,
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
        """Global min-max normalization with epsilon shift."""
        col_min = X.min(axis=0)
        col_max = X.max(axis=0)
        denom = col_max - col_min
        denom[denom < self.epsilon] = self.epsilon

        X_norm = (X - col_min) / denom + self.epsilon
        return X_norm

    # =====================================================================
    # BAYESIAN BOOTSTRAP
    # =====================================================================

    def _bayesian_bootstrap(self, X_norm: np.ndarray, criteria_cols: List[str]) -> Dict:
        """Bayesian Bootstrap with Dirichlet resampling."""
        N, p = X_norm.shape
        B = self.bootstrap_iterations
        rng = np.random.RandomState(self.seed)

        all_weights = np.zeros((B, p))

        for b in range(B):
            # Draw Dirichlet(1,...,1) weights
            g = rng.exponential(1.0, size=N)
            obs_weights = g / g.sum()

            try:
                # Resample according to Dirichlet weights
                X_df = pd.DataFrame(X_norm, columns=criteria_cols)
                indices = rng.choice(N, size=N, replace=True, p=obs_weights)
                X_boot = X_df.iloc[indices].reset_index(drop=True)
                
                # Compute weights on bootstrap sample
                W_e = np.array([self.entropy_calc.calculate(X_boot).weights[c] 
                               for c in criteria_cols])
                W_c = np.array([self.critic_calc.calculate(X_boot).weights[c] 
                               for c in criteria_cols])
                W_m = np.array([self.merec_calc.calculate(X_boot).weights[c] 
                               for c in criteria_cols])
                W_s = np.array([self.sd_calc.calculate(X_boot).weights[c] 
                               for c in criteria_cols])
                
                # Fuse
                weight_vectors = {
                    'entropy': W_e, 'critic': W_c, 
                    'merec': W_m, 'std_dev': W_s
                }
                W_fused, _, _ = self.fusion.reliability_weighted_fusion(weight_vectors)
                all_weights[b, :] = W_fused
            except Exception:
                # Fallback to previous iteration on failure
                all_weights[b, :] = all_weights[max(0, b-1), :]

        # Posterior statistics
        mean_weights = all_weights.mean(axis=0)
        mean_weights /= mean_weights.sum()

        std_weights = all_weights.std(axis=0, ddof=1)

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
        criteria_cols: List[str]
    ) -> Dict:
        """Split-half temporal stability verification."""
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

        # Compute weights on each half
        X_first = X_raw[mask_first]
        X_second = X_raw[mask_second]

        X_first_norm = self._global_min_max_normalize(X_first)
        X_second_norm = self._global_min_max_normalize(X_second)

        W_first = self._compute_fused_weights(X_first_norm, criteria_cols)
        W_second = self._compute_fused_weights(X_second_norm, criteria_cols)

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

    def _compute_fused_weights(self, X_norm: np.ndarray, criteria_cols: List[str]) -> np.ndarray:
        """Compute fused weights (helper for stability check)."""
        X_df = pd.DataFrame(X_norm, columns=criteria_cols)
        
        W_e = np.array([self.entropy_calc.calculate(X_df).weights[c] for c in criteria_cols])
        W_c = np.array([self.critic_calc.calculate(X_df).weights[c] for c in criteria_cols])
        W_m = np.array([self.merec_calc.calculate(X_df).weights[c] for c in criteria_cols])
        W_s = np.array([self.sd_calc.calculate(X_df).weights[c] for c in criteria_cols])
        
        weight_vectors = {'entropy': W_e, 'critic': W_c, 'merec': W_m, 'std_dev': W_s}
        W_fused, _, _ = self.fusion.reliability_weighted_fusion(weight_vectors)
        
        return W_fused
