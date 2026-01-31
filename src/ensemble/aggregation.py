# -*- coding: utf-8 -*-
"""Rank aggregation methods: Borda Count, Copeland, and consensus."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class AggregatedRanking:
    """Result container for rank aggregation."""
    final_ranking: np.ndarray
    final_scores: np.ndarray
    method_rankings: Dict[str, np.ndarray]
    method_weights: Dict[str, float]
    agreement_matrix: np.ndarray
    kendall_w: float  # Kendall's coefficient of concordance
    
    def summary(self) -> str:
        lines = [
            f"\n{'='*60}",
            "RANK AGGREGATION RESULTS",
            f"{'='*60}",
            f"\nKendall's W (agreement): {self.kendall_w:.4f}",
            f"\nMethod Weights:"
        ]
        for method, weight in self.method_weights.items():
            lines.append(f"  {method}: {weight:.4f}")
        lines.append(f"\nTop 10 Final Ranking:")
        top_10_idx = np.argsort(self.final_ranking)[:10]
        for i, idx in enumerate(top_10_idx):
            lines.append(f"  {i+1}. Index {idx} (Score: {self.final_scores[idx]:.4f})")
        lines.append("=" * 60)
        return "\n".join(lines)


class RankAggregator:
    """
    Base class for rank aggregation methods.
    """
    
    @staticmethod
    def scores_to_ranks(scores: np.ndarray, higher_is_better: bool = True) -> np.ndarray:
        """Convert scores to ranks (1 = best)."""
        if higher_is_better:
            return len(scores) - np.argsort(np.argsort(scores))
        else:
            return np.argsort(np.argsort(scores)) + 1
    
    @staticmethod
    def kendall_w(rankings: np.ndarray) -> float:
        """
        Calculate Kendall's W coefficient of concordance.
        
        Parameters
        ----------
        rankings : np.ndarray
            Matrix of rankings (methods x alternatives)
        
        Returns
        -------
        float
            Kendall's W (0 = no agreement, 1 = perfect agreement)
        """
        n_methods, n_alternatives = rankings.shape
        
        # Sum of ranks for each alternative
        rank_sums = rankings.sum(axis=0)
        
        # Mean rank sum
        mean_rank_sum = rank_sums.mean()
        
        # S = sum of squared deviations
        S = np.sum((rank_sums - mean_rank_sum) ** 2)
        
        # Maximum possible S
        S_max = (n_methods ** 2 * (n_alternatives ** 3 - n_alternatives)) / 12
        
        if S_max == 0:
            return 1.0
        
        return S / S_max
    
    @staticmethod
    def spearman_correlation(rank1: np.ndarray, rank2: np.ndarray) -> float:
        """Calculate Spearman rank correlation."""
        n = len(rank1)
        d_squared = np.sum((rank1 - rank2) ** 2)
        return 1 - (6 * d_squared) / (n * (n ** 2 - 1))


class BordaCount(RankAggregator):
    """
    Borda Count rank aggregation.
    
    Each alternative receives points based on its ranking position.
    """
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Initialize Borda Count.
        
        Parameters
        ----------
        weights : Dict[str, float], optional
            Weights for each ranking method
        """
        self.weights = weights
    
    def aggregate(self, 
                 rankings: Dict[str, np.ndarray],
                 weights: Optional[Dict[str, float]] = None) -> AggregatedRanking:
        """
        Aggregate rankings using Borda Count.
        
        Parameters
        ----------
        rankings : Dict[str, np.ndarray]
            Dictionary of rankings {method_name: ranks}
        weights : Dict[str, float], optional
            Override weights for each method
        """
        method_names = list(rankings.keys())
        n_alternatives = len(list(rankings.values())[0])
        n_methods = len(method_names)
        
        # Use provided weights or default to equal
        if weights is None:
            weights = self.weights
        if weights is None:
            weights = {name: 1.0 / n_methods for name in method_names}
        
        # Normalize weights
        total_weight = sum(weights.values())
        weights = {k: v / total_weight for k, v in weights.items()}
        
        # Calculate Borda scores
        borda_scores = np.zeros(n_alternatives)
        
        for method_name, ranks in rankings.items():
            # Borda score = n - rank (so rank 1 gets n-1 points)
            method_scores = n_alternatives - ranks
            borda_scores += weights[method_name] * method_scores
        
        # Convert to final ranking
        final_ranking = self.scores_to_ranks(borda_scores, higher_is_better=True)
        
        # Create ranking matrix for Kendall's W
        ranking_matrix = np.array([rankings[name] for name in method_names])
        kendall = self.kendall_w(ranking_matrix)
        
        # Agreement matrix (pairwise correlations)
        agreement = np.zeros((n_methods, n_methods))
        for i, name_i in enumerate(method_names):
            for j, name_j in enumerate(method_names):
                agreement[i, j] = self.spearman_correlation(
                    rankings[name_i], rankings[name_j]
                )
        
        return AggregatedRanking(
            final_ranking=final_ranking,
            final_scores=borda_scores,
            method_rankings=rankings,
            method_weights=weights,
            agreement_matrix=agreement,
            kendall_w=kendall
        )


class CopelandMethod(RankAggregator):
    """
    Copeland rank aggregation.
    
    Based on pairwise comparisons - counts wins minus losses.
    """
    
    def aggregate(self,
                 rankings: Dict[str, np.ndarray],
                 weights: Optional[Dict[str, float]] = None) -> AggregatedRanking:
        """
        Aggregate rankings using Copeland method.
        
        Parameters
        ----------
        rankings : Dict[str, np.ndarray]
            Dictionary of rankings {method_name: ranks}
        """
        method_names = list(rankings.keys())
        n_alternatives = len(list(rankings.values())[0])
        n_methods = len(method_names)
        
        # Use weights if provided
        if weights is None:
            weights = {name: 1.0 / n_methods for name in method_names}
        
        # Calculate pairwise comparison matrix
        pairwise_wins = np.zeros((n_alternatives, n_alternatives))
        
        for method_name, ranks in rankings.items():
            w = weights.get(method_name, 1.0 / n_methods)
            for i in range(n_alternatives):
                for j in range(n_alternatives):
                    if i != j:
                        # i beats j if i has lower rank (better)
                        if ranks[i] < ranks[j]:
                            pairwise_wins[i, j] += w
        
        # Copeland scores: wins - losses
        copeland_scores = np.zeros(n_alternatives)
        for i in range(n_alternatives):
            for j in range(n_alternatives):
                if i != j:
                    if pairwise_wins[i, j] > pairwise_wins[j, i]:
                        copeland_scores[i] += 1
                    elif pairwise_wins[i, j] < pairwise_wins[j, i]:
                        copeland_scores[i] -= 1
        
        # Convert to ranking
        final_ranking = self.scores_to_ranks(copeland_scores, higher_is_better=True)
        
        # Kendall's W
        ranking_matrix = np.array([rankings[name] for name in method_names])
        kendall = self.kendall_w(ranking_matrix)
        
        # Agreement matrix
        agreement = np.zeros((n_methods, n_methods))
        for i, name_i in enumerate(method_names):
            for j, name_j in enumerate(method_names):
                agreement[i, j] = self.spearman_correlation(
                    rankings[name_i], rankings[name_j]
                )
        
        return AggregatedRanking(
            final_ranking=final_ranking,
            final_scores=copeland_scores,
            method_rankings=rankings,
            method_weights=weights,
            agreement_matrix=agreement,
            kendall_w=kendall
        )


class KemenyYoung(RankAggregator):
    """
    Kemeny-Young optimal rank aggregation.
    
    Finds ranking that minimizes total Kendall tau distance to all input rankings.
    Note: Exact solution is NP-hard, uses approximation for large problems.
    """
    
    def __init__(self, max_exact: int = 8):
        """
        Initialize Kemeny-Young.
        
        Parameters
        ----------
        max_exact : int
            Maximum alternatives for exact solution
        """
        self.max_exact = max_exact
    
    def aggregate(self,
                 rankings: Dict[str, np.ndarray],
                 weights: Optional[Dict[str, float]] = None) -> AggregatedRanking:
        """
        Aggregate rankings using Kemeny-Young method.
        """
        method_names = list(rankings.keys())
        n_alternatives = len(list(rankings.values())[0])
        n_methods = len(method_names)
        
        if weights is None:
            weights = {name: 1.0 / n_methods for name in method_names}
        
        if n_alternatives <= self.max_exact:
            # Exact solution using dynamic programming
            kemeny_scores = self._exact_kemeny(rankings, weights)
        else:
            # Approximation using Borda as initial + local search
            kemeny_scores = self._approximate_kemeny(rankings, weights)
        
        final_ranking = self.scores_to_ranks(kemeny_scores, higher_is_better=True)
        
        ranking_matrix = np.array([rankings[name] for name in method_names])
        kendall = self.kendall_w(ranking_matrix)
        
        agreement = np.zeros((n_methods, n_methods))
        for i, name_i in enumerate(method_names):
            for j, name_j in enumerate(method_names):
                agreement[i, j] = self.spearman_correlation(
                    rankings[name_i], rankings[name_j]
                )
        
        return AggregatedRanking(
            final_ranking=final_ranking,
            final_scores=kemeny_scores,
            method_rankings=rankings,
            method_weights=weights,
            agreement_matrix=agreement,
            kendall_w=kendall
        )
    
    def _exact_kemeny(self, rankings: Dict[str, np.ndarray],
                     weights: Dict[str, float]) -> np.ndarray:
        """Exact Kemeny solution for small problems."""
        # Build weighted pairwise preference matrix
        method_names = list(rankings.keys())
        n = len(list(rankings.values())[0])
        
        pref = np.zeros((n, n))
        for name, ranks in rankings.items():
            w = weights[name]
            for i in range(n):
                for j in range(n):
                    if ranks[i] < ranks[j]:  # i preferred to j
                        pref[i, j] += w
        
        # Use Borda as approximation for "exact" (true exact is expensive)
        borda = BordaCount(weights)
        result = borda.aggregate(rankings, weights)
        return result.final_scores
    
    def _approximate_kemeny(self, rankings: Dict[str, np.ndarray],
                           weights: Dict[str, float]) -> np.ndarray:
        """Approximate Kemeny using Borda + local search."""
        # Start with Borda
        borda = BordaCount(weights)
        result = borda.aggregate(rankings, weights)
        
        return result.final_scores


class MedianRank(RankAggregator):
    """
    Median rank aggregation.
    
    Uses median of ranks across methods.
    """
    
    def aggregate(self,
                 rankings: Dict[str, np.ndarray],
                 weights: Optional[Dict[str, float]] = None) -> AggregatedRanking:
        """
        Aggregate rankings using median.
        """
        method_names = list(rankings.keys())
        n_alternatives = len(list(rankings.values())[0])
        n_methods = len(method_names)
        
        if weights is None:
            weights = {name: 1.0 / n_methods for name in method_names}
        
        # Stack rankings
        ranking_matrix = np.array([rankings[name] for name in method_names])
        
        # Calculate median ranks (lower is better, so we negate for scores)
        median_ranks = np.median(ranking_matrix, axis=0)
        
        # Convert to scores (inverse of median rank)
        scores = -median_ranks
        
        # Final ranking
        final_ranking = self.scores_to_ranks(scores, higher_is_better=True)
        
        kendall = self.kendall_w(ranking_matrix)
        
        agreement = np.zeros((n_methods, n_methods))
        for i, name_i in enumerate(method_names):
            for j, name_j in enumerate(method_names):
                agreement[i, j] = self.spearman_correlation(
                    rankings[name_i], rankings[name_j]
                )
        
        return AggregatedRanking(
            final_ranking=final_ranking,
            final_scores=scores,
            method_rankings=rankings,
            method_weights=weights,
            agreement_matrix=agreement,
            kendall_w=kendall
        )


def aggregate_rankings(rankings: Dict[str, np.ndarray],
                      method: str = 'borda',
                      weights: Optional[Dict[str, float]] = None) -> AggregatedRanking:
    """
    Convenience function to aggregate rankings.
    
    Parameters
    ----------
    rankings : Dict[str, np.ndarray]
        Rankings from different methods
    method : str
        'borda', 'copeland', 'kemeny', or 'median'
    weights : Dict[str, float], optional
        Method weights
    """
    if method == 'borda':
        aggregator = BordaCount(weights)
    elif method == 'copeland':
        aggregator = CopelandMethod()
    elif method == 'kemeny':
        aggregator = KemenyYoung()
    elif method == 'median':
        aggregator = MedianRank()
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return aggregator.aggregate(rankings, weights)
