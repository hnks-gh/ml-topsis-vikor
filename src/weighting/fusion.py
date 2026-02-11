# -*- coding: utf-8 -*-
"""
Advanced weight fusion methods for combining multiple objective weighting methods.

Implements: Reliability-Weighted, Min Cross-Entropy, Bayesian Model Averaging, OWA.
See technical documentation for detailed formulas and references.
"""

import numpy as np
from typing import Dict, List, Optional
from scipy.optimize import minimize
from scipy.stats import entropy as scipy_entropy


class AdvancedWeightFusion:
    """Advanced fusion methods for combining weight vectors."""
    
    def __init__(self, epsilon: float = 1e-10):
        self.epsilon = epsilon
    
    def reliability_weighted_fusion(
        self,
        weight_vectors: Dict[str, np.ndarray],
        bootstrap_results: Optional[Dict[str, Dict]] = None
    ) -> tuple:
        """
        Adaptive fusion weighted by method reliability (decisiveness, stability, informativeness).
        
        Returns (fused_weights, reliability_scores, details_dict).
        """
        n_criteria = len(next(iter(weight_vectors.values())))
        n_methods = len(weight_vectors)
        
        # Calculate reliability scores for each method
        reliability = {}
        
        for method, weights in weight_vectors.items():
            # Ensure weights are valid probability distribution
            w_safe = np.clip(weights, self.epsilon, None)
            w_norm = w_safe / w_safe.sum()
            
            # Entropy-based confidence (lower entropy = more decisive)
            H = scipy_entropy(w_norm) / np.log(n_criteria)
            entropy_confidence = 1.0 - H
            
            # Stability score (if bootstrap available)
            if bootstrap_results and method in bootstrap_results:
                std_weights = bootstrap_results[method].get('std', np.zeros(n_criteria))
                cv = np.mean(std_weights / (w_norm + self.epsilon))
                stability_score = np.exp(-cv)
            else:
                stability_score = 1.0
            
            # Uniformity penalty
            max_entropy = -np.log(1.0 / n_criteria)
            current_entropy = scipy_entropy(w_norm)
            uniformity_penalty = 1.0 - np.exp(-(max_entropy - current_entropy))
            
            # Combined reliability
            reliability[method] = (
                0.5 * entropy_confidence + 0.3 * stability_score + 0.2 * uniformity_penalty
            )
        
        # Normalize reliability scores to get method weights
        total_reliability = sum(reliability.values())
        alpha = {m: r / total_reliability for m, r in reliability.items()}
        
        # Fuse weights using reliability-weighted combination
        fused = np.zeros(n_criteria)
        for method, weights in weight_vectors.items():
            fused += alpha[method] * weights
        
        # Normalize
        fused = np.clip(fused, self.epsilon, None)
        fused = fused / fused.sum()
        
        details = {
            "method": "reliability_weighted_fusion",
            "reliability_scores": reliability,
            "fusion_alphas": alpha,
            "entropy_values": {
                method: float(scipy_entropy(np.clip(w, self.epsilon, None)) / np.log(n_criteria))
                for method, w in weight_vectors.items()
            },
        }
        
        return fused, reliability, details
    
    def minimum_crossentropy_fusion(
        self,
        weight_vectors: Dict[str, np.ndarray],
        prior: Optional[np.ndarray] = None,
        constraints: Optional[Dict] = None
    ) -> tuple:
        """
        Info-theoretically optimal fusion minimizing KL-divergence from prior.
        
        Returns (fused_weights, converged, details_dict).
        """
        # Default prior: uniform
        if prior is None:
            prior = np.ones(n_criteria) / n_criteria
        
        # Objective: KL-divergence
        def objective(w):
            w_safe = np.clip(w, self.epsilon, None)
            return np.sum(w_safe * np.log(w_safe / (prior + self.epsilon)))
        
        # Constraints
        cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
        
        if constraints and 'preserve_means' in constraints:
            for method, weights_m in weight_vectors.items():
                target_mean = np.mean(weights_m)
                cons.append({'type': 'eq', 'fun': lambda w, t=target_mean: np.mean(w) - t})
        
        bounds = [(self.epsilon, 1.0) for _ in range(n_criteria)]
        
        # Initial guess: geometric mean
        w0 = np.ones(n_criteria)
        for weights in weight_vectors.values():
            w_safe = np.clip(weights, self.epsilon, None)
            w0 *= w_safe ** (1.0 / len(weight_vectors))
        w0 = w0 / w0.sum()
        
        # Optimize
        result = minimize(
            objective,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )
        
        fused = result.x
        fused = np.clip(fused, self.epsilon, None)
        fused = fused / fused.sum()
        
        details = {
            "method": "minimum_crossentropy_fusion",
            "converged": result.success,
            "kl_divergence": result.fun,
            "iterations": result.nit,
        }
        
        return fused, result.success, details
    
    def bayesian_model_averaging(
        self,
        weight_vectors: Dict[str, np.ndarray],
        performance_metrics: Dict[str, float]
    ) -> tuple:
        """
        BMA: weight methods by predictive performance (lower IC = better).
        
        Returns (fused_weights, posterior_probs, details_dict).
        """
        n_criteria = len(next(iter(weight_vectors.values())))
        
        # Calculate posterior probabilities from performance metrics
        # Using exp(-IC/2) as likelihood (common in model averaging)
        log_likelihoods = {
            method: -metric / 2.0 
            for method, metric in performance_metrics.items()
        }
        
        # Convert to probabilities (softmax)
        max_ll = max(log_likelihoods.values())
        exp_ll = {m: np.exp(ll - max_ll) for m, ll in log_likelihoods.items()}
        total = sum(exp_ll.values())
        posterior = {m: exp_ll[m] / total for m in exp_ll}
        
        # BMA fusion
        fused = np.zeros(n_criteria)
        for method, weights in weight_vectors.items():
            if method in posterior:
                fused += posterior[method] * weights
        
        # Normalize
        fused = np.clip(fused, self.epsilon, None)
        fused = fused / fused.sum()
        
        details = {
            "method": "bayesian_model_averaging",
            "posterior_probabilities": posterior,
            "performance_metrics": performance_metrics,
        }
        
        return fused, posterior, details
    
    def ordered_weighted_averaging(
        self,
        weight_vectors: Dict[str, np.ndarray],
        orness: float = 0.5
    ) -> tuple:
        """
        OWA aggregation: sorts values before weighting (robust to outliers).
        orness: 1=max, 0.5=average, 0=min.
        
        Returns (fused_weights, owa_weights, details_dict).
        """
        n_criteria = len(next(iter(weight_vectors.values())))
        n_methods = len(weight_vectors)
        
        # Generate OWA weights
        if orness == 0.5:
            owa_weights = np.ones(n_methods) / n_methods
        else:
            alpha = (1.0 - orness) / orness if orness > 0 else 10.0
            ranks = np.arange(1, n_methods + 1)
            w = np.exp(-alpha * (ranks - 1) / (n_methods - 1))
            owa_weights = w / w.sum()
        
        # Apply OWA
        for j in range(n_criteria):
            values = np.array([weight_vectors[m][j] for m in weight_vectors.keys()])
            sorted_values = np.sort(values)[::-1]
            fused[j] = np.dot(owa_weights, sorted_values)
        
        # Normalize
        fused = np.clip(fused, self.epsilon, None)
        fused = fused / fused.sum()
        
        details = {
            "method": "ordered_weighted_averaging",
            "orness": orness,
            "owa_weights": owa_weights.tolist(),
        }
        
        return fused, owa_weights, details
