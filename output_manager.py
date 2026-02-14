# -*- coding: utf-8 -*-
"""
Output Management for ML-MCDM Analysis Results
================================================

Provides the ``OutputManager`` class for persisting analysis artefacts
(CSV, JSON, text report) into an organised directory structure::

    outputs/
    ├── results/   — numerical data  (CSV, JSON)
    ├── figures/   — visualisation charts  (PNG)
    └── reports/   — comprehensive text report

This module is *not* called by the pipeline directly; the pipeline does
its own inline saving via ``_save_all_results``.  ``OutputManager`` is
kept available for advanced / ad-hoc usage and backward compatibility.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import json


# =========================================================================
# Utility
# =========================================================================

def to_array(x: Any) -> np.ndarray:
    """Convert Series / list / scalar to plain ndarray."""
    if x is None:
        return np.array([])
    if hasattr(x, 'values'):
        return x.values
    return np.asarray(x) if not isinstance(x, np.ndarray) else x


# =========================================================================
# OutputManager
# =========================================================================

class OutputManager:
    """
    Manages structured output to ``results/``, ``figures/``, ``reports/``.

    Updated for the IFS + Evidential Reasoning architecture.
    """

    def __init__(self, base_output_dir: str = 'outputs'):
        self.base_dir = Path(base_output_dir)
        self.results_dir = self.base_dir / 'results'
        self.figures_dir = self.base_dir / 'figures'
        self.reports_dir = self.base_dir / 'reports'
        self._setup_directories()
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    def _setup_directories(self) -> None:
        for d in [self.results_dir, self.figures_dir, self.reports_dir]:
            d.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------
    # Weight export
    # -----------------------------------------------------------------

    def save_weights(
        self,
        weights: Dict[str, np.ndarray],
        subcriteria_names: List[str],
    ) -> str:
        """Save subcriteria weights from all four methods + fused."""
        df = pd.DataFrame({'Subcriteria': subcriteria_names})
        for method, w in weights.items():
            if isinstance(w, np.ndarray) and len(w) == len(subcriteria_names):
                df[method.title()] = w
        df = df.sort_values(
            df.columns[-1], ascending=False
        ).reset_index(drop=True)
        path = self.results_dir / 'weights_analysis.csv'
        df.to_csv(path, index=False, float_format='%.6f')
        return str(path)

    # -----------------------------------------------------------------
    # Ranking export
    # -----------------------------------------------------------------

    def save_rankings(
        self,
        ranking_result: Any,
        provinces: List[str],
    ) -> str:
        """Save final ER rankings to CSV."""
        df = pd.DataFrame({
            'Province': ranking_result.final_ranking.index,
            'ER_Score': ranking_result.final_scores.values,
            'ER_Rank': ranking_result.final_ranking.values,
        }).sort_values('ER_Rank').reset_index(drop=True)
        df.insert(0, 'Rank', range(1, len(df) + 1))
        path = self.results_dir / 'final_rankings.csv'
        df.to_csv(path, index=False, float_format='%.6f')
        return str(path)

    # -----------------------------------------------------------------
    # MCDM scores per criterion
    # -----------------------------------------------------------------

    def save_mcdm_scores_by_criterion(
        self,
        ranking_result: Any,
        provinces: List[str],
    ) -> Dict[str, str]:
        """Save per-criterion MCDM method scores."""
        saved = {}
        for crit_id, method_scores in ranking_result.criterion_method_scores.items():
            df = pd.DataFrame(method_scores)
            df.index = provinces
            df.index.name = 'Province'
            path = self.results_dir / f'mcdm_scores_{crit_id}.csv'
            df.to_csv(path, float_format='%.6f')
            saved[crit_id] = str(path)
        return saved

    # -----------------------------------------------------------------
    # ML results
    # -----------------------------------------------------------------

    def save_ml_results(
        self,
        ml_results: Dict[str, Any],
    ) -> Dict[str, str]:
        saved: Dict[str, str] = {}
        if ml_results.get('rf_importance'):
            imp_df = pd.DataFrame([
                {'Feature': k, 'Importance': v}
                for k, v in sorted(
                    ml_results['rf_importance'].items(),
                    key=lambda x: x[1], reverse=True,
                )
            ])
            path = self.results_dir / 'feature_importance.csv'
            imp_df.to_csv(path, index=False, float_format='%.6f')
            saved['feature_importance'] = str(path)
        return saved

    # -----------------------------------------------------------------
    # Analysis results
    # -----------------------------------------------------------------

    def save_analysis_results(
        self,
        analysis_results: Dict[str, Any],
    ) -> Dict[str, str]:
        saved: Dict[str, str] = {}
        sens = analysis_results.get('sensitivity')
        if sens and hasattr(sens, 'weight_sensitivity'):
            df = pd.DataFrame([
                {'Criterion': k, 'Sensitivity': v}
                for k, v in sorted(
                    sens.weight_sensitivity.items(),
                    key=lambda x: x[1], reverse=True,
                )
            ])
            path = self.results_dir / 'sensitivity_analysis.csv'
            df.to_csv(path, index=False, float_format='%.6f')
            saved['sensitivity'] = str(path)

            robust_df = pd.DataFrame([{
                'Robustness': sens.overall_robustness,
            }])
            path = self.results_dir / 'robustness_summary.csv'
            robust_df.to_csv(path, index=False, float_format='%.6f')
            saved['robustness'] = str(path)
        return saved

    # -----------------------------------------------------------------
    # Execution summary
    # -----------------------------------------------------------------

    def save_execution_summary(
        self,
        execution_time: float,
    ) -> str:
        summary = {
            'timestamp': datetime.now().isoformat(),
            'execution_time_seconds': round(execution_time, 2),
        }
        path = self.results_dir / 'execution_summary.json'
        with open(path, 'w') as f:
            json.dump(summary, f, indent=2)
        return str(path)

    # -----------------------------------------------------------------
    # Config snapshot
    # -----------------------------------------------------------------

    def save_config_snapshot(self, config: Any) -> str:
        path = self.results_dir / 'config_snapshot.json'
        config.save(path)
        return str(path)


# =========================================================================
# Factory
# =========================================================================

def create_output_manager(output_dir: str = 'outputs') -> OutputManager:
    """Factory function to create an OutputManager."""
    return OutputManager(output_dir)
