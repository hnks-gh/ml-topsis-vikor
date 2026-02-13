# -*- coding: utf-8 -*-
"""
ML-MCDM Pipeline Orchestrator
==============================

Eight-phase production pipeline:

  Phase 1  Data Loading
  Phase 2  Weight Calculation       (GTWC + Bayesian Bootstrap)
  Phase 3  Hierarchical Ranking     (12 MCDM + two-stage ER)
  Phase 4  ML Feature Importance    (Random Forest, optional)
  Phase 5  Sensitivity Analysis     (Monte Carlo weight perturbation)
  Phase 6  Visualisation            (high-resolution figures)
  Phase 7  Result Export            (CSV / JSON / text report)

Note: ML forecasting (Phase 6 old) is **isolated** from the workflow
and will not execute unless explicitly re-enabled in a future release.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import warnings
import time
import json

warnings.filterwarnings('ignore')

# Internal imports
from .config import Config, get_default_config
from .logger import setup_logger, ProgressLogger
from .data_loader import DataLoader, PanelData

from .mcdm.traditional import TOPSISCalculator
from .ranking import HierarchicalRankingPipeline, HierarchicalRankingResult
from .analysis import SensitivityAnalysis
from .visualization import PanelVisualizer


def _to_array(x) -> np.ndarray:
    """Convert Series / list / array to plain ndarray."""
    if x is None:
        return np.array([])
    if hasattr(x, 'values'):
        return x.values
    return np.asarray(x)


# =========================================================================
# Result container
# =========================================================================

@dataclass
class PipelineResult:
    """Container for all pipeline results."""
    # Data
    panel_data: PanelData
    decision_matrix: np.ndarray

    # Weights
    entropy_weights: np.ndarray
    critic_weights: np.ndarray
    merec_weights: np.ndarray
    std_dev_weights: np.ndarray
    fused_weights: np.ndarray
    weight_details: Dict[str, Any]

    # Hierarchical Ranking (IFS + ER)
    ranking_result: HierarchicalRankingResult

    # ML Results (optional — may be empty)
    rf_feature_importance: Dict[str, float]

    # Analysis
    sensitivity_result: Any

    # Meta
    execution_time: float = 0.0
    config: Optional[Config] = None

    # ---- convenience accessors ----

    def get_final_ranking_df(self) -> pd.DataFrame:
        """Sorted DataFrame with province, rank, score, Kendall's W."""
        return pd.DataFrame({
            'Province': self.ranking_result.final_ranking.index,
            'ER_Score': self.ranking_result.final_scores.values,
            'ER_Rank': self.ranking_result.final_ranking.values,
        }).sort_values('ER_Rank').reset_index(drop=True)


# =========================================================================
# Pipeline
# =========================================================================

class MLMCDMPipeline:
    """
    Production-grade ML-MCDM pipeline for panel data analysis.

    Integrates
    ----------
    * Robust Global Hybrid Weighting (Entropy + CRITIC + MEREC + SD)
    * 12 MCDM methods per criterion group:
        Traditional : TOPSIS, VIKOR, PROMETHEE, COPRAS, EDAS, SAW
        IFS         : IFS-TOPSIS, IFS-VIKOR, IFS-PROMETHEE, IFS-COPRAS,
                      IFS-EDAS, IFS-SAW
    * Two-stage Evidential Reasoning aggregation (Yang & Xu, 2002)
    * Random Forest feature importance (optional)
    * Monte Carlo sensitivity analysis
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config or get_default_config()

        # Output directories
        self._setup_output_directory()

        # Logger
        debug_file = Path(self.config.output_dir) / 'logs' / 'debug.log'
        self.logger = setup_logger('ml_mcdm', debug_file=debug_file)

        # Visualiser & output helpers
        self.visualizer = PanelVisualizer(
            output_dir=str(Path(self.config.output_dir) / 'figures'),
            dpi=self.config.visualization.dpi,
        )

    # -----------------------------------------------------------------
    # Setup
    # -----------------------------------------------------------------

    def _setup_output_directory(self) -> None:
        out = Path(self.config.output_dir)
        for sub in ('figures', 'results', 'reports', 'logs'):
            (out / sub).mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------
    # Main entry point
    # -----------------------------------------------------------------

    def run(self, data_path: Optional[str] = None) -> PipelineResult:
        """Execute full analysis pipeline and return results."""
        start_time = time.time()

        self.logger.info("=" * 60)
        self.logger.info("ML-MCDM PANEL DATA ANALYSIS PIPELINE")
        self.logger.info("IFS + Evidential Reasoning Architecture")
        self.logger.info("=" * 60)

        # Phase 1: Data Loading
        with ProgressLogger(self.logger, "Phase 1: Data Loading"):
            panel_data = self._load_data(data_path)

        # Phase 2: Weight Calculation
        with ProgressLogger(self.logger, "Phase 2: Weight Calculation (GTWC)"):
            weights = self._calculate_weights(panel_data)

        # Phase 3: Hierarchical Ranking (12 MCDM methods + ER)
        with ProgressLogger(self.logger, "Phase 3: Hierarchical Ranking (IFS + ER)"):
            ranking_result = self._run_hierarchical_ranking(panel_data, weights)

        # Phase 4: ML Feature Importance (optional)
        ml_results: Dict[str, Any] = {'rf_importance': {}}
        with ProgressLogger(self.logger, "Phase 4: ML Feature Importance"):
            try:
                ml_results = self._run_ml(panel_data, ranking_result)
            except Exception as e:
                self.logger.warning(f"ML analysis skipped: {e}")

        # Phase 5: Sensitivity Analysis
        analysis_results: Dict[str, Any] = {'sensitivity': None}
        with ProgressLogger(self.logger, "Phase 5: Sensitivity Analysis"):
            try:
                analysis_results = self._run_analysis(
                    panel_data, ranking_result, weights)
            except Exception as e:
                self.logger.warning(f"Sensitivity analysis skipped: {e}")

        execution_time = time.time() - start_time

        # Phase 6: Visualisations
        with ProgressLogger(self.logger, "Phase 6: Generating Figures"):
            try:
                self._generate_all_visualizations(
                    panel_data, weights, ranking_result,
                    analysis_results, ml_results=ml_results,
                )
            except Exception as e:
                self.logger.warning(f"Visualisation failed: {e}")

        # Phase 7: Save Results
        with ProgressLogger(self.logger, "Phase 7: Saving Results"):
            try:
                self._save_all_results(
                    panel_data, weights, ranking_result, ml_results,
                    analysis_results, execution_time,
                )
            except Exception as e:
                self.logger.warning(f"Result saving failed: {e}")
                import traceback
                self.logger.debug(traceback.format_exc())

        self.logger.info("=" * 60)
        self.logger.info(f"Pipeline completed in {execution_time:.2f}s")
        self.logger.info(f"Outputs → {self.config.output_dir}")
        self.logger.info("=" * 60)

        # Build result object
        subcriteria = weights['subcriteria']
        latest_year = max(panel_data.years)
        decision_matrix = panel_data.subcriteria_cross_section[latest_year][subcriteria].values

        return PipelineResult(
            panel_data=panel_data,
            decision_matrix=decision_matrix,
            entropy_weights=weights['entropy'],
            critic_weights=weights['critic'],
            merec_weights=weights['merec'],
            std_dev_weights=weights['std_dev'],
            fused_weights=weights['fused'],
            weight_details=weights.get('details', {}),
            ranking_result=ranking_result,
            rf_feature_importance=ml_results.get('rf_importance', {}),
            sensitivity_result=analysis_results.get('sensitivity'),
            execution_time=execution_time,
            config=self.config,
        )

    # -----------------------------------------------------------------
    # Phase 1: Data Loading
    # -----------------------------------------------------------------

    def _load_data(self, data_path: Optional[str]) -> PanelData:
        loader = DataLoader(self.config)
        panel_data = loader.load()
        self.logger.info(
            f"Loaded: {len(panel_data.provinces)} provinces, "
            f"{len(panel_data.years)} years, "
            f"{panel_data.n_subcriteria} subcriteria, "
            f"{panel_data.n_criteria} criteria"
        )
        return panel_data

    # -----------------------------------------------------------------
    # Phase 2: Weight Calculation
    # -----------------------------------------------------------------

    def _calculate_weights(self, panel_data: PanelData) -> Dict[str, Any]:
        """Run GTWC hybrid weighting on the full panel."""
        from .weighting import RobustGlobalWeighting

        subcriteria = panel_data.hierarchy.all_subcriteria
        panel_df = panel_data.subcriteria_long.copy()

        self.logger.info("GTWC weighting pipeline")
        self.logger.info(
            f"  Panel: {len(panel_df)} obs "
            f"({len(panel_data.years)} yr x "
            f"{len(panel_data.provinces)} prov x "
            f"{len(subcriteria)} sub)"
        )

        calc = RobustGlobalWeighting(
            bootstrap_iterations=self.config.weighting.bootstrap_iterations,
            stability_threshold=self.config.weighting.stability_threshold,
            epsilon=self.config.weighting.epsilon,
            seed=self.config.random.seed,
        )

        result = calc.calculate(
            panel_df,
            entity_col='Province',
            time_col='Year',
            criteria_cols=subcriteria,
        )

        # Extract individual method weights
        indiv = result.details["individual_weights"]
        entropy_w = np.array([indiv["entropy"][c] for c in subcriteria])
        critic_w = np.array([indiv["critic"][c] for c in subcriteria])
        merec_w = np.array([indiv["merec"][c] for c in subcriteria])
        stddev_w = np.array([indiv["std_dev"][c] for c in subcriteria])
        fused_w = np.array([result.weights[c] for c in subcriteria])

        self.logger.info(f"  Entropy  : [{entropy_w.min():.4f}, {entropy_w.max():.4f}]")
        self.logger.info(f"  CRITIC   : [{critic_w.min():.4f}, {critic_w.max():.4f}]")
        self.logger.info(f"  MEREC    : [{merec_w.min():.4f}, {merec_w.max():.4f}]")
        self.logger.info(f"  StdDev   : [{stddev_w.min():.4f}, {stddev_w.max():.4f}]")
        self.logger.info(f"  Fused    : [{fused_w.min():.4f}, {fused_w.max():.4f}]")

        # Reliability & stability
        fusion_info = result.details.get("fusion", {})
        rel = fusion_info.get("reliability_scores", {})
        if rel:
            self.logger.info("  Reliability: " +
                             ", ".join(f"{k}={v:.4f}" for k, v in rel.items()))

        stab = result.details.get("stability", {})
        if stab:
            self.logger.info(
                f"  Stability: cos={stab.get('cosine_similarity', 0):.4f}, "
                f"spearman={stab.get('spearman_correlation', 0):.4f}, "
                f"stable={stab.get('is_stable', 'N/A')}"
            )

        boot = result.details.get("bootstrap", {})
        if boot:
            mean_std = np.mean([boot["std_weights"][c] for c in subcriteria])
            self.logger.info(
                f"  Bootstrap ({boot.get('iterations', '?')} iters): "
                f"mean σ_w = {mean_std:.6f}"
            )

        fused_dict = {sc: result.weights[sc] for sc in subcriteria}

        return {
            'entropy': entropy_w,
            'critic': critic_w,
            'merec': merec_w,
            'std_dev': stddev_w,
            'fused': fused_w,
            'fused_dict': fused_dict,
            'subcriteria': subcriteria,
            'details': result.details,
        }

    # -----------------------------------------------------------------
    # Phase 3: Hierarchical Ranking
    # -----------------------------------------------------------------

    def _run_hierarchical_ranking(
        self,
        panel_data: PanelData,
        weights: Dict[str, Any],
    ) -> HierarchicalRankingResult:
        pipeline = HierarchicalRankingPipeline(
            n_grades=self.config.ifs.n_grades,
            method_weight_scheme=self.config.er.method_weight_scheme,
            ifs_spread_factor=self.config.ifs.spread_factor,
        )
        return pipeline.rank(
            panel_data=panel_data,
            subcriteria_weights=weights['fused_dict'],
        )

    # -----------------------------------------------------------------
    # Phase 4: ML Feature Importance (optional)
    # -----------------------------------------------------------------

    def _run_ml(
        self,
        panel_data: PanelData,
        ranking_result: HierarchicalRankingResult,
    ) -> Dict[str, Any]:
        """Train Random Forest on ER scores for feature importance."""
        from .ml import RandomForestTS

        results: Dict[str, Any] = {'rf_result': None, 'rf_importance': {}}
        subcriteria = panel_data.hierarchy.all_subcriteria

        features_df = panel_data.subcriteria_long.copy()

        # Map ER scores to each province
        score_map = ranking_result.final_scores.to_dict()
        features_df['target_score'] = features_df['Province'].map(score_map)
        features_df = features_df.dropna(subset=['target_score'])

        unique_years = features_df['Year'].nunique() if 'Year' in features_df.columns else 0
        rf_splits = self.config.random_forest.n_splits

        if unique_years < rf_splits + 1 and unique_years >= 2:
            rf_splits = max(1, unique_years - 1)

        if len(features_df) < 10 or unique_years < 2:
            self.logger.warning(
                f"Skipping RF: insufficient data "
                f"({len(features_df)} samples, {unique_years} years)"
            )
            return results

        rf = RandomForestTS(
            n_estimators=self.config.random_forest.n_estimators,
            max_depth=self.config.random_forest.max_depth,
            min_samples_split=self.config.random_forest.min_samples_split,
            min_samples_leaf=self.config.random_forest.min_samples_leaf,
            max_features=self.config.random_forest.max_features,
            n_splits=rf_splits,
        )

        rf_result = rf.fit_predict(features_df, 'target_score', subcriteria)
        results['rf_result'] = rf_result
        results['rf_importance'] = rf_result.feature_importance.to_dict()

        cv_r2 = np.mean(rf_result.cv_scores.get('r2', [0]))
        self.logger.info(f"RF CV R² = {cv_r2:.4f}")

        return results

    # -----------------------------------------------------------------
    # Phase 5: Sensitivity Analysis
    # -----------------------------------------------------------------

    def _run_analysis(
        self,
        panel_data: PanelData,
        ranking_result: HierarchicalRankingResult,
        weights: Dict[str, Any],
    ) -> Dict[str, Any]:
        subcriteria = panel_data.hierarchy.all_subcriteria
        latest_year = max(panel_data.years)
        matrix = panel_data.subcriteria_cross_section[latest_year][subcriteria].values

        def topsis_ranking_func(m, w):
            df = pd.DataFrame(m, columns=subcriteria)
            w_dict = dict(zip(subcriteria, w))
            calc = TOPSISCalculator()
            res = calc.calculate(df, w_dict)
            return res.ranks.values

        sensitivity = SensitivityAnalysis(
            n_simulations=self.config.validation.n_simulations,
        )
        sens_result = sensitivity.analyze(
            matrix,
            weights['fused'],
            topsis_ranking_func,
            criteria_names=subcriteria,
            alternative_names=panel_data.provinces,
        )
        self.logger.info(
            f"Sensitivity: robustness = {sens_result.overall_robustness:.4f}"
        )
        return {'sensitivity': sens_result}

    # -----------------------------------------------------------------
    # Phase 6: Visualisations
    # -----------------------------------------------------------------

    def _generate_all_visualizations(
        self,
        panel_data: PanelData,
        weights: Dict[str, Any],
        ranking_result: HierarchicalRankingResult,
        analysis_results: Dict[str, Any],
        ml_results: Optional[Dict[str, Any]] = None,
    ) -> None:
        fig_count = 0

        try:
            # 1. Final ER ranking bar chart
            self.visualizer.plot_final_ranking_summary(
                panel_data.provinces,
                _to_array(ranking_result.final_scores),
                _to_array(ranking_result.final_ranking),
                title='Hierarchical ER Final Ranking',
                save_name='01_er_final_ranking.png',
            )
            fig_count += 1

            # 2. Score distribution histogram
            self.visualizer.plot_score_distribution(
                _to_array(ranking_result.final_scores),
                title='ER Score Distribution',
                save_name='02_er_score_distribution.png',
            )
            fig_count += 1

            # 3. Weight comparison grouped bar
            subcriteria = weights['subcriteria']
            weight_dict_for_plot = {
                'Entropy': weights['entropy'],
                'CRITIC': weights['critic'],
                'MEREC': weights['merec'],
                'Std Dev': weights['std_dev'],
                'Fused': weights['fused'],
            }
            self.visualizer.plot_weights_comparison(
                weight_dict_for_plot, subcriteria,
                title='Subcriteria Weight Comparison',
                save_name='03_weights_comparison.png',
            )
            fig_count += 1

            # 4. Sensitivity analysis
            if analysis_results.get('sensitivity'):
                self.visualizer.plot_sensitivity_analysis(
                    analysis_results['sensitivity'].weight_sensitivity,
                    title='Weight Sensitivity Analysis',
                    save_name='04_sensitivity_analysis.png',
                )
                fig_count += 1

            # 5. Feature importance
            if ml_results and ml_results.get('rf_importance'):
                self.visualizer.plot_feature_importance_single(
                    ml_results['rf_importance'],
                    title='Random Forest Feature Importance',
                    save_name='05_feature_importance.png',
                )
                fig_count += 1

            self.logger.info(f"Figures generated: {fig_count}")

        except Exception as e:
            self.logger.warning(f"Visualisation error: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())

    # -----------------------------------------------------------------
    # Phase 7: Save Results
    # -----------------------------------------------------------------

    def _save_all_results(
        self,
        panel_data: PanelData,
        weights: Dict[str, Any],
        ranking_result: HierarchicalRankingResult,
        ml_results: Dict[str, Any],
        analysis_results: Dict[str, Any],
        execution_time: float,
    ) -> None:
        results_dir = Path(self.config.output_dir) / 'results'
        results_dir.mkdir(parents=True, exist_ok=True)
        reports_dir = Path(self.config.output_dir) / 'reports'
        reports_dir.mkdir(parents=True, exist_ok=True)

        # 1. Final rankings
        ranking_df = pd.DataFrame({
            'Province': ranking_result.final_ranking.index,
            'ER_Score': ranking_result.final_scores.values,
            'ER_Rank': ranking_result.final_ranking.values,
        }).sort_values('ER_Rank').reset_index(drop=True)
        ranking_df.insert(0, 'Rank', range(1, len(ranking_df) + 1))
        ranking_df.to_csv(results_dir / 'final_rankings.csv', index=False)
        self.logger.info("Saved: final_rankings.csv")

        # 2. Criterion weights used by ER Stage 2
        crit_w = ranking_result.criterion_weights_used
        pd.DataFrame([crit_w]).to_csv(
            results_dir / 'criterion_weights.csv', index=False
        )
        self.logger.info("Saved: criterion_weights.csv")

        # 3. MCDM scores per criterion group
        for crit_id, method_scores in ranking_result.criterion_method_scores.items():
            crit_df = pd.DataFrame(method_scores)
            crit_df.index = panel_data.provinces
            crit_df.index.name = 'Province'
            crit_df.to_csv(results_dir / f'mcdm_scores_{crit_id}.csv')
        self.logger.info(
            f"Saved: mcdm_scores_*.csv ({len(ranking_result.criterion_method_scores)} files)"
        )

        # 4. MCDM rank comparison across all methods (latest-year global)
        all_method_ranks = {}
        for crit_id, method_ranks in ranking_result.criterion_method_ranks.items():
            for method, ranks_series in method_ranks.items():
                col = f"{crit_id}_{method}"
                all_method_ranks[col] = ranks_series.values
        rank_comparison_df = pd.DataFrame(
            all_method_ranks, index=panel_data.provinces
        )
        rank_comparison_df.index.name = 'Province'
        rank_comparison_df.to_csv(results_dir / 'mcdm_rank_comparison.csv')
        self.logger.info("Saved: mcdm_rank_comparison.csv")

        # 5. ER uncertainty
        ranking_result.er_result.uncertainty.to_csv(
            results_dir / 'prediction_uncertainty_er.csv'
        )
        self.logger.info("Saved: prediction_uncertainty_er.csv")

        # 6. Subcriteria weights
        subcriteria = weights['subcriteria']
        weights_df = pd.DataFrame({
            'Subcriteria': subcriteria,
            'Entropy': weights['entropy'],
            'CRITIC': weights['critic'],
            'MEREC': weights['merec'],
            'Std_Dev': weights['std_dev'],
            'Fused': weights['fused'],
        })
        weights_df.to_csv(results_dir / 'weights_analysis.csv', index=False,
                          float_format='%.6f')
        self.logger.info("Saved: weights_analysis.csv")

        # 7. RF feature importance
        if ml_results.get('rf_result'):
            rf = ml_results['rf_result']
            rf.feature_importance.to_frame('Importance').to_csv(
                results_dir / 'feature_importance.csv'
            )
            self.logger.info("Saved: feature_importance.csv")

            if rf.cv_scores:
                cv_df = pd.DataFrame(rf.cv_scores)
                cv_df.index.name = 'Fold'
                cv_df.to_csv(results_dir / 'cv_scores.csv')
                self.logger.info("Saved: cv_scores.csv")

        # 8. Sensitivity
        if analysis_results.get('sensitivity'):
            sens = analysis_results['sensitivity']
            if hasattr(sens, 'weight_sensitivity') and sens.weight_sensitivity:
                sens_df = pd.DataFrame([
                    {'Criterion': k, 'Sensitivity': v}
                    for k, v in sorted(
                        sens.weight_sensitivity.items(),
                        key=lambda x: x[1], reverse=True,
                    )
                ])
                sens_df.to_csv(results_dir / 'sensitivity_analysis.csv', index=False,
                               float_format='%.6f')
                self.logger.info("Saved: sensitivity_analysis.csv")

            if hasattr(sens, 'overall_robustness'):
                pd.DataFrame([{
                    'Robustness': sens.overall_robustness,
                }]).to_csv(results_dir / 'robustness_summary.csv', index=False,
                           float_format='%.6f')
                self.logger.info("Saved: robustness_summary.csv")

        # 9. Data summary statistics
        latest_year = max(panel_data.years)
        summary_df = panel_data.subcriteria_cross_section[latest_year].describe().T
        summary_df.index.name = 'Subcriteria'
        summary_df.to_csv(results_dir / 'data_summary_statistics.csv',
                          float_format='%.6f')
        self.logger.info("Saved: data_summary_statistics.csv")

        # 10. Execution summary (JSON)
        summary = {
            'execution_time_seconds': round(execution_time, 2),
            'n_provinces': len(panel_data.provinces),
            'n_years': len(panel_data.years),
            'years': panel_data.years,
            'n_subcriteria': panel_data.n_subcriteria,
            'n_criteria': panel_data.n_criteria,
            'n_mcdm_methods': len(ranking_result.methods_used),
            'methods_used': ranking_result.methods_used,
            'target_year': ranking_result.target_year,
            'kendall_w': ranking_result.kendall_w,
            'aggregation': 'Evidential Reasoning (Yang & Xu, 2002)',
            'fuzzy_extension': 'Intuitionistic Fuzzy Sets (Atanassov, 1986)',
        }
        with open(results_dir / 'execution_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        self.logger.info("Saved: execution_summary.json")

        # 11. Config snapshot
        try:
            self.config.save(results_dir / 'config_snapshot.json')
            self.logger.info("Saved: config_snapshot.json")
        except Exception:
            pass

        # 12. Text report
        try:
            report = self._build_text_report(
                panel_data, weights, ranking_result,
                ml_results, analysis_results, execution_time,
            )
            with open(reports_dir / 'report.txt', 'w', encoding='utf-8') as f:
                f.write(report)
            self.logger.info("Saved: report.txt")
        except Exception as e:
            self.logger.warning(f"Report generation failed: {e}")

        self.logger.info(f"All results saved to {results_dir}")

    # -----------------------------------------------------------------
    # Report builder
    # -----------------------------------------------------------------

    def _build_text_report(
        self,
        panel_data: PanelData,
        weights: Dict[str, Any],
        ranking_result: HierarchicalRankingResult,
        ml_results: Dict[str, Any],
        analysis_results: Dict[str, Any],
        execution_time: float,
    ) -> str:
        from datetime import datetime

        lines: List[str] = []
        w = 80

        lines.append("=" * w)
        lines.append("  ML-MCDM ANALYSIS REPORT")
        lines.append("  IFS + Evidential Reasoning Hierarchical Ranking")
        lines.append("=" * w)
        lines.append(f"  Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"  Runtime   : {execution_time:.2f}s")
        lines.append("")

        # --- Data overview ---
        lines.append("-" * w)
        lines.append("  1. DATA OVERVIEW")
        lines.append("-" * w)
        lines.append(f"  Provinces   : {len(panel_data.provinces)}")
        lines.append(f"  Years       : {min(panel_data.years)}-{max(panel_data.years)} "
                     f"({len(panel_data.years)} years)")
        lines.append(f"  Subcriteria : {panel_data.n_subcriteria}")
        lines.append(f"  Criteria    : {panel_data.n_criteria}")
        lines.append("")

        # --- Weighting ---
        lines.append("-" * w)
        lines.append("  2. WEIGHTING (GTWC)")
        lines.append("-" * w)
        subcriteria = weights['subcriteria']
        lines.append(f"  {'Subcriteria':<12} {'Entropy':>10} {'CRITIC':>10} "
                     f"{'MEREC':>10} {'StdDev':>10} {'Fused':>10}")
        lines.append("  " + "-" * 62)
        for i, sc in enumerate(subcriteria):
            lines.append(
                f"  {sc:<12} {weights['entropy'][i]:>10.4f} {weights['critic'][i]:>10.4f} "
                f"{weights['merec'][i]:>10.4f} {weights['std_dev'][i]:>10.4f} "
                f"{weights['fused'][i]:>10.4f}"
            )
        lines.append("")

        # --- Ranking ---
        lines.append("-" * w)
        lines.append("  3. HIERARCHICAL ER RANKING")
        lines.append("-" * w)
        lines.append(f"  Methods     : {len(ranking_result.methods_used)} "
                     f"({', '.join(ranking_result.methods_used)})")
        lines.append(f"  Kendall W   : {ranking_result.kendall_w:.4f}")
        lines.append(f"  Target year : {ranking_result.target_year}")
        lines.append("")
        lines.append(f"  {'Rank':<6} {'Province':<25} {'ER Score':>10}")
        lines.append("  " + "-" * 42)
        ranking_df = pd.DataFrame({
            'Province': ranking_result.final_ranking.index,
            'Score': ranking_result.final_scores.values,
            'Rank': ranking_result.final_ranking.values,
        }).sort_values('Rank')
        for _, row in ranking_df.iterrows():
            lines.append(f"  {int(row['Rank']):<6} {row['Province']:<25} {row['Score']:>10.4f}")
        lines.append("")

        # --- Sensitivity ---
        if analysis_results.get('sensitivity'):
            sens = analysis_results['sensitivity']
            lines.append("-" * w)
            lines.append("  4. SENSITIVITY ANALYSIS")
            lines.append("-" * w)
            lines.append(f"  Overall robustness : {sens.overall_robustness:.4f}")
            if hasattr(sens, 'weight_sensitivity') and sens.weight_sensitivity:
                lines.append(f"  {'Criterion':<20} {'Sensitivity':>12}")
                lines.append("  " + "-" * 32)
                for k, v in sorted(sens.weight_sensitivity.items(),
                                   key=lambda x: x[1], reverse=True)[:10]:
                    lines.append(f"  {k:<20} {v:>12.4f}")
            lines.append("")

        # --- ML Feature Importance ---
        if ml_results.get('rf_importance'):
            lines.append("-" * w)
            lines.append("  5. RF FEATURE IMPORTANCE")
            lines.append("-" * w)
            sorted_imp = sorted(ml_results['rf_importance'].items(),
                                key=lambda x: x[1], reverse=True)
            for feat, imp in sorted_imp[:15]:
                lines.append(f"  {feat:<20} {imp:.4f}")
            lines.append("")

        lines.append("=" * w)
        lines.append("  END OF REPORT")
        lines.append("=" * w)
        return "\n".join(lines)


# =========================================================================
# Convenience function
# =========================================================================

def run_pipeline(
    data_path: Optional[str] = None,
    config: Optional[Config] = None,
) -> PipelineResult:
    """Run the full pipeline. Returns PipelineResult."""
    pipeline = MLMCDMPipeline(config)
    return pipeline.run(data_path)
