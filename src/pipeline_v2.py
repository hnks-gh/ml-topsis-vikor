# -*- coding: utf-8 -*-
"""
Refactored Pipeline v2.0
=========================

New architecture:
- Traditional MCDM methods: Use LAST YEAR only
- Fuzzy MCDM methods: Use LAST YEAR with uncertainty from temporal variance  
- ML Forecasting: Use ALL historical data to predict NEXT YEAR
- Combines Traditional + Fuzzy MCDM with ML predictions

Methods:
- TOPSIS (Traditional + Fuzzy)
- VIKOR (Traditional + Fuzzy)  
- PROMETHEE (Traditional + Fuzzy)
- COPRAS (Traditional + Fuzzy)
- EDAS (Traditional + Fuzzy)

Author: ML-MCDM Research Team
Version: 2.0.0
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import warnings
import time

warnings.filterwarnings('ignore')


@dataclass
class MCDMScore:
    """Container for a single MCDM method's results."""
    method_name: str
    variant: str  # 'traditional' or 'fuzzy'
    scores: np.ndarray
    rankings: np.ndarray
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass  
class PipelineResultV2:
    """Container for v2 pipeline results."""
    
    # Panel data
    panel_data: Any
    entities: List[str]
    components: List[str]
    years: List[int]
    
    # MCDM Results (last year)
    traditional_mcdm: Dict[str, MCDMScore]  # Method name -> scores
    fuzzy_mcdm: Dict[str, MCDMScore]        # Method name -> fuzzy scores
    
    # Weights
    weights: Dict[str, np.ndarray]
    
    # ML Forecasting (next year prediction)
    ml_forecast: Optional[Any]  # UnifiedForecastResult
    predicted_next_year: Optional[pd.DataFrame]
    prediction_uncertainty: Optional[pd.DataFrame]
    
    # Aggregated results
    final_rankings: pd.DataFrame
    consensus_ranking: np.ndarray
    
    # Metadata
    execution_time: float
    
    def summary(self) -> str:
        """Generate summary report."""
        lines = [
            "\n" + "=" * 80,
            "ML-MCDM PIPELINE v2.0 RESULTS SUMMARY",
            "=" * 80,
            "",
            f"Entities: {len(self.entities)}",
            f"Components: {len(self.components)}",
            f"Years analyzed: {self.years}",
            f"Last year used for MCDM: {max(self.years)}",
            "",
            "## Traditional MCDM Rankings (Last Year)",
            "-" * 40,
        ]
        
        for method, score in self.traditional_mcdm.items():
            top_idx = np.argmin(score.rankings)
            lines.append(f"  {method:12s}: Top = {self.entities[top_idx]} "
                        f"(Score: {score.scores[top_idx]:.4f})")
        
        lines.extend([
            "",
            "## Fuzzy MCDM Rankings (Last Year with Uncertainty)",
            "-" * 40,
        ])
        
        for method, score in self.fuzzy_mcdm.items():
            top_idx = np.argmin(score.rankings)
            lines.append(f"  {method:12s}: Top = {self.entities[top_idx]} "
                        f"(Score: {score.scores[top_idx]:.4f})")
        
        if self.ml_forecast is not None:
            lines.extend([
                "",
                "## ML Forecasting Performance",
                "-" * 40,
            ])
            for model, weight in self.ml_forecast.model_contributions.items():
                lines.append(f"  {model:25s}: Weight = {weight:.3f}")
        
        lines.extend([
            "",
            "## Final Consensus Ranking",
            "-" * 40,
        ])
        
        top_5 = self.final_rankings.nsmallest(5, 'consensus_rank')
        for _, row in top_5.iterrows():
            lines.append(f"  Rank {int(row['consensus_rank']):2d}: {row['entity']}")
        
        lines.extend([
            "",
            f"Execution Time: {self.execution_time:.2f} seconds",
            "=" * 80,
        ])
        
        return "\n".join(lines)


class PipelineV2:
    """
    Refactored ML-MCDM Pipeline v2.0
    
    Architecture:
    1. Load panel data
    2. Calculate weights from last year
    3. Apply Traditional MCDM to last year data
    4. Apply Fuzzy MCDM to last year (with temporal variance as uncertainty)
    5. Use ML forecasting to predict next year values
    6. Aggregate all rankings
    """
    
    def __init__(self, 
                 output_dir: str = "outputs",
                 ml_mode: str = "fast",
                 verbose: bool = True):
        self.output_dir = Path(output_dir)
        self.ml_mode = ml_mode
        self.verbose = verbose
        
        self._setup_directories()
    
    def _setup_directories(self):
        """Create output directories."""
        (self.output_dir / 'figures').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'results').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'reports').mkdir(parents=True, exist_ok=True)
    
    def _log(self, msg: str):
        """Print message if verbose."""
        if self.verbose:
            print(f"[Pipeline] {msg}")
    
    def _ensure_panel_data(self, data):
        """Convert DataFrame to PanelData if needed."""
        from .data_loader import PanelData, PanelDataLoader
        
        if isinstance(data, PanelData):
            return data
        
        # Convert DataFrame to PanelData
        if isinstance(data, pd.DataFrame):
            loader = PanelDataLoader()
            return loader.load_from_dataframe(data)
        
        raise ValueError(f"Expected DataFrame or PanelData, got {type(data)}")
    
    def run(self, data) -> PipelineResultV2:
        """
        Execute the full pipeline.
        
        Parameters:
            data: DataFrame or PanelData object with historical data
        
        Returns:
            PipelineResultV2 with all results
        """
        start_time = time.time()
        
        self._log("=" * 60)
        self._log("ML-MCDM PIPELINE v2.0")
        self._log("=" * 60)
        
        # Convert to PanelData if needed
        panel_data = self._ensure_panel_data(data)
        
        # Extract info
        entities = panel_data.provinces
        components = panel_data.components
        years = sorted(panel_data.years)
        last_year = max(years)
        
        self._log(f"Entities: {len(entities)}")
        self._log(f"Components: {len(components)}")
        self._log(f"Years: {years}")
        self._log(f"Using last year ({last_year}) for MCDM")
        
        # Step 1: Calculate weights from last year
        self._log("\n[1/5] Calculating weights...")
        weights = self._calculate_weights(panel_data, last_year)
        
        # Step 2: Traditional MCDM on last year
        self._log("\n[2/5] Running Traditional MCDM methods...")
        traditional_results = self._run_traditional_mcdm(panel_data, weights, last_year)
        
        # Step 3: Fuzzy MCDM on last year
        self._log("\n[3/5] Running Fuzzy MCDM methods...")
        fuzzy_results = self._run_fuzzy_mcdm(panel_data, weights)
        
        # Step 4: ML Forecasting
        self._log("\n[4/5] Running ML Forecasting...")
        ml_forecast, predictions, uncertainty = self._run_ml_forecasting(panel_data)
        
        # Step 5: Aggregate rankings
        self._log("\n[5/5] Aggregating rankings...")
        final_rankings, consensus = self._aggregate_rankings(
            entities, traditional_results, fuzzy_results
        )
        
        execution_time = time.time() - start_time
        
        result = PipelineResultV2(
            panel_data=panel_data,
            entities=entities,
            components=components,
            years=years,
            traditional_mcdm=traditional_results,
            fuzzy_mcdm=fuzzy_results,
            weights=weights,
            ml_forecast=ml_forecast,
            predicted_next_year=predictions,
            prediction_uncertainty=uncertainty,
            final_rankings=final_rankings,
            consensus_ranking=consensus,
            execution_time=execution_time
        )
        
        self._log(f"\nPipeline completed in {execution_time:.2f} seconds")
        
        return result
    
    def _calculate_weights(self, panel_data, last_year: int) -> Dict[str, np.ndarray]:
        """Calculate weights using Entropy and CRITIC."""
        from .mcdm import EntropyWeightCalculator, CRITICWeightCalculator
        
        # Get last year data
        df = panel_data.cross_section[last_year][panel_data.components]
        
        # Entropy weights
        entropy = EntropyWeightCalculator()
        entropy_result = entropy.calculate(df)
        
        # CRITIC weights
        critic = CRITICWeightCalculator()
        critic_result = critic.calculate(df)
        
        # Ensemble (average)
        components = panel_data.components
        entropy_arr = np.array([entropy_result.weights[c] for c in components])
        critic_arr = np.array([critic_result.weights[c] for c in components])
        ensemble_arr = (entropy_arr + critic_arr) / 2
        
        self._log(f"  Entropy weights calculated")
        self._log(f"  CRITIC weights calculated")
        
        return {
            'entropy': entropy_arr,
            'critic': critic_arr,
            'ensemble': ensemble_arr,
            'weights_dict': {c: ensemble_arr[i] for i, c in enumerate(components)}
        }
    
    def _run_traditional_mcdm(self, panel_data, weights: Dict, 
                              last_year: int) -> Dict[str, MCDMScore]:
        """Run all traditional MCDM methods on last year data."""
        from .mcdm import TOPSISCalculator, VIKORCalculator
        from .mcdm.promethee import PROMETHEECalculator
        from .mcdm.copras import COPRASCalculator
        from .mcdm.edas import EDASCalculator
        
        results = {}
        df = panel_data.cross_section[last_year][panel_data.components]
        w = weights['weights_dict']
        
        # TOPSIS
        try:
            topsis = TOPSISCalculator()
            tr = topsis.calculate(df, w)
            results['TOPSIS'] = MCDMScore(
                method_name='TOPSIS',
                variant='traditional',
                scores=tr.scores.values,
                rankings=tr.ranks.values,
                details={'ideal_solution': tr.ideal_solution.to_dict(), 
                         'anti_ideal_solution': tr.anti_ideal_solution.to_dict()}
            )
            self._log(f"  TOPSIS: Done")
        except Exception as e:
            self._log(f"  TOPSIS: Failed - {e}")
        
        # VIKOR
        try:
            vikor = VIKORCalculator(v=0.5)
            vr = vikor.calculate(df, w)
            results['VIKOR'] = MCDMScore(
                method_name='VIKOR',
                variant='traditional',
                scores=vr.Q,
                rankings=vr.final_ranks,
                details={'S': vr.S, 'R': vr.R}
            )
            self._log(f"  VIKOR: Done")
        except Exception as e:
            self._log(f"  VIKOR: Failed - {e}")
        
        # PROMETHEE
        try:
            promethee = PROMETHEECalculator()
            pr = promethee.calculate(df, w)
            results['PROMETHEE'] = MCDMScore(
                method_name='PROMETHEE',
                variant='traditional',
                scores=pr.phi_net.values,
                rankings=pr.ranks_promethee_ii.values,
                details={'positive_flow': pr.phi_positive.values, 'negative_flow': pr.phi_negative.values}
            )
            self._log(f"  PROMETHEE: Done")
        except Exception as e:
            self._log(f"  PROMETHEE: Failed - {e}")
        
        # COPRAS
        try:
            copras = COPRASCalculator()
            cr = copras.calculate(df, w)
            results['COPRAS'] = MCDMScore(
                method_name='COPRAS',
                variant='traditional',
                scores=cr.utility_degree.values,
                rankings=cr.ranks.values,
                details={'S_plus': cr.S_plus.values, 'S_minus': cr.S_minus.values}
            )
            self._log(f"  COPRAS: Done")
        except Exception as e:
            self._log(f"  COPRAS: Failed - {e}")
        
        # EDAS
        try:
            edas = EDASCalculator()
            er = edas.calculate(df, w)
            results['EDAS'] = MCDMScore(
                method_name='EDAS',
                variant='traditional',
                scores=er.AS.values,
                rankings=er.ranks.values,
                details={'SP': er.SP.values, 'SN': er.SN.values}
            )
            self._log(f"  EDAS: Done")
        except Exception as e:
            self._log(f"  EDAS: Failed - {e}")
        
        return results
    
    def _run_fuzzy_mcdm(self, panel_data, weights: Dict) -> Dict[str, MCDMScore]:
        """Run all fuzzy MCDM methods using temporal variance as uncertainty."""
        from .mcdm import (
            FuzzyTOPSIS, FuzzyVIKOR, FuzzyPROMETHEE, 
            FuzzyCOPRAS, FuzzyEDAS
        )
        
        results = {}
        w = weights['weights_dict']  # Dict[str, float]
        
        # Fuzzy TOPSIS
        try:
            fuzzy_topsis = FuzzyTOPSIS()
            ftr = fuzzy_topsis.calculate_from_panel(panel_data, w)
            results['Fuzzy_TOPSIS'] = MCDMScore(
                method_name='Fuzzy TOPSIS',
                variant='fuzzy',
                scores=ftr.scores.values,
                rankings=ftr.ranks.values,
                details={'d_positive': ftr.d_positive.values, 'd_negative': ftr.d_negative.values}
            )
            self._log(f"  Fuzzy TOPSIS: Done")
        except Exception as e:
            self._log(f"  Fuzzy TOPSIS: Failed - {e}")
        
        # Fuzzy VIKOR
        try:
            fuzzy_vikor = FuzzyVIKOR()
            fvr = fuzzy_vikor.calculate_from_panel(panel_data, w)
            results['Fuzzy_VIKOR'] = MCDMScore(
                method_name='Fuzzy VIKOR',
                variant='fuzzy',
                scores=fvr.Q.values,
                rankings=fvr.ranks_Q.values,
                details={'S': fvr.S.values, 'R': fvr.R.values}
            )
            self._log(f"  Fuzzy VIKOR: Done")
        except Exception as e:
            self._log(f"  Fuzzy VIKOR: Failed - {e}")
        
        # Fuzzy PROMETHEE
        try:
            fuzzy_promethee = FuzzyPROMETHEE()
            fpr = fuzzy_promethee.calculate_from_panel(panel_data, w)
            results['Fuzzy_PROMETHEE'] = MCDMScore(
                method_name='Fuzzy PROMETHEE',
                variant='fuzzy',
                scores=fpr.phi_net.values,
                rankings=fpr.ranks.values,
                details={'phi_positive': fpr.phi_positive.values, 'phi_negative': fpr.phi_negative.values}
            )
            self._log(f"  Fuzzy PROMETHEE: Done")
        except Exception as e:
            self._log(f"  Fuzzy PROMETHEE: Failed - {e}")
        
        # Fuzzy COPRAS
        try:
            fuzzy_copras = FuzzyCOPRAS()
            fcr = fuzzy_copras.calculate_from_panel(panel_data, w)
            results['Fuzzy_COPRAS'] = MCDMScore(
                method_name='Fuzzy COPRAS',
                variant='fuzzy',
                scores=fcr.utility_degree.values,
                rankings=fcr.ranks.values,
                details={'Q': fcr.Q.values}
            )
            self._log(f"  Fuzzy COPRAS: Done")
        except Exception as e:
            self._log(f"  Fuzzy COPRAS: Failed - {e}")
        
        # Fuzzy EDAS
        try:
            fuzzy_edas = FuzzyEDAS()
            fer = fuzzy_edas.calculate_from_panel(panel_data, w)
            results['Fuzzy_EDAS'] = MCDMScore(
                method_name='Fuzzy EDAS',
                variant='fuzzy',
                scores=fer.AS.values,
                rankings=fer.ranks.values,
                details={'NSP': fer.NSP.values, 'NSN': fer.NSN.values}
            )
            self._log(f"  Fuzzy EDAS: Done")
        except Exception as e:
            self._log(f"  Fuzzy EDAS: Failed - {e}")
        
        return results
    
    def _run_ml_forecasting(self, panel_data) -> Tuple[Any, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Run advanced ML forecasting for next year."""
        try:
            from .ml import UnifiedForecaster, ForecastMode
            
            forecaster = UnifiedForecaster(
                mode=ForecastMode(self.ml_mode),
                include_neural=True,
                include_tree_ensemble=True,
                include_linear=True,
                verbose=self.verbose
            )
            
            result = forecaster.forecast(panel_data)
            
            self._log(f"  Forecast complete: {len(result.predictions)} entities")
            self._log(f"  Models used: {list(result.model_contributions.keys())}")
            
            return result, result.predictions, result.uncertainty
            
        except Exception as e:
            self._log(f"  ML Forecasting failed: {e}")
            import traceback
            if self.verbose:
                traceback.print_exc()
            return None, None, None
    
    def _aggregate_rankings(self, entities: List[str],
                           traditional: Dict[str, MCDMScore],
                           fuzzy: Dict[str, MCDMScore]) -> Tuple[pd.DataFrame, np.ndarray]:
        """Aggregate all rankings into consensus."""
        n = len(entities)
        
        # Collect all rankings
        all_rankings = {}
        
        for name, score in traditional.items():
            all_rankings[name] = score.rankings
        
        for name, score in fuzzy.items():
            all_rankings[name] = score.rankings
        
        # Borda count aggregation
        borda_scores = np.zeros(n)
        
        for rankings in all_rankings.values():
            # Borda score: n - rank (higher is better)
            borda_scores += (n - rankings)
        
        # Final ranking
        consensus_ranking = np.argsort(np.argsort(-borda_scores)) + 1
        
        # Create DataFrame
        df = pd.DataFrame({
            'entity': entities,
            'borda_score': borda_scores,
            'consensus_rank': consensus_ranking
        })
        
        # Add individual rankings
        for name, rankings in all_rankings.items():
            df[f'{name}_rank'] = rankings
        
        df = df.sort_values('consensus_rank')
        
        return df, consensus_ranking


def run_pipeline_v2(data_path: str,
                    output_dir: str = "outputs",
                    ml_mode: str = "balanced") -> PipelineResultV2:
    """
    Convenience function to run the v2 pipeline.
    
    Parameters:
        data_path: Path to panel data CSV
        output_dir: Output directory
        ml_mode: 'fast', 'balanced', 'accurate', 'neural', or 'ensemble'
    
    Returns:
        PipelineResultV2 with all results
    """
    from .data_loader import PanelDataLoader
    from .config import get_default_config
    
    # Load data
    config = get_default_config()
    loader = PanelDataLoader(config)
    panel_data = loader.load(data_path)
    
    # Run pipeline
    pipeline = PipelineV2(output_dir=output_dir, ml_mode=ml_mode)
    result = pipeline.run(panel_data)
    
    return result
