# -*- coding: utf-8 -*-
"""
Visualization Module
====================

Comprehensive panel data and MCDM visualization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

# Import matplotlib with non-interactive backend fallback
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.patches import Polygon
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None


class PanelVisualizer:
    """
    Professional visualization for panel data MCDM analysis.
    """
    
    def __init__(self,
                 output_dir: str = 'outputs/figures',
                 style: str = 'seaborn-v0_8-whitegrid',
                 figsize: Tuple[int, int] = (12, 8),
                 dpi: int = 300):
        """
        Initialize visualizer.
        
        Parameters
        ----------
        output_dir : str
            Directory for saving figures
        style : str
            Matplotlib style
        figsize : Tuple[int, int]
            Default figure size
        dpi : int
            Figure resolution
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figsize = figsize
        self.dpi = dpi
        
        if HAS_MATPLOTLIB:
            try:
                plt.style.use(style)
            except:
                plt.style.use('default')
        
        # Color palette
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'tertiary': '#F18F01',
            'quaternary': '#C73E1D',
            'success': '#2E7D32',
            'warning': '#F57C00',
            'info': '#1976D2',
            'light': '#E0E0E0',
            'dark': '#212121'
        }
        
        self.cmap = 'viridis'
    
    def plot_score_evolution(self,
                            scores: pd.DataFrame,
                            entity_col: str = 'province',
                            time_col: str = 'year',
                            score_col: str = 'score',
                            top_n: int = 10,
                            bottom_n: int = 5,
                            title: str = 'Score Evolution Over Time',
                            save_name: str = 'score_evolution.png') -> Optional[str]:
        """
        Plot score evolution for top and bottom performers.
        """
        if not HAS_MATPLOTLIB:
            return None
        
        fig, axes = plt.subplots(1, 2, figsize=(self.figsize[0], self.figsize[1] * 0.7))
        
        # Pivot data
        wide = scores.pivot(index=time_col, columns=entity_col, values=score_col)
        
        # Get top and bottom performers (by mean score)
        mean_scores = wide.mean().sort_values(ascending=False)
        top_entities = mean_scores.head(top_n).index
        bottom_entities = mean_scores.tail(bottom_n).index
        
        # Plot top performers
        ax = axes[0]
        for entity in top_entities:
            ax.plot(wide.index, wide[entity], marker='o', linewidth=2, 
                   label=entity[:15], markersize=6)
        ax.set_xlabel('Year', fontsize=11)
        ax.set_ylabel('Score', fontsize=11)
        ax.set_title(f'Top {top_n} Performers', fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Plot bottom performers
        ax = axes[1]
        for entity in bottom_entities:
            ax.plot(wide.index, wide[entity], marker='s', linewidth=2,
                   label=entity[:15], markersize=6, linestyle='--')
        ax.set_xlabel('Year', fontsize=11)
        ax.set_ylabel('Score', fontsize=11)
        ax.set_title(f'Bottom {bottom_n} Performers', fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        save_path = self.output_dir / save_name
        fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close(fig)
        
        return str(save_path)
    
    def plot_rank_heatmap(self,
                         rankings: pd.DataFrame,
                         entity_col: str = 'province',
                         time_col: str = 'year',
                         rank_col: str = 'rank',
                         title: str = 'Ranking Evolution Heatmap',
                         save_name: str = 'rank_heatmap.png') -> Optional[str]:
        """
        Plot heatmap of rankings over time.
        """
        if not HAS_MATPLOTLIB:
            return None
        
        # Pivot to wide format
        wide = rankings.pivot(index=entity_col, columns=time_col, values=rank_col)
        
        # Sort by mean rank
        wide = wide.loc[wide.mean(axis=1).sort_values().index]
        
        # Limit to top 30 for readability
        if len(wide) > 30:
            wide = wide.head(30)
        
        fig, ax = plt.subplots(figsize=(self.figsize[0], len(wide) * 0.4 + 2))
        
        # Create heatmap
        im = ax.imshow(wide.values, aspect='auto', cmap='RdYlGn_r')
        
        # Labels
        ax.set_yticks(range(len(wide)))
        ax.set_yticklabels([str(e)[:20] for e in wide.index], fontsize=9)
        ax.set_xticks(range(len(wide.columns)))
        ax.set_xticklabels(wide.columns, fontsize=10)
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Rank (1=Best)', fontsize=10)
        
        # Add text annotations
        for i in range(len(wide)):
            for j in range(len(wide.columns)):
                rank = wide.iloc[i, j]
                if not np.isnan(rank):
                    text_color = 'white' if rank > len(wide) / 2 else 'black'
                    ax.text(j, i, f'{int(rank)}', ha='center', va='center',
                           color=text_color, fontsize=8)
        
        ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
        ax.set_xlabel('Year', fontsize=11)
        ax.set_ylabel('Province', fontsize=11)
        
        plt.tight_layout()
        
        save_path = self.output_dir / save_name
        fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close(fig)
        
        return str(save_path)
    
    def plot_method_comparison(self,
                              rankings_dict: Dict[str, np.ndarray],
                              entity_names: List[str],
                              title: str = 'MCDM Method Comparison',
                              save_name: str = 'method_comparison.png') -> Optional[str]:
        """
        Compare rankings from different MCDM methods.
        """
        if not HAS_MATPLOTLIB:
            return None
        
        methods = list(rankings_dict.keys())
        n_methods = len(methods)
        n_entities = len(entity_names)
        
        fig, ax = plt.subplots(figsize=(self.figsize[0], max(8, n_entities * 0.3)))
        
        # Create rank matrix
        rank_matrix = np.column_stack([rankings_dict[m] for m in methods])
        
        # Sort by first method
        sort_idx = np.argsort(rank_matrix[:, 0])[:30]  # Top 30
        rank_matrix = rank_matrix[sort_idx]
        sorted_names = [entity_names[i][:20] for i in sort_idx]
        
        # Parallel coordinates style
        x_positions = np.arange(n_methods)
        
        for i, name in enumerate(sorted_names):
            ranks = rank_matrix[i]
            color = plt.cm.viridis(i / len(sorted_names))
            ax.plot(x_positions, ranks, '-o', color=color, alpha=0.7,
                   linewidth=1.5, markersize=6, label=name if i < 10 else '')
        
        ax.set_xticks(x_positions)
        ax.set_xticklabels(methods, fontsize=11, rotation=15)
        ax.set_ylabel('Rank', fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.invert_yaxis()  # Lower rank is better
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        save_path = self.output_dir / save_name
        fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close(fig)
        
        return str(save_path)
    
    def plot_convergence(self,
                        sigma_by_year: Dict[int, float],
                        beta_coefficient: float,
                        half_life: float,
                        title: str = 'Convergence Analysis',
                        save_name: str = 'convergence.png') -> Optional[str]:
        """
        Plot convergence analysis results.
        """
        if not HAS_MATPLOTLIB:
            return None
        
        fig, axes = plt.subplots(1, 2, figsize=self.figsize)
        
        # Sigma convergence
        ax = axes[0]
        years = list(sigma_by_year.keys())
        sigmas = list(sigma_by_year.values())
        
        ax.plot(years, sigmas, 'o-', linewidth=2, markersize=8,
               color=self.colors['primary'])
        
        # Trend line
        z = np.polyfit(range(len(years)), sigmas, 1)
        p = np.poly1d(z)
        ax.plot(years, p(range(len(years))), '--', 
               color=self.colors['secondary'], linewidth=2, label='Trend')
        
        ax.set_xlabel('Year', fontsize=11)
        ax.set_ylabel('Coefficient of Variation (σ)', fontsize=11)
        ax.set_title('Sigma Convergence', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Beta convergence info
        ax = axes[1]
        ax.axis('off')
        
        convergence_type = 'CONVERGING' if beta_coefficient < 0 else 'DIVERGING'
        color = self.colors['success'] if beta_coefficient < 0 else self.colors['quaternary']
        
        info_text = f"""
        Beta Convergence Analysis
        ─────────────────────────
        
        β coefficient: {beta_coefficient:.4f}
        
        Interpretation:
        {"Provinces are converging" if beta_coefficient < 0 else "Provinces are diverging"}
        {"(poorer catching up to richer)" if beta_coefficient < 0 else "(gap is widening)"}
        
        Convergence Speed: {abs(beta_coefficient):.4f}
        
        Half-life: {half_life:.1f} years
        {"(time to halve the gap)" if half_life < 100 else "(very slow convergence)"}
        
        Status: {convergence_type}
        """
        
        ax.text(0.1, 0.5, info_text, transform=ax.transAxes, fontsize=11,
               verticalalignment='center', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = self.output_dir / save_name
        fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close(fig)
        
        return str(save_path)
    
    def plot_weight_sensitivity(self,
                               weight_sensitivity: Dict[str, float],
                               title: str = 'Weight Sensitivity Analysis',
                               save_name: str = 'weight_sensitivity.png') -> Optional[str]:
        """
        Plot weight sensitivity bar chart.
        """
        if not HAS_MATPLOTLIB:
            return None
        
        fig, ax = plt.subplots(figsize=(self.figsize[0], self.figsize[1] * 0.6))
        
        criteria = list(weight_sensitivity.keys())
        sensitivities = list(weight_sensitivity.values())
        
        # Sort by sensitivity
        sorted_idx = np.argsort(sensitivities)[::-1]
        criteria = [criteria[i] for i in sorted_idx]
        sensitivities = [sensitivities[i] for i in sorted_idx]
        
        # Color gradient
        colors = [plt.cm.RdYlGn_r(s) for s in sensitivities]
        
        bars = ax.barh(range(len(criteria)), sensitivities, color=colors)
        
        ax.set_yticks(range(len(criteria)))
        ax.set_yticklabels([c[:25] for c in criteria], fontsize=10)
        ax.set_xlabel('Sensitivity Index', fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.invert_yaxis()
        
        # Add value labels
        for bar, val in zip(bars, sensitivities):
            ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{val:.3f}', va='center', fontsize=9)
        
        ax.set_xlim(0, max(sensitivities) * 1.15)
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        save_path = self.output_dir / save_name
        fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close(fig)
        
        return str(save_path)
    
    def plot_feature_importance(self,
                               importance_dict: Dict[str, float],
                               title: str = 'Feature Importance',
                               save_name: str = 'feature_importance.png',
                               top_n: int = 20) -> Optional[str]:
        """
        Plot feature importance from ML models.
        """
        if not HAS_MATPLOTLIB:
            return None
        
        fig, ax = plt.subplots(figsize=(self.figsize[0], self.figsize[1] * 0.7))
        
        # Sort and take top N
        sorted_items = sorted(importance_dict.items(), 
                             key=lambda x: x[1], reverse=True)[:top_n]
        features = [item[0] for item in sorted_items]
        importances = [item[1] for item in sorted_items]
        
        # Color gradient
        colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(features)))[::-1]
        
        bars = ax.barh(range(len(features)), importances, color=colors)
        
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels([f[:30] for f in features], fontsize=10)
        ax.set_xlabel('Importance', fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.invert_yaxis()
        
        # Add value labels
        for bar, val in zip(bars, importances):
            ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                   f'{val:.3f}', va='center', fontsize=9)
        
        ax.set_xlim(0, max(importances) * 1.15)
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        save_path = self.output_dir / save_name
        fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close(fig)
        
        return str(save_path)
    
    def plot_radar_chart(self,
                        values: Dict[str, np.ndarray],
                        categories: List[str],
                        title: str = 'Multi-Criteria Profile',
                        save_name: str = 'radar_chart.png') -> Optional[str]:
        """
        Plot radar chart for multi-criteria comparison.
        """
        if not HAS_MATPLOTLIB:
            return None
        
        n_categories = len(categories)
        angles = np.linspace(0, 2 * np.pi, n_categories, endpoint=False).tolist()
        angles += angles[:1]  # Close the polygon
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        colors = plt.cm.Set2(np.linspace(0, 1, len(values)))
        
        for (name, vals), color in zip(values.items(), colors):
            vals = list(vals) + [vals[0]]  # Close
            ax.plot(angles, vals, 'o-', linewidth=2, label=name[:20], color=color)
            ax.fill(angles, vals, alpha=0.15, color=color)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([c[:15] for c in categories], fontsize=10)
        ax.set_title(title, fontsize=13, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=9)
        
        plt.tight_layout()
        
        save_path = self.output_dir / save_name
        fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close(fig)
        
        return str(save_path)
    
    def plot_ensemble_weights(self,
                             weights: Dict[str, float],
                             title: str = 'Ensemble Model Weights',
                             save_name: str = 'ensemble_weights.png') -> Optional[str]:
        """
        Plot pie chart of ensemble weights.
        """
        if not HAS_MATPLOTLIB:
            return None
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        labels = list(weights.keys())
        sizes = list(weights.values())
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
        
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%',
                                          colors=colors, startangle=90,
                                          explode=[0.02] * len(labels))
        
        for text in texts:
            text.set_fontsize(10)
        for autotext in autotexts:
            autotext.set_fontsize(9)
            autotext.set_fontweight('bold')
        
        ax.set_title(title, fontsize=13, fontweight='bold')
        
        plt.tight_layout()
        
        save_path = self.output_dir / save_name
        fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close(fig)
        
        return str(save_path)
    
    def create_dashboard(self,
                        results: Dict[str, Any],
                        save_name: str = 'dashboard.png') -> Optional[str]:
        """
        Create comprehensive results dashboard.
        """
        if not HAS_MATPLOTLIB:
            return None
        
        fig = plt.figure(figsize=(20, 16))
        
        # Create grid
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Title
        fig.suptitle('ML-MCDM Panel Analysis Dashboard', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # 1. Score distribution
        ax1 = fig.add_subplot(gs[0, 0])
        if 'scores' in results:
            ax1.hist(results['scores'], bins=30, color=self.colors['primary'],
                    edgecolor='white', alpha=0.7)
            ax1.set_title('Score Distribution', fontsize=11, fontweight='bold')
            ax1.set_xlabel('Score')
            ax1.set_ylabel('Frequency')
        
        # 2. Top 10 bar chart
        ax2 = fig.add_subplot(gs[0, 1])
        if 'top_10' in results:
            names, scores = zip(*results['top_10'])
            colors = plt.cm.Greens(np.linspace(0.3, 0.9, 10))[::-1]
            ax2.barh(range(10), scores, color=colors)
            ax2.set_yticks(range(10))
            ax2.set_yticklabels([n[:15] for n in names], fontsize=9)
            ax2.set_title('Top 10 Provinces', fontsize=11, fontweight='bold')
            ax2.invert_yaxis()
        
        # 3. Method agreement
        ax3 = fig.add_subplot(gs[0, 2])
        if 'agreement_matrix' in results:
            im = ax3.imshow(results['agreement_matrix'], cmap='RdYlGn', 
                           vmin=-1, vmax=1)
            ax3.set_title('Method Agreement', fontsize=11, fontweight='bold')
            plt.colorbar(im, ax=ax3, shrink=0.8)
        
        # 4-6. More visualizations based on available results
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.text(0.5, 0.5, 'Panel\nAnalysis\nSummary', ha='center', va='center',
                fontsize=14, transform=ax4.transAxes)
        ax4.axis('off')
        
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.text(0.5, 0.5, 'Temporal\nTrends', ha='center', va='center',
                fontsize=14, transform=ax5.transAxes)
        ax5.axis('off')
        
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.text(0.5, 0.5, 'Convergence\nStatus', ha='center', va='center',
                fontsize=14, transform=ax6.transAxes)
        ax6.axis('off')
        
        # 7-9. Bottom row
        ax7 = fig.add_subplot(gs[2, :])
        if 'summary_table' in results:
            ax7.axis('off')
            # Create summary table
            table_data = results['summary_table']
            table = ax7.table(cellText=table_data['values'],
                             colLabels=table_data['columns'],
                             rowLabels=table_data['rows'],
                             loc='center',
                             cellLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.2, 1.5)
        else:
            ax7.text(0.5, 0.5, 'Summary Statistics', ha='center', va='center',
                    fontsize=14, transform=ax7.transAxes)
            ax7.axis('off')
        
        save_path = self.output_dir / save_name
        fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close(fig)
        
        return str(save_path)


def create_visualizer(output_dir: str = 'outputs/figures') -> PanelVisualizer:
    """Factory function to create visualizer."""
    return PanelVisualizer(output_dir=output_dir)
