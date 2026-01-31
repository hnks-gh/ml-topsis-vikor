# -*- coding: utf-8 -*-
"""
Ensemble Module
===============

Stacking meta-learner and rank aggregation methods.
"""

from .stacking import StackingEnsemble, StackingResult
from .aggregation import (
    RankAggregator, BordaCount, CopelandMethod, 
    AggregatedRanking, aggregate_rankings
)

__all__ = [
    'StackingEnsemble', 'StackingResult',
    'RankAggregator', 'BordaCount', 'CopelandMethod', 
    'AggregatedRanking', 'aggregate_rankings'
]
