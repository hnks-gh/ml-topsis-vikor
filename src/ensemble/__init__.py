# -*- coding: utf-8 -*-
"""Ensemble methods: stacking and rank aggregation."""

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
