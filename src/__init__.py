# -*- coding: utf-8 -*-
"""ML-MCDM: Machine Learning enhanced Multi-Criteria Decision Making framework."""

from .config import Config, get_default_config
from .logger import setup_logger
from .data_loader import PanelDataLoader, PanelData
from .pipeline import MLTOPSISPipeline, run_pipeline, PipelineResult
from .output_manager import OutputManager, create_output_manager
from .visualization import PanelVisualizer, create_visualizer

__version__ = '2.0.0'

__all__ = [
    'Config', 'get_default_config',
    'setup_logger',
    'PanelDataLoader', 'PanelData',
    'MLTOPSISPipeline', 'run_pipeline', 'PipelineResult',
    'OutputManager', 'create_output_manager',
    'PanelVisualizer', 'create_visualizer',
]
