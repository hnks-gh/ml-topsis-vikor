# -*- coding: utf-8 -*-
"""Logging utilities with colored console output."""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


class ColoredFormatter(logging.Formatter):
    """Custom formatter with ANSI color codes."""
    
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
    }
    RESET = '\033[0m'
    BOLD = '\033[1m'
    
    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, '')
        record.levelname = f"{color}{self.BOLD}{record.levelname}{self.RESET}"
        record.msg = f"{color}{record.msg}{self.RESET}"
        return super().format(record)


def setup_logger(
    name: str = "ml_topsis",
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    console: bool = True
) -> logging.Logger:
    """Setup and configure logger with console and file handlers."""
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers = []
    
    # Console handler with colors
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_fmt = ColoredFormatter(
            '%(asctime)s │ %(levelname)-8s │ %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_fmt)
        logger.addHandler(console_handler)
    
    # File handler without colors
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_fmt = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_fmt)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "ml_topsis") -> logging.Logger:
    """Get existing logger or create default one."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logger(name)
    return logger


class ProgressLogger:
    """Context manager for logging progress of operations."""
    
    def __init__(self, logger: logging.Logger, operation: str, total: int = 0):
        self.logger = logger
        self.operation = operation
        self.total = total
        self.current = 0
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.info(f"▶ Starting: {self.operation}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = (datetime.now() - self.start_time).total_seconds()
        if exc_type is None:
            self.logger.info(f"✓ Completed: {self.operation} ({elapsed:.2f}s)")
        else:
            self.logger.error(f"✗ Failed: {self.operation} ({elapsed:.2f}s)")
        return False
    
    def update(self, n: int = 1, message: str = ""):
        self.current += n
        if self.total > 0:
            pct = (self.current / self.total) * 100
            self.logger.debug(f"  Progress: {self.current}/{self.total} ({pct:.1f}%) {message}")
