"""
Mean Reversion Trading System

A comprehensive cryptocurrency and stock factor research and backtesting platform
focused on mean reversion trading strategies.
"""

__version__ = "0.1.0"
__author__ = "Mean Reversion Trading Team"
__email__ = "team@meanreversion.com"

from .utils.config import Config
from .utils.logger import get_logger

# Initialize global configuration
config = Config()
logger = get_logger(__name__)

__all__ = [
    "config",
    "logger",
    "__version__",
    "__author__",
    "__email__",
]
