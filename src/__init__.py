"""
Momentum Strategy Package

A time-series momentum strategy implementation based on:
- "Trend Following: Equity and Bond Crisis Alpha" by Hamill, Rattray, Van Hemert (AQR)
- "Time Series Momentum" by Moskowitz, Ooi, Pedersen (2012)
"""

from .data import DataFetcher, load_prices, DEFAULT_UNIVERSE
from .strategy import MomentumStrategy, Backtester, run_momentum_backtest
from .visualization import create_full_report

__version__ = '1.0.0'
__author__ = 'Eduardo Medina'

__all__ = [
    'DataFetcher',
    'load_prices',
    'DEFAULT_UNIVERSE',
    'MomentumStrategy',
    'Backtester',
    'run_momentum_backtest',
    'create_full_report',
]
