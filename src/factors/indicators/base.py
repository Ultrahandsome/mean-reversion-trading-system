"""Base technical indicator class for the Mean Reversion Trading System."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union
import pandas as pd
import numpy as np

from ...data.models import MarketData
from ...utils.logger import get_logger

logger = get_logger(__name__)


class TechnicalIndicator(ABC):
    """Abstract base class for technical indicators."""
    
    def __init__(self, name: str):
        """
        Initialize technical indicator.
        
        Args:
            name: Indicator name
        """
        self.name = name
        self._cache = {}
    
    @abstractmethod
    def calculate(self, market_data: MarketData, **kwargs) -> pd.Series:
        """
        Calculate the technical indicator.
        
        Args:
            market_data: Market data to calculate indicator on
            **kwargs: Additional parameters
            
        Returns:
            Pandas Series with indicator values
        """
        pass
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get indicator parameters."""
        return {}
    
    def validate_data(self, market_data: MarketData, min_periods: int = 1) -> None:
        """
        Validate input data.
        
        Args:
            market_data: Market data to validate
            min_periods: Minimum number of periods required
        """
        if len(market_data.data) < min_periods:
            raise ValueError(
                f"Insufficient data for {self.name}. "
                f"Required: {min_periods}, Available: {len(market_data.data)}"
            )
    
    def _get_cache_key(self, market_data: MarketData, **kwargs) -> str:
        """Generate cache key for the calculation."""
        symbol = market_data.symbol
        data_hash = hash(str(market_data.data.index[-1]) + str(len(market_data.data)))
        params_hash = hash(str(sorted(kwargs.items())))
        return f"{symbol}_{data_hash}_{params_hash}"
    
    def calculate_with_cache(self, market_data: MarketData, **kwargs) -> pd.Series:
        """
        Calculate indicator with caching.
        
        Args:
            market_data: Market data
            **kwargs: Additional parameters
            
        Returns:
            Calculated indicator values
        """
        cache_key = self._get_cache_key(market_data, **kwargs)
        
        if cache_key in self._cache:
            logger.debug(f"Cache hit for {self.name}")
            return self._cache[cache_key]
        
        result = self.calculate(market_data, **kwargs)
        self._cache[cache_key] = result
        
        logger.debug(f"Calculated and cached {self.name}")
        return result
    
    def clear_cache(self) -> None:
        """Clear indicator cache."""
        self._cache.clear()
    
    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(name='{self.name}')"


class TechnicalIndicators:
    """Collection of technical indicators."""
    
    def __init__(self):
        """Initialize technical indicators collection."""
        self.indicators = {}
        self._setup_indicators()
    
    def _setup_indicators(self):
        """Setup available indicators."""
        from .moving_averages import MovingAverages
        from .volatility import VolatilityIndicators
        from .momentum import MomentumIndicators
        from .mean_reversion import MeanReversionIndicators
        
        self.moving_averages = MovingAverages()
        self.volatility = VolatilityIndicators()
        self.momentum = MomentumIndicators()
        self.mean_reversion = MeanReversionIndicators()
    
    def calculate_all(self, market_data: MarketData, config: Dict[str, Any]) -> pd.DataFrame:
        """
        Calculate all configured indicators.
        
        Args:
            market_data: Market data
            config: Configuration for indicators
            
        Returns:
            DataFrame with all indicator values
        """
        results = pd.DataFrame(index=market_data.data.index)
        
        # Add price data
        results['close'] = market_data.data['close']
        results['volume'] = market_data.data['volume']
        
        # Calculate moving averages
        if 'moving_averages' in config:
            ma_config = config['moving_averages']
            for period in ma_config.get('periods', [20, 50]):
                results[f'sma_{period}'] = self.moving_averages.sma(market_data, period)
                results[f'ema_{period}'] = self.moving_averages.ema(market_data, period)
        
        # Calculate volatility indicators
        if 'volatility' in config:
            vol_config = config['volatility']
            period = vol_config.get('period', 20)
            results[f'volatility_{period}'] = self.volatility.historical_volatility(market_data, period)
            
            # Bollinger Bands
            bb_period = vol_config.get('bollinger_period', 20)
            bb_std = vol_config.get('bollinger_std', 2.0)
            bb_upper, bb_middle, bb_lower = self.volatility.bollinger_bands(market_data, bb_period, bb_std)
            results[f'bb_upper_{bb_period}'] = bb_upper
            results[f'bb_middle_{bb_period}'] = bb_middle
            results[f'bb_lower_{bb_period}'] = bb_lower
        
        # Calculate momentum indicators
        if 'momentum' in config:
            mom_config = config['momentum']
            rsi_period = mom_config.get('rsi_period', 14)
            results[f'rsi_{rsi_period}'] = self.momentum.rsi(market_data, rsi_period)
        
        # Calculate mean reversion indicators
        if 'mean_reversion' in config:
            mr_config = config['mean_reversion']
            zscore_period = mr_config.get('zscore_period', 20)
            results[f'zscore_{zscore_period}'] = self.mean_reversion.zscore(market_data, zscore_period)
        
        return results
    
    def get_indicator_config(self) -> Dict[str, Any]:
        """Get default indicator configuration."""
        return {
            'moving_averages': {
                'periods': [10, 20, 50, 200]
            },
            'volatility': {
                'period': 20,
                'bollinger_period': 20,
                'bollinger_std': 2.0
            },
            'momentum': {
                'rsi_period': 14
            },
            'mean_reversion': {
                'zscore_period': 20
            }
        }
