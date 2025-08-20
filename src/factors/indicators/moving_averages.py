"""Moving average indicators for the Mean Reversion Trading System."""

import pandas as pd
import numpy as np
from typing import Optional

from .base import TechnicalIndicator
from ...data.models import MarketData


class MovingAverages:
    """Collection of moving average indicators."""
    
    def sma(self, market_data: MarketData, period: int, price_column: str = 'close') -> pd.Series:
        """
        Simple Moving Average.
        
        Args:
            market_data: Market data
            period: Period for moving average
            price_column: Price column to use
            
        Returns:
            SMA values
        """
        if len(market_data.data) < period:
            raise ValueError(f"Insufficient data for SMA({period})")
        
        return market_data.data[price_column].rolling(window=period).mean()
    
    def ema(self, market_data: MarketData, period: int, price_column: str = 'close') -> pd.Series:
        """
        Exponential Moving Average.
        
        Args:
            market_data: Market data
            period: Period for moving average
            price_column: Price column to use
            
        Returns:
            EMA values
        """
        if len(market_data.data) < period:
            raise ValueError(f"Insufficient data for EMA({period})")
        
        return market_data.data[price_column].ewm(span=period).mean()
    
    def wma(self, market_data: MarketData, period: int, price_column: str = 'close') -> pd.Series:
        """
        Weighted Moving Average.
        
        Args:
            market_data: Market data
            period: Period for moving average
            price_column: Price column to use
            
        Returns:
            WMA values
        """
        if len(market_data.data) < period:
            raise ValueError(f"Insufficient data for WMA({period})")
        
        prices = market_data.data[price_column]
        weights = np.arange(1, period + 1)
        
        def weighted_mean(x):
            return np.average(x, weights=weights)
        
        return prices.rolling(window=period).apply(weighted_mean, raw=True)
    
    def hull_ma(self, market_data: MarketData, period: int, price_column: str = 'close') -> pd.Series:
        """
        Hull Moving Average.
        
        Args:
            market_data: Market data
            period: Period for moving average
            price_column: Price column to use
            
        Returns:
            Hull MA values
        """
        if len(market_data.data) < period:
            raise ValueError(f"Insufficient data for Hull MA({period})")
        
        prices = market_data.data[price_column]
        
        # Calculate WMA with half period and full period
        half_period = int(period / 2)
        wma_half = prices.ewm(span=half_period).mean()
        wma_full = prices.ewm(span=period).mean()
        
        # Calculate raw Hull MA
        raw_hma = 2 * wma_half - wma_full
        
        # Smooth with WMA of sqrt(period)
        smooth_period = int(np.sqrt(period))
        return raw_hma.ewm(span=smooth_period).mean()
    
    def vwma(self, market_data: MarketData, period: int, price_column: str = 'close') -> pd.Series:
        """
        Volume Weighted Moving Average.
        
        Args:
            market_data: Market data
            period: Period for moving average
            price_column: Price column to use
            
        Returns:
            VWMA values
        """
        if len(market_data.data) < period:
            raise ValueError(f"Insufficient data for VWMA({period})")
        
        prices = market_data.data[price_column]
        volumes = market_data.data['volume']
        
        # Calculate price * volume
        pv = prices * volumes
        
        # Rolling sums
        pv_sum = pv.rolling(window=period).sum()
        volume_sum = volumes.rolling(window=period).sum()
        
        # VWMA = sum(price * volume) / sum(volume)
        return pv_sum / volume_sum
    
    def tema(self, market_data: MarketData, period: int, price_column: str = 'close') -> pd.Series:
        """
        Triple Exponential Moving Average.
        
        Args:
            market_data: Market data
            period: Period for moving average
            price_column: Price column to use
            
        Returns:
            TEMA values
        """
        if len(market_data.data) < period * 3:
            raise ValueError(f"Insufficient data for TEMA({period})")
        
        prices = market_data.data[price_column]
        
        # First EMA
        ema1 = prices.ewm(span=period).mean()
        
        # Second EMA (EMA of EMA)
        ema2 = ema1.ewm(span=period).mean()
        
        # Third EMA (EMA of EMA of EMA)
        ema3 = ema2.ewm(span=period).mean()
        
        # TEMA = 3*EMA1 - 3*EMA2 + EMA3
        return 3 * ema1 - 3 * ema2 + ema3
    
    def kama(self, market_data: MarketData, period: int = 10, fast_sc: int = 2, slow_sc: int = 30, price_column: str = 'close') -> pd.Series:
        """
        Kaufman's Adaptive Moving Average.
        
        Args:
            market_data: Market data
            period: Period for efficiency ratio calculation
            fast_sc: Fast smoothing constant
            slow_sc: Slow smoothing constant
            price_column: Price column to use
            
        Returns:
            KAMA values
        """
        if len(market_data.data) < period + 1:
            raise ValueError(f"Insufficient data for KAMA({period})")
        
        prices = market_data.data[price_column]
        
        # Calculate change and volatility
        change = abs(prices - prices.shift(period))
        volatility = abs(prices - prices.shift(1)).rolling(window=period).sum()
        
        # Efficiency ratio
        er = change / volatility
        
        # Smoothing constants
        fast_alpha = 2.0 / (fast_sc + 1)
        slow_alpha = 2.0 / (slow_sc + 1)
        
        # Scaled smoothing constant
        sc = (er * (fast_alpha - slow_alpha) + slow_alpha) ** 2
        
        # Calculate KAMA
        kama = pd.Series(index=prices.index, dtype=float)
        kama.iloc[period] = prices.iloc[period]
        
        for i in range(period + 1, len(prices)):
            kama.iloc[i] = kama.iloc[i-1] + sc.iloc[i] * (prices.iloc[i] - kama.iloc[i-1])
        
        return kama
    
    def ma_envelope(self, market_data: MarketData, period: int, envelope_pct: float = 0.025, ma_type: str = 'sma', price_column: str = 'close') -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Moving Average Envelope.
        
        Args:
            market_data: Market data
            period: Period for moving average
            envelope_pct: Envelope percentage (e.g., 0.025 for 2.5%)
            ma_type: Type of moving average ('sma', 'ema', 'wma')
            price_column: Price column to use
            
        Returns:
            Tuple of (upper_envelope, middle_ma, lower_envelope)
        """
        # Calculate base moving average
        if ma_type == 'sma':
            ma = self.sma(market_data, period, price_column)
        elif ma_type == 'ema':
            ma = self.ema(market_data, period, price_column)
        elif ma_type == 'wma':
            ma = self.wma(market_data, period, price_column)
        else:
            raise ValueError(f"Unsupported MA type: {ma_type}")
        
        # Calculate envelopes
        upper_envelope = ma * (1 + envelope_pct)
        lower_envelope = ma * (1 - envelope_pct)
        
        return upper_envelope, ma, lower_envelope
    
    def ma_crossover(self, market_data: MarketData, fast_period: int, slow_period: int, ma_type: str = 'sma', price_column: str = 'close') -> pd.Series:
        """
        Moving Average Crossover Signal.
        
        Args:
            market_data: Market data
            fast_period: Fast MA period
            slow_period: Slow MA period
            ma_type: Type of moving average
            price_column: Price column to use
            
        Returns:
            Crossover signals (1 for bullish, -1 for bearish, 0 for no signal)
        """
        if fast_period >= slow_period:
            raise ValueError("Fast period must be less than slow period")
        
        # Calculate moving averages
        if ma_type == 'sma':
            fast_ma = self.sma(market_data, fast_period, price_column)
            slow_ma = self.sma(market_data, slow_period, price_column)
        elif ma_type == 'ema':
            fast_ma = self.ema(market_data, fast_period, price_column)
            slow_ma = self.ema(market_data, slow_period, price_column)
        else:
            raise ValueError(f"Unsupported MA type: {ma_type}")
        
        # Calculate crossover signals
        signals = pd.Series(0, index=market_data.data.index)
        
        # Bullish crossover: fast MA crosses above slow MA
        bullish_cross = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))
        signals[bullish_cross] = 1
        
        # Bearish crossover: fast MA crosses below slow MA
        bearish_cross = (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))
        signals[bearish_cross] = -1
        
        return signals
