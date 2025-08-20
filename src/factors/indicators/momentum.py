"""Momentum indicators for the Mean Reversion Trading System."""

import pandas as pd
import numpy as np
from typing import Optional

from ...data.models import MarketData


class MomentumIndicators:
    """Collection of momentum indicators."""
    
    def rsi(self, market_data: MarketData, period: int = 14, price_column: str = 'close') -> pd.Series:
        """
        Relative Strength Index.
        
        Args:
            market_data: Market data
            period: Period for RSI calculation
            price_column: Price column to use
            
        Returns:
            RSI values
        """
        if len(market_data.data) < period + 1:
            raise ValueError(f"Insufficient data for RSI({period})")
        
        prices = market_data.data[price_column]
        
        # Calculate price changes
        delta = prices.diff()
        
        # Separate gains and losses
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        
        # Calculate average gains and losses
        avg_gains = gains.rolling(window=period).mean()
        avg_losses = losses.rolling(window=period).mean()
        
        # Calculate RS and RSI
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def stochastic(self, market_data: MarketData, k_period: int = 14, d_period: int = 3) -> tuple[pd.Series, pd.Series]:
        """
        Stochastic Oscillator.
        
        Args:
            market_data: Market data
            k_period: Period for %K calculation
            d_period: Period for %D smoothing
            
        Returns:
            Tuple of (%K, %D) values
        """
        if len(market_data.data) < k_period:
            raise ValueError(f"Insufficient data for Stochastic({k_period})")
        
        high = market_data.data['high']
        low = market_data.data['low']
        close = market_data.data['close']
        
        # Calculate %K
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        k_percent = 100 * (close - lowest_low) / (highest_high - lowest_low)
        
        # Calculate %D (smoothed %K)
        d_percent = k_percent.rolling(window=d_period).mean()
        
        return k_percent, d_percent
    
    def macd(self, market_data: MarketData, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9, price_column: str = 'close') -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        MACD (Moving Average Convergence Divergence).
        
        Args:
            market_data: Market data
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line EMA period
            price_column: Price column to use
            
        Returns:
            Tuple of (MACD line, Signal line, Histogram)
        """
        if len(market_data.data) < slow_period:
            raise ValueError(f"Insufficient data for MACD({fast_period}, {slow_period})")
        
        prices = market_data.data[price_column]
        
        # Calculate EMAs
        ema_fast = prices.ewm(span=fast_period).mean()
        ema_slow = prices.ewm(span=slow_period).mean()
        
        # Calculate MACD line
        macd_line = ema_fast - ema_slow
        
        # Calculate signal line
        signal_line = macd_line.ewm(span=signal_period).mean()
        
        # Calculate histogram
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def williams_r(self, market_data: MarketData, period: int = 14) -> pd.Series:
        """
        Williams %R.
        
        Args:
            market_data: Market data
            period: Period for calculation
            
        Returns:
            Williams %R values
        """
        if len(market_data.data) < period:
            raise ValueError(f"Insufficient data for Williams %R({period})")
        
        high = market_data.data['high']
        low = market_data.data['low']
        close = market_data.data['close']
        
        # Calculate highest high and lowest low
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        
        # Calculate Williams %R
        williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
        
        return williams_r
    
    def cci(self, market_data: MarketData, period: int = 20, constant: float = 0.015) -> pd.Series:
        """
        Commodity Channel Index.
        
        Args:
            market_data: Market data
            period: Period for calculation
            constant: Constant factor (typically 0.015)
            
        Returns:
            CCI values
        """
        if len(market_data.data) < period:
            raise ValueError(f"Insufficient data for CCI({period})")
        
        # Calculate typical price
        typical_price = (market_data.data['high'] + market_data.data['low'] + market_data.data['close']) / 3
        
        # Calculate moving average of typical price
        sma_tp = typical_price.rolling(window=period).mean()
        
        # Calculate mean deviation
        mean_deviation = typical_price.rolling(window=period).apply(
            lambda x: np.mean(np.abs(x - x.mean())), raw=True
        )
        
        # Calculate CCI
        cci = (typical_price - sma_tp) / (constant * mean_deviation)
        
        return cci
    
    def momentum(self, market_data: MarketData, period: int = 10, price_column: str = 'close') -> pd.Series:
        """
        Price Momentum.
        
        Args:
            market_data: Market data
            period: Period for momentum calculation
            price_column: Price column to use
            
        Returns:
            Momentum values
        """
        if len(market_data.data) < period + 1:
            raise ValueError(f"Insufficient data for Momentum({period})")
        
        prices = market_data.data[price_column]
        
        # Calculate momentum as current price / price n periods ago
        momentum = prices / prices.shift(period)
        
        return momentum
    
    def roc(self, market_data: MarketData, period: int = 10, price_column: str = 'close') -> pd.Series:
        """
        Rate of Change.
        
        Args:
            market_data: Market data
            period: Period for ROC calculation
            price_column: Price column to use
            
        Returns:
            ROC values
        """
        if len(market_data.data) < period + 1:
            raise ValueError(f"Insufficient data for ROC({period})")
        
        prices = market_data.data[price_column]
        
        # Calculate ROC as percentage change
        roc = ((prices - prices.shift(period)) / prices.shift(period)) * 100
        
        return roc
