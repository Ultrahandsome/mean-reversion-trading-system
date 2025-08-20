"""Volatility indicators for the Mean Reversion Trading System."""

import pandas as pd
import numpy as np
from typing import Tuple

from ...data.models import MarketData


class VolatilityIndicators:
    """Collection of volatility indicators."""
    
    def historical_volatility(self, market_data: MarketData, period: int = 20, annualize: bool = True, price_column: str = 'close') -> pd.Series:
        """
        Historical Volatility.
        
        Args:
            market_data: Market data
            period: Period for volatility calculation
            annualize: Whether to annualize the volatility
            price_column: Price column to use
            
        Returns:
            Historical volatility values
        """
        if len(market_data.data) < period + 1:
            raise ValueError(f"Insufficient data for Historical Volatility({period})")
        
        prices = market_data.data[price_column]
        
        # Calculate log returns
        log_returns = np.log(prices / prices.shift(1))
        
        # Calculate rolling standard deviation
        volatility = log_returns.rolling(window=period).std()
        
        # Annualize if requested (assuming 252 trading days per year)
        if annualize:
            volatility = volatility * np.sqrt(252)
        
        return volatility
    
    def bollinger_bands(self, market_data: MarketData, period: int = 20, num_std: float = 2.0, price_column: str = 'close') -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Bollinger Bands.
        
        Args:
            market_data: Market data
            period: Period for moving average
            num_std: Number of standard deviations
            price_column: Price column to use
            
        Returns:
            Tuple of (upper_band, middle_band, lower_band)
        """
        if len(market_data.data) < period:
            raise ValueError(f"Insufficient data for Bollinger Bands({period})")
        
        prices = market_data.data[price_column]
        
        # Calculate middle band (SMA)
        middle_band = prices.rolling(window=period).mean()
        
        # Calculate standard deviation
        std = prices.rolling(window=period).std()
        
        # Calculate upper and lower bands
        upper_band = middle_band + (std * num_std)
        lower_band = middle_band - (std * num_std)
        
        return upper_band, middle_band, lower_band
    
    def atr(self, market_data: MarketData, period: int = 14) -> pd.Series:
        """
        Average True Range.
        
        Args:
            market_data: Market data
            period: Period for ATR calculation
            
        Returns:
            ATR values
        """
        if len(market_data.data) < period + 1:
            raise ValueError(f"Insufficient data for ATR({period})")
        
        high = market_data.data['high']
        low = market_data.data['low']
        close = market_data.data['close']
        
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate ATR as exponential moving average of True Range
        atr = true_range.ewm(span=period).mean()
        
        return atr
    
    def keltner_channels(self, market_data: MarketData, period: int = 20, multiplier: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Keltner Channels.
        
        Args:
            market_data: Market data
            period: Period for EMA and ATR
            multiplier: ATR multiplier
            
        Returns:
            Tuple of (upper_channel, middle_line, lower_channel)
        """
        if len(market_data.data) < period + 1:
            raise ValueError(f"Insufficient data for Keltner Channels({period})")
        
        # Calculate middle line (EMA of typical price)
        typical_price = (market_data.data['high'] + market_data.data['low'] + market_data.data['close']) / 3
        middle_line = typical_price.ewm(span=period).mean()
        
        # Calculate ATR
        atr = self.atr(market_data, period)
        
        # Calculate channels
        upper_channel = middle_line + (multiplier * atr)
        lower_channel = middle_line - (multiplier * atr)
        
        return upper_channel, middle_line, lower_channel
    
    def donchian_channels(self, market_data: MarketData, period: int = 20) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Donchian Channels.
        
        Args:
            market_data: Market data
            period: Period for channel calculation
            
        Returns:
            Tuple of (upper_channel, middle_line, lower_channel)
        """
        if len(market_data.data) < period:
            raise ValueError(f"Insufficient data for Donchian Channels({period})")
        
        high = market_data.data['high']
        low = market_data.data['low']
        
        # Calculate channels
        upper_channel = high.rolling(window=period).max()
        lower_channel = low.rolling(window=period).min()
        middle_line = (upper_channel + lower_channel) / 2
        
        return upper_channel, middle_line, lower_channel
    
    def volatility_ratio(self, market_data: MarketData, short_period: int = 10, long_period: int = 30, price_column: str = 'close') -> pd.Series:
        """
        Volatility Ratio (short-term vol / long-term vol).
        
        Args:
            market_data: Market data
            short_period: Short-term volatility period
            long_period: Long-term volatility period
            price_column: Price column to use
            
        Returns:
            Volatility ratio values
        """
        if len(market_data.data) < long_period + 1:
            raise ValueError(f"Insufficient data for Volatility Ratio({short_period}, {long_period})")
        
        prices = market_data.data[price_column]
        log_returns = np.log(prices / prices.shift(1))
        
        # Calculate short and long-term volatilities
        short_vol = log_returns.rolling(window=short_period).std()
        long_vol = log_returns.rolling(window=long_period).std()
        
        # Calculate ratio
        vol_ratio = short_vol / long_vol
        
        return vol_ratio
    
    def parkinson_volatility(self, market_data: MarketData, period: int = 20, annualize: bool = True) -> pd.Series:
        """
        Parkinson Volatility Estimator (uses high-low range).
        
        Args:
            market_data: Market data
            period: Period for volatility calculation
            annualize: Whether to annualize the volatility
            
        Returns:
            Parkinson volatility values
        """
        if len(market_data.data) < period:
            raise ValueError(f"Insufficient data for Parkinson Volatility({period})")
        
        high = market_data.data['high']
        low = market_data.data['low']
        
        # Calculate log(high/low)^2
        log_hl_squared = (np.log(high / low)) ** 2
        
        # Calculate rolling mean and apply Parkinson formula
        parkinson_vol = np.sqrt(log_hl_squared.rolling(window=period).mean() / (4 * np.log(2)))
        
        # Annualize if requested
        if annualize:
            parkinson_vol = parkinson_vol * np.sqrt(252)
        
        return parkinson_vol
    
    def garman_klass_volatility(self, market_data: MarketData, period: int = 20, annualize: bool = True) -> pd.Series:
        """
        Garman-Klass Volatility Estimator.
        
        Args:
            market_data: Market data
            period: Period for volatility calculation
            annualize: Whether to annualize the volatility
            
        Returns:
            Garman-Klass volatility values
        """
        if len(market_data.data) < period + 1:
            raise ValueError(f"Insufficient data for Garman-Klass Volatility({period})")
        
        high = market_data.data['high']
        low = market_data.data['low']
        close = market_data.data['close']
        open_price = market_data.data['open']
        
        # Calculate components
        log_hl = np.log(high / low)
        log_co = np.log(close / open_price)
        
        # Garman-Klass formula
        gk_component = 0.5 * (log_hl ** 2) - (2 * np.log(2) - 1) * (log_co ** 2)
        
        # Calculate rolling mean
        gk_vol = np.sqrt(gk_component.rolling(window=period).mean())
        
        # Annualize if requested
        if annualize:
            gk_vol = gk_vol * np.sqrt(252)
        
        return gk_vol
    
    def vix_like_index(self, market_data: MarketData, period: int = 30, price_column: str = 'close') -> pd.Series:
        """
        VIX-like volatility index.
        
        Args:
            market_data: Market data
            period: Period for volatility calculation
            price_column: Price column to use
            
        Returns:
            VIX-like index values
        """
        if len(market_data.data) < period + 1:
            raise ValueError(f"Insufficient data for VIX-like index({period})")
        
        prices = market_data.data[price_column]
        
        # Calculate log returns
        log_returns = np.log(prices / prices.shift(1))
        
        # Calculate rolling volatility
        rolling_vol = log_returns.rolling(window=period).std()
        
        # Annualize and convert to percentage (like VIX)
        vix_like = rolling_vol * np.sqrt(252) * 100
        
        return vix_like
