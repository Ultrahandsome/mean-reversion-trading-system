"""Mean reversion indicators for the Mean Reversion Trading System."""

import pandas as pd
import numpy as np
from typing import Optional, Tuple
from scipy import stats

# Removed relative imports for standalone testing
# from .base import TechnicalIndicator
# from ...data.models import MarketData


class MeanReversionIndicators:
    """Collection of mean reversion indicators."""
    
    def zscore(self, market_data, period: int, price_column: str = 'close') -> pd.Series:
        """
        Z-Score indicator for mean reversion.
        
        Args:
            market_data: Market data
            period: Lookback period for mean and std calculation
            price_column: Price column to use
            
        Returns:
            Z-score values
        """
        if len(market_data.data) < period:
            raise ValueError(f"Insufficient data for Z-Score({period})")
        
        prices = market_data.data[price_column]
        
        # Calculate rolling mean and standard deviation
        rolling_mean = prices.rolling(window=period).mean()
        rolling_std = prices.rolling(window=period).std()
        
        # Calculate z-score
        zscore = (prices - rolling_mean) / rolling_std
        
        return zscore
    
    def bollinger_position(self, market_data, period: int = 20, num_std: float = 2.0, price_column: str = 'close') -> pd.Series:
        """
        Bollinger Band Position (0 = lower band, 1 = upper band).
        
        Args:
            market_data: Market data
            period: Period for moving average
            num_std: Number of standard deviations
            price_column: Price column to use
            
        Returns:
            Bollinger position values
        """
        if len(market_data.data) < period:
            raise ValueError(f"Insufficient data for Bollinger Position({period})")
        
        prices = market_data.data[price_column]
        
        # Calculate Bollinger Bands
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        
        # Calculate position within bands
        position = (prices - lower_band) / (upper_band - lower_band)
        
        return position
    
    def mean_reversion_ratio(self, market_data, period: int, price_column: str = 'close') -> pd.Series:
        """
        Mean Reversion Ratio: current price / moving average.
        
        Args:
            market_data: Market data
            period: Period for moving average
            price_column: Price column to use
            
        Returns:
            Mean reversion ratio values
        """
        if len(market_data.data) < period:
            raise ValueError(f"Insufficient data for Mean Reversion Ratio({period})")
        
        prices = market_data.data[price_column]
        sma = prices.rolling(window=period).mean()
        
        return prices / sma
    
    def distance_from_mean(self, market_data, period: int, normalize: bool = True, price_column: str = 'close') -> pd.Series:
        """
        Distance from moving average.
        
        Args:
            market_data: Market data
            period: Period for moving average
            normalize: Whether to normalize by standard deviation
            price_column: Price column to use
            
        Returns:
            Distance from mean values
        """
        if len(market_data.data) < period:
            raise ValueError(f"Insufficient data for Distance from Mean({period})")
        
        prices = market_data.data[price_column]
        sma = prices.rolling(window=period).mean()
        
        distance = prices - sma
        
        if normalize:
            std = prices.rolling(window=period).std()
            distance = distance / std
        
        return distance
    
    def hurst_exponent(self, market_data, period: int, price_column: str = 'close') -> pd.Series:
        """
        Rolling Hurst Exponent for mean reversion detection.
        H < 0.5 indicates mean reversion, H > 0.5 indicates trending.
        
        Args:
            market_data: Market data
            period: Rolling window period
            price_column: Price column to use
            
        Returns:
            Hurst exponent values
        """
        if len(market_data.data) < period * 2:
            raise ValueError(f"Insufficient data for Hurst Exponent({period})")
        
        prices = market_data.data[price_column]
        log_prices = np.log(prices)
        
        def calculate_hurst(series):
            """Calculate Hurst exponent for a series."""
            if len(series) < 10:
                return np.nan
            
            # Calculate log returns
            returns = np.diff(np.log(series))
            
            # Calculate R/S statistic for different lags
            lags = range(2, min(len(returns) // 2, 20))
            rs_values = []
            
            for lag in lags:
                # Split returns into chunks
                n_chunks = len(returns) // lag
                if n_chunks < 2:
                    continue
                
                chunks = [returns[i*lag:(i+1)*lag] for i in range(n_chunks)]
                rs_chunk = []
                
                for chunk in chunks:
                    if len(chunk) == 0:
                        continue
                    
                    # Calculate mean
                    mean_chunk = np.mean(chunk)
                    
                    # Calculate cumulative deviations
                    deviations = np.cumsum(chunk - mean_chunk)
                    
                    # Calculate range and standard deviation
                    R = np.max(deviations) - np.min(deviations)
                    S = np.std(chunk)
                    
                    if S > 0:
                        rs_chunk.append(R / S)
                
                if rs_chunk:
                    rs_values.append((lag, np.mean(rs_chunk)))
            
            if len(rs_values) < 3:
                return np.nan
            
            # Linear regression to find Hurst exponent
            lags_log = [np.log(x[0]) for x in rs_values]
            rs_log = [np.log(x[1]) for x in rs_values if x[1] > 0]
            
            if len(lags_log) != len(rs_log) or len(rs_log) < 3:
                return np.nan
            
            try:
                slope, _, _, _, _ = stats.linregress(lags_log, rs_log)
                return slope
            except:
                return np.nan
        
        # Calculate rolling Hurst exponent
        hurst_values = log_prices.rolling(window=period).apply(calculate_hurst, raw=True)
        
        return hurst_values
    
    def adf_test_statistic(self, market_data, period: int, price_column: str = 'close') -> pd.Series:
        """
        Rolling Augmented Dickey-Fuller test statistic for stationarity.
        More negative values indicate stronger mean reversion.
        
        Args:
            market_data: Market data
            period: Rolling window period
            price_column: Price column to use
            
        Returns:
            ADF test statistic values
        """
        if len(market_data.data) < period:
            raise ValueError(f"Insufficient data for ADF test({period})")
        
        prices = market_data.data[price_column]
        
        def calculate_adf(series):
            """Calculate ADF test statistic."""
            if len(series) < 10:
                return np.nan
            
            try:
                from statsmodels.tsa.stattools import adfuller
                result = adfuller(series, maxlag=1, regression='c', autolag=None)
                return result[0]  # ADF statistic
            except:
                return np.nan
        
        # Calculate rolling ADF statistic
        adf_values = prices.rolling(window=period).apply(calculate_adf, raw=True)
        
        return adf_values
    
    def variance_ratio(self, market_data, period: int, lag: int = 2, price_column: str = 'close') -> pd.Series:
        """
        Rolling Variance Ratio test for mean reversion.
        VR < 1 indicates mean reversion, VR > 1 indicates momentum.
        
        Args:
            market_data: Market data
            period: Rolling window period
            lag: Lag for variance ratio calculation
            price_column: Price column to use
            
        Returns:
            Variance ratio values
        """
        if len(market_data.data) < period:
            raise ValueError(f"Insufficient data for Variance Ratio({period})")
        
        prices = market_data.data[price_column]
        log_prices = np.log(prices)
        
        def calculate_vr(series, k=lag):
            """Calculate variance ratio."""
            if len(series) < k * 2:
                return np.nan
            
            # Calculate returns
            returns = np.diff(series)
            
            if len(returns) < k:
                return np.nan
            
            # Calculate k-period returns
            k_returns = []
            for i in range(k, len(returns) + 1):
                k_returns.append(sum(returns[i-k:i]))
            
            if len(k_returns) < 2:
                return np.nan
            
            # Calculate variances
            var_1 = np.var(returns, ddof=1) if len(returns) > 1 else np.nan
            var_k = np.var(k_returns, ddof=1) / k if len(k_returns) > 1 else np.nan
            
            if var_1 > 0 and not np.isnan(var_k):
                return var_k / var_1
            else:
                return np.nan
        
        # Calculate rolling variance ratio
        vr_values = log_prices.rolling(window=period).apply(lambda x: calculate_vr(x, lag), raw=True)
        
        return vr_values
    
    def mean_reversion_speed(self, market_data, period: int, price_column: str = 'close') -> pd.Series:
        """
        Estimate mean reversion speed using AR(1) model.
        
        Args:
            market_data: Market data
            period: Rolling window period
            price_column: Price column to use
            
        Returns:
            Mean reversion speed values
        """
        if len(market_data.data) < period:
            raise ValueError(f"Insufficient data for Mean Reversion Speed({period})")
        
        prices = market_data.data[price_column]
        
        def calculate_ar1_coeff(series):
            """Calculate AR(1) coefficient."""
            if len(series) < 10:
                return np.nan
            
            # Calculate returns
            returns = np.diff(np.log(series))
            
            if len(returns) < 2:
                return np.nan
            
            # AR(1) regression: r_t = alpha + beta * r_{t-1} + error
            y = returns[1:]
            x = returns[:-1]
            
            if len(y) != len(x) or len(y) < 2:
                return np.nan
            
            try:
                # Simple linear regression
                coeff = np.corrcoef(x, y)[0, 1] * (np.std(y) / np.std(x))
                return coeff
            except:
                return np.nan
        
        # Calculate rolling AR(1) coefficient
        ar1_coeff = prices.rolling(window=period).apply(calculate_ar1_coeff, raw=True)
        
        # Mean reversion speed = -ln(1 + AR1_coeff)
        # More negative AR1 coefficient indicates faster mean reversion
        speed = -np.log(1 + ar1_coeff.clip(lower=-0.99))
        
        return speed
