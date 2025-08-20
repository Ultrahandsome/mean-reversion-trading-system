"""Z-Score based mean reversion strategy."""

from typing import Dict, List, Any
import pandas as pd
import numpy as np

# Fixed imports for standalone testing
# from ..base import BaseStrategy
# from ...data.models import MarketData
# from ...factors.signals.base import Signal
# from ...factors.indicators.mean_reversion import MeanReversionIndicators
# from ...utils.logger import get_logger

# Simple logger replacement
class SimpleLogger:
    def info(self, msg): print(f"INFO: {msg}")
    def warning(self, msg): print(f"WARNING: {msg}")
    def debug(self, msg): pass
    def error(self, msg): print(f"ERROR: {msg}")

logger = SimpleLogger()

# Simple base strategy class
class BaseStrategy:
    def __init__(self, name: str):
        self.name = name
        self.parameters = {}

    def set_parameters(self, **kwargs):
        self.parameters.update(kwargs)

    def get_parameters(self):
        return self.parameters.copy()

# Simple Signal class
class Signal:
    def __init__(self, timestamp, symbol, signal_type, direction, strength, price, confidence=0.5, metadata=None):
        self.timestamp = timestamp
        self.symbol = symbol
        self.signal_type = signal_type
        self.direction = direction
        self.strength = strength
        self.price = price
        self.confidence = confidence
        self.metadata = metadata or {}


class ZScoreMeanReversion(BaseStrategy):
    """Z-Score based mean reversion strategy."""
    
    def __init__(
        self,
        lookback_period: int = 20,
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.5,
        min_volume: float = 1000000,
        volatility_filter: bool = True
    ):
        """
        Initialize Z-Score mean reversion strategy.
        
        Args:
            lookback_period: Period for mean and std calculation
            entry_threshold: Z-score threshold for entry signals
            exit_threshold: Z-score threshold for exit signals
            min_volume: Minimum daily volume filter
            volatility_filter: Whether to apply volatility filter
        """
        super().__init__("Z-Score Mean Reversion")
        
        self.parameters = {
            'lookback_period': lookback_period,
            'entry_threshold': entry_threshold,
            'exit_threshold': exit_threshold,
            'min_volume': min_volume,
            'volatility_filter': volatility_filter
        }
        
        # Import MeanReversionIndicators dynamically
        import importlib.util
        spec = importlib.util.spec_from_file_location("mr_indicators", "src/factors/indicators/mean_reversion.py")
        mr_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mr_module)

        self.mr_indicators = mr_module.MeanReversionIndicators()
        self.positions = {}  # Track current positions per symbol

        logger.info(f"Initialized {self.name} strategy")

    def validate_market_data(self, market_data, min_periods=1):
        """Simple validation for market data."""
        if not market_data:
            raise ValueError("Market data cannot be empty")

        for symbol, data in market_data.items():
            if len(data.data) < min_periods:
                raise ValueError(f"Insufficient data for {symbol}. Required: {min_periods}, Available: {len(data.data)}")

    def generate_signals(self, market_data):
        """
        Generate Z-Score based mean reversion signals.
        
        Args:
            market_data: Dictionary of symbol -> MarketData
            
        Returns:
            List of trading signals
        """
        # Validate data
        min_periods = self.parameters['lookback_period'] + 10
        self.validate_market_data(market_data, min_periods)
        
        all_signals = []
        
        for symbol, data in market_data.items():
            try:
                signals = self._generate_symbol_signals(symbol, data)
                all_signals.extend(signals)
            except Exception as e:
                logger.error(f"Error generating signals for {symbol}: {e}")
                continue
        
        logger.info(f"Generated {len(all_signals)} Z-Score signals")
        return all_signals
    
    def _generate_symbol_signals(self, symbol: str, market_data):
        """Generate signals for a single symbol."""
        lookback_period = self.parameters['lookback_period']
        entry_threshold = self.parameters['entry_threshold']
        exit_threshold = self.parameters['exit_threshold']
        min_volume = self.parameters['min_volume']
        
        # Calculate Z-Score
        zscore = self.mr_indicators.zscore(market_data, lookback_period)
        prices = market_data.data['close']
        volumes = market_data.data['volume']
        
        # Apply volume filter
        if min_volume > 0:
            volume_mask = volumes >= min_volume
            zscore = zscore[volume_mask]
            prices = prices[volume_mask]
        
        # Apply volatility filter if enabled
        if self.parameters['volatility_filter']:
            zscore, prices = self._apply_volatility_filter(zscore, prices, market_data)
        
        signals = []
        current_position = self.positions.get(symbol, 0)  # 0: no position, 1: long, -1: short
        
        for i in range(len(zscore)):
            if pd.isna(zscore.iloc[i]):
                continue
            
            timestamp = zscore.index[i]
            price = prices.iloc[i]
            z_value = zscore.iloc[i]
            
            # Entry signals
            if current_position == 0:
                if z_value <= -entry_threshold:
                    # Oversold - Long entry
                    signal = Signal(
                        timestamp=timestamp,
                        symbol=symbol,
                        signal_type='entry',
                        direction=1,
                        strength=min(abs(z_value) / entry_threshold, 1.0),
                        price=price,
                        confidence=self._calculate_confidence(z_value, market_data, i),
                        metadata={
                            'zscore': z_value,
                            'indicator': 'zscore',
                            'strategy': self.name
                        }
                    )
                    signals.append(signal)
                    current_position = 1
                
                elif z_value >= entry_threshold:
                    # Overbought - Short entry
                    signal = Signal(
                        timestamp=timestamp,
                        symbol=symbol,
                        signal_type='entry',
                        direction=-1,
                        strength=min(abs(z_value) / entry_threshold, 1.0),
                        price=price,
                        confidence=self._calculate_confidence(z_value, market_data, i),
                        metadata={
                            'zscore': z_value,
                            'indicator': 'zscore',
                            'strategy': self.name
                        }
                    )
                    signals.append(signal)
                    current_position = -1
            
            # Exit signals
            elif current_position == 1 and z_value >= -exit_threshold:
                # Exit long position
                signal = Signal(
                    timestamp=timestamp,
                    symbol=symbol,
                    signal_type='exit',
                    direction=0,
                    strength=0.8,
                    price=price,
                    confidence=0.8,
                    metadata={
                        'zscore': z_value,
                        'indicator': 'zscore',
                        'strategy': self.name
                    }
                )
                signals.append(signal)
                current_position = 0
            
            elif current_position == -1 and z_value <= exit_threshold:
                # Exit short position
                signal = Signal(
                    timestamp=timestamp,
                    symbol=symbol,
                    signal_type='exit',
                    direction=0,
                    strength=0.8,
                    price=price,
                    confidence=0.8,
                    metadata={
                        'zscore': z_value,
                        'indicator': 'zscore',
                        'strategy': self.name
                    }
                )
                signals.append(signal)
                current_position = 0
        
        # Update position tracking
        self.positions[symbol] = current_position
        
        return signals
    
    def _apply_volatility_filter(
        self,
        zscore: pd.Series,
        prices: pd.Series,
        market_data
    ):
        """Apply volatility filter to signals."""
        # Calculate rolling volatility
        returns = market_data.data['close'].pct_change()
        volatility = returns.rolling(window=20).std()
        
        # Filter for periods with sufficient volatility
        min_volatility = 0.01  # 1% daily volatility threshold
        vol_mask = volatility >= min_volatility
        
        # Align with zscore index
        aligned_vol_mask = vol_mask.reindex(zscore.index, fill_value=False)
        
        filtered_zscore = zscore[aligned_vol_mask]
        filtered_prices = prices[aligned_vol_mask]
        
        return filtered_zscore, filtered_prices
    
    def _calculate_confidence(
        self,
        zscore: float,
        market_data,
        current_index: int
    ) -> float:
        """
        Calculate signal confidence based on multiple factors.
        
        Args:
            zscore: Current Z-score value
            market_data: Market data
            current_index: Current data index
            
        Returns:
            Confidence score (0-1)
        """
        confidence = 0.5  # Base confidence
        
        # Increase confidence for extreme Z-scores
        if abs(zscore) > 2.5:
            confidence += 0.2
        elif abs(zscore) > 3.0:
            confidence += 0.3
        
        # Check trend consistency
        if current_index >= 5:
            recent_prices = market_data.data['close'].iloc[current_index-5:current_index]
            if len(recent_prices) >= 2:
                trend = recent_prices.iloc[-1] - recent_prices.iloc[0]
                
                # Higher confidence if price trend aligns with mean reversion signal
                if (zscore < -2 and trend < 0) or (zscore > 2 and trend > 0):
                    confidence += 0.1
        
        # Volume confirmation
        if current_index >= 10:
            recent_volume = market_data.data['volume'].iloc[current_index-10:current_index]
            avg_volume = recent_volume.mean()
            current_volume = market_data.data['volume'].iloc[current_index]
            
            if current_volume > avg_volume * 1.2:  # Above average volume
                confidence += 0.1
        
        return min(confidence, 1.0)
    
    def get_strategy_description(self) -> str:
        """Get strategy description."""
        return (
            f"Z-Score Mean Reversion Strategy:\n"
            f"- Lookback Period: {self.parameters['lookback_period']} days\n"
            f"- Entry Threshold: ±{self.parameters['entry_threshold']}\n"
            f"- Exit Threshold: ±{self.parameters['exit_threshold']}\n"
            f"- Min Volume: {self.parameters['min_volume']:,}\n"
            f"- Volatility Filter: {self.parameters['volatility_filter']}\n\n"
            f"Logic:\n"
            f"- Long when Z-score <= -{self.parameters['entry_threshold']} (oversold)\n"
            f"- Short when Z-score >= {self.parameters['entry_threshold']} (overbought)\n"
            f"- Exit when Z-score returns to ±{self.parameters['exit_threshold']}"
        )
    
    def reset_positions(self) -> None:
        """Reset position tracking (useful for backtesting)."""
        self.positions.clear()
    
    def get_current_positions(self) -> Dict[str, int]:
        """Get current position tracking."""
        return self.positions.copy()
