"""Mean reversion signal generators for the Mean Reversion Trading System."""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime

from .base import SignalGenerator, Signal
from ..indicators.mean_reversion import MeanReversionIndicators
from ..indicators.moving_averages import MovingAverages
from ..indicators.volatility import VolatilityIndicators
from ...data.models import MarketData
from ...utils.logger import get_logger

logger = get_logger(__name__)


class MeanReversionSignals(SignalGenerator):
    """Mean reversion signal generator."""
    
    def __init__(self):
        """Initialize mean reversion signal generator."""
        super().__init__("Mean Reversion Signals")
        
        self.mr_indicators = MeanReversionIndicators()
        self.ma_indicators = MovingAverages()
        self.vol_indicators = VolatilityIndicators()
        
        # Default parameters
        self.parameters = {
            'zscore_period': 20,
            'zscore_entry_threshold': 2.0,
            'zscore_exit_threshold': 0.5,
            'bollinger_period': 20,
            'bollinger_std': 2.0,
            'bollinger_entry_threshold': 0.95,
            'bollinger_exit_threshold': 0.5,
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'rsi_exit': 50,
            'min_volume': 1000,
            'volatility_filter': True,
            'volatility_period': 20,
            'volatility_threshold': 0.02
        }
    
    def generate_signals(self, market_data: MarketData, **kwargs) -> List[Signal]:
        """
        Generate mean reversion signals.
        
        Args:
            market_data: Market data to analyze
            **kwargs: Additional parameters
            
        Returns:
            List of generated signals
        """
        # Update parameters
        params = self.parameters.copy()
        params.update(kwargs)
        
        # Validate data
        min_periods = max(params['zscore_period'], params['bollinger_period'], params['rsi_period'])
        self.validate_data(market_data, min_periods)
        
        signals = []
        
        # Generate Z-Score signals
        zscore_signals = self._generate_zscore_signals(market_data, params)
        signals.extend(zscore_signals)
        
        # Generate Bollinger Band signals
        bb_signals = self._generate_bollinger_signals(market_data, params)
        signals.extend(bb_signals)
        
        # Generate RSI signals
        rsi_signals = self._generate_rsi_signals(market_data, params)
        signals.extend(rsi_signals)
        
        # Apply filters
        if params.get('volatility_filter', True):
            signals = self._apply_volatility_filter(signals, market_data, params)
        
        if params.get('min_volume', 0) > 0:
            signals = self._apply_volume_filter(signals, market_data, params)
        
        # Sort by timestamp
        signals.sort(key=lambda x: x.timestamp)
        
        logger.info(f"Generated {len(signals)} mean reversion signals for {market_data.symbol}")
        
        return signals
    
    def _generate_zscore_signals(self, market_data: MarketData, params: Dict[str, Any]) -> List[Signal]:
        """Generate Z-Score based signals."""
        period = params['zscore_period']
        entry_threshold = params['zscore_entry_threshold']
        exit_threshold = params['zscore_exit_threshold']
        
        # Calculate Z-Score
        zscore = self.mr_indicators.zscore(market_data, period)
        prices = market_data.data['close']
        
        signals = []
        position = 0  # 0: no position, 1: long, -1: short
        
        for i in range(len(zscore)):
            if pd.isna(zscore.iloc[i]):
                continue
            
            timestamp = zscore.index[i]
            price = prices.iloc[i]
            z_value = zscore.iloc[i]
            
            # Entry signals
            if position == 0:
                if z_value <= -entry_threshold:
                    # Oversold - Long entry
                    signal = Signal(
                        timestamp=timestamp,
                        symbol=market_data.symbol,
                        signal_type='entry',
                        direction=1,
                        strength=min(abs(z_value) / entry_threshold, 1.0),
                        price=price,
                        confidence=0.7,
                        metadata={'zscore': z_value, 'indicator': 'zscore'}
                    )
                    signals.append(signal)
                    position = 1
                
                elif z_value >= entry_threshold:
                    # Overbought - Short entry
                    signal = Signal(
                        timestamp=timestamp,
                        symbol=market_data.symbol,
                        signal_type='entry',
                        direction=-1,
                        strength=min(abs(z_value) / entry_threshold, 1.0),
                        price=price,
                        confidence=0.7,
                        metadata={'zscore': z_value, 'indicator': 'zscore'}
                    )
                    signals.append(signal)
                    position = -1
            
            # Exit signals
            elif position == 1 and z_value >= -exit_threshold:
                # Exit long position
                signal = Signal(
                    timestamp=timestamp,
                    symbol=market_data.symbol,
                    signal_type='exit',
                    direction=0,
                    strength=0.8,
                    price=price,
                    confidence=0.8,
                    metadata={'zscore': z_value, 'indicator': 'zscore'}
                )
                signals.append(signal)
                position = 0
            
            elif position == -1 and z_value <= exit_threshold:
                # Exit short position
                signal = Signal(
                    timestamp=timestamp,
                    symbol=market_data.symbol,
                    signal_type='exit',
                    direction=0,
                    strength=0.8,
                    price=price,
                    confidence=0.8,
                    metadata={'zscore': z_value, 'indicator': 'zscore'}
                )
                signals.append(signal)
                position = 0
        
        return signals
    
    def _generate_bollinger_signals(self, market_data: MarketData, params: Dict[str, Any]) -> List[Signal]:
        """Generate Bollinger Band based signals."""
        period = params['bollinger_period']
        num_std = params['bollinger_std']
        entry_threshold = params['bollinger_entry_threshold']
        exit_threshold = params['bollinger_exit_threshold']
        
        # Calculate Bollinger position
        bb_position = self.mr_indicators.bollinger_position(market_data, period, num_std)
        prices = market_data.data['close']
        
        signals = []
        position = 0
        
        for i in range(len(bb_position)):
            if pd.isna(bb_position.iloc[i]):
                continue
            
            timestamp = bb_position.index[i]
            price = prices.iloc[i]
            bb_pos = bb_position.iloc[i]
            
            # Entry signals
            if position == 0:
                if bb_pos <= (1 - entry_threshold):
                    # Near lower band - Long entry
                    signal = Signal(
                        timestamp=timestamp,
                        symbol=market_data.symbol,
                        signal_type='entry',
                        direction=1,
                        strength=1 - bb_pos,
                        price=price,
                        confidence=0.6,
                        metadata={'bb_position': bb_pos, 'indicator': 'bollinger'}
                    )
                    signals.append(signal)
                    position = 1
                
                elif bb_pos >= entry_threshold:
                    # Near upper band - Short entry
                    signal = Signal(
                        timestamp=timestamp,
                        symbol=market_data.symbol,
                        signal_type='entry',
                        direction=-1,
                        strength=bb_pos,
                        price=price,
                        confidence=0.6,
                        metadata={'bb_position': bb_pos, 'indicator': 'bollinger'}
                    )
                    signals.append(signal)
                    position = -1
            
            # Exit signals
            elif position == 1 and bb_pos >= exit_threshold:
                # Exit long position
                signal = Signal(
                    timestamp=timestamp,
                    symbol=market_data.symbol,
                    signal_type='exit',
                    direction=0,
                    strength=0.7,
                    price=price,
                    confidence=0.7,
                    metadata={'bb_position': bb_pos, 'indicator': 'bollinger'}
                )
                signals.append(signal)
                position = 0
            
            elif position == -1 and bb_pos <= (1 - exit_threshold):
                # Exit short position
                signal = Signal(
                    timestamp=timestamp,
                    symbol=market_data.symbol,
                    signal_type='exit',
                    direction=0,
                    strength=0.7,
                    price=price,
                    confidence=0.7,
                    metadata={'bb_position': bb_pos, 'indicator': 'bollinger'}
                )
                signals.append(signal)
                position = 0
        
        return signals
    
    def _generate_rsi_signals(self, market_data: MarketData, params: Dict[str, Any]) -> List[Signal]:
        """Generate RSI based signals."""
        from ..indicators.momentum import MomentumIndicators
        
        period = params['rsi_period']
        oversold = params['rsi_oversold']
        overbought = params['rsi_overbought']
        exit_level = params['rsi_exit']
        
        momentum = MomentumIndicators()
        rsi = momentum.rsi(market_data, period)
        prices = market_data.data['close']
        
        signals = []
        position = 0
        
        for i in range(len(rsi)):
            if pd.isna(rsi.iloc[i]):
                continue
            
            timestamp = rsi.index[i]
            price = prices.iloc[i]
            rsi_value = rsi.iloc[i]
            
            # Entry signals
            if position == 0:
                if rsi_value <= oversold:
                    # Oversold - Long entry
                    signal = Signal(
                        timestamp=timestamp,
                        symbol=market_data.symbol,
                        signal_type='entry',
                        direction=1,
                        strength=(oversold - rsi_value) / oversold,
                        price=price,
                        confidence=0.5,
                        metadata={'rsi': rsi_value, 'indicator': 'rsi'}
                    )
                    signals.append(signal)
                    position = 1
                
                elif rsi_value >= overbought:
                    # Overbought - Short entry
                    signal = Signal(
                        timestamp=timestamp,
                        symbol=market_data.symbol,
                        signal_type='entry',
                        direction=-1,
                        strength=(rsi_value - overbought) / (100 - overbought),
                        price=price,
                        confidence=0.5,
                        metadata={'rsi': rsi_value, 'indicator': 'rsi'}
                    )
                    signals.append(signal)
                    position = -1
            
            # Exit signals
            elif position == 1 and rsi_value >= exit_level:
                # Exit long position
                signal = Signal(
                    timestamp=timestamp,
                    symbol=market_data.symbol,
                    signal_type='exit',
                    direction=0,
                    strength=0.6,
                    price=price,
                    confidence=0.6,
                    metadata={'rsi': rsi_value, 'indicator': 'rsi'}
                )
                signals.append(signal)
                position = 0
            
            elif position == -1 and rsi_value <= exit_level:
                # Exit short position
                signal = Signal(
                    timestamp=timestamp,
                    symbol=market_data.symbol,
                    signal_type='exit',
                    direction=0,
                    strength=0.6,
                    price=price,
                    confidence=0.6,
                    metadata={'rsi': rsi_value, 'indicator': 'rsi'}
                )
                signals.append(signal)
                position = 0
        
        return signals
    
    def _apply_volatility_filter(self, signals: List[Signal], market_data: MarketData, params: Dict[str, Any]) -> List[Signal]:
        """Filter signals based on volatility."""
        period = params.get('volatility_period', 20)
        threshold = params.get('volatility_threshold', 0.02)
        
        # Calculate volatility
        returns = market_data.data['close'].pct_change()
        volatility = returns.rolling(window=period).std()
        
        filtered_signals = []
        
        for signal in signals:
            # Find volatility at signal timestamp
            try:
                vol_value = volatility.loc[signal.timestamp]
                if not pd.isna(vol_value) and vol_value >= threshold:
                    filtered_signals.append(signal)
            except KeyError:
                # If timestamp not found, include signal
                filtered_signals.append(signal)
        
        return filtered_signals
    
    def _apply_volume_filter(self, signals: List[Signal], market_data: MarketData, params: Dict[str, Any]) -> List[Signal]:
        """Filter signals based on volume."""
        min_volume = params.get('min_volume', 1000)
        
        filtered_signals = []
        
        for signal in signals:
            try:
                volume = market_data.data.loc[signal.timestamp, 'volume']
                if volume >= min_volume:
                    filtered_signals.append(signal)
            except KeyError:
                # If timestamp not found, include signal
                filtered_signals.append(signal)
        
        return filtered_signals
