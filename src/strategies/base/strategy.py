"""Base strategy class for the Mean Reversion Trading System."""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import pandas as pd

from ...data.models import MarketData
from ...factors.signals.base import Signal, SignalGenerator
from ...utils.logger import get_logger

logger = get_logger(__name__)


class BaseStrategy(ABC):
    """Abstract base class for trading strategies."""
    
    def __init__(self, name: str):
        """
        Initialize strategy.
        
        Args:
            name: Strategy name
        """
        self.name = name
        self.parameters = {}
        self.signal_generators: List[SignalGenerator] = []
        
        logger.info(f"Initialized strategy: {name}")
    
    @abstractmethod
    def generate_signals(self, market_data: Dict[str, MarketData]) -> List[Signal]:
        """
        Generate trading signals from market data.
        
        Args:
            market_data: Dictionary of symbol -> MarketData
            
        Returns:
            List of trading signals
        """
        pass
    
    def set_parameters(self, **kwargs) -> None:
        """Set strategy parameters."""
        self.parameters.update(kwargs)
        logger.debug(f"Updated parameters for {self.name}: {kwargs}")
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get strategy parameters."""
        return self.parameters.copy()
    
    def add_signal_generator(self, generator: SignalGenerator) -> None:
        """Add a signal generator to the strategy."""
        self.signal_generators.append(generator)
        logger.debug(f"Added signal generator {generator.name} to {self.name}")
    
    def validate_market_data(self, market_data: Dict[str, MarketData], min_periods: int = 1) -> None:
        """
        Validate market data for strategy requirements.
        
        Args:
            market_data: Market data dictionary
            min_periods: Minimum number of periods required
        """
        if not market_data:
            raise ValueError("Market data cannot be empty")
        
        for symbol, data in market_data.items():
            if len(data.data) < min_periods:
                raise ValueError(
                    f"Insufficient data for {symbol}. "
                    f"Required: {min_periods}, Available: {len(data.data)}"
                )
    
    def filter_signals(
        self,
        signals: List[Signal],
        min_strength: float = 0.0,
        min_confidence: float = 0.0
    ) -> List[Signal]:
        """
        Filter signals based on strength and confidence.
        
        Args:
            signals: List of signals to filter
            min_strength: Minimum signal strength
            min_confidence: Minimum confidence level
            
        Returns:
            Filtered signals
        """
        filtered = []
        
        for signal in signals:
            if signal.strength >= min_strength and signal.confidence >= min_confidence:
                filtered.append(signal)
        
        logger.debug(f"Filtered {len(signals)} signals to {len(filtered)}")
        return filtered
    
    def combine_signals(
        self,
        signal_lists: List[List[Signal]],
        method: str = 'union'
    ) -> List[Signal]:
        """
        Combine signals from multiple generators.
        
        Args:
            signal_lists: List of signal lists
            method: Combination method ('union', 'intersection', 'weighted')
            
        Returns:
            Combined signals
        """
        if not signal_lists:
            return []
        
        if len(signal_lists) == 1:
            return signal_lists[0]
        
        if method == 'union':
            # Simple union of all signals
            combined = []
            for signal_list in signal_lists:
                combined.extend(signal_list)
            
            # Sort by timestamp
            combined.sort(key=lambda x: x.timestamp)
            return combined
        
        elif method == 'intersection':
            # Only signals that appear in all lists
            if len(signal_lists) < 2:
                return signal_lists[0] if signal_lists else []
            
            # Create sets of (timestamp, symbol) tuples
            signal_sets = []
            signal_maps = []
            
            for signal_list in signal_lists:
                signal_set = set()
                signal_map = {}
                
                for signal in signal_list:
                    key = (signal.timestamp, signal.symbol)
                    signal_set.add(key)
                    signal_map[key] = signal
                
                signal_sets.append(signal_set)
                signal_maps.append(signal_map)
            
            # Find intersection
            common_keys = signal_sets[0]
            for signal_set in signal_sets[1:]:
                common_keys = common_keys.intersection(signal_set)
            
            # Create combined signals
            combined = []
            for key in common_keys:
                # Use signal from first list as base
                base_signal = signal_maps[0][key]
                
                # Average strength and confidence
                total_strength = sum(signal_maps[i][key].strength for i in range(len(signal_maps)))
                total_confidence = sum(signal_maps[i][key].confidence for i in range(len(signal_maps)))
                
                combined_signal = Signal(
                    timestamp=base_signal.timestamp,
                    symbol=base_signal.symbol,
                    signal_type=base_signal.signal_type,
                    direction=base_signal.direction,
                    strength=total_strength / len(signal_maps),
                    price=base_signal.price,
                    confidence=total_confidence / len(signal_maps),
                    metadata={'combined_from': len(signal_maps)}
                )
                
                combined.append(combined_signal)
            
            combined.sort(key=lambda x: x.timestamp)
            return combined
        
        else:
            raise ValueError(f"Unsupported combination method: {method}")
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get strategy information."""
        return {
            'name': self.name,
            'parameters': self.parameters,
            'signal_generators': [gen.name for gen in self.signal_generators],
            'class': self.__class__.__name__
        }
    
    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(name='{self.name}')"
