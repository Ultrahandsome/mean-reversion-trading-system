"""Base signal generator for the Mean Reversion Trading System."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
import pandas as pd
import numpy as np

from ...data.models import MarketData
from ...utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Signal:
    """Represents a trading signal."""
    
    timestamp: datetime
    symbol: str
    signal_type: str  # 'entry', 'exit', 'stop'
    direction: int    # 1 for long, -1 for short, 0 for neutral
    strength: float   # Signal strength (0-1)
    price: float      # Price at signal generation
    confidence: float = 0.5  # Confidence level (0-1)
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate signal data."""
        if self.direction not in [-1, 0, 1]:
            raise ValueError("Direction must be -1, 0, or 1")
        
        if not 0 <= self.strength <= 1:
            raise ValueError("Strength must be between 0 and 1")
        
        if not 0 <= self.confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1")
        
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert signal to dictionary."""
        return {
            'timestamp': self.timestamp,
            'symbol': self.symbol,
            'signal_type': self.signal_type,
            'direction': self.direction,
            'strength': self.strength,
            'price': self.price,
            'confidence': self.confidence,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Signal':
        """Create signal from dictionary."""
        return cls(**data)
    
    def is_entry_signal(self) -> bool:
        """Check if this is an entry signal."""
        return self.signal_type == 'entry' and self.direction != 0
    
    def is_exit_signal(self) -> bool:
        """Check if this is an exit signal."""
        return self.signal_type == 'exit'
    
    def is_long_signal(self) -> bool:
        """Check if this is a long signal."""
        return self.direction == 1
    
    def is_short_signal(self) -> bool:
        """Check if this is a short signal."""
        return self.direction == -1


class SignalGenerator(ABC):
    """Abstract base class for signal generators."""
    
    def __init__(self, name: str):
        """
        Initialize signal generator.
        
        Args:
            name: Generator name
        """
        self.name = name
        self.parameters = {}
    
    @abstractmethod
    def generate_signals(self, market_data: MarketData, **kwargs) -> List[Signal]:
        """
        Generate trading signals from market data.
        
        Args:
            market_data: Market data to analyze
            **kwargs: Additional parameters
            
        Returns:
            List of generated signals
        """
        pass
    
    def set_parameters(self, **kwargs) -> None:
        """Set generator parameters."""
        self.parameters.update(kwargs)
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get generator parameters."""
        return self.parameters.copy()
    
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
    
    def filter_signals(
        self,
        signals: List[Signal],
        min_strength: float = 0.0,
        min_confidence: float = 0.0,
        signal_types: Optional[List[str]] = None
    ) -> List[Signal]:
        """
        Filter signals based on criteria.
        
        Args:
            signals: List of signals to filter
            min_strength: Minimum signal strength
            min_confidence: Minimum confidence level
            signal_types: List of allowed signal types
            
        Returns:
            Filtered signals
        """
        filtered = []
        
        for signal in signals:
            # Check strength
            if signal.strength < min_strength:
                continue
            
            # Check confidence
            if signal.confidence < min_confidence:
                continue
            
            # Check signal type
            if signal_types and signal.signal_type not in signal_types:
                continue
            
            filtered.append(signal)
        
        return filtered
    
    def signals_to_dataframe(self, signals: List[Signal]) -> pd.DataFrame:
        """
        Convert signals to DataFrame.
        
        Args:
            signals: List of signals
            
        Returns:
            DataFrame with signal data
        """
        if not signals:
            return pd.DataFrame()
        
        data = [signal.to_dict() for signal in signals]
        df = pd.DataFrame(data)
        
        # Set timestamp as index
        if 'timestamp' in df.columns:
            df.set_index('timestamp', inplace=True)
        
        return df
    
    def combine_signals(
        self,
        signal_lists: List[List[Signal]],
        method: str = 'union'
    ) -> List[Signal]:
        """
        Combine multiple signal lists.
        
        Args:
            signal_lists: List of signal lists to combine
            method: Combination method ('union', 'intersection', 'weighted')
            
        Returns:
            Combined signals
        """
        if not signal_lists:
            return []
        
        if method == 'union':
            # Simple union of all signals
            combined = []
            for signal_list in signal_lists:
                combined.extend(signal_list)
            
            # Sort by timestamp
            combined.sort(key=lambda x: x.timestamp)
            return combined
        
        elif method == 'intersection':
            # Only signals that appear in all lists (same timestamp and symbol)
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
    
    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(name='{self.name}')"
