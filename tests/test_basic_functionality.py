"""Basic functionality tests for the Mean Reversion Trading System."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.models import PriceData, MarketData, Symbol
from data.storage.sqlite_storage import SQLiteStorage
from factors.indicators.mean_reversion import MeanReversionIndicators
from factors.signals.mean_reversion_signals import MeanReversionSignals
from strategies.mean_reversion.zscore_strategy import ZScoreMeanReversion
from backtesting.portfolio import Portfolio, Position


class TestDataModels:
    """Test data models."""
    
    def test_price_data_creation(self):
        """Test PriceData creation and validation."""
        price_data = PriceData(
            symbol="AAPL",
            timestamp=datetime(2023, 1, 1),
            open=150.0,
            high=155.0,
            low=149.0,
            close=154.0,
            volume=1000000
        )
        
        assert price_data.symbol == "AAPL"
        assert price_data.open == 150.0
        assert price_data.high == 155.0
        assert price_data.low == 149.0
        assert price_data.close == 154.0
        assert price_data.volume == 1000000
        assert price_data.adjusted_close == 154.0  # Should default to close
    
    def test_price_data_validation(self):
        """Test PriceData validation."""
        # Test invalid OHLC relationship
        with pytest.raises(ValueError):
            PriceData(
                symbol="AAPL",
                timestamp=datetime(2023, 1, 1),
                open=150.0,
                high=149.0,  # High < Low
                low=151.0,
                close=154.0,
                volume=1000000
            )
        
        # Test negative volume
        with pytest.raises(ValueError):
            PriceData(
                symbol="AAPL",
                timestamp=datetime(2023, 1, 1),
                open=150.0,
                high=155.0,
                low=149.0,
                close=154.0,
                volume=-1000  # Negative volume
            )
    
    def test_market_data_creation(self):
        """Test MarketData creation."""
        # Create sample data
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        data = pd.DataFrame({
            'open': np.random.uniform(100, 110, 10),
            'high': np.random.uniform(110, 120, 10),
            'low': np.random.uniform(90, 100, 10),
            'close': np.random.uniform(100, 110, 10),
            'volume': np.random.uniform(1000000, 2000000, 10)
        }, index=dates)
        
        # Ensure OHLC relationships are valid
        data['high'] = np.maximum(data[['open', 'close']].max(axis=1), data['high'])
        data['low'] = np.minimum(data[['open', 'close']].min(axis=1), data['low'])
        
        market_data = MarketData(data, "AAPL")
        
        assert market_data.symbol == "AAPL"
        assert len(market_data.data) == 10
        assert 'returns' in market_data.data.columns
        assert 'log_returns' in market_data.data.columns
    
    def test_symbol_creation(self):
        """Test Symbol creation."""
        symbol = Symbol(
            symbol="AAPL",
            name="Apple Inc.",
            exchange="NASDAQ",
            asset_type="stock",
            currency="USD",
            sector="Technology"
        )
        
        assert symbol.symbol == "AAPL"
        assert symbol.name == "Apple Inc."
        assert symbol.exchange == "NASDAQ"
        assert symbol.asset_type == "stock"


class TestIndicators:
    """Test technical indicators."""
    
    def create_sample_market_data(self, length=50):
        """Create sample market data for testing."""
        dates = pd.date_range('2023-01-01', periods=length, freq='D')
        
        # Create realistic price data with some mean reversion
        np.random.seed(42)  # For reproducible tests
        prices = 100 + np.cumsum(np.random.normal(0, 1, length) * 0.5)
        
        data = pd.DataFrame({
            'open': prices * (1 + np.random.normal(0, 0.01, length)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.02, length))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.02, length))),
            'close': prices,
            'volume': np.random.uniform(1000000, 2000000, length)
        }, index=dates)
        
        return MarketData(data, "TEST")
    
    def test_zscore_calculation(self):
        """Test Z-score calculation."""
        market_data = self.create_sample_market_data(50)
        indicators = MeanReversionIndicators()
        
        zscore = indicators.zscore(market_data, period=20)
        
        assert len(zscore) == 50
        assert not zscore.iloc[-1] is np.nan  # Should have valid values at the end
        
        # Z-score should have reasonable range (mostly between -3 and 3)
        valid_zscores = zscore.dropna()
        assert valid_zscores.min() > -5
        assert valid_zscores.max() < 5
    
    def test_bollinger_position(self):
        """Test Bollinger Band position calculation."""
        market_data = self.create_sample_market_data(50)
        indicators = MeanReversionIndicators()
        
        bb_position = indicators.bollinger_position(market_data, period=20, num_std=2.0)
        
        assert len(bb_position) == 50
        
        # Bollinger position should be between 0 and 1
        valid_positions = bb_position.dropna()
        assert valid_positions.min() >= 0
        assert valid_positions.max() <= 1


class TestStrategy:
    """Test trading strategies."""
    
    def create_sample_market_data(self, length=100):
        """Create sample market data for testing."""
        dates = pd.date_range('2023-01-01', periods=length, freq='D')
        
        # Create price data with clear mean reversion patterns
        np.random.seed(42)
        base_price = 100
        prices = []
        
        for i in range(length):
            # Add some mean reversion behavior
            if i > 0:
                deviation = prices[-1] - base_price
                mean_revert = -0.1 * deviation  # Mean reversion component
                random_walk = np.random.normal(0, 1)
                price_change = mean_revert + random_walk
                prices.append(prices[-1] + price_change)
            else:
                prices.append(base_price)
        
        data = pd.DataFrame({
            'open': [p * (1 + np.random.normal(0, 0.005)) for p in prices],
            'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'close': prices,
            'volume': np.random.uniform(2000000, 3000000, length)
        }, index=dates)
        
        return MarketData(data, "TEST")
    
    def test_zscore_strategy_creation(self):
        """Test Z-Score strategy creation."""
        strategy = ZScoreMeanReversion(
            lookback_period=20,
            entry_threshold=2.0,
            exit_threshold=0.5
        )
        
        assert strategy.name == "Z-Score Mean Reversion"
        assert strategy.parameters['lookback_period'] == 20
        assert strategy.parameters['entry_threshold'] == 2.0
        assert strategy.parameters['exit_threshold'] == 0.5
    
    def test_zscore_strategy_signals(self):
        """Test Z-Score strategy signal generation."""
        strategy = ZScoreMeanReversion(
            lookback_period=20,
            entry_threshold=1.5,  # Lower threshold for testing
            exit_threshold=0.5,
            min_volume=0,  # Disable volume filter
            volatility_filter=False  # Disable volatility filter
        )
        
        market_data = self.create_sample_market_data(100)
        market_data_dict = {"TEST": market_data}
        
        signals = strategy.generate_signals(market_data_dict)
        
        # Should generate some signals
        assert len(signals) > 0
        
        # Check signal properties
        for signal in signals:
            assert signal.symbol == "TEST"
            assert signal.signal_type in ['entry', 'exit']
            assert signal.direction in [-1, 0, 1]
            assert 0 <= signal.strength <= 1
            assert 0 <= signal.confidence <= 1


class TestPortfolio:
    """Test portfolio management."""
    
    def test_portfolio_creation(self):
        """Test portfolio creation."""
        portfolio = Portfolio(
            initial_capital=100000,
            commission=0.001,
            slippage=0.0005
        )
        
        assert portfolio.initial_capital == 100000
        assert portfolio.cash == 100000
        assert portfolio.commission == 0.001
        assert portfolio.slippage == 0.0005
        assert len(portfolio.positions) == 0
    
    def test_position_opening(self):
        """Test opening positions."""
        portfolio = Portfolio(initial_capital=100000)
        
        success = portfolio.open_position(
            symbol="AAPL",
            quantity=100,
            price=150.0,
            timestamp=datetime(2023, 1, 1),
            direction=1
        )
        
        assert success
        assert portfolio.has_position("AAPL")
        assert len(portfolio.positions) == 1
        
        position = portfolio.get_position("AAPL")
        assert position.symbol == "AAPL"
        assert position.quantity == 100
        assert position.entry_price == 150.0
        assert position.direction == 1
    
    def test_position_closing(self):
        """Test closing positions."""
        portfolio = Portfolio(initial_capital=100000)
        
        # Open position
        portfolio.open_position(
            symbol="AAPL",
            quantity=100,
            price=150.0,
            timestamp=datetime(2023, 1, 1),
            direction=1
        )
        
        # Close position
        success = portfolio.close_position(
            symbol="AAPL",
            price=160.0,
            timestamp=datetime(2023, 1, 2)
        )
        
        assert success
        assert not portfolio.has_position("AAPL")
        assert len(portfolio.closed_positions) == 1
        
        closed_position = portfolio.closed_positions[0]
        assert closed_position['symbol'] == "AAPL"
        assert closed_position['pnl'] > 0  # Should be profitable


class TestStorage:
    """Test data storage."""
    
    def test_sqlite_storage_creation(self):
        """Test SQLite storage creation."""
        storage = SQLiteStorage(":memory:")  # In-memory database for testing
        
        with storage:
            storage.create_tables()
            
            # Test symbol operations
            symbol = Symbol(symbol="TEST", name="Test Symbol", asset_type="stock")
            storage.save_symbol(symbol)
            
            retrieved_symbol = storage.get_symbol("TEST")
            assert retrieved_symbol is not None
            assert retrieved_symbol.symbol == "TEST"
            assert retrieved_symbol.name == "Test Symbol"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
