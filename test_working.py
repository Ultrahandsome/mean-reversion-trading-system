#!/usr/bin/env python3
"""Working unit tests for the Mean Reversion Trading System."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
import importlib.util

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def load_module(name, path):
    """Load a module from a file path."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Load modules
models = load_module("models", "src/data/models.py")
mr_indicators = load_module("mr_indicators", "src/factors/indicators/mean_reversion.py")
portfolio_module = load_module("portfolio", "src/backtesting/portfolio.py")

class TestDataModels:
    """Test data models."""
    
    def test_price_data_creation(self):
        """Test PriceData creation and validation."""
        price_data = models.PriceData(
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
            models.PriceData(
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
            models.PriceData(
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
        np.random.seed(42)  # For reproducible tests
        prices = 100 + np.cumsum(np.random.normal(0, 1, 10) * 0.5)
        
        data = pd.DataFrame({
            'open': prices * (1 + np.random.normal(0, 0.01, 10)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.02, 10))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.02, 10))),
            'close': prices,
            'volume': np.random.uniform(1000000, 2000000, 10)
        }, index=dates)
        
        # Ensure OHLC relationships are valid
        data['high'] = np.maximum(data[['open', 'close']].max(axis=1), data['high'])
        data['low'] = np.minimum(data[['open', 'close']].min(axis=1), data['low'])
        
        market_data = models.MarketData(data, "AAPL")
        
        assert market_data.symbol == "AAPL"
        assert len(market_data.data) == 10
        assert 'returns' in market_data.data.columns
        assert 'log_returns' in market_data.data.columns
    
    def test_symbol_creation(self):
        """Test Symbol creation."""
        symbol = models.Symbol(
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
        
        # Ensure OHLC relationships are valid
        data['high'] = np.maximum(data[['open', 'close']].max(axis=1), data['high'])
        data['low'] = np.minimum(data[['open', 'close']].min(axis=1), data['low'])
        
        return models.MarketData(data, "TEST")
    
    def test_zscore_calculation(self):
        """Test Z-score calculation."""
        market_data = self.create_sample_market_data(50)
        indicators = mr_indicators.MeanReversionIndicators()
        
        zscore = indicators.zscore(market_data, period=20)
        
        assert len(zscore) == 50
        assert not pd.isna(zscore.iloc[-1])  # Should have valid values at the end
        
        # Z-score should have reasonable range (mostly between -3 and 3)
        valid_zscores = zscore.dropna()
        assert valid_zscores.min() > -5
        assert valid_zscores.max() < 5
    
    def test_bollinger_position(self):
        """Test Bollinger Band position calculation."""
        market_data = self.create_sample_market_data(50)
        indicators = mr_indicators.MeanReversionIndicators()
        
        bb_position = indicators.bollinger_position(market_data, period=20, num_std=2.0)
        
        assert len(bb_position) == 50
        
        # Bollinger position should be between 0 and 1 (mostly)
        valid_positions = bb_position.dropna()
        # Allow some tolerance for extreme values
        assert valid_positions.min() >= -0.5
        assert valid_positions.max() <= 1.5


class TestPortfolio:
    """Test portfolio management."""
    
    def test_portfolio_creation(self):
        """Test portfolio creation."""
        portfolio = portfolio_module.Portfolio(
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
        portfolio = portfolio_module.Portfolio(initial_capital=100000)
        
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
        portfolio = portfolio_module.Portfolio(initial_capital=100000)
        
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


def run_tests():
    """Run all tests manually."""
    print("🧪 Running Unit Tests...")
    print("=" * 50)
    
    test_results = []
    
    # Test Data Models
    print("1. Testing Data Models...")
    try:
        test_data = TestDataModels()
        test_data.test_price_data_creation()
        test_data.test_price_data_validation()
        test_data.test_market_data_creation()
        test_data.test_symbol_creation()
        print("   ✓ All data model tests passed")
        test_results.append(("Data Models", True))
    except Exception as e:
        print(f"   ❌ Data model tests failed: {e}")
        test_results.append(("Data Models", False))
    
    # Test Indicators
    print("\n2. Testing Indicators...")
    try:
        test_indicators = TestIndicators()
        test_indicators.test_zscore_calculation()
        test_indicators.test_bollinger_position()
        print("   ✓ All indicator tests passed")
        test_results.append(("Indicators", True))
    except Exception as e:
        print(f"   ❌ Indicator tests failed: {e}")
        test_results.append(("Indicators", False))
    
    # Test Portfolio
    print("\n3. Testing Portfolio...")
    try:
        test_portfolio = TestPortfolio()
        test_portfolio.test_portfolio_creation()
        test_portfolio.test_position_opening()
        test_portfolio.test_position_closing()
        print("   ✓ All portfolio tests passed")
        test_results.append(("Portfolio", True))
    except Exception as e:
        print(f"   ❌ Portfolio tests failed: {e}")
        test_results.append(("Portfolio", False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    passed = sum(1 for _, result in test_results if result)
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} test suites passed")
    
    if passed == total:
        print("🎉 ALL UNIT TESTS PASSED!")
        return True
    else:
        print("⚠ Some unit tests failed")
        return False


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
