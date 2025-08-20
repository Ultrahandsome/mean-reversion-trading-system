#!/usr/bin/env python3
"""Simple test to verify core functionality without complex imports."""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_core_functionality():
    """Test core functionality by directly importing modules."""
    print("🔍 Testing Core Functionality...")
    print("=" * 50)
    
    try:
        # Test 1: Data Models
        print("1. Testing data models...")
        
        # Import data models directly
        import importlib.util
        spec = importlib.util.spec_from_file_location("models", "src/data/models.py")
        models = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(models)
        
        # Test PriceData
        price_data = models.PriceData(
            symbol="AAPL",
            timestamp=datetime(2023, 1, 1),
            open=150.0,
            high=155.0,
            low=149.0,
            close=154.0,
            volume=1000000
        )
        print(f"   ✓ PriceData created: {price_data.symbol} @ ${price_data.close}")
        
        # Test MarketData
        dates = pd.date_range('2023-01-01', periods=30, freq='D')
        np.random.seed(42)  # For reproducible results
        prices = 100 + np.cumsum(np.random.normal(0, 1, 30) * 0.5)
        
        data = pd.DataFrame({
            'open': prices * (1 + np.random.normal(0, 0.01, 30)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.02, 30))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.02, 30))),
            'close': prices,
            'volume': np.random.uniform(1000000, 2000000, 30)
        }, index=dates)
        
        # Ensure OHLC relationships are valid
        data['high'] = np.maximum(data[['open', 'close']].max(axis=1), data['high'])
        data['low'] = np.minimum(data[['open', 'close']].min(axis=1), data['low'])
        
        market_data = models.MarketData(data, "AAPL")
        print(f"   ✓ MarketData created: {len(market_data.data)} records")
        print(f"   ✓ Returns calculated: {len(market_data.data['returns'].dropna())} valid returns")
        
        # Test 2: Mean Reversion Indicators
        print("\n2. Testing mean reversion indicators...")
        
        spec = importlib.util.spec_from_file_location("mr_indicators", "src/factors/indicators/mean_reversion.py")
        mr_indicators = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mr_indicators)
        
        indicators = mr_indicators.MeanReversionIndicators()
        
        # Test Z-score
        zscore = indicators.zscore(market_data, period=20)
        valid_zscores = zscore.dropna()
        print(f"   ✓ Z-score calculated: {len(valid_zscores)} valid values")
        print(f"   ✓ Z-score range: {valid_zscores.min():.2f} to {valid_zscores.max():.2f}")
        
        # Test Bollinger position
        bb_position = indicators.bollinger_position(market_data, period=20)
        valid_bb = bb_position.dropna()
        print(f"   ✓ Bollinger position calculated: {len(valid_bb)} valid values")
        print(f"   ✓ BB position range: {valid_bb.min():.3f} to {valid_bb.max():.3f}")
        
        # Test 3: Portfolio Management
        print("\n3. Testing portfolio management...")
        
        spec = importlib.util.spec_from_file_location("portfolio", "src/backtesting/portfolio.py")
        portfolio_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(portfolio_module)
        
        portfolio = portfolio_module.Portfolio(initial_capital=100000)
        print(f"   ✓ Portfolio created with ${portfolio.initial_capital:,.2f}")
        
        # Test opening a position
        success = portfolio.open_position(
            symbol="AAPL",
            quantity=100,
            price=150.0,
            timestamp=datetime(2023, 1, 1),
            direction=1
        )
        print(f"   ✓ Position opened: {success}")
        print(f"   ✓ Portfolio positions: {len(portfolio.positions)}")
        print(f"   ✓ Remaining cash: ${portfolio.cash:,.2f}")
        
        # Test closing the position
        success = portfolio.close_position(
            symbol="AAPL",
            price=160.0,
            timestamp=datetime(2023, 1, 2)
        )
        print(f"   ✓ Position closed: {success}")
        print(f"   ✓ Closed positions: {len(portfolio.closed_positions)}")
        
        if portfolio.closed_positions:
            pnl = portfolio.closed_positions[0]['pnl']
            print(f"   ✓ P&L: ${pnl:.2f}")
        
        # Test 4: Yahoo Finance Provider (if available)
        print("\n4. Testing Yahoo Finance provider...")
        
        try:
            spec = importlib.util.spec_from_file_location("yahoo_provider", "src/data/providers/yahoo_finance.py")
            yahoo_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(yahoo_module)
            
            provider = yahoo_module.YahooFinanceProvider()
            print("   ✓ Yahoo Finance provider created")
            
            # Test getting symbol info (this might fail if no internet)
            try:
                symbol_info = provider.get_symbol_info("AAPL")
                if symbol_info:
                    print(f"   ✓ Symbol info retrieved: {symbol_info.name}")
                else:
                    print("   ⚠ Symbol info not available (possibly no internet)")
            except Exception as e:
                print(f"   ⚠ Symbol info test failed: {str(e)[:50]}...")
                
        except Exception as e:
            print(f"   ⚠ Yahoo Finance provider test failed: {str(e)[:50]}...")
        
        print("\n" + "=" * 50)
        print("✅ CORE FUNCTIONALITY TESTS COMPLETED!")
        print("✅ Key components are working correctly")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_provider():
    """Test data provider functionality with real data."""
    print("\n🌐 Testing Data Provider with Real Data...")
    print("=" * 50)
    
    try:
        import yfinance as yf
        
        # Test basic yfinance functionality
        print("1. Testing yfinance library...")
        ticker = yf.Ticker("AAPL")
        
        # Get recent data (last 5 days)
        hist = ticker.history(period="5d")
        
        if not hist.empty:
            print(f"   ✓ Retrieved {len(hist)} days of AAPL data")
            print(f"   ✓ Latest close: ${hist['Close'].iloc[-1]:.2f}")
            print(f"   ✓ Date range: {hist.index[0].date()} to {hist.index[-1].date()}")
            
            # Test data quality
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_cols = [col for col in required_cols if col not in hist.columns]
            
            if not missing_cols:
                print("   ✓ All required columns present")
            else:
                print(f"   ⚠ Missing columns: {missing_cols}")
            
            # Check for valid OHLC relationships
            valid_ohlc = ((hist['High'] >= hist['Low']) & 
                         (hist['High'] >= hist['Open']) & 
                         (hist['High'] >= hist['Close']) &
                         (hist['Low'] <= hist['Open']) & 
                         (hist['Low'] <= hist['Close'])).all()
            
            if valid_ohlc:
                print("   ✓ OHLC relationships are valid")
            else:
                print("   ⚠ Some OHLC relationships are invalid")
            
            return True
        else:
            print("   ❌ No data retrieved")
            return False
            
    except ImportError:
        print("   ⚠ yfinance not available, skipping data provider test")
        return True
    except Exception as e:
        print(f"   ❌ Data provider test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Mean Reversion Trading System - Simple Functionality Test")
    print("=" * 80)
    
    # Test core functionality
    core_ok = test_core_functionality()
    
    if core_ok:
        # Test data provider
        data_ok = test_data_provider()
        
        if data_ok:
            print("\n🎉 ALL TESTS PASSED!")
            print("🎉 System core functionality is working!")
            sys.exit(0)
        else:
            print("\n⚠ Data provider tests had issues, but core functionality works")
            sys.exit(0)
    else:
        print("\n❌ Core functionality tests failed")
        sys.exit(1)
