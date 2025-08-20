#!/usr/bin/env python3
"""Test script to verify all imports work correctly."""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test all core imports."""
    print("🔍 Testing Mean Reversion Trading System Imports...")
    print("=" * 60)
    
    try:
        # Test core libraries
        print("1. Testing core libraries...")
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        print("   ✓ Core libraries (numpy, pandas, matplotlib) imported successfully")
        
        # Test data models
        print("\n2. Testing data models...")
        import data.models
        PriceData = data.models.PriceData
        MarketData = data.models.MarketData
        Symbol = data.models.Symbol
        print("   ✓ Data models imported successfully")

        # Test data providers (skip storage for now due to relative imports)
        print("\n3. Testing data providers...")
        import data.providers.yahoo_finance
        YahooFinanceProvider = data.providers.yahoo_finance.YahooFinanceProvider
        print("   ✓ Data providers imported successfully")
        
        # Test indicators
        print("\n4. Testing indicators...")
        import factors.indicators.mean_reversion
        MeanReversionIndicators = factors.indicators.mean_reversion.MeanReversionIndicators
        print("   ✓ Mean reversion indicators imported successfully")

        # Test strategies
        print("\n5. Testing strategies...")
        import strategies.mean_reversion.zscore_strategy
        ZScoreMeanReversion = strategies.mean_reversion.zscore_strategy.ZScoreMeanReversion
        print("   ✓ Z-Score strategy imported successfully")
        
        # Test backtesting
        print("\n6. Testing backtesting components...")
        import backtesting.portfolio
        Portfolio = backtesting.portfolio.Portfolio
        print("   ✓ Portfolio imported successfully")

        # Test utilities
        print("\n7. Testing utilities...")
        import utils.config
        import utils.logger
        Config = utils.config.Config
        get_logger = utils.logger.get_logger
        print("   ✓ Utility modules imported successfully")
        
        print("\n" + "=" * 60)
        print("✅ ALL IMPORTS SUCCESSFUL!")
        print("✅ Project structure is properly set up")
        return True
        
    except ImportError as e:
        print(f"\n❌ Import Error: {e}")
        print("❌ Some modules are missing or have import issues")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected Error: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality of core components."""
    print("\n🔧 Testing Basic Functionality...")
    print("=" * 60)
    
    try:
        from datetime import datetime
        import numpy as np
        import pandas as pd
        import data.models
        PriceData = data.models.PriceData
        MarketData = data.models.MarketData
        Symbol = data.models.Symbol
        
        # Test PriceData creation
        print("1. Testing PriceData creation...")
        price_data = PriceData(
            symbol="TEST",
            timestamp=datetime(2023, 1, 1),
            open=100.0,
            high=105.0,
            low=99.0,
            close=104.0,
            volume=1000000
        )
        print(f"   ✓ PriceData created: {price_data.symbol} @ ${price_data.close}")
        
        # Test MarketData creation
        print("\n2. Testing MarketData creation...")
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
        
        market_data = MarketData(data, "TEST")
        print(f"   ✓ MarketData created: {len(market_data.data)} records for {market_data.symbol}")
        
        # Test Symbol creation
        print("\n3. Testing Symbol creation...")
        symbol = Symbol(
            symbol="TEST",
            name="Test Symbol",
            exchange="TEST_EXCHANGE",
            asset_type="stock"
        )
        print(f"   ✓ Symbol created: {symbol.symbol} ({symbol.name})")
        
        # Test indicators
        print("\n4. Testing technical indicators...")
        import factors.indicators.mean_reversion
        MeanReversionIndicators = factors.indicators.mean_reversion.MeanReversionIndicators
        indicators = MeanReversionIndicators()
        
        # Create more data for indicators
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        data = pd.DataFrame({
            'open': 100 + np.cumsum(np.random.normal(0, 1, 50) * 0.5),
            'high': 100 + np.cumsum(np.random.normal(0, 1, 50) * 0.5) + np.abs(np.random.normal(0, 2, 50)),
            'low': 100 + np.cumsum(np.random.normal(0, 1, 50) * 0.5) - np.abs(np.random.normal(0, 2, 50)),
            'close': 100 + np.cumsum(np.random.normal(0, 1, 50) * 0.5),
            'volume': np.random.uniform(1000000, 2000000, 50)
        }, index=dates)
        
        # Ensure OHLC relationships
        data['high'] = np.maximum(data[['open', 'close']].max(axis=1), data['high'])
        data['low'] = np.minimum(data[['open', 'close']].min(axis=1), data['low'])
        
        market_data = MarketData(data, "TEST")
        
        zscore = indicators.zscore(market_data, period=20)
        print(f"   ✓ Z-score calculated: {len(zscore.dropna())} valid values")
        
        print("\n" + "=" * 60)
        print("✅ BASIC FUNCTIONALITY TESTS PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ Functionality Test Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Mean Reversion Trading System - Import & Functionality Test")
    print("=" * 80)
    
    # Test imports
    imports_ok = test_imports()
    
    if imports_ok:
        # Test basic functionality
        functionality_ok = test_basic_functionality()
        
        if functionality_ok:
            print("\n🎉 ALL TESTS PASSED!")
            print("🎉 System is ready for use!")
            sys.exit(0)
        else:
            print("\n❌ Functionality tests failed")
            sys.exit(1)
    else:
        print("\n❌ Import tests failed")
        sys.exit(1)
