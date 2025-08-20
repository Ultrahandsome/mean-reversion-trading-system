#!/usr/bin/env python3
"""Integration test for the Mean Reversion Trading System."""

import sys
import os
from datetime import datetime, timedelta
import importlib.util
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def load_module(name, path):
    """Load a module from a file path."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def create_mock_data_provider():
    """Create a mock data provider for testing."""
    
    class MockDataProvider:
        def __init__(self):
            self.name = "Mock Provider"
        
        def get_price_data(self, symbol, start_date, end_date, interval="1d"):
            """Generate mock price data."""
            # Generate realistic price data
            dates = pd.date_range(start_date, end_date, freq='D')
            
            # Filter to weekdays only (simulate trading days)
            dates = dates[dates.weekday < 5]
            
            if len(dates) == 0:
                return None
            
            np.random.seed(hash(symbol) % 2**32)  # Consistent data per symbol
            
            # Create mean-reverting price series
            n_days = len(dates)
            base_price = 100 + hash(symbol) % 100  # Different base price per symbol
            
            # Generate mean-reverting returns
            returns = []
            current_deviation = 0
            
            for i in range(n_days):
                # Mean reversion component
                mean_revert = -0.1 * current_deviation
                # Random component
                random_component = np.random.normal(0, 0.02)
                # Combine
                daily_return = mean_revert + random_component
                returns.append(daily_return)
                current_deviation += daily_return
            
            # Convert to prices
            prices = [base_price]
            for ret in returns[1:]:
                prices.append(prices[-1] * (1 + ret))
            
            # Create OHLC data
            data = pd.DataFrame(index=dates)
            data['close'] = prices[:len(dates)]
            data['open'] = data['close'].shift(1).fillna(data['close'].iloc[0])
            
            # Add some intraday volatility
            daily_vol = np.abs(np.random.normal(0, 0.01, len(dates)))
            data['high'] = data[['open', 'close']].max(axis=1) * (1 + daily_vol)
            data['low'] = data[['open', 'close']].min(axis=1) * (1 - daily_vol)
            data['volume'] = np.random.uniform(1000000, 5000000, len(dates))
            
            # Load MarketData class
            models = load_module("models", "src/data/models.py")
            return models.MarketData(data, symbol)
        
        def get_latest_price(self, symbol):
            """Get latest price (not implemented for mock)."""
            return None
        
        def is_market_open(self):
            """Check if market is open."""
            return True
    
    return MockDataProvider()

def test_integration():
    """Run integration test."""
    print("🔗 Integration Test - Mean Reversion Trading System")
    print("=" * 60)
    
    try:
        # Load required modules
        print("1. Loading modules...")
        models = load_module("models", "src/data/models.py")
        mr_indicators = load_module("mr_indicators", "src/factors/indicators/mean_reversion.py")
        zscore_strategy = load_module("zscore_strategy", "src/strategies/mean_reversion/zscore_strategy.py")
        portfolio_module = load_module("portfolio", "src/backtesting/portfolio.py")
        print("   ✓ All modules loaded successfully")
        
        # Create mock data provider
        print("\n2. Setting up mock data provider...")
        data_provider = create_mock_data_provider()
        print("   ✓ Mock data provider created")
        
        # Test data retrieval
        print("\n3. Testing data retrieval...")
        symbols = ['AAPL', 'MSFT', 'GOOGL']
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 6, 30)
        
        market_data = {}
        for symbol in symbols:
            data = data_provider.get_price_data(symbol, start_date, end_date)
            if data:
                market_data[symbol] = data
                print(f"   ✓ Retrieved {len(data.data)} records for {symbol}")
            else:
                print(f"   ❌ No data for {symbol}")
        
        if not market_data:
            print("   ❌ No market data retrieved")
            return False
        
        # Test strategy signal generation
        print("\n4. Testing strategy signal generation...")
        strategy = zscore_strategy.ZScoreMeanReversion(
            lookback_period=20,
            entry_threshold=1.5,  # Lower threshold for more signals
            exit_threshold=0.5,
            min_volume=0,  # Disable volume filter
            volatility_filter=False  # Disable volatility filter
        )
        
        signals = strategy.generate_signals(market_data)
        print(f"   ✓ Generated {len(signals)} signals")
        
        if signals:
            entry_signals = [s for s in signals if s.signal_type == 'entry']
            exit_signals = [s for s in signals if s.signal_type == 'exit']
            long_signals = [s for s in entry_signals if s.direction == 1]
            short_signals = [s for s in entry_signals if s.direction == -1]
            
            print(f"   ✓ Entry signals: {len(entry_signals)} (Long: {len(long_signals)}, Short: {len(short_signals)})")
            print(f"   ✓ Exit signals: {len(exit_signals)}")
        
        # Test portfolio simulation
        print("\n5. Testing portfolio simulation...")
        portfolio = portfolio_module.Portfolio(initial_capital=100000)
        
        # Simple simulation: process signals chronologically
        signals.sort(key=lambda x: x.timestamp)
        
        processed_signals = 0
        for signal in signals[:10]:  # Process first 10 signals
            if signal.signal_type == 'entry' and not portfolio.has_position(signal.symbol):
                # Calculate position size (simple: $10,000 per position)
                position_value = 10000
                quantity = position_value / signal.price
                
                success = portfolio.open_position(
                    symbol=signal.symbol,
                    quantity=quantity,
                    price=signal.price,
                    timestamp=signal.timestamp,
                    direction=signal.direction
                )
                
                if success:
                    processed_signals += 1
            
            elif signal.signal_type == 'exit' and portfolio.has_position(signal.symbol):
                success = portfolio.close_position(
                    symbol=signal.symbol,
                    price=signal.price,
                    timestamp=signal.timestamp
                )
                
                if success:
                    processed_signals += 1
        
        print(f"   ✓ Processed {processed_signals} signals")
        print(f"   ✓ Open positions: {len(portfolio.positions)}")
        print(f"   ✓ Closed positions: {len(portfolio.closed_positions)}")
        print(f"   ✓ Current cash: ${portfolio.cash:,.2f}")
        
        # Calculate basic performance metrics
        if portfolio.closed_positions:
            total_pnl = sum(pos['pnl'] for pos in portfolio.closed_positions)
            winning_trades = sum(1 for pos in portfolio.closed_positions if pos['pnl'] > 0)
            total_trades = len(portfolio.closed_positions)
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            print(f"   ✓ Total P&L: ${total_pnl:.2f}")
            print(f"   ✓ Win rate: {win_rate:.2%} ({winning_trades}/{total_trades})")
        
        # Test indicators on real data
        print("\n6. Testing indicators on market data...")
        test_symbol = list(market_data.keys())[0]
        test_data = market_data[test_symbol]
        
        indicators = mr_indicators.MeanReversionIndicators()
        
        # Test Z-score
        zscore = indicators.zscore(test_data, period=20)
        valid_zscores = zscore.dropna()
        
        if len(valid_zscores) > 0:
            print(f"   ✓ Z-score for {test_symbol}: {len(valid_zscores)} values")
            print(f"   ✓ Z-score range: {valid_zscores.min():.2f} to {valid_zscores.max():.2f}")
            
            # Count extreme values
            extreme_low = (valid_zscores <= -2).sum()
            extreme_high = (valid_zscores >= 2).sum()
            print(f"   ✓ Extreme values: {extreme_low} oversold, {extreme_high} overbought")
        
        print("\n" + "=" * 60)
        print("✅ INTEGRATION TEST COMPLETED SUCCESSFULLY!")
        print("✅ All components work together correctly")
        return True
        
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_integration()
    sys.exit(0 if success else 1)
