#!/usr/bin/env python3
"""Performance evaluation with real market data."""

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

def test_yahoo_finance_provider():
    """Test Yahoo Finance provider with real data."""
    print("📈 Testing Yahoo Finance Provider with Real Data")
    print("=" * 60)
    
    try:
        # Load Yahoo Finance provider
        yahoo_module = load_module("yahoo_provider", "src/data/providers/yahoo_finance.py")
        provider = yahoo_module.YahooFinanceProvider()
        
        print("1. Testing data retrieval...")
        
        # Test with a small date range
        symbols = ['AAPL', 'MSFT']
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 3, 31)  # 3 months of data
        
        market_data = {}
        
        for symbol in symbols:
            try:
                print(f"   Fetching {symbol}...")
                data = provider.get_price_data(symbol, start_date, end_date)
                
                if data and len(data.data) > 0:
                    market_data[symbol] = data
                    print(f"   ✓ {symbol}: {len(data.data)} records")
                    print(f"     Date range: {data.data.index[0].date()} to {data.data.index[-1].date()}")
                    print(f"     Price range: ${data.data['close'].min():.2f} - ${data.data['close'].max():.2f}")
                else:
                    print(f"   ⚠ {symbol}: No data retrieved")
                    
            except Exception as e:
                print(f"   ❌ {symbol}: Error - {str(e)[:100]}...")
        
        if not market_data:
            print("   ⚠ No real market data available, using mock data for performance test")
            return None
        
        print(f"\n   ✓ Successfully retrieved data for {len(market_data)} symbols")
        return market_data
        
    except Exception as e:
        print(f"   ❌ Yahoo Finance provider test failed: {e}")
        return None

def run_performance_backtest(market_data):
    """Run a performance backtest."""
    print("\n🎯 Running Performance Backtest")
    print("=" * 60)
    
    try:
        # Load required modules
        zscore_strategy = load_module("zscore_strategy", "src/strategies/mean_reversion/zscore_strategy.py")
        portfolio_module = load_module("portfolio", "src/backtesting/portfolio.py")
        
        # Initialize strategy
        strategy = zscore_strategy.ZScoreMeanReversion(
            lookback_period=20,
            entry_threshold=2.0,
            exit_threshold=0.5,
            min_volume=0,  # Disable volume filter for testing
            volatility_filter=False  # Disable volatility filter
        )
        
        print(f"1. Strategy: {strategy.name}")
        print(f"   Parameters: {strategy.get_parameters()}")
        
        # Generate signals
        print("\n2. Generating signals...")
        signals = strategy.generate_signals(market_data)
        
        if not signals:
            print("   ⚠ No signals generated")
            return None
        
        entry_signals = [s for s in signals if s.signal_type == 'entry']
        exit_signals = [s for s in signals if s.signal_type == 'exit']
        long_signals = [s for s in entry_signals if s.direction == 1]
        short_signals = [s for s in entry_signals if s.direction == -1]
        
        print(f"   ✓ Total signals: {len(signals)}")
        print(f"   ✓ Entry signals: {len(entry_signals)} (Long: {len(long_signals)}, Short: {len(short_signals)})")
        print(f"   ✓ Exit signals: {len(exit_signals)}")
        
        # Initialize portfolio
        initial_capital = 100000
        portfolio = portfolio_module.Portfolio(
            initial_capital=initial_capital,
            commission=0.001,  # 0.1% commission
            slippage=0.0005    # 0.05% slippage
        )
        
        print(f"\n3. Portfolio simulation (Initial capital: ${initial_capital:,})")
        
        # Sort signals by timestamp
        signals.sort(key=lambda x: x.timestamp)
        
        # Simple backtesting simulation
        processed_signals = 0
        max_positions = 5  # Limit concurrent positions
        position_size = 0.15  # 15% of portfolio per position
        
        for signal in signals:
            if signal.signal_type == 'entry' and not portfolio.has_position(signal.symbol):
                # Check if we can open new positions
                if len(portfolio.positions) >= max_positions:
                    continue
                
                # Calculate position size
                portfolio_value = portfolio.total_value
                position_value = portfolio_value * position_size
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
        
        # Close any remaining positions at the last available price
        for symbol in list(portfolio.positions.keys()):
            if symbol in market_data:
                last_price = market_data[symbol].data['close'].iloc[-1]
                last_date = market_data[symbol].data.index[-1]
                
                portfolio.close_position(
                    symbol=symbol,
                    price=last_price,
                    timestamp=last_date
                )
        
        print(f"   ✓ Processed {processed_signals} signals")
        print(f"   ✓ Final positions: {len(portfolio.positions)}")
        print(f"   ✓ Completed trades: {len(portfolio.closed_positions)}")
        
        # Calculate performance metrics
        print("\n4. Performance Analysis")
        
        if portfolio.closed_positions:
            # Basic metrics
            total_pnl = sum(pos['pnl'] for pos in portfolio.closed_positions)
            total_return = total_pnl / initial_capital
            
            winning_trades = [pos for pos in portfolio.closed_positions if pos['pnl'] > 0]
            losing_trades = [pos for pos in portfolio.closed_positions if pos['pnl'] < 0]
            
            win_rate = len(winning_trades) / len(portfolio.closed_positions)
            avg_win = np.mean([pos['pnl'] for pos in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([pos['pnl'] for pos in losing_trades]) if losing_trades else 0
            
            profit_factor = abs(sum(pos['pnl'] for pos in winning_trades) / sum(pos['pnl'] for pos in losing_trades)) if losing_trades else float('inf')
            
            # Holding periods
            holding_periods = [pos['holding_period'] for pos in portfolio.closed_positions]
            avg_holding_period = np.mean(holding_periods) if holding_periods else 0
            
            print(f"   📊 Total Return: {total_return:.2%}")
            print(f"   📊 Total P&L: ${total_pnl:.2f}")
            print(f"   📊 Final Portfolio Value: ${portfolio.total_value:.2f}")
            print(f"   📊 Number of Trades: {len(portfolio.closed_positions)}")
            print(f"   📊 Win Rate: {win_rate:.2%}")
            print(f"   📊 Average Win: ${avg_win:.2f}")
            print(f"   📊 Average Loss: ${avg_loss:.2f}")
            print(f"   📊 Profit Factor: {profit_factor:.2f}")
            print(f"   📊 Average Holding Period: {avg_holding_period:.1f} days")
            
            # Trade breakdown by symbol
            print("\n   📈 Trade Breakdown by Symbol:")
            symbol_stats = {}
            for pos in portfolio.closed_positions:
                symbol = pos['symbol']
                if symbol not in symbol_stats:
                    symbol_stats[symbol] = {'trades': 0, 'pnl': 0, 'wins': 0}
                
                symbol_stats[symbol]['trades'] += 1
                symbol_stats[symbol]['pnl'] += pos['pnl']
                if pos['pnl'] > 0:
                    symbol_stats[symbol]['wins'] += 1
            
            for symbol, stats in symbol_stats.items():
                win_rate_symbol = stats['wins'] / stats['trades'] if stats['trades'] > 0 else 0
                print(f"     {symbol}: {stats['trades']} trades, ${stats['pnl']:.2f} P&L, {win_rate_symbol:.1%} win rate")
            
            return {
                'total_return': total_return,
                'total_pnl': total_pnl,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'num_trades': len(portfolio.closed_positions),
                'avg_holding_period': avg_holding_period
            }
        else:
            print("   ⚠ No completed trades to analyze")
            return None
            
    except Exception as e:
        print(f"   ❌ Performance backtest failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_mock_data_for_performance():
    """Create mock data for performance testing if real data unavailable."""
    print("📊 Creating Mock Data for Performance Testing")
    print("=" * 60)
    
    models = load_module("models", "src/data/models.py")
    
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2023, 12, 31)
    
    market_data = {}
    
    for symbol in symbols:
        # Generate 2 years of daily data
        dates = pd.date_range(start_date, end_date, freq='D')
        dates = dates[dates.weekday < 5]  # Weekdays only
        
        np.random.seed(hash(symbol) % 2**32)  # Consistent per symbol
        
        # Create mean-reverting price series
        n_days = len(dates)
        base_price = 100 + hash(symbol) % 200  # Different base price per symbol
        
        # Generate mean-reverting returns with trends
        returns = []
        current_deviation = 0
        trend = np.random.normal(0, 0.0002)  # Small trend component
        
        for i in range(n_days):
            # Mean reversion component
            mean_revert = -0.05 * current_deviation
            # Random component
            random_component = np.random.normal(0, 0.015)
            # Trend component
            trend_component = trend
            # Combine
            daily_return = mean_revert + random_component + trend_component
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
        
        # Add intraday volatility
        daily_vol = np.abs(np.random.normal(0, 0.008, len(dates)))
        data['high'] = data[['open', 'close']].max(axis=1) * (1 + daily_vol)
        data['low'] = data[['open', 'close']].min(axis=1) * (1 - daily_vol)
        data['volume'] = np.random.uniform(2000000, 10000000, len(dates))
        
        market_data[symbol] = models.MarketData(data, symbol)
        print(f"   ✓ {symbol}: {len(data)} records, price range ${data['close'].min():.2f}-${data['close'].max():.2f}")
    
    return market_data

def main():
    """Main performance evaluation."""
    print("🚀 Mean Reversion Trading System - Performance Evaluation")
    print("=" * 80)
    
    # Try to get real data first
    real_data = test_yahoo_finance_provider()
    
    if real_data:
        print("\n✅ Using real market data for performance evaluation")
        market_data = real_data
    else:
        print("\n📊 Using mock data for performance evaluation")
        market_data = create_mock_data_for_performance()
    
    # Run performance backtest
    results = run_performance_backtest(market_data)
    
    if results:
        print("\n" + "=" * 80)
        print("🎉 PERFORMANCE EVALUATION COMPLETED!")
        print("=" * 80)
        
        # Performance summary
        print(f"📈 Strategy Performance Summary:")
        print(f"   Total Return: {results['total_return']:.2%}")
        print(f"   Win Rate: {results['win_rate']:.2%}")
        print(f"   Profit Factor: {results['profit_factor']:.2f}")
        print(f"   Number of Trades: {results['num_trades']}")
        print(f"   Average Holding Period: {results['avg_holding_period']:.1f} days")
        
        # Performance assessment
        if results['total_return'] > 0:
            print("✅ Strategy shows positive returns")
        else:
            print("⚠ Strategy shows negative returns")
            
        if results['win_rate'] > 0.5:
            print("✅ Strategy has good win rate (>50%)")
        else:
            print("⚠ Strategy has low win rate (<50%)")
            
        if results['profit_factor'] > 1.0:
            print("✅ Strategy has positive profit factor")
        else:
            print("⚠ Strategy has negative profit factor")
        
        return True
    else:
        print("\n❌ Performance evaluation failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
