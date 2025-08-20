#!/usr/bin/env python3
"""
Basic backtest example for the Mean Reversion Trading System.

This example demonstrates how to:
1. Set up a data provider
2. Create a mean reversion strategy
3. Run a backtest
4. Analyze results
"""

import sys
import os
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.providers.yahoo_finance import YahooFinanceProvider
from strategies.mean_reversion.zscore_strategy import ZScoreMeanReversion
from backtesting.engine.base import BacktestEngine
from utils.logger import get_logger

logger = get_logger(__name__)


def main():
    """Run basic backtest example."""
    print("="*60)
    print("MEAN REVERSION TRADING SYSTEM - BASIC BACKTEST EXAMPLE")
    print("="*60)
    
    # Configuration
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2023, 12, 31)
    initial_capital = 100000
    
    print(f"\nBacktest Configuration:")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Period: {start_date.date()} to {end_date.date()}")
    print(f"Initial Capital: ${initial_capital:,}")
    
    try:
        # 1. Initialize data provider
        print("\n1. Initializing data provider...")
        data_provider = YahooFinanceProvider()
        
        # 2. Create strategy
        print("2. Creating Z-Score mean reversion strategy...")
        strategy = ZScoreMeanReversion(
            lookback_period=20,
            entry_threshold=2.0,
            exit_threshold=0.5,
            min_volume=1000000,
            volatility_filter=True
        )
        
        print(f"Strategy: {strategy.name}")
        print(f"Parameters: {strategy.get_parameters()}")
        
        # 3. Initialize backtesting engine
        print("\n3. Initializing backtesting engine...")
        engine = BacktestEngine(
            initial_capital=initial_capital,
            commission=0.001,  # 0.1% commission
            slippage=0.0005    # 0.05% slippage
        )
        
        # 4. Run backtest
        print("4. Running backtest...")
        print("This may take a few minutes to download data and run simulation...")
        
        results = engine.run_backtest(
            strategy=strategy,
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            data_provider=data_provider,
            benchmark_symbol='SPY',  # S&P 500 as benchmark
            rebalance_frequency='daily'
        )
        
        # 5. Display results
        print("\n5. Backtest Results:")
        results.print_summary()
        
        # Additional analysis
        print("\n6. Additional Analysis:")
        
        # Trade statistics
        if not results.trades_df.empty:
            print(f"\nTrade Details:")
            print(f"Total Trades: {len(results.trades_df)}")
            
            entry_trades = results.trades_df[results.trades_df['action'] == 'open']
            if not entry_trades.empty:
                print(f"Long Trades: {len(entry_trades[entry_trades['direction'] == 1])}")
                print(f"Short Trades: {len(entry_trades[entry_trades['direction'] == -1])}")
        
        # Position statistics
        if not results.positions_df.empty:
            print(f"\nPosition Statistics:")
            avg_holding_period = results.positions_df['holding_period'].mean()
            print(f"Average Holding Period: {avg_holding_period:.1f} days")
            
            profitable_positions = results.positions_df[results.positions_df['pnl'] > 0]
            if len(profitable_positions) > 0:
                avg_profit = profitable_positions['pnl'].mean()
                print(f"Average Profit per Winning Trade: ${avg_profit:.2f}")
            
            losing_positions = results.positions_df[results.positions_df['pnl'] < 0]
            if len(losing_positions) > 0:
                avg_loss = losing_positions['pnl'].mean()
                print(f"Average Loss per Losing Trade: ${avg_loss:.2f}")
        
        # Risk metrics
        print(f"\nRisk Metrics:")
        if results.volatility > 0:
            print(f"Annualized Volatility: {results.volatility:.2%}")
        if results.max_drawdown < 0:
            print(f"Maximum Drawdown: {results.max_drawdown:.2%}")
            print(f"Max Drawdown Duration: {results.max_drawdown_duration} days")
        
        # Benchmark comparison
        if results.benchmark_return is not None:
            print(f"\nBenchmark Comparison:")
            print(f"Strategy Return: {results.total_return:.2%}")
            print(f"Benchmark Return: {results.benchmark_return:.2%}")
            print(f"Excess Return: {results.excess_return:.2%}")
            if results.beta is not None:
                print(f"Beta: {results.beta:.2f}")
            if results.alpha is not None:
                print(f"Alpha: {results.alpha:.2%}")
        
        print(f"\n7. Data Export:")
        
        # Save results to CSV files
        if not results.equity_curve.empty:
            equity_file = "backtest_equity_curve.csv"
            results.equity_curve.to_csv(equity_file)
            print(f"Equity curve saved to: {equity_file}")
        
        if not results.trades_df.empty:
            trades_file = "backtest_trades.csv"
            results.trades_df.to_csv(trades_file)
            print(f"Trades saved to: {trades_file}")
        
        if not results.positions_df.empty:
            positions_file = "backtest_positions.csv"
            results.positions_df.to_csv(positions_file)
            print(f"Positions saved to: {positions_file}")
        
        print(f"\n{'='*60}")
        print("BACKTEST COMPLETED SUCCESSFULLY!")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        print(f"\nERROR: {e}")
        print("\nPlease check the logs for more details.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
