#!/usr/bin/env python3
"""
Visualization example for the Mean Reversion Trading System.

This example demonstrates how to:
1. Run a backtest
2. Create various performance visualizations
3. Generate an interactive dashboard
"""

import sys
import os
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.providers.yahoo_finance import YahooFinanceProvider
from strategies.mean_reversion.zscore_strategy import ZScoreMeanReversion
from backtesting.engine.base import BacktestEngine
from visualization.performance import PerformanceVisualizer
from utils.logger import get_logger

logger = get_logger(__name__)


def main():
    """Run visualization example."""
    print("="*60)
    print("MEAN REVERSION TRADING SYSTEM - VISUALIZATION EXAMPLE")
    print("="*60)
    
    # Configuration
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    start_date = datetime(2021, 1, 1)
    end_date = datetime(2023, 12, 31)
    initial_capital = 100000
    
    print(f"\nBacktest Configuration:")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Period: {start_date.date()} to {end_date.date()}")
    print(f"Initial Capital: ${initial_capital:,}")
    
    try:
        # 1. Run backtest
        print("\n1. Running backtest...")
        
        data_provider = YahooFinanceProvider()
        
        strategy = ZScoreMeanReversion(
            lookback_period=20,
            entry_threshold=2.0,
            exit_threshold=0.5,
            min_volume=1000000,
            volatility_filter=True
        )
        
        engine = BacktestEngine(
            initial_capital=initial_capital,
            commission=0.001,
            slippage=0.0005
        )
        
        results = engine.run_backtest(
            strategy=strategy,
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            data_provider=data_provider,
            benchmark_symbol='SPY'
        )
        
        print("Backtest completed successfully!")
        results.print_summary()
        
        # 2. Create visualizations
        print("\n2. Creating visualizations...")
        
        visualizer = PerformanceVisualizer(results)
        
        # Create output directory
        output_dir = "visualization_output"
        os.makedirs(output_dir, exist_ok=True)
        
        # Equity curve
        print("   - Equity curve...")
        visualizer.plot_equity_curve(
            figsize=(14, 8),
            save_path=os.path.join(output_dir, "equity_curve.png")
        )
        
        # Drawdown analysis
        print("   - Drawdown analysis...")
        visualizer.plot_drawdown(
            figsize=(14, 6),
            save_path=os.path.join(output_dir, "drawdown.png")
        )
        
        # Returns distribution
        print("   - Returns distribution...")
        visualizer.plot_returns_distribution(
            figsize=(14, 6),
            save_path=os.path.join(output_dir, "returns_distribution.png")
        )
        
        # Monthly returns heatmap
        print("   - Monthly returns heatmap...")
        visualizer.plot_monthly_returns(
            figsize=(14, 8),
            save_path=os.path.join(output_dir, "monthly_returns.png")
        )
        
        # Comprehensive performance summary
        print("   - Performance summary...")
        visualizer.plot_performance_summary(
            figsize=(16, 12),
            save_path=os.path.join(output_dir, "performance_summary.png")
        )
        
        # 3. Create interactive dashboard
        print("\n3. Creating interactive dashboard...")
        
        try:
            interactive_fig = visualizer.create_interactive_dashboard()
            
            # Save interactive dashboard
            dashboard_path = os.path.join(output_dir, "interactive_dashboard.html")
            interactive_fig.write_html(dashboard_path)
            print(f"   Interactive dashboard saved to: {dashboard_path}")
            
            # Show in browser (optional)
            # interactive_fig.show()
            
        except ImportError:
            print("   Plotly not available, skipping interactive dashboard")
        except Exception as e:
            print(f"   Error creating interactive dashboard: {e}")
        
        # 4. Generate detailed analysis report
        print("\n4. Generating analysis report...")
        
        report_path = os.path.join(output_dir, "analysis_report.txt")
        with open(report_path, 'w') as f:
            f.write("MEAN REVERSION TRADING SYSTEM - ANALYSIS REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            # Strategy information
            f.write("STRATEGY INFORMATION\n")
            f.write("-" * 20 + "\n")
            f.write(f"Strategy: {results.strategy_name}\n")
            f.write(f"Period: {results.start_date.date()} to {results.end_date.date()}\n")
            f.write(f"Symbols: {', '.join(symbols)}\n")
            f.write(f"Initial Capital: ${results.initial_capital:,.2f}\n\n")
            
            # Performance metrics
            f.write("PERFORMANCE METRICS\n")
            f.write("-" * 20 + "\n")
            summary = results.summary()
            for key, value in summary.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
            
            # Risk metrics
            f.write("RISK ANALYSIS\n")
            f.write("-" * 20 + "\n")
            f.write(f"Maximum Drawdown: {results.max_drawdown:.2%}\n")
            f.write(f"Max Drawdown Duration: {results.max_drawdown_duration} days\n")
            f.write(f"Volatility: {results.volatility:.2%}\n")
            f.write(f"Sharpe Ratio: {results.sharpe_ratio:.2f}\n")
            f.write(f"Sortino Ratio: {results.sortino_ratio:.2f}\n")
            f.write(f"Calmar Ratio: {results.calmar_ratio:.2f}\n\n")
            
            # Trade analysis
            if not results.positions_df.empty:
                f.write("TRADE ANALYSIS\n")
                f.write("-" * 20 + "\n")
                f.write(f"Total Trades: {len(results.positions_df)}\n")
                f.write(f"Winning Trades: {results.winning_trades}\n")
                f.write(f"Losing Trades: {results.losing_trades}\n")
                f.write(f"Win Rate: {results.win_rate:.2%}\n")
                f.write(f"Average Win: ${results.avg_win:.2f}\n")
                f.write(f"Average Loss: ${results.avg_loss:.2f}\n")
                f.write(f"Profit Factor: {results.profit_factor:.2f}\n")
                
                # Holding period analysis
                avg_holding = results.positions_df['holding_period'].mean()
                max_holding = results.positions_df['holding_period'].max()
                min_holding = results.positions_df['holding_period'].min()
                
                f.write(f"Average Holding Period: {avg_holding:.1f} days\n")
                f.write(f"Max Holding Period: {max_holding} days\n")
                f.write(f"Min Holding Period: {min_holding} days\n\n")
            
            # Benchmark comparison
            if results.benchmark_return is not None:
                f.write("BENCHMARK COMPARISON\n")
                f.write("-" * 20 + "\n")
                f.write(f"Strategy Return: {results.total_return:.2%}\n")
                f.write(f"Benchmark Return: {results.benchmark_return:.2%}\n")
                f.write(f"Excess Return: {results.excess_return:.2%}\n")
                if results.beta is not None:
                    f.write(f"Beta: {results.beta:.2f}\n")
                if results.alpha is not None:
                    f.write(f"Alpha: {results.alpha:.2%}\n")
                if results.information_ratio is not None:
                    f.write(f"Information Ratio: {results.information_ratio:.2f}\n")
        
        print(f"   Analysis report saved to: {report_path}")
        
        print(f"\n5. Summary:")
        print(f"   All visualizations saved to: {output_dir}/")
        print(f"   Files created:")
        print(f"   - equity_curve.png")
        print(f"   - drawdown.png") 
        print(f"   - returns_distribution.png")
        print(f"   - monthly_returns.png")
        print(f"   - performance_summary.png")
        print(f"   - interactive_dashboard.html")
        print(f"   - analysis_report.txt")
        
        print(f"\n{'='*60}")
        print("VISUALIZATION EXAMPLE COMPLETED SUCCESSFULLY!")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Visualization example failed: {e}")
        print(f"\nERROR: {e}")
        print("\nPlease check the logs for more details.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
