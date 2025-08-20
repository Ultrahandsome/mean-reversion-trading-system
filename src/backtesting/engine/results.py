"""Backtesting results container."""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

from ..portfolio import Portfolio
from ...utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class BacktestResults:
    """Container for backtesting results."""
    
    # Basic information
    strategy_name: str
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float
    
    # Portfolio data
    portfolio: Portfolio
    
    # Performance metrics
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    
    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    
    # Additional metrics
    calmar_ratio: float
    sortino_ratio: float
    beta: Optional[float] = None
    alpha: Optional[float] = None
    
    # Benchmark comparison (if available)
    benchmark_return: Optional[float] = None
    excess_return: Optional[float] = None
    information_ratio: Optional[float] = None
    
    # Raw data
    equity_curve: Optional[pd.DataFrame] = None
    trades_df: Optional[pd.DataFrame] = None
    positions_df: Optional[pd.DataFrame] = None
    
    def __post_init__(self):
        """Initialize additional data."""
        if self.equity_curve is None:
            self.equity_curve = self.portfolio.get_equity_dataframe()
        
        if self.trades_df is None:
            self.trades_df = self.portfolio.get_trades_dataframe()
        
        if self.positions_df is None:
            self.positions_df = self.portfolio.get_closed_positions_dataframe()
    
    @classmethod
    def from_portfolio(
        cls,
        portfolio: Portfolio,
        strategy_name: str,
        start_date: datetime,
        end_date: datetime,
        benchmark_returns: Optional[pd.Series] = None,
        risk_free_rate: float = 0.02
    ) -> 'BacktestResults':
        """
        Create BacktestResults from Portfolio.
        
        Args:
            portfolio: Portfolio object
            strategy_name: Name of the strategy
            start_date: Backtest start date
            end_date: Backtest end date
            benchmark_returns: Benchmark returns for comparison
            risk_free_rate: Risk-free rate for Sharpe ratio calculation
            
        Returns:
            BacktestResults object
        """
        # Get data
        equity_df = portfolio.get_equity_dataframe()
        trades_df = portfolio.get_trades_dataframe()
        positions_df = portfolio.get_closed_positions_dataframe()
        
        if equity_df.empty:
            logger.warning("Empty equity curve, creating minimal results")
            return cls._create_empty_results(portfolio, strategy_name, start_date, end_date)
        
        # Calculate basic metrics
        initial_capital = portfolio.initial_capital
        final_capital = equity_df['total_value'].iloc[-1]
        total_return = (final_capital - initial_capital) / initial_capital
        
        # Calculate returns series
        returns = equity_df['daily_return'].dropna()
        
        if len(returns) == 0:
            logger.warning("No returns data available")
            return cls._create_empty_results(portfolio, strategy_name, start_date, end_date)
        
        # Performance metrics
        annualized_return = cls._calculate_annualized_return(returns)
        volatility = cls._calculate_volatility(returns)
        sharpe_ratio = cls._calculate_sharpe_ratio(returns, risk_free_rate)
        max_drawdown, max_dd_duration = cls._calculate_max_drawdown(equity_df['total_value'])
        
        # Trade statistics
        trade_stats = cls._calculate_trade_statistics(positions_df)
        
        # Additional metrics
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        sortino_ratio = cls._calculate_sortino_ratio(returns, risk_free_rate)
        
        # Benchmark comparison
        benchmark_return = None
        excess_return = None
        information_ratio = None
        beta = None
        alpha = None
        
        if benchmark_returns is not None:
            benchmark_stats = cls._calculate_benchmark_metrics(
                returns, benchmark_returns, risk_free_rate
            )
            benchmark_return = benchmark_stats['benchmark_return']
            excess_return = benchmark_stats['excess_return']
            information_ratio = benchmark_stats['information_ratio']
            beta = benchmark_stats['beta']
            alpha = benchmark_stats['alpha']
        
        return cls(
            strategy_name=strategy_name,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            final_capital=final_capital,
            portfolio=portfolio,
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_duration=max_dd_duration,
            total_trades=trade_stats['total_trades'],
            winning_trades=trade_stats['winning_trades'],
            losing_trades=trade_stats['losing_trades'],
            win_rate=trade_stats['win_rate'],
            avg_win=trade_stats['avg_win'],
            avg_loss=trade_stats['avg_loss'],
            profit_factor=trade_stats['profit_factor'],
            calmar_ratio=calmar_ratio,
            sortino_ratio=sortino_ratio,
            beta=beta,
            alpha=alpha,
            benchmark_return=benchmark_return,
            excess_return=excess_return,
            information_ratio=information_ratio,
            equity_curve=equity_df,
            trades_df=trades_df,
            positions_df=positions_df
        )
    
    @classmethod
    def _create_empty_results(
        cls,
        portfolio: Portfolio,
        strategy_name: str,
        start_date: datetime,
        end_date: datetime
    ) -> 'BacktestResults':
        """Create empty results for failed backtests."""
        return cls(
            strategy_name=strategy_name,
            start_date=start_date,
            end_date=end_date,
            initial_capital=portfolio.initial_capital,
            final_capital=portfolio.initial_capital,
            portfolio=portfolio,
            total_return=0.0,
            annualized_return=0.0,
            volatility=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            max_drawdown_duration=0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            profit_factor=0.0,
            calmar_ratio=0.0,
            sortino_ratio=0.0
        )
    
    @staticmethod
    def _calculate_annualized_return(returns: pd.Series) -> float:
        """Calculate annualized return."""
        if len(returns) == 0:
            return 0.0
        
        total_return = (1 + returns).prod() - 1
        years = len(returns) / 252  # Assuming 252 trading days per year
        
        if years <= 0:
            return 0.0
        
        return (1 + total_return) ** (1 / years) - 1
    
    @staticmethod
    def _calculate_volatility(returns: pd.Series) -> float:
        """Calculate annualized volatility."""
        if len(returns) == 0:
            return 0.0
        
        return returns.std() * np.sqrt(252)
    
    @staticmethod
    def _calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns - risk_free_rate / 252
        
        if excess_returns.std() == 0:
            return 0.0
        
        return excess_returns.mean() / excess_returns.std() * np.sqrt(252)
    
    @staticmethod
    def _calculate_max_drawdown(equity_curve: pd.Series) -> tuple[float, int]:
        """Calculate maximum drawdown and duration."""
        if len(equity_curve) == 0:
            return 0.0, 0
        
        # Calculate running maximum
        running_max = equity_curve.expanding().max()
        
        # Calculate drawdown
        drawdown = (equity_curve - running_max) / running_max
        
        # Find maximum drawdown
        max_drawdown = drawdown.min()
        
        # Calculate maximum drawdown duration
        max_dd_duration = 0
        current_dd_duration = 0
        
        for dd in drawdown:
            if dd < 0:
                current_dd_duration += 1
                max_dd_duration = max(max_dd_duration, current_dd_duration)
            else:
                current_dd_duration = 0
        
        return max_drawdown, max_dd_duration
    
    @staticmethod
    def _calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float) -> float:
        """Calculate Sortino ratio."""
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns - risk_free_rate / 252
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0
        
        return excess_returns.mean() / downside_returns.std() * np.sqrt(252)
    
    @staticmethod
    def _calculate_trade_statistics(positions_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate trade statistics."""
        if positions_df.empty:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0
            }
        
        total_trades = len(positions_df)
        winning_trades = len(positions_df[positions_df['pnl'] > 0])
        losing_trades = len(positions_df[positions_df['pnl'] < 0])
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        
        wins = positions_df[positions_df['pnl'] > 0]['pnl']
        losses = positions_df[positions_df['pnl'] < 0]['pnl']
        
        avg_win = wins.mean() if len(wins) > 0 else 0.0
        avg_loss = losses.mean() if len(losses) > 0 else 0.0
        
        total_wins = wins.sum() if len(wins) > 0 else 0.0
        total_losses = abs(losses.sum()) if len(losses) > 0 else 0.0
        
        profit_factor = total_wins / total_losses if total_losses > 0 else 0.0
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor
        }
    
    @staticmethod
    def _calculate_benchmark_metrics(
        returns: pd.Series,
        benchmark_returns: pd.Series,
        risk_free_rate: float
    ) -> Dict[str, Any]:
        """Calculate benchmark comparison metrics."""
        # Align returns
        aligned_returns = returns.align(benchmark_returns, join='inner')
        strategy_returns = aligned_returns[0].dropna()
        bench_returns = aligned_returns[1].dropna()
        
        if len(strategy_returns) == 0 or len(bench_returns) == 0:
            return {
                'benchmark_return': 0.0,
                'excess_return': 0.0,
                'information_ratio': 0.0,
                'beta': 0.0,
                'alpha': 0.0
            }
        
        # Calculate benchmark return
        benchmark_return = (1 + bench_returns).prod() - 1
        
        # Calculate excess return
        excess_return = (1 + strategy_returns).prod() - 1 - benchmark_return
        
        # Calculate tracking error and information ratio
        excess_returns = strategy_returns - bench_returns
        tracking_error = excess_returns.std() * np.sqrt(252)
        information_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() > 0 else 0.0
        
        # Calculate beta and alpha
        covariance = np.cov(strategy_returns, bench_returns)[0, 1]
        benchmark_variance = bench_returns.var()
        beta = covariance / benchmark_variance if benchmark_variance > 0 else 0.0
        
        strategy_mean = strategy_returns.mean() * 252
        benchmark_mean = bench_returns.mean() * 252
        alpha = strategy_mean - (risk_free_rate + beta * (benchmark_mean - risk_free_rate))
        
        return {
            'benchmark_return': benchmark_return,
            'excess_return': excess_return,
            'information_ratio': information_ratio,
            'beta': beta,
            'alpha': alpha
        }
    
    def summary(self) -> Dict[str, Any]:
        """Get summary of results."""
        return {
            'Strategy': self.strategy_name,
            'Period': f"{self.start_date.date()} to {self.end_date.date()}",
            'Initial Capital': f"${self.initial_capital:,.2f}",
            'Final Capital': f"${self.final_capital:,.2f}",
            'Total Return': f"{self.total_return:.2%}",
            'Annualized Return': f"{self.annualized_return:.2%}",
            'Volatility': f"{self.volatility:.2%}",
            'Sharpe Ratio': f"{self.sharpe_ratio:.2f}",
            'Max Drawdown': f"{self.max_drawdown:.2%}",
            'Calmar Ratio': f"{self.calmar_ratio:.2f}",
            'Total Trades': self.total_trades,
            'Win Rate': f"{self.win_rate:.2%}",
            'Profit Factor': f"{self.profit_factor:.2f}"
        }
    
    def print_summary(self) -> None:
        """Print results summary."""
        summary = self.summary()
        
        print("\n" + "="*50)
        print("BACKTEST RESULTS SUMMARY")
        print("="*50)
        
        for key, value in summary.items():
            print(f"{key:<20}: {value}")
        
        print("="*50)
