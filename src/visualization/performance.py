"""Performance visualization tools for backtesting results."""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..backtesting.engine.results import BacktestResults
from ..utils.logger import get_logger

logger = get_logger(__name__)

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class PerformanceVisualizer:
    """Visualization tools for backtesting performance analysis."""
    
    def __init__(self, results: BacktestResults):
        """
        Initialize performance visualizer.
        
        Args:
            results: BacktestResults object
        """
        self.results = results
        
    def plot_equity_curve(self, figsize: tuple = (12, 6), save_path: Optional[str] = None) -> None:
        """
        Plot equity curve.
        
        Args:
            figsize: Figure size
            save_path: Path to save the plot
        """
        if self.results.equity_curve.empty:
            logger.warning("No equity curve data to plot")
            return
        
        fig, ax = plt.subplots(figsize=figsize)
        
        equity_curve = self.results.equity_curve
        
        # Plot equity curve
        ax.plot(equity_curve.index, equity_curve['total_value'], 
                linewidth=2, label='Portfolio Value', color='blue')
        
        # Add initial capital line
        ax.axhline(y=self.results.initial_capital, color='gray', 
                  linestyle='--', alpha=0.7, label='Initial Capital')
        
        # Formatting
        ax.set_title(f'{self.results.strategy_name} - Equity Curve', fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Portfolio Value ($)', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format y-axis as currency
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Equity curve saved to {save_path}")
        
        plt.show()
    
    def plot_drawdown(self, figsize: tuple = (12, 6), save_path: Optional[str] = None) -> None:
        """
        Plot drawdown chart.
        
        Args:
            figsize: Figure size
            save_path: Path to save the plot
        """
        if self.results.equity_curve.empty:
            logger.warning("No equity curve data to plot")
            return
        
        fig, ax = plt.subplots(figsize=figsize)
        
        equity_curve = self.results.equity_curve['total_value']
        
        # Calculate drawdown
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max
        
        # Plot drawdown
        ax.fill_between(drawdown.index, drawdown, 0, alpha=0.7, color='red', label='Drawdown')
        ax.plot(drawdown.index, drawdown, color='darkred', linewidth=1)
        
        # Add max drawdown line
        ax.axhline(y=self.results.max_drawdown, color='red', 
                  linestyle='--', alpha=0.8, label=f'Max Drawdown ({self.results.max_drawdown:.2%})')
        
        # Formatting
        ax.set_title(f'{self.results.strategy_name} - Drawdown Analysis', fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Drawdown (%)', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Drawdown chart saved to {save_path}")
        
        plt.show()
    
    def plot_returns_distribution(self, figsize: tuple = (12, 6), save_path: Optional[str] = None) -> None:
        """
        Plot returns distribution.
        
        Args:
            figsize: Figure size
            save_path: Path to save the plot
        """
        if self.results.equity_curve.empty:
            logger.warning("No equity curve data to plot")
            return
        
        returns = self.results.equity_curve['daily_return'].dropna()
        
        if returns.empty:
            logger.warning("No returns data to plot")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Histogram
        ax1.hist(returns, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax1.axvline(returns.mean(), color='red', linestyle='--', 
                   label=f'Mean: {returns.mean():.4f}')
        ax1.axvline(returns.median(), color='green', linestyle='--', 
                   label=f'Median: {returns.median():.4f}')
        
        ax1.set_title('Daily Returns Distribution', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Daily Return', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(returns, dist="norm", plot=ax2)
        ax2.set_title('Q-Q Plot (Normal Distribution)', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Returns distribution saved to {save_path}")
        
        plt.show()
    
    def plot_monthly_returns(self, figsize: tuple = (14, 8), save_path: Optional[str] = None) -> None:
        """
        Plot monthly returns heatmap.
        
        Args:
            figsize: Figure size
            save_path: Path to save the plot
        """
        if self.results.equity_curve.empty:
            logger.warning("No equity curve data to plot")
            return
        
        returns = self.results.equity_curve['daily_return'].dropna()
        
        if returns.empty:
            logger.warning("No returns data to plot")
            return
        
        # Calculate monthly returns
        monthly_returns = (1 + returns).resample('M').prod() - 1
        
        # Create pivot table for heatmap
        monthly_returns.index = pd.to_datetime(monthly_returns.index)
        monthly_returns_pivot = monthly_returns.groupby([
            monthly_returns.index.year,
            monthly_returns.index.month
        ]).first().unstack()
        
        # Create month labels
        month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        sns.heatmap(monthly_returns_pivot, 
                   annot=True, 
                   fmt='.2%', 
                   cmap='RdYlGn', 
                   center=0,
                   xticklabels=month_labels,
                   yticklabels=True,
                   ax=ax)
        
        ax.set_title(f'{self.results.strategy_name} - Monthly Returns Heatmap', 
                    fontsize=16, fontweight='bold')
        ax.set_xlabel('Month', fontsize=12)
        ax.set_ylabel('Year', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Monthly returns heatmap saved to {save_path}")
        
        plt.show()
    
    def plot_performance_summary(self, figsize: tuple = (16, 12), save_path: Optional[str] = None) -> None:
        """
        Plot comprehensive performance summary.
        
        Args:
            figsize: Figure size
            save_path: Path to save the plot
        """
        fig = plt.figure(figsize=figsize)
        
        # Create subplots
        gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 1], hspace=0.3, wspace=0.3)
        
        # 1. Equity Curve
        ax1 = fig.add_subplot(gs[0, :])
        if not self.results.equity_curve.empty:
            equity_curve = self.results.equity_curve
            ax1.plot(equity_curve.index, equity_curve['total_value'], 
                    linewidth=2, color='blue', label='Portfolio Value')
            ax1.axhline(y=self.results.initial_capital, color='gray', 
                       linestyle='--', alpha=0.7, label='Initial Capital')
            ax1.set_title(f'{self.results.strategy_name} - Performance Summary', 
                         fontsize=16, fontweight='bold')
            ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # 2. Drawdown
        ax2 = fig.add_subplot(gs[1, 0])
        if not self.results.equity_curve.empty:
            equity_curve = self.results.equity_curve['total_value']
            running_max = equity_curve.expanding().max()
            drawdown = (equity_curve - running_max) / running_max
            ax2.fill_between(drawdown.index, drawdown, 0, alpha=0.7, color='red')
            ax2.set_title('Drawdown', fontsize=14, fontweight='bold')
            ax2.set_ylabel('Drawdown (%)', fontsize=10)
            ax2.grid(True, alpha=0.3)
            ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
        
        # 3. Returns Distribution
        ax3 = fig.add_subplot(gs[1, 1])
        if not self.results.equity_curve.empty:
            returns = self.results.equity_curve['daily_return'].dropna()
            if not returns.empty:
                ax3.hist(returns, bins=30, alpha=0.7, color='blue', edgecolor='black')
                ax3.axvline(returns.mean(), color='red', linestyle='--')
                ax3.set_title('Daily Returns Distribution', fontsize=14, fontweight='bold')
                ax3.set_xlabel('Daily Return', fontsize=10)
                ax3.set_ylabel('Frequency', fontsize=10)
                ax3.grid(True, alpha=0.3)
        
        # 4. Performance Metrics Table
        ax4 = fig.add_subplot(gs[2, :])
        ax4.axis('off')
        
        # Create metrics table
        metrics_data = [
            ['Total Return', f'{self.results.total_return:.2%}'],
            ['Annualized Return', f'{self.results.annualized_return:.2%}'],
            ['Volatility', f'{self.results.volatility:.2%}'],
            ['Sharpe Ratio', f'{self.results.sharpe_ratio:.2f}'],
            ['Max Drawdown', f'{self.results.max_drawdown:.2%}'],
            ['Calmar Ratio', f'{self.results.calmar_ratio:.2f}'],
            ['Total Trades', f'{self.results.total_trades}'],
            ['Win Rate', f'{self.results.win_rate:.2%}'],
            ['Profit Factor', f'{self.results.profit_factor:.2f}']
        ]
        
        # Split into two columns
        mid_point = len(metrics_data) // 2
        left_metrics = metrics_data[:mid_point]
        right_metrics = metrics_data[mid_point:]
        
        # Create table
        table_data = []
        for i in range(max(len(left_metrics), len(right_metrics))):
            row = []
            if i < len(left_metrics):
                row.extend(left_metrics[i])
            else:
                row.extend(['', ''])
            
            if i < len(right_metrics):
                row.extend(right_metrics[i])
            else:
                row.extend(['', ''])
            
            table_data.append(row)
        
        table = ax4.table(cellText=table_data,
                         colLabels=['Metric', 'Value', 'Metric', 'Value'],
                         cellLoc='center',
                         loc='center',
                         colWidths=[0.2, 0.15, 0.2, 0.15])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style the table
        for i in range(len(table_data) + 1):
            for j in range(4):
                cell = table[(i, j)]
                if i == 0:  # Header row
                    cell.set_facecolor('#4CAF50')
                    cell.set_text_props(weight='bold', color='white')
                else:
                    if j % 2 == 0:  # Metric columns
                        cell.set_facecolor('#f0f0f0')
                    else:  # Value columns
                        cell.set_facecolor('#ffffff')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Performance summary saved to {save_path}")
        
        plt.show()
    
    def create_interactive_dashboard(self) -> go.Figure:
        """
        Create interactive dashboard using Plotly.
        
        Returns:
            Plotly figure object
        """
        if self.results.equity_curve.empty:
            logger.warning("No data available for interactive dashboard")
            return go.Figure()
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Equity Curve', 'Drawdown', 
                          'Daily Returns', 'Monthly Returns',
                          'Trade Distribution', 'Performance Metrics'),
            specs=[[{"colspan": 2}, None],
                   [{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "table"}]],
            vertical_spacing=0.08
        )
        
        equity_curve = self.results.equity_curve
        
        # 1. Equity Curve
        fig.add_trace(
            go.Scatter(x=equity_curve.index, y=equity_curve['total_value'],
                      mode='lines', name='Portfolio Value',
                      line=dict(color='blue', width=2)),
            row=1, col=1
        )
        
        # Add initial capital line
        fig.add_hline(y=self.results.initial_capital, 
                     line_dash="dash", line_color="gray",
                     annotation_text="Initial Capital",
                     row=1, col=1)
        
        # 2. Drawdown
        running_max = equity_curve['total_value'].expanding().max()
        drawdown = (equity_curve['total_value'] - running_max) / running_max
        
        fig.add_trace(
            go.Scatter(x=drawdown.index, y=drawdown,
                      fill='tonexty', mode='lines',
                      name='Drawdown', line=dict(color='red')),
            row=2, col=1
        )
        
        # 3. Daily Returns
        returns = equity_curve['daily_return'].dropna()
        if not returns.empty:
            fig.add_trace(
                go.Scatter(x=returns.index, y=returns,
                          mode='markers', name='Daily Returns',
                          marker=dict(color='green', size=3)),
                row=2, col=2
            )
        
        # 4. Monthly Returns (if enough data)
        if len(returns) > 30:
            monthly_returns = (1 + returns).resample('M').prod() - 1
            fig.add_trace(
                go.Bar(x=monthly_returns.index, y=monthly_returns,
                      name='Monthly Returns', marker_color='orange'),
                row=3, col=1
            )
        
        # 5. Performance Metrics Table
        metrics_data = [
            ['Total Return', f'{self.results.total_return:.2%}'],
            ['Annualized Return', f'{self.results.annualized_return:.2%}'],
            ['Sharpe Ratio', f'{self.results.sharpe_ratio:.2f}'],
            ['Max Drawdown', f'{self.results.max_drawdown:.2%}'],
            ['Win Rate', f'{self.results.win_rate:.2%}'],
            ['Total Trades', f'{self.results.total_trades}']
        ]
        
        fig.add_trace(
            go.Table(
                header=dict(values=['Metric', 'Value'],
                           fill_color='lightblue',
                           align='left'),
                cells=dict(values=list(zip(*metrics_data)),
                          fill_color='white',
                          align='left')
            ),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(
            title=f'{self.results.strategy_name} - Interactive Performance Dashboard',
            height=1000,
            showlegend=True
        )
        
        return fig
