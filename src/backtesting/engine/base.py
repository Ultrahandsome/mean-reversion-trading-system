"""Main backtesting engine for the Mean Reversion Trading System."""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np

from .results import BacktestResults
from ..portfolio import Portfolio
from ..risk import RiskManager
from ...data.models import MarketData
from ...data.providers.base import DataProvider
from ...factors.signals.base import SignalGenerator, Signal
from ...strategies.base import BaseStrategy
from ...utils.logger import get_logger
from ...utils.config import get_config

logger = get_logger(__name__)


class BacktestEngine:
    """Main backtesting engine."""
    
    def __init__(
        self,
        initial_capital: float = 100000,
        commission: float = 0.001,
        slippage: float = 0.0005,
        risk_manager: Optional[RiskManager] = None
    ):
        """
        Initialize backtesting engine.
        
        Args:
            initial_capital: Initial capital for backtesting
            commission: Commission rate per trade
            slippage: Slippage rate per trade
            risk_manager: Risk manager instance
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        
        self.risk_manager = risk_manager or RiskManager()
        self.config = get_config()
        
        # State variables
        self.portfolio: Optional[Portfolio] = None
        self.current_date: Optional[datetime] = None
        self.market_data: Dict[str, MarketData] = {}
        self.current_prices: Dict[str, float] = {}
        
        logger.info(f"Initialized backtesting engine with ${initial_capital:,.2f} capital")
    
    def run_backtest(
        self,
        strategy: BaseStrategy,
        symbols: List[str],
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        data_provider: DataProvider,
        benchmark_symbol: Optional[str] = None,
        rebalance_frequency: str = 'daily'
    ) -> BacktestResults:
        """
        Run backtest for a strategy.
        
        Args:
            strategy: Trading strategy to backtest
            symbols: List of symbols to trade
            start_date: Backtest start date
            end_date: Backtest end date
            data_provider: Data provider for market data
            benchmark_symbol: Benchmark symbol for comparison
            rebalance_frequency: How often to rebalance ('daily', 'weekly', 'monthly')
            
        Returns:
            BacktestResults object
        """
        # Convert dates
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        
        logger.info(f"Starting backtest: {strategy.name} from {start_date.date()} to {end_date.date()}")
        
        # Initialize portfolio
        self.portfolio = Portfolio(
            initial_capital=self.initial_capital,
            commission=self.commission,
            slippage=self.slippage
        )
        
        # Load market data
        self._load_market_data(symbols, start_date, end_date, data_provider)
        
        # Load benchmark data if specified
        benchmark_returns = None
        if benchmark_symbol:
            benchmark_returns = self._load_benchmark_data(
                benchmark_symbol, start_date, end_date, data_provider
            )
        
        # Get trading dates
        trading_dates = self._get_trading_dates(start_date, end_date)
        
        if not trading_dates:
            logger.error("No trading dates found")
            return BacktestResults._create_empty_results(
                self.portfolio, strategy.name, start_date, end_date
            )
        
        # Run backtest simulation
        self._run_simulation(strategy, trading_dates, rebalance_frequency)
        
        # Create results
        results = BacktestResults.from_portfolio(
            portfolio=self.portfolio,
            strategy_name=strategy.name,
            start_date=start_date,
            end_date=end_date,
            benchmark_returns=benchmark_returns,
            risk_free_rate=self.config.get('backtesting.risk_free_rate', 0.02)
        )
        
        logger.info(f"Backtest completed: Total Return {results.total_return:.2%}")
        
        return results
    
    def _load_market_data(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        data_provider: DataProvider
    ) -> None:
        """Load market data for all symbols."""
        logger.info(f"Loading market data for {len(symbols)} symbols")
        
        # Add some buffer for indicators
        buffer_days = 100
        data_start = start_date - timedelta(days=buffer_days)
        
        for symbol in symbols:
            try:
                market_data = data_provider.get_price_data(
                    symbol=symbol,
                    start_date=data_start,
                    end_date=end_date,
                    interval='1d'
                )
                
                if market_data and len(market_data.data) > 0:
                    self.market_data[symbol] = market_data
                    logger.debug(f"Loaded {len(market_data.data)} records for {symbol}")
                else:
                    logger.warning(f"No data found for {symbol}")
                    
            except Exception as e:
                logger.error(f"Error loading data for {symbol}: {e}")
    
    def _load_benchmark_data(
        self,
        benchmark_symbol: str,
        start_date: datetime,
        end_date: datetime,
        data_provider: DataProvider
    ) -> Optional[pd.Series]:
        """Load benchmark data for comparison."""
        try:
            benchmark_data = data_provider.get_price_data(
                symbol=benchmark_symbol,
                start_date=start_date,
                end_date=end_date,
                interval='1d'
            )
            
            if benchmark_data:
                returns = benchmark_data.data['close'].pct_change().dropna()
                logger.info(f"Loaded benchmark data for {benchmark_symbol}")
                return returns
            
        except Exception as e:
            logger.error(f"Error loading benchmark data: {e}")
        
        return None
    
    def _get_trading_dates(self, start_date: datetime, end_date: datetime) -> List[datetime]:
        """Get list of trading dates from market data."""
        all_dates = set()
        
        for market_data in self.market_data.values():
            dates = market_data.data.index
            filtered_dates = dates[(dates >= start_date) & (dates <= end_date)]
            all_dates.update(filtered_dates)
        
        trading_dates = sorted(list(all_dates))
        logger.info(f"Found {len(trading_dates)} trading dates")
        
        return trading_dates
    
    def _run_simulation(
        self,
        strategy: BaseStrategy,
        trading_dates: List[datetime],
        rebalance_frequency: str
    ) -> None:
        """Run the main simulation loop."""
        logger.info("Starting simulation loop")
        
        rebalance_dates = self._get_rebalance_dates(trading_dates, rebalance_frequency)
        
        for i, current_date in enumerate(trading_dates):
            self.current_date = current_date
            
            # Update current prices
            self._update_current_prices(current_date)
            
            # Check for stop losses and take profits
            self._check_risk_exits(current_date)
            
            # Generate signals on rebalance dates
            if current_date in rebalance_dates:
                self._process_strategy_signals(strategy, current_date)
            
            # Update portfolio equity curve
            self.portfolio.update_equity_curve(current_date, self.current_prices)
            
            # Log progress
            if i % 100 == 0:
                progress = (i + 1) / len(trading_dates) * 100
                logger.debug(f"Simulation progress: {progress:.1f}%")
        
        logger.info("Simulation completed")
    
    def _get_rebalance_dates(
        self,
        trading_dates: List[datetime],
        frequency: str
    ) -> List[datetime]:
        """Get rebalancing dates based on frequency."""
        if frequency == 'daily':
            return trading_dates
        
        elif frequency == 'weekly':
            # Rebalance on Mondays (or first trading day of week)
            rebalance_dates = []
            current_week = None
            
            for date in trading_dates:
                week = date.isocalendar()[1]
                if week != current_week:
                    rebalance_dates.append(date)
                    current_week = week
            
            return rebalance_dates
        
        elif frequency == 'monthly':
            # Rebalance on first trading day of month
            rebalance_dates = []
            current_month = None
            
            for date in trading_dates:
                month = date.month
                if month != current_month:
                    rebalance_dates.append(date)
                    current_month = month
            
            return rebalance_dates
        
        else:
            logger.warning(f"Unknown rebalance frequency: {frequency}, using daily")
            return trading_dates
    
    def _update_current_prices(self, current_date: datetime) -> None:
        """Update current prices for all symbols."""
        self.current_prices.clear()
        
        for symbol, market_data in self.market_data.items():
            try:
                # Find the price for current date or the most recent available
                available_dates = market_data.data.index
                valid_dates = available_dates[available_dates <= current_date]
                
                if len(valid_dates) > 0:
                    latest_date = valid_dates[-1]
                    price = market_data.data.loc[latest_date, 'close']
                    self.current_prices[symbol] = price
                
            except Exception as e:
                logger.debug(f"Could not get price for {symbol} on {current_date}: {e}")
    
    def _check_risk_exits(self, current_date: datetime) -> None:
        """Check for risk-based exits (stop loss, take profit)."""
        positions_to_close = []
        
        for symbol, position in self.portfolio.positions.items():
            if symbol not in self.current_prices:
                continue
            
            current_price = self.current_prices[symbol]
            
            # Check stop loss
            if position.should_stop_loss(current_price):
                positions_to_close.append((symbol, 'stop_loss'))
            
            # Check take profit
            elif position.should_take_profit(current_price):
                positions_to_close.append((symbol, 'take_profit'))
            
            # Check risk manager rules
            elif self.risk_manager.should_exit_position(position, current_price, self.portfolio):
                positions_to_close.append((symbol, 'risk_management'))
        
        # Close positions
        for symbol, reason in positions_to_close:
            if symbol in self.current_prices:
                self.portfolio.close_position(
                    symbol=symbol,
                    price=self.current_prices[symbol],
                    timestamp=current_date,
                    reason=reason
                )
    
    def _process_strategy_signals(self, strategy: BaseStrategy, current_date: datetime) -> None:
        """Process signals from strategy."""
        try:
            # Get available market data up to current date
            current_market_data = {}
            
            for symbol, market_data in self.market_data.items():
                # Filter data up to current date
                filtered_data = market_data.data[market_data.data.index <= current_date]
                
                if not filtered_data.empty:
                    current_market_data[symbol] = MarketData(filtered_data, symbol)
            
            # Generate signals
            signals = strategy.generate_signals(current_market_data)
            
            # Process each signal
            for signal in signals:
                self._process_signal(signal, current_date)
                
        except Exception as e:
            logger.error(f"Error processing strategy signals: {e}")
    
    def _process_signal(self, signal: Signal, current_date: datetime) -> None:
        """Process a single trading signal."""
        symbol = signal.symbol
        
        if symbol not in self.current_prices:
            logger.warning(f"No price data for signal symbol: {symbol}")
            return
        
        current_price = self.current_prices[symbol]
        
        # Entry signals
        if signal.is_entry_signal():
            # Check if we already have a position
            if self.portfolio.has_position(symbol):
                logger.debug(f"Already have position in {symbol}, ignoring entry signal")
                return
            
            # Calculate position size
            position_size = self._calculate_position_size(signal, current_price)
            
            if position_size > 0:
                # Calculate stop loss and take profit
                stop_loss, take_profit = self._calculate_risk_levels(
                    signal, current_price
                )
                
                # Open position
                success = self.portfolio.open_position(
                    symbol=symbol,
                    quantity=position_size,
                    price=current_price,
                    timestamp=current_date,
                    direction=signal.direction,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    metadata={
                        'signal_strength': signal.strength,
                        'signal_confidence': signal.confidence,
                        'signal_metadata': signal.metadata
                    }
                )
                
                if success:
                    logger.debug(f"Opened position: {symbol} {signal.direction} @ ${current_price:.2f}")
        
        # Exit signals
        elif signal.is_exit_signal():
            if self.portfolio.has_position(symbol):
                success = self.portfolio.close_position(
                    symbol=symbol,
                    price=current_price,
                    timestamp=current_date,
                    reason='signal'
                )
                
                if success:
                    logger.debug(f"Closed position: {symbol} @ ${current_price:.2f}")
    
    def _calculate_position_size(self, signal: Signal, price: float) -> float:
        """Calculate position size for a signal."""
        # Get position sizing parameters
        max_position_size = self.config.get('risk_management.max_position_size', 0.1)
        position_sizing_method = self.config.get('risk_management.position_sizing_method', 'equal_weight')
        
        # Calculate base position size
        portfolio_value = self.portfolio.total_value
        max_position_value = portfolio_value * max_position_size
        
        # Adjust based on signal strength and confidence
        signal_adjustment = (signal.strength + signal.confidence) / 2
        adjusted_position_value = max_position_value * signal_adjustment
        
        # Calculate quantity
        position_size = adjusted_position_value / price
        
        # Apply risk manager constraints
        position_size = self.risk_manager.adjust_position_size(
            position_size, signal.symbol, price, self.portfolio
        )
        
        return position_size
    
    def _calculate_risk_levels(
        self,
        signal: Signal,
        current_price: float
    ) -> tuple[Optional[float], Optional[float]]:
        """Calculate stop loss and take profit levels."""
        stop_loss_pct = self.config.get('risk_management.stop_loss', 0.05)
        take_profit_pct = self.config.get('risk_management.take_profit', 0.15)
        
        if signal.is_long_signal():
            stop_loss = current_price * (1 - stop_loss_pct)
            take_profit = current_price * (1 + take_profit_pct)
        else:
            stop_loss = current_price * (1 + stop_loss_pct)
            take_profit = current_price * (1 - take_profit_pct)
        
        return stop_loss, take_profit
