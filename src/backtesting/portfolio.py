"""Portfolio and position management for backtesting."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np

# from ..utils.logger import get_logger

# Simple logger replacement for standalone testing
class SimpleLogger:
    def info(self, msg): print(f"INFO: {msg}")
    def warning(self, msg): print(f"WARNING: {msg}")
    def debug(self, msg): pass  # Skip debug messages
    def error(self, msg): print(f"ERROR: {msg}")

logger = SimpleLogger()


@dataclass
class Position:
    """Represents a trading position."""
    
    symbol: str
    quantity: float
    entry_price: float
    entry_time: datetime
    direction: int  # 1 for long, -1 for short
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate position data."""
        if self.direction not in [-1, 1]:
            raise ValueError("Direction must be 1 (long) or -1 (short)")
        
        if self.quantity <= 0:
            raise ValueError("Quantity must be positive")
        
        if self.entry_price <= 0:
            raise ValueError("Entry price must be positive")
    
    @property
    def market_value(self) -> float:
        """Get current market value (requires current price)."""
        # This will be calculated by the portfolio using current market price
        return abs(self.quantity) * self.entry_price
    
    @property
    def is_long(self) -> bool:
        """Check if position is long."""
        return self.direction == 1
    
    @property
    def is_short(self) -> bool:
        """Check if position is short."""
        return self.direction == -1
    
    def unrealized_pnl(self, current_price: float) -> float:
        """
        Calculate unrealized P&L.
        
        Args:
            current_price: Current market price
            
        Returns:
            Unrealized P&L
        """
        if self.is_long:
            return self.quantity * (current_price - self.entry_price)
        else:
            return self.quantity * (self.entry_price - current_price)
    
    def unrealized_pnl_pct(self, current_price: float) -> float:
        """
        Calculate unrealized P&L percentage.
        
        Args:
            current_price: Current market price
            
        Returns:
            Unrealized P&L percentage
        """
        pnl = self.unrealized_pnl(current_price)
        return pnl / (abs(self.quantity) * self.entry_price)
    
    def should_stop_loss(self, current_price: float) -> bool:
        """Check if stop loss should be triggered."""
        if self.stop_loss is None:
            return False
        
        if self.is_long:
            return current_price <= self.stop_loss
        else:
            return current_price >= self.stop_loss
    
    def should_take_profit(self, current_price: float) -> bool:
        """Check if take profit should be triggered."""
        if self.take_profit is None:
            return False
        
        if self.is_long:
            return current_price >= self.take_profit
        else:
            return current_price <= self.take_profit
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert position to dictionary."""
        return {
            'symbol': self.symbol,
            'quantity': self.quantity,
            'entry_price': self.entry_price,
            'entry_time': self.entry_time,
            'direction': self.direction,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'metadata': self.metadata
        }


class Portfolio:
    """Portfolio management for backtesting."""
    
    def __init__(self, initial_capital: float, commission: float = 0.001, slippage: float = 0.0005):
        """
        Initialize portfolio.
        
        Args:
            initial_capital: Initial capital amount
            commission: Commission rate per trade
            slippage: Slippage rate per trade
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.commission = commission
        self.slippage = slippage
        
        self.positions: Dict[str, Position] = {}
        self.closed_positions: List[Dict[str, Any]] = []
        self.trades: List[Dict[str, Any]] = []
        
        # Performance tracking
        self.equity_curve: List[Dict[str, Any]] = []
        self.daily_returns: List[float] = []
        
        logger.info(f"Initialized portfolio with ${initial_capital:,.2f}")
    
    @property
    def total_value(self) -> float:
        """Get total portfolio value."""
        return self.cash + sum(pos.market_value for pos in self.positions.values())
    
    @property
    def positions_value(self) -> float:
        """Get total value of all positions."""
        return sum(pos.market_value for pos in self.positions.values())
    
    @property
    def num_positions(self) -> int:
        """Get number of open positions."""
        return len(self.positions)
    
    @property
    def position_symbols(self) -> List[str]:
        """Get list of symbols with open positions."""
        return list(self.positions.keys())
    
    def has_position(self, symbol: str) -> bool:
        """Check if portfolio has position in symbol."""
        return symbol in self.positions
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for symbol."""
        return self.positions.get(symbol)
    
    def calculate_position_size(
        self,
        symbol: str,
        price: float,
        risk_amount: float,
        stop_loss: Optional[float] = None,
        method: str = 'fixed_amount'
    ) -> float:
        """
        Calculate position size based on risk management rules.
        
        Args:
            symbol: Trading symbol
            price: Entry price
            risk_amount: Amount to risk
            stop_loss: Stop loss price
            method: Position sizing method
            
        Returns:
            Position size (number of shares/units)
        """
        if method == 'fixed_amount':
            return risk_amount / price
        
        elif method == 'fixed_percent':
            portfolio_value = self.total_value
            risk_value = portfolio_value * risk_amount
            return risk_value / price
        
        elif method == 'volatility_adjusted':
            # This would require volatility data
            # For now, use fixed amount
            return risk_amount / price
        
        elif method == 'kelly':
            # Kelly criterion would require win rate and average win/loss
            # For now, use fixed amount
            return risk_amount / price
        
        else:
            raise ValueError(f"Unsupported position sizing method: {method}")
    
    def open_position(
        self,
        symbol: str,
        quantity: float,
        price: float,
        timestamp: datetime,
        direction: int,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Open a new position.
        
        Args:
            symbol: Trading symbol
            quantity: Position quantity
            price: Entry price
            timestamp: Entry timestamp
            direction: Position direction (1 for long, -1 for short)
            stop_loss: Stop loss price
            take_profit: Take profit price
            metadata: Additional metadata
            
        Returns:
            True if position opened successfully
        """
        if symbol in self.positions:
            logger.warning(f"Position already exists for {symbol}")
            return False
        
        # Calculate costs
        trade_value = abs(quantity) * price
        commission_cost = trade_value * self.commission
        slippage_cost = trade_value * self.slippage
        total_cost = trade_value + commission_cost + slippage_cost
        
        # Check if we have enough cash
        if total_cost > self.cash:
            logger.warning(f"Insufficient cash for {symbol} position. Required: ${total_cost:.2f}, Available: ${self.cash:.2f}")
            return False
        
        # Create position
        position = Position(
            symbol=symbol,
            quantity=abs(quantity),
            entry_price=price,
            entry_time=timestamp,
            direction=direction,
            stop_loss=stop_loss,
            take_profit=take_profit,
            metadata=metadata or {}
        )
        
        # Update portfolio
        self.positions[symbol] = position
        self.cash -= total_cost
        
        # Record trade
        trade = {
            'timestamp': timestamp,
            'symbol': symbol,
            'action': 'open',
            'quantity': quantity,
            'price': price,
            'direction': direction,
            'commission': commission_cost,
            'slippage': slippage_cost,
            'cash_after': self.cash
        }
        self.trades.append(trade)
        
        logger.info(f"Opened {direction} position in {symbol}: {quantity} @ ${price:.2f}")
        return True
    
    def close_position(
        self,
        symbol: str,
        price: float,
        timestamp: datetime,
        reason: str = 'signal'
    ) -> bool:
        """
        Close an existing position.
        
        Args:
            symbol: Trading symbol
            price: Exit price
            timestamp: Exit timestamp
            reason: Reason for closing
            
        Returns:
            True if position closed successfully
        """
        if symbol not in self.positions:
            logger.warning(f"No position to close for {symbol}")
            return False
        
        position = self.positions[symbol]
        
        # Calculate P&L
        pnl = position.unrealized_pnl(price)
        pnl_pct = position.unrealized_pnl_pct(price)
        
        # Calculate costs
        trade_value = position.quantity * price
        commission_cost = trade_value * self.commission
        slippage_cost = trade_value * self.slippage
        
        # Update cash
        proceeds = trade_value - commission_cost - slippage_cost
        self.cash += proceeds
        
        # Record closed position
        closed_position = {
            'symbol': symbol,
            'entry_time': position.entry_time,
            'exit_time': timestamp,
            'entry_price': position.entry_price,
            'exit_price': price,
            'quantity': position.quantity,
            'direction': position.direction,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'holding_period': (timestamp - position.entry_time).days,
            'reason': reason,
            'commission': commission_cost,
            'slippage': slippage_cost
        }
        self.closed_positions.append(closed_position)
        
        # Record trade
        trade = {
            'timestamp': timestamp,
            'symbol': symbol,
            'action': 'close',
            'quantity': position.quantity,
            'price': price,
            'direction': -position.direction,  # Opposite direction for closing
            'pnl': pnl,
            'commission': commission_cost,
            'slippage': slippage_cost,
            'cash_after': self.cash,
            'reason': reason
        }
        self.trades.append(trade)
        
        # Remove position
        del self.positions[symbol]
        
        logger.info(f"Closed position in {symbol}: P&L ${pnl:.2f} ({pnl_pct:.2%})")
        return True
    
    def update_equity_curve(self, timestamp: datetime, market_prices: Dict[str, float]) -> None:
        """
        Update equity curve with current market values.
        
        Args:
            timestamp: Current timestamp
            market_prices: Dictionary of symbol -> current price
        """
        # Calculate current portfolio value
        positions_value = 0
        unrealized_pnl = 0
        
        for symbol, position in self.positions.items():
            if symbol in market_prices:
                current_price = market_prices[symbol]
                pos_value = position.quantity * current_price
                positions_value += pos_value
                unrealized_pnl += position.unrealized_pnl(current_price)
        
        total_value = self.cash + positions_value
        
        # Calculate daily return
        if self.equity_curve:
            prev_value = self.equity_curve[-1]['total_value']
            daily_return = (total_value - prev_value) / prev_value
            self.daily_returns.append(daily_return)
        else:
            daily_return = 0.0
            self.daily_returns.append(0.0)
        
        # Record equity point
        equity_point = {
            'timestamp': timestamp,
            'cash': self.cash,
            'positions_value': positions_value,
            'total_value': total_value,
            'unrealized_pnl': unrealized_pnl,
            'daily_return': daily_return,
            'num_positions': len(self.positions)
        }
        self.equity_curve.append(equity_point)
    
    def get_equity_dataframe(self) -> pd.DataFrame:
        """Get equity curve as DataFrame."""
        if not self.equity_curve:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.equity_curve)
        df.set_index('timestamp', inplace=True)
        return df
    
    def get_trades_dataframe(self) -> pd.DataFrame:
        """Get trades as DataFrame."""
        if not self.trades:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.trades)
        df.set_index('timestamp', inplace=True)
        return df
    
    def get_closed_positions_dataframe(self) -> pd.DataFrame:
        """Get closed positions as DataFrame."""
        if not self.closed_positions:
            return pd.DataFrame()
        
        return pd.DataFrame(self.closed_positions)
