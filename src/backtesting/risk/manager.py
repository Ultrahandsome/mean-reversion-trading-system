"""Risk management for backtesting."""

from typing import Optional, Dict, Any
import numpy as np

from ..portfolio import Portfolio, Position
from ...utils.logger import get_logger
from ...utils.config import get_config

logger = get_logger(__name__)


class RiskManager:
    """Risk management for backtesting."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize risk manager.
        
        Args:
            config: Risk management configuration
        """
        self.config = config or get_config().risk_config
        
        # Risk parameters
        self.max_position_size = self.config.get('max_position_size', 0.1)
        self.max_portfolio_exposure = self.config.get('max_portfolio_exposure', 0.8)
        self.max_drawdown = self.config.get('max_drawdown', 0.2)
        self.stop_loss = self.config.get('stop_loss', 0.05)
        self.take_profit = self.config.get('take_profit', 0.15)
        
        logger.info("Initialized risk manager")
    
    def should_exit_position(
        self,
        position: Position,
        current_price: float,
        portfolio: Portfolio
    ) -> bool:
        """
        Check if position should be exited based on risk rules.
        
        Args:
            position: Position to check
            current_price: Current market price
            portfolio: Portfolio instance
            
        Returns:
            True if position should be exited
        """
        # Check portfolio-level drawdown
        if self._check_portfolio_drawdown(portfolio):
            logger.warning(f"Portfolio drawdown exceeded, exiting {position.symbol}")
            return True
        
        # Check position-level risk
        unrealized_pnl_pct = position.unrealized_pnl_pct(current_price)
        
        # Check if loss exceeds maximum allowed
        if unrealized_pnl_pct < -self.stop_loss:
            logger.info(f"Stop loss triggered for {position.symbol}: {unrealized_pnl_pct:.2%}")
            return True
        
        # Check if profit target reached
        if unrealized_pnl_pct > self.take_profit:
            logger.info(f"Take profit triggered for {position.symbol}: {unrealized_pnl_pct:.2%}")
            return True
        
        return False
    
    def adjust_position_size(
        self,
        proposed_size: float,
        symbol: str,
        price: float,
        portfolio: Portfolio
    ) -> float:
        """
        Adjust position size based on risk constraints.
        
        Args:
            proposed_size: Proposed position size
            symbol: Trading symbol
            price: Entry price
            portfolio: Portfolio instance
            
        Returns:
            Adjusted position size
        """
        # Check maximum position size
        portfolio_value = portfolio.total_value
        max_position_value = portfolio_value * self.max_position_size
        max_size_by_position = max_position_value / price
        
        adjusted_size = min(proposed_size, max_size_by_position)
        
        # Check maximum portfolio exposure
        current_exposure = portfolio.positions_value / portfolio_value
        position_value = adjusted_size * price
        new_exposure = (portfolio.positions_value + position_value) / portfolio_value
        
        if new_exposure > self.max_portfolio_exposure:
            # Reduce position size to stay within exposure limit
            max_additional_value = (self.max_portfolio_exposure * portfolio_value) - portfolio.positions_value
            max_additional_size = max(0, max_additional_value / price)
            adjusted_size = min(adjusted_size, max_additional_size)
        
        # Check available cash
        required_cash = adjusted_size * price * (1 + portfolio.commission + portfolio.slippage)
        if required_cash > portfolio.cash:
            max_size_by_cash = portfolio.cash / (price * (1 + portfolio.commission + portfolio.slippage))
            adjusted_size = min(adjusted_size, max_size_by_cash)
        
        # Ensure minimum position size
        min_position_value = 100  # Minimum $100 position
        min_size = min_position_value / price
        
        if adjusted_size < min_size:
            logger.debug(f"Position size too small for {symbol}, skipping")
            return 0
        
        if adjusted_size != proposed_size:
            logger.debug(f"Adjusted position size for {symbol}: {proposed_size:.2f} -> {adjusted_size:.2f}")
        
        return adjusted_size
    
    def _check_portfolio_drawdown(self, portfolio: Portfolio) -> bool:
        """Check if portfolio drawdown exceeds maximum allowed."""
        if not portfolio.equity_curve:
            return False
        
        # Get current portfolio value
        current_value = portfolio.total_value
        
        # Calculate peak value
        peak_value = max(point['total_value'] for point in portfolio.equity_curve)
        
        # Calculate drawdown
        drawdown = (current_value - peak_value) / peak_value
        
        return drawdown < -self.max_drawdown
    
    def calculate_var(
        self,
        portfolio: Portfolio,
        confidence_level: float = 0.05,
        time_horizon: int = 1
    ) -> float:
        """
        Calculate Value at Risk (VaR).
        
        Args:
            portfolio: Portfolio instance
            confidence_level: Confidence level (e.g., 0.05 for 95% VaR)
            time_horizon: Time horizon in days
            
        Returns:
            VaR value
        """
        if len(portfolio.daily_returns) < 30:
            return 0.0
        
        returns = np.array(portfolio.daily_returns[-252:])  # Last year of returns
        
        # Calculate VaR using historical simulation
        var = np.percentile(returns, confidence_level * 100) * np.sqrt(time_horizon)
        
        return abs(var) * portfolio.total_value
    
    def calculate_expected_shortfall(
        self,
        portfolio: Portfolio,
        confidence_level: float = 0.05
    ) -> float:
        """
        Calculate Expected Shortfall (Conditional VaR).
        
        Args:
            portfolio: Portfolio instance
            confidence_level: Confidence level
            
        Returns:
            Expected Shortfall value
        """
        if len(portfolio.daily_returns) < 30:
            return 0.0
        
        returns = np.array(portfolio.daily_returns[-252:])
        
        # Calculate VaR threshold
        var_threshold = np.percentile(returns, confidence_level * 100)
        
        # Calculate Expected Shortfall
        tail_returns = returns[returns <= var_threshold]
        
        if len(tail_returns) == 0:
            return 0.0
        
        expected_shortfall = np.mean(tail_returns)
        
        return abs(expected_shortfall) * portfolio.total_value
    
    def check_concentration_risk(self, portfolio: Portfolio) -> Dict[str, float]:
        """
        Check concentration risk in portfolio.
        
        Args:
            portfolio: Portfolio instance
            
        Returns:
            Dictionary of concentration metrics
        """
        if not portfolio.positions:
            return {'max_weight': 0.0, 'herfindahl_index': 0.0}
        
        portfolio_value = portfolio.total_value
        
        # Calculate position weights
        weights = []
        for position in portfolio.positions.values():
            weight = position.market_value / portfolio_value
            weights.append(weight)
        
        # Maximum weight
        max_weight = max(weights) if weights else 0.0
        
        # Herfindahl-Hirschman Index
        herfindahl_index = sum(w**2 for w in weights)
        
        return {
            'max_weight': max_weight,
            'herfindahl_index': herfindahl_index,
            'num_positions': len(weights)
        }
    
    def get_risk_metrics(self, portfolio: Portfolio) -> Dict[str, Any]:
        """
        Get comprehensive risk metrics.
        
        Args:
            portfolio: Portfolio instance
            
        Returns:
            Dictionary of risk metrics
        """
        metrics = {}
        
        # Portfolio exposure
        portfolio_value = portfolio.total_value
        exposure = portfolio.positions_value / portfolio_value if portfolio_value > 0 else 0
        metrics['portfolio_exposure'] = exposure
        
        # Drawdown
        if portfolio.equity_curve:
            current_value = portfolio.total_value
            peak_value = max(point['total_value'] for point in portfolio.equity_curve)
            drawdown = (current_value - peak_value) / peak_value
            metrics['current_drawdown'] = drawdown
        else:
            metrics['current_drawdown'] = 0.0
        
        # Concentration risk
        concentration = self.check_concentration_risk(portfolio)
        metrics.update(concentration)
        
        # VaR and Expected Shortfall
        metrics['var_95'] = self.calculate_var(portfolio, 0.05)
        metrics['expected_shortfall_95'] = self.calculate_expected_shortfall(portfolio, 0.05)
        
        # Volatility
        if len(portfolio.daily_returns) > 1:
            volatility = np.std(portfolio.daily_returns) * np.sqrt(252)
            metrics['annualized_volatility'] = volatility
        else:
            metrics['annualized_volatility'] = 0.0
        
        return metrics
    
    def check_risk_limits(self, portfolio: Portfolio) -> List[str]:
        """
        Check all risk limits and return violations.
        
        Args:
            portfolio: Portfolio instance
            
        Returns:
            List of risk limit violations
        """
        violations = []
        
        # Check portfolio exposure
        portfolio_value = portfolio.total_value
        exposure = portfolio.positions_value / portfolio_value if portfolio_value > 0 else 0
        
        if exposure > self.max_portfolio_exposure:
            violations.append(f"Portfolio exposure ({exposure:.2%}) exceeds limit ({self.max_portfolio_exposure:.2%})")
        
        # Check drawdown
        if portfolio.equity_curve:
            current_value = portfolio.total_value
            peak_value = max(point['total_value'] for point in portfolio.equity_curve)
            drawdown = (current_value - peak_value) / peak_value
            
            if drawdown < -self.max_drawdown:
                violations.append(f"Drawdown ({drawdown:.2%}) exceeds limit ({-self.max_drawdown:.2%})")
        
        # Check position concentration
        concentration = self.check_concentration_risk(portfolio)
        if concentration['max_weight'] > self.max_position_size:
            violations.append(f"Position concentration ({concentration['max_weight']:.2%}) exceeds limit ({self.max_position_size:.2%})")
        
        return violations
