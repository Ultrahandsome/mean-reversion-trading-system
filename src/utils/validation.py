"""Data validation utilities for the Mean Reversion Trading System."""

import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Union
from .logger import get_logger

logger = get_logger(__name__)


def validate_data(
    data: pd.DataFrame,
    required_columns: Optional[List[str]] = None,
    min_rows: int = 1,
    check_nulls: bool = True,
    check_duplicates: bool = True
) -> bool:
    """
    Validate DataFrame for trading data requirements.
    
    Args:
        data: DataFrame to validate
        required_columns: List of required column names
        min_rows: Minimum number of rows required
        check_nulls: Whether to check for null values
        check_duplicates: Whether to check for duplicate rows
        
    Returns:
        True if validation passes
        
    Raises:
        ValueError: If validation fails
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Data must be a pandas DataFrame")
    
    # Check minimum rows
    if len(data) < min_rows:
        raise ValueError(f"Data must have at least {min_rows} rows, got {len(data)}")
    
    # Check required columns
    if required_columns:
        missing_columns = set(required_columns) - set(data.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Check for null values
    if check_nulls:
        null_counts = data.isnull().sum()
        null_columns = null_counts[null_counts > 0]
        if not null_columns.empty:
            logger.warning(f"Found null values in columns: {null_columns.to_dict()}")
    
    # Check for duplicates
    if check_duplicates:
        duplicate_count = data.duplicated().sum()
        if duplicate_count > 0:
            logger.warning(f"Found {duplicate_count} duplicate rows")
    
    logger.debug(f"Data validation passed for DataFrame with shape {data.shape}")
    return True


def validate_price_data(data: pd.DataFrame) -> bool:
    """
    Validate price data DataFrame.
    
    Args:
        data: Price data DataFrame
        
    Returns:
        True if validation passes
    """
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    
    # Basic validation
    validate_data(data, required_columns=required_columns, min_rows=1)
    
    # Price-specific validations
    price_columns = ['open', 'high', 'low', 'close']
    
    # Check for negative prices
    for col in price_columns:
        if (data[col] < 0).any():
            raise ValueError(f"Found negative prices in column '{col}'")
    
    # Check OHLC relationships
    if not ((data['high'] >= data['low']) & 
            (data['high'] >= data['open']) & 
            (data['high'] >= data['close']) &
            (data['low'] <= data['open']) & 
            (data['low'] <= data['close'])).all():
        raise ValueError("OHLC price relationships are invalid")
    
    # Check for zero volume
    if (data['volume'] == 0).any():
        logger.warning("Found zero volume periods")
    
    # Check for negative volume
    if (data['volume'] < 0).any():
        raise ValueError("Found negative volume values")
    
    logger.debug("Price data validation passed")
    return True


def validate_returns(returns: Union[pd.Series, np.ndarray]) -> bool:
    """
    Validate returns data.
    
    Args:
        returns: Returns data
        
    Returns:
        True if validation passes
    """
    if isinstance(returns, pd.Series):
        returns_array = returns.values
    else:
        returns_array = np.array(returns)
    
    # Check for infinite values
    if np.isinf(returns_array).any():
        raise ValueError("Returns contain infinite values")
    
    # Check for extreme returns (> 100% or < -100%)
    extreme_returns = np.abs(returns_array) > 1.0
    if extreme_returns.any():
        extreme_count = extreme_returns.sum()
        logger.warning(f"Found {extreme_count} extreme returns (>100% or <-100%)")
    
    logger.debug("Returns validation passed")
    return True


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration dictionary.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if validation passes
    """
    required_sections = ['data', 'strategies', 'risk_management', 'backtesting']
    
    # Check required sections
    missing_sections = set(required_sections) - set(config.keys())
    if missing_sections:
        raise ValueError(f"Missing required configuration sections: {missing_sections}")
    
    # Validate data section
    data_config = config.get('data', {})
    if 'providers' not in data_config:
        raise ValueError("Data configuration must include 'providers' section")
    
    # Validate risk management section
    risk_config = config.get('risk_management', {})
    required_risk_params = ['max_position_size', 'max_portfolio_exposure']
    missing_risk_params = set(required_risk_params) - set(risk_config.keys())
    if missing_risk_params:
        raise ValueError(f"Missing required risk management parameters: {missing_risk_params}")
    
    # Validate risk parameter ranges
    max_position_size = risk_config.get('max_position_size', 0)
    if not 0 < max_position_size <= 1:
        raise ValueError("max_position_size must be between 0 and 1")
    
    max_portfolio_exposure = risk_config.get('max_portfolio_exposure', 0)
    if not 0 < max_portfolio_exposure <= 1:
        raise ValueError("max_portfolio_exposure must be between 0 and 1")
    
    # Validate backtesting section
    backtesting_config = config.get('backtesting', {})
    initial_capital = backtesting_config.get('initial_capital', 0)
    if initial_capital <= 0:
        raise ValueError("initial_capital must be positive")
    
    logger.debug("Configuration validation passed")
    return True


def validate_symbol(symbol: str) -> bool:
    """
    Validate trading symbol format.
    
    Args:
        symbol: Trading symbol
        
    Returns:
        True if validation passes
    """
    if not isinstance(symbol, str):
        raise ValueError("Symbol must be a string")
    
    if not symbol.strip():
        raise ValueError("Symbol cannot be empty")
    
    # Basic symbol format validation
    symbol = symbol.upper().strip()
    
    # Check for valid characters (letters, numbers, hyphens, dots)
    import re
    if not re.match(r'^[A-Z0-9.-]+$', symbol):
        raise ValueError(f"Invalid symbol format: {symbol}")
    
    logger.debug(f"Symbol validation passed for: {symbol}")
    return True


def validate_date_range(start_date: str, end_date: str) -> bool:
    """
    Validate date range.
    
    Args:
        start_date: Start date string
        end_date: End date string
        
    Returns:
        True if validation passes
    """
    try:
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
    except Exception as e:
        raise ValueError(f"Invalid date format: {e}")
    
    if start >= end:
        raise ValueError("Start date must be before end date")
    
    # Check if dates are too far in the future
    now = pd.Timestamp.now()
    if start > now or end > now:
        logger.warning("Date range extends into the future")
    
    logger.debug(f"Date range validation passed: {start_date} to {end_date}")
    return True
