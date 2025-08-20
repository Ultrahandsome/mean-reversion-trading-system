# API Reference

## Core Modules

### Data Infrastructure

#### `src.data.models`

**PriceData**
```python
@dataclass
class PriceData:
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    adjusted_close: Optional[float] = None
```

**MarketData**
```python
class MarketData:
    def __init__(self, data: pd.DataFrame, symbol: str)
    def get_price_data(self, start_date: Optional[datetime] = None, 
                      end_date: Optional[datetime] = None) -> pd.DataFrame
    def get_returns(self, start_date: Optional[datetime] = None,
                   end_date: Optional[datetime] = None,
                   log_returns: bool = False) -> pd.Series
    def resample(self, frequency: str) -> 'MarketData'
```

**Symbol**
```python
@dataclass
class Symbol:
    symbol: str
    name: Optional[str] = None
    exchange: Optional[str] = None
    asset_type: str = "stock"
    currency: str = "USD"
    sector: Optional[str] = None
    industry: Optional[str] = None
    market_cap: Optional[float] = None
    is_active: bool = True
    metadata: Optional[Dict[str, Any]] = None
```

#### `src.data.providers`

**DataProvider (Base Class)**
```python
class DataProvider(ABC):
    @abstractmethod
    def get_price_data(self, symbol: str, start_date: datetime, 
                      end_date: datetime, interval: str = "1d") -> Optional[MarketData]
    
    @abstractmethod
    def get_latest_price(self, symbol: str) -> Optional[PriceData]
    
    @abstractmethod
    def search_symbols(self, query: str) -> List[Symbol]
    
    @abstractmethod
    def get_symbol_info(self, symbol: str) -> Optional[Symbol]
```

**YahooFinanceProvider**
```python
class YahooFinanceProvider(DataProvider):
    def __init__(self, rate_limit: int = 2000)
    def get_price_data(self, symbol: str, start_date: datetime, 
                      end_date: datetime, interval: str = "1d") -> Optional[MarketData]
    def get_dividends(self, symbol: str, start_date: datetime, 
                     end_date: datetime) -> pd.DataFrame
    def get_splits(self, symbol: str, start_date: datetime, 
                  end_date: datetime) -> pd.DataFrame
```

#### `src.data.storage`

**SQLiteStorage**
```python
class SQLiteStorage(DataStorage):
    def __init__(self, db_path: str = "data/market_data.db")
    def save_symbol(self, symbol: Symbol) -> None
    def get_symbol(self, symbol: str) -> Optional[Symbol]
    def save_price_data(self, price_data: List[PriceData]) -> None
    def get_price_data(self, symbol: str, start_date: Optional[datetime] = None,
                      end_date: Optional[datetime] = None) -> Optional[MarketData]
```

### Factor Research Framework

#### `src.factors.indicators`

**MeanReversionIndicators**
```python
class MeanReversionIndicators:
    def zscore(self, market_data: MarketData, period: int, 
              price_column: str = 'close') -> pd.Series
    def bollinger_position(self, market_data: MarketData, period: int = 20, 
                          num_std: float = 2.0) -> pd.Series
    def mean_reversion_ratio(self, market_data: MarketData, period: int) -> pd.Series
    def hurst_exponent(self, market_data: MarketData, period: int) -> pd.Series
    def variance_ratio(self, market_data: MarketData, period: int, lag: int = 2) -> pd.Series
```

**MovingAverages**
```python
class MovingAverages:
    def sma(self, market_data: MarketData, period: int) -> pd.Series
    def ema(self, market_data: MarketData, period: int) -> pd.Series
    def wma(self, market_data: MarketData, period: int) -> pd.Series
    def hull_ma(self, market_data: MarketData, period: int) -> pd.Series
    def vwma(self, market_data: MarketData, period: int) -> pd.Series
```

**VolatilityIndicators**
```python
class VolatilityIndicators:
    def historical_volatility(self, market_data: MarketData, period: int = 20) -> pd.Series
    def bollinger_bands(self, market_data: MarketData, period: int = 20, 
                       num_std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]
    def atr(self, market_data: MarketData, period: int = 14) -> pd.Series
    def keltner_channels(self, market_data: MarketData, period: int = 20) -> Tuple[pd.Series, pd.Series, pd.Series]
```

**MomentumIndicators**
```python
class MomentumIndicators:
    def rsi(self, market_data: MarketData, period: int = 14) -> pd.Series
    def stochastic(self, market_data: MarketData, k_period: int = 14, 
                  d_period: int = 3) -> tuple[pd.Series, pd.Series]
    def macd(self, market_data: MarketData, fast_period: int = 12, 
            slow_period: int = 26, signal_period: int = 9) -> tuple[pd.Series, pd.Series, pd.Series]
    def williams_r(self, market_data: MarketData, period: int = 14) -> pd.Series
```

#### `src.factors.signals`

**Signal**
```python
@dataclass
class Signal:
    timestamp: datetime
    symbol: str
    signal_type: str  # 'entry', 'exit', 'stop'
    direction: int    # 1 for long, -1 for short, 0 for neutral
    strength: float   # Signal strength (0-1)
    price: float      # Price at signal generation
    confidence: float = 0.5  # Confidence level (0-1)
    metadata: Optional[Dict[str, Any]] = None
```

**SignalGenerator (Base Class)**
```python
class SignalGenerator(ABC):
    @abstractmethod
    def generate_signals(self, market_data: MarketData, **kwargs) -> List[Signal]
    
    def filter_signals(self, signals: List[Signal], min_strength: float = 0.0,
                      min_confidence: float = 0.0) -> List[Signal]
    def combine_signals(self, signal_lists: List[List[Signal]], 
                       method: str = 'union') -> List[Signal]
```

### Trading Strategies

#### `src.strategies.base`

**BaseStrategy**
```python
class BaseStrategy(ABC):
    def __init__(self, name: str)
    
    @abstractmethod
    def generate_signals(self, market_data: Dict[str, MarketData]) -> List[Signal]
    
    def set_parameters(self, **kwargs) -> None
    def get_parameters(self) -> Dict[str, Any]
    def add_signal_generator(self, generator: SignalGenerator) -> None
```

#### `src.strategies.mean_reversion`

**ZScoreMeanReversion**
```python
class ZScoreMeanReversion(BaseStrategy):
    def __init__(self, lookback_period: int = 20, entry_threshold: float = 2.0,
                exit_threshold: float = 0.5, min_volume: float = 1000000,
                volatility_filter: bool = True)
    
    def generate_signals(self, market_data: Dict[str, MarketData]) -> List[Signal]
    def get_strategy_description(self) -> str
    def reset_positions(self) -> None
```

### Backtesting Engine

#### `src.backtesting.portfolio`

**Position**
```python
@dataclass
class Position:
    symbol: str
    quantity: float
    entry_price: float
    entry_time: datetime
    direction: int  # 1 for long, -1 for short
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    
    def unrealized_pnl(self, current_price: float) -> float
    def unrealized_pnl_pct(self, current_price: float) -> float
    def should_stop_loss(self, current_price: float) -> bool
    def should_take_profit(self, current_price: float) -> bool
```

**Portfolio**
```python
class Portfolio:
    def __init__(self, initial_capital: float, commission: float = 0.001, 
                slippage: float = 0.0005)
    
    def open_position(self, symbol: str, quantity: float, price: float,
                     timestamp: datetime, direction: int, 
                     stop_loss: Optional[float] = None,
                     take_profit: Optional[float] = None) -> bool
    
    def close_position(self, symbol: str, price: float, timestamp: datetime,
                      reason: str = 'signal') -> bool
    
    def update_equity_curve(self, timestamp: datetime, 
                           market_prices: Dict[str, float]) -> None
    
    @property
    def total_value(self) -> float
    @property
    def positions_value(self) -> float
    @property
    def num_positions(self) -> int
```

#### `src.backtesting.engine`

**BacktestEngine**
```python
class BacktestEngine:
    def __init__(self, initial_capital: float = 100000, commission: float = 0.001,
                slippage: float = 0.0005, risk_manager: Optional[RiskManager] = None)
    
    def run_backtest(self, strategy: BaseStrategy, symbols: List[str],
                    start_date: Union[str, datetime], end_date: Union[str, datetime],
                    data_provider: DataProvider, benchmark_symbol: Optional[str] = None,
                    rebalance_frequency: str = 'daily') -> BacktestResults
```

**BacktestResults**
```python
@dataclass
class BacktestResults:
    strategy_name: str
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    total_trades: int
    win_rate: float
    profit_factor: float
    
    def summary(self) -> Dict[str, Any]
    def print_summary(self) -> None
```

#### `src.backtesting.risk`

**RiskManager**
```python
class RiskManager:
    def __init__(self, config: Optional[Dict[str, Any]] = None)
    
    def should_exit_position(self, position: Position, current_price: float,
                           portfolio: Portfolio) -> bool
    
    def adjust_position_size(self, proposed_size: float, symbol: str,
                           price: float, portfolio: Portfolio) -> float
    
    def calculate_var(self, portfolio: Portfolio, confidence_level: float = 0.05,
                     time_horizon: int = 1) -> float
    
    def get_risk_metrics(self, portfolio: Portfolio) -> Dict[str, Any]
```

### Visualization

#### `src.visualization.performance`

**PerformanceVisualizer**
```python
class PerformanceVisualizer:
    def __init__(self, results: BacktestResults)
    
    def plot_equity_curve(self, figsize: tuple = (12, 6), 
                         save_path: Optional[str] = None) -> None
    
    def plot_drawdown(self, figsize: tuple = (12, 6), 
                     save_path: Optional[str] = None) -> None
    
    def plot_returns_distribution(self, figsize: tuple = (12, 6), 
                                 save_path: Optional[str] = None) -> None
    
    def plot_monthly_returns(self, figsize: tuple = (14, 8), 
                            save_path: Optional[str] = None) -> None
    
    def plot_performance_summary(self, figsize: tuple = (16, 12), 
                                save_path: Optional[str] = None) -> None
    
    def create_interactive_dashboard(self) -> go.Figure
```

### Utilities

#### `src.utils.config`

**Config**
```python
class Config:
    def __init__(self, config_path: Optional[str] = None)
    def get(self, key: str, default: Any = None) -> Any
    def set(self, key: str, value: Any) -> None
    def reload(self) -> None
    
    @property
    def data_config(self) -> Dict[str, Any]
    @property
    def strategy_config(self) -> Dict[str, Any]
    @property
    def risk_config(self) -> Dict[str, Any]
    @property
    def backtesting_config(self) -> Dict[str, Any]
```

#### `src.utils.validation`

**Validation Functions**
```python
def validate_data(data: pd.DataFrame, required_columns: Optional[List[str]] = None,
                 min_rows: int = 1, check_nulls: bool = True,
                 check_duplicates: bool = True) -> bool

def validate_price_data(data: pd.DataFrame) -> bool

def validate_returns(returns: Union[pd.Series, np.ndarray]) -> bool

def validate_config(config: Dict[str, Any]) -> bool

def validate_symbol(symbol: str) -> bool

def validate_date_range(start_date: str, end_date: str) -> bool
```

## Configuration Schema

### Main Configuration (`configs/config.yaml`)

```yaml
data:
  providers:
    yahoo_finance:
      enabled: true
      rate_limit: 2000
    alpha_vantage:
      enabled: false
      api_key: ${ALPHA_VANTAGE_API_KEY}
    binance:
      enabled: false
      api_key: ${BINANCE_API_KEY}
      api_secret: ${BINANCE_API_SECRET}

strategies:
  mean_reversion:
    zscore:
      lookback_period: 20
      entry_threshold: 2.0
      exit_threshold: 0.5

risk_management:
  max_position_size: 0.1
  max_portfolio_exposure: 0.8
  stop_loss: 0.05
  take_profit: 0.15

backtesting:
  initial_capital: 100000
  commission: 0.001
  slippage: 0.0005
  benchmark: "SPY"
  risk_free_rate: 0.02
```

## Error Handling

All modules include comprehensive error handling with informative error messages. Common exceptions:

- `ValueError`: Invalid parameters or data
- `FileNotFoundError`: Missing configuration or data files
- `RuntimeError`: System state errors (e.g., not connected to database)
- `ConnectionError`: Network or API connection issues

## Logging

The system uses structured logging with configurable levels:

```python
from src.utils.logger import get_logger

logger = get_logger(__name__)
logger.info("Information message")
logger.warning("Warning message")
logger.error("Error message")
```

Log levels: DEBUG, INFO, WARNING, ERROR
Default log file: `logs/trading_system.log`
