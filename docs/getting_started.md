# Getting Started with Mean Reversion Trading System

## Overview

The Mean Reversion Trading System is a comprehensive platform for researching, backtesting, and implementing mean reversion trading strategies across both cryptocurrency and traditional stock markets.

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository:**
```bash
git clone <repository-url>
cd mean-reversion
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Install in development mode:**
```bash
pip install -e .
```

4. **Set up environment variables:**
```bash
cp .env.example .env
# Edit .env with your API keys
```

## Quick Start

### Basic Backtest Example

```python
from datetime import datetime
from src.data.providers.yahoo_finance import YahooFinanceProvider
from src.strategies.mean_reversion.zscore_strategy import ZScoreMeanReversion
from src.backtesting.engine.base import BacktestEngine

# 1. Initialize data provider
data_provider = YahooFinanceProvider()

# 2. Create strategy
strategy = ZScoreMeanReversion(
    lookback_period=20,
    entry_threshold=2.0,
    exit_threshold=0.5
)

# 3. Initialize backtesting engine
engine = BacktestEngine(
    initial_capital=100000,
    commission=0.001,
    slippage=0.0005
)

# 4. Run backtest
results = engine.run_backtest(
    strategy=strategy,
    symbols=['AAPL', 'MSFT', 'GOOGL'],
    start_date=datetime(2020, 1, 1),
    end_date=datetime(2023, 12, 31),
    data_provider=data_provider
)

# 5. View results
results.print_summary()
```

### Running the Example

```bash
cd examples
python basic_backtest.py
```

## Core Components

### 1. Data Infrastructure

- **Data Providers**: Yahoo Finance, Alpha Vantage, Binance
- **Storage**: SQLite-based data storage with caching
- **Models**: Structured data models for prices, symbols, and market data

### 2. Factor Research Framework

- **Technical Indicators**: Moving averages, volatility, momentum
- **Mean Reversion Indicators**: Z-score, Bollinger Bands, Hurst exponent
- **Signal Generation**: Configurable signal generators with confidence scoring

### 3. Backtesting Engine

- **Portfolio Management**: Position tracking, risk management
- **Performance Metrics**: Returns, Sharpe ratio, drawdown analysis
- **Transaction Costs**: Commission and slippage modeling

### 4. Trading Strategies

- **Z-Score Mean Reversion**: Statistical mean reversion based on Z-scores
- **Bollinger Bands**: Mean reversion using Bollinger Band positions
- **RSI Contrarian**: RSI-based contrarian signals

## Configuration

The system uses YAML configuration files in the `configs/` directory:

```yaml
# configs/config.yaml
strategies:
  mean_reversion:
    zscore:
      lookback_period: 20
      entry_threshold: 2.0
      exit_threshold: 0.5

risk_management:
  max_position_size: 0.1
  stop_loss: 0.05
  take_profit: 0.15

backtesting:
  initial_capital: 100000
  commission: 0.001
  slippage: 0.0005
```

## Data Providers

### Yahoo Finance (Default)

```python
from src.data.providers.yahoo_finance import YahooFinanceProvider

provider = YahooFinanceProvider()
market_data = provider.get_price_data(
    symbol='AAPL',
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2023, 12, 31)
)
```

### Alpha Vantage

```python
from src.data.providers.alpha_vantage import AlphaVantageProvider

provider = AlphaVantageProvider(api_key='your_api_key')
# Usage similar to Yahoo Finance
```

### Binance (Cryptocurrency)

```python
from src.data.providers.binance import BinanceProvider

provider = BinanceProvider(
    api_key='your_api_key',
    api_secret='your_api_secret'
)
# Usage similar to other providers
```

## Strategy Development

### Creating a Custom Strategy

```python
from src.strategies.base import BaseStrategy
from src.factors.signals.base import Signal

class MyMeanReversionStrategy(BaseStrategy):
    def __init__(self):
        super().__init__("My Mean Reversion Strategy")
        
    def generate_signals(self, market_data):
        signals = []
        
        for symbol, data in market_data.items():
            # Your signal generation logic here
            signal = Signal(
                timestamp=data.data.index[-1],
                symbol=symbol,
                signal_type='entry',
                direction=1,  # 1 for long, -1 for short
                strength=0.8,
                price=data.data['close'].iloc[-1],
                confidence=0.7
            )
            signals.append(signal)
            
        return signals
```

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_basic_functionality.py

# Run with coverage
pytest --cov=src tests/
```

## Common Issues

### 1. Data Provider Errors

**Problem**: API rate limits or connection errors
**Solution**: 
- Check your API keys in `.env`
- Reduce the number of symbols or date range
- Add delays between requests

### 2. Insufficient Data

**Problem**: Strategy requires more historical data
**Solution**:
- Increase the lookback period in data loading
- Use longer historical periods
- Check data availability for your symbols

### 3. Memory Issues

**Problem**: Large datasets causing memory issues
**Solution**:
- Process symbols in batches
- Use data filtering to reduce memory usage
- Consider using a more powerful machine

## Next Steps

1. **Explore Examples**: Check the `examples/` directory for more use cases
2. **Read Documentation**: Browse the `docs/` directory for detailed guides
3. **Customize Strategies**: Modify existing strategies or create new ones
4. **Run Backtests**: Test your strategies on historical data
5. **Analyze Results**: Use the built-in performance metrics and visualization tools

## Support

- **Documentation**: See the `docs/` directory
- **Examples**: Check the `examples/` directory
- **Tests**: Run `pytest` to verify your installation
- **Issues**: Check the logs in the `logs/` directory for debugging

## License

This project is licensed under the MIT License. See the LICENSE file for details.
