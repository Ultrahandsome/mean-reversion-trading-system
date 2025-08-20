# Mean Reversion Trading System

A comprehensive cryptocurrency and stock factor research and backtesting platform focused on mean reversion trading strategies.

## Overview

This project implements a sophisticated mean reversion trading system that:
- Identifies overbought/oversold conditions when asset prices deviate from historical means
- Executes contrarian trades (short when overbought, long when oversold)
- Provides comprehensive backtesting and performance analysis
- Supports both cryptocurrency and traditional stock market data

## Project Structure

```
mean_reversion/
├── src/
│   ├── data/                    # Data infrastructure
│   │   ├── providers/           # Data source integrations
│   │   ├── storage/             # Data storage and management
│   │   └── preprocessing/       # Data cleaning and preparation
│   ├── factors/                 # Factor research framework
│   │   ├── indicators/          # Technical indicators
│   │   ├── signals/             # Signal generation
│   │   └── analysis/            # Statistical analysis tools
│   ├── backtesting/             # Backtesting engine
│   │   ├── engine/              # Core backtesting logic
│   │   ├── metrics/             # Performance metrics
│   │   └── risk/                # Risk management
│   ├── strategies/              # Trading strategies
│   │   ├── mean_reversion/      # Mean reversion strategies
│   │   └── base/                # Base strategy classes
│   ├── portfolio/               # Portfolio management
│   └── utils/                   # Utility functions
├── tests/                       # Unit tests
├── configs/                     # Configuration files
├── data/                        # Data storage
├── notebooks/                   # Jupyter notebooks for research
├── docs/                        # Documentation
└── examples/                    # Usage examples
```

## Core Features

### Data Infrastructure
- Multi-source data ingestion (Yahoo Finance, Alpha Vantage, Binance, etc.)
- Historical and real-time data support
- Efficient data storage and retrieval
- Data quality validation and cleaning

### Factor Research Framework
- Multiple mean calculation methods (SMA, EMA, LWMA)
- Volatility and deviation measurements
- Statistical significance testing
- Signal strength and confidence scoring

### Backtesting Engine
- Historical strategy simulation
- Comprehensive performance metrics
- Risk management and position sizing
- Transaction cost modeling
- Portfolio-level analysis

### Mean Reversion Strategies
- Z-score based mean reversion
- Bollinger Bands mean reversion
- RSI-based contrarian strategies
- Multi-timeframe analysis

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd mean-reversion

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

## Quick Start

```python
from src.strategies.mean_reversion import ZScoreMeanReversion
from src.backtesting.engine import BacktestEngine
from src.data.providers import YahooFinanceProvider

# Initialize data provider
data_provider = YahooFinanceProvider()

# Create strategy
strategy = ZScoreMeanReversion(
    lookback_period=20,
    entry_threshold=2.0,
    exit_threshold=0.5
)

# Run backtest
engine = BacktestEngine(
    strategy=strategy,
    data_provider=data_provider,
    initial_capital=100000
)

results = engine.run_backtest(
    symbols=['AAPL', 'MSFT', 'BTC-USD'],
    start_date='2020-01-01',
    end_date='2023-12-31'
)

# Analyze results
print(f"Total Return: {results.total_return:.2%}")
print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
print(f"Max Drawdown: {results.max_drawdown:.2%}")
```

## Configuration

The system uses YAML configuration files for easy parameter management:

```yaml
# configs/strategy_config.yaml
mean_reversion:
  lookback_period: 20
  entry_threshold: 2.0
  exit_threshold: 0.5
  position_size: 0.1
  max_positions: 10

risk_management:
  max_position_size: 0.2
  stop_loss: 0.05
  max_drawdown: 0.15
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

MIT License - see LICENSE file for details.


# Mean Reversion Trading System 用户指南

## 1. 项目介绍

### 系统概述
Mean Reversion Trading System（均值回归交易系统）是一个专业的量化交易平台，专注于股票和加密货币市场的均值回归策略研究与回测。该系统基于统计学原理，当资产价格显著偏离历史均值时执行反向交易：
- **超买信号**：价格高于均值时做空
- **超卖信号**：价格低于均值时做多

### 核心功能
- 🔄 **多资产支持**：股票、加密货币、外汇等
- 📊 **因子研究框架**：Z-score、布林带、RSI等技术指标
- 🎯 **策略回测引擎**：完整的历史模拟与性能分析
- 📈 **可视化工具**：交互式图表和性能报告
- ⚡ **实时数据支持**：支持实时和历史数据获取
- 🛡️ **风险管理**：仓位管理、止损止盈、回撤控制

### 目标用户
- **量化交易员**：开发和测试均值回归策略
- **金融研究员**：分析市场均值回归特性
- **Python开发者**：构建定制化交易系统

## 2. 系统架构

### 模块化设计
系统采用分层模块化架构，便于扩展和维护：

```
数据层 → 因子层 → 策略层 → 回测层 → 可视化层
```

### 核心组件

**数据基础设施 (Data Infrastructure)**
- 多数据源支持（Yahoo Finance、Alpha Vantage、Binance）
- SQLite数据存储与缓存
- 数据预处理和验证

**因子研究框架 (Factor Research)**
- 技术指标库（移动平均、波动率、动量）
- 均值回归指标（Z-score、布林带位置、Hurst指数）
- 信号生成与置信度评分

**回测引擎 (Backtesting Engine)**
- 投资组合管理
- 性能指标计算
- 风险管理系统
- 交易成本建模

**交易策略 (Trading Strategies)**
- Z-score均值回归策略
- 布林带均值回归策略
- RSI反向策略

## 3. 安装与配置

### 系统要求
- Python 3.8+
- 8GB+ RAM（推荐）
- 网络连接（获取市场数据）

### 安装步骤

1. **克隆项目**
```bash
git clone <repository-url>
cd mean-reversion
```

2. **安装依赖**
```bash
pip install -r requirements.txt
pip install -e .
```

3. **配置环境变量**
```bash
cp .env.example .env
# 编辑 .env 文件，添加API密钥
```

4. **验证安装**
```bash
python -c "import src; print('安装成功！')"
```

### 配置设置

**API密钥配置** (`.env`文件)：
```bash
# Alpha Vantage API密钥
ALPHA_VANTAGE_API_KEY=your_api_key_here

# Binance API密钥（加密货币）
BINANCE_API_KEY=your_binance_key
BINANCE_API_SECRET=your_binance_secret
```

**策略参数配置** (`configs/config.yaml`)：
```yaml
strategies:
  mean_reversion:
    zscore:
      lookback_period: 20      # 回望期
      entry_threshold: 2.0     # 入场阈值
      exit_threshold: 0.5      # 出场阈值

risk_management:
  max_position_size: 0.1       # 最大仓位10%
  stop_loss: 0.05             # 止损5%
  take_profit: 0.15           # 止盈15%
```

## 4. 快速开始

### 基础回测示例

```python
from datetime import datetime
from src.data.providers.yahoo_finance import YahooFinanceProvider
from src.strategies.mean_reversion.zscore_strategy import ZScoreMeanReversion
from src.backtesting.engine.base import BacktestEngine

# 1. 初始化数据提供商
data_provider = YahooFinanceProvider()

# 2. 创建Z-score均值回归策略
strategy = ZScoreMeanReversion(
    lookback_period=20,    # 20日均值
    entry_threshold=2.0,   # 2倍标准差入场
    exit_threshold=0.5     # 0.5倍标准差出场
)

# 3. 初始化回测引擎
engine = BacktestEngine(
    initial_capital=100000,  # 初始资金10万
    commission=0.001,        # 手续费0.1%
    slippage=0.0005         # 滑点0.05%
)

# 4. 运行回测
results = engine.run_backtest(
    strategy=strategy,
    symbols=['AAPL', 'MSFT', 'GOOGL'],  # 测试股票
    start_date=datetime(2020, 1, 1),
    end_date=datetime(2023, 12, 31),
    data_provider=data_provider,
    benchmark_symbol='SPY'  # 基准指数
)

# 5. 查看结果
results.print_summary()
```

### 运行示例
```bash
cd examples
python basic_backtest.py
```

### 预期输出
```
==================================================
BACKTEST RESULTS SUMMARY
==================================================
Strategy            : Z-Score Mean Reversion
Period              : 2020-01-01 to 2023-12-31
Initial Capital     : $100,000.00
Final Capital       : $125,430.50
Total Return        : 25.43%
Annualized Return   : 7.32%
Sharpe Ratio        : 1.24
Max Drawdown        : -8.45%
Total Trades        : 156
Win Rate            : 58.33%
Profit Factor       : 1.67
==================================================
```

## 5. 核心功能详解

### 数据提供商使用

**Yahoo Finance（免费，推荐）**
```python
from src.data.providers.yahoo_finance import YahooFinanceProvider

provider = YahooFinanceProvider()
market_data = provider.get_price_data(
    symbol='AAPL',
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2023, 12, 31)
)
```

**Alpha Vantage（需要API密钥）**
```python
from src.data.providers.alpha_vantage import AlphaVantageProvider

provider = AlphaVantageProvider(api_key='your_key')
# 使用方法与Yahoo Finance相同
```

**Binance（加密货币）**
```python
from src.data.providers.binance import BinanceProvider

provider = BinanceProvider(
    api_key='your_key',
    api_secret='your_secret'
)
```

### 均值回归策略

**Z-Score策略参数**：
- `lookback_period`: 计算均值和标准差的回望期
- `entry_threshold`: 入场Z-score阈值（通常2.0-2.5）
- `exit_threshold`: 出场Z-score阈值（通常0.5-1.0）
- `min_volume`: 最小成交量过滤
- `volatility_filter`: 是否启用波动率过滤

**布林带策略**：
```python
from src.strategies.mean_reversion.bollinger_strategy import BollingerBandsMeanReversion

strategy = BollingerBandsMeanReversion(
    period=20,              # 布林带周期
    num_std=2.0,           # 标准差倍数
    entry_threshold=0.95,   # 入场阈值（接近带边）
    exit_threshold=0.5      # 出场阈值（回到中轨）
)
```

### 回测工作流程

1. **数据准备**：获取历史价格数据
2. **信号生成**：根据策略规则生成交易信号
3. **模拟交易**：执行买卖操作，考虑交易成本
4. **性能计算**：计算收益率、夏普比率等指标
5. **结果分析**：生成报告和可视化图表

## 6. 高级用法

### 创建自定义策略

```python
from src.strategies.base import BaseStrategy
from src.factors.signals.base import Signal

class MyMeanReversionStrategy(BaseStrategy):
    def __init__(self):
        super().__init__("我的均值回归策略")
        
    def generate_signals(self, market_data):
        signals = []
        
        for symbol, data in market_data.items():
            # 自定义信号逻辑
            if self._is_oversold(data):
                signal = Signal(
                    timestamp=data.data.index[-1],
                    symbol=symbol,
                    signal_type='entry',
                    direction=1,  # 做多
                    strength=0.8,
                    price=data.data['close'].iloc[-1],
                    confidence=0.7
                )
                signals.append(signal)
                
        return signals
    
    def _is_oversold(self, data):
        # 实现超卖判断逻辑
        pass
```

### 风险管理配置

```yaml
risk_management:
  max_position_size: 0.1        # 单个仓位最大10%
  max_portfolio_exposure: 0.8   # 总仓位最大80%
  stop_loss: 0.05              # 止损5%
  take_profit: 0.15            # 止盈15%
  max_drawdown: 0.2            # 最大回撤20%
  position_sizing_method: "equal_weight"  # 等权重
```

### 可视化和报告

```python
from src.visualization.performance import PerformanceVisualizer

# 创建可视化工具
visualizer = PerformanceVisualizer(results)

# 生成各种图表
visualizer.plot_equity_curve()           # 净值曲线
visualizer.plot_drawdown()               # 回撤分析
visualizer.plot_returns_distribution()   # 收益分布
visualizer.plot_monthly_returns()        # 月度收益热力图
visualizer.plot_performance_summary()    # 综合性能报告

# 创建交互式仪表板
interactive_fig = visualizer.create_interactive_dashboard()
interactive_fig.show()
```

## 7. 故障排除

### 常见问题

**问题1：数据获取失败**
```
错误：ConnectionError: Failed to fetch data
解决：检查网络连接和API密钥配置
```

**问题2：内存不足**
```
错误：MemoryError: Unable to allocate array
解决：减少回测时间范围或股票数量，或增加系统内存
```

**问题3：策略无信号**
```
错误：Generated 0 signals
解决：调整策略参数，降低入场阈值或检查数据质量
```

### 调试技巧

**启用详细日志**：
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

**检查数据质量**：
```python
# 验证价格数据
from src.utils.validation import validate_price_data
validate_price_data(market_data.data)
```

**测试策略参数**：
```python
# 使用较短时间段测试
results = engine.run_backtest(
    strategy=strategy,
    symbols=['AAPL'],  # 单个股票
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2023, 6, 30),  # 6个月测试
    data_provider=data_provider
)
```

### 性能优化建议

1. **批量处理**：一次处理多个股票而非逐个处理
2. **数据缓存**：使用SQLite存储避免重复下载
3. **参数调优**：使用网格搜索优化策略参数
4. **并行计算**：对于大规模回测考虑使用多进程

### 获取帮助

- **文档**：查看 `docs/` 目录获取详细文档
- **示例**：参考 `examples/` 目录的使用示例
- **测试**：运行 `pytest` 验证系统功能
- **日志**：检查 `logs/` 目录的错误日志

---

通过本指南，您应该能够快速上手Mean Reversion Trading System，开始您的量化交易研究之旅。系统的模块化设计使其易于扩展和定制，适合各种均值回归策略的研究和实现。
