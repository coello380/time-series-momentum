# Time-Series Momentum Strategy

A Python implementation of the time-series momentum strategy from **"Trend Following: Equity and Bond Crisis Alpha"** by Hamill, Rattray, and Van Hemert (Man AHL, 2016).

This project replicates the **mom(1,4)** strategy, a volatility-scaled momentum signal using the past 4 months of returns—and adapts it for trading equity ETFs.

---

## Results

Backtest on S&P 500 sector ETFs (2018-2025):

| Metric | Strategy | Benchmark |
|--------|----------|-----------|
| Total Return | 32.0% | 47.1% |
| Annualized Return | 5.8% | 8.2% |
| Annualized Volatility | 9.0% | 14.6% |
| Sharpe Ratio | **0.64** | 0.56 |
| Max Drawdown | -12.8% | — |
| Calmar Ratio | 0.46 | — |

The strategy achieves a **higher risk-adjusted return** (Sharpe 0.64 vs 0.56) with **significantly lower volatility** (9% vs 15%), demonstrating effective volatility targeting.


---

## Methodology

### Signal Construction

The momentum signal is calculated as:

$$mom_{t} = \frac{\sum_{i=1}^{4} w_i \cdot R_{t-i}}{\sigma_{t} \cdot \sqrt{\sum w_i^2}}$$

Where:
- $R_{t-i}$ = Monthly return at lag $i$
- $w_i$ = Weight for each lag (equal weights: 0.25 each)
- $\sigma_t$ = Volatility estimate at time $t$
- $\sqrt{\sum w_i^2}$ = 0.5 for equal weights

### Volatility Estimation

Volatility uses exponentially weighted moving standard deviation with a floor:

$$\sigma_t = \max(\sigma_{6m}, 0.5 \times \sigma_{24m})$$

Where:
- $\sigma_{6m}$ = EWM std with 6-month half-life
- $\sigma_{24m}$ = EWM std with 24-month half-life

The floor prevents over-leveraging during low-volatility regimes.

### Position Sizing

Positions are scaled by dividing the signal by volatility again:

$$position_t = \frac{mom_t}{\sigma_t}$$

This ensures equal risk contribution across assets regardless of their volatility.

### Portfolio Gearing

The portfolio is scaled to achieve 10% target volatility:

$$gearing = \frac{\sigma_{target}}{\sqrt{w' \Omega w}}$$

Where $\Omega$ is the correlation matrix estimated with a rolling 24-month window.

### Signal Capping

Signals are capped at ±2 to prevent extreme positions.

---

## Project Structure

```
├── main.py                 # CLI entry point
├── src/
│   ├── strategy.py         # MomentumStrategy and Backtester classes
│   ├── data.py             # Data fetching (yfinance, Alpaca)
│   └── visualization.py    # Performance charts
├── results/                # Backtest outputs
└── requirements.txt
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/trend-following.git
cd trend-following

# Install dependencies
pip install -r requirements.txt
```

### Requirements

```
pandas
numpy
matplotlib
yfinance
alpaca-py        # Optional: for Alpaca data
python-dotenv    # Optional: for .env file support
```

---

## Usage

### Basic Usage

```bash
# Run with default settings (sector ETFs, 5 years, yfinance)
python main.py

# Specify parameters
python main.py --years 7 --lookback 4 --vol-target 0.10
```

### Command Line Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--tickers` | — | Custom ticker list (e.g., `AAPL MSFT GOOGL`) |
| `--universe` | `etfs` | Predefined universe: `etfs`, `liquid`, `sp500` |
| `--years` | 5 | Years of historical data |
| `--lookback` | 4 | Momentum lookback in months |
| `--vol-target` | 0.10 | Target portfolio volatility |
| `--cost` | 0.001 | Transaction cost (decimal) |
| `--source` | `yfinance` | Data source: `yfinance` or `alpaca` |
| `--output` | `results` | Output directory |

### Using Alpaca Data

```bash
# Set environment variables (or use .env file)
export ALPACA_API_KEY="your_key"
export ALPACA_SECRET_KEY="your_secret"

python main.py --source alpaca --years 7
```

### Python API

```python
from src.data import load_prices
from src.strategy import run_momentum_backtest

# Load data
prices = load_prices(
    tickers=['XLK', 'XLF', 'XLV', 'XLE'],
    start_date='2019-01-01'
)

# Run backtest
results = run_momentum_backtest(
    prices=prices,
    n_lags=4,
    target_vol=0.10,
    transaction_cost=0.0002
)

# Access results
print(f"Sharpe Ratio: {results['metrics']['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {results['metrics']['max_drawdown']*100:.1f}%")
```

---

## Key Findings

### Why Momentum Works

1. **Behavioral biases**: Investors underreact to new information, causing trends to persist
2. **Institutional flows**: Large funds rebalance slowly, creating predictable price patterns
3. **Risk premia**: Trend following provides "crisis alpha"—positive returns during market crashes

### Performance Characteristics

- **Positive skewness**: The strategy profits from large moves in either direction
- **Crisis alpha**: Outperformed during the 2022 bear market (visible in equity curve)
- **Low correlation**: Returns have low correlation with buy-and-hold equity

### Limitations

- **Whipsaw risk**: Choppy, mean-reverting markets hurt performance (see 2023-2024)
- **Capacity constraints**: Strategy works best with liquid, diversified instruments
- **Look-ahead bias**: Careful to use only information available at decision time

---

## References

- Hamill, C., Rattray, S., & Van Hemert, O. (2016). **Trend Following: Equity and Bond Crisis Alpha**. Man AHL Working Paper.
- Moskowitz, T. J., Ooi, Y. H., & Pedersen, L. H. (2012). **Time Series Momentum**. Journal of Financial Economics, 104(2), 228-250.
- Hurst, B., Ooi, Y. H., & Pedersen, L. H. (2017). **A Century of Evidence on Trend-Following Investing**. AQR Capital Management.

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Author

**Eduardo Medina (coello)**  
MSc Quantitative Finance, University of Amsterdam  
[LinkedIn](https://linkedin.com/in/yourprofile) | [GitHub](https://github.com/yourusername)
