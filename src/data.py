"""
Data fetching module for momentum strategy.
Supports multiple data sources: yfinance (default) and Alpaca.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional, Union
from dotenv import load_dotenv
from alpaca.data.enums import DataFeed
load_dotenv()
import os

# Primary: yfinance (no API key required)
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

# Secondary: Alpaca (requires API key)
try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False


# ============================================================================
# UNIVERSE DEFINITIONS
# ============================================================================

# S&P 500 sector ETFs - good starting universe for momentum
SECTOR_ETFS = [
    'XLK',  # Technology
    'XLF',  # Financials
    'XLV',  # Healthcare
    'XLE',  # Energy
    'XLI',  # Industrials
    'XLP',  # Consumer Staples
    'XLY',  # Consumer Discretionary
    'XLB',  # Materials
    'XLU',  # Utilities
    'XLRE', # Real Estate
    'XLC',  # Communication Services
]

# Larger universe: liquid US stocks (example)
LIQUID_STOCKS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B',
    'JPM', 'JNJ', 'V', 'PG', 'UNH', 'HD', 'MA', 'DIS', 'PYPL', 'ADBE',
    'NFLX', 'CRM', 'PFE', 'KO', 'PEP', 'TMO', 'ABBV', 'COST', 'AVGO',
    'WMT', 'MRK', 'CVX', 'XOM', 'LLY', 'BAC', 'ORCL', 'AMD', 'QCOM',
]

# Default universe for backtesting
DEFAULT_UNIVERSE = SECTOR_ETFS


class DataFetcher:
    """
    Unified interface for fetching historical price data.
    Supports yfinance (free) and Alpaca (requires API key).
    """
    
    def __init__(self, source: str = 'yfinance', alpaca_key: str = None, alpaca_secret: str = None):
        """
        Initialize the data fetcher.
        
        Args:
            source: 'yfinance' or 'alpaca'
            alpaca_key: Alpaca API key (optional, can use env var ALPACA_API_KEY)
            alpaca_secret: Alpaca secret key (optional, can use env var ALPACA_SECRET_KEY)
        """
        self.source = source.lower()
        
        if self.source == 'yfinance':
            if not YFINANCE_AVAILABLE:
                raise ImportError("yfinance not installed. Run: pip install yfinance")
        
        elif self.source == 'alpaca':
            if not ALPACA_AVAILABLE:
                raise ImportError("alpaca-py not installed. Run: pip install alpaca-py")
            
            # Get API keys from arguments or environment
            self.alpaca_key = alpaca_key or os.getenv('ALPACA_API_KEY')
            self.alpaca_secret = alpaca_secret or os.getenv('ALPACA_SECRET_KEY')
            
            if not self.alpaca_key or not self.alpaca_secret:
                raise ValueError(
                    "Alpaca API credentials required. Either pass them as arguments "
                    "or set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables."
                )
            
            self.alpaca_client = StockHistoricalDataClient(self.alpaca_key, self.alpaca_secret)
        
        else:
            raise ValueError(f"Unknown data source: {source}. Use 'yfinance' or 'alpaca'.")
    
    def get_prices(
        self,
        tickers: List[str],
        start_date: Union[str, datetime],
        end_date: Union[str, datetime] = None,
        price_col: str = 'Adj Close'
    ) -> pd.DataFrame:
        """
        Fetch historical prices for a list of tickers.
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date (str 'YYYY-MM-DD' or datetime)
            end_date: End date (default: today)
            price_col: Which price to use ('Adj Close', 'Close', 'Open', etc.)
        
        Returns:
            DataFrame with dates as index and tickers as columns
        """
        if end_date is None:
            end_date = datetime.now()
        
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d')
        
        if self.source == 'yfinance':
            return self._fetch_yfinance(tickers, start_date, end_date, price_col)
        elif self.source == 'alpaca':
            return self._fetch_alpaca(tickers, start_date, end_date)
    
    def _fetch_yfinance(
        self,
        tickers: List[str],
        start_date: datetime,
        end_date: datetime,
        price_col: str
    ) -> pd.DataFrame:
        """Fetch data using yfinance."""
        data = yf.download(
            tickers,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=False
        )
        
        if len(tickers) == 1:
            # Single ticker returns different structure
            prices = data[[price_col]].copy()
            prices.columns = tickers
        else:
            prices = data[price_col].copy()
        
        return prices
    
    def _fetch_alpaca(
        self,
        tickers: List[str],
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Fetch data using Alpaca API."""
        request_params = StockBarsRequest(
            symbol_or_symbols=tickers,
            timeframe=TimeFrame.Day,
            start=start_date,
            end=end_date,
            feed=DataFeed.IEX
        )
        
        bars = self.alpaca_client.get_stock_bars(request_params)
        
        # Convert to DataFrame
        data_dict = {}
        for ticker in tickers:
            if ticker in bars.data:
                ticker_bars = bars.data[ticker]
                data_dict[ticker] = pd.Series(
                    [bar.close for bar in ticker_bars],
                    index=[bar.timestamp for bar in ticker_bars]
                )
        
        prices = pd.DataFrame(data_dict)
        prices.index = pd.to_datetime(prices.index).tz_localize(None)
        
        return prices


def get_sp500_tickers() -> List[str]:
    """
    Fetch current S&P 500 constituents from Wikipedia.
    
    Returns:
        List of ticker symbols
    """
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    tables = pd.read_html(url)
    sp500_table = tables[0]
    tickers = sp500_table['Symbol'].tolist()
    # Clean up tickers (replace . with - for yfinance compatibility)
    tickers = [t.replace('.', '-') for t in tickers]
    return tickers


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def load_prices(
    tickers: List[str] = None,
    start_date: str = None,
    end_date: str = None,
    source: str = 'yfinance'
) -> pd.DataFrame:
    """
    Quick helper to load price data with sensible defaults.
    
    Args:
        tickers: List of tickers (default: sector ETFs)
        start_date: Start date (default: 5 years ago)
        end_date: End date (default: today)
        source: Data source ('yfinance' or 'alpaca')
    
    Returns:
        DataFrame of adjusted close prices
    """
    if tickers is None:
        tickers = DEFAULT_UNIVERSE
    
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
    
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    fetcher = DataFetcher(source=source)
    return fetcher.get_prices(tickers, start_date, end_date)


if __name__ == '__main__':
    # Example usage
    print("Fetching sector ETF data...")
    prices = load_prices()
    print(f"\nLoaded {len(prices)} days of data for {len(prices.columns)} tickers")
    print(f"Date range: {prices.index[0]} to {prices.index[-1]}")
    print(f"\nTickers: {list(prices.columns)}")
    print(f"\nSample data:\n{prices.tail()}")
