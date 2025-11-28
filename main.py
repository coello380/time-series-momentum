#!/usr/bin/env python3
"""
Momentum Strategy - Main Entry Point

This script runs a complete momentum strategy backtest:
1. Fetches historical price data
2. Generates momentum signals
3. Runs backtest with transaction costs
4. Outputs performance metrics and visualizations

Usage:
    python main.py                          # Run with defaults (sector ETFs, 5 years)
    python main.py --tickers AAPL MSFT GOOGL --years 3
    python main.py --universe sp500 --years 10

Author: Eduardo Medina
"""

import argparse
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from data import DataFetcher, load_prices, DEFAULT_UNIVERSE, LIQUID_STOCKS, get_sp500_tickers
from strategy import MomentumStrategy, Backtester, run_momentum_backtest
from visualization import create_full_report, plot_equity_curve


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run momentum strategy backtest',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py                                    # Default: sector ETFs, 5 years
    python main.py --tickers AAPL MSFT GOOGL AMZN    # Specific tickers
    python main.py --universe liquid                  # Use liquid stock universe
    python main.py --years 10 --lookback 6           # 10 years, 6-month lookback
        """
    )
    
    parser.add_argument(
        '--tickers', 
        nargs='+', 
        default=None,
        help='List of ticker symbols (e.g., AAPL MSFT GOOGL)'
    )
    
    parser.add_argument(
        '--universe',
        choices=['etfs', 'liquid', 'sp500'],
        default='etfs',
        help='Predefined universe: etfs (sector ETFs), liquid (large caps), sp500'
    )
    
    parser.add_argument(
        '--years',
        type=int,
        default=5,
        help='Number of years of historical data (default: 5)'
    )
    
    parser.add_argument(
        '--lookback',
        type=int,
        default=4,
        help='Momentum lookback period in months (default: 4)'
    )
    
    parser.add_argument(
        '--vol-target',
        type=float,
        default=0.10,
        help='Target volatility for position sizing (default: 0.10)'
    )
    
    parser.add_argument(
        '--cost',
        type=float,
        default=0.001,
        help='Transaction cost as decimal (default: 0.001 = 10bps)'
    )
    
    parser.add_argument(
        '--source',
        choices=['yfinance', 'alpaca'],
        default='yfinance',
        help='Data source (default: yfinance)'
    )
    
    parser.add_argument(
        '--output',
        default='results',
        help='Output directory for results (default: results)'
    )
    
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip generating plots'
    )
    
    return parser.parse_args()


def get_universe(args) -> list:
    """Get the ticker universe based on arguments."""
    if args.tickers:
        return args.tickers
    
    if args.universe == 'etfs':
        return DEFAULT_UNIVERSE
    elif args.universe == 'liquid':
        return LIQUID_STOCKS
    elif args.universe == 'sp500':
        print("Fetching S&P 500 constituents...")
        return get_sp500_tickers()
    
    return DEFAULT_UNIVERSE


def main():
    """Main entry point."""
    args = parse_args()
    
    print("=" * 60)
    print("MOMENTUM STRATEGY BACKTEST")
    print("=" * 60)
    
    # Get tickers
    tickers = get_universe(args)
    print(f"\nUniverse: {len(tickers)} tickers")
    if len(tickers) <= 15:
        print(f"Tickers: {', '.join(tickers)}")
    else:
        print(f"Tickers: {', '.join(tickers[:10])}... and {len(tickers)-10} more")
    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.years * 365)
    
    print(f"\nDate range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"Data source: {args.source}")
    
    # Fetch data
    print("\nFetching price data...")
    try:
        fetcher = DataFetcher(source=args.source)
        prices = fetcher.get_prices(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date
        )
    except Exception as e:
        print(f"Error fetching data: {e}")
        sys.exit(1)
    
    # Drop tickers with too much missing data
    missing_pct = prices.isnull().sum() / len(prices)
    valid_tickers = missing_pct[missing_pct < 0.1].index.tolist()
    
    if len(valid_tickers) < len(tickers):
        dropped = set(tickers) - set(valid_tickers)
        print(f"Dropped {len(dropped)} tickers with >10% missing data: {dropped}")
        prices = prices[valid_tickers]
    
    # Forward fill small gaps, then drop remaining NaN
    prices = prices.ffill().dropna()
    
    if len(prices) == 0 or len(prices.columns) == 0:
        print("Error: No valid price data loaded. Try again later or use --source alpaca")
        sys.exit

    print(f"Loaded {len(prices)} trading days for {len(prices.columns)} tickers")
    
    # Run backtest
    print("\nRunning backtest...")
    print(f"  Lookback: {args.lookback} months")
    print(f"  Vol target: {args.vol_target*100:.0f}%")
    print(f"  Transaction cost: {args.cost*10000:.0f} bps")
    
    results = run_momentum_backtest(
        prices=prices,
        n_lags=args.lookback,
        target_vol=args.vol_target,
        transaction_cost=args.cost
    )
    
    # Calculate benchmark (equal-weight buy & hold)
    returns = np.log(prices / prices.shift(1)).dropna()
    #gemini fix:::
    # --- ADD THIS RESAMPLING STEP ---
    # Calculate cumulative daily returns, then resample to monthly
    # The benchmark should reflect the returns over the same period as the strategy
    monthly_benchmark_returns = (1 + returns.mean(axis=1)).resample('ME').prod() - 1 
    benchmark_returns = monthly_benchmark_returns.dropna() 
    # ---------------------------------
    
    # Align benchmark with strategy returns (now the intersection should work)
    common_idx = results['portfolio_returns'].index.intersection(benchmark_returns.index)
    benchmark_returns = benchmark_returns.loc[common_idx]
    # Check for empty Series before division (gemini fix?)
    if len(benchmark_returns) == 0:
        print("Error: After alignment, benchmark returns are empty. Cannot calculate metrics.")
        # Set safe defaults or exit gracefully
        bench_cumulative = 0
        bench_ann_return = 0
        bench_vol = 0
        bench_sharpe = 0
    else:
        # Benchmark metrics calculation (lines 202-207)
        bench_cumulative = (1 + benchmark_returns).prod() - 1
        # Use 12 for periods per year since the data is now monthly
        bench_ann_return = (1 + bench_cumulative) ** (12 / len(benchmark_returns)) - 1 
        bench_vol = benchmark_returns.std() * np.sqrt(12)
        bench_sharpe = bench_ann_return / bench_vol if bench_vol > 0 else 0
        
    
    # Add benchmark metrics (commented it out as per gemini request)
    #bench_cumulative = (1 + benchmark_returns).prod() - 1
    #bench_ann_return = (1 + bench_cumulative) ** (252 / len(benchmark_returns)) - 1
    #bench_vol = benchmark_returns.std() * np.sqrt(252)
    #bench_sharpe = bench_ann_return / bench_vol if bench_vol > 0 else 0
    
    # Print results
    metrics = results['metrics']
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\n{'Metric':<25} {'Strategy':>15} {'Benchmark':>15}")
    print("-" * 55)
    print(f"{'Total Return':<25} {metrics['total_return']*100:>14.2f}% {bench_cumulative*100:>14.2f}%")
    print(f"{'Annualized Return':<25} {metrics['annualized_return']*100:>14.2f}% {bench_ann_return*100:>14.2f}%")
    print(f"{'Annualized Volatility':<25} {metrics['annualized_volatility']*100:>14.2f}% {bench_vol*100:>14.2f}%")
    print(f"{'Sharpe Ratio':<25} {metrics['sharpe_ratio']:>15.2f} {bench_sharpe:>15.2f}")
    print(f"{'Max Drawdown':<25} {metrics['max_drawdown']*100:>14.2f}%")
    print(f"{'Calmar Ratio':<25} {metrics['calmar_ratio']:>15.2f}")
    print(f"{'Win Rate':<25} {metrics['win_rate']*100:>14.1f}%")
    print("=" * 60)
    
    # Generate visualizations
    if not args.no_plots:
        print(f"\nGenerating visualizations in '{args.output}/'...")
        os.makedirs(args.output, exist_ok=True)
        
        try:
            create_full_report(
                results=results,
                benchmark_returns=benchmark_returns,
                output_dir=args.output
            )
        except Exception as e:
            print(f"Warning: Could not generate some plots: {e}")
    
    # Save results to CSV
    results_df = pd.DataFrame({
        'strategy_return': results['portfolio_returns'],
        'benchmark_return': benchmark_returns
    })
    results_df.to_csv(os.path.join(args.output, 'returns.csv'))
    print(f"Returns saved to: {args.output}/returns.csv")
    
    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(args.output, 'metrics.csv'), index=False)
    print(f"Metrics saved to: {args.output}/metrics.csv")
    
    print("\nDone!")
    
    return results


if __name__ == '__main__':
    main()
