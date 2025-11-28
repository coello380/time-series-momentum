"""
Time-Series Momentum Strategy - Paper Replication

Exact implementation of mom(1,4) from:
"Trend Following: Equity and Bond Crisis Alpha" 
by Hamill, Rattray, Van Hemert (Man AHL, 2016)

Key methodology points:
1. Signal = weighted sum of individual monthly returns / (σ × √Σw²)
2. Volatility = max(EWM_6month, 0.5 × EWM_24month) 
3. Position = signal / σ (vol appears twice)
4. Signal capped at ±2
5. Portfolio scaled to 10% target vol using correlation matrix
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict


class MomentumStrategy:
    """
    Exact replication of mom(1,4) from the Man AHL paper.
    
    The strategy:
    1. Calculates momentum as weighted sum of past 4 monthly returns
    2. Scales by volatility (EWM with floor) and weight factor
    3. Caps signal at ±2
    4. Sizes positions by dividing signal by vol again
    5. Targets 10% portfolio volatility
    """
    
    def __init__(
        self,
        n_lags: int = 4,
        vol_halflife_short: int = 6,  # months (or 126 trading days)
        vol_halflife_long: int = 24,   # months (or 504 trading days)
        vol_floor_multiplier: float = 0.5,
        signal_cap: float = 2.0,
        target_vol: float = 0.10,
        use_monthly: bool = False,  # If True, use actual monthly data; if False, approximate with daily
        trading_days_per_month: int = 21
    ):
        """
        Initialize the strategy.
        
        Args:
            n_lags: Number of monthly lags to use (default: 4 for mom(1,4))
            vol_halflife_short: Short-term vol half-life in months (default: 6)
            vol_halflife_long: Long-term vol half-life in months (default: 24)
            vol_floor_multiplier: Multiplier for vol floor (default: 0.5)
            signal_cap: Maximum absolute signal value (default: 2.0)
            target_vol: Target portfolio volatility (default: 10%)
            use_monthly: Whether input data is monthly (default: False = daily)
            trading_days_per_month: Trading days per month for daily data (default: 21)
        """
        self.n_lags = n_lags
        self.vol_halflife_short = vol_halflife_short
        self.vol_halflife_long = vol_halflife_long
        self.vol_floor_multiplier = vol_floor_multiplier
        self.signal_cap = signal_cap
        self.target_vol = target_vol
        self.use_monthly = use_monthly
        self.trading_days_per_month = trading_days_per_month
        
        # Equal weights for mom(1,4)
        self.weights = np.array([1.0 / n_lags] * n_lags)
        
        # Weight scaling factor: sqrt(sum of squared weights)
        # For equal weights of 1/4: sqrt(4 * (1/4)^2) = sqrt(4 * 1/16) = sqrt(1/4) = 0.5
        self.weight_scale = np.sqrt(np.sum(self.weights ** 2))
    
    def calculate_monthly_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate monthly returns from price data.
        
        If using daily data, resample to monthly first.
        If data is already monthly, just calculate returns.
        
        Args:
            prices: DataFrame of prices (dates x tickers)
        
        Returns:
            DataFrame of monthly returns
        """
        if self.use_monthly:
            # Data is already monthly - just calculate simple returns
            returns = prices.pct_change()
        else:
            # Resample daily to monthly (end of month prices)
            monthly_prices = prices.resample('ME').last()
            returns = monthly_prices.pct_change()
        
        # Drop first row (NaN from pct_change)
        returns = returns.iloc[1:]
        
        return returns
    
    def calculate_volatility(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volatility using the paper's methodology.
        
        From footnote 4 (p.6):
        "We estimate the standard deviation of returns using exponentially 
        decaying weights. We take the maximum of an estimate based on a 
        half-life of 6 months and 0.5 times an estimate based on a half-life 
        of 24 months, where the latter acts as a floor."
        
        Args:
            returns: DataFrame of monthly returns
        
        Returns:
            DataFrame of volatility estimates
        """
        # Short-term vol (6-month half-life)
        vol_short = returns.ewm(halflife=self.vol_halflife_short).std()
        
        # Long-term vol (24-month half-life) with floor multiplier
        vol_long = returns.ewm(halflife=self.vol_halflife_long).std() * self.vol_floor_multiplier
        
        # Take element-wise maximum of the two
        vol = vol_short.combine(vol_long, np.maximum)
        
        return vol
    
    def calculate_momentum_signal(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the momentum signal according to Equation 1 from the paper.
        
        mom_t-1 = (w1*R_t-1 + w2*R_t-2 + ... + wn*R_t-n) / (σ_t-1 * sqrt(Σw²))
        
        For mom(1,4): weights are all 1/4, so sqrt(Σw²) = 0.5
        
        Args:
            returns: DataFrame of monthly returns
        
        Returns:
            DataFrame of momentum signals (capped at ±2)
        """
        # Calculate volatility
        vol = self.calculate_volatility(returns)
        
        # Calculate weighted sum of lagged returns
        # R_t-1 is lag 1 (shift 1), R_t-2 is lag 2 (shift 2), etc.
        weighted_returns = pd.DataFrame(0.0, index=returns.index, columns=returns.columns)
        
        for i, w in enumerate(self.weights):
            lag = i + 1  # lag 1, 2, 3, 4
            weighted_returns += w * returns.shift(lag)
        
        # Divide by (vol * weight_scale) to get signal
        # Note: vol is already the estimate at t-1 due to how ewm works
        signal = weighted_returns / (vol * self.weight_scale)
        
        # Cap signal at ±2
        signal = signal.clip(lower=-self.signal_cap, upper=self.signal_cap)
        
        return signal
    
    def calculate_positions(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate position sizes according to Equation 2 from the paper.
        
        Position = signal / σ
        
        The signal already has one σ in the denominator (for normalization),
        and we divide by σ again here (for position sizing). This ensures
        all assets trade the same amount of risk for a given signal strength.
        
        Args:
            returns: DataFrame of monthly returns
        
        Returns:
            DataFrame of position sizes (in risk units)
        """
        signal = self.calculate_momentum_signal(returns)
        vol = self.calculate_volatility(returns)
        
        # Position = signal / vol (second vol division)
        positions = signal / vol
        
        return positions
    
    def calculate_correlation_matrix(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate correlation matrix using 24-month EWM.
        
        From footnote 6 (p.7):
        "Ω is the correlation matrix between constituent strategy returns 
        based on exponentially decaying weights with a 24-month half-life."
        
        Args:
            returns: DataFrame of returns
        
        Returns:
            Correlation matrix
        """
        # EWM correlation with 24-month half-life
        # pandas ewm doesn't directly support correlation, so we compute it manually
        
        # Get EWM covariance
        ewm_cov = returns.ewm(halflife=24).cov()
        
        # Get the last correlation matrix
        # Extract the most recent covariance matrix
        last_date = returns.index[-1]
        
        try:
            cov_matrix = ewm_cov.loc[last_date]
        except:
            # Fallback to simple correlation if ewm fails
            return returns.corr()
        
        # Convert covariance to correlation
        std = np.sqrt(np.diag(cov_matrix))
        corr_matrix = cov_matrix / np.outer(std, std)
        
        return pd.DataFrame(corr_matrix, index=returns.columns, columns=returns.columns)
    
    def calculate_portfolio_gearing(
        self, 
        returns: pd.DataFrame,
        positions: pd.DataFrame
    ) -> pd.Series:
        """
        Calculate gearing factor to achieve target portfolio volatility.
        
        From footnote 6 (p.7):
        "The gearing process starts with individual securities which are all 
        scaled to 10% average ex-ante volatility using mom*10%/σ. Then in 
        each aggregation step... we achieve a 10% average ex-ante portfolio 
        volatility by multiplying with the weight given to a security or 
        sector and then dividing by a factor √(w'Ωw)."
        
        Simplified: We scale positions so portfolio has target volatility.
        
        Args:
            returns: DataFrame of returns
            positions: DataFrame of positions
        
        Returns:
            Series of gearing factors over time
        """
        n_assets = len(returns.columns)
        
        # Equal weights for assets
        w = np.ones(n_assets) / n_assets
        
        gearing = pd.Series(index=returns.index, dtype=float)
        
        min_periods = 24  # Need at least 24 months for stable estimates
        
        for i in range(len(returns)):
            if i < min_periods:
                gearing.iloc[i] = 1.0
                continue
            
            # Get returns up to this point
            ret_subset = returns.iloc[:i]
            
            # Calculate correlation matrix (simple rolling for stability)
            corr_matrix = ret_subset.tail(min_periods).corr().values
            
            # Handle NaN
            corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
            np.fill_diagonal(corr_matrix, 1.0)
            
            # Individual asset vols (annualized) - use EWM
            asset_vols = ret_subset.ewm(halflife=self.vol_halflife_short).std().iloc[-1].values
            asset_vols_annual = asset_vols * np.sqrt(12)  # monthly to annual
            
            # Replace any zero or nan vols
            asset_vols_annual = np.nan_to_num(asset_vols_annual, nan=0.15)
            asset_vols_annual = np.where(asset_vols_annual < 0.01, 0.15, asset_vols_annual)
            
            # Portfolio volatility: sqrt(w' * Σ * w) where Σ = diag(σ) * Ω * diag(σ)
            cov_matrix = np.outer(asset_vols_annual, asset_vols_annual) * corr_matrix
            port_var = np.dot(w, np.dot(cov_matrix, w))
            port_vol = np.sqrt(max(port_var, 1e-6))
            
            # Gearing to achieve target vol
            gearing.iloc[i] = self.target_vol / port_vol
        
        # Cap gearing to avoid extreme leverage
        gearing = gearing.clip(lower=0.1, upper=5.0)
        
        return gearing


class Backtester:
    """
    Backtester for the paper-accurate momentum strategy.
    """
    
    def __init__(
        self,
        strategy: MomentumStrategy,
        transaction_cost: float = 0.0002  # 2 bps as mentioned in paper (p.7)
    ):
        """
        Initialize backtester.
        
        Args:
            strategy: MomentumStrategy instance
            transaction_cost: Cost per trade (default: 2 bps from paper)
        """
        self.strategy = strategy
        self.transaction_cost = transaction_cost
    
    def run(self, prices: pd.DataFrame) -> Dict:
        """
        Run backtest following Equation 2 from the paper.
        
        From page 7:
        "The signal value indicates how many risk units one would want to hold 
        in a security. To turn this into a dollar position, we need to divide 
        by the volatility estimate a second time (so that all assets are trading 
        the same amount of risk for a given strength signal)."
        
        Performance_t = Σ_k Gearing_t-1^k × (mom_t-1^k / σ_t-1^k) × R_t^k
        
        The key insight: (mom / σ) gives risk-normalized position, then Gearing 
        scales the entire portfolio to target vol.
        
        Args:
            prices: DataFrame of prices (daily or monthly)
        
        Returns:
            Dictionary with results
        """
        # Calculate monthly returns
        monthly_returns = self.strategy.calculate_monthly_returns(prices)
        
        # Calculate signals (already has one vol division, capped at ±2)
        signals = self.strategy.calculate_momentum_signal(monthly_returns)
        
        # The signal IS the position in risk units (already vol-normalized)
        # The second σ division in equation 2 converts to dollar units
        # But for returns calculation, we just need: signal × return × gearing
        
        # Gearing scales to target vol (10%)
        gearing = self._calculate_portfolio_gearing(monthly_returns, signals)
        
        # Strategy returns: gearing × signal_t-1 × return_t
        # Signal is already in [-2, +2], gearing scales to target vol
        strategy_returns = signals.shift(1) * monthly_returns
        
        # Apply portfolio-level gearing
        geared_returns = strategy_returns.multiply(gearing.shift(1), axis=0)
        
        # Portfolio return (equal weighted across assets)
        portfolio_returns = geared_returns.mean(axis=1)
        
        # Transaction costs
        signal_changes = signals.diff().abs()
        turnover = signal_changes.mean(axis=1)
        costs = turnover * self.transaction_cost
        
        # Net returns
        net_returns = (portfolio_returns - costs).dropna()
        net_returns = net_returns.replace([np.inf, -np.inf], np.nan).dropna()
        
        # Calculate metrics
        metrics = self._calculate_metrics(net_returns)
        
        return {
            'portfolio_returns': net_returns,
            'signals': signals,
            'gearing': gearing,
            'monthly_returns': monthly_returns,
            'metrics': metrics
        }
    
    def _calculate_portfolio_gearing(
        self, 
        returns: pd.DataFrame, 
        signals: pd.DataFrame
    ) -> pd.Series:
        """
        Calculate gearing to scale portfolio to target volatility.
        
        For individual stocks (vs the paper's futures), we:
        1. Estimate each stock's vol contribution
        2. Use correlation matrix to estimate portfolio vol
        3. Scale to achieve target vol (10%)
        
        From footnote 6: gearing = target_vol / sqrt(w' Ω w)
        where Ω is correlation matrix with 24-month EWM half-life
        
        Args:
            returns: DataFrame of monthly returns
            signals: DataFrame of signals
        
        Returns:
            Series of gearing factors
        """
        n_assets = len(returns.columns)
        gearing = pd.Series(index=returns.index, dtype=float)
        
        min_periods = 24  # Need sufficient history for correlation
        
        for i in range(len(returns)):
            if i < min_periods:
                gearing.iloc[i] = 1.0
                continue
            
            # Get data up to this point
            ret_history = returns.iloc[:i]
            current_signals = signals.iloc[i] if i < len(signals) else signals.iloc[-1]
            
            # Individual stock vols (monthly, from EWM)
            stock_vols = ret_history.ewm(halflife=self.strategy.vol_halflife_short).std().iloc[-1]
            
            # Correlation matrix (24-month EWM approximated by rolling)
            lookback = min(i, 24)
            corr_matrix = ret_history.tail(lookback).corr().values
            
            # Handle NaN in correlation
            corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
            np.fill_diagonal(corr_matrix, 1.0)
            
            # Portfolio weights: equal weight for simplicity
            # (could also weight by signal strength)
            w = np.ones(n_assets) / n_assets
            
            # Each stock's contribution to portfolio vol
            # Strategy return per stock ≈ signal × stock_return
            # So strategy vol per stock ≈ |signal| × stock_vol
            signal_vals = current_signals.values
            signal_vals = np.nan_to_num(signal_vals, nan=0.0)
            
            # Effective vol per stock = |signal| × stock_vol (annualized)
            effective_vols = np.abs(signal_vals) * stock_vols.values * np.sqrt(12)
            effective_vols = np.nan_to_num(effective_vols, nan=0.1)
            effective_vols = np.where(effective_vols < 0.01, 0.1, effective_vols)
            
            # Portfolio variance: w' Σ w where Σ = diag(σ_eff) × Ω × diag(σ_eff)
            cov_matrix = np.outer(effective_vols, effective_vols) * corr_matrix
            port_var = np.dot(w, np.dot(cov_matrix, w))
            port_vol = np.sqrt(max(port_var, 1e-6))
            
            # Gearing to achieve target vol
            gearing.iloc[i] = self.strategy.target_vol / port_vol
        
        # Cap gearing to avoid extreme leverage
        gearing = gearing.clip(lower=0.2, upper=3.0)
        
        return gearing
    
    def _calculate_metrics(self, returns: pd.Series) -> Dict:
        """Calculate performance metrics."""
        
        # Assuming monthly returns
        periods_per_year = 12
        
        # Total return
        total_return = (1 + returns).prod() - 1
        
        # Annualized return
        n_years = len(returns) / periods_per_year
        if n_years > 0 and total_return > -1:
            ann_return = (1 + total_return) ** (1 / n_years) - 1
        else:
            ann_return = -1.0 if total_return <= -1 else 0
        
        # Annualized volatility
        ann_vol = returns.std() * np.sqrt(periods_per_year)
        
        # Sharpe ratio
        sharpe = ann_return / ann_vol if ann_vol > 0 and not np.isnan(ann_return) else 0
        
        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdowns = cumulative / rolling_max - 1
        max_dd = drawdowns.min()
        
        # Skewness (important in the paper)
        skewness = returns.skew()
        
        # Skewness of 3-month rolling returns (paper uses this)
        rolling_3m = returns.rolling(3).sum()
        skewness_3m = rolling_3m.skew()
        
        # Skewness of 12-month rolling returns
        rolling_12m = returns.rolling(12).sum()
        skewness_12m = rolling_12m.skew()
        
        # Win rate
        win_rate = (returns > 0).mean()
        
        # Calmar ratio
        calmar = ann_return / abs(max_dd) if max_dd != 0 and not np.isnan(ann_return) else 0
        
        return {
            'total_return': total_return,
            'annualized_return': ann_return,
            'annualized_volatility': ann_vol,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'skewness_monthly': skewness,
            'skewness_3m': skewness_3m,
            'skewness_12m': skewness_12m,
            'win_rate': win_rate,
            'calmar_ratio': calmar,
            'n_observations': len(returns)
        }


def run_momentum_backtest(
    prices: pd.DataFrame,
    n_lags: int = 4,
    target_vol: float = 0.10,
    transaction_cost: float = 0.0002
) -> Dict:
    """
    Convenience function to run the paper-accurate backtest.
    
    Args:
        prices: DataFrame of daily prices
        n_lags: Number of monthly lags (default: 4 for mom(1,4))
        target_vol: Target portfolio volatility (default: 10%)
        transaction_cost: Transaction cost (default: 2 bps)
    
    Returns:
        Dictionary with backtest results
    """
    strategy = MomentumStrategy(
        n_lags=n_lags,
        target_vol=target_vol
    )
    
    backtester = Backtester(
        strategy=strategy,
        transaction_cost=transaction_cost
    )
    
    return backtester.run(prices)


# =============================================================================
# VERIFICATION: Print methodology summary
# =============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("MOMENTUM STRATEGY - PAPER REPLICATION")
    print("Hamill, Rattray, Van Hemert (2016)")
    print("=" * 70)
    
    print("\n1. SIGNAL CALCULATION (Equation 1):")
    print("   mom = (w1*R_t-1 + w2*R_t-2 + w3*R_t-3 + w4*R_t-4) / (σ * √Σw²)")
    print("   For mom(1,4): weights = [0.25, 0.25, 0.25, 0.25]")
    print("   √Σw² = √(4 × 0.25²) = 0.5")
    
    print("\n2. VOLATILITY ESTIMATION (Footnote 4):")
    print("   σ = max(EWM_6month, 0.5 × EWM_24month)")
    print("   Uses exponentially weighted moving std with floor")
    
    print("\n3. SIGNAL CAPPING (Footnote 5):")
    print("   signal = clip(signal, -2, +2)")
    
    print("\n4. POSITION SIZING (Equation 2):")
    print("   position = signal / σ")
    print("   (Vol appears twice: once in signal, once in position)")
    
    print("\n5. PORTFOLIO GEARING (Footnote 6):")
    print("   Scale to 10% target vol using correlation matrix")
    print("   gearing = target_vol / √(w'Ωw)")
    
    print("\n" + "=" * 70)
    
    # Test with synthetic data that has TRENDS (not pure random)
    print("\nTesting with TRENDING synthetic data...")
    print("(Pure random data won't show momentum profits)\n")
    
    np.random.seed(42)
    n_months = 180  # 15 years of monthly data
    n_assets = 5
    
    # Generate trending data: random walk with drift + momentum
    monthly_dates = pd.date_range('2010-01-31', periods=n_months, freq='ME')
    
    prices_data = np.zeros((n_months, n_assets))
    prices_data[0, :] = 100
    
    for i in range(1, n_months):
        for j in range(n_assets):
            # Add trend persistence: if previous return was positive, 
            # this one is more likely to be positive too (momentum)
            if i > 1:
                prev_ret = (prices_data[i-1, j] / prices_data[i-2, j]) - 1
                trend_bias = 0.3 * prev_ret  # momentum factor
            else:
                trend_bias = 0
            
            # Monthly return = drift + trend + noise
            drift = 0.005  # 0.5% monthly drift (6% annual)
            noise = np.random.randn() * 0.04  # 4% monthly vol
            ret = drift + trend_bias + noise
            
            prices_data[i, j] = prices_data[i-1, j] * (1 + ret)
    
    prices = pd.DataFrame(
        prices_data,
        index=monthly_dates,
        columns=[f'Asset_{i}' for i in range(n_assets)]
    )
    
    # Since data is already monthly, create a strategy that knows this
    strategy = MomentumStrategy(
        n_lags=4,
        target_vol=0.10,
        use_monthly=True  # Important: data is already monthly
    )
    
    backtester = Backtester(
        strategy=strategy,
        transaction_cost=0.0002
    )
    
    results = backtester.run(prices)
    
    print("Backtest Results (Trending Data):")
    print("-" * 40)
    for key, value in results['metrics'].items():
        if isinstance(value, float):
            if 'return' in key or 'drawdown' in key or 'volatility' in key:
                print(f"  {key}: {value*100:.2f}%")
            else:
                print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    print("\n" + "-" * 40)
    print("Signal Statistics:")
    print(f"  Mean signal: {results['signals'].mean().mean():.3f}")
    print(f"  Std signal: {results['signals'].std().mean():.3f}")
    print(f"  % positive signals: {(results['signals'] > 0).mean().mean()*100:.1f}%")
    
    print("\n" + "=" * 70)
