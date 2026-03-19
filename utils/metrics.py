import torch
import numpy as np
from typing import Optional, Union, Tuple


# =============================================================================
# Basic Regression Metrics
# =============================================================================

def mean_squared_error(y_true, y_pred):    
    """Calculate Mean Squared Error."""
    mse = torch.mean((y_true - y_pred) ** 2)
    return mse.item()


def mean_absolute_error(y_true, y_pred):
    """Calculate Mean Absolute Error."""
    mae = torch.mean(torch.abs(y_true - y_pred))
    return mae.item()


def r2_score(y_true, y_pred):
    """
    Calculate R² Score (Coefficient of Determination).
    
    R² = 1 - (SS_res / SS_tot)
    where SS_res = Σ(y_true - y_pred)²
          SS_tot = Σ(y_true - mean(y_true))²
    """
    total_sum_squares = torch.sum((y_true - torch.mean(y_true)) ** 2)
    residual_sum_squares = torch.sum((y_true - y_pred) ** 2)
    r2 = 1 - (residual_sum_squares / (total_sum_squares + 1e-8))
    return r2.item()


def mean_absolute_percentage_error(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error."""
    mape = torch.mean(torch.abs((y_true - y_pred) / (y_true + 1e-8)))
    return mape.item()


# =============================================================================
# Economic Performance Metrics (Portfolio Level)
# Based on "(Re-)Imag(in)ing Price Trends" paper
# =============================================================================

def sharpe_ratio(returns: Union[torch.Tensor, np.ndarray], 
                 risk_free_rate: float = 0.0,
                 periods_per_year: int = 252,
                 annualize: bool = True) -> float:
    """
    Calculate Sharpe Ratio.
    
    Sharpe Ratio = (E[R] - Rf) / σ[R]
    
    Args:
        returns: Portfolio returns (can be daily, weekly, or monthly)
        risk_free_rate: Risk-free rate (annualized)
        periods_per_year: Trading periods per year (252 for daily, 52 for weekly, 12 for monthly)
        annualize: Whether to annualize the Sharpe ratio
        
    Returns:
        Sharpe ratio (annualized if annualize=True)
    """
    if isinstance(returns, torch.Tensor):
        returns = returns.detach().cpu().numpy()
    
    returns = np.asarray(returns).flatten()
    
    # Remove NaN values
    returns = returns[~np.isnan(returns)]
    
    if len(returns) == 0:
        return 0.0
    
    # Calculate excess returns
    excess_returns = returns - (risk_free_rate / periods_per_year)
    
    # Calculate Sharpe ratio
    mean_excess = np.mean(excess_returns)
    std_excess = np.std(excess_returns, ddof=1)
    
    if std_excess == 0:
        return 0.0
    
    sharpe = mean_excess / std_excess
    
    if annualize:
        sharpe = sharpe * np.sqrt(periods_per_year)
    
    return float(sharpe)


def portfolio_turnover(weights_t: Union[torch.Tensor, np.ndarray],
                      weights_t_minus_1: Union[torch.Tensor, np.ndarray],
                      returns_t: Union[torch.Tensor, np.ndarray]) -> float:
    """
    Calculate portfolio turnover rate.
    
    Turnover = Σ|w_{i,t+1} - w_{i,t}(1 + r_{i,t+1}) / (1 + Σw_{j,t}r_{j,t+1})|
    
    This measures the fraction of the portfolio that is rebalanced.
    
    Args:
        weights_t: Portfolio weights at time t (after rebalancing)
        weights_t_minus_1: Portfolio weights at time t-1
        returns_t: Asset returns from t-1 to t
        
    Returns:
        Turnover rate (0 to 2, where 0 = no trading, 2 = complete replacement)
    """
    if isinstance(weights_t, torch.Tensor):
        weights_t = weights_t.detach().cpu().numpy()
    if isinstance(weights_t_minus_1, torch.Tensor):
        weights_t_minus_1 = weights_t_minus_1.detach().cpu().numpy()
    if isinstance(returns_t, torch.Tensor):
        returns_t = returns_t.detach().cpu().numpy()
    
    # Calculate portfolio return
    portfolio_return = np.sum(weights_t_minus_1 * returns_t)
    
    # Calculate weights after drift (before rebalancing)
    weights_after_drift = weights_t_minus_1 * (1 + returns_t) / (1 + portfolio_return)
    
    # Calculate turnover
    turnover = np.sum(np.abs(weights_t - weights_after_drift))
    
    return float(turnover)


def net_of_cost_sharpe(returns: Union[torch.Tensor, np.ndarray],
                       turnover_rates: Union[torch.Tensor, np.ndarray],
                       transaction_cost_bps: float = 10.0,
                       periods_per_year: int = 252,
                       risk_free_rate: float = 0.0) -> float:
    """
    Calculate Sharpe ratio after deducting transaction costs.
    
    Net Return_t = Gross Return_t - (Turnover_t × Transaction Cost)
    
    Args:
        returns: Gross portfolio returns
        turnover_rates: Portfolio turnover at each period
        transaction_cost_bps: Transaction cost in basis points (e.g., 10 for 10 bps)
        periods_per_year: Trading periods per year
        risk_free_rate: Risk-free rate (annualized)
        
    Returns:
        Annualized Sharpe ratio after transaction costs
    """
    if isinstance(returns, torch.Tensor):
        returns = returns.detach().cpu().numpy()
    if isinstance(turnover_rates, torch.Tensor):
        turnover_rates = turnover_rates.detach().cpu().numpy()
    
    returns = np.asarray(returns).flatten()
    turnover_rates = np.asarray(turnover_rates).flatten()
    
    # Calculate transaction costs
    transaction_costs = turnover_rates * (transaction_cost_bps / 10000)
    
    # Calculate net returns
    net_returns = returns - transaction_costs
    
    # Calculate Sharpe ratio on net returns
    return sharpe_ratio(net_returns, risk_free_rate, periods_per_year, annualize=True)


def long_short_portfolio_returns(predictions: Union[torch.Tensor, np.ndarray],
                                 actual_returns: Union[torch.Tensor, np.ndarray],
                                 n_quantiles: int = 10,
                                 long_quantile: int = 10,
                                 short_quantile: int = 1,
                                 weight_method: str = 'equal') -> np.ndarray:
    """
    Calculate returns of a long-short portfolio based on predictions.
    
    This constructs a High-Low (H-L) portfolio by:
    1. Ranking assets by predicted probability/score
    2. Going long the top quantile
    3. Going short the bottom quantile
    
    Args:
        predictions: Predicted scores/probabilities for each asset at each time (T × N)
        actual_returns: Actual returns for each asset at each time (T × N)
        n_quantiles: Number of quantiles to divide assets into (default: 10 for deciles)
        long_quantile: Which quantile to long (10 = top decile)
        short_quantile: Which quantile to short (1 = bottom decile)
        weight_method: 'equal' for equal-weight or 'value' for value-weight
        
    Returns:
        Array of portfolio returns at each time period (T,)
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(actual_returns, torch.Tensor):
        actual_returns = actual_returns.detach().cpu().numpy()
    
    # Ensure 2D arrays
    if predictions.ndim == 1:
        predictions = predictions.reshape(1, -1)
    if actual_returns.ndim == 1:
        actual_returns = actual_returns.reshape(1, -1)
    
    T, N = predictions.shape
    portfolio_returns = np.zeros(T)
    
    for t in range(T):
        pred_t = predictions[t]
        ret_t = actual_returns[t]
        
        # Remove NaN values
        valid_mask = ~(np.isnan(pred_t) | np.isnan(ret_t))
        pred_valid = pred_t[valid_mask]
        ret_valid = ret_t[valid_mask]
        
        if len(pred_valid) < n_quantiles:
            portfolio_returns[t] = np.nan
            continue
        
        # Rank predictions and assign quantiles
        quantile_labels = np.zeros(len(pred_valid))
        sorted_indices = np.argsort(pred_valid)
        quantile_size = len(pred_valid) // n_quantiles
        
        for q in range(1, n_quantiles + 1):
            start_idx = (q - 1) * quantile_size
            end_idx = q * quantile_size if q < n_quantiles else len(pred_valid)
            quantile_labels[sorted_indices[start_idx:end_idx]] = q
        
        # Select long and short positions
        long_mask = quantile_labels == long_quantile
        short_mask = quantile_labels == short_quantile
        
        # Calculate portfolio return
        if weight_method == 'equal':
            long_return = np.mean(ret_valid[long_mask]) if np.sum(long_mask) > 0 else 0
            short_return = np.mean(ret_valid[short_mask]) if np.sum(short_mask) > 0 else 0
            portfolio_returns[t] = long_return - short_return
        else:
            # For value-weighted, you would need market cap data
            raise NotImplementedError("Value-weighted portfolios require market cap data")
    
    return portfolio_returns


def cumulative_volatility_adjusted_returns(returns: Union[torch.Tensor, np.ndarray],
                                          target_volatility: float = 0.2,
                                          periods_per_year: int = 252) -> np.ndarray:
    """
    Calculate cumulative returns adjusted to a target volatility level.
    
    This normalizes different strategies to the same risk level for comparison.
    
    Args:
        returns: Strategy returns
        target_volatility: Target annualized volatility (e.g., 0.2 for 20%)
        periods_per_year: Trading periods per year
        
    Returns:
        Cumulative volatility-adjusted returns
    """
    if isinstance(returns, torch.Tensor):
        returns = returns.detach().cpu().numpy()
    
    returns = np.asarray(returns).flatten()
    
    # Calculate current volatility
    current_vol = np.std(returns, ddof=1) * np.sqrt(periods_per_year)
    
    if current_vol == 0:
        return np.zeros_like(returns)
    
    # Scale returns to target volatility
    scaling_factor = target_volatility / current_vol
    adjusted_returns = returns * scaling_factor
    
    # Calculate cumulative returns
    cumulative_returns = np.cumprod(1 + adjusted_returns) - 1
    
    return cumulative_returns


# =============================================================================
# Statistical Prediction Metrics (Stock Level)
# =============================================================================

def cross_entropy_loss(y_true: Union[torch.Tensor, np.ndarray],
                      y_pred: Union[torch.Tensor, np.ndarray],
                      eps: float = 1e-8) -> float:
    """
    Calculate binary cross-entropy loss.
    
    L(y, ŷ) = -y·log(ŷ) - (1-y)·log(1-ŷ)
    
    Args:
        y_true: True binary labels (0 or 1)
        y_pred: Predicted probabilities (0 to 1)
        eps: Small constant for numerical stability
        
    Returns:
        Mean cross-entropy loss
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    
    # Clip predictions to avoid log(0)
    y_pred = np.clip(y_pred, eps, 1 - eps)
    
    # Calculate cross-entropy
    ce = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    return float(np.mean(ce))


def prediction_accuracy(y_true: Union[torch.Tensor, np.ndarray],
                       y_pred: Union[torch.Tensor, np.ndarray],
                       threshold: float = 0.5) -> float:
    """
    Calculate classification accuracy.
    
    Args:
        y_true: True binary labels (0 or 1)
        y_pred: Predicted probabilities (0 to 1)
        threshold: Threshold for converting probabilities to binary predictions
        
    Returns:
        Accuracy (0 to 1)
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    # Convert probabilities to binary predictions
    y_pred_binary = (y_pred >= threshold).astype(int)
    
    # Calculate accuracy
    accuracy = np.mean(y_true == y_pred_binary)
    
    return float(accuracy)


def mcfadden_pseudo_r2(y_true: Union[torch.Tensor, np.ndarray],
                       y_pred: Union[torch.Tensor, np.ndarray],
                       eps: float = 1e-8) -> float:
    """
    Calculate McFadden's Pseudo R².
    
    Pseudo R² = 1 - (log-likelihood of model / log-likelihood of null model)
    
    The null model predicts the sample mean probability for all observations.
    
    Args:
        y_true: True binary labels (0 or 1)
        y_pred: Predicted probabilities from the model (0 to 1)
        eps: Small constant for numerical stability
        
    Returns:
        McFadden's Pseudo R² (typically 0.2-0.4 is considered good fit)
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    # Clip predictions
    y_pred = np.clip(y_pred, eps, 1 - eps)
    
    # Log-likelihood of the model
    log_likelihood = np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    # Null model: predict sample mean for all observations
    y_mean = np.mean(y_true)
    y_mean = np.clip(y_mean, eps, 1 - eps)
    log_likelihood_null = np.sum(y_true * np.log(y_mean) + (1 - y_true) * np.log(1 - y_mean))
    
    # Calculate pseudo R²
    if log_likelihood_null == 0:
        return 0.0
    
    pseudo_r2 = 1 - (log_likelihood / log_likelihood_null)
    
    return float(pseudo_r2)


def decile_analysis(predictions: Union[torch.Tensor, np.ndarray],
                   actual_returns: Union[torch.Tensor, np.ndarray],
                   n_quantiles: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Analyze prediction accuracy by sorting into quantiles.
    
    This checks if higher predicted probabilities correspond to higher actual returns
    (monotonicity test).
    
    Args:
        predictions: Predicted scores/probabilities (T × N or flattened)
        actual_returns: Actual returns (T × N or flattened)
        n_quantiles: Number of quantiles (default: 10 for deciles)
        
    Returns:
        Tuple of (quantile_labels, average_returns_per_quantile)
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(actual_returns, torch.Tensor):
        actual_returns = actual_returns.detach().cpu().numpy()
    
    predictions = predictions.flatten()
    actual_returns = actual_returns.flatten()
    
    # Remove NaN values
    valid_mask = ~(np.isnan(predictions) | np.isnan(actual_returns))
    predictions = predictions[valid_mask]
    actual_returns = actual_returns[valid_mask]
    
    # Assign quantiles
    quantile_labels = np.zeros(len(predictions), dtype=int)
    sorted_indices = np.argsort(predictions)
    quantile_size = len(predictions) // n_quantiles
    
    for q in range(1, n_quantiles + 1):
        start_idx = (q - 1) * quantile_size
        end_idx = q * quantile_size if q < n_quantiles else len(predictions)
        quantile_labels[sorted_indices[start_idx:end_idx]] = q
    
    # Calculate average return for each quantile
    avg_returns = np.zeros(n_quantiles)
    for q in range(1, n_quantiles + 1):
        mask = quantile_labels == q
        if np.sum(mask) > 0:
            avg_returns[q - 1] = np.mean(actual_returns[mask])
    
    return quantile_labels, avg_returns


# =============================================================================
# Correlation and Regression Utilities
# =============================================================================

def cross_sectional_correlation(signal_1: Union[torch.Tensor, np.ndarray],
                                signal_2: Union[torch.Tensor, np.ndarray],
                                average_over_time: bool = True) -> Union[float, np.ndarray]:
    """
    Calculate cross-sectional correlation between two signals.
    
    For each time period, calculate correlation across assets.
    
    Args:
        signal_1: First signal (T × N)
        signal_2: Second signal (T × N)
        average_over_time: If True, return average correlation; if False, return time series
        
    Returns:
        Average correlation (if average_over_time=True) or array of correlations over time
    """
    if isinstance(signal_1, torch.Tensor):
        signal_1 = signal_1.detach().cpu().numpy()
    if isinstance(signal_2, torch.Tensor):
        signal_2 = signal_2.detach().cpu().numpy()
    
    # Ensure 2D
    if signal_1.ndim == 1:
        signal_1 = signal_1.reshape(1, -1)
    if signal_2.ndim == 1:
        signal_2 = signal_2.reshape(1, -1)
    
    T = signal_1.shape[0]
    correlations = np.zeros(T)
    
    for t in range(T):
        s1 = signal_1[t]
        s2 = signal_2[t]
        
        # Remove NaN
        valid_mask = ~(np.isnan(s1) | np.isnan(s2))
        s1_valid = s1[valid_mask]
        s2_valid = s2[valid_mask]
        
        if len(s1_valid) > 1:
            correlations[t] = np.corrcoef(s1_valid, s2_valid)[0, 1]
        else:
            correlations[t] = np.nan
    
    if average_over_time:
        return float(np.nanmean(correlations))
    else:
        return correlations


def rank_information_coefficient(predictions: Union[torch.Tensor, np.ndarray],
                                 actual_returns: Union[torch.Tensor, np.ndarray]) -> float:
    """
    Calculate Rank Information Coefficient (Rank IC).

    Rank IC is Spearman rank correlation between model scores and realized returns.
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(actual_returns, torch.Tensor):
        actual_returns = actual_returns.detach().cpu().numpy()

    predictions = np.asarray(predictions).flatten()
    actual_returns = np.asarray(actual_returns).flatten()

    if len(predictions) != len(actual_returns) or len(predictions) < 2:
        return 0.0

    # Remove NaN/Inf pairs
    valid_mask = np.isfinite(predictions) & np.isfinite(actual_returns)
    predictions = predictions[valid_mask]
    actual_returns = actual_returns[valid_mask]

    if len(predictions) < 2:
        return 0.0

    def _average_tie_ranks(x: np.ndarray) -> np.ndarray:
        # Stable sort to preserve deterministic tie handling
        sorter = np.argsort(x, kind='mergesort')
        sorted_x = x[sorter]
        n = len(sorted_x)
        ranks_sorted = np.zeros(n, dtype=float)

        i = 0
        while i < n:
            j = i
            while j + 1 < n and sorted_x[j + 1] == sorted_x[i]:
                j += 1
            # 1-based average rank for ties
            avg_rank = (i + j) / 2.0 + 1.0
            ranks_sorted[i:j + 1] = avg_rank
            i = j + 1

        ranks = np.empty(n, dtype=float)
        ranks[sorter] = ranks_sorted
        return ranks

    pred_ranks = _average_tie_ranks(predictions)
    ret_ranks = _average_tie_ranks(actual_returns)

    pred_std = np.std(pred_ranks, ddof=1)
    ret_std = np.std(ret_ranks, ddof=1)
    if pred_std < 1e-12 or ret_std < 1e-12:
        return 0.0

    ric = np.corrcoef(pred_ranks, ret_ranks)[0, 1]
    if np.isnan(ric):
        return 0.0
    return float(ric)


# =============================================================================
# Additional Helper Functions
# =============================================================================

def annualized_return(returns: Union[torch.Tensor, np.ndarray],
                     periods_per_year: int = 252) -> float:
    """Calculate annualized return from period returns."""
    if isinstance(returns, torch.Tensor):
        returns = returns.detach().cpu().numpy()
    
    returns = np.asarray(returns).flatten()
    returns = returns[~np.isnan(returns)]
    
    if len(returns) == 0:
        return 0.0
    
    # Compound returns
    total_return = np.prod(1 + returns) - 1
    n_periods = len(returns)
    
    # Annualize
    annualized = (1 + total_return) ** (periods_per_year / n_periods) - 1
    
    return float(annualized)


def annualized_volatility(returns: Union[torch.Tensor, np.ndarray],
                         periods_per_year: int = 252) -> float:
    """Calculate annualized volatility from period returns."""
    if isinstance(returns, torch.Tensor):
        returns = returns.detach().cpu().numpy()
    
    returns = np.asarray(returns).flatten()
    returns = returns[~np.isnan(returns)]
    
    if len(returns) == 0:
        return 0.0
    
    vol = np.std(returns, ddof=1) * np.sqrt(periods_per_year)
    
    return float(vol)


def maximum_drawdown(returns: Union[torch.Tensor, np.ndarray]) -> float:
    """
    Calculate maximum drawdown.
    
    MDD = max(peak - trough) / peak
    """
    if isinstance(returns, torch.Tensor):
        returns = returns.detach().cpu().numpy()
    
    returns = np.asarray(returns).flatten()
    
    # Calculate cumulative returns
    cumulative = np.cumprod(1 + returns)
    
    # Calculate running maximum
    running_max = np.maximum.accumulate(cumulative)
    
    # Calculate drawdown
    drawdown = (cumulative - running_max) / running_max
    
    # Maximum drawdown
    max_dd = np.min(drawdown)
    
    return float(abs(max_dd))


