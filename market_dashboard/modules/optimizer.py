import pandas as pd
from modules.strategies import MovingAverageCrossover, RSIStrategy, BollingerBandsStrategy
from modules.utils import compute_returns

STRATEGY_MAPPING = {
    'MA Crossover': MovingAverageCrossover,
    'RSI (Threshold)': lambda **kwargs: RSIStrategy(mode='threshold', **kwargs),
    'RSI (Mean-Reversion)': lambda **kwargs: RSIStrategy(mode='mean_reversion', **kwargs),
    'Bollinger Bands': BollingerBandsStrategy
}


def grid_search_strategy(price, indicators, strategy_name, params_grid, interval='1d'):
    """Perform parameter grid search and return best config by Sharpe."""
    from modules.portfolio import sharpe_ratio, max_drawdown, win_rate

    best = None
    results = []

    for config in params_grid:
        strategy = STRATEGY_MAPPING[strategy_name](**config)
        signals = strategy.generate_signals(price, indicators)
        backtest = strategy.compute_positions_and_equity(signals, price, initial_equity=100)
        metrics = strategy.compute_metrics(backtest['equity'], backtest['daily_return'], interval=interval)

        summary = {
            'config': config,
            'total_return': metrics['total_return'],
            'sharpe_ratio': metrics['sharpe_ratio'],
            'max_drawdown': metrics['max_drawdown'],
            'win_rate': metrics['win_rate']
        }
        results.append(summary)

        if best is None or summary['sharpe_ratio'] > best['sharpe_ratio']:
            best = summary

    return {
        'best': best,
        'results': pd.DataFrame(results)
    }
