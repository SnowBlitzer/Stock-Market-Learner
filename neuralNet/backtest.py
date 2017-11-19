# backtest.py


#######################################################################################################################
# STRATEGY - A Strategy class receives a Pandas DataFrame of bars, i.e. a list of Open-High-Low-Close-Volume (OHLCV) 
# data points at a particular frequency. The Strategy will produce a list of signals, which consist of a timestamp 
# and an element from the set {1,0,−1} indicating a long (buy), hold, or short (sell) signal respectively.
# PORTFOLIO - The majority of the backtesting work will occur in the Portfolio class. It will receive a set of signals
# (as described above) and create a series of positions, allocated against a cash component. The job of the Portfolio
# object is to produce an equity curve, incorporate basic transaction costs and keep track of trades.
# PERFORMANCE - The Performance object takes a portfolio and produces a set of statistics about its performance. In 
# particular it will output risk/return characteristics (Sharpe, Sortino and Information Ratios), trade/profit 
# metrics and drawdown information.
#######################################################################################################################

from abc import ABCMeta, abstractmethod

class Strategy(object):
    """Strategy is an abstract base class providing an interface for
    all subsequent (inherited) trading strategies.

    The goal of a (derived) Strategy object is to output a list of signals,
    which has the form of a time series indexed pandas DataFrame.

    In this instance only a single symbol/instrument is supported."""

    __metaclass__ = ABCMeta

    @abstractmethod
    def generate_signals(self):
        """An implementation is required to return the DataFrame of symbols 
        containing the signals to go long, short or hold (1, -1 or 0)."""
        raise NotImplementedError("Should implement generate_signals()!")
		
class Portfolio(object):
    """An abstract base class representing a portfolio of 
    positions (including both instruments and cash), determined
    on the basis of a set of signals provided by a Strategy."""

    __metaclass__ = ABCMeta

    @abstractmethod
    def generate_positions(self):
        """Provides the logic to determine how the portfolio 
        positions are allocated on the basis of forecasting
        signals and available cash."""
        raise NotImplementedError("Should implement generate_positions()!")

    @abstractmethod
    def backtest_portfolio(self):
        """Provides the logic to generate the trading orders
        and subsequent equity curve (i.e. growth of total equity),
        as a sum of holdings and cash, and the bar-period returns
        associated with this curve based on the 'positions' DataFrame.

        Produces a portfolio object that can be examined by 
        other classes/functions."""
        raise NotImplementedError("Should implement backtest_portfolio()!")

