import numpy as np
import pandas as pd
from typing import List, Tuple
from Classes.Result import Result

class Backtester:

    def __init__(self, data):

        self.data = data
        #self.returns = self.data.pct_change().iloc[1:]

    def run(self, start_date=None, end_date=None, freq=30, window=365, aum=100, transaction_cost=0.0, strategy=None):

        if start_date is None:
            start_date = self.returns.index[0]

        if end_date is None:
            end_date = self.returns.index[-1]

        self.start_date = start_date
        self.end_date = end_date

        self.freq = freq
        self.window = window

        self.aum = aum
        self.transaction_cost = transaction_cost

        self.calculate_weights(strategy)
        self.calculate_performance()

        return Result(self.performance, self.weights, self.total_transaction_costs)

    def calculate_weights(self, strategy):
        # Initialize lists to collect weights and dates
        weights_list = []
        dates_list = []

        # Define rebalancing frequency and training window
        freq_dt = pd.DateOffset(days=self.freq)
        window_dt = pd.DateOffset(days=self.window)

        # Calculate start date with window
        start_date_with_window = pd.to_datetime(self.start_date) - window_dt

        # Get price data within the window
        prices = self.data[start_date_with_window:self.end_date]

        # Generate rebalancing dates in reverse order
        rebalancing_dates = []
        current_date = prices.index[-1]
        while current_date >= prices.index[0] + window_dt:
            rebalancing_dates.append(current_date)
            current_date -= freq_dt

        # Reverse the list to have dates in ascending order
        rebalancing_dates.reverse()

        # Initialize last_weights as zeros
        last_weights = pd.Series(0.0, index=prices.columns)

        for current_date in rebalancing_dates:
            # Define training period
            train_start = current_date - window_dt
            train_end = current_date - pd.DateOffset(days=1)

            # Get training data
            price_window = prices[train_start:train_end]

            # Drop columns with missing values
            price_window_filtered = price_window.dropna(axis=1)

            # Get new weights from strategy
            final_optimal_weights = strategy.get_position(price_window_filtered, last_weights)
            last_weights = final_optimal_weights

            # Collect weights and date
            weights_list.append(final_optimal_weights)
            dates_list.append(current_date)

        # Create DataFrame from collected weights
        optimal_weights_df = pd.DataFrame(weights_list, index=dates_list)

        # Assign the calculated weights
        self.weights = optimal_weights_df.fillna(0.0)

    def calculate_performance(self):
        balance = self.aum

        # Get the first date where weights are available
        first_valid_date = self.weights.first_valid_index()

        # Get the data within the specified date range
        df = self.data[self.start_date:self.end_date]

        # Backfill missing values to handle days where weights are not available
        df = df.bfill()
        df = df.ffill()

        # Calculate returns
        returns = df.pct_change(fill_method=None).dropna()

        # Initialize total transaction costs and previous weights
        self.total_transaction_costs = 0
        previous_weights = pd.Series(0.0, index=self.weights.columns)

        # Initialize lists to store portfolio values and dates
        portfolio_values = []
        dates = []

        # Get the list of dates to iterate over
        date_range = returns.loc[first_valid_date:].index

        for date in date_range:
            # Update weights if new weights are available
            if date in self.weights.index:
                current_weights = self.weights.loc[date]

                # Calculate changes in positions
                changes = (current_weights - previous_weights) * balance

                # Calculate transaction costs
                transaction_costs = changes.abs().sum() * (self.transaction_cost / 100)

                # Update total transaction costs and subtract from balance
                self.total_transaction_costs += transaction_costs
                balance -= transaction_costs

                # Update previous weights
                previous_weights = current_weights.copy()
            else:
                current_weights = previous_weights.copy()

            # Calculate portfolio return
            portfolio_return = (current_weights * returns.loc[date]).sum()

            # Update balance
            balance *= (1 + portfolio_return)

            # Store the portfolio value and date
            portfolio_values.append(balance)
            dates.append(date)

        # Create a Series for the portfolio performance
        self.performance = pd.Series(portfolio_values, index=dates)

