import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict
from Classes.Result import Result
from Classes.Strategy import Strategy
 
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class Backtester:
    def __init__(self, data: pd.DataFrame, dates_universe: Dict[str, List[str]] = None) -> None:
        """
        Initialise le backtester avec les données de prix et optionnellement un dictionnaire d'univers par date.

        Args:
            data: pd.DataFrame ou pd.Series avec les prix
            dates_universe: Dict[str, List[str]], dictionnaire avec dates (YYYY-MM-DD) et listes de tickers
        """
        # Gestion des données
        if isinstance(data, pd.Series):
            self.data = pd.DataFrame(data)
            if self.data.columns[0] == 0:
                self.data.columns = ['Asset']
        else:
            self.data = data

        if not isinstance(self.data, pd.DataFrame):
            raise TypeError("Les données doivent être un DataFrame ou une Series")

        # Gestion du dictionnaire d'univers par date
        self.dates_universe = {}
        if dates_universe is not None:
            if not isinstance(dates_universe, dict):
                raise TypeError("dates_universe doit être un dictionnaire")
            
            # Vérifie que chaque clé est une date valide et chaque valeur une liste de strings
            for date_str, tickers in dates_universe.items():
                # Vérifie le format de la date
                try:
                    pd.to_datetime(date_str)
                except ValueError:
                    raise ValueError(f"La clé {date_str} n'est pas une date valide au format YYYY-MM-DD")
                
                if not isinstance(tickers, list) or not all(isinstance(t, str) for t in tickers):
                    raise ValueError(f"Les tickers pour la date {date_str} doivent être une liste de strings")
                
                # Vérifie que les tickers existent dans data
                invalid_tickers = [t for t in tickers if t not in self.data.columns]
                if invalid_tickers:
                    raise ValueError(f"Tickers non trouvés dans les données: {invalid_tickers}")
            
            self.dates_universe = dates_universe

    def run(self, 
            start_date: Optional[pd.Timestamp] = None, 
            end_date: Optional[pd.Timestamp] = None, 
            freq: int = 30, 
            window: int = 365, 
            aum: float = 100, 
            transaction_cost: float = 0.0, 
            strategy: Strategy = None) -> Result:
        """
        Exécute le backtest avec les paramètres spécifiés.

        Args:
        - start_date (pd.Timestamp): Date de début du backtest.
        - end_date (pd.Timestamp): Date de fin du backtest.
        - freq (int): Fréquence de rééquilibrage en jours.
        - window (int): Fenêtre de formation en jours.
        - aum (float): Actifs sous gestion.
        - transaction_cost (float): Coût de transaction en pourcentage.
        - strategy (Strategy): Stratégie de trading à tester.

        Returns:
        - Result: Résultats du backtest.
        """

        if strategy is None:
            raise ValueError("A strategy must be provided to run the backtest.")
        
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

        self.handle_missing_data()
        self.calculate_weights(strategy)
        self.calculate_performance()

        if not hasattr(strategy, 'name'):
            strategy.name = strategy.__class__.__name__

        return Result(self.performance, self.weights, self.total_transaction_costs, strategy.name)

    def handle_missing_data(self) -> None:
        # Supprimer les colonnes avec toutes les valeurs manquantes
        self.data = self.data.dropna(axis=1, how='all')

        # Sélectionner uniquement les colonnes numériques
        self.data = self.data.select_dtypes(include=[np.number])

        # Remplir les valeurs manquantes entre le premier et le dernier index valides pour chaque colonne
        for col in self.data.columns:
            first_valid_index = self.data[col].first_valid_index()
            last_valid_index = self.data[col].last_valid_index()

            if first_valid_index is not None and last_valid_index is not None:
                self.data.loc[first_valid_index:last_valid_index, col] = self.data.loc[first_valid_index:last_valid_index, col].ffill()

        if self.data.empty:
            raise ValueError("No data available after handling missing values.")


    def calculate_weights(self, strategy: Strategy) -> None: 
        """
        Calcule les poids optimaux pour chaque date de rééquilibrage.

        Args:
        - strategy (Strategy): Stratégie de trading à tester.

        Returns:
        - None
        """

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

        # Initialize lists to collect weights and dates
        weights_list = [last_weights]
        dates_list = [(current_date - pd.DateOffset(days=1))]

        for current_date in rebalancing_dates:
            # Define training period
            train_start = current_date - window_dt
            train_end = current_date - pd.DateOffset(days=1)

            # Get training data
            price_window = prices[train_start:train_end]

            # Filtrer selon l'univers de dates
            if self.dates_universe:
                # Convertir toutes les dates du dictionnaire en datetime
                universe_dates = [pd.to_datetime(date) for date in self.dates_universe.keys()]
                
                # Trouver la date d'univers la plus récente avant la date courante
                available_dates = [date for date in universe_dates if date <= current_date]
                
                if available_dates:
                    reference_date = max(available_dates)
                    active_tickers = self.dates_universe[reference_date.strftime('%Y-%m-%d')]
                    price_window = price_window[active_tickers]
                else:
                    print(f"Pas d'univers défini avant {current_date}")
                    price_window = pd.DataFrame()  # DataFrame vide si pas de date valide

            # Drop columns with missing values
            price_window_filtered = price_window.dropna(axis=1)
            if price_window_filtered.empty:
                print(f"No data available for {current_date}. Skipping...")
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

    def calculate_performance(self) -> None:
        """
        Calcule la performance du portefeuille en utilisant les poids calculés.

        Returns:
        - None
        """
        
        balance = self.aum

        # Get the first date where weights are available
        first_valid_date = self.weights.first_valid_index()
        
        # Get the data within the specified date range
        df = self.data[self.start_date:self.end_date]

        # Calculate returns
        returns = df.pct_change()[1:]

        # Initialize total transaction costs and previous weights
        self.total_transaction_costs = 0
        previous_weights = pd.Series(0.0, index=self.weights.columns)

        # Initialize lists to store portfolio values and dates
        portfolio_values = [self.aum]
        dates = [first_valid_date - pd.DateOffset(days=1)]

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
