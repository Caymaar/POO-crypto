import pandas as pd
import numpy as np
from scipy.optimize import minimize

class StrategyLowVol:
    """
   Stratégie basée sur la volatilité faible. Cette classe vise à identifier et à investir dans 
   les actifs crypto avec la plus faible volatilité historique dans un ensemble donné de données.

   Attributes:
       historical_data (pd.DataFrame): Données historiques des actifs incluant les prix de clôture.
       num_assets (int): Nombre d'actifs à sélectionner pour la stratégie.

   Methods:
       calculate_returns(): Calcule les rendements quotidiens des actifs.
       calculate_volatility(): Calcule la volatilité historique des actifs.
       select_assets(): Sélectionne les actifs avec la plus faible volatilité historique.
       calculate_initial_weights(): Calcule les poids initiaux basés sur la capitalisation boursière.
       adjust_weights_over_time(): Ajuste les poids des actifs au fil du temps en fonction de leur volatilité.
       run_strategy(): Exécute l'ensemble de la stratégie en appelant les méthodes nécessaires et renvoie les symboles sélectionnés et les poids ajustés.
   """
   
    def __init__(self, historical_data, num_assets=10):
        self.historical_data = historical_data
        self.num_assets = num_assets
    
    def calculate_returns(self):
        """Calcule les rendements quotidiens des actifs."""
        self.historical_data['Close Price'] = pd.to_numeric(self.historical_data['Close Price'], errors='coerce')

        # S'assurer que l'index est unique pour chaque paire de 'Date' et 'symbol'
        if not self.historical_data.index.is_unique:
           # Gérer les doublons ici
           self.historical_data = self.historical_data[~self.historical_data.index.duplicated(keep='last')]

        # Calcul des rendements une fois que l'index est confirmé comme unique
        self.returns = self.historical_data['Close Price'].unstack(level='symbol').pct_change()        
      
    def calculate_volatility(self):
        """Calcule la volatilité historique (écart-type des rendements) des actifs."""
        # Utiliser une fenêtre glissante pour calculer la volatilité moyenne
        self.volatility = self.returns.rolling(window=30, min_periods=1).std().mean()

    def select_assets(self):
        """ Sélection des actifs avec la plus faible volatilité moyenne. """
        self.low_vol_symbols = self.volatility.nsmallest(self.num_assets).index

    def calculate_initial_weights(self):
         """ Calculer les poids initiaux en fonction de la capitalisation boursière. """
         # S'assurer que les capitalisations boursières sont numériques
         self.historical_data['market_cap'] = pd.to_numeric(self.historical_data['market_cap'], errors='coerce')
         
         # Utiliser la capitalisation boursière de la première période disponible
         initial_market_caps = self.historical_data['market_cap'].unstack(level='symbol').iloc[0, :]
         selected_initial_market_caps = initial_market_caps.loc[self.low_vol_symbols]
         self.initial_weights = selected_initial_market_caps / selected_initial_market_caps.sum()
    
    def adjust_weights_over_time(self):
        """ Ajuster les poids à chaque période en fonction de la volatilité. """
        # Calculer la volatilité roulante pour chaque symbole sélectionné
        rolling_volatility = self.returns[self.low_vol_symbols].rolling(window=30, min_periods=1).std()
        
        # Calculer les poids relatifs en divisant la volatilité de chaque actif par la volatilité totale à chaque période
        relative_weights = rolling_volatility.div(rolling_volatility.sum(axis=1), axis=0)
        self.adjusted_weights_over_time = relative_weights.fillna(0)

    def run_strategy(self):
        """Exécute la stratégie complète."""
        self.calculate_returns()
        self.calculate_volatility()
        self.select_assets()
        self.calculate_initial_weights()
        self.adjust_weights_over_time()

        return self.low_vol_symbols, self.adjusted_weights_over_time
    

class StrategyMomentum:
    """
    Stratégie basée sur le momentum. Cette classe vise à capitaliser sur la continuation de la tendance 
    existante des actifs crypto en sélectionnant ceux qui ont performé le mieux sur une période donnée.

    Attributes:
        historical_data (pd.DataFrame): Données historiques des actifs incluant les prix de clôture.
        num_assets (int): Nombre d'actifs à sélectionner pour la stratégie.
        lookback_period (int): Période de rétrospection en jours pour calculer le momentum.

    Methods:
        calculate_returns(): Calcule les rendements des actifs.
        select_assets(): Sélectionne les actifs avec le meilleur momentum.
        calculate_weights(): Calcule les poids des actifs basés sur leur performance relative.
        run_strategy(): Exécute l'ensemble de la stratégie en appelant les méthodes nécessaires et renvoie les symboles sélectionnés et les poids ajustés.
    """
    def __init__(self, historical_data, num_assets=10, lookback_period=90):
        self.historical_data = historical_data
        self.num_assets = num_assets
        self.lookback_period = lookback_period

    def calculate_returns(self):
        """Calcule les rendements totaux sur la période spécifiée pour chaque actif."""
        # S'assurer que les prix sont numériques
        self.historical_data['Close Price'] = pd.to_numeric(self.historical_data['Close Price'], errors='coerce')

        # Assurer l'unicité de l'index avant d'unstack
        if not self.historical_data.index.is_unique:
            # Eliminer les doublons en gardant la dernière occurrence
            self.historical_data = self.historical_data[~self.historical_data.index.duplicated(keep='last')]

        # Calculer les rendements
        self.returns = self.historical_data['Close Price'].unstack(level='symbol').pct_change()
        
    def select_assets(self):
        """Sélectionne les actifs avec la meilleure performance sur la période de lookback."""
        total_returns = self.returns.rolling(window=self.lookback_period).apply(lambda x: (1 + x).prod() - 1)
        self.top_momentum_symbols = total_returns.iloc[-1].nlargest(self.num_assets).index

    def calculate_weights(self):
        """Calcule les poids des actifs basés sur la variation de la performance."""
        # Calculer la variation quotidienne de la performance pour les actifs sélectionnés
        daily_variation = self.returns[self.top_momentum_symbols].diff()
        
        # Calculer les poids relatifs en divisant la variation quotidienne de chaque actif par la variation totale
        relative_weights = daily_variation.div(daily_variation.sum(axis=1), axis=0)
        
        # Remplacer les NaN par des zéros
        self.adjusted_weights_over_time = relative_weights.fillna(0)

    def run_strategy(self):
        """Exécute la stratégie complète."""
        self.calculate_returns()
        self.select_assets()
        self.calculate_weights()

        return self.top_momentum_symbols, self.adjusted_weights_over_time

class StrategyMaximisationSharpeRatio:
    """
    Stratégie visant à maximiser le ratio de Sharpe du portefeuille, c'est-à-dire le rendement ajusté au risque. 
    Elle utilise une optimisation pour déterminer la meilleure combinaison d'actifs qui maximise ce ratio.

    Attributes:
        historical_data (pd.DataFrame): Données historiques des actifs incluant les prix de clôture.
        num_assets (int): Nombre d'actifs à sélectionner pour la stratégie.
        risk_free_symbol (str): Symbole représentant l'actif sans risque (e.g., BNBUSDT).

    Methods:
        calculate_returns(): Calcule les rendements des actifs.
        select_assets(): Sélectionne les actifs pour la stratégie basée sur la capitalisation boursière.
        optimize_portfolio(): Utilise une fonction d'optimisation pour maximiser le ratio de Sharpe.
        run_strategy(): Exécute l'ensemble de la stratégie en appelant les méthodes nécessaires et renvoie les symboles sélectionnés et les poids optimisés.
    """
    def __init__(self, historical_data, num_assets=5, risk_free_symbol='BNBUSDT'):
        self.historical_data = historical_data.dropna()  
        self.num_assets = num_assets
        self.risk_free_symbol = risk_free_symbol

    def calculate_returns(self):
        self.historical_data['Close Price'] = pd.to_numeric(self.historical_data['Close Price'], errors='coerce')
        self.historical_data = self.historical_data[~self.historical_data.index.duplicated(keep='first')]
        self.returns = self.historical_data['Close Price'].unstack(level='symbol').pct_change().dropna()

    def select_assets(self):
        latest_market_caps = self.historical_data['market_cap'].unstack(level='symbol').iloc[-1]
        top_symbols = latest_market_caps.nlargest(self.num_assets + 1).index  

        if self.risk_free_symbol in top_symbols:
            self.selected_symbols = top_symbols[:self.num_assets]  # Ajoute le risk-free
            self.include_risk_free = True
        else:
            self.selected_symbols = top_symbols[:self.num_assets]  # Exclu le risk-free 
            self.include_risk_free = False

    def optimize_portfolio(self):
        optimized_weights = pd.DataFrame(index=self.returns.index, columns=self.selected_symbols)

        for date, return_slice in self.returns.iterrows():
            cov_matrix = self.returns.loc[:date, self.selected_symbols].cov()
            avg_returns = self.returns.loc[:date, self.selected_symbols].mean()

            if self.include_risk_free:
                risk_free_rate = self.returns.loc[date, self.risk_free_symbol]
            else:
                risk_free_rate = 0  

            def neg_sharpe_ratio(weights, avg_returns, cov_matrix, risk_free_rate):
                port_return = np.dot(weights, avg_returns) - risk_free_rate
                port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                sharpe_ratio = port_return / port_volatility
                return -sharpe_ratio  

            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            bounds = tuple((0, 1) for _ in range(self.num_assets))

            init_guess = [1./self.num_assets] * self.num_assets
            opt_results = minimize(neg_sharpe_ratio, init_guess, args=(avg_returns, cov_matrix, risk_free_rate), method='SLSQP', bounds=bounds, constraints=constraints)
            optimized_weights.loc[date] = opt_results.x

        self.optimized_weights_over_time = optimized_weights

    def run_strategy(self):
        self.calculate_returns()
        self.select_assets()
        self.optimize_portfolio()
        return self.selected_symbols, self.optimized_weights_over_time


class StrategyEqualWeighting:
    """
   Stratégie de pondération égale. Chaque actif sélectionné dans le portefeuille reçoit un poids égal. 
   Cette stratégie est simple et ne prend pas en compte la volatilité ou le rendement des actifs.

   Attributes:
       historical_data (pd.DataFrame): Données historiques des actifs incluant les prix de clôture.
       num_assets (int): Nombre d'actifs à sélectionner pour la stratégie.

   Methods:
       select_assets(): Sélectionne les actifs basés sur la capitalisation boursière.
       calculate_weights(): Calcule les poids égaux pour tous les actifs sélectionnés.
       run_strategy(): Exécute l'ensemble de la stratégie en appelant les méthodes nécessaires et renvoie les symboles sélectionnés et les poids égaux.
   """
    def __init__(self, historical_data, num_assets=15):
        self.historical_data = historical_data
        self.num_assets = num_assets

    def select_assets(self):
        """
        Sélectionne les actifs en fonction de la capitalisation boursière.
        Assurez-vous que l'index est unique pour éviter les erreurs lors du unstacking.
        """
        if not self.historical_data.index.is_unique:
            self.historical_data = self.historical_data[~self.historical_data.index.duplicated(keep='first')]

        # Sélectionner les actifs
        latest_market_caps = self.historical_data['market_cap'].unstack(level='symbol').iloc[-1]
        self.selected_symbols = latest_market_caps.nlargest(self.num_assets).index

    def calculate_weights(self):
        """Calcule des poids égaux pour les actifs sélectionnés."""
        equal_weight = 1.0 / self.num_assets
        self.weights = pd.DataFrame(equal_weight, index=self.selected_symbols, columns=['Poids'])

    def run_strategy(self):
        """Exécute la stratégie complète."""
        self.select_assets()
        self.calculate_weights()
        return self.selected_symbols, self.weights