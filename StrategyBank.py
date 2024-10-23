import numpy as np
import pandas as pd
from Classes.Strategy import Strategy, RankedStrategy, OptimizationStrategy

class MinVarianceStrategy(OptimizationStrategy):
    def objective_function(self, weights, expected_returns, cov_matrix):
        # Fonction objectif : variance du portefeuille
        portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        return portfolio_variance
    
class MaxSharpeStrategy(OptimizationStrategy):
    def objective_function(self, weights, expected_returns, cov_matrix):
        portfolio_return = np.dot(weights, expected_returns) * 252  # Annualisé
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)) * 252)
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
        # Nous voulons maximiser le ratio de Sharpe, donc nous minimisons son opposé
        return -sharpe_ratio
    
class EqualRiskContributionStrategy(OptimizationStrategy):
    def objective_function(self, weights, expected_returns, cov_matrix):
        # Calcul de la contribution au risque de chaque actif
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        marginal_risk_contribution = np.dot(cov_matrix, weights) / portfolio_volatility
        risk_contributions = weights * marginal_risk_contribution

        # Calcul de l'objectif ERC
        target_risk = portfolio_volatility / len(weights)
        return np.sum((risk_contributions - target_risk) ** 2)
    
class EqualRiskContributionStrategy(OptimizationStrategy):
    def __init__(self, lmd_mu=0.25, lmd_var=0.1, **kwargs):
        super().__init__(**kwargs)
        self.lmd_mu = lmd_mu
        self.lmd_var = lmd_var

    def objective_function(self, weights, expected_returns, cov_matrix):
        N = len(weights)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        marginal_risk_contribution = np.dot(cov_matrix, weights) / portfolio_volatility
        risk_contributions = weights * marginal_risk_contribution

        # Calcul de l'objectif ERC avec les paramètres lmd_mu et lmd_var
        risk_objective = np.sum((risk_contributions - portfolio_volatility / N) ** 2)
        return_value_objective = -self.lmd_mu * np.dot(weights, expected_returns)
        variance_objective = self.lmd_var * portfolio_volatility ** 2
        return risk_objective + return_value_objective + variance_objective
    
class EqualWeightStrategy(Strategy):
    def get_position(self, historical_data, current_position):
        num_assets = historical_data.shape[1]
        weights = pd.Series(1 / num_assets, index=historical_data.columns)
        return weights
    
class RandomStrategy(Strategy):
    def get_position(self, historical_data, current_position):
        weights = np.random.rand(len(historical_data.columns))
        weights /= weights.sum()
        return pd.Series(weights, index=historical_data.columns)

class ValueStrategy:
    def __init__(self, historical_data, lookback_period=90, alpha = 1):
        self.lookback_period = lookback_period
        self.historical_data = pd.read_csv(historical_data)
        self.lookback_period = lookback_period
        self.historical_data['Date']  = pd.to_datetime(self.historical_data['Date'])
        self.historical_data.set_index('Date', inplace=True)
        self.alpha = alpha
    
    def rank_assets(self):
        """
        on classe les actifs puis on leur ajoute un parametre alpha
        qui determine le coefficient de risque du portefeuill
        """
        last_prices = self.historical_data.iloc[-1]  # Dernier prix de chaque actif
        prices_one_year_ago = self.historical_data.iloc[-self.lookback_period]  # Prix d'il y a un an

        # Calcul du coefficient
        coef_asset = last_prices / prices_one_year_ago

        # On peut éventuellement normaliser ou utiliser ce coefficient dans d'autres calculs
        coef_asset = coef_asset.fillna(0)  # Remplacer les NaN par 0 si nécessaire

        # Afficher ou retourner le coefficient

        # latest_returns = total_returns.iloc[-1]
        ranked_assets = coef_asset.rank(ascending=False, method='first').sort_values()
        num_assets = ranked_assets.count()
        sum_of_ranks = ranked_assets.sum()
        average = sum_of_ranks / num_assets
        weights = (ranked_assets - average) * self.alpha
        
        # Normalisation pour que les poids soient entre -1 et 1
        max_weight = weights.max()
        min_weight = weights.min()
        
        # Normaliser les poids
        weights = (weights - average) / (max_weight - min_weight)  # Ajustement centré


        # Ajustement final pour garantir que la somme des poids soit 0
        weights = weights - weights.mean()  # Centre autour de 0

        # S'assurer que les poids sont dans l'intervalle [-1, 1]
        weights = weights.clip(lower=-1, upper=1)
        
        # Ajustement final pour garantir que la somme des poids soit 1
        if weights.sum() != 0:
            weights /= weights.sum()  # Normaliser pour que la somme soit 1

        # Remise à l'échelle pour que la somme soit 1
        weights = (weights + 1) / 2  # Passer de [-1, 1] à [0, 1]
        weights /= weights.sum()  # Normaliser pour que la somme soit 1
        
        
        return weights

class MomentumStrategy:
    
    def __init__(self, historical_data, lookback_period=365):
        self.lookback_period = lookback_period
        self.historical_data = pd.read_csv(historical_data)
        self.lookback_period = lookback_period
        self.historical_data['Date']  = pd.to_datetime(self.historical_data['Date'])
        self.historical_data.set_index('Date', inplace=True)
    
    def rank_assets(self):
        """
        On alloue plus de poids au actifs qui ont surperformé
        """
        returns = self.historical_data.pct_change().dropna()
        total_returns = returns.rolling(window=self.lookback_period - 30).apply(lambda x: (1 + x).prod() - 1)
        latest_returns = total_returns.iloc[-1]
        ranked_assets = latest_returns.rank(ascending=False, method='first').sort_values()
        print(ranked_assets)
        # Nombre total d'actifs
        num_assets = ranked_assets.count()

        # Déterminer la moitié supérieure et inférieure
        split_index = num_assets // 2  # Indice pour séparer les longs et shorts
        # Sélectionner les actifs longs et shorts
        long_assets = ranked_assets.index[:split_index]  # Actifs de la moitié supérieure pour long
        short_assets = ranked_assets.index[split_index:]  # Actifs de la moitié inférieure pour short
        
        # Initialiser les poids
        weights = pd.Series(0, index=latest_returns.index)
        
        # Affecter des poids longs (positifs) à la moitié supérieure
        weights[long_assets] = (1 / split_index) / 2  # Distribution égale des poids longs

        # Affecter des poids shorts (négatifs) à la moitié inférieure
        weights[short_assets] = (-1 / split_index) / 2 # Distribution égale des poids shorts
        
        return weights

class MinVolStrategy(RankedStrategy):
    pass

