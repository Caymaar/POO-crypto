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

class MomentumStrategy(RankedStrategy):
    pass

class ValueStrategy(RankedStrategy):
    pass

class MivVolStrategy(RankedStrategy):
    pass