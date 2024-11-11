import numpy as np
import pandas as pd
from Classes.Strategy import Strategy, RankedStrategy, OptimizationStrategy

class MinVarianceStrategy(OptimizationStrategy):
    def objective_function(self, weights: np.ndarray, expected_returns: pd.Series, cov_matrix: pd.DataFrame) -> float:
        """
        Fonction objectif pour minimiser la variance du portefeuille.

        Args:
            weights (np.ndarray): Poids du portefeuille.
            expected_returns (pd.Series): Rendements attendus des actifs.
            cov_matrix (pd.DataFrame): Matrice de covariance.

        Returns:
            float: Variance du portefeuille.
        """
        # Fonction objectif : variance du portefeuille
        portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        return portfolio_variance
    
class MaxSharpeStrategy(OptimizationStrategy):
    def objective_function(self, weights: np.ndarray, expected_returns: pd.Series, cov_matrix: pd.DataFrame) -> float:
        """
        Fonction objectif pour maximiser le ratio de Sharpe du portefeuille.

        Args:
            weights (np.ndarray): Poids du portefeuille.
            expected_returns (pd.Series): Rendements attendus des actifs.
            cov_matrix (pd.DataFrame): Matrice de covariance.

        Returns:
            float: Négatif du ratio de Sharpe (pour minimisation).
        """
        portfolio_return = np.dot(weights, expected_returns) * 252  # Annualisé
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)) * 252)
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
        # Nous voulons maximiser le ratio de Sharpe, donc nous minimisons son opposé
        return -sharpe_ratio

class EqualRiskContributionStrategy(OptimizationStrategy):
    def __init__(self, lmd_mu: float = 0.25, lmd_var: float = 0.1, **kwargs) -> None:
        """
        Initialise la stratégie Equal Risk Contribution.

        Args:
            lmd_mu (float): Paramètre de pondération pour le retour.
            lmd_var (float): Paramètre de pondération pour la variance.
        """
        super().__init__(**kwargs)
        self.lmd_mu = lmd_mu
        self.lmd_var = lmd_var

    def objective_function(self, weights: np.ndarray, expected_returns: pd.Series, cov_matrix: pd.DataFrame) -> float:
        """
        Fonction objectif pour la stratégie Equal Risk Contribution.

        Args:
            weights (np.ndarray): Poids du portefeuille.
            expected_returns (pd.Series): Rendements attendus des actifs.
            cov_matrix (pd.DataFrame): Matrice de covariance.

        Returns:
            float: Valeur de la fonction objectif ERC.
        """
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
    def get_position(self, historical_data: pd.DataFrame, current_position: pd.Series) -> pd.Series:
        """
        Retourne une position avec des poids égaux pour chaque actif.

        Args:
            historical_data (pd.DataFrame): Les données historiques.
            current_position (pd.Series): La position actuelle.

        Returns:
            pd.Series: Nouvelle position avec des poids égaux.
        """
        num_assets = historical_data.shape[1]
        weights = pd.Series(1 / num_assets, index=historical_data.columns)
        return weights
    
class RandomStrategy(Strategy):
    def get_position(self, historical_data: pd.DataFrame, current_position: pd.Series) -> pd.Series:
        """
        Retourne une position avec des poids aléatoires normalisés.

        Args:
            historical_data (pd.DataFrame): Les données historiques.
            current_position (pd.Series): La position actuelle.

        Returns:
            pd.Series: Nouvelle position avec des poids aléatoires.
        """
        weights = np.random.rand(len(historical_data.columns))
        weights /= weights.sum()
        return pd.Series(weights, index=historical_data.columns)

class ValueStrategy(RankedStrategy):
    def rank_assets(self, historical_data: pd.DataFrame) -> pd.Series:
        """
        Classe les actifs en fonction de leur valeur (ratio prix actuel / prix il y a un an).

        Args:
            historical_data (pd.DataFrame): Les données historiques.

        Returns:
            pd.Series: Classement des actifs, où les actifs moins chers ont un rang plus élevé.
        """
        last_prices = historical_data.iloc[-1]  # Dernier prix de chaque actif
        prices_one_year_ago = historical_data.iloc[0]  # Prix d'il y a un an
        coef_asset = last_prices / prices_one_year_ago
        coef_asset = coef_asset.dropna()
        return coef_asset.rank(ascending=False, method='first').sort_values()

class MomentumStrategy(RankedStrategy):
    def rank_assets(self, historical_data: pd.DataFrame) -> pd.Series:
        """
        Classe les actifs en fonction de leur performance passée.

        Args:
            historical_data (pd.DataFrame): Les données historiques.

        Returns:
            pd.Series: Classement des actifs, où les actifs performants ont un rang plus élevé.
        """
        returns = historical_data.pct_change().dropna()
        len_window = len(returns)
        delta = int(np.ceil(len_window*(1/12)))
        total_returns = returns.rolling(window=len_window - delta).apply(lambda x: (1 + x).prod() - 1)
        latest_returns = total_returns.iloc[-delta]
        latest_returns = latest_returns.dropna()
        return latest_returns.rank(ascending=True, method='first').sort_values()

class MinVolStrategy(RankedStrategy):
    def rank_assets(self, historical_data: pd.DataFrame) -> pd.Series:
        """
        Classe les actifs en fonction de leur volatilité, où les actifs moins volatils sont favorisés.

        Args:
            historical_data (pd.DataFrame): Les données historiques.

        Returns:
            pd.Series: Classement des actifs en fonction de la volatilité.
        """
        returns = historical_data.pct_change().dropna()
        volatility = returns.std()
        volatility.dropna()
        return volatility.rank(ascending=False, method='first').sort_values()
        
        
    
    