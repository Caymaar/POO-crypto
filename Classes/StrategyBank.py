import numpy as np
import pandas as pd
from Classes.Strategy import Strategy, RankedStrategy, OptimizationStrategy, filter_with_signals

class EqualWeightStrategy(Strategy):
    @filter_with_signals
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

        if num_assets == 0:
            return pd.Series()
        
        weights = pd.Series(1 / num_assets, index=historical_data.columns)
        return weights
    
class RandomStrategy(Strategy):
    @filter_with_signals
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
    def __init__(self, lmd_mu: float = 0.0, lmd_var: float = 0.0, **kwargs) -> None:
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

        def _minimize_risk_concentration(weights, cov_matrix):
            """
            Fonction objectif pour minimiser la concentration de risque dans un portefeuille.

            Parameters:
            weights (numpy.array): Vecteur des poids des actifs dans le portefeuille.
            covariance_matrix (numpy.array): Matrice de covariance des rendements des actifs.

            Returns:
            float: Valeur de la fonction objectif.
            """
            N = len(weights)
            risk_contributions = np.dot(cov_matrix, weights)
            objective_value = 0
            for i in range(N):
                for j in range(N):
                    objective_value += (weights[i] * risk_contributions[i] - weights[j] * risk_contributions[j])
            return objective_value ** 2

        risk_contributions = ((cov_matrix @ weights) * weights) / np.sqrt((weights.T @ cov_matrix @ weights))
        risk_objective = np.sum((risk_contributions - 1 / len(weights))**2)
        # risk_objective = _minimize_risk_concentration(weights, cov_matrix) # ou "np.sum((risk_contributions - 1 / num_assets)**2)" Les deux fonctionnent, mais différement, j'ai du mal à cerner si l'une est meilleure que l'autre.
        return_value_objective = -self.lmd_mu * weights.T @ expected_returns
        variance_objective = self.lmd_var * weights.T @ cov_matrix @ weights
        return risk_objective #+ return_value_objective + variance_objective
    
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

class CrossingMovingAverage(EqualWeightStrategy):
    def __init__(self, fast_period: int=30, slow_period: int=90) -> None:
        """
        Initialise la stratégie de croisement de moyennes mobiles.
        
        Args:
            fast_period (int): Période de la moyenne mobile rapide.
            slow_period (int): Période de la moyenne mobile lente.
        """
        super().__init__()
        self.fast_period = fast_period
        self.slow_period = slow_period

        self.name += f"\nFast: {fast_period}, Slow: {slow_period}"

    def signals(self, data: pd.DataFrame) -> list:
        """
        Retourne les actifs avec un croisement de moyennes mobiles.

        Args:
          x  data (pd.DataFrame): Les données historiques.

        Returns:
            list: Liste des actifs avec un croisement de moyennes mobiles.
        """
        # Calcul des moyennes mobiles
        fast_ma = data.rolling(window=self.fast_period).mean()
        slow_ma = data.rolling(window=self.slow_period).mean()

        # Vérification des croisements : rapide > lente aujourd'hui et rapide <= lente hier
        crossover = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))

        # Sélectionne les colonnes avec un croisement au dernier jour
        last_day_crossover = crossover.iloc[-1]
        return last_day_crossover[last_day_crossover].index.tolist()     
        
    
    