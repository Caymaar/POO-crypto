from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from scipy.optimize import minimize

class Strategy(ABC):
    """
    Classe abstraite pour définir une stratégie d'investissement.
    """

    @abstractmethod
    def get_position(self, historical_data, current_position):
        """
        Méthode obligatoire pour déterminer la position actuelle.

        Args:
            historical_data (pd.DataFrame): Les données historiques.
            current_position (pd.Series): La position actuelle.

        Returns:
            pd.Series: La nouvelle position.
        """
        pass

class OptimizationStrategy(Strategy):
    def __init__(self, num_clusters=5, max_weight=0.1, min_weight=0.0,
                 min_weight_sector=0.0, max_weight_sector=0.3,
                 risk_free_rate=0.01, total_exposure=1.0):
        self.num_clusters = num_clusters
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.min_weight_sector = min_weight_sector
        self.max_weight_sector = max_weight_sector
        self.risk_free_rate = risk_free_rate
        self.total_exposure = total_exposure

    def get_position(self, historical_data, current_position):
        # Calculer les rendements
        returns = historical_data.pct_change().dropna()

        # Vérifier qu'il y a suffisamment de données pour l'optimisation
        if len(returns) < 2:
            # Retourner la position actuelle si pas assez de données
            return current_position

        # Exclure les colonnes avec des valeurs manquantes
        returns = returns.dropna(axis=1, how='any')

        # Vérifier qu'il reste des actifs après suppression
        if returns.empty:
            return current_position

        # Créer les contraintes du portefeuille
        portfolio_constraints = self.create_portfolio_constraints(returns)

        # Calculer la matrice de covariance
        cov_matrix = returns.cov()

        # Calculer les rendements attendus
        expected_returns = returns.mean()

        # Définir les poids minimum et maximum pour chaque action
        bounds = tuple((0, 1) for _ in range(returns.shape[1]))

        # Initialiser les poids de manière égale
        initial_weights = np.array([1 / returns.shape[1]] * returns.shape[1])

        # Effectuer l'optimisation
        result = minimize(
            fun=self.objective_function,
            x0=initial_weights,
            args=(expected_returns, cov_matrix),
            method='SLSQP',
            bounds=bounds,
            constraints=portfolio_constraints
        )

        if result.success:
            # Créer une série de poids avec tous les actifs, en mettant zéro pour les actifs exclus
            weights = pd.Series(0.0, index=historical_data.columns)
            weights.update(pd.Series(result.x, index=returns.columns))
            return weights
        else:
            import warnings
            warnings.warn("L'optimisation n'a pas réussi: " + result.message + ". Utilisation des poids précédents.")
            return current_position

    def create_portfolio_constraints(self, returns):
        # Créer les contraintes du portefeuille
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - self.total_exposure},
            {'type': 'ineq', 'fun': lambda x: self.max_weight - x},
            {'type': 'ineq', 'fun': lambda x: x - self.min_weight}
        ]
        # Ajoutez ici les contraintes supplémentaires si nécessaire
        return constraints

    @abstractmethod
    def objective_function(self, weights, expected_returns, cov_matrix):
        """
        Fonction objectif à minimiser. Doit être implémentée par les sous-classes.

        Args:
            weights (numpy.array): Poids du portefeuille.
            expected_returns (pd.Series): Rendements attendus des actifs.
            cov_matrix (pd.DataFrame): Matrice de covariance.

        Returns:
            float: Valeur de la fonction objectif.
        """
        pass

class RankedStrategy(Strategy):
    def get_position(self, historical_data, current_position):
        pass

    @abstractmethod
    def rank_assets(self, historical_data):
        """
        Classe abstraite pour déterminer le classement des actifs.
        """
        pass
