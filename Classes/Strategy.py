from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from typing import List, Dict

class Strategy(ABC):
    """
    Classe abstraite pour définir une stratégie d'investissement.
    """
    def __init__(self) -> None:
        """
        Initialise la stratégie avec le nom de la classe fille.
        """
        self.name: str = self.__class__.__name__

    @abstractmethod
    def get_position(self, historical_data: pd.DataFrame, current_position: pd.Series) -> pd.Series:
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
    def __init__(self, num_clusters: int = 5, max_weight: float = 0.1, min_weight: float = 0.0,
                 min_weight_sector: float = 0.0, max_weight_sector: float = 0.3,
                 risk_free_rate: float = 0.01, total_exposure: float = 1.0) -> None:
        """
        Initialise la stratégie d'optimisation avec des paramètres spécifiques.

        Args:
            num_clusters (int): Nombre de clusters pour le regroupement d'actifs.
            max_weight (float): Poids maximum par actif.
            min_weight (float): Poids minimum par actif.
            min_weight_sector (float): Poids minimum par secteur.
            max_weight_sector (float): Poids maximum par secteur.
            risk_free_rate (float): Taux sans risque.
            total_exposure (float): Exposition totale du portefeuille.
        """
        super().__init__()
        self.num_clusters = num_clusters
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.min_weight_sector = min_weight_sector
        self.max_weight_sector = max_weight_sector
        self.risk_free_rate = risk_free_rate
        self.total_exposure = total_exposure

    def get_position(self, historical_data: pd.DataFrame, current_position: pd.Series) -> pd.Series:
        """
        Détermine la nouvelle position en fonction des données historiques.

        Args:
            historical_data (pd.DataFrame): Les données historiques.
            current_position (pd.Series): La position actuelle.

        Returns:
            pd.Series: La nouvelle position calculée par optimisation.
        """

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

    def create_portfolio_constraints(self) -> List[Dict[str, any]]:
        """
        Crée les contraintes pour l'optimisation du portefeuille.

        Returns:
            List[Dict[str, any]]: Liste de contraintes pour l'optimisation.
        """
        # Créer les contraintes du portefeuille
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - self.total_exposure},
            {'type': 'ineq', 'fun': lambda x: self.max_weight - x},
            {'type': 'ineq', 'fun': lambda x: x - self.min_weight}
        ]
        # Ajoutez ici les contraintes supplémentaires si nécessaire
        return constraints

    @abstractmethod
    def objective_function(self, weights: np.ndarray, expected_returns: pd.Series, cov_matrix: pd.DataFrame) -> float:
        """
        Fonction objectif à minimiser, définie par les sous-classes.

        Args:
            weights (numpy.ndarray): Poids du portefeuille.
            expected_returns (pd.Series): Rendements attendus des actifs.
            cov_matrix (pd.DataFrame): Matrice de covariance.

        Returns:
            float: Valeur de la fonction objectif.
        """
        pass

class RankedStrategy(Strategy):
    def get_position(self, historical_data: pd.DataFrame, current_position: pd.Series) -> pd.Series:
        """
        Calcule la position en classant les actifs.

        Args:
            historical_data (pd.DataFrame): Les données historiques.
            current_position (pd.Series): La position actuelle.

        Returns:
            pd.Series: La nouvelle position basée sur le classement des actifs.
        """
        ranked_assets = self.rank_assets(historical_data)

        num_assets = ranked_assets.count()
        sum_of_ranks = ranked_assets.sum()
        average = sum_of_ranks / num_assets
        weights = (ranked_assets - average)

        total_abs_ranks = sum(abs(weights))

        normalized_weights = weights / total_abs_ranks

        return normalized_weights

    
    @abstractmethod
    def rank_assets(self, historical_data: pd.DataFrame) -> pd.Series:
        """
        Classe abstraite pour déterminer le classement des actifs.

        Args:
            historical_data (pd.DataFrame): Les données historiques.

        Returns:
            pd.Series: Classement des actifs.
        """
        pass
