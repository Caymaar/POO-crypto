import pandas as pd
import matplotlib.pyplot as plt

class Result:
    """
    Classe pour stocker les résultats du backtest.
    """

    def __init__(self, performance: pd.Series, weight: pd.DataFrame, total_transactions_cost: float, name: str = None):
        self.performance = performance
        self.weights = weight
        self.total_transactions_cost = total_transactions_cost
        self.name = name

    def show(self):
        self.performance.plot(figsize=(10, 6), title='Performance du portefeuille')

    def periods_freq(self, series: pd.Series) -> int:
        serie_length = len(series)
        num_of_days = (series.index[-1] - series.index[0]).days
        ratio = serie_length / num_of_days
        
        if abs(ratio - 1) < abs(ratio - (252 / 365)):
            return 365
        else:
            return 252
        
    def volatility(self, prices: pd.Series) -> float:
        """
        Calcule la volatilité annualisée à partir d'une série de prix.
        """
        returns = prices.pct_change().dropna()
        return returns.std() * (self.periods_freq(prices) ** 0.5)
    
    def perf(self, prices: pd.Series) -> float:
        """
        Calcule la performance totale à partir d'une série de prix.
        """
        return prices[-1] / prices[0] - 1
    
    def cagr(self, prices: pd.Series) -> float:
        """
        Calcule le taux de croissance annuel composé (CAGR) à partir d'une série de prix.
        """
        total_periods = len(prices)
        total_years = total_periods / self.periods_freq(prices)
        return (self.perf(prices) + 1) ** (1 / total_years) - 1
    
    def max_drawdown(self, prices: pd.Series) -> float:
        """
        Calcule le maximum drawdown à partir d'une série de prix.
        """
        drawdown = (prices / prices.cummax() - 1)
        return drawdown.min()

    def sharpe_ratio(self, prices: pd.Series, risk_free_rate: float = 0.0) -> float:
        """
        Calcule le ratio de Sharpe à partir d'une série de prix.
        """
        returns = prices.pct_change().dropna()
        excess_returns = returns - risk_free_rate / self.periods_freq(prices)
        return excess_returns.mean() / excess_returns.std() * (self.periods_freq(prices) ** 0.5)
    
    def calculate_metrics(self) -> None:
        """
        Calcule les métriques de performance du portefeuille.
        """
        self.metrics = {
            'Performance': self.perf(self.performance),
            'CAGR': self.cagr(self.performance),
            'Volatility': self.volatility(self.performance),
            'Max Drawdown': self.max_drawdown(self.performance),
            'Sharpe Ratio': self.sharpe_ratio(self.performance)
        }

    def show_metrics(self) -> None:
        """
        Affiche les métriques de performance du portefeuille.
        """
        self.calculate_metrics()
        print(pd.Series(self.metrics))

    def compare_with(self, *other_results: 'Result') -> None:
            """
            Compare les performances avec d'autres objets Result et trace les performances.

            Args:
                other_results (Result): Un ou plusieurs objets Result à comparer.
            """
            plt.figure(figsize=(12, 8))
            
            # Performance de l'objet courant
            plt.plot(self.performance.index, self.performance, label=self.name, linewidth=2, color='blue')

            # Pointillés pour la valeur initiale
            plt.axhline(self.performance.iloc[0], color='blue', linestyle='--')

            # Liste pour stocker les métriques
            metrics_list = []

            for idx, other in enumerate(other_results):
                # Tracer la performance des autres objets
                plt.plot(other.performance.index, other.performance, label=other.name, linewidth=2)
                other.calculate_metrics()
                metrics_list.append(other.metrics)

            plt.title('Comparaison des performances')
            plt.xlabel('Date')
            plt.ylabel('Performance')
            plt.legend()
            plt.grid()
            plt.show()

            # Création d'un DataFrame pour les métriques
            metrics_df = pd.DataFrame(metrics_list)
            metrics_df.index = [other.name for other in other_results]

            # Ajouter les métriques de l'objet courant
            self.calculate_metrics()
            metrics_df.loc[self.name] = self.metrics

            # Mettre en forme le tableau avec l'objet courant en gras
            print(metrics_df.T.to_markdown())