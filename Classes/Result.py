import pandas as pd

class Result:
    """
    Classe pour stocker les résultats du backtest.
    """

    def __init__(self, performance, weight, total_transactions_cost):
        self.performance = performance
        self.weights = weight
        self.total_transactions_cost = total_transactions_cost

    def show(self):
        self.performance.plot(figsize=(10, 6), title='Performance du portefeuille')
