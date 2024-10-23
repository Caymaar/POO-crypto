from Classes.Univers import Univers
from Classes.StrategyBank import MinVarianceStrategy, MaxSharpeStrategy, EqualRiskContributionStrategy
from Classes.Backtester import Backtester
import pandas as pd
from datetime import datetime, timedelta

# Instancier les stratégies avec les paramètres appropriés
min_variance_strategy = MinVarianceStrategy(
    num_clusters=5,
    max_weight=1,
    min_weight=0.0,
    risk_free_rate=0.01,
    total_exposure=1.0
)

max_sharpe_strategy = MaxSharpeStrategy(
    num_clusters=5,
    max_weight=1,
    min_weight=0.0,
    risk_free_rate=0.01,
    total_exposure=1.0
)

equal_risk_strategy = EqualRiskContributionStrategy(
    lmd_mu=0.25,
    lmd_var=0.1,
    num_clusters=5,
    max_weight=1,
    min_weight=0.0,
    risk_free_rate=0.01,
    total_exposure=1.0
)

# Exemple d'utilisation

start_date = "2023-01-01"
end_date = (datetime.today() - timedelta(days=1)).strftime('%Y-%m-%d')
univers_layer = Univers('layer-1', start_date, end_date, 20, verbose=True)

# Créer une instance du backtester
backtester = Backtester(univers_layer.data)

# Exécuter les backtests
result_min_variance = backtester.run(start_date, end_date, 30, 90, 100, 0.0, strategy=min_variance_strategy)
result_min_variance.show()

result_max_sharpe = backtester.run(start_date, end_date, 30, 90, 100, 0.05, strategy=max_sharpe_strategy)
result_max_sharpe.show()

result_equal_risk = backtester.run(start_date, end_date, 30, 90, 100, 0.0, strategy=equal_risk_strategy)
result_equal_risk.show()