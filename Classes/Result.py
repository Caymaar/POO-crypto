import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
import numpy as np
import matplotlib.dates as mdates

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class Result:
    """
    Classe pour stocker les résultats du backtest et comparer les stratégies avec des subplots soignés.
    """

    def __init__(self, performance: pd.Series, weight: pd.DataFrame, total_transactions_cost: float, name: str = None):
        self.performance = performance
        self.weights = weight
        self.total_transactions_cost = total_transactions_cost
        self.name = name


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
    
    def get_metrics(self) -> None:
        """
        Calcule les métriques de performance du portefeuille.
        """
        return {
            'Performance': f"{self.perf(self.performance):.2%}",
            'CAGR': f"{self.cagr(self.performance):.2%}",
            'Volatility': f"{self.volatility(self.performance):.2%}",
            'Max Drawdown': f"{self.max_drawdown(self.performance):.2%}",
            'Sharpe Ratio': f"{self.sharpe_ratio(self.performance):.2f}"
        }

    def show_metrics(self) -> None:
        """
        Affiche les métriques de performance du portefeuille.
        """
        metrics = self.get_metrics()
        print(pd.Series(metrics))

    def calculate_drawdown(self):
        return self.performance / self.performance.cummax() - 1

    def plot_dashboard(self, *other_results: 'Result'):
        """
        Affiche une figure avec :
        - Performance des stratégies (un seul graphique).
        - Drawdowns (graphiques collés verticalement avec une ordonnée commune).
        - Histogrammes des rendements (graphiques collés horizontalement avec une ordonnée commune).
        """
        results = [self] + list(other_results)

        # Préparation des données
        performances = [result.performance for result in results]
        drawdowns = [result.calculate_drawdown() for result in results]
        returns = [result.performance.pct_change().dropna() for result in results]
        names = [result.name for result in results]

        # Création de la figure
        num_results = len(results)
        fig, axs = plt.subplots(3, max(2, num_results), figsize=(15, 10), 
                                gridspec_kw={'hspace': 0.4, 'wspace': 0.4}, 
                                constrained_layout=True)

        # Performance (1er graphique, toutes les stratégies superposées)
        sns.set(style="whitegrid")
        for perf, name in zip(performances, names):
            axs[0, 0].plot(perf.index, perf, label=name, linewidth=2)
        axs[0, 0].set_title("Performance des stratégies", fontsize=14)
        axs[0, 0].set_xlabel("Date")
        axs[0, 0].set_ylabel("Valeur")
        axs[0, 0].legend()
        axs[0, 0].grid(True)

        # Supprimer les autres colonnes de la ligne performance
        for i in range(1, max(2, num_results)):
            fig.delaxes(axs[0, i])

        # Drawdowns (chacun sur un graphique distinct mais même ordonnée)
        drawdown_min = min([dd.min() for dd in drawdowns])
        drawdown_max = max([dd.max() for dd in drawdowns])
        for i, (dd, name) in enumerate(zip(drawdowns, names)):
            sns.lineplot(ax=axs[1, i], x=dd.index, y=dd, label=name, color='red')
            axs[1, i].set_title(f"Drawdown: {name}", fontsize=12)
            axs[1, i].set_xlabel("Date")
            axs[1, i].set_ylabel("Drawdown")
            axs[1, i].set_ylim(drawdown_min, drawdown_max)
            axs[1, i].axhline(0, color='black', linestyle='--', linewidth=1)
            axs[1, i].grid(True)

        # Histogrammes des rendements (chacun sur un graphique distinct mais même ordonnée)
        return_min = min([r.min() for r in returns])
        return_max = max([r.max() for r in returns])
        for i, (r, name) in enumerate(zip(returns, names)):
            sns.histplot(ax=axs[2, i], data=r, kde=True, bins=30, color='blue')
            axs[2, i].set_title(f"Rendements: {name}", fontsize=12)
            axs[2, i].set_xlabel("Rendements")
            axs[2, i].set_ylabel("Fréquence")
            axs[2, i].set_xlim(return_min, return_max)

        # Suppression des axes inutilisés si moins de colonnes
        for i in range(num_results, max(2, num_results)):
            if 1 in axs.shape:
                fig.delaxes(axs[1, i])
            if 2 in axs.shape:
                fig.delaxes(axs[2, i])

        plt.show()

    def compare(self, *other_results):
        results = [self] + list(other_results)

        # Préparation des données]
        performances = [result.performance for result in results]
        metrics = [result.get_metrics() for result in results]
        drawdowns = [result.calculate_drawdown() for result in results]
        returns = [result.performance.pct_change().dropna() for result in results]
        names = [result.name for result in results]

        # Calcul des rendements annuels (EOY Returns)
        eoy_returns = []
        for perf in performances:
            annual_returns = perf.resample('YE').last().pct_change(fill_method=None).dropna()
            eoy_returns.append(annual_returns)

        # Création de la figure avec GridSpec
        num_results = len(results)
        fig = plt.figure(figsize=(12 + len(names)*2, 22))  # Ajustement de la largeur
        gs = fig.add_gridspec(6, max(1, num_results), hspace=0.6, wspace=0.03)  # hspace ajusté, wspace réduit

        # Performance (prend deux lignes)
        ax_perf = fig.add_subplot(gs[0:2, :])
        sns.set(style="whitegrid")
        for perf, name in zip(performances, names):
            ax_perf.plot(perf.index, perf, label=name, linewidth=2)
        ax_perf.set_title("Performance des stratégies", fontsize=16)
        ax_perf.set_ylabel("Valeur")
        ax_perf.legend(loc="upper left", fontsize=10)
        # Ligne de base à 0 en pointillés
        ax_perf.axhline(perf.iloc[0], color='black', linestyle='--', linewidth=1)
        ax_perf.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{(x - perf.iloc[0])/perf.iloc[0]:.0%}'))
        ax_perf.grid(True)

        # Rendements annuels (EOY Returns)
        ax_eoy = fig.add_subplot(gs[2, :])
        bar_width = 0.8 / num_results  # Largeur des barres pour chaque stratégie
        for i, (eoy, name) in enumerate(zip(eoy_returns, names)):
            positions = np.arange(len(eoy)) + i * bar_width
            ax_eoy.bar(positions, eoy.values, width=bar_width, label=name)
        ax_eoy.set_title("Rendement annuel (EOY Returns)", fontsize=14)
        ax_eoy.set_ylabel("Rendement (%)")
        ax_eoy.set_xticks(np.arange(len(eoy_returns[0])) + bar_width * (num_results - 1) / 2)
        ax_eoy.set_xticklabels([date.year for date in eoy_returns[0].index], rotation=45)
        ax_eoy.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
        ax_eoy.legend(loc="upper left", fontsize=10)
        ax_eoy.grid(True)

        # Drawdowns
        drawdown_min = min(dd.min() for dd in drawdowns)
        drawdown_max = max(dd.max() for dd in drawdowns)
        date_min = min(dd.index.min() for dd in drawdowns)
        date_max = max(dd.index.max() for dd in drawdowns)
        for i, (dd, name) in enumerate(zip(drawdowns, names)):
            ax_dd = fig.add_subplot(gs[3, i])
            sns.lineplot(ax=ax_dd, x=dd.index, y=dd, color='red')
            ax_dd.fill_between(dd.index, dd, 0, color='red', alpha=0.3)
            ax_dd.set_title(name, fontsize=14, fontweight='bold')
            if i == 0:
                ax_dd.set_ylabel("Drawdown")
                ax_dd.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
            else:
                ax_dd.set_ylabel("")
                ax_dd.yaxis.set_ticklabels([])  # Supprimer les labels y
            ax_dd.set_ylim(drawdown_min, drawdown_max)
            ax_dd.set_xlim(date_min, date_max)
            ax_dd.axhline(0, color='black', linestyle='--', linewidth=1)
            ax_dd.grid(True)
            # Suppression des labels x
            ax_dd.set_xlabel("")
            # Réduction de la densité des dates
            ax_dd.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=5))
            ax_dd.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))


        # Histogrammes
        ret_min_x = min(r.min() for r in returns)
        ret_max_x = max(r.max() for r in returns)

        # Calculer les limites en y pour les histogrammes
        hist_data = []
        for r in returns:
            hist, _ = np.histogram(r, bins=30, range=(ret_min_x, ret_max_x))
            hist_data.append(hist)

        ret_min_y = 0
        ret_max_y = max(h.max() for h in hist_data)

        for i, (r, name) in enumerate(zip(returns, names)):
            ax_ret = fig.add_subplot(gs[4, i])
            sns.histplot(ax=ax_ret, data=r, kde=True, bins=30, color='blue')
            if i == 0:
                ax_ret.set_ylabel("Rendements")
            else:
                ax_ret.set_ylabel("")
                ax_ret.yaxis.set_ticklabels([])  # Supprimer les labels y
            ax_ret.set_xlim(ret_min_x, ret_max_x)
            ax_ret.set_ylim(ret_min_y, ret_max_y)
            ax_ret.set_xlabel("Rendements (%)")
            ax_ret.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
            ax_ret.grid(True)

        plt.show()

                
        # Création d'un DataFrame pour les métriques
        metrics_df = pd.DataFrame(metrics)
        metrics_df.index = names
        metrics_df.loc[self.name] = metrics[0]

        # Mettre en forme le tableau avec l'objet courant en gras
        print(metrics_df.T.to_markdown())

    def visualize(self):
        self.compare()

    def positions(self):
        """
        Visualise les positions sous forme de graphique en aires empilées interactif avec Plotly.
        
        Args:
            weights (pd.DataFrame): DataFrame contenant les poids des positions, indexé par date.
            title (str): Titre du graphique.
        """
        # Filtrer les colonnes pertinentes (éliminer celles avec uniquement des 0)
        weights_filtered = self.weights.loc[:, (self.weights != 0).any(axis=0)]

        # Créer une figure Plotly
        fig = go.Figure()

        # Ajouter chaque colonne comme une trace dans le graphique en aires empilées
        for column in weights_filtered.columns:
            y_values = weights_filtered[column] * 100  # Conversion en %

            # Créer une série de textes conditionnels pour le hover
            hover_text = y_values.apply(
                lambda x: f"<b>{column}</b><br>Poids: {x:.2f}%" if x != 0 else ""
            )

            fig.add_trace(go.Scatter(
                x=weights_filtered.index,
                y=y_values,
                mode='lines',
                name=column,
                stackgroup='one',  # Utilisation de stackgroup pour empiler les aires
                text=hover_text,    # Assignation du texte conditionnel
                hovertemplate='%{text}<extra></extra>'  # Utilisation du texte conditionnel dans le hover
            ))

        # Personnalisation de la figure
        fig.update_layout(
            title=f"Évolution des positions - {self.name}",
            xaxis_title="Date",
            yaxis_title="Poids (%)",
            yaxis=dict(ticksuffix="%", showgrid=True),  # Affiche l'ordonnée en %
            legend_title="Positions",
            hovermode="x unified",  # Affiche toutes les positions pour une date donnée
            template="plotly_white",
            width=1000,  # Ajustement de la largeur
            height=600   # Ajustement de la hauteur
        )

        # Afficher la figure
        fig.show()
