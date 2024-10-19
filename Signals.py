import requests
import pandas as pd
import time
from pandas.tseries.offsets import DateOffset

from DataBase import DataBase


class Signals:
    """
    Classe pour analyser les signaux de trading à partir d'un fichier CSV contenant des données de crypto-monnaies.
    """

    def __init__(self, file_path, alpha=1.0, verbose=False):
        """
        Initialise la classe avec le chemin du fichier et les paramètres de configuration.

        :param file_path: Chemin vers le fichier CSV contenant les données.
        :param alpha: Coefficient pour le calcul des pondérations.
        :param verbose: Booléen pour activer les messages d'information.
        """
        self.file_path = file_path
        self.alpha = alpha
        self.verbose = verbose
        self.database = self.load_data()
        self.monthly_returns = self.calculate_monthly_returns()

    def load_data(self):
        """
        Charge le fichier CSV et prétraité les données.

        :return: DataFrame contenant les données prétraitées.
        """
        # Charger le CSV
        database = pd.read_csv(self.file_path)
        database = database[["Date", "ID", "Close"]]

        # Convertir la colonne 'Date' en format datetime
        database['Date'] = pd.to_datetime(database['Date'], format='%Y-%m-%d')
        
        return database

    def Value_strat(self): # Return sur 1 mois
        """
        Calcule les retours mensuels et les pondérations.

        :return: DataFrame contenant les retours mensuels, les rangs et les pondérations.
        """
        # Obtenir la date la plus récente dans la colonne 'Date'
        most_recent_date = self.database['Date'].max()

        # Calculer un mois en arrière
        one_month_ago = most_recent_date - DateOffset(months=1)

        # Filtrer les données pour ne garder que celles entre un mois en arrière et la date la plus récente
        filtered_data = self.database[(self.database['Date'] > one_month_ago) & (self.database['Date'] <= most_recent_date)]

        # Calculer les returns (pourcentage de changement quotidien)
        filtered_data['returns'] = filtered_data['Close'].pct_change()

        # Extraire l'année et le mois de la colonne 'Date'
        filtered_data['Month'] = filtered_data['Date'].dt.to_period('M')

        # Ne garder que les données du mois le plus récent
        latest_month = filtered_data['Month'].max()
        filtered_data = filtered_data[filtered_data['Month'] == latest_month]

        # Grouper par 'ID' (actif) et calculer la moyenne des returns
        monthly_returns = filtered_data.groupby(['ID', 'Month'])['returns'].mean().reset_index().sort_values(by='returns', ascending=False).reset_index(drop=True)

        # Calculer le rank des returns (du plus grand au plus petit)
        monthly_returns['Rank'] = monthly_returns['returns'].rank(ascending=False)

        # Calculer la somme des ranks
        sum_ranks = monthly_returns['Rank'].sum()

        # Calculer le nombre d'actifs uniques
        num_assets = monthly_returns['ID'].nunique()

        # Calculer la pondération
        monthly_returns['Weight'] = self.alpha * (monthly_returns['Rank'] - (sum_ranks / num_assets))

        return monthly_returns

    def display_results(self):
        """
        Affiche les résultats des retours mensuels, rangs et pondérations.
        """
        print(self.monthly_returns[['ID', 'Rank', 'Weight']])



