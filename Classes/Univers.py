import requests
import pandas as pd
import time

from Classes.DataBase import DataBase


class Univers:
    """
    La classe Univers interagit avec l'API CoinGecko pour récupérer et structurer des données 
    de marché sur différentes catégories de cryptomonnaies. Elle permet de récupérer les symboles, 
    leur historique de prix et de comparer les catégories d'actifs.

    Attributes:
        nb_actif (int): Nombre d'actifs à récupérer par catégorie.
        start_date (str): Date de début pour les données historiques au format 'YYYY-MM-DD'.
        end_date (str): Date de fin pour les données historiques au format 'YYYY-MM-DD'.
        categories (list): Liste des catégories d'actifs à analyser.
        verbose (bool): Indicateur pour afficher ou non les messages de débogage.
        data_merged (pd.DataFrame): DataFrame contenant les données récupérées.
        all_symbols (list): Liste de tous les symboles récupérés.
        category_dict (dict): Dictionnaire contenant les symboles classés par catégorie.
        notlisted_symbols (list): Liste des symboles non disponibles sur Binance.
        data (pd.DataFrame): Données historiques filtrées récupérées de la base de données.
    
    Methods:
        get_symbols(): Récupère les symboles pour chaque catégorie spécifiée depuis l'API CoinGecko.
        """

    def __init__(self, categories, start_date, end_date, nb_actif=10, verbose=False):
        """
        Initialise la classe Univers avec les catégories d'actifs, la période temporelle, et d'autres options.

        Args:
            categories (list or str): Liste des catégories ou une seule catégorie d'actifs à analyser.
            start_date (str): Date de début pour les données au format 'YYYY-MM-DD'.
            end_date (str): Date de fin pour les données au format 'YYYY-MM-DD'.
            nb_actif (int): Nombre d'actifs à récupérer par catégorie. Par défaut 10.
            verbose (bool): Indique si les messages de débogage doivent être affichés. Par défaut False.
        """
        self.nb_actif = nb_actif
        self.start_date = start_date
        self.end_date = end_date
        self.verbose = verbose
        self.categories = categories if isinstance(categories, list) else [categories]
        
        # Initialisation du dataframe final pour stocker les données
        self.data_merged = pd.DataFrame()

        # Récupérer les symboles pour les catégories spécifiées
        self.get_symbols()

        print("Univers initialisé avec succès.")

        # Initialiser la base de données 
        self.db = DataBase(self.all_symbols, self.start_date, self.end_date, verbose=verbose)

        # Mettre à jour la base de données et obtenir la liste des symboles non disponibles
        self.notlisted_symbols = self.db.update_database()

        # Récupérer les données historiques pour les symboles disponibles
        self.data = self.db.from_ohlcv_to_close(self.db.get_data(self.all_symbols, self.start_date, self.end_date))

        # Afficher les symboles non disponibles
        if self.verbose:
            if self.notlisted_symbols:
                print("Les symboles suivants ne sont pas listés sur Binance :", self.notlisted_symbols)
            else:
                print("Tous les symboles ont été récupérés avec succès.")

    def get_symbols(self):
        """
        Récupère les symboles des catégories spécifiées depuis l'API CoinGecko et les structure 
        dans un DataFrame. Ces symboles sont ensuite utilisés pour récupérer les données historiques.

        Raises:
            ValueError: Si aucune donnée n'est trouvée pour une catégorie spécifique.
        """
        self.coingecko_markets_url = 'https://api.coingecko.com/api/v3/coins/markets'
        
        if self.verbose:
            print(f"Récupération des symboles pour les catégories : {self.categories}")

        for category in self.categories:
            params = {
                'vs_currency': 'usd',
                'category': category,
                'order': 'market_cap_desc',
                'per_page': self.nb_actif,
                'page': 1,
                'sparkline': False,
                'price_change_percentage': '24h',
                'locale': 'en',
                'precision': 3
            }
            
            response = requests.get(self.coingecko_markets_url, params=params)
            data_json = response.json()
            
            if isinstance(data_json, list) and len(data_json) > 0:
                data = pd.DataFrame(data_json)
                data['symbol'] = data['symbol'].str.upper() + 'USDT'
                data['category'] = category
                data = data[['id', 'symbol', 'name', 'current_price', 'market_cap', 'market_cap_rank', 'category']]
                self.data_merged = pd.concat([self.data_merged, data], ignore_index=True)
            else:
                print(f"Erreur ou aucune donnée pour la catégorie {category}: {data_json}")
            #time.sleep(50)  # Pause facultative pour éviter de surcharger l'API
            
        # Récupérer la liste des symboles qui composent l'univers
        self.all_symbols = list(self.data_merged['symbol'].unique())

        # Créer un dictionnaire pour stocker les symboles par catégorie
        self.category_dict = {}
        for index, row in self.data_merged.iterrows():
            category = row['category']
            symbol = row['symbol']
            
            if category in self.category_dict:
                self.category_dict[category].append(symbol)
            else:
                self.category_dict[category] = [symbol]