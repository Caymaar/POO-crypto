import requests
import pandas as pd
import time

from binance.client import Client as BinanceClient


class Univers:
    """
        Univers est une classe conçue pour interagir avec l'API CoinGecko afin de récupérer 
        et de structurer des données de marché pour diverses catégories de cryptomonnaies. 

        Attributes:
            symbol : Le ticker attendu
            start_date : une date de début pour l'historique des prix
            end_date : une date de fin pour l'historique des prix
            nb_actif : un nombre d'actif par catégories
            categories : Les différentes catégories d'actifs

        Methods:
                Get_data : Récupére les datas de coingecko par catégories, on va ensuite récupérer les symboles unique pour faire un appel via notre Database
                et récupérer l'historique des prix.
                On sauvegarde aussi un dictionaire des compositions des indices qui nous permettra de pouvoir comparer les performances des différentes catégories d'indices.

        Raises:
            ValueError: Si il n'y a pas de catégories, on oblige l'utilisateur à en choisir une.
    """

    def __init__(self, symbol : str, start_date : str, end_date : str , nb_actif, categories=None):
        self.nb_actif = nb_actif
        self.start_date = start_date
        self.end_date = end_date

        # On force des catégories à mettre
        if not categories:
            raise ValueError("Please select at least one category")
        else:
            self.categories = categories if isinstance(categories, list) else [categories]
        
            
        self.base_url = 'https://api.coingecko.com/api/v3/coins/markets'
        # Initialisation du dataframe final
        self.data_merged = pd.DataFrame()
    
    # Récupérer les symboles des catégories de coingecko
    def get_data(self):          
                                            
        self.base_url = 'https://api.coingecko.com/api/v3/coins/markets'
        
        for cate in self.categories:
            params = {
                        'vs_currency': 'usd',
                        'category': cate,
                        'order': 'market_cap_desc',
                        'per_page': self.nb_actif,
                        'page': 1,
                        'sparkline': False,
                        'price_change_percentage': '24h',
                        'locale': 'en',
                        'precision': 3
                    }
            
            response = requests.get(self.base_url, params=params)
            data_json = response.json()
            
            if isinstance(data_json, list) and len(data_json) > 0:
                
                data = pd.DataFrame(data_json)
                data['symbol'] = data['symbol'].str.upper() + 'USDT'
                data['category'] = cate
                data = data[['id', 'symbol', 'name', 'current_price', 'market_cap', 'market_cap_rank', 'category']]
                self.data_merged = pd.concat([self.data_merged, data], ignore_index=True)
            else:
                print(f"Erreur ou aucune donnée pour la catégorie {cate}: {data_json}")
            time.sleep(50)
            
        # Récupérer les symboles qui composent l'univers
        self.all_symbols = list(self.data_merged['symbol'].unique())

        # Créer un dictionnaire de chaque actif qui compose l'indice pour pouvoir les comparer entre eux
        self.category_dict = {}

        for index, row in self.data_merged.iterrows():
            category = row['category']
            symbol = row['symbol']
            
            if category in self.category_dict:
                self.category_dict[category].append(symbol)
            else:
                self.category_dict[category] = [symbol]


        return self.all_symbols
            


