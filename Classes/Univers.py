import requests
import pandas as pd
from typing import List, Dict, Union
from Classes.DataBase import DataBase


class Univers:
    """
    La classe Univers interagit avec l'API CoinGecko pour récupérer et structurer des données 
    de marché sur différentes catégories de cryptomonnaies.
    """

    def __init__(self, verbose: bool = False) -> None:
        """
        Initialise la classe Univers avec les catégories d'actifs, la période temporelle, et d'autres options.

        Args:
            verbose (bool): Indique si les messages de débogage doivent être affichés. Par défaut False.
        """

        self.verbose = verbose

        self.db = DataBase(self.verbose)
        

    def get_crypto_symbols(self, categories: Union[List[str], str], nb_actif: int = 10, format: str = "list") -> Union[List[str], Dict[str, List[str]]]:

        categories = categories if isinstance(categories, list) else [categories]

        coingecko_markets_url = 'https://api.coingecko.com/api/v3/coins/markets'
        
        data_merged = pd.DataFrame()
        
        if self.verbose:
            print(f"Récupération des symboles pour les catégories : {categories}")

        for category in categories:
            params = {
                'vs_currency': 'usd',
                'category': category,
                'order': 'market_cap_desc',
                'per_page': nb_actif,
                'page': 1,
                'sparkline': False,
                'price_change_percentage': '24h',
                'locale': 'en',
                'precision': 3
            }
            
            response = requests.get(coingecko_markets_url, params=params)
            data_json = response.json()
            
            if isinstance(data_json, list) and len(data_json) > 0:
                data = pd.DataFrame(data_json)
                data['symbol'] = data['symbol'].str.upper() + 'USDT'
                data['category'] = category
                data = data[['id', 'symbol', 'name', 'current_price', 'market_cap', 'market_cap_rank', 'category']]
                data_merged = pd.concat([data_merged, data], ignore_index=True)
            else:
                print(f"Erreur ou aucune donnée pour la catégorie {category}: {data_json}")
            #time.sleep(50)  # Pause facultative pour éviter de surcharger l'API
            
        if format == "list":
            return list(data_merged['symbol'].unique())

        if format == "dict":
            category_dict = {}
            for index, row in data_merged.iterrows():
                category = row['category']
                symbol = row['symbol']
                
                if category in category_dict:
                    category_dict[category].append(symbol)
                else:
                    category_dict[category] = [symbol]

            return category_dict
        
    def get_crypto_prices(self, symbols: List[str], start_date: str, end_date: str) -> pd.DataFrame:

        self.db.update_database(symbols, start_date, end_date, 'binance')
        
        return self.db.get_data(symbols, start_date, end_date)
    
    def get_equity_prices(self, symbols: List[str], start_date: str, end_date: str) -> pd.DataFrame:

        self.db.update_database(symbols, start_date, end_date, 'yfinance')

        return self.db.get_data(symbols, start_date, end_date)