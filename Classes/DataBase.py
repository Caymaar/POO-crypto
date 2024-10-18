import pandas as pd
from binance.client import Client
from datetime import datetime

class DataBase:
    """
    Classe pour gérer une base de données d'actifs financiers à partir de Binance,
    permettant de récupérer, filtrer et mettre à jour des données historiques.
    """
    def __init__(self, symbols, start_date, end_date, verbose=False):
        """
        Initialise la base de données en définissant les symboles, les dates de début et de fin,
        et la connexion à l'API Binance.

        Args:
            symbols (list): Liste des symboles à suivre.
            start_date (str): Date de début des données au format 'YYYY-MM-DD'.
            end_date (str): Date de fin des données au format 'YYYY-MM-DD'.
            verbose (bool): Active les messages de débogage si True.
        """
        self.api_key = 'AAJWE5TewkT5QCivRev9s5r2MpmZMUFXXGokxJL9mlZkZadRiKCEky0tho7OMGxW'
        self.api_secret = 'TxA2VRCyvVHLkn4DZvucbvCkTpNWYDQJeHVcKCDiJD7G5usNd7CrBNKd8rea1vPP'
        self.client = Client(self.api_key, self.api_secret)
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.verbose = verbose

        if self.verbose:
            print("Initialisation de la base de données...")
        self.database = pd.read_csv('data/database.csv')

    def get_binance_historical_data(self, symbol, start_date, end_date):
        """
        Récupère les données historiques pour un symbole depuis Binance entre deux dates.

        Args:
            symbol (str): Le symbole de l'actif à récupérer.
            start_date (str): Date de début des données au format 'YYYY-MM-DD'.
            end_date (str): Date de fin des données au format 'YYYY-MM-DD'.

        Returns:
            pd.DataFrame: Données historiques sous forme de DataFrame, ou None si une erreur survient.
        """
        try:
            klines = self.client.get_historical_klines(
                symbol,
                Client.KLINE_INTERVAL_1DAY,
                start_date,
                end_date
            )
            df = pd.DataFrame(klines, columns=[
                'Open Time', 'Open', 'High', 'Low', 'Close', 'Volume', 
                'Close Time', 'Quote Asset Volume', 'Number of Trades',
                'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume', 'Ignore'
            ])
            df['Date'] = pd.to_datetime(df['Open Time'], unit='ms').dt.strftime('%Y-%m-%d')
            df['Open'] = df['Open'].astype(float)
            df['High'] = df['High'].astype(float)
            df['Low'] = df['Low'].astype(float)
            df['Close'] = df['Close'].astype(float)
            df['Volume'] = df['Volume'].astype(float)
            df['Adj Close'] = df['Close']
            df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
            df['ID'] = symbol
            return df
        except Exception as e:
            if self.verbose:
                print(f"Erreur lors de la récupération des données pour {symbol}: {e}")
            return None

    def get_symbol_date_range(self, symbol):
        """
        Retourne les dates minimum et maximum pour un symbole donné dans la base de données.

        Args:
            symbol (str): Le symbole à vérifier.

        Returns:
            tuple: Date minimale et maximale sous forme de tuple (min_date, max_date), ou (None, None) si le symbole n'est pas présent.
        """
        df_temp = self.database[self.database['ID'] == symbol]
        if df_temp.empty:
            return None, None
        return df_temp['Date'].min(), df_temp['Date'].max()

    def get_data(self, symbols, start_date, end_date):
        """
        Filtre et retourne les données de la base de données pour les symboles et la période spécifiés.

        Args:
            symbols (list): Liste des symboles à filtrer.
            start_date (str): Date de début au format 'YYYY-MM-DD'.
            end_date (str): Date de fin au format 'YYYY-MM-DD'.

        Returns:
            pd.DataFrame: Données filtrées correspondant aux critères.
        """
        filtered_data = self.database[
            (self.database['ID'].isin(symbols)) &
            (self.database['Date'] >= start_date) &
            (self.database['Date'] <= end_date)
        ]
        return filtered_data
    
    def update_database(self):
        """
        Vérifie et met à jour la base de données avec les nouvelles données pour chaque symbole.
        Si de nouvelles données sont récupérées, la base de données est mise à jour et sauvegardée.

        Returns:
            list: Liste des symboles non disponibles sur Binance.
        """
        notlisted = []
        modified = False

        for symbol in self.symbols:
            if self.verbose:
                print(f"Vérification des données pour {symbol}...")
            
            min_date, max_date = self.get_symbol_date_range(symbol)

            if min_date is None or max_date < self.end_date:
                start_date = max_date if max_date else self.start_date
                new_data = self.get_binance_historical_data(symbol, start_date, self.end_date)
                
                if new_data is None:
                    notlisted.append(symbol)
                    continue

                self.database = pd.concat([self.database, new_data], ignore_index=True)
                modified = True
                if self.verbose:
                    print(f"Données mises à jour pour {symbol} ({start_date} - {self.end_date}).")
            else:
                if self.verbose:
                    print(f"Les données pour {symbol} sont à jour.")
        
        if modified:
            self.database.sort_values(['ID', 'Date'], inplace=True)
            self.database.to_csv('data/database.csv', index=False)
            if self.verbose:
                print("Base de données mise à jour.")
        else:
            if self.verbose:
                print("Aucune mise à jour nécessaire.")
        
        return notlisted