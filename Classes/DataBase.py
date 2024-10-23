import pandas as pd
import os
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

        self.load_database()

    def load_database(self):
        """
        Charge la base de données à partir du fichier CSV 'database.csv'.
        Si le fichier ou le dossier n'existe pas, les crée avec les colonnes spécifiées.
        """
        directory = 'data'
        file_path = os.path.join(directory, 'database.csv')
        
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        if not os.path.exists(file_path):
            # Créer un DataFrame vide avec les colonnes spécifiées
            columns = ['ID', 'Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
            df = pd.DataFrame(columns=columns)
            # Sauvegarder le DataFrame dans un fichier CSV
            df.to_csv(file_path, index=False)
        
        # Charger le fichier CSV dans self.database
        self.database = pd.read_csv(file_path)

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
    
    def del_data(self, symbols, dates=None):
        """
        Supprime les lignes de la base de données pour les symboles et les dates spécifiés.

        Args:
            symbols (list): Liste des symboles à filtrer.
            dates (list, optional): Liste des dates au format 'YYYY-MM-DD' à supprimer. Si None, supprime pour toutes les dates.

        Returns:
            None
        """
        if dates is None:
            # Filtrer les données à supprimer uniquement par symboles
            mask = self.database['ID'].isin(symbols)
        else:
            # Filtrer les données à supprimer par symboles et dates
            mask = (self.database['ID'].isin(symbols)) & (self.database['Date'].isin(dates))
        
        # Supprimer les lignes correspondantes
        self.database = self.database[~mask]

    def save_database(self):
        """
        Sauvegarde la base de données dans le fichier CSV 'database.csv'.
        """
        self.database = self.database.sort_values(['ID', 'Date'])
        self.database.to_csv('data/database.csv', index=False)
        if self.verbose:
            print("Base de données sauvegardée.")
    
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
            self.save_database()
        else:
            if self.verbose:
                print("Aucune mise à jour nécessaire.")
        
        return notlisted
    
    def from_ohlcv_to_close(self, ohlcv_df):
        """
        Transforme ohlcv_df, un DataFrame OHLCV en un DataFrame avec les dates en index,
        les IDs en colonnes, et les prix de clôture en valeurs.
        """

        ohlcv_df = ohlcv_df[['Date', 'ID', 'Close']].copy()

        # Assurez-vous que la colonne 'Date' est au format datetime
        ohlcv_df['Date'] = pd.to_datetime(ohlcv_df['Date'])
        
        # Gérer les doublons en gardant la dernière entrée pour chaque combinaison 'Date' et 'ID'
        ohlcv_df = ohlcv_df.sort_values('Date')
        ohlcv_df = ohlcv_df.drop_duplicates(subset=['Date', 'ID'], keep='last')
        
        # Pivotement du DataFrame pour obtenir le format désiré
        close_df = ohlcv_df.pivot(index='Date', columns='ID', values='Close')
        
        return close_df

