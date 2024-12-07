import pandas as pd
import os
from binance.client import Client
import yfinance as yf
from typing import List, Tuple, Optional


class DataBase:

    def __init__(self, verbose: bool = False) -> None:
        """
        Initialise la base de données. Si la connexion à Binance échoue, 
        continue en mode hors-ligne.

        Args:
            verbose (bool): Active les messages de débogage si True.
        """
        self.api_key = 'AAJWE5TewkT5QCivRev9s5r2MpmZMUFXXGokxJL9mlZkZadRiKCEky0tho7OMGxW'
        self.api_secret = 'TxA2VRCyvVHLkn4DZvucbvCkTpNWYDQJeHVcKCDiJD7G5usNd7CrBNKd8rea1vPP'
        self.verbose = verbose
        self.is_online = False

        try:
            self.client = Client(self.api_key, self.api_secret)
            self.is_online = True
            if self.verbose:
                print("Connexion à Binance établie.")
        except Exception as e:
            if self.verbose:
                print(f"Impossible de se connecter à Binance: {e}")
                print("La base de données fonctionnera en mode hors-ligne.")

        if self.verbose:
            print("Initialisation de la base de données...")

        self.load_database()

    def load_database(self) -> None:

        directory = 'data'
        database_file = 'database.csv'

        self.file_path = os.path.join(directory, database_file)
        
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        if not os.path.exists(self.file_path):
            # Créer un DataFrame vide avec la date comme index
            df = pd.DataFrame(columns=['Date'])
            # Sauvegarder le DataFrame vide
            df.to_csv(self.file_path)
        
        # Charger le fichier CSV dans self.database avec la date comme index
        self.database = pd.read_csv(self.file_path, index_col='Date', parse_dates=True)

        #self.database.index = pd.to_datetime(self.database.index)


    def get_historical_close(self, symbols: List[str], start_date: str, end_date: str, backend: str) -> Optional[pd.DataFrame]:

        if backend == 'binance':
            return self.get_binance_historical_close(symbols, start_date, end_date)
        elif backend == 'yfinance':
            return self.get_yfinance_historical_close(symbols, start_date, end_date)
        else:
            raise ValueError("Backend non supporté. Utilisez 'binance' ou 'yfinance'.")

    def get_binance_historical_close(self, symbols: List[str], start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        try:
            data = {}
            
            # Boucle sur chaque symbole pour récupérer les données de clôture
            for symbol in symbols:
                klines = self.client.get_historical_klines(
                    symbol,
                    Client.KLINE_INTERVAL_1DAY,
                    start_date,
                    end_date
                )
                
                # Extraction des dates et des prix de clôture
                close_data = [(pd.to_datetime(kline[0], unit='ms'), float(kline[4])) for kline in klines]
                df = pd.DataFrame(close_data, columns=['date', 'close']).set_index('date')
                data[symbol] = df['close']
            
            # Combine les DataFrames pour chaque symbole en un seul
            result_df = pd.concat(data.values(), axis=1, keys=data.keys())
            result_df.index.name = 'Date'  # Définit le nom de l'index

            # Met l'index en datetime
            result_df.index = pd.to_datetime(result_df.index)

            
            return result_df
        except Exception as e:
            if self.verbose:
                print(f"Erreur lors de la récupération des données pour les symboles {symbols}: {e}")
            return None

    def get_yfinance_historical_close(self, symbols: List[str], start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        data = yf.download(symbols, start=start_date, end=end_date, progress=self.verbose)
        return data['Close']
    
    def _get_symbol_date_range(self, symbol: str) -> Tuple[Optional[str], Optional[str]]:

        if symbol not in self.database.columns:
            # Le symbole n'existe pas encore dans la base de données
            return None, None

        # Utiliser first_valid_index et last_valid_index pour les dates valides
        first_date = self.database[symbol].first_valid_index()
        last_date = self.database[symbol].last_valid_index()

        if first_date is None or last_date is None:
            # Si aucune donnée valide n'est trouvée
            return None, None

        # Retourner les dates sous forme de chaîne (format YYYY-MM-DD)
        return first_date.strftime('%Y-%m-%d'), last_date.strftime('%Y-%m-%d')

    def get_data(self, symbols: List[str], start_date: str, end_date: str) -> pd.DataFrame:

        # Filtrer les symboles qui sont dans la base de données et pas dans notlisted
        valid_symbols = []
        invalid_symbols = []

        for s in symbols:
            if s not in self.notlisted:
                if s in self.database.columns:
                    valid_symbols.append(s)
                else:
                    invalid_symbols.append(s)
                    if self.verbose:
                        print(f"Le symbole {s} n'est pas présent dans la base de données")

        if not valid_symbols:
            if self.verbose:
                print("Aucun symbole valide trouvé.")
            return pd.DataFrame()

        # Conversion des dates
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        # Filtrage des données
        filtered_data = self.database.loc[start_date:end_date, valid_symbols]

        # Drop les lignes où toutes les valeurs sont NaN
        filtered_data = filtered_data.dropna(how='all')
        
        if self.verbose:
            print(f"Données extraites pour {len(valid_symbols)} symboles du {start_date} au {end_date}")
        
        return filtered_data
    
    def del_data(self, symbols: List[str], dates: Optional[List[str]] = None) -> None:

        # Si dates est None, supprimer les colonnes entières
        if dates is None:
            self.database = self.database.drop(columns=symbols)
        else:
            # Convertir les dates en datetime si ce n'est pas déjà fait
            dates = pd.to_datetime(dates)
            # Garder toutes les lignes sauf celles spécifiées dans dates
            self.database = self.database.drop(index=dates)
            
        if self.verbose:
            print(f"Données supprimées pour {symbols}")

    def save_database(self) -> None:

        self.database = self.database.sort_index()
        self.database.to_csv(self.file_path, index=True)
        if self.verbose:
            print("Base de données sauvegardée.")
        
    def update_database(self, symbols: List[str], start_date: str, end_date: str, backend: str) -> List[str]:

        self.notlisted = []
        modified = False

        if not self.is_online:
            if self.verbose:
                print("Base de données en mode hors ligne. Mise à jour impossible.")
            return

        for symbol in symbols:
            if self.verbose:
                print(f"Vérification des données pour {symbol}...")
            
            # Obtenir la plage de dates actuelle pour le symbole
            min_date, max_date = self._get_symbol_date_range(symbol)
            
            # Déterminer la nouvelle plage de dates à récupérer
            if min_date is None or pd.to_datetime(max_date) < pd.to_datetime(end_date):
                new_start_date = max_date if max_date else start_date
                if self.verbose:
                    print(f"Récupération des données pour {symbol} de {new_start_date} à {end_date}...")
                
                # Récupérer les données manquantes
                new_data = self.get_historical_close([symbol], new_start_date, end_date, backend)
                
                if new_data is None:
                    if self.verbose:
                        print(f"Les données pour {symbol} ne sont pas disponibles.")
                    self.notlisted.append(symbol)
                    continue
                
                # Ajouter les nouvelles données à la base de données
                self.database = self.database.combine_first(new_data)
                modified = True
                
                if self.verbose:
                    print(f"Données mises à jour pour {symbol} ({new_start_date} - {end_date}).")
            else:
                if self.verbose:
                    print(f"Les données pour {symbol} sont déjà à jour.")

        # Sauvegarder les modifications si nécessaire
        if modified:
            self.save_database()
        else:
            if self.verbose:
                print("Aucune mise à jour nécessaire.")

            
    @staticmethod
    def from_ohlcv_to_close(ohlcv_df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforme ohlcv_df, un DataFrame OHLCV en un DataFrame avec les dates en index,
        les IDs en colonnes, et les prix de clôture en valeurs.

        Args:
            ohlcv_df (pd.DataFrame): DataFrame contenant les données OHLCV.

        Returns:
            pd.DataFrame: DataFrame pivoté avec les prix de clôture.
        """
        ohlcv_df.columns = [col.upper() for col in ohlcv_df.columns]

        ohlcv_df = ohlcv_df[['DATE', 'ID', 'CLOSE']].copy()

        # Assurez-vous que la colonne 'Date' est au format datetime
        ohlcv_df['DATE'] = pd.to_datetime(ohlcv_df['DATE'])
        
        # Gérer les doublons en gardant la dernière entrée pour chaque combinaison 'Date' et 'ID'
        ohlcv_df = ohlcv_df.sort_values('DATE')
        ohlcv_df = ohlcv_df.drop_duplicates(subset=['DATE', 'ID'], keep='last')
        
        # Pivotement du DataFrame pour obtenir le format désiré
        close_df = ohlcv_df.pivot(index='DATE', columns='ID', values='CLOSE')
        
        return close_df

