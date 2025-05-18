# chargement des packages 
import yfinance as yf 
import pandas as pd 
import matplotlib.pyplot as plt
import os
from datetime import date
from typing import Dict, List

# Dictionnaire des compagnies
companies: Dict[str, str]= {"Apple": "AAPL","Microsoft": "MSFT","Amazon": "AMZN","Alphabet": "GOOGL","Meta": "META","Tesla": "TSLA",
 "NVIDIA": "NVDA","Samsung": "005930.KS","Tencent": "TCEHY","Alibaba": "BABA","IBM": "IBM","Intel": "INTC","Oracle": "ORCL",
 "Sony": "SONY","Adobe": "ADBE","Netflix": "NFLX","AMD": "AMD","Qualcomm": "QCOM","Cisco": "CSCO","JP Morgan": "JPM",
 "Goldman Sachs": "GS","Visa": "V","Johnson & Johnson": "JNJ","Pfizer": "PFE","ExxonMobil": "XOM","ASML": "ASML.AS","SAP": "SAP.DE",
 "Siemens": "SIE.DE","Louis Vuitton (LVMH)": "MC.PA","TotalEnergies": "TTE.PA","Shell": "SHEL.L","Baidu": "BIDU","JD.com": "JD",
 "BYD": "BYDDY","ICBC": "1398.HK","Toyota": "TM","SoftBank": "9984.T","Nintendo": "NTDOY","Hyundai": "HYMTF",
 "Reliance Industries": "RELIANCE.NS","Tata Consultancy Services": "TCS.NS"}

# Liste des ratios à récupérer
ratio_names: List[str] = [
    "forwardPE", "beta", "priceToBook", "priceToSales", "dividendYield",
    "trailingEps", "debtToEquity", "currentRatio", "quickRatio",
    "returnOnEquity", "returnOnAssets", "operatingMargins", "profitMargins"
]

# -------------------------------
# 1.1  Scrapping Ratios financiers
# -------------------------------
def scrap_ratios(companies: Dict[str, str], ratio_names) -> None:
    """
    Récupère les ratios fondamentaux de chaque entreprise et les enregistre dans un fichier CSV.

    Args:
        companies (Dict[str, str]): Dictionnaire associant noms d'entreprises à leurs tickers.
    """

    print("Scraping ratios financiers...")

    # Initialiser les colonnes avec des listes vides
    ratios: Dict[str, List[float]] = {key: [] for key in ratio_names}
    company_names: List[str] = []

    for name, symbol in companies.items():
        print(f" - {name} ({symbol})")
        ticker = yf.Ticker(symbol)
        info = ticker.info
        company_names.append(name)

        # Récupérer chaque ratio ou None si indisponible
        for ratio in ratio_names:
            value = info.get(ratio, None)
            ratios[ratio].append(value)

    # Créer un DataFrame et exporter en CSV
    df_ratios = pd.DataFrame(ratios, index=company_names)
    df_ratios.to_csv("companies_ratio.csv")
    print("Ratios sauvegardés dans 'company_ratios.csv'")


# ---------------------------------------
# 1.2  Scrapping variations des stocks
# ---------------------------------------
def scrap_historical_data(companies: Dict[str, str], start: str="2019-01-01", end: str="2024-12-31") -> None:
    """
    Récupère les données historiques de cours de clôture pour chaque entreprise et calcule les rendements.

    Args:
        companies (Dict[str, str]): Dictionnaire des entreprises à scraper.
        start (str): Date de début au format 'YYYY-MM-DD'.
        end (str): Date de fin au format 'YYYY-MM-DD'.
    """

    print("Scraping des données historiques...")
    output_dir = "Companies_historical_data"
    os.makedirs(output_dir, exist_ok=True)

    for name, symbol in companies.items():
        print(f" - {name} ({symbol})")
        try:
            # Télécharger les données
            data = yf.download(symbol, start=start, end=end)
            if data.empty:
                print(f"Aucune donnée pour {symbol}")
                continue

            # Calcul du rendement journalier basé sur la clôture
            data.columns = data.columns.get_level_values(0)
            data = data[["Close"]].copy()
            data["Next Day Close"] = data["Close"].shift(-1)
            data["Rendement"] = data["Next Day Close"] / data["Close"] - 1

            # Exporter dans un CSV
            file_path = os.path.join(output_dir, f"{name.replace(' ', '_')}.csv")
            data.to_csv(file_path)
        except Exception as e:
            print(f"Erreur pour {symbol} : {e}")

    print(f"Données sauvegardées dans le dossier '{output_dir}'")


# ---------------------------------------
# 1.3  Visualisation des données
# ---------------------------------------

def plot_stock_data(data, path, ticker_symbol=None, company_name=None):
    # Vérification des entrées
    if not ticker_symbol and not company_name:
        print("Vous devez renseigner soit un ticker, soit un nom d'entreprise.")
        return

    # Déduction du nom de l'entreprise si on a le ticker
    if ticker_symbol and not company_name:
        company_name = next((k for k, v in companies.items() if v == ticker_symbol), None)
        if company_name is None:
            print(f"Ticker '{ticker_symbol}' introuvable dans la liste.")
            return

    # Construction du chemin vers le fichier CSV
    file_path = os.path.join(path,f"{company_name}_data.csv")

    # Vérification de l'existence du fichier
    if not os.path.exists(file_path):
        print(f"Fichier non trouvé pour l'entreprise : {company_name}")
        return

    # Chargement des données
    df = pd.read_csv(file_path)

    # Vérification de la colonne demandée
    if data not in df.columns:
        print(f"La colonne '{data}' n'existe pas dans les données de {company_name}.")
        print(f"Colonnes disponibles : {list(df.columns)}")
        return

    # Tracé du graphique
    plt.figure(figsize=(14, 7))
    plt.plot(df['Date'], df[data], label=data)
    plt.title(f"{data} - {company_name}")
    plt.xlabel("Date")
    plt.ylabel(data)
    plt.xticks(df.index[::100], df['Date'][::100], rotation=90)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_stock_data_comparison(data, path, ticker_symbols=None, company_names=None):
    if not ticker_symbols and not company_names:
        print("Vous devez renseigner au moins un ticker ou un nom d'entreprise.")
        return

    # Si des tickers sont fournis, les convertir en noms d'entreprise
    selected_companies = []
    if ticker_symbols:
        for ticker in ticker_symbols:
            company = next((k for k, v in companies.items() if v == ticker), None)
            if company:
                selected_companies.append(company)
            else:
                print(f"Ticker '{ticker}' non trouvé.")

    # Ajouter les noms d'entreprise s’ils sont fournis directement
    if company_names:
        selected_companies.extend(company_names)

    # Suppression des doublons
    selected_companies = list(set(selected_companies))

    # Initialisation du graphique
    plt.figure(figsize=(14, 7))

    for company in selected_companies:
        file_path = os.path.join(path, f"{company}_data.csv")
        if not os.path.exists(file_path):
            print(f"Fichier non trouvé pour l'entreprise : {company}")
            continue

        df = pd.read_csv(file_path)

        if 'Date' not in df.columns or data not in df.columns:
            print(f"Données manquantes dans {company} : 'Date' ou '{data}'")
            continue

        # S'assurer que Date est bien en format datetime
        df['Date'] = pd.to_datetime(df['Date'])

        plt.plot(df['Date'], df[data], label=company)

    plt.title(f"Comparaison des {data} entre entreprises")
    plt.xlabel("Date")
    plt.ylabel(data)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ---------------------------------------
# 1.4  Fonction a run 
# ---------------------------------------

def get_companies_financial_ratios_and_historical_data():
    scrap_historical_data(companies=companies)
    scrap_ratios(companies=companies, ratio_names=ratio_names)


