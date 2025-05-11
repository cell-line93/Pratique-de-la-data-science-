# chargement des packages 
import yfinance as yf 
import pandas as pd 
import os
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
def scrap_ratios(companies: Dict[str, str]) -> None:
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
    df_ratios.to_csv("company_ratios.csv")
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