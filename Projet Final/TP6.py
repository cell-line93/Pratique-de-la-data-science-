import requests
import json
from datetime import datetime, timedelta
import os
import glob 

# Clé API personnelle à récupérer sur le site NewsAPI
api_key = "add233078aa847728b005b68d19394cb"

# Sources d'actualités financières (ajustez selon votre besoin)
sources = 'financial-post, the-wall-street-journal, bloomberg, the-washington-post, australian-financial-review, bbc-news, cnn'

# Fonction pour initialiser les paramètres de la requête
def initialize_params(company_name: str):
    last_day = datetime.today().strftime('%Y-%m-%d')
    first_day = (datetime.today() - timedelta(days=20)).strftime('%Y-%m-%d')
    
    params = {
        "sources": sources,
        "q": company_name,
        "apiKey": api_key,
        "language": "en",
        "pageSize": 100,
        "from": first_day,
        "to": last_day,
    }
    
    return params

#Sauvegarder les actualités mises à jour
def save_news(news_dict: dict, company_name: str, directory: str):
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, f"{company_name.lower()}_news.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(news_dict, f, indent=4, ensure_ascii=False)


def get_news(company_name: str, path: str, save:bool):
    url = 'https://newsapi.org/v2/everything'
    params = initialize_params(company_name)
    
    # Faire la requête GET
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        news_data = response.json()
        news_dict = {}

        # Parcourir les articles et les filtrer par entreprise
        for article in news_data['articles']:
            title = article['title']
            description = article['description']
            published_at = article['publishedAt'].split("T")[0]
            source_name = article['source']['name']

            # Vérifier que le nom de l'entreprise est mentionné dans le titre ou la description
            if company_name.lower() in title.lower() or company_name.lower() in description.lower():
                if published_at not in news_dict:
                    news_dict[published_at] = []

                news_dict[published_at].append({
                    'title': title,
                    'description': description,
                    'publishedAt': published_at,
                    'source': source_name,
                    'url': article['url']
                })
        
        if save:
            save_news(news_dict, company_name, path)

        return news_dict
    else:
        print(f"Error: {response.status_code}")
        return {}

# Charger les actualités existantes (si fichier JSON déjà présent)
def load_existing_news(company_name: str, directory: str = "news_data") -> dict:
    filepath = os.path.join(directory, f"{company_name.lower()}_news.json")
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def update_news(company_name: str, file_path: str):
    # Charger les actualités existantes
    existing_news = load_existing_news(company_name, file_path)
    
    # Scraper les dernières actualités
    new_news = get_news(company_name, file_path, False)
    
    # Ajouter uniquement les nouveaux articles non présents
    for date, articles in new_news.items():
        if date not in existing_news:
            existing_news[date] = []
        for article in articles:
            # Vérifier si l'article existe déjà (en fonction du titre)
            if not any(existing_article['title'] == article['title'] for existing_article in existing_news[date]):
                existing_news[date].append(article)
    
    # Sauvegarder les nouvelles actualités dans le fichier JSON
    with open(file_path, 'w') as file:
        json.dump(existing_news, file, indent=4)

    return existing_news
    

def get_news_all_companies(new_data_path:str, company_name_path: str):
    filepaths = glob.glob(f"{company_name_path}/*.csv")
    news_all_companies = {}

    for file in filepaths:
        filename = os.path.basename(file)
        filename = os.path.splitext(filename)[0].split('_')[0]
        news_company = get_news(filename,new_data_path, True)
        news_all_companies[filename] = {"news": news_company}
    
    save_path = os.path.join("Projet Final\\NLP\\Resultats", "news_all_companies.json")
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(news_all_companies, f, ensure_ascii=False, indent=4)
    

def update_news_all_companies(news_data_path: str, company_name_path: str):
    filepaths = glob.glob(f"{company_name_path}/*.csv")
    news_all_companies = {}
    
    for file in filepaths:
        filename = os.path.basename(file)
        company_name = os.path.splitext(filename)[0].split('_')[0].lower()
        json_path = os.path.join(news_data_path, f"{company_name}_news.json")
        news_company = update_news(company_name, json_path)
        news_all_companies[filename] = {"news": news_company}

    save_path = os.path.join("Projet Final\\NLP\\Resultats", "news_all_companies.json")
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(news_all_companies, f, ensure_ascii=False, indent=4)

def news_scraping_all_companies():
    
    path = "Projet Final\\Companies_news_data"
    existing_json_files = glob.glob(f"{path}/*.json")

    if len(existing_json_files) == 0:
        get_news_all_companies("Projet Final\\Companies_news_data", "Projet Final\\Companies_historical_data")
    else:
        update_news_all_companies("Projet Final\\Companies_news_data", "Projet Final\\Companies_historical_data")

    