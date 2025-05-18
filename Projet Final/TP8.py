import pytz
from datetime import datetime
import glob
import os
import json
from collections import Counter, defaultdict
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import yfinance as yf
from transformers import BertTokenizer, BertForSequenceClassification
import torch


# --- Extraction des textes et timestamps ---
def get_texts_timestamps(news_data):
    eastern = pytz.timezone('America/New_York')
    texts, timestamps = [], []

    for article in news_data:
        ts_utc = datetime.fromisoformat(article['publishedAt'].replace('Z', '+00:00'))
        ts_ny = ts_utc.astimezone(eastern)
        ts_ny_hour = ts_ny.replace(minute=0, second=0, microsecond=0)

        full_text = f"{article.get('title', '')} {article.get('description', '')}"
        texts.append(full_text.strip())
        timestamps.append(ts_ny_hour)

    return texts, timestamps


# --- Prédiction des sentiments avec un modèle BERT ---
def get_sentiments(model_path, texts):
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    model.eval()

    sentiments = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
            pred = torch.argmax(outputs.logits, dim=1).item()
        sentiments.append(pred)

    return sentiments


# --- Alignement des timestamps sur les heures de marché ---
def align_timestamps(timestamps):
    aligned = []
    for ts in timestamps:
        hour = ts.hour + ts.minute / 60
        if 9.5 <= hour < 15:
            aligned.append(ts.replace(minute=0, second=0, microsecond=0))
        elif 15 <= hour < 24:
            aligned.append(ts.replace(hour=15, minute=0, second=0, microsecond=0))
        else:
            aligned.append((ts - pd.Timedelta(days=1)).replace(hour=15, minute=0, second=0, microsecond=0))
    return aligned


# --- Affichage comparatif des sentiments sur le graphique de prix ---
def plot_comparison(df, sentiments_a, sentiments_b, timestamps, title_a, title_b):
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = df.set_index("Datetime")
    ts_aligned = align_timestamps(timestamps)

    def group_sentiments(sentiments):
        grouped = defaultdict(list)
        for ts, s in zip(ts_aligned, sentiments):
            grouped[ts].append(s)
        return grouped

    colors = {0: 'gold', 1: 'green', 2: 'red'}  # Neutre, Positif, Négatif
    offset = {0: 0.5, 1: 1, 2: 1.5}

    fig, axs = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    for sentiments, title, ax in zip([sentiments_a, sentiments_b], [title_a, title_b], axs):
        ax.plot(df.index, df["Close"], label="Prix", color="black")
        grouped = group_sentiments(sentiments)

        for ts, s_list in grouped.items():
            if ts not in df.index:
                idx = df.index.get_indexer([ts], method='nearest')[0]
                ts = df.index[idx]

            for s in s_list:
                y = df["Close"].get(ts, None)
                if y is not None:
                    ax.scatter(ts, y + offset[s]*0.5, color=colors[s], s=50)

        ax.set_title(title)
        ax.grid(True)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Positif', markerfacecolor='green', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Neutre', markerfacecolor='gold', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Négatif', markerfacecolor='red', markersize=10),
        Line2D([0], [0], color='black', label='Prix')
    ]
    axs[1].legend(handles=legend_elements)
    plt.tight_layout()
    plt.show()


# --- Analyse d'impact du sentiment sur une entreprise donnée ---
def analyze_company_sentiment_impact(ticker_symbol, json_path, model_path_finetuned, start_date="2025-01-01"):
    ticker = yf.Ticker(ticker_symbol)
    df = ticker.history(start=start_date, interval="60m").reset_index()

    with open(json_path, "r") as f:
        news_data = json.load(f)

    if isinstance(news_data, dict):
        news_data = [article for articles in news_data.values() for article in articles]

    texts, timestamps = get_texts_timestamps(news_data)

    sentiments_original = get_sentiments("yiyanghkust/finbert-tone", texts)
    sentiments_finetuned = get_sentiments(model_path_finetuned, texts)

    plot_comparison(
        df,
        sentiments_original,
        sentiments_finetuned,
        timestamps,
        title_a="FinBERT (original)",
        title_b="FinBERT (fine-tuné)"
    )


# --- Fonction commune pour récupérer sentiments d'une liste de textes (optimisation interne) ---
def _get_sentiments_for_texts(model_path, texts):
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    model.eval()

    sentiments = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
            pred = torch.argmax(outputs.logits, dim=1).item()
        sentiments.append(pred)

    return sentiments


# --- Calcul des sentiments par entreprise depuis un répertoire JSON ---
def sentiments_par_entreprise(json_dir, model_path, min_articles=10):
    results = {}
    files = glob.glob(os.path.join(json_dir, "*.json"))

    for path in files:
        company = os.path.basename(path).replace("_news.json", "").replace("news_", "").lower()

        with open(path, "r") as f:
            news_data = json.load(f)

        if isinstance(news_data, dict):
            all_articles = []
            for date in news_data:
                all_articles.extend(news_data[date])
        else:
            continue  # format invalide

        if len(all_articles) < min_articles:
            continue

        texts = [f"{article.get('title', '')} {article.get('description', '')}".strip() for article in all_articles]
        sentiments = _get_sentiments_for_texts(model_path, texts)
        results[company] = sentiments

    return results


# --- Résumé des sentiments par entreprise (counts et ratios) ---
def resumer_sentiments_par_entreprise(min_articles=10):
    summary = {}
    json_dir = "Projet Final\\NLP\\Resultats"
    model_path = "Projet Final\\NLP\\Fine-tuning"

    files = glob.glob(os.path.join(json_dir, "*.json"))

    for path in files:
        company = os.path.basename(path).replace("_news.json", "").replace("news_", "").lower()

        with open(path, "r") as f:
            news_data = json.load(f)

        if isinstance(news_data, dict):
            all_articles = []
            for date in news_data:
                all_articles.extend(news_data[date])
        else:
            continue

        if len(all_articles) < min_articles:
            continue

        texts = [f"{article.get('title', '')} {article.get('description', '')}".strip() for article in all_articles]
        sentiments = _get_sentiments_for_texts(model_path, texts)
        counts = Counter(sentiments)
        total = sum(counts.values())

        summary[company] = {
            "total_articles": total,
            "positive": counts.get(1, 0),
            "neutral": counts.get(0, 0),
            "negative": counts.get(2, 0),
            "positive_ratio": counts.get(1, 0) / total if total > 0 else 0,
            "neutral_ratio": counts.get(0, 0) / total if total > 0 else 0,
            "negative_ratio": counts.get(2, 0) / total if total > 0 else 0,
        }

    return summary
