import json 

def get_recommendation_for_all_companies(wanna_run_all):

    if wanna_run_all == True:
        from TP1 import get_companies_financial_ratios_and_historical_data
        from TP2 import get_similar_companie_dico
        from TP3 import give_actual_recommendation_all_companies
        from TP4 import give_next_close_all_companies
        from TP6 import news_scraping_all_companies
        from TP8 import resumer_sentiments_par_entreprise
        """
            Scraping pour Récupèrer les ratios financiers et prix historiques des entreprises
        """ 
        #get_companies_financial_ratios_and_historical_data()


        """
            Clustering pour determiner les entreprises similaires selon le critère risque, rendement, et corrélation 
            + Evaluation des modèles de clustering utilisés
        """  
        similar_companies = get_similar_companie_dico()
        #print(similar_companies)

        """
            Classification supervisée pour donner une recommendation d'achat de vente pour chaque entreprise à la dernière date 
            du dataset
        """  
        companies_recommendation = give_actual_recommendation_all_companies()
        #print(companies_recommendation)

        """
            Regression avec ML pour donner prevoir le close en N+1 pour chaque entreprise à la dernière date 
            du dataset
        """  
        next_close = give_next_close_all_companies()
        # print(next_close)

        """
            Scraper les news des entreprise sous format JSON
        """
        news_scraping_all_companies()

        """
            Analyse de sentiments 
        """
        sentiment_analysis = resumer_sentiments_par_entreprise()
        #print(sentiment_analysis)

        """
            Agregation en un seul JSON
        """
    else:
        similar_companies_json_path = "Projet Final\\Clustering\\Resultats\\similar_companies.json"

        with open(similar_companies_json_path, "r", encoding="utf-8") as f:
            similar_companies = json.load(f)

        companies_recommendation_json_path = "Projet Final\\Classification\\Resultats\\recommendation_companies.json"

        with open(companies_recommendation_json_path, "r", encoding="utf-8") as f:
            companies_recommendation = json.load(f)
        
        next_close_json_path = "Projet Final\\Regression\\ML\\Resultats\\next_close_companies.json"

        with open(next_close_json_path, "r", encoding="utf-8") as f:
            next_close = json.load(f)

        sentiment_analysis_json_path = "Projet Final\\NLP\\Resultats\\resultats_bert.json"

        with open(sentiment_analysis_json_path, "r", encoding="utf-8") as f:
            sentiment_analysis = json.load(f)
         
    aggregated_json = {}
    for key in next_close.keys():

        if companies_recommendation[key] == "buy" and sentiment_analysis[key.lower()]["Sentiment global"] == "positif":
            signal_final = "Il faut acheter l'actif si on ne l'a pas et renforcer si on l'a dejà"

        elif companies_recommendation[key] == "sell" and sentiment_analysis[key.lower()]["Sentiment global"] == "n\u00e9gatif":
            signal_final = "Il faut vendre l'actif si on l'a et ne pas l'acheter ou le shorter si on ne l'a pas"

        else:
            signal_final = "Les signaux sont contrarian. On vend si l' pour se désensibiliser à l'actif sinon on ne fait aucun mouvement"

        
        aggregated_json[key]= {"entreprise similaire": similar_companies[key],
                              "preconisation positionnement portfeuille": companies_recommendation[key],
                              "prediction du prix futur": next_close[key],
                              "Sentiment global sur le stock": sentiment_analysis[key.lower()]["Sentiment global"],
                              "Recommendation finale": signal_final
                            }
        
    path_aggregated_json = "Projet Final\\Recommendation_du_jour.json"
    with open(path_aggregated_json, 'w', encoding='utf-8') as f:
        json.dump(aggregated_json, f, ensure_ascii=False, indent=4)
    
    return aggregated_json
    
   
    

if __name__ == "__main__":
    get_recommendation_for_all_companies(wanna_run_all=False)