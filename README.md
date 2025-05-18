# 📈 Projet Final – Recommandation d'Investissement Multimodale

## Auteurs 
Hassna MARJANE
Celine CAI 
Brice TAGO

## 🎯 Objectif

Ce projet vise à développer un **pipeline automatisé** permettant d’agréger quotidiennement des signaux issus de modèles de **clustering**, **classification**, **régression**, et **analyse de texte (NLP)** pour formuler des **recommandations d’investissement ("buy", "hold", "sell")** sur des actions cotées.

## Exemple de sortie

"Adobe": {
        "entreprise similaire": {
            "ratios financiers": [],
            "rendements": [
                "Apple",
                "Microsoft",
                "Alphabet",
                "Meta",
                "NVIDIA",
                "IBM",
                "Intel",
                "Oracle",
                "Sony",
                "Qualcomm",
                "Cisco",
                "Goldman Sachs",
                "Visa",
                "Johnson & Johnson",
                "Pfizer",
                "ExxonMobil",
                "ASML",
                "SAP",
                "Siemens",
                "Louis Vuitton (LVMH)",
                "TotalEnergies",
                "Shell",
                "Toyota",
                "SoftBank",
                "Tata Consultancy Services",
                "Amazon",
                "AMD",
                "ICBC",
                "JP Morgan",
                "Netflix",
                "Nintendo",
                "Reliance Industries",
                "Samsung"
            ],
            "correlation des rendements": [
                "Apple",
                "Microsoft",
                "Alphabet",
                "Meta",
                "NVIDIA",
                "Qualcomm",
                "Amazon",
                "AMD",
                "Netflix",
                "Tesla"
            ]
        },
        "preconisation positionnement portfeuille": "sell",
        "prediction du prix futur": 404.90759588281713,
        "Sentiment global sur le stock": "neutre",
        "Recommendation finale": "Les signaux sont contrarian. On vend si l' pour se désensibiliser à l'actif sinon on ne fait aucun mouvement"
    },
    "Alibaba": {
        "entreprise similaire": {
            "ratios financiers": [
                "Tencent",
                "IBM",
                "Intel",
                "Sony",
                "Cisco",
                "Johnson & Johnson"
            ],
            "rendements": [
                "Tencent",
                "JD.com",
                "Baidu"
            ],
            "correlation des rendements": [
                "Tencent",
                "JD.com",
                "BYD",
                "Baidu"
            ]
        },
        "preconisation positionnement portfeuille": "sell",
        "prediction du prix futur": 124.90092335205166,
        "Sentiment global sur le stock": "neutre",
        "Recommendation finale": "Les signaux sont contrarian. On vend si l' pour se désensibiliser à l'actif sinon on ne fait aucun mouvement"
    }