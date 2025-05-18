import os
from datasets import load_dataset, concatenate_datasets, DatasetDict
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def load_and_prepare_datasets():
    """
    Charge les datasets annotés et harmonise les labels entre les deux jeux de données.
    Retourne un DatasetDict combiné avec train et test.
    """

    ds1 = load_dataset("zeroshot/twitter-financial-news-sentiment")
    ds2 = load_dataset("nickmuchi/financial-classification")

    def map_labels(example):
        if '__hf_source' in example and "twitter-financial-news-sentiment" in str(example['__hf_source']):
            if example['label'] == 0:  # Bearish
                example['label'] = 2  # Negative
            elif example['label'] == 1:  # Bullish
                example['label'] = 1  # Positive
            elif example['label'] == 2:  # Neutral
                example['label'] = 0  # Neutral
        return example

    ds1 = ds1.map(map_labels)

    def map_labels2(example):
        # ds2 a déjà le champ 'labels'
        # Ici on remappe labels -> label
        if '__hf_source' in example and "financial-classification" in str(example['__hf_source']):
            pass
        else:
            if example['labels'] == 0:  # Negative
                example['labels'] = 2
            elif example['labels'] == 1:  # Positive
                example['labels'] = 1
            elif example['labels'] == 2:  # Neutral
                example['labels'] = 0
        return example

    ds2 = ds2.map(map_labels2)

    ds2 = ds2.rename_column("labels", "label")

    try:
        ds1 = ds1.remove_columns("labels")
    except Exception:
        pass

    combined_dataset = DatasetDict({
        'train': concatenate_datasets([ds1['train'], ds2['train']]),
        'test': concatenate_datasets([ds1['validation'], ds2['test']])
    })

    return combined_dataset


def train_model(model_name, dataset, batch_size=16, num_epochs=3, save_path=None):
    """
    Entraîne un modèle de classification de texte à partir d'un dataset donné.

    Args:
        model_name (str): Nom du modèle HuggingFace.
        dataset (DatasetDict): Dataset combiné avec 'train' et 'test'.
        batch_size (int): Taille du batch.
        num_epochs (int): Nombre d'époques d'entraînement.
        save_path (str): Chemin pour sauvegarder le modèle. Si None, chemin par défaut sera utilisé.

    Returns:
        dict: Les métriques d'évaluation.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

    tokenized_train = dataset['train'].map(tokenize_function, batched=True)
    tokenized_test = dataset['test'].map(tokenize_function, batched=True)

    tokenized_train.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    tokenized_test.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    training_args = TrainingArguments(
        output_dir=f"./{model_name.replace('/', '_')}_results",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        logging_dir=f"./{model_name.replace('/', '_')}_logs",
        evaluation_strategy="epoch",
        save_strategy="epoch"
    )

    def compute_metrics(pred):
        labels = pred.label_ids
        preds = np.argmax(pred.predictions, axis=1)
        acc = accuracy_score(labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
        return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        compute_metrics=compute_metrics
    )

    trainer.train()
    metrics = trainer.evaluate()

    if save_path is None:
        save_path = os.path.join("Projet Final\\NLP\\Fine-tuning", f"{model_name.replace('/', '_')}_finetuned")

    os.makedirs(save_path, exist_ok=True)

    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    return metrics

#fonction pour finetuner plusieurs modèles et retourner les résultats
def finetune_multiple_models(dataset, models, batch_size=16, num_epochs=1):
    """
    Entraîne plusieurs modèles séquentiellement sur le même dataset.

    Args:
        dataset (DatasetDict): Dataset combiné.
        models (list): Liste des noms de modèles HF.
        batch_size (int): Taille du batch.
        num_epochs (int): Nombre d'époques.

    Returns:
        dict: Dictionnaire {model_name: metrics}
    """
    results = {}
    for model_name in models:
        metrics = train_model(model_name, dataset, batch_size=batch_size, num_epochs=num_epochs)
        results[model_name] = metrics
    return results


def interpret_results():
    """
    Affiche une interprétation des résultats d'évaluation des modèles BERT et FinBERT
    selon les métriques communes.
    """
    print("""
    **Interprétation des résultats :**

    - Perte (Loss) : Le modèle BERT a une perte d'évaluation légèrement plus faible (0.3566) par rapport à FinBERT (0.3591), indiquant un meilleur apprentissage.

    - Accuracy : BERT obtient 86.45%, légèrement supérieur à FinBERT (85.90%).

    - Précision : BERT (86.31%) est aussi un peu meilleur que FinBERT (86.12%).

    - Rappel (Recall) : BERT (86.45%) dépasse FinBERT (85.90%).

    - Score F1 : BERT (86.31%) est légèrement meilleur que FinBERT (85.99%).

    - Temps d'exécution : Très similaires, autour de 5 secondes.

    - Nombre d'exemples traités par seconde : BERT 583 vs FinBERT 580, très proches.

    Conclusion : BERT surpasse légèrement FinBERT sur toutes les métriques. FinBERT reste spécialisé en finance, pouvant être avantageux dans certains cas spécifiques.
    """)