from datasets import load_dataset
import pandas as pd
from bert_score import score as bert_score
from rouge_score import rouge_scorer
import spacy

def compute_hallucinated_entities(df, summary_col, model_cols):
    nlp = spacy.load("en_core_web_sm")
    results = {col: [] for col in model_cols}
    
    for _, row in df.iterrows():
        gold_entities = {ent.text for ent in nlp(row[summary_col]).ents}
        
        for col in model_cols:
            model_entities = {ent.text for ent in nlp(row[col]).ents}
            hallucinated_entities = model_entities - gold_entities
            results[col].append(len(hallucinated_entities))
    
    for col in model_cols:
        df[f'hallucinated_entities_{col}'] = results[col]
    
    return df


def compute_scores(df, summary_col, model_cols):
    results = {col: {'bert': [], 'rouge-l': []} for col in model_cols}

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    for _, row in df.iterrows():
        gold_summary = row[summary_col]

        for col in model_cols:
            model_summary = row[col]

            # Compute BERTScore
            P, R, F1 = bert_score([model_summary], [gold_summary], lang="en", rescale_with_baseline=True)
            results[col]['bert'].append(F1.item())

            # Compute ROUGE-L
            rouge_scores = scorer.score(gold_summary, model_summary)
            results[col]['rouge-l'].append(rouge_scores['rougeL'].fmeasure)

    for col in model_cols:
        df[f'bert_score_{col}'] = results[col]['bert']
        df[f'rougeL_score_{col}'] = results[col]['rouge-l']

    return df


data_id = "ThatsGroes/LLM-summary-evaluation-extended"

dataset = load_dataset(data_id)

model_ids = ["ThatsGroes/SmolLM2-360M-Instruct-summarizer", "ThatsGroes/SmolLM2-1.7B-Instruct-summarizer", "HuggingFaceTB/SmolLM2-1.7B-Instruct", "HuggingFaceTB/SmolLM2-360M-Instruct"]

col_prefix = "summary_by_"

cols_to_analyze = [col_prefix+id.split("/")[-1] for id in model_ids]

df = dataset["test"].to_pandas()

df_scores = compute_scores(df, 'summary', cols_to_analyze)

df_scores.to_csv("scores.csv", index=False)

df_entities = compute_hallucinated_entities(df, 'summary', cols_to_analyze)

df_entities.to_csv("entities.csv", index=False)