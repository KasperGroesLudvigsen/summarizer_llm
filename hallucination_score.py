from datasets import load_dataset
import pandas as pd
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


data_id = "ThatsGroes/LLM-summary-evaluation-extended"

dataset = load_dataset(data_id)

model_ids = ["ThatsGroes/SmolLM2-360M-Instruct-summarizer", "ThatsGroes/SmolLM2-1.7B-Instruct-summarizer", "HuggingFaceTB/SmolLM2-1.7B-Instruct", "HuggingFaceTB/SmolLM2-360M-Instruct"]

col_prefix = "summary_by_"

cols_to_analyze = [col_prefix+id.split("/")[-1] for id in model_ids]

df = dataset["test"].to_pandas()

df_entities = compute_hallucinated_entities(df, 'summary', cols_to_analyze)

df_entities.to_csv("entities.csv", index=False)