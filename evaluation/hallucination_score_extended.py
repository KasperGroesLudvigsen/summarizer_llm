from datasets import load_dataset
import pandas as pd
import spacy
from tqdm import tqdm

def compute_hallucinated_entities(df, summary_col, model_cols):
    nlp = spacy.load("en_core_web_sm")

    num_hallucinated_ent = {col: [] for col in model_cols}
    num_missed_ent = {col: [] for col in model_cols}

    hallucinated_entities = {col: [] for col in model_cols}
    missed_entities = {col: [] for col in model_cols}

    all_gold_entities = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Computing hallucinated entities"):
        gold_entities = {ent.text for ent in nlp(row[summary_col]).ents}
        all_gold_entities.append(gold_entities)
        
        for col in model_cols:
            model_entities = {ent.text for ent in nlp(row[col]).ents}
            hallucinated = model_entities - gold_entities
            missed_entities = gold_entities - model_entities

            # Collect results
            num_hallucinated_ent[col].append(len(hallucinated))

            num_missed_ent[col].append(len(missed_entities))

            hallucinated_entities[col].append(list(hallucinated))

            missed_entities[col].append(list(missed_entities))

    for col in model_cols:
        df[f'num_hallucinated_entities_{col}'] = num_hallucinated_ent[col]
        df[f'hallucinated_entities_{col}'] = hallucinated_entities[col]
        df[f'num_missed_entities_{col}'] = num_missed_ent[col]
        df[f'missed_entities_{col}'] = missed_entities[col]
        

    df["gold_entities"] = all_gold_entities
    
    return df


data_id = "ThatsGroes/LLM-summary-evaluation"

dataset = load_dataset(data_id)

model_ids = ["ThatsGroes/SmolLM2-360M-Instruct-summarizer", "ThatsGroes/SmolLM2-1.7B-Instruct-summarizer", "HuggingFaceTB/SmolLM2-1.7B-Instruct", "HuggingFaceTB/SmolLM2-360M-Instruct"]

col_prefix = "summary_by_"

cols_to_analyze = [col_prefix+id.split("/")[-1] for id in model_ids]

df = dataset["test"].to_pandas()

df_entities = compute_hallucinated_entities(df, 'summary', cols_to_analyze)

df_entities.to_csv("entities.csv", index=False)