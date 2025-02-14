import pandas as pd
from datasets import load_dataset
from bert_score import BERTScorer
from rouge_score import rouge_scorer
from flair.data import Sentence
from flair.models import SequenceTagger
from typing import Set, List
from tqdm import tqdm
from datetime import date
import argparse

# Enable tqdm for pandas
tqdm.pandas()


############ 
# NER
############
# downloads to:
# ~/.flair/models/ner-english-large 
# C:\Users\<YourUsername>\.flair\models\ner-english-large
# load from local: tagger = SequenceTagger.load("/my_models/flair_ner_large")
# Copy folder:
# mkdir -p /my_models/flair_ner_large
# cp -r ~/.flair/models/ner-english-large/* /my_models/flair_ner_large/


tagger = SequenceTagger.load("flair/ner-english-large")
 
def extract_named_entities(text: str, allowed_entity_types: List[str]=None):
    """Extract named entities from text using Flair's NER model."""
    if pd.isna(text) or not isinstance(text, str):  # Handle NaN or non-string values
        return []
    
    sentence = Sentence(text)
    tagger.predict(sentence)

    # Define allowed entity types

    # Extract only the entities of interest
    filtered_entities = {
        entity.text for entity in sentence.get_spans('ner') if entity.tag in allowed_entity_types
    }
        
    return filtered_entities


def difference_entities(row, col1: str="model_entities", col2: str="dialog_entities"):
    """Find entities in column 'b' that are not in column 'c'."""
    return row[col1] - row[col2]


def compute_named_entities(df: pd.DataFrame, model_input_col: str, model_output_col: str, allowed_entity_types: List) -> pd.DataFrame:

    df[f"entities_in_{model_input_col}"] = df[model_input_col].progress_apply(extract_named_entities, args=(allowed_entity_types,))

    df[f"entities_in_{model_output_col}"] = df[model_output_col].progress_apply(extract_named_entities, args=(allowed_entity_types,))

    df["hallucinated_entities"] = df.apply(difference_entities, axis=1, args=(f"entities_in_{model_output_col}", f"entities_in_{model_input_col}"))

    df["missed_entities"] = df.apply(difference_entities, axis=1, args=("dialog_entities", "model_entities"))

    df["num_missed_entities"] = df["missed_entities"].apply(lambda x: len(x))

    df["num_hallucinated_entities"] = df["hallucinated_entities"].apply(lambda x: len(x))

    return df




############
# BERT and ROUGE-L
###########
bertscorer = BERTScorer(model_type="microsoft/deberta-xlarge-mnli", device="cuda") #BERTScorer(lang="en", device="cuda")
rougescorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def compute_f1_bertscore(y_preds, y_trues):    

    (P, R, F1), hash_value = bertscorer.score(y_preds, y_trues, return_hash=True)

    #print(type(P))
    return P.item(), R.item(), F1.item()


def compute_rouge_l(y_preds, y_trues):
    # Compute ROUGE-L
    rouge_scores = rougescorer.score(y_trues, y_preds)
    return rouge_scores['rougeL'].fmeasure
    

if __name__ == "__main__":


    parser = argparse.ArgumentParser(description="Evaluate LLM summaries using BERTScore, ROUGE-L, and NER.")
    parser.add_argument("path", type=str, help="Path to the CSV file containing LLM outputs.")
    args = parser.parse_args()


    x_col = "model_output"

    y_col = "summary"

    # load data with llm outputs
    df = pd.read_csv(args.path)

    allowed_entity_types = ["PER", "ORG", "LOC"]

    df = compute_named_entities(
        model_input_col=y_col,
        model_output_col=x_col,
        df=df,
        allowed_entitiy_types=allowed_entity_types,
        )

    df[["bert_precision", "bert_recall", "bert_f1"]] = df.apply(
        lambda row: compute_f1_bertscore(
            [row[x_col]], 
            [row[y_col]]), 
            axis=1,
            result_type="expand"
            )

    df["rouge_l"] = df.apply(
        lambda row: compute_rouge_l(
            row[x_col], 
            row[y_col]
            ), 
            axis=1)


    df.to_csv(f"evaluation_results_{str(date.today())}.csv", index=False)


