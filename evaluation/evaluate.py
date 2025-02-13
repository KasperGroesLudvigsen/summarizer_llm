import pandas as pd
from datasets import load_dataset
from bert_score import BERTScorer
from rouge_score import rouge_scorer
from flair.data import Sentence
from flair.models import SequenceTagger
from typing import Set, List
from tqdm import tqdm

# Enable tqdm for pandas
tqdm.pandas()

def get_data():

    data = load_dataset("ThatsGroes/LLM-summary-evaluation", trust_remote_code=True)

    df = data["test"].to_pandas()

    id_vars = ["summary", "dialog", "system_prompt", "messages", "text", "prompt"]

    df = df.melt(id_vars=id_vars, var_name="model", value_name="model_output")

    df["model"] = df["model"].str.replace("summary_by_", "")

    df.dropna(subset=["summary", "model_output"], inplace=True)

    df["dialog"] = df["dialog"].apply(lambda x: x.split("\n\n **Conversation:** \n\n")[-1].strip())

    return df

df = get_data()

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


allowed_entity_types = ["PER", "ORG", "LOC"]

df["dialog_entities"] = df["dialog"].progress_apply(extract_named_entities, args=(allowed_entity_types,))

df["model_entities"] = df["model_output"].progress_apply(extract_named_entities, args=(allowed_entity_types,))

df["hallucinated_entities"] = df.apply(difference_entities, axis=1, args=("model_entities", "dialog_entities"))

df["missed_entities"] = df.apply(difference_entities, axis=1, args=("dialog_entities", "model_entities"))

df["num_missed_entities"] = df["missed_entities"].apply(lambda x: len(x))

df["num_hallucinated_entities"] = df["hallucinated_entities"].apply(lambda x: len(x))

df.to_csv("llm_summary_entities.csv", index=False)

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
    



x_col = "model_output"

y_col = "summary"

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

df.to_csv("llm_summary_evaluation_results.csv", index=False)

