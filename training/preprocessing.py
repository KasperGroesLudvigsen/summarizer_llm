"""
Convert the raw dataset into a format that can be used to train SmolLLM
"""
import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv
import os 
from huggingface_hub import login


load_dotenv()

token = os.getenv("HF_TOKEN")

login(token, add_to_git_credential=True)
#csv_path_eval = "summaries5.csv"

def format_example(example: dict) -> dict:
    return {
        "messages": [
            {"content": example["system_prompt"], "role": "system"},
            {"content": example["dialog"], "role": "user"},
            {"content" : example["summary"], "role" : "assistant"}
            ]
            }

def create_dialog_summary_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Produce columns "summary" and "dialog". 

    In kobprof/skolegpt-instruct, the column "question" corresponds to our "dialog"
    column, and the "response" columns corresponds to our "summary" column
    """
    system_prompt = "You are an expert in summarizing texts. Extract and present the main key point of the input text in one short sentence, including essential details like dates, locations, persons and organizations if necessary."

    # Create "summary" and "dialog" columns
    df['summary'] = df['response'].apply(lambda x: x.split("**Summary")[-1])

    df['summary'] = df['summary'].str.replace(r'[\n\*\:]', '', regex=True).str.strip()

    df['dialog'] = df['response'].apply(lambda x: x.split("**Summary")[0])

    df["dialog"] = df["dialog"].str.replace(r'[\n\*\:]', '', regex=True).str.strip()

    df["dialog"] = df["dialog"].apply(generate_prompt)

    df["system_prompt"] = system_prompt

    return df


def generate_prompt(dialog: str):
    prompt = (
        f"Write one sentence that summarizes this conversation, emphasizing any meetings, persons or places mentioned in the conversation. \n\n **Conversation:** \n\n {dialog}"
    )
    return prompt

def load_data():
    df = load_dataset("ThatsGroes/synthetic-dialog-summaries-raw", token=token)
    df = df["train"].to_pandas()

    csv_path_raw = "summaries.csv"
    csv_path_train_ready = "dialog_summaries_ready.csv"

    #df = pd.read_csv(csv_path_raw)

    df = create_dialog_summary_cols(df)

    keep = ["system_prompt", "dialog", "summary"]
    df = df.drop([col for col in df.columns if col not in keep], axis=1)

    df.to_csv(csv_path_train_ready, index=False)

    #df.dialog[0]

    data_files = {"train": csv_path_train_ready} #, "test": "test.csv"}

    dataset = load_dataset("csv", data_files=csv_path_train_ready)

    return dataset


def main():
    """
    This is the last post processing step.
    """

    dataset = load_data()

    dataset = dataset.map(format_example)

    dataset = dataset.train_test_split(test_size=0.05)

    dataset.push_to_hub("ThatsGroes/synthetic-dialog-summaries-processed")


main()

"""
[ 
    { "content": "Extract and present the main key point of the input text in one very short sentence, including essential details like dates, locations, persons and organizations if necessary.", "role": "system" }, 
    { "content": "Hi Sarah,\n\nI hope you're doing well! I wanted to reach out because I've been struggling with a student in my class who is significantly behind in reading comprehension. I remember you mentioning some effective strategies during our last conversation, and I was wondering if you could share some resources or tips that might help me support this student better.\n\nAny advice would be greatly appreciated! Let me know if you have time to chat further about this.\n\nBest,\nEmily", "role": "user" }, 
    { "content": "Emily is seeking advice on strategies for a struggling reader in her class.", "role": "assistant" } 
]
"""

