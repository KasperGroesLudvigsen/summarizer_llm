from inference import run_inference
from datasets import load_dataset
from utils import make_prompt
import pandas as pd
import re

def remove_person_tags(text: str) -> str:
    """Removes #Person<number># patterns from text."""
    if pd.isna(text) or not isinstance(text, str):  # Handle NaN or non-string values
        return text

    text = re.sub(r"#Person\d+#[:\s]*", "", text)
    return text.replace("\n", " ")


if __name__ == "__main__":

    # push llm outputs to
    push_to = "ThatsGroes/LLM-summary-evaluation-dialogsum"

    temperature = 0.2
    top_p = 0.8
    max_tokens = 8192//4

    model_ids = ["ThatsGroes/SmolLM2-360M-Instruct-summarizer", "ThatsGroes/SmolLM2-1.7B-Instruct-summarizer", "HuggingFaceTB/SmolLM2-1.7B-Instruct", "HuggingFaceTB/SmolLM2-360M-Instruct"]

    df = load_dataset("knkarthick/dialogsum")

    df = df["test"].to_pandas()

    df["clean_dialog"] = df["dialogue"].apply(remove_person_tags)

    dialogs = df.drop_duplicates(subset=["clean_dialog"])["clean_dialog"].to_list()

    user_prompt = "Write one sentence that summarizes this conversation, emphasizing any meetings, persons or places mentioned in the conversation. **Conversation:** {model_input}"

    system_prompt = "You are an expert in summarizing texts. Extract and present the main key point of the input text in one short sentence, including essential details like dates, locations, persons and organizations if necessary."

    prompts = [make_prompt(user_prompt.format(model_input=dialog), system_prompt) for dialog in dialogs]

    prompts = prompts[:5]
    
    results = run_inference(model_input=prompts,
                  model_ids=model_ids, 
                  temperature=temperature, 
                  top_p=top_p, 
                  max_tokens=max_tokens)

    try:
        results.push_to_hub(push_to)

    except Exception as e:
        print(f"Could not push to hub due to exception:\n{e}\nWill save to disk")
        results.save_to_disk("output_data")

    results = results.to_pandas()

    results.to_csv("model_output_dialogsum.csv", index=False)
