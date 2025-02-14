"""
Prepare knkarthick/dialogsum for evaluation. E.g. remove diarization
"""
from datasets import load_dataset, Dataset, concatenate_datasets
import pandas as pd
import re
from vllm import LLM, SamplingParams
from codecarbon import EmissionsTracker
from copy import deepcopy
from torch.cuda import empty_cache, is_available, get_device_capability
from typing import List, Dict
from utils import make_prompt
import torch

def is_bf16_supported():
    """Checks if the GPU supports bfloat16 precision."""
    if not is_available():
        return False

    device = torch.device("cuda")
    capability = get_device_capability(device)
    
    # Ampere (8.0+) and later architectures support bfloat16
    return capability[0] >= 8

# Example usage
print("BF16 supported:", is_bf16_supported())

def remove_person_tags(text: str) -> str:
    """Removes #Person<number># patterns from text."""
    if pd.isna(text) or not isinstance(text, str):  # Handle NaN or non-string values
        return text

    text = re.sub(r"#Person\d+#[:\s]*", "", text)
    return text.replace("\n", " ")

def run_inference(model_input: List[List[Dict[str, str]]], # prompts made with utils.make_prompt()
                  model_ids: List[str], 
                  temperature: float, 
                  top_p: float, 
                  max_tokens: int):


    sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens)

    dataset = Dataset.from_dict({"model_input": model_input})

    results = []

    dtype = "auto" if is_bf16_supported() else "half"

    for model_id in model_ids:

        subset = deepcopy(dataset) 

        subset = subset.add_column("model", [model_id for i in range(len(subset))])

        llm = LLM(model=model_id, max_seq_len_to_capture=max_tokens, dtype=dtype)

        print("Starting inference..")
        tracker = EmissionsTracker(project_name=model_id, measure_power_secs=1)
        tracker.start()
        outputs = llm.chat(subset["model_input"], sampling_params)
        emissions = tracker.stop()

        subset = subset.add_column("llm_output", [output.outputs[0].text for output in outputs])

        results.append(subset)

        empty_cache()

        # torch.cuda.empty_cache does not properly free up memory
        del llm 

    results = concatenate_datasets(results)

    return results


if __name__ == "__main__":

    # push llm outputs to
    push_to = "ThatsGroes/LLM-summary-evaluation-dialogsum"

    temperature = 0.2
    top_p = 0.8
    max_tokens = 8192//4

    model_ids = ["ThatsGroes/SmolLM2-360M-Instruct-summarizer", "ThatsGroes/SmolLM2-1.7B-Instruct-summarizer", "HuggingFaceTB/SmolLM2-1.7B-Instruct", "HuggingFaceTB/SmolLM2-360M-Instruct"]

    dataset = load_dataset("knkarthick/dialogsum")

    df = dataset["test"].to_pandas()

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
