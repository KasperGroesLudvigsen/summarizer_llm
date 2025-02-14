"""
Generalized script for running inference with several models as part of evaluation

The script iterates over a list of HF model IDs and runs inference on the passed
model input. Returns a datasets.Dataset object

"""
from datasets import load_dataset, Dataset, concatenate_datasets
import pandas as pd
import re
from vllm import LLM, SamplingParams
from codecarbon import EmissionsTracker
from copy import deepcopy
from torch.cuda import empty_cache, is_available, get_device_capability
from torch import device
from typing import List, Dict
from utils import make_prompt

def is_bf16_supported():
    """Checks if the GPU supports bfloat16 precision."""
    if not is_available():
        return False

    capability = get_device_capability(device("cuda"))
    
    # Ampere (8.0+) and later architectures support bfloat16
    return capability[0] >= 8

print("BF16 supported:", is_bf16_supported())


def run_inference(model_input: List[List[Dict[str, str]]], # prompts made with utils.make_prompt()
                  model_ids: List[str], 
                  temperature: float, 
                  top_p: float, 
                  max_tokens: int) -> Dataset:
    
    """
    Iterates over a list of HF model IDs and runs inference with each using VLLM.
    Returns a Dataset object with three columns: model_input, model_output and model (the HF ID)
    
    """


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

        subset = subset.add_column("model_output", [output.outputs[0].text for output in outputs])

        results.append(subset)

        empty_cache()

        # torch.cuda.empty_cache does not properly free up memory
        del llm 

    results = concatenate_datasets(results)

    return results
