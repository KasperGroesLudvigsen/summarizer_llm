"""
Evaluate a list of models on a dataset

To DO: Calculate and saved tokens per second according to: https://github.com/vllm-project/vllm/issues/4968
"""

from datasets import load_dataset
import pandas as pd
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from codecarbon import EmissionsTracker
import argparse
from huggingface_hub import login
from datasets import load_dataset, concatenate_datasets
from copy import copy
import torch 
import os
import random 


def make_prompt(example: dict) -> dict:
    topic = random.choice(topics)
    #prompt = f"""f{example["instruction"]}: **TEXT:** {example["text"]}"""

    prompt = f"Please write a text that could pass as a transcription of an everyday conversation between two or more people on the topic of: {topic}. Do not indicate speaker turns and do not use quotation marks. Just write the transcription as on long text. Then, write one sentence that summarizes the transcription, emphasizing any meeting, persons or places mentioned in the conversation"
    return {"prompt": [{"role": "user", "content": prompt}]}



token = os.getenv("HF_TOKEN") 

dataset_path = "mock_evaluation_data.csv"

dataset = load_dataset("csv", data_files=dataset_path)

dataset = dataset.map(make_prompt)

models = [
    "google/gemma-2-27b-it",
    "google/gemma-2-9b-it", 
    "ThatsGroes/Llama-3-8B-instruct-AI-Sweden-SkoleGPT",
    "ThatsGroes/Llama-3-8B-instruct-AI-Sweden-SkoleGPT",
    "AI-Sweden-Models/Llama-3-8B-instruct",
    "AI-Sweden-Models/Llama-3-8B-instruct-Q4_K_M-gguf",
    "mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated",
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "Qwen/Qwen2.5-Coder-32B-Instruct"
    ]


all_results = []

energy_use = []

for model in models:

    results = copy(dataset)

    tokenizer = AutoTokenizer.from_pretrained(model, token=token)

    sampling_params = SamplingParams(temperature=0.2, top_p=0.95, max_tokens=512, seed=123456789)

    llm = LLM(model=model, max_seq_len_to_capture=8000)

    # Log some GPU stats before we start inference
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(
        f"You're using the {gpu_stats.name} GPU, which has {max_memory:.2f} GB of memory "
        f"in total, of which {start_gpu_memory:.2f}GB has been reserved already."
    )

    tracker = EmissionsTracker()
    tracker.start()
    outputs = llm.chat(dataset["prompt"], sampling_params)
    emissions = tracker.stop()
    print(f"Emissions from generating queries with {model}:\n {emissions}")
    energy_consumption_kwh = tracker._total_energy.kWh  # Total energy in kWh
    print(f"Energy consumption from generating queries with {model}:\n {emissions}")

    responses = [output.outputs[0].text for output in outputs]

    results = results.add_column("summary", responses)

    results = results.add_column("model", [model for _ in range(len(results))])
    
    # number of tokens in the prompt and response. Used for calculcating kwh/token
    results = results.add_column("num_tokens_query", [len(tokenizer.encode(text, add_special_tokens=False)) for text in responses]) # [len(tokenizer.encode(text, add_special_tokens=False)) for text in responses]

    # each element in results["prompt"] is a list with a dictionary with two keys: "content" and "role"
    results = results.add_column("num_tokens_prompt", [len(tokenizer.encode(text[0]["content"], add_special_tokens=False)) for text in results["prompt"]])

    all_results.append(results)

    # Print some post inference GPU stats
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_inference = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    inference_percentage = round(used_memory_inference / max_memory * 100, 3)

    energy_use.append({
        "model" : model, 
        "energy_use_kwh" : energy_consumption_kwh, 
        "num_tokens_query" : sum(results["num_tokens_query"]), 
        "num_tokens_prompt" : sum(results["num_tokens_prompt"]),
        "num_tokens_total" : sum(results["num_tokens_query"]) + sum(results["num_tokens_prompt"]),
        "used_memory_inference": used_memory_inference
        })

    print(
        f"We ended up using {used_memory:.2f} GB GPU memory ({used_percentage:.2f}%), "
        f"of which {used_memory_inference:.2f} GB ({inference_percentage:.2f}%) "
        "was used for inference."
    )

    torch.cuda.empty_cache()

    # torch.cuda.empty_cache does not properly free up memory
    del llm 

energy_use = pd.DataFrame.from_records(energy_use)

energy_use["energy_per_token_total"] = energy_use["energy_use_kwh"] / energy_use["num_tokens_total"]

energy_use.to_csv("energy_use_per_model.csv", index=False)

final_dataset = concatenate_datasets(all_results)

print(f"Final dataset: \n {final_dataset}")

final_dataset.to_csv("summaries.csv")



from vllm import LLM, SamplingParams

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Create an LLM.
llm = LLM(model="facebook/opt-125m")
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)

print(outputs)


# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")