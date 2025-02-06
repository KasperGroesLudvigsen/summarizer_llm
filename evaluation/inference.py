"""
TODO: Evaluate on google/Synthetic-Persona-Chat
"""
from datasets import load_dataset
from vllm import LLM, SamplingParams
from codecarbon import EmissionsTracker
from datasets import load_dataset
import torch 
from ..utils import make_prompt


# push llm outputs to
push_to = "ThatsGroes/LLM-summary-evaluation"

temperature = 0.2
top_p = 0.8
max_tokens = 8192
num_samples = 100 # 10000

model_ids = ["ThatsGroes/SmolLM2-360M-Instruct-summarizer", "ThatsGroes/SmolLM2-1.7B-Instruct-summarizer", "HuggingFaceTB/SmolLM2-1.7B-Instruct", "HuggingFaceTB/SmolLM2-360M-Instruct"]
#model_ids = ["ThatsGroes/SmolLM2-360M-Instruct-summarizer"]

# Load dataset1
dataset_id1 = "ThatsGroes/synthetic-dialog-summaries-processed-clean-chatml" #"ThatsGroes/synthetic-dialog-summaries-processed-clean"

dataset = load_dataset(dataset_id1, split="test") #dataset.map(formatting_prompts_func, batched = True, fn_kwargs={"tokenizer": tokenizer, "col_to_process": col_to_process})

dataset = dataset.shuffle(seed=90201)

# only take from 5000 and up because the first 5000 were used as evaluation in training
dataset = dataset.select(range(5000,num_samples))

prompts = [make_prompt(prompt) for prompt in dataset["dialog"]]

dataset = dataset.add_column("prompt", prompts)

system_prompt = "You are an expert in summarizing texts. Extract and present the main key point of the input text in one short sentence, including essential details like dates, locations, persons and organizations if necessary."

model_suffixes = []

for model_id in model_ids:

    print(f"\nWill run inference with: {model_id}\n")

    model_suffix = model_id.split("/")[-1]

    model_suffixes.append(model_suffix)

    sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens)

    llm = LLM(model=model_id, max_seq_len_to_capture=max_tokens)

    print("Starting inference..")
    tracker = EmissionsTracker(project_name=model_id, measure_power_secs=1)
    tracker.start()
    outputs = llm.chat(dataset["prompt"], sampling_params)
    emissions = tracker.stop()

    responses = [output.outputs[0].text for output in outputs]

    dataset = dataset.add_column(f"summary_by_{model_suffix}", responses)

    torch.cuda.empty_cache()

    # torch.cuda.empty_cache does not properly free up memory
    del llm 

try:
    new_order = ["summary"]
    new_order.extend(model_suffixes)
    new_order.extend(["dialog", "system_prompt", "messages", "text"])
    dataset = dataset.select_columns(new_order)
except:
    print("Did not rearrange column order")


try:
    dataset.push_to_hub(push_to)

except Exception as e:
    print(f"Could not push to hub due to exception:\n{e}\nWill save to disk")
    dataset.save_to_disk("output_data")