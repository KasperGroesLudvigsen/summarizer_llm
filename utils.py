from datasets import Dataset, load_dataset

def formatting_prompts_func(samples: Dataset, tokenizer, col_to_process: str="messages"):
    """
    Create a new column called "text" by applying the tokenizer's chat template to `col_to_process`
    Args:

        samples: A Huggingface-style dataset
        tokenizer: an Unsloth tokenizer
        col_to_process: The column containing the samples in the dataset. The entries in this column will be converted to the chat template format specified in the tokenizer.
    """
    convos = samples[col_to_process]
    texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
    return { "text" : texts, }

def get_data():

    data = load_dataset("ThatsGroes/LLM-summary-evaluation", trust_remote_code=True)

    df = data["test"].to_pandas()

    id_vars = ["summary", "dialog", "system_prompt", "messages", "text", "prompt"]

    df = df.melt(id_vars=id_vars, var_name="model", value_name="model_output")

    df["model"] = df["model"].str.replace("summary_by_", "")

    df.dropna(subset=["summary", "model_output"], inplace=True)

    df["dialog"] = df["dialog"].apply(lambda x: x.split("\n\n **Conversation:** \n\n")[-1].strip())

    return df

def make_prompt(user_prompt: str, system_prompt: str=None) -> dict:

    if system_prompt:

        return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

    return [{"role": "user", "content": user_prompt}]

