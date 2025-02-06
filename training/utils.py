from datasets import Dataset

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


def make_prompt(user_prompt: str, system_prompt: str=None) -> dict:

    if system_prompt:

        return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

    return [{"role": "user", "content": user_prompt}]

