from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from datasets import load_dataset, Dataset, concatenate_datasets
import os
import yaml
import utils
import pathlib
from huggingface_hub import HfApi
import os
import wandb
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from dotenv import load_dotenv

load_dotenv()

token = os.getenv("HF_TOKEN")

wandb_key = os.getenv("WANDB_KEY")

if wandb_key:
    print("\n WANDB KEY SET SUCCESSFULLY\n")
else:
    print("\n ERROR: WANDB KEY **NOT** SET SUCCESSFULLY\n")


def train(config):    
    
    print(f"#### Starting a fine tuning run with the following configuration:\n\n{config}")

    wandb_config = config["wandb_config"]

    model_config = config["model_config"]

    lora_config = config["lora_config"]

    sft_config = config["sft_config"]
 
    #####################
    # Load model and tokenizer
    #####################

    max_seq_length = model_config["max_seq_length"] # Choose any! We auto support RoPE Scaling internally!
    dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = False # Use 4bit quantization to reduce memory usage. Can be False.
    model_id = model_config["model_id"]

    ### Load model and tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_id, # or choose "unsloth/Llama-3.2-1B-Instruct"
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit)

    # defaults to using chatml as per https://github.com/unslothai/unsloth/blob/main/unsloth/chat_templates.py
    tokenizer = get_chat_template(
        tokenizer
    )

    #####################
    # Load datasets
    #####################

    # Load dataset1
    dataset_id1 = "ThatsGroes/synthetic-dialog-summaries-processed-clean-chatml" #"ThatsGroes/synthetic-dialog-summaries-processed-clean"
    dataset_id2 = ("HuggingFaceTB/smoltalk", "smol-summarize")
    col_to_process = "messages" # the name of the column in the dataset containing the actual training data

    dataset = load_dataset(dataset_id1) #dataset.map(formatting_prompts_func, batched = True, fn_kwargs={"tokenizer": tokenizer, "col_to_process": col_to_process})

    # Load dataset2
    smoltalk = load_dataset(dataset_id2[0], dataset_id2[1])

    # Convert dataset2 to chatml format using the built in method `map` and our own `formatting_prompts_func` and pass arguments using `fn_kwargs`
    smoltalk = smoltalk.map(utils.formatting_prompts_func, batched = True, fn_kwargs={"tokenizer": tokenizer, "col_to_process": col_to_process})

    # Concat datasets to training data
    training_dataset = concatenate_datasets(
        [
            dataset["train"],
            smoltalk["train"]
        ]
        )

    # Create evaluation data
    eval_dataset = concatenate_datasets(
        [
            dataset["test"],
            smoltalk["test"]
        ]
        )
    
    eval_dataset = eval_dataset.shuffle(seed=90201)

    eval_dataset = eval_dataset.select(range(5000))

    #####################
    # Set up LoRA
    #####################

    print("\nSETTING UP LORA\n")

    rank = lora_config["r"]
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"]
    use_rslora = True
    lora_alpha = lora_config["lora_alpha"]


    model = FastLanguageModel.get_peft_model(
        model,
        r = rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128 # A lower rank means fewer parameters are being learned, leading to a more parameter-efficient method
        target_modules = target_modules,
        lora_alpha = lora_alpha, # standard practice seems to be to set this to 16. Mlabonne says its usually set to 1-2x the rank
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 3407,
        use_rslora = use_rslora,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )

    print("\nGOING TO ATTEMPT TO LOG INTO WANDB\n")
    wandb.login(key=wandb_key)
    os.environ["WANDB_PROJECT"]= wandb_config["wandb_project"] #"llm_dialog_summarizer"
    os.environ["WANDB_LOG_MODEL"] = "end"


    print("\nINSTANTIATING SFTTRAINER\n")

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = training_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field = sft_config["dataset_text_field"], # the key of the dict returned in formatting_prompts_func()
        max_seq_length = max_seq_length,
        dataset_num_proc = 4, # Number of processes to use for processing the dataset. Only used when packing = False
        packing = False, # Can make training 5x faster for short sequences.
        eval_strategy = "steps",
        eval_steps = 2000, # this should probably be much higher or eval datset should be significantly smaller
        args = TrainingArguments(
            per_device_train_batch_size = 8,
            gradient_accumulation_steps = 2,
            num_train_epochs = sft_config["epochs"], # Set this for 1 full training run. OpenAI's default is 4 https://community.openai.com/t/how-many-epochs-for-fine-tunes/7027/5
            learning_rate = sft_config["learning_rate"],
            warmup_steps=sft_config["warmup_steps"],
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = 100,
            optim = sft_config["optimizer"],
            weight_decay = sft_config["weight_decay"],
            lr_scheduler_type = sft_config["lr_scheduler_type"], # linear or cosine are most common
            seed = sft_config["seed"],
            output_dir = "outputs",
            report_to="wandb",
            run_name = wandb_config["run_name"], #f"Baseline-{model_id.split('/')[-1]}",
            save_steps=10000
        ),
    )

    print("\nBEGINNING TRAINING\n")

    trainer_stats = trainer.train()

    print(f"\nTrainer stats:\n {trainer_stats}\n")

    save_suffix = "-summarizer"
    hf_user = "ThatsGroes"

    new_model_id = f"{hf_user}/{model_id.split('/')[-1]}{save_suffix}"
    model.push_to_hub_merged(new_model_id, tokenizer, save_method="merged_16bit")

    api = HfApi()
    api.upload_file(
        path_or_fileobj=config["config_path"],
        path_in_repo="fine_tuning_configuration.yml",
        repo_id=new_model_id,
        repo_type="model",
    )
