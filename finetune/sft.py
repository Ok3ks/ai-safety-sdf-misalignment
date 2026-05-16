import torch
from transformers import AutoTokenizer, AutoModelForImageTextToText, BitsAndBytesConfig
from peft import LoraConfig
import torch
from trl import SFTConfig, SFTTrainer
from peft import PeftModel
import wandb

#peft wandb trl torch transformers pandas inspect_petri inspect_ai pydantic_core pydantic_ai open_ai vllm

def init_model(model_id = "google/gemma-4-E2B"):
    """
    
    Initialize Gemma
    Returns Model and Tokenizer
    
    """
    # Check if GPU benefits from bfloat16
    if torch.cuda.get_device_capability()[0] >= 8:
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float16

    # Define model init arguments
    model_kwargs = dict(
        dtype=torch_dtype,
        device_map="auto", # Let torch decide how to load the model
    )

    # BitsAndBytesConfig: Enables 4-bit quantization to reduce model size/memory usage
    model_kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=model_kwargs['dtype'],
        bnb_4bit_quant_storage=model_kwargs['dtype'],
    )

    model = AutoModelForImageTextToText.from_pretrained(model_id, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(f"{model_id}-it") #use instruction tuned variant

    return model, tokenizer

def peft_config():
    """Returns a Peft config for initialisation"""

    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=16,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
        modules_to_save=["lm_head", "embed_tokens"], # make sure to save the lm_head and embed_tokens as you train the special tokens
        ensure_weight_tying=True,
    )

    return peft_config

def sft_config(model, dataset, tokenizer, peft_config, output_dir):
    """Returns a configured trainer object"""

    args = SFTConfig(
        output_dir= output_dir,         # directory to save and repository id
        max_length=1024,                         # max length for model and packing of the dataset 
        #find average length of dataset, critique
        num_train_epochs=5,                     # number of training epochs
        per_device_train_batch_size=1,          # batch size per device during training
        optim="adamw_torch_fused",              # use fused adamw optimizer
        logging_steps=10,                       # log every 10 steps
        save_strategy="epoch",                  # save checkpoint every epoch
        eval_strategy="epoch",                  # evaluate checkpoint every epoch
        learning_rate=2e-4,                     # learning rate
        fp16=True if model.dtype == torch.float16 else False,   # use float16 precision
        bf16=True if model.dtype == torch.bfloat16 else False,   # use bfloat16 precision
        max_grad_norm=0.3,                 # max gradient norm based on QLoRA paper
        lr_scheduler_type="linear",           # use constant learning rate scheduler
        push_to_hub=True,                           # push model to hub
        report_to="wandb",  
        run_name=f"{output_dir}-run-1",              # report metrics to tensorboard
        dataset_kwargs={
            "add_special_tokens": False, # Template with special tokens
            "append_concat_token": True, # Add EOS token as separator token between examples
        }
    )  

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    return trainer, args

def free_memory(model, trainer):
    """Free Memory of Machine"""

    del model
    del trainer
    torch.cuda.empty_cache()


if __name__ == "__main__":
    from datasets import load_dataset
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_id", choices=["google/gemma-4-E2B", "google/gemma-4-E4B", "google/gemma-4-26B-A4B", "google/gemma-4-31B"])
    parser.add_argument("-p", "--path")
    parser.add_argument("-d", "--dataset_name")

    #pass config file for hyperparameter tuning

    args = parser.parse_args()
    model, tokenizer = init_model(model_id=args.model_id)

    pft_config = peft_config()
    dataset = load_dataset(
        "json",
        data_files=args.path,
        split="train"
    )

    dataset = dataset.train_test_split(test_size=0.3, seed=42)
    dataset["train"] = dataset["train"].shuffle(seed=42).select(range(1000))
    dataset["test"] = dataset["test"].shuffle(seed=42).select(range(1000))

    dataset = dataset.map(lambda x: {"messages": x["messages"]["messages"]})

    run_id = f"{args.model_id.split("/")[-1]}-{args.dataset_name}"
    trainer, args = sft_config(
        model, dataset, tokenizer, pft_config, 
        f"{run_id}"
    )

    run = wandb.init(
        entity="freelanceokeks",
        project="SDF misalignment",
    )

    trainer.train()
    # run.log_model("cake_bake")
    trainer.save_model("artifact/models")