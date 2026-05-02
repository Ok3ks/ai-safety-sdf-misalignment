import torch
from transformers import AutoTokenizer, AutoModelForImageTextToText, BitsAndBytesConfig
from peft import LoraConfig
import torch
from trl import SFTConfig, SFTTrainer
from peft import PeftModel

# Hugging Face model id
model_id = "google/gemma-4-E2B" # @param ["google/gemma-4-E2B","google/gemma-4-E4B"] {"allow-input":true}

def init_model():
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
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-4-E2B-it")

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

def sft_config(model, dataset, tokenizer):
    """Returns a configured trainer object"""

    args = SFTConfig(
        output_dir="gemma-text-to-sql",         # directory to save and repository id
        max_length=512,                         # max length for model and packing of the dataset
        num_train_epochs=3,                     # number of training epochs
        per_device_train_batch_size=1,          # batch size per device during training
        optim="adamw_torch_fused",              # use fused adamw optimizer
        logging_steps=10,                       # log every 10 steps
        save_strategy="epoch",                  # save checkpoint every epoch
        eval_strategy="epoch",                  # evaluate checkpoint every epoch
        learning_rate=5e-5,                     # learning rate
        fp16=True if model.dtype == torch.float16 else False,   # use float16 precision
        bf16=True if model.dtype == torch.bfloat16 else False,   # use bfloat16 precision
        max_grad_norm=0.3,                      # max gradient norm based on QLoRA paper
        lr_scheduler_type="constant",           # use constant learning rate scheduler
        push_to_hub=True,                           # push model to hub
        report_to="tensorboard",                # report metrics to tensorboard
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

def merge_model(args: SFTConfig):
    """Merge Peft Model"""

    model = AutoModelForImageTextToText.from_pretrained(model_id, low_cpu_mem_usage=True)

    peft_model = PeftModel.from_pretrained(model, args.output_dir)
    merged_model = peft_model.merge_and_unload()
    merged_model.save_pretrained("merged_model", safe_serialization=True, max_shard_size="2GB")

    processor = AutoTokenizer.from_pretrained(args.output_dir)
    processor.save_pretrained("merged_model")
    
    return model

def free_memory():
    """Free Memory of Machine"""

    del model
    del trainer
    torch.cuda.empty_cache()