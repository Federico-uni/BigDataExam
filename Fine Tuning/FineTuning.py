import os
import torch
from huggingface_hub import login
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

# =========================================================
# CONFIG
# =========================================================
HF_TOKEN="token_HF"
if not HF_TOKEN:
    raise ValueError("HF_TOKEN not present.")

login(token=HF_TOKEN)

MODEL_ID = "meta-llama/Llama-3.1-8B"
OUT_DIR = "./ft_llama31_8b_lora"
TOKENIZER_DIR = os.path.join(OUT_DIR, "tokenizer")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(TOKENIZER_DIR, exist_ok=True)

# =========================================================
# DATASET
# =========================================================
cartella = "path"

TRAIN_FILES = []
for nome_file in os.listdir(cartella):
    percorso = os.path.join(cartella, nome_file)
    if os.path.isfile(percorso):
        TRAIN_FILES.append(percorso)

if not TRAIN_FILES:
    raise ValueError(f"Nessun file trovato nella cartella: {cartella}")

print("File di training trovati:")
for f in TRAIN_FILES:
    print("-", f)

ds = load_dataset("parquet", data_files={"train": TRAIN_FILES})["train"]

def preprocess(ex):
    return {
        "prompt": ex["prompt"],
        "completion": ex["text"]
    }

ds = ds.map(preprocess, remove_columns=ds.column_names)

# =========================================================
# CONTROLLO GPU / BF16
# =========================================================
use_gpu = torch.cuda.is_available()
device = "cuda" if use_gpu else "cpu"

use_bf16 = False
use_fp16 = False

print(f"\nCUDA disponibile: {use_gpu}")
if use_gpu:
    print(f"GPU rilevata: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version torch: {torch.version.cuda}")

    # bf16 solo se realmente supportato
    if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
        use_bf16 = True
        use_fp16 = False
        compute_dtype = torch.bfloat16
        print("bf16 not supported: using bfloat16")
    else:
        use_bf16 = False
        use_fp16 = True
        compute_dtype = torch.float16
        print("bf16 not supported: using float16")
else:
    compute_dtype = torch.float32
    print("No GPU : using CPU")

# =========================================================
# TOKENIZER
# =========================================================
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    token=HF_TOKEN,
    use_fast=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.save_pretrained(TOKENIZER_DIR)
print(f"Tokenizer salvato in: {TOKENIZER_DIR}")

# =========================================================
# MODELLO
# =========================================================
if use_gpu:
    print("\nModel loading in 4-bit...")

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        attn_implementation="sdpa", # "flash_attention_2"
        token=HF_TOKEN,
        quantization_config=bnb,
        device_map="auto",
    )

    optimizer_name = "paged_adamw_8bit"
    use_gradient_checkpointing = True

else:
    print("\nModel loading in CPU...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        token=HF_TOKEN,
    )
    model.to(device)

    optimizer_name = "adamw_torch"
    use_gradient_checkpointing = False

model.config.use_cache = False

# =========================================================
# LORA
# =========================================================
peft = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

# =========================================================
# TRAIN CONFIG
# =========================================================
args = SFTConfig(
    output_dir=OUT_DIR,
    packing=False,
    per_device_train_batch_size=2,
    max_length=1024, # 2048
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    num_train_epochs=5,
    logging_steps=10,
    save_steps=1000, # 200
    save_total_limit=2,
    optim=optimizer_name,
    bf16=use_bf16,
    fp16=use_fp16,
    gradient_checkpointing=use_gradient_checkpointing,
    report_to="none",
)

# =========================================================
# TRAINER
# =========================================================
trainer = SFTTrainer(
    model=model,
    train_dataset=ds,
    peft_config=peft,
    processing_class=tokenizer,
    args=args,
)

# =========================================================
# TRAIN
# =========================================================
trainer.train()

# =========================================================
# SALVATAGGIO FINALE
# =========================================================
trainer.model.save_pretrained(OUT_DIR)
tokenizer.save_pretrained(TOKENIZER_DIR)

print("\nFinished.")
print("Model saved in:", OUT_DIR)
print("Tokenizer saved in:", TOKENIZER_DIR)
