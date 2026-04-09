"""
train_qlora.py — QLoRA instruction fine-tuning for Qwen2.5-1.5B-Instruct.

Hardware target: NVIDIA GTX 1650 (4 GB VRAM)
Method:         QLoRA = 4-bit quantisation + LoRA adapters
Trainer:        TRL SFTTrainer with assistant-only loss

Usage:
    python src/train_qlora.py

Outputs:
    outputs/adapter/   — LoRA adapter weights (load on top of base model)
    outputs/eval/      — training_log.json (loss curve data)
"""

import json
import os
import sys
import warnings
import logging
import datetime

# ── 🚨 CRITICAL FIX: PyTorch must check CUDA *BEFORE* bitsandbytes/transformers 
# are imported. On Windows, importing PEFT/Transformers can eagerly lock the GPU context. 
# If VRAM is highly fragmented, it silently falls back to CPU and permanently breaks CUDA for this script.
import torch

def _check_hardware_early():
    if not torch.cuda.is_available():
        print("⚠️ WARNING: GPU not found natively by PyTorch before imports.")
        return "CPU"
    return f"GPU ({torch.cuda.get_device_name(0)})"

DEVICE_NAME = _check_hardware_early()

from datasets import load_from_disk
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import transformers
from trl import SFTConfig, SFTTrainer

# Allow imports from src/ when running from project root
sys.path.insert(0, os.path.dirname(__file__))

from config import (
    ADAPTER_DIR,
    DATA_PROCESSED,
    EOS_TOKEN,
    EVAL_DIR,
    LORA_CONFIG,
    MODEL_ID,
    TRAIN_CONFIG,
)

# Suppress noisy-but-harmless FutureWarnings from bitsandbytes / PyTorch
warnings.filterwarnings("ignore", message=".*_check_is_size will be removed.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*use_reentrant parameter should be passed explicitly.*", category=UserWarning)


def setup_logging():
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join("logs", f"training_debug_{timestamp}.log")

    logging.basicConfig(
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8")
        ]
    )

    # Force HuggingFace to output rich info logs to our files
    transformers.logging.set_verbosity_info()
    transformers.logging.add_handler(logging.FileHandler(log_file, encoding="utf-8"))

    return logging.getLogger(__name__), log_file


def main():
    logger, log_file = setup_logging()
    logger.info(f"Started training script. Detailed logs writing to: {log_file}")

    # ── HARDWARE CHECK: Log GPU/CPU ──────────────────────────────────────────
    if DEVICE_NAME == "CPU":
        logger.warning("⚠️ WARNING: GPU not found or CUDA is not configured properly.")
        logger.warning("Model training is starting on CPU. This will be very slow.")
    else:
        logger.info(f"✅ GPU hardware verified! Model training is starting on: {DEVICE_NAME}")

    # ── Create output directories ──────────────────────────────────────────────
    logger.info("Initializing output directories...")
    os.makedirs(ADAPTER_DIR, exist_ok=True)
    os.makedirs(EVAL_DIR, exist_ok=True)
    os.makedirs("offload", exist_ok=True)

    # ── 1. Load dataset ────────────────────────────────────────────────────────
    logger.info("Loading dataset...")
    ds = load_from_disk(DATA_PROCESSED)
    logger.info(f"Train samples: {len(ds['train'])} | Validation samples: {len(ds['validation'])}")

    # IMPORTANT for GTX 1650: subset the data for your first run.
    # To run a ~30 minute test, keep this at 1000. For overnight training, COMMENT THIS OUT.
    ds["train"]      = ds["train"].select(range(5000))
    ds["validation"] = ds["validation"].select(range(1000))

    # ── 2. Load tokenizer ──────────────────────────────────────────────────────
    logger.info(f"Loading tokenizer: {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"   # right-padding during training

    # ── 3. Load base model in 4-bit ───────────────────────────────────────────
    logger.info("Loading base model in 4-bit (QLoRA)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,  # NO BF16 on GTX 1650
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,  # CRITICAL: Override Qwen's default BF16 which crashes Turing GPUs
        device_map="auto",
        offload_folder="offload",
    )

    # ── 4. Prepare model for k-bit training ───────────────────────────────────
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )
    model.config.use_cache = False

    # ── 5. Auto-detect LoRA target modules ────────────────────────────────────
    # Qwen2.5 uses standard transformer projection layer names.
    candidate_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"]
    module_names = {name for name, _ in model.named_modules()}
    target_modules = [m for m in candidate_modules
                      if any(n.endswith(m) for n in module_names)]

    if not target_modules:
        logger.error("No standard LoRA target modules found!")
        raise ValueError("Inspect model.named_modules() and set target_modules manually.")
    logger.info(f"Auto-configured LoRA target modules: {target_modules}")

    # ── 6. LoRA config ────────────────────────────────────────────────────────
    lora_config = LoraConfig(
        **LORA_CONFIG,
        target_modules=target_modules,
    )

    # ── 7. Training config ────────────────────────────────────────────────────
    sft_config = SFTConfig(
        output_dir=ADAPTER_DIR,
        eos_token=EOS_TOKEN,
        gradient_checkpointing_kwargs={"use_reentrant": False},  # modern PyTorch best practice
        **TRAIN_CONFIG,
    )

    # ── 8. Trainer ────────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    # ── 9. Train ──────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STARTING FINE-TUNING")
    logger.info("=" * 60)
    logger.info(f"Epochs             : {TRAIN_CONFIG['num_train_epochs']}")
    logger.info(f"Batch size         : {TRAIN_CONFIG['per_device_train_batch_size']}")
    logger.info(f"Grad accumulation  : {TRAIN_CONFIG['gradient_accumulation_steps']}")
    effective_batch = TRAIN_CONFIG['per_device_train_batch_size'] * TRAIN_CONFIG['gradient_accumulation_steps']
    logger.info(f"Effective batch    : {effective_batch}")
    logger.info(f"Learning rate      : {TRAIN_CONFIG['learning_rate']}")
    logger.info("=" * 60)

    # ── CRITICAL GTX 1650 FIX: Purge hidden bfloat16 tensors ─────────────────
    # The SFTTrainer sometimes auto-initializes LoRA weights or positional embeddings
    # to BFloat16 if the original Qwen config requested it, regardless of torch_dtype.
    # We MUST downcast them to float32 before training or the Turing GPU crashes.
    for name, param in trainer.model.named_parameters():
        if param.dtype == torch.bfloat16:
            param.data = param.data.to(torch.float32)

    train_result = trainer.train()

    # ── 10. Save adapter + tokenizer ──────────────────────────────────────────
    logger.info("Saving adapter and tokenizer...")
    trainer.save_model(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)
    logger.info(f"Adapter saved successfully to: {ADAPTER_DIR}")

    # ── 11. Save training log (for plotting later) ────────────────────────────
    log_path = os.path.join(EVAL_DIR, "training_log.json")
    history = trainer.state.log_history
    with open(log_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info(f"Training evaluations JSON saved to: {log_path}")

    logger.info("=" * 60)
    logger.info("✅ TRAINING COMPLETE!")
    logger.info(f"Total steps : {train_result.global_step}")
    logger.info(f"Final loss  : {train_result.training_loss:.4f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
