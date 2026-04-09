"""
config.py — Single source of truth for all project constants.

Every script imports from here. Change a value once, it updates everywhere.
"""

# ── Model ──────────────────────────────────────────────────────────────────────
MODEL_ID   = "Qwen/Qwen2.5-1.5B-Instruct"
EOS_TOKEN  = "<|im_end|>"          # Qwen2.5 chat-format end token

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_PROCESSED = "data/processed/customer_support_chat"
DATA_RAW       = "data/raw/customer_support_raw"
ADAPTER_DIR    = "outputs/adapter"
MERGED_DIR     = "outputs/merged"
EVAL_DIR       = "outputs/eval"

# ── System Prompt ──────────────────────────────────────────────────────────────
# THIS MUST MATCH EXACTLY between prepare_data.py, train, infer, and evaluate.
# It is the instruction the model reads at the start of every conversation.
SYSTEM_PROMPT = (
    "You are a helpful customer support assistant. "
    "Answer clearly, politely, and concisely. "
    "If needed, ask one short follow-up question."
)

# ── Training Hyper-parameters (optimised for GTX 1650, 4 GB VRAM) ─────────────
TRAIN_CONFIG = dict(
    max_length                 = 256,   # fits within 4 GB; 33% of data > 256 tokens
    packing                    = False,
    num_train_epochs           = 1,     # start with 1; scale after it works
    per_device_train_batch_size= 1,     # MUST be 1 on 4 GB
    per_device_eval_batch_size = 1,     # CRITICAL FIX: trainer defaults to 8, which causes OOM crashes
    gradient_accumulation_steps= 16,    # simulates effective batch size of 16
    learning_rate              = 2e-4,
    warmup_ratio               = 0.03,
    lr_scheduler_type          = "cosine",
    logging_steps              = 10,
    eval_strategy              = "epoch",
    save_strategy              = "epoch",
    save_total_limit           = 1,
    bf16                       = False,  # GTX 1650 is Turing — NO BF16 support
    fp16                       = True,   # use FP16 instead
    gradient_checkpointing     = True,   # critical memory saver
    report_to                  = "none",
    remove_unused_columns      = True,
    assistant_only_loss        = False,  # disabled because Qwen's template lacks TRL's generation tag
)

# ── LoRA Hyper-parameters ──────────────────────────────────────────────────────
LORA_CONFIG = dict(
    r            = 8,      # rank — lower = less memory. 8 is safe for 4 GB
    lora_alpha   = 16,     # scaling factor; keep = 2 × r
    lora_dropout = 0.05,
    bias         = "none",
    task_type    = "CAUSAL_LM",
)

# ── Evaluation ─────────────────────────────────────────────────────────────────
EVAL_SAMPLE_SIZE = 50    # how many test examples to evaluate (50 is fast, 200 is thorough)
MAX_NEW_TOKENS   = 200   # max tokens to generate per answer during eval/infer
