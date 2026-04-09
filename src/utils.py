"""
utils.py — Shared model loading and text generation utilities.

Used by infer.py, evaluate.py, and report.py.
Centralises the fix for the BatchEncoding crash in modern transformers.
"""

import gc
import os
import warnings

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from config import ADAPTER_DIR, EOS_TOKEN, MAX_NEW_TOKENS, MODEL_ID, SYSTEM_PROMPT

# Suppress noisy-but-harmless FutureWarnings from bitsandbytes / PyTorch internals
warnings.filterwarnings(
    "ignore",
    message=".*_check_is_size will be removed.*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message=".*use_reentrant parameter should be passed explicitly.*",
    category=UserWarning,
)


def _bnb_config() -> BitsAndBytesConfig:
    """4-bit NF4 quantisation config — required for GTX 1650 (4 GB VRAM)."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,  # GTX 1650 has NO BF16 support
    )


def load_tokenizer() -> AutoTokenizer:
    """Load the Qwen2.5 tokenizer and ensure a pad token is set."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"   # required for batch generation with causal LMs
    return tokenizer


def load_model(use_adapter: bool = False) -> AutoModelForCausalLM:
    """
    Load the base model in 4-bit, optionally attaching the LoRA adapter.

    Args:
        use_adapter: If True, load the fine-tuned LoRA adapter on top.

    Returns:
        Model ready for inference (eval mode, on device).
    """
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=_bnb_config(),
        torch_dtype=torch.float16,
        device_map="auto",
        offload_folder="offload",   # spill to CPU RAM if VRAM is full
    )

    if use_adapter:
        if not os.path.isdir(ADAPTER_DIR):
            raise FileNotFoundError(
                f"Adapter not found at '{ADAPTER_DIR}'. "
                "Run src/train_qlora.py first."
            )
        model = PeftModel.from_pretrained(model, ADAPTER_DIR)

    model.eval()
    return model


def build_model(use_adapter: bool = False):
    """Convenience wrapper — returns (model, tokenizer) tuple."""
    tokenizer = load_tokenizer()
    model     = load_model(use_adapter=use_adapter)
    return model, tokenizer


def free_model(model) -> None:
    """
    Fully release a model from GPU and CPU memory.
    Call this between loading base and fine-tuned models during evaluation
    to avoid OOM crashes on the GTX 1650.
    """
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


@torch.no_grad()
def generate_answer(
    model,
    tokenizer,
    user_text: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
    system_prompt: str | None = None,
) -> str:
    """
    Generate a response for a user message.

    CRASH FIX: Modern transformers returns a BatchEncoding dict from
    apply_chat_template, NOT a raw tensor. We use return_dict=True and
    unpack with **inputs so model.generate receives the correct types.

    Args:
        model:          Loaded causal LM (base or fine-tuned).
        tokenizer:      Matching tokenizer.
        user_text:      The user's question / message.
        max_new_tokens: Maximum tokens to generate.
        system_prompt:  Override the default system prompt (optional).

    Returns:
        Generated assistant response as a plain string.
    """
    prompt = system_prompt or SYSTEM_PROMPT
    messages = [
        {"role": "system",  "content": prompt},
        {"role": "user",    "content": user_text},
    ]

    # ── CRASH FIX ────────────────────────────────────────────────────────────
    # apply_chat_template with return_dict=True returns a dict with
    # 'input_ids' and 'attention_mask'. We move both to the model device
    # and unpack with **inputs — model.generate never sees a bare tensor.
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[-1]
    # ─────────────────────────────────────────────────────────────────────────

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,                    # greedy — deterministic, reproducible
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.convert_tokens_to_ids(EOS_TOKEN),
    )

    # Slice off the prompt tokens — only return the newly generated part
    generated = output_ids[0][input_len:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()
