"""
prepare_data.py — Download, clean, and format the Bitext Customer Support
dataset for instruction fine-tuning with Qwen2.5.

Cleaning pipeline:
  1. Remove empty / null rows
  2. Replace all {{placeholder}} template markers with realistic values
  3. Deduplicate by user message (keep first occurrence only)
  4. Filter out examples that are too long for the token budget
  5. Filter out extremely short / low-quality user messages
  6. Convert to system/user/assistant chat format
  7. Split 80 / 10 / 10 and save both processed + raw versions
"""

import re
import random
from datasets import load_dataset, DatasetDict

random.seed(42)

DATASET_ID = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"

SYSTEM_PROMPT = (
    "You are a helpful customer support assistant. "
    "Answer clearly, politely, and concisely. "
    "If needed, ask one short follow-up question."
)

# ── Placeholder replacement mappings ────────────────────────────────────
# These turn the unusable {{template}} markers into natural-sounding text
# so the model never learns to output raw placeholders.

# Static replacements (always the same value)
STATIC_PLACEHOLDERS = {
    "{{Customer Support Phone Number}}": "1-800-555-0199",
    "{{Customer Support Email}}":       "support@company.com",
    "{{Customer Support Hours}}":       "Monday-Friday, 9 AM to 6 PM EST",
    "{{Website URL}}":                  "www.company.com/support",
    "{{App}}":                          "our mobile app",
    "{{App Name}}":                     "our mobile app",
    "{{Company}}":                      "our company",
    "{{Company Name}}":                 "our company",
    "{{Live Chat}}":                    "our live chat on the website",
    "{{Customer Support Channel}}":     "our customer support page",
    "{{Customer Account Page}}":        "your account settings page",
    "{{Password Reset URL}}":           "www.company.com/reset-password",
    "{{Subscription Plan}}":            "your current plan",
    "{{Billing Address}}":              "your billing address on file",
    "{{Shipping Address}}":             "your shipping address on file",
    "{{Currency Symbol}}":              "$",
    "{{Warranty Period}}":              "12 months",
}

# Dynamic replacements (random realistic values per occurrence)
DYNAMIC_POOLS = {
    "{{Delivery City}}": [
        "New York", "Los Angeles", "Chicago", "Houston", "Phoenix",
        "London", "Toronto", "Mumbai", "Sydney", "Berlin",
    ],
    "{{Product Name}}": [
        "the product", "your item", "the ordered product",
    ],
    "{{Order Number}}": None,       # generated randomly
    "{{Tracking Number}}": None,    # generated randomly
    "{{Invoice Number}}": None,     # generated randomly
    "{{Account Email}}": [
        "your registered email address",
    ],
    "{{Account Number}}": None,     # generated randomly
    "{{Refund Amount}}": None,      # generated randomly
    "{{Delivery Date}}": [
        "within 3-5 business days", "within 5-7 business days",
        "by the estimated delivery date",
    ],
    "{{Promo Code}}": [
        "SAVE10", "WELCOME20", "LOYALTY15", "FIRSTORDER",
    ],
}


def _generate_dynamic_value(placeholder_name: str) -> str:
    """Generate a realistic random value for a dynamic placeholder."""
    if placeholder_name in ("{{Order Number}}",):
        return f"ORD-{random.randint(100000, 999999)}"
    elif placeholder_name in ("{{Tracking Number}}",):
        return f"TRK-{random.randint(1000000, 9999999)}"
    elif placeholder_name in ("{{Invoice Number}}",):
        return f"INV-{random.randint(10000, 99999)}"
    elif placeholder_name in ("{{Account Number}}",):
        return f"ACC-{random.randint(10000, 99999)}"
    elif placeholder_name in ("{{Refund Amount}}",):
        return f"${random.choice([9.99, 14.99, 24.99, 39.99, 49.99, 79.99]):.2f}"

    # If it has a pool, pick from it
    pool = DYNAMIC_POOLS.get(placeholder_name)
    if pool:
        return random.choice(pool)

    # Fallback — should not reach here often
    return "the relevant details"


def clean_placeholders(text: str) -> str:
    """Replace every {{...}} placeholder with a realistic value."""
    # 1) Static replacements first (exact match, fast)
    for placeholder, value in STATIC_PLACEHOLDERS.items():
        text = text.replace(placeholder, value)

    # 2) Dynamic replacements (known patterns)
    for placeholder in DYNAMIC_POOLS:
        while placeholder in text:
            text = text.replace(placeholder, _generate_dynamic_value(placeholder), 1)

    # 3) Catch ANY remaining {{...}} we didn't explicitly handle
    #    Use a function so we can add smart spacing
    def _fallback_replace(match):
        # Check what surrounds the placeholder to pick a natural replacement
        return "your reference number"

    text = re.sub(r"\{\{[^}]+\}\}", _fallback_replace, text)

    # 4) Post-clean: fix artifacts from replacement
    text = re.sub(r"  +", " ", text)              # collapse double spaces
    text = re.sub(r" ([.,;:!?])", r"\1", text)     # remove space before punctuation
    text = text.strip()

    return text


def clean_row(example):
    """Clean both instruction and response text for a single row."""
    instruction = clean_placeholders(example["instruction"].strip())
    response    = clean_placeholders(example["response"].strip())
    return {
        **example,
        "instruction": instruction,
        "response": response,
    }


def is_quality_row(example) -> bool:
    """Filter out rows that are empty, too short, or too long."""
    instruction = example["instruction"]
    response    = example["response"]

    # Must exist and not be empty
    if not instruction or not response:
        return False
    if not instruction.strip() or not response.strip():
        return False

    # User message too short to be meaningful (< 5 characters)
    if len(instruction.strip()) < 5:
        return False

    # Filter by total character length as a proxy for token budget.
    # With max_length=256 tokens ≈ 1024 chars (system + user + assistant + template overhead).
    # We use 950 chars to leave headroom for chat-template special tokens.
    total_chars = len(SYSTEM_PROMPT) + len(instruction) + len(response)
    if total_chars > 950:
        return False

    return True


def to_messages(example):
    """Convert a cleaned row into the chat-message format for SFTTrainer."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": example["instruction"].strip()},
        {"role": "assistant", "content": example["response"].strip()},
    ]
    return {"messages": messages}


def deduplicate(dataset):
    """Remove duplicate user messages, keeping the first occurrence."""
    seen = set()
    keep_indices = []
    for i, example in enumerate(dataset):
        key = example["instruction"].strip().lower()
        if key not in seen:
            seen.add(key)
            keep_indices.append(i)
    return dataset.select(keep_indices)


def main():
    print("=" * 60)
    print("Step 1/7: Loading raw dataset from Hugging Face...")
    raw = load_dataset(DATASET_ID, split="train")
    print(f"  Loaded {len(raw)} rows")

    # ── Step 2: Basic filtering (nulls, empties) ──
    print("\nStep 2/7: Removing null / empty rows...")
    raw = raw.filter(
        lambda x: x["instruction"] is not None
        and x["response"] is not None
        and x["instruction"].strip() != ""
        and x["response"].strip() != ""
    )
    print(f"  After null filter: {len(raw)} rows")

    # ── Step 3: Clean placeholders ──
    print("\nStep 3/7: Cleaning {{placeholder}} template markers...")
    raw = raw.map(clean_row)
    # Verify cleaning worked
    sample_check = sum(1 for ex in raw if "{{" in ex["response"])
    print(f"  Remaining rows with placeholders: {sample_check}")

    # ── Step 4: Shuffle ──
    print("\nStep 4/7: Shuffling dataset...")
    raw = raw.shuffle(seed=42)

    # ── Step 5: Deduplicate ──
    print("\nStep 5/7: Removing duplicate user messages...")
    before_dedup = len(raw)
    raw = deduplicate(raw)
    print(f"  Removed {before_dedup - len(raw)} duplicates")
    print(f"  After dedup: {len(raw)} rows")

    # ── Step 6: Quality / length filter ──
    print("\nStep 6/7: Filtering by quality and token budget...")
    before_filter = len(raw)
    raw = raw.filter(is_quality_row)
    print(f"  Removed {before_filter - len(raw)} rows (too short or too long)")
    print(f"  After quality filter: {len(raw)} rows")

    # ── Step 7: Split, format, and save ──
    print("\nStep 7/7: Splitting 80/10/10 and converting to chat format...")
    split_80_20   = raw.train_test_split(test_size=0.2, seed=42)
    split_80_10_10 = split_80_20["test"].train_test_split(test_size=0.5, seed=42)

    train_raw = split_80_20["train"]
    val_raw   = split_80_10_10["train"]
    test_raw  = split_80_10_10["test"]

    # Convert to chat-message format
    train = train_raw.map(to_messages, remove_columns=train_raw.column_names)
    val   = val_raw.map(to_messages,   remove_columns=val_raw.column_names)
    test  = test_raw.map(to_messages,  remove_columns=test_raw.column_names)

    processed = DatasetDict({"train": train, "validation": val, "test": test})

    # Save processed dataset (chat format)
    processed.save_to_disk("data/processed/customer_support_chat")

    # Save raw splits (original columns preserved — useful for evaluation)
    raw_splits = DatasetDict(
        {"train": train_raw, "validation": val_raw, "test": test_raw}
    )
    raw_splits.save_to_disk("data/raw/customer_support_raw")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("DONE — Dataset processing complete!")
    print("=" * 60)
    print(f"  Train:      {len(train):,} examples")
    print(f"  Validation: {len(val):,} examples")
    print(f"  Test:       {len(test):,} examples")
    print(f"  Total:      {len(train) + len(val) + len(test):,} examples")
    print()
    print("  Saved processed dataset -> data/processed/customer_support_chat")
    print("  Saved raw splits        -> data/raw/customer_support_raw")
    print()

    # Show a sample to verify
    print("Sample cleaned example:")
    print("-" * 40)
    import json
    print(json.dumps(train[0], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
