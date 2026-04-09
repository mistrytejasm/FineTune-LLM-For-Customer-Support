# 🧠 Qwen2.5 Customer Support AI (Fine-Tuned)

This repository contains a complete, end-to-end Machine Learning pipeline to fine-tune a Large Language Model (**Qwen2.5-1.5B**) on a 15,000-row customer support dataset. 

This project was heavily heavily engineered to execute entirely locally on a **4GB VRAM Consumer GPU (NVIDIA GTX 1650)** using QLoRA 4-bit quantisation, aggressive gradient checkpointing, and PCIe CPU offloading architectures.

## ✨ Results Summary
The fine-tuned model completely learned the company's de-escalation persona and formatting instructions.
* **ROUGE-1:** Improved from `0.34` to `0.50` (+44%)
* **BLEU:** Improved from `0.05` to `0.19` (+280%)
* **Perplexity:** Dropped from `4.8` to `2.97` 

**Angry Customer:** *"I want to edit my f\*\*\*ing user, will you help me?"*
* **Base Model (Refusal):** *"I'm sorry, but I can't assist with that..."*
* **Fine-Tuned Model (Empathetic De-escalation):** *"We're here for you! I'm fully aware of your frustration in trying to edit your user information. Rest assured, we are committed to assisting you..."*

## 🚀 How to Run the Inference (Plug-and-Play)
Because the heavy original dataset and log checkpoints are ignored by git, this repository is highly lightweight. The custom fine-tuned weights (`outputs/adapter`) are only **36 Megabytes**.

When you clone this repo, you can immediately talk to the fine-tuned model. The Python script will automatically download the required 3GB base model to your cache and inject the 36MB custom adapter block into it:

1. Clone the repository: `git clone https://github.com/mistrytejasm/FineTune-LLM-For-Customer-Support.git`
2. Create and activate a virtual environment.
3. Install dependencies: `pip install -r requirements.txt`
4. Chat with the finished model:
```bash
python src/infer.py
```

---

## 🛠️ Pipeline Architecture (Step-by-Step)
If you wish to re-run the entire pipeline from the ground up, follow this exact sequence of scripts:

### 1. Data Preparation
```bash
python src/prepare_data.py
```
* **What it does:** Downloads the `bitext/Bitext-customer-support-llm-chatbot-training-dataset` from Hugging Face. Cleans the data, applies a strict conversational prompt template (`<|im_start|>system...`), and serialises the 15,000+ interactions into Apache Arrow arrays.

### 2. QLoRA Training
```bash
python src/train_qlora.py
```
* **What it does:** The main training engine. It loads Qwen2.5 in strictly 4-bit NormalFloat format. A strict script-sweep guarantees all tensors initialize in `float32/fp16` to block `bfloat16` hardware crashes on older NVIDIA architectures. Configured explicitly for `Batch = 1` with 16 Accumulation steps to survive within 4GB VRAM limits.
* **Output:** Saves the newly trained LoRA weights to `outputs/adapter/adapter_model.safetensors`.

### 3. Metric Evaluation
```bash
python src/run_eval.py
```
* **What it does:** Sequentially loads the Base model and asks it 50 random test questions. It unloads the model from RAM, loads the New Fine-Tuned model, and asks it the exact same 50 questions. It then runs a multi-threaded evaluation algorithm mapping BLEU and ROUGE-n similarities against the "Ground Truth" company responses. 
* **Output:** Saves the algorithmic comparison numbers to `outputs/eval/metrics.json`.

### 4. Report Generation
```bash
python src/report.py
```
* **What it does:** Reads the `metrics.json` file and the `training_log.json` to generate robust graphical plots (Bar charts, Heatmaps, Loss Curves) using `matplotlib`. It compiles all data into a comprehensive Markdown presentation.
* **Output:** Generates PNG charts in `outputs/eval/plots/` and `outputs/eval/report.md`.

---

### Project Configuration
All hyper-parameters (learning rates, batch sizes, inference tokens, max lengths, system prompts) are centrally managed inside a single source of truth: `src/config.py`. Edit parameters there to immediately update all surrounding scripts globally.
