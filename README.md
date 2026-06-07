# Detecting LLM Hallucinations via Hidden-State Probing

CS 455 term project. A single notebook (`HALUPROBE.ipynb`) that extracts hidden states from
**Mistral-7B-v0.1** on factual QA, trains probes (logistic regression / linear SVM / MLP) across a
layer × position grid under three prompting strategies, and evaluates hallucination detection
(primary metric: **AUROC**), including a TriviaQA out-of-distribution check.

## Contents
- `HALUPROBE.ipynb` — the full pipeline (data → extraction → probing → evaluation).
- `report.pdf` — final report.
- `README.md` — this file.

## Requirements
- **A CUDA GPU.** The model is loaded in 4-bit (`bitsandbytes`), so a free-tier Colab/Kaggle **T4
  (16 GB)** is enough. 
- **Python 3.10+** with `torch`, `transformers`, `datasets`, `bitsandbytes` (>= 0.46.1),
  `huggingface_hub`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `tqdm`. On
  Colab/Kaggle only `bitsandbytes` needs installing — the first two cells handle it.

## Setup
1. Open `HALUPROBE.ipynb` in **Google Colab** (recommended) or Kaggle and turn the **GPU runtime on**.

2. **Cache location.** On Colab the first cell mounts Google Drive and `CACHE_DIR` points to
   `/content/drive/MyDrive/HALLUPROBE/cache`. If you are **not** on Colab, edit `CACHE_DIR` in the
   config cell to any writable local folder and skip the Drive-mount cell.

## Data — nothing to download by hand
Both datasets are **public** and stream automatically from the Hugging Face Hub the first time you
run the data cells; they cache into the local `data/` folder (`DATA_DIR`). You do **not** download,
unzip, or place any files manually — just run the cells.

- **HaluEval-QA** — `load_dataset("pminervini/HaluEval", "qa_samples")`. ~10k QA items. The notebook
  keeps the gold (non-hallucinated) answers, de-duplicates by question, and samples a fixed
  **3,000-question** subset with `seed=42`. Mistral then answers each question and we label its
  output by substring-match against the gold answer.
- **TriviaQA (OOD)** — `load_dataset("mandarjoshi/trivia_qa", "rc.wikipedia")`.  This config is
  **large** (several GB, downloaded in ~26 shards) — allow disk space and a few minutes on the first
  run. We sample 3,000 questions and label against the gold answer or its aliases.



## How to run
Run the cells **top to bottom**:
1. **Installs + imports + config cell.** Set the model, layers, positions, seed, and paths here.
   Change `CACHE_DIR` if not on Colab; lower `BATCH_SIZE` (16 → 4–8) if you hit out-of-memory.
2. **Data loading + labelling** — downloads HaluEval and TriviaQA and assigns binary labels.
3. **Model load (4-bit) + hidden-state extraction.** Activations are cached to `CACHE_DIR` as
   `mistralai_Mistral-7B-v0.1_halueval_<strategy>.pkl`. The notebook runs all three strategies:
   `zero_shot`, `factuality_aware`, `chain_of_thought`.
4. **Probe training** + the layer × position ablation heatmaps.
5. **Output-space baselines**, the **cross-strategy comparison**, and the **TriviaQA OOD evaluation**.

## Reproducibility / skipping the GPU
- Everything is seeded (`SEED = 42`) and decoding is greedy, so runs are deterministic.
- **Extraction is the only GPU step.** It writes activation pickles to `CACHE_DIR`; with
  `FORCE_EXTRACT = False` (default) later runs reuse them, so the probing and evaluation cells rerun
  on **CPU in minutes**. If you place the cached `.pkl` files in `CACHE_DIR`, the entire analysis
  reproduces **without a GPU**.
- Set `FORCE_EXTRACT = True` only to regenerate activations from scratch.
