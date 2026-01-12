# Robust LLM Retriever Eval

This repository contains the code for our paper: *"On the Robustness of LLM-Based Dense Retrievers: A Systematic Analysis of Generalizability and Stability"*.

We use this codebase to evaluate the robustness of state-of-the-art (SOTA) open-source LLM-based dense retrievers.

[//]: # (To reproduce the results reported in the paper, please follow the steps below.)

We are cleaning our code and will publish it as soon as possible.

# Datasets

The datasets used in the experiments are from the [Extended Beir Datasets](https://github.com/liyongkang123/extended_beir_datasets) code repository. The datasets are stored in the datasets/ folder. These datasets will download automatically when you run the code.

# Environment Setup

You can set up the environment using either conda or pip.

- **Conda**: Use `environment.yml`
- **Pip**: Use `requirements.txt`

We recommend using pip for installation. In addition to common deep learning and NLP libraries such as PyTorch and Transformers, this project requires the following dependencies:

- [BEIR](https://github.com/beir-cellar/beir) - Benchmarking IR library
- [Gensim](https://radimrehurek.com/gensim/) - Topic modeling and word embeddings
- [rpy2](https://rpy2.github.io/) - R language interface for statistical analysis

### Quick Start

```bash
# Using pip (recommended)
pip install -r requirements.txt

# Using conda
conda env create -f environment.yml
conda activate <env_name>
```

### R Language Support

For statistical analysis (e.g., Linear Mixed Models), R language support is required via `rpy2`:

```bash
conda install conda-forge::rpy2
```

# Reproducing the Main Results

## RQ1: How generalizable are LLM-based dense retrievers across diverse conditions?

1. **Run evaluation on all models and datasets:**
   ```bash
   bash scripts/eval.sh
   ```
   This script evaluates all models on all datasets at the query level. Embeddings are saved to `./embeddings/clean/`, and query-level results are stored in the `output/` folder. This produces the data for **Table 7**.

2. **Classify queries by type:**
   ```bash
   python nf_cats_all_datasets.py
   ```
   This script classifies all queries across datasets. The classification results are saved to `data/nf_cats_all_datasets.csv`. It then combines query categories with the scores from Step 1 to generate `data/nf_cats_all_datasets_with_scores.csv`.

3. **Prepare data for LMM analysis:**
   
   Run all cells in `nf_cats_all_datasets_scores_analysis and LMM data preprocess.ipynb` to prepare data for Linear Mixed Model (LMM) analysis. This produces the data for **Table 8** and generates `data/df_long_all_model_scores.csv`.

4. **Run LMM and ANOVA analysis:**
   ```bash
   python LMM_ANOVA_Analysis_R.py
   ```
   This script performs LMM analysis and produces the data for **Table 1** and **Table 2**.

## RQ2: How stable are LLM-based dense retrievers under perturbations?

1. **Generate query variations:**
   ```bash
   bash scripts/augment_query_variation_text.sh
   ```
   Query variations are saved to `output_attack/attacked_text/query/`.

2. **Evaluate query variations:**
   ```bash
   bash scripts/eval_attack_query.sh
   ```

3. **Generate adversarial documents (corpus poisoning):**
   
   Use the [HotFlip Corpus Poisoning](https://github.com/liyongkang123/hotflip_corpus_poisoning) repository to generate adversarial documents for all LLM-based retrievers. Place the generated documents in `output_attack/attacked_text/document/`.

4. **Evaluate corpus poisoning:**
   ```bash
   bash scripts/eval_attack_document.sh
   ```

5. **Generate statistics and figures:**
   ```bash
   python eval_attack_data_statistics.py
   ```
   This produces **Figure 1** and **Table 3** for RQ2.

## RQ3: What factors are predictive of robustness in LLM-based retrievers?

1. **Generate embedding analysis figures:**
   ```bash
   python draw_isoscore.py
   ```
   - Call `model_embedding_isoscore_and_avg_cosine_and_query_variations()` to generate **Figure 2**
   - Call `model_embedding_isoscore_and_avg_cosine_and_asr_20_horizontal()` to generate **Figure 3**

2. **Compute average cosine similarity:**
   
   The average cosine similarity for each model on each dataset is computed in `draw_cosine.py` (lines 67-128). Results are saved to `data/average_cosine_similarity_data_100k.json`.

3. **Compute IsoScore:**
   
   The IsoScore for each model on each dataset is computed using the `IsoScore_numpy` function in `draw_isoscore.py`. Results are saved to `data/isoscore_data_100k.json`.

4. **Compute spectral norm:**
   ```bash
   python draw_spectral_norm.py
   ```
   This computes the average spectral norm for each model.
