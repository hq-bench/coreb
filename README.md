# CoREB: Code Retrieval and Reranking Benchmark

[![PyPI version](https://img.shields.io/pypi/v/coreb)](https://pypi.org/project/coreb/)
[![Downloads](https://img.shields.io/pypi/dm/coreb)](https://pypi.org/project/coreb/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![Dataset](https://img.shields.io/badge/HuggingFace-hq--bench%2Fcoreb-yellow)](https://huggingface.co/datasets/hq-bench/coreb)
[![Reranker](https://img.shields.io/badge/HuggingFace-coreb--code--reranker-yellow)](https://huggingface.co/hq-bench/coreb-code-reranker)
[![arXiv](https://img.shields.io/badge/arXiv-2605.04615-b31b1b)](https://arxiv.org/abs/2605.04615)

**CoREB** is a graded-relevance benchmark for evaluating code retrieval and reranking models across three tasks:

| Task | Query | Target | Example |
|------|-------|--------|---------|
| **Text-to-Code** (T2C) | Natural language description | Code solution | "Find the longest substring without repeating characters" → Python solution |
| **Code-to-Code** (C2C) | Code in language A | Equivalent code in language B | Python solution → Java translation |
| **Code-to-Text** (C2T) | Code snippet | Problem description | Python solution → problem statement |

## Key Features

- **Graded relevance**: 3-level qrel scheme (rel=2: positive, rel=1: hard negative, rel=0: irrelevant) — hard negatives are same-problem distractors that penalize nDCG when retrieved above true positives
- **5 programming languages**: Python, C++, Java, Go, Ruby
- **Problem-disjoint train/test splits**: v202602 (training) and v202603 (testing) cover non-overlapping contest windows
- **Two-stage evaluation**: benchmarks both retrieval (embedding models) and reranking (cross-encoders)
- **Drop-in evaluation**: compatible with standard IR evaluation (pytrec\_eval) with `relevance_level=2`

## Installation

```bash
pip install coreb
```

For HuggingFace model support:
```bash
pip install coreb[hf]        # transformers backend
pip install coreb[gemini]    # Google Gemini API
pip install coreb[all]       # everything
```

## Quick Start

### Load the Dataset

```python
from datasets import load_dataset

# Load v202603 release (latest)
code_corpus = load_dataset("hq-bench/coreb", "code_corpus", split="release_v2603")
text_corpus = load_dataset("hq-bench/coreb", "text_corpus", split="release_v2603")

# Load task-specific queries and qrels
t2c_queries = load_dataset("hq-bench/coreb", "text2code_queries", split="release_v2603")
t2c_qrels = load_dataset("hq-bench/coreb", "text2code_qrels", split="release_v2603")

print(f"Code corpus: {len(code_corpus)} documents")
print(f"T2C queries: {len(t2c_queries)} queries, {len(t2c_qrels)} qrels")
```

### Run Evaluation

```python
from coreb_runner.benchmark import (
    load_jsonl,
    convert_corpus_to_coir_format,
    convert_queries_to_coir_format,
    convert_qrels_to_coir_format,
    EvaluateRetrieval,
    DenseRetrievalExactSearch,
    create_model_wrapper,
)

# Load data (from local JSONL files or convert from HF datasets)
corpus = convert_corpus_to_coir_format(load_jsonl("code_corpus.jsonl"))
queries = convert_queries_to_coir_format(load_jsonl("text2code_queries.jsonl"))
qrels = convert_qrels_to_coir_format(load_jsonl("text2code_qrels.jsonl"))

# Create model wrapper
model = create_model_wrapper("jinaai/jina-embeddings-v3", model_type="huggingface")

# Run retrieval + evaluation
retriever = DenseRetrievalExactSearch(model, batch_size=64)
evaluator = EvaluateRetrieval(retriever, k_values=[1, 3, 5, 10])
results = evaluator.retrieve(corpus, queries)
ndcg, _map, recall, precision = evaluator.evaluate(qrels, results, evaluator.k_values)

print(f"nDCG@10: {ndcg['NDCG@10']:.4f}")
print(f"Recall@10: {recall['Recall@10']:.4f}")
```

### Evaluation with Graded Relevance

CoREB uses `relevance_level=2` — only rel>=2 items count as relevant for binary metrics (Recall, MAP, Precision). Hard negatives (rel=1) penalize nDCG by occupying top ranks with zero gain but do not inflate Recall/MRR.

```python
# The EvaluateRetrieval class handles this automatically:
# - rel=1 (hard negatives) are zeroed out for nDCG computation
# - relevance_level=2 is set for pytrec_eval binary metrics
print(f"Relevance threshold: {EvaluateRetrieval.RELEVANCE_LEVEL}")  # 2
```

## Dataset Structure

Available on HuggingFace: [`hq-bench/coreb`](https://huggingface.co/datasets/hq-bench/coreb)

**8 configs** x **2 splits** (`release_v2602`, `release_v2603`):

| Config | v2603 Rows | Description |
|--------|-----------|-------------|
| `code_corpus` | 1,744 | Code solutions (5 languages, 2 generator models) |
| `text_corpus` | 875 | Problem descriptions (175 original + 700 LLM noise) |
| `text2code_queries` | 1,123 | T2C queries (canonical, full, search subtasks) |
| `text2code_qrels` | 5,950 | T2C relevance judgments (2,814 pos + 3,136 hard neg) |
| `code2code_queries` | 278 | C2C queries (cross-language) |
| `code2code_qrels` | 1,457 | C2C relevance judgments (623 pos + 834 hard neg) |
| `code2text_queries` | 1,200 | C2T queries (canonical, full, match subtasks) |
| `code2text_qrels` | 4,610 | C2T relevance judgments (820 pos + 2,650 hard neg) |

## Benchmark Results (v202603, nDCG@10)

| Rank | Model | Avg | T2C | C2C | C2T |
|------|-------|-----|-----|-----|-----|
| 1 | GemEmb-2 | 0.639 | 0.434 | 0.698 | 0.784 |
| 2 | C2LLM-7B | 0.623 | 0.443 | 0.659 | 0.766 |
| 3 | jina-code-1.5b | 0.607 | 0.414 | 0.671 | 0.735 |
| 4 | C2LLM-0.5B | 0.604 | 0.430 | 0.657 | 0.725 |
| 5 | jina-code-0.5b | 0.596 | 0.386 | 0.677 | 0.725 |
| 6 | F2LLM-4B | 0.547 | 0.407 | 0.500 | 0.735 |
| 7 | Qwen3-Emb-4B | 0.495 | 0.390 | 0.392 | 0.704 |
| 8 | F2LLM-1.7B | 0.485 | 0.383 | 0.383 | 0.690 |
| 9 | Qwen3-Emb-0.6B | 0.443 | 0.349 | 0.384 | 0.597 |
| 10 | F2LLM-0.6B | 0.439 | 0.344 | 0.334 | 0.641 |
| 11 | Qwen3-Emb-8B | 0.428 | 0.328 | 0.320 | 0.635 |

## CoREB-Reranker

[`hq-bench/coreb-code-reranker`](https://huggingface.co/hq-bench/coreb-code-reranker) is a code reranker fine-tuned from Qwen3-Reranker-4B via LoRA. It is the first reranker to achieve consistent gains across all three code search tasks.

**Reranking delta (nDCG@10 %):**

| Reranker | T2C | C2C | C2T |
|----------|-----|-----|-----|
| **CoREB-Reranker** | **+1.1** | **+5.1** | **+0.8** |

## Tutorials

Interactive Colab notebooks to get started:

| Notebook | Description |
|----------|-------------|
| [01 — Download & Analyze Data](notebooks/01_download_and_analyze_data.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hq-bench/coreb/blob/main/notebooks/01_download_and_analyze_data.ipynb) | Load the dataset from HuggingFace, explore corpus/queries/qrels, and analyze statistics |
| [02 — Run Evaluation](notebooks/02_run_evaluation.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hq-bench/coreb/blob/main/notebooks/02_run_evaluation.ipynb) | Two-stage evaluation: dense retrieval + reranking with CoREB-Reranker |

## Citation

```bibtex
@article{xue2025coreb,
  title   = {Beyond Retrieval: A Multitask Benchmark and Model for Code Search},
  author  = {Xue, Siqiao and Liao, Zihan and Qin, Jin and Zhang, Ziyin and Mu, Yixiang and Zhou, Fan and Yu, Hang},
  journal = {arXiv preprint arXiv:2605.04615},
  year    = {2025},
  url     = {https://arxiv.org/abs/2605.04615}
}
```

## License

This project is licensed under the Apache License 2.0 — see [LICENSE](LICENSE) for details.
