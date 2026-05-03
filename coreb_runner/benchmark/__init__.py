"""
CoREB (Code Retrieval Embedding Benchmark) Package

This package contains utilities for CoREB benchmark:
- models: Embedding model wrappers
- data_loader: Data loading and format conversion utilities
- evaluation: Retrieval evaluation with graded relevance (relevance_level=2)
"""

from coreb_runner.benchmark.models import (
    BaseEmbeddingWrapper,
    HuggingFaceWrapper,
    GeminiEmbeddingWrapper,
    create_model_wrapper
)

from coreb_runner.benchmark.data_loader import (
    load_jsonl,
    convert_corpus_to_coir_format,
    convert_queries_to_coir_format,
    convert_qrels_to_coir_format,
    load_custom_dataset,
    group_queries_by_subtask,
    filter_queries_by_subtask
)

_evaluation_import_error = None
try:
    from coreb_runner.benchmark.evaluation import (
        CoREBEvaluation,
        EvaluateRetrieval,
        DenseRetrievalExactSearch,
    )
except Exception as e:
    _evaluation_import_error = e

__all__ = [
    # Models
    'BaseEmbeddingWrapper',
    'HuggingFaceWrapper',
    'GeminiEmbeddingWrapper',
    'create_model_wrapper',
    # Data Loaders
    'load_jsonl',
    'convert_corpus_to_coir_format',
    'convert_queries_to_coir_format',
    'convert_qrels_to_coir_format',
    'load_custom_dataset',
    'group_queries_by_subtask',
    'filter_queries_by_subtask',
]

if _evaluation_import_error is None:
    __all__.extend(
        [
            # Evaluation
            'CoREBEvaluation',
            'EvaluateRetrieval',
            'DenseRetrievalExactSearch',
        ]
    )

