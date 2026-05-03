"""
Code Embedding Data Loaders

Utilities for loading and converting data formats for code embedding evaluation.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict

try:
    from easyllm_kit.utils import get_logger

    logger = get_logger(__name__)
except Exception:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    logger = logging.getLogger(__name__)
    logger.warning("easyllm_kit logger unavailable; using stdlib logging fallback.")


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """
    Load JSONL file.
    
    Args:
        file_path: Path to JSONL file
    
    Returns:
        List of dictionaries
    """
    data = []
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON at line {line_num}: {e}")
                continue
    
    logger.info(f"Loaded {len(data)} entries from {file_path}")
    return data


def convert_corpus_to_coir_format(corpus: List[Dict[str, Any]]) -> Dict[str, Dict[str, str]]:
    """
    Convert our corpus format to CoIR format.
    
    Our format: {code_id: str, code: str, language: str, ...}
    CoIR format: {doc_id: {"text": str, ...}}
    
    Args:
        corpus: List of corpus documents in our format
    
    Returns:
        Dictionary in CoIR format
    """
    coir_corpus = {}
    
    for doc in corpus:
        # Extract document ID (try multiple possible keys)
        doc_id = doc.get('code_id') or doc.get('doc_id') or doc.get('id') or doc.get('text_id')
        if not doc_id:
            logger.warning(f"Document missing ID: {doc}")
            continue
        
        # Extract text content (try multiple possible keys)
        text = doc.get('code') or doc.get('text') or doc.get('content')
        if not text:
            logger.warning(f"Document {doc_id} missing text content")
            continue
        
        coir_corpus[doc_id] = {"text": text}
        
        # Optionally include metadata
        for key in ['language', 'model', 'problem_id']:
            if key in doc:
                coir_corpus[doc_id][key] = doc[key]
    
    logger.info(f"Converted {len(coir_corpus)} corpus documents to CoIR format")
    return coir_corpus


def convert_queries_to_coir_format(queries: List[Dict[str, Any]]) -> Dict[str, str]:
    """
    Convert our queries format to CoIR format.
    
    Our format: {query_id: str, query: str, ...}
    CoIR format: {query_id: query_text}
    
    Args:
        queries: List of queries in our format
    
    Returns:
        Dictionary in CoIR format
    """
    coir_queries = {}
    
    for query in queries:
        query_id = query.get('query_id')
        if not query_id:
            logger.warning(f"Query missing ID: {query}")
            continue
        
        # Extract query text (try multiple possible keys)
        query_text = (
            query.get('query') or 
            query.get('code') or 
            query.get('text') or 
            query.get('content')
        )
        if not query_text:
            logger.warning(f"Query {query_id} missing text content")
            continue
        
        coir_queries[query_id] = query_text
    
    logger.info(f"Converted {len(coir_queries)} queries to CoIR format")
    return coir_queries


def convert_qrels_to_coir_format(qrels: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    """
    Convert our qrels format to CoIR format.

    Our format: {query_id: str, doc_id: str, relevance: int}
    CoIR format: {query_id: {doc_id: score}}

    The v202601 dataset uses the field name "relevance" (not "score").
    We check both names so that legacy files still work, but "relevance"
    takes precedence when present.

    Args:
        qrels: List of qrels in our format

    Returns:
        Dictionary in CoIR format
    """
    coir_qrels = defaultdict(dict)

    for qrel in qrels:
        query_id = qrel.get('query_id')
        doc_id = qrel.get('doc_id')
        # v202601 uses "relevance"; fall back to "score" for legacy files.
        # Do NOT default to 1 — a missing relevance field is a data error.
        if 'relevance' in qrel:
            score = qrel['relevance']
        elif 'score' in qrel:
            score = qrel['score']
        else:
            logger.warning(f"Qrel missing both 'relevance' and 'score' fields: {qrel}")
            score = 1

        if not query_id or not doc_id:
            logger.warning(f"Qrel missing query_id or doc_id: {qrel}")
            continue

        coir_qrels[query_id][doc_id] = score
    
    total_qrels = sum(len(docs) for docs in coir_qrels.values())
    logger.info(f"Converted {total_qrels} qrels for {len(coir_qrels)} queries to CoIR format")
    
    return dict(coir_qrels)


def load_custom_dataset(
    corpus_path: str,
    queries_path: str,
    qrels_path: str
) -> Tuple[Dict[str, Dict], Dict[str, str], Dict[str, Dict[str, int]]]:
    """
    Load custom dataset and convert to CoIR format.
    
    Args:
        corpus_path: Path to corpus JSONL file
        queries_path: Path to queries JSONL file
        qrels_path: Path to qrels JSONL file
    
    Returns:
        Tuple of (corpus, queries, qrels) in CoIR format
    """
    logger.info("Loading custom dataset...")
    
    # Load raw data
    corpus_raw = load_jsonl(corpus_path)
    queries_raw = load_jsonl(queries_path)
    qrels_raw = load_jsonl(qrels_path)
    
    # Convert to CoIR format
    corpus = convert_corpus_to_coir_format(corpus_raw)
    queries = convert_queries_to_coir_format(queries_raw)
    qrels = convert_qrels_to_coir_format(qrels_raw)
    
    # Validation
    logger.info("Validating dataset...")
    _validate_dataset(corpus, queries, qrels)
    
    return corpus, queries, qrels


def _validate_dataset(
    corpus: Dict[str, Dict],
    queries: Dict[str, str],
    qrels: Dict[str, Dict[str, int]]
):
    """Validate dataset consistency."""
    # Check for missing query IDs in qrels
    missing_queries = set(qrels.keys()) - set(queries.keys())
    if missing_queries:
        logger.warning(f"Found {len(missing_queries)} queries in qrels but not in queries")
    
    # Check for missing doc IDs in corpus
    all_doc_ids_in_qrels = set()
    for docs in qrels.values():
        all_doc_ids_in_qrels.update(docs.keys())
    
    missing_docs = all_doc_ids_in_qrels - set(corpus.keys())
    if missing_docs:
        logger.warning(f"Found {len(missing_docs)} documents in qrels but not in corpus")
        logger.warning(f"Sample missing docs: {list(missing_docs)[:5]}")
    
    # Print stats
    logger.info(f"Dataset validation complete:")
    logger.info(f"  • Corpus: {len(corpus)} documents")
    logger.info(f"  • Queries: {len(queries)} queries")
    logger.info(f"  • Qrels: {len(qrels)} queries with relevance judgments")
    logger.info(f"  • Avg qrels per query: {sum(len(docs) for docs in qrels.values()) / len(qrels):.2f}")


def group_queries_by_subtask(queries: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Group queries by subtask.
    
    Args:
        queries: List of query dictionaries
    
    Returns:
        Dictionary mapping subtask names to lists of queries
    """
    grouped = defaultdict(list)
    
    for query in queries:
        subtask = query.get('subtask', 'unknown')
        grouped[subtask].append(query)
    
    logger.info(f"Grouped {len(queries)} queries into {len(grouped)} subtasks")
    for subtask, subtask_queries in grouped.items():
        logger.info(f"  • {subtask}: {len(subtask_queries)} queries")
    
    return dict(grouped)


def filter_queries_by_subtask(
    queries: List[Dict[str, Any]],
    qrels: List[Dict[str, Any]],
    subtasks: List[str]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Filter queries and qrels by subtask names.
    
    Args:
        queries: List of query dictionaries
        qrels: List of qrel dictionaries
        subtasks: List of subtask names to keep
    
    Returns:
        Tuple of (filtered_queries, filtered_qrels)
    """
    # Filter queries
    filtered_queries = [q for q in queries if q.get('subtask') in subtasks]
    
    # Get query IDs
    query_ids = {q['query_id'] for q in filtered_queries}
    
    # Filter qrels
    filtered_qrels = [qr for qr in qrels if qr.get('query_id') in query_ids]
    
    logger.info(f"Filtered to {len(filtered_queries)} queries and {len(filtered_qrels)} qrels")
    logger.info(f"Subtasks: {', '.join(subtasks)}")
    
    return filtered_queries, filtered_qrels

