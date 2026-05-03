"""
CoREB Retrieval Evaluation Module

Replaces CoIR's evaluation pipeline with graded-relevance-aware metrics.
Key difference: uses relevance_level=2 so that hard negatives (relevance=1)
are NOT counted as relevant for binary metrics (Recall, MAP, Precision, MRR),
while nDCG properly uses graded relevance values.

Relevance scale:
    2 = true positive (correct match)
    1 = hard negative (same problem, wrong/partial match)
    0 = irrelevant (not used in qrels; unjudged docs are implicitly 0)
"""

import logging
import math
import heapq
import torch
import numpy as np
from typing import Dict, List, Tuple, Any

logger = logging.getLogger(__name__)

# ── Similarity functions ─────────────────────────────────────────────────────

def cos_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute pairwise cosine similarity: res[i][j] = cos_sim(a[i], b[j])."""
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)
    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)
    if len(a.shape) == 1:
        a = a.unsqueeze(0)
    if len(b.shape) == 1:
        b = b.unsqueeze(0)
    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))


def dot_score(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute pairwise dot product: res[i][j] = dot(a[i], b[j])."""
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)
    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)
    if len(a.shape) == 1:
        a = a.unsqueeze(0)
    if len(b.shape) == 1:
        b = b.unsqueeze(0)
    return torch.mm(a, b.transpose(0, 1))


# ── Dense Retrieval ──────────────────────────────────────────────────────────

class DenseRetrievalExactSearch:
    """
    Dense retrieval via brute-force cosine/dot similarity.

    Encodes queries and corpus in batches, computes similarity, and returns
    top-k results per query.  Compatible with any model that exposes
    ``encode_queries()`` and ``encode_corpus()``.
    """

    def __init__(self, model, batch_size: int = 128,
                 corpus_chunk_size: int = 50000, **kwargs):
        self.model = model
        self.batch_size = batch_size
        self.corpus_chunk_size = corpus_chunk_size
        self.show_progress_bar = kwargs.get("show_progress_bar", True)
        self.convert_to_tensor = kwargs.get("convert_to_tensor", True)
        self.score_functions = {'cos_sim': cos_sim, 'dot': dot_score}
        self.results: Dict[str, Dict[str, float]] = {}

    def search(
        self,
        corpus: Dict[str, Dict[str, str]],
        queries: Dict[str, str],
        top_k: int,
        score_function: str = "cos_sim",
        return_sorted: bool = False,
        **kwargs
    ) -> Dict[str, Dict[str, float]]:
        if score_function not in self.score_functions:
            raise ValueError(
                f"score_function must be 'cos_sim' or 'dot', got '{score_function}'"
            )

        logger.info("Encoding Queries...")
        query_ids = list(queries.keys())
        self.results = {qid: {} for qid in query_ids}
        query_texts = [queries[qid] for qid in query_ids]
        query_embeddings = self.model.encode_queries(
            query_texts,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar,
            convert_to_tensor=self.convert_to_tensor,
        )

        logger.info("Sorting Corpus by document length (Longest first)...")
        corpus_ids = sorted(
            corpus,
            key=lambda k: len(corpus[k].get("title", "") + corpus[k].get("text", "")),
            reverse=True,
        )
        corpus_list = [corpus[cid] for cid in corpus_ids]

        logger.info("Encoding Corpus in batches...")
        score_fn = self.score_functions[score_function]

        result_heaps: Dict[str, list] = {qid: [] for qid in query_ids}

        for batch_num, start in enumerate(
            range(0, len(corpus_list), self.corpus_chunk_size)
        ):
            end = min(start + self.corpus_chunk_size, len(corpus_list))
            logger.info(
                f"Encoding Batch {batch_num + 1}/"
                f"{math.ceil(len(corpus_list) / self.corpus_chunk_size)}..."
            )

            sub_corpus_embeddings = self.model.encode_corpus(
                corpus_list[start:end],
                batch_size=self.batch_size,
                show_progress_bar=self.show_progress_bar,
                convert_to_tensor=self.convert_to_tensor,
            )

            scores = score_fn(query_embeddings, sub_corpus_embeddings)
            scores[torch.isnan(scores)] = -1

            top_k_values, top_k_idx = torch.topk(
                scores, min(top_k + 1, scores.shape[1]), dim=1,
                largest=True, sorted=return_sorted,
            )
            top_k_values = top_k_values.cpu().tolist()
            top_k_idx = top_k_idx.cpu().tolist()

            for qi in range(len(query_embeddings)):
                qid = query_ids[qi]
                for sub_idx, sc in zip(top_k_idx[qi], top_k_values[qi]):
                    cid = corpus_ids[start + sub_idx]
                    if cid != qid:
                        if len(result_heaps[qid]) < top_k:
                            heapq.heappush(result_heaps[qid], (sc, cid))
                        else:
                            heapq.heappushpop(result_heaps[qid], (sc, cid))

        for qid in result_heaps:
            for sc, cid in result_heaps[qid]:
                self.results[qid][cid] = sc

        return self.results


# ── Metric computation ───────────────────────────────────────────────────────

class EvaluateRetrieval:
    """
    Retrieval evaluation with graded relevance support.

    Key difference from CoIR/BEIR: ``relevance_level=2`` so that only true
    positives (relevance >= 2) count as "relevant" for binary metrics
    (Recall, MAP, Precision).  nDCG uses the raw graded values natively.
    Hard negatives (relevance=1) lower nDCG by occupying top ranks with
    sub-maximal gain, but do NOT inflate Recall/MRR/MAP.
    """

    RELEVANCE_LEVEL = 2  # minimum relevance to be "relevant" for binary metrics

    def __init__(self, retriever=None,
                 k_values: List[int] = None,
                 score_function: str = "cos_sim"):
        self.k_values = k_values or [1, 3, 5, 10, 100, 1000]
        self.top_k = max(self.k_values)
        self.retriever = retriever
        self.score_function = score_function

    def retrieve(
        self,
        corpus: Dict[str, Dict[str, str]],
        queries: Dict[str, str],
        **kwargs
    ) -> Dict[str, Dict[str, float]]:
        if not self.retriever:
            raise ValueError("No retriever provided.")
        return self.retriever.search(
            corpus, queries, self.top_k, self.score_function, **kwargs
        )

    def rerank(
        self,
        corpus: Dict[str, Dict[str, str]],
        queries: Dict[str, str],
        results: Dict[str, Dict[str, float]],
        top_k: int,
    ) -> Dict[str, Dict[str, float]]:
        new_corpus = {}
        for qid in results:
            ranked = sorted(results[qid].items(), key=lambda x: x[1], reverse=True)
            for doc_id, _ in ranked[:top_k]:
                new_corpus[doc_id] = corpus[doc_id]
        return self.retriever.search(new_corpus, queries, top_k, self.score_function)

    @staticmethod
    def evaluate(
        qrels: Dict[str, Dict[str, int]],
        results: Dict[str, Dict[str, float]],
        k_values: List[int],
        ignore_identical_ids: bool = True,
    ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float]]:
        """
        Evaluate retrieval results using pytrec_eval with relevance_level=2.

        Returns (ndcg, map, recall, precision) dicts keyed like "NDCG@10".
        """
        import pytrec_eval

        if ignore_identical_ids:
            for qid, rels in results.items():
                for pid in list(rels):
                    if qid == pid:
                        results[qid].pop(pid)

        ndcg, _map, recall, precision = {}, {}, {}, {}
        for k in k_values:
            ndcg[f"NDCG@{k}"] = 0.0
            _map[f"MAP@{k}"] = 0.0
            recall[f"Recall@{k}"] = 0.0
            precision[f"P@{k}"] = 0.0

        map_string = "map_cut." + ",".join(str(k) for k in k_values)
        ndcg_string = "ndcg_cut." + ",".join(str(k) for k in k_values)
        recall_string = "recall." + ",".join(str(k) for k in k_values)
        precision_string = "P." + ",".join(str(k) for k in k_values)

        # Zero out hard negatives (relevance < RELEVANCE_LEVEL) so they
        # contribute zero gain to nDCG as well as binary metrics.
        # pytrec_eval's relevance_level only affects binary metrics (Recall,
        # MAP, Precision) — nDCG always uses the raw graded values.  By
        # zeroing out sub-threshold entries we ensure nDCG treats hard
        # negatives (relevance=1) as irrelevant rather than partially relevant.
        filtered_qrels = {}
        for qid, docs in qrels.items():
            filtered_qrels[qid] = {
                did: rel if rel >= EvaluateRetrieval.RELEVANCE_LEVEL else 0
                for did, rel in docs.items()
            }

        evaluator = pytrec_eval.RelevanceEvaluator(
            filtered_qrels,
            {map_string, ndcg_string, recall_string, precision_string},
            relevance_level=EvaluateRetrieval.RELEVANCE_LEVEL,
        )
        scores = evaluator.evaluate(results)

        for qid in scores:
            for k in k_values:
                ndcg[f"NDCG@{k}"] += scores[qid][f"ndcg_cut_{k}"]
                _map[f"MAP@{k}"] += scores[qid][f"map_cut_{k}"]
                recall[f"Recall@{k}"] += scores[qid][f"recall_{k}"]
                precision[f"P@{k}"] += scores[qid][f"P_{k}"]

        n = len(scores)
        for k in k_values:
            ndcg[f"NDCG@{k}"] = round(ndcg[f"NDCG@{k}"] / n, 5) if n else 0.0
            _map[f"MAP@{k}"] = round(_map[f"MAP@{k}"] / n, 5) if n else 0.0
            recall[f"Recall@{k}"] = round(recall[f"Recall@{k}"] / n, 5) if n else 0.0
            precision[f"P@{k}"] = round(precision[f"P@{k}"] / n, 5) if n else 0.0

        for metric_dict in [ndcg, _map, recall, precision]:
            logger.info("")
            for key, val in metric_dict.items():
                logger.info(f"{key}: {val:.4f}")

        return ndcg, _map, recall, precision

    @staticmethod
    def evaluate_custom(
        qrels: Dict[str, Dict[str, int]],
        results: Dict[str, Dict[str, float]],
        k_values: List[int],
        metric: str,
    ) -> Dict[str, float]:
        """Compute custom metrics (MRR, R_cap, Accuracy) with graded relevance."""
        metric_lower = metric.lower()
        if metric_lower in ("mrr", "mrr@k", "mrr_cut"):
            return mrr(qrels, results, k_values)
        elif metric_lower in ("recall_cap", "r_cap", "r_cap@k"):
            return recall_cap(qrels, results, k_values)
        elif metric_lower in ("acc", "top_k_acc", "accuracy", "accuracy@k", "top_k_accuracy"):
            return top_k_accuracy(qrels, results, k_values)
        else:
            raise ValueError(f"Unknown custom metric: {metric}")


# ── Custom metrics (graded-relevance aware) ──────────────────────────────────
# These check relevance > 1 (i.e. only true positives with relevance >= 2)
# so hard negatives (relevance=1) do NOT count as hits.

_RELEVANCE_THRESHOLD = 1  # doc is "relevant" if relevance > this value


def mrr(
    qrels: Dict[str, Dict[str, int]],
    results: Dict[str, Dict[str, float]],
    k_values: List[int],
) -> Dict[str, float]:
    """Mean Reciprocal Rank — only true positives (relevance > 1) are hits."""
    MRR = {f"MRR@{k}": 0.0 for k in k_values}
    k_max = max(k_values)

    top_hits = {}
    for qid, doc_scores in results.items():
        top_hits[qid] = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:k_max]

    for qid in top_hits:
        relevant = {
            did for did, rel in qrels.get(qid, {}).items()
            if rel > _RELEVANCE_THRESHOLD
        }
        for k in k_values:
            for rank, (did, _) in enumerate(top_hits[qid][:k]):
                if did in relevant:
                    MRR[f"MRR@{k}"] += 1.0 / (rank + 1)
                    break

    n = len(qrels)
    for k in k_values:
        MRR[f"MRR@{k}"] = round(MRR[f"MRR@{k}"] / n, 5) if n else 0.0
        logger.info(f"MRR@{k}: {MRR[f'MRR@{k}']:.4f}")

    return MRR


def recall_cap(
    qrels: Dict[str, Dict[str, int]],
    results: Dict[str, Dict[str, float]],
    k_values: List[int],
) -> Dict[str, float]:
    """Capped recall — only true positives (relevance > 1) are relevant."""
    capped = {f"R_cap@{k}": 0.0 for k in k_values}
    k_max = max(k_values)

    for qid, doc_scores in results.items():
        top_hits = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:k_max]
        relevant = [
            did for did, rel in qrels.get(qid, {}).items()
            if rel > _RELEVANCE_THRESHOLD
        ]
        for k in k_values:
            retrieved = [
                did for did, _ in top_hits[:k]
                if qrels.get(qid, {}).get(did, 0) > _RELEVANCE_THRESHOLD
            ]
            denom = min(len(relevant), k)
            if denom > 0:
                capped[f"R_cap@{k}"] += len(retrieved) / denom

    n = len(qrels)
    for k in k_values:
        capped[f"R_cap@{k}"] = round(capped[f"R_cap@{k}"] / n, 5) if n else 0.0
        logger.info(f"R_cap@{k}: {capped[f'R_cap@{k}']:.4f}")

    return capped


def top_k_accuracy(
    qrels: Dict[str, Dict[str, int]],
    results: Dict[str, Dict[str, float]],
    k_values: List[int],
) -> Dict[str, float]:
    """Top-k accuracy — a query is a hit if any true positive is in top-k."""
    acc = {f"Accuracy@{k}": 0.0 for k in k_values}
    k_max = max(k_values)

    top_hits = {}
    for qid, doc_scores in results.items():
        top_hits[qid] = [
            did for did, _ in sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:k_max]
        ]

    for qid in top_hits:
        relevant = {
            did for did, rel in qrels.get(qid, {}).items()
            if rel > _RELEVANCE_THRESHOLD
        }
        for k in k_values:
            if relevant & set(top_hits[qid][:k]):
                acc[f"Accuracy@{k}"] += 1.0

    n = len(qrels)
    for k in k_values:
        acc[f"Accuracy@{k}"] = round(acc[f"Accuracy@{k}"] / n, 5) if n else 0.0
        logger.info(f"Accuracy@{k}: {acc[f'Accuracy@{k}']:.4f}")

    return acc


# ── Orchestrator (replaces coir.evaluation.COIR) ────────────────────────────

class CoREBEvaluation:
    """
    Evaluation orchestrator — drop-in replacement for ``coir.evaluation.COIR``.

    Runs dense retrieval then evaluates with graded-relevance-aware metrics.
    """

    def __init__(self, tasks, batch_size: int = 128):
        self.tasks = tasks
        self.batch_size = batch_size

    def run(self, model, output_folder: str) -> Dict[str, Any]:
        import os
        import json

        results = {}
        for task_name, task_data in self.tasks.items():
            output_file = os.path.join(output_folder, f"{task_name}.json")

            if os.path.exists(output_file):
                logger.info(f"Results for {task_name} already exist. Skipping.")
                continue

            corpus, queries, qrels = task_data

            retriever_model = DenseRetrievalExactSearch(model, batch_size=self.batch_size)
            retriever = EvaluateRetrieval(retriever_model, score_function="cos_sim")

            task_results = retriever.retrieve(corpus, queries)

            ndcg, _map, recall, precision = retriever.evaluate(
                qrels, task_results, retriever.k_values
            )
            metrics = {
                "NDCG": ndcg,
                "MAP": _map,
                "Recall": recall,
                "Precision": precision,
            }

            # Save predictions for downstream recomputation (e.g. C2C anchor exclusion)
            os.makedirs(output_folder, exist_ok=True)
            predictions = {}
            for qid, doc_scores in task_results.items():
                ranked = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
                predictions[qid] = [
                    {"corpus_id": did, "score": sc, "rank": i + 1}
                    for i, (did, sc) in enumerate(ranked)
                ]

            pred_file = os.path.join(output_folder, f"{task_name}_predictions.json")
            with open(pred_file, 'w') as f:
                json.dump(predictions, f)

            with open(output_file, 'w') as f:
                json.dump({"metrics": metrics}, f, indent=4)

            logger.info(f"Results for {task_name} saved to {output_folder}")
            results[task_name] = metrics

        return results
