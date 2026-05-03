"""
Code Embedding Evaluation Runner

This runner evaluates code embedding models using the CoIR framework.
It supports both standard CoIR tasks and custom corpus evaluation.

Features:
- Multiple model types (Sentence-Transformers, HuggingFace, Custom)
- Standard CoIR benchmarks (10 datasets)
- Custom corpus evaluation (our own datasets)
- Config-controlled evaluation options
- Comprehensive metrics (NDCG, MAP, Recall, MRR, Precision)
"""

import json
from pathlib import Path
from typing import Dict, Any

from coreb_runner.runners.base_runner import Runner
from coreb_runner.benchmark.models import create_model_wrapper
from coreb_runner.benchmark.data_loader import (
    load_custom_dataset,
    group_queries_by_subtask,
    filter_queries_by_subtask,
    load_jsonl
)
from coreb_runner.benchmark.evaluation import CoREBEvaluation
from coreb_runner.utils.tasks import get_coreb_tasks
from easyllm_kit.utils import get_logger

logger = get_logger('code_emb_eval_runner')


@Runner.register("code_emb_eval")
class CodeEmbEvalRunner(Runner):
    """
    Runner for evaluating code embedding models.
    
    This runner supports:
    1. Standard CoIR benchmark evaluation (10 datasets)
    2. Custom corpus evaluation (our own datasets)
    3. Subtask-specific evaluation
    4. Multiple model types
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Code Embedding Evaluation Runner.
        
        Args:
            config: Configuration dictionary with structure:
                {
                    "runner_name": "code_emb_eval",
                    "evaluation": {
                        "model": {
                            "type": "huggingface|custom",
                            "model_name": "model-name-or-path",
                            "model_kwargs": {...}
                        },
                        "custom_corpus": {
                            "enabled": true,
                            "corpus_dir": "path/to/corpus.jsonl",
                            "queries_dir": "path/to/queries.jsonl",
                            "qrels_dir": "path/to/qrels.jsonl",
                            "subtasks": ["subtask1", "subtask2", ...]  # Optional
                        },
                        "batch_size": 128,
                        "output_dir": "results/code_emb_eval",
                        "save_embeddings": false,
                        "compute_metrics": true
                    }
                }
        """
        # Initialize without calling super().__init__ since base Runner doesn't have __init__

        self.eval_config = config.get('evaluation', {})
        self.model_config = self.eval_config.get('model', {})

        # Custom corpus configuration
        self.custom_config = self.eval_config.get('custom_corpus', {})
        self.custom_enabled = self.custom_config.get('enabled', True)

        # General configuration
        self.batch_size = self.eval_config.get('batch_size', 128)
        self.output_dir = Path(self.eval_config.get('output_dir', 'results/code_emb_eval'))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Evaluation options
        self.save_embeddings = self.eval_config.get('save_embeddings', False)
        self.compute_metrics = self.eval_config.get('compute_metrics', True)

        # Initialize model
        logger.info("Initializing embedding model...")
        self.model = self._initialize_model()

        # Results storage
        self.results = {}

        logger.info(f"CodeEmbEvalRunner initialized")
        logger.info(f"  • Model: {self.model_config.get('model_name')}")
        logger.info(f"  • Batch size: {self.batch_size}")
        logger.info(f"  • Output dir: {self.output_dir}")

    def _initialize_model(self):
        """Initialize the embedding model based on configuration."""
        model_name = self.model_config.get('model_name')
        model_kwargs = self.model_config.get('model_kwargs', {})

        return create_model_wrapper(model_name, **model_kwargs)

    def run(self):
        """Run code embedding evaluation."""
        logger.info("🚀 Starting Code Embedding Evaluation")

        try:
            # Run custom corpus evaluation
            self._run_custom_evaluation()

            # Save results
            self._save_results()

            # Print summary
            self._print_summary()

            logger.info("✅ Code embedding evaluation completed successfully")

        except Exception as e:
            logger.error(f"❌ Error in code embedding evaluation: {str(e)}")
            raise

    def _run_custom_evaluation(self):
        """Run evaluation on custom corpus (our own dataset)."""
        if not self.custom_enabled:
            logger.info("Custom corpus evaluation disabled, skipping")
            return

        logger.info("Running evaluation on custom corpus...")

        # Get task configuration
        task_name = self.custom_config.get('task_name', 'text2code')
        subtasks = self.custom_config.get('subtasks', [])

        # Determine data directory from corpus_dir
        corpus_dir = self.custom_config.get('corpus_dir')
        if corpus_dir:
            data_dir = str(Path(corpus_dir).parent)
        else:
            data_dir = './scripts/preprocess/ddb_storage'

        logger.info(f"Loading task: {task_name}")
        if subtasks:
            logger.info(f"  • Filtering to subtasks: {', '.join(subtasks)}")

        # Map task_name to our task names
        task_name_map = {
            'text2code_retrieval': 'text2code',
            'code2code_similarity': 'code2code', 
            'code2text_retrieval': 'code2text',
            'custom_code_retrieval': 'text2code',  # default
            'text2code': 'text2code',
            'code2code': 'code2code',
            'code2text': 'code2text',
        }

        benchmark_task_name = task_name_map.get(task_name, task_name)
        
        # Validate task name
        valid_tasks = ['text2code', 'code2code', 'code2text']
        if benchmark_task_name not in valid_tasks:
            raise ValueError(f"Invalid task name: {task_name}. Must be one of: {valid_tasks}")

        # Validate subtasks if provided
        if subtasks:
            from coreb_runner.utils.tasks import CoREBTasks
            task_loader = CoREBTasks()
            available_subtasks = task_loader.get_available_subtasks(benchmark_task_name)
            invalid_subtasks = [s for s in subtasks if s not in available_subtasks]
            if invalid_subtasks:
                logger.warning(f"Invalid subtasks for {benchmark_task_name}: {invalid_subtasks}")
                logger.info(f"Available subtasks: {available_subtasks}")
                # Filter out invalid subtasks
                subtasks = [s for s in subtasks if s in available_subtasks]
                if not subtasks:
                    logger.info("No valid subtasks remaining, using all subtasks")
                    subtasks = None

        # Load task using our task definitions
        subtask_dict = {benchmark_task_name: subtasks} if subtasks else None

        try:
            tasks = get_coreb_tasks(
                task_names=[benchmark_task_name],
                data_dir=data_dir,
                subtasks=subtask_dict
            )

            if not tasks:
                raise ValueError(f"No tasks loaded for: {benchmark_task_name}")

            task = tasks[0]
            
            # Log task information
            logger.info(f"✅ Task loaded: {task.name}")
            logger.info(f"  • Corpus: {len(task.corpus)} documents")
            logger.info(f"  • Queries: {len(task.queries)} queries")
            logger.info(f"  • Qrels: {len(task.qrels)} query-doc pairs")

        except Exception as e:
            logger.error(f"Failed to load task using task definitions: {e}")
            logger.info("Falling back to manual data loading...")

            # Fallback to manual loading
            corpus_dir = self.custom_config.get('corpus_dir')
            queries_dir = self.custom_config.get('queries_dir')
            qrels_dir = self.custom_config.get('qrels_dir')

            if not all([corpus_dir, queries_dir, qrels_dir]):
                raise ValueError(
                    "For custom corpus evaluation, must specify: "
                    "corpus_dir, queries_dir, qrels_dir"
                )

            corpus, queries, qrels = load_custom_dataset(corpus_dir, queries_dir, qrels_dir)

            if subtasks:
                queries_raw = load_jsonl(queries_dir)
                qrels_raw = load_jsonl(qrels_dir)

                filtered_queries, filtered_qrels = filter_queries_by_subtask(
                    queries_raw, qrels_raw, subtasks
                )

                from coreb_runner.benchmark.data_loader import (
                    convert_queries_to_coir_format,
                    convert_qrels_to_coir_format
                )
                queries = convert_queries_to_coir_format(filtered_queries)
                qrels = convert_qrels_to_coir_format(filtered_qrels)

            # Use our custom Task class
            from coreb_runner.utils.tasks import Task
            task = Task(
                name=task_name,
                corpus=corpus,
                queries=queries,
                qrels=qrels
            )

        # Run evaluation using CoREB framework (graded relevance, relevance_level=2)
        evaluation = CoREBEvaluation(tasks=[task], batch_size=self.batch_size)
        
        logger.info("Encoding queries and corpus...")
        custom_results = evaluation.run(
            self.model,
            output_folder=str(self.output_dir / 'custom')
        )

        # ── C2C anchor exclusion ──────────────────────────────────────────────
        # For Code-to-Code tasks the anchor code item (i.e. the query itself) is
        # present in the shared corpus and is always retrieved at rank 1 with
        # near-perfect similarity.  Because the anchor is NOT a positive in the
        # qrels, rank 1 is structurally wasted and nDCG@1 / MRR@1 are 0 for
        # every model by construction.  We recompute metrics after stripping the
        # anchor from each query's ranked list so that the reported numbers
        # reflect actual cross-language retrieval ability.
        anchor_map = getattr(task, 'c2c_anchor_map', {})
        if anchor_map:
            custom_results = self._recompute_c2c_without_anchor(
                custom_results, task, anchor_map
            )
            logger.info("✅ C2C metrics recomputed with anchor excluded from ranked lists")
        # ─────────────────────────────────────────────────────────────────────

        # Store results
        self.results['custom'] = custom_results

        logger.info(f"✅ Custom evaluation completed")

        # If compute_metrics enabled, calculate additional metrics
        if self.compute_metrics:
            self._compute_additional_metrics(task.queries, task.qrels, task.name)

    def _recompute_c2c_without_anchor(self, results: Dict, task, anchor_map: Dict[str, str]) -> Dict:
        """
        Recompute Code-to-Code metrics after removing the anchor code item from
        each query's ranked list.

        The anchor (the query's own code snippet) is always retrieved at rank 1
        with cosine similarity ≈ 1.0.  It is not a positive in the qrels, so
        rank 1 is structurally wasted.  Stripping it gives a fair measurement of
        cross-language retrieval ability.

        Args:
            results:    Raw CoIR result dict for this task.
            task:       Task object (provides qrels).
            anchor_map: {query_id: anchor_code_id} built by CoREBTasks.get_code2code_task().

        Returns:
            Updated results dict with corrected metrics for the C2C task.
        """
        import math

        # Relevance threshold: only true positives (relevance >= 2) count as
        # "relevant" for binary metrics.  Hard negatives (relevance=1) contribute
        # graded gain to nDCG but do NOT count as hits for Recall/MRR.
        _REL_THRESH = 1  # relevant if relevance > _REL_THRESH (i.e. >= 2)

        def _ndcg_at_k(ranked_ids, relevant: dict, k: int) -> float:
            # Zero out hard negatives so they contribute no gain to nDCG,
            # consistent with how EvaluateRetrieval.evaluate() handles them.
            filtered = {did: rel if rel > _REL_THRESH else 0 for did, rel in relevant.items()}
            dcg = sum(
                filtered.get(doc_id, 0) / math.log2(i + 2)
                for i, doc_id in enumerate(ranked_ids[:k])
            )
            ideal = sorted([v for v in filtered.values() if v > 0], reverse=True)[:k]
            idcg = sum(v / math.log2(i + 2) for i, v in enumerate(ideal))
            return dcg / idcg if idcg else 0.0

        def _recall_at_k(ranked_ids, relevant: dict, k: int) -> float:
            # Only true positives (relevance > threshold) count
            hits = sum(1 for d in ranked_ids[:k] if relevant.get(d, 0) > _REL_THRESH)
            total_relevant = sum(1 for v in relevant.values() if v > _REL_THRESH)
            return hits / total_relevant if total_relevant else 0.0

        def _mrr_at_k(ranked_ids, relevant: dict, k: int) -> float:
            for i, doc_id in enumerate(ranked_ids[:k]):
                if relevant.get(doc_id, 0) > _REL_THRESH:
                    return 1.0 / (i + 1)
            return 0.0

        # CoIR stores predictions under the output folder; we re-read them from
        # the saved predictions file if available, otherwise fall back to the
        # results dict (which may already contain per-query data).
        pred_file = self.output_dir / 'custom' / f'{task.name}_predictions.json'
        if not pred_file.exists():
            logger.warning(
                "C2C predictions file not found; skipping anchor-exclusion recomputation. "
                f"Expected: {pred_file}"
            )
            return results

        with open(pred_file) as f:
            predictions = json.load(f)  # {query_id: [{corpus_id, score, rank}, ...]}

        k_values = [1, 5, 10, 20]
        qrels_dict = task.qrels  # {query_id: {doc_id: relevance}}

        per_query_metrics = {k: {'ndcg': [], 'recall': [], 'mrr': []} for k in k_values}

        for query_id, ranked_list in predictions.items():
            anchor_id = anchor_map.get(query_id, '')
            # Strip the anchor from the ranked list and re-rank
            filtered = [item['corpus_id'] for item in ranked_list if item['corpus_id'] != anchor_id]
            relevant = qrels_dict.get(query_id, {})
            for k in k_values:
                per_query_metrics[k]['ndcg'].append(_ndcg_at_k(filtered, relevant, k))
                per_query_metrics[k]['recall'].append(_recall_at_k(filtered, relevant, k))
                per_query_metrics[k]['mrr'].append(_mrr_at_k(filtered, relevant, k))

        # Build corrected metrics dict in CoIR format
        corrected: Dict = {}
        for k in k_values:
            n = len(per_query_metrics[k]['ndcg'])
            corrected[f'ndcg_at_{k}'] = sum(per_query_metrics[k]['ndcg']) / n if n else 0.0
            corrected[f'recall_at_{k}'] = sum(per_query_metrics[k]['recall']) / n if n else 0.0
            corrected[f'mrr_at_{k}'] = sum(per_query_metrics[k]['mrr']) / n if n else 0.0

        logger.info("C2C corrected metrics (anchor excluded):")
        for metric, val in corrected.items():
            logger.info(f"  {metric}: {val:.4f}")

        # Patch the corrected metrics back into the results dict
        task_key = task.name
        if task_key in results:
            if isinstance(results[task_key], dict):
                results[task_key].update(corrected)
        return results

    def _compute_additional_metrics(self, queries: Dict, qrels: Dict, task_name: str):
        """Compute additional metrics beyond standard CoIR metrics."""
        logger.info("Computing additional metrics...")

        additional_metrics = {
            'subtask_breakdown': {},
            'summary': {
                'total_queries': len(queries),
                'total_qrels': sum(len(docs) for docs in qrels.values()),
                'avg_qrels_per_query': sum(len(docs) for docs in qrels.values()) / len(qrels) if qrels else 0,
                'task_name': task_name
            }
        }

        # Store additional metrics
        if 'custom' not in self.results:
            self.results['custom'] = {}

        self.results['custom']['additional_metrics'] = additional_metrics

        logger.info("✅ Additional metrics computed")

    def _save_results(self):
        """Save evaluation results."""
        results_file = self.output_dir / 'evaluation_results.json'

        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)

        logger.info(f"Results saved to {results_file}")

        # Save summary
        summary_file = self.output_dir / 'evaluation_summary.txt'
        with open(summary_file, 'w') as f:
            f.write(self._generate_summary_text())

        logger.info(f"Summary saved to {summary_file}")

    def _generate_summary_text(self) -> str:
        """Generate summary text."""
        lines = []
        lines.append("=" * 80)
        lines.append("CODE EMBEDDING EVALUATION SUMMARY")
        lines.append("=" * 80)
        lines.append(f"Model: {self.model_config.get('model_name')}")
        lines.append(f"Batch Size: {self.batch_size}")
        lines.append("")

        # Custom results
        if 'custom' in self.results:
            lines.append("\n📊 CUSTOM CORPUS EVALUATION")
            lines.append("-" * 80)

            for task_name, task_results in self.results['custom'].items():
                if task_name == 'additional_metrics':
                    # Show summary
                    summary = task_results.get('summary', {})
                    lines.append(f"\nDataset Summary:")
                    lines.append(f"  • Total Queries: {summary.get('total_queries', 0)}")
                    lines.append(f"  • Total Qrels: {summary.get('total_qrels', 0)}")
                    lines.append(f"  • Avg Qrels/Query: {summary.get('avg_qrels_per_query', 0):.2f}")
                    lines.append(f"  • Task: {summary.get('task_name', 'Unknown')}")
                else:
                    lines.append(f"\n{task_name}:")
                    if isinstance(task_results, dict):
                        for metric_name, metric_value in task_results.items():
                            if isinstance(metric_value, (int, float)):
                                lines.append(f"  • {metric_name}: {metric_value:.4f}")

        lines.append("\n" + "=" * 80)
        return "\n".join(lines)

    def _print_summary(self):
        """Print evaluation summary to console."""
        print("\n" + self._generate_summary_text())
    
    def list_available_subtasks(self, task_name: str = None):
        """List available subtasks for a given task or all tasks."""
        from coreb_runner.utils.tasks import CoREBTasks, print_available_subtasks
        
        if task_name:
            task_loader = CoREBTasks()
            available_subtasks = task_loader.get_available_subtasks(task_name)
            logger.info(f"Available subtasks for {task_name}:")
            for subtask in available_subtasks:
                subtask_info = task_loader.get_subtask_info(task_name, subtask)
                logger.info(f"  • {subtask}: {subtask_info.get('description', '')}")
        else:
            print_available_subtasks()


# Register the runner
CodeEmbEvalRunner.register('code_emb_eval')
