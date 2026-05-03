"""
CoREB (Code Retrieval Embedding Benchmark) Tasks

Define CoREB benchmark tasks with graded relevance support.
Evaluation uses our own CoREBEvaluation module (relevance_level=2).

CoREB includes three main tasks:
- CoREB-Text2Code (coreb_t2c_*): Retrieve code from natural language
- CoREB-Code2Code (coreb_c2c_*): Find similar code implementations  
- CoREB-Code2Text (coreb_c2t_*): Retrieve descriptions from code
"""

from typing import Dict, List, Optional, Any
from pathlib import Path

# CoREB uses its own evaluation module — no CoIR dependency required
from easyllm_kit.utils import get_logger

from coreb_runner.benchmark.data_loader import (
    load_custom_dataset,
    filter_queries_by_subtask,
    load_jsonl
)

logger = get_logger(__name__)


class CoREBTasks:
    """
    CoREB (Code Retrieval Embedding Benchmark) Task Definitions.
    
    This class creates CoIR/MTEB compatible task objects from CoREB benchmark data.
    
    Task naming convention:
    - coreb_t2c_* : Text-to-Code retrieval subtasks
    - coreb_c2c_* : Code-to-Code similarity subtasks
    - coreb_c2t_* : Code-to-Text retrieval subtasks
    """
    
    # Task metadata following MTEB/CoIR conventions
    TASK_METADATA = {
        'text2code': {
            'name': 'CoREB-Text2Code',
            'description': 'CoREB: Retrieve code implementations from natural language descriptions using canonical (abbreviated) problem statements',
            'type': 'Retrieval',
            'category': 'code',
            'modalities': ['text', 'code'],
            'eval_splits': ['test'],
            'eval_langs': ['eng'],  # English queries
            'main_score': 'ndcg_at_10',
            'domains': ['Programming', 'Algorithm'],
            'task_subtypes': ['Code Search', 'Text-to-Code Retrieval'],
            'license': 'apache-2.0',
            'annotations_creators': 'expert-generated',
            'dialect': [],
            'sample_creation': 'found',
            'bibtex_citation': '',  # Add your paper citation
            'subtasks': {
                # Canonical Retro (abbreviated) subtasks
                't2c_canonical_retro_any': {
                    'name': 'Text2Code Canonical Retro Any',
                    'description': 'Language-agnostic retrieval using canonical (abbreviated) problem descriptions',
                    'query_type': 'language_agnostic',
                    'text_type': 'canonical',
                    'language_constraint': 'none',
                    'count': 84
                },
                't2c_canonical_retro_python': {
                    'name': 'Text2Code Canonical Retro Python',
                    'description': 'Python-specific retrieval using canonical problem descriptions',
                    'query_type': 'language_specific',
                    'text_type': 'canonical',
                    'language_constraint': 'python',
                    'count': 73
                },
                't2c_canonical_retro_java': {
                    'name': 'Text2Code Canonical Retro Java',
                    'description': 'Java-specific retrieval using canonical problem descriptions',
                    'query_type': 'language_specific',
                    'text_type': 'canonical',
                    'language_constraint': 'java',
                    'count': 55
                },
                't2c_canonical_retro_cpp': {
                    'name': 'Text2Code Canonical Retro C++',
                    'description': 'C++-specific retrieval using canonical problem descriptions',
                    'query_type': 'language_specific',
                    'text_type': 'canonical',
                    'language_constraint': 'cpp',
                    'count': 48
                },
                't2c_canonical_retro_ruby': {
                    'name': 'Text2Code Canonical Retro Ruby',
                    'description': 'Ruby-specific retrieval using canonical problem descriptions',
                    'query_type': 'language_specific',
                    'text_type': 'canonical',
                    'language_constraint': 'ruby',
                    'count': 43
                },
                't2c_canonical_retro_go': {
                    'name': 'Text2Code Canonical Retro Go',
                    'description': 'Go-specific retrieval using canonical problem descriptions',
                    'query_type': 'language_specific',
                    'text_type': 'canonical',
                    'language_constraint': 'go',
                    'count': 39
                },
                # Full Retro (complete) subtasks
                't2c_full_retro_any': {
                    'name': 'Text2Code Full Retro Any',
                    'description': 'Language-agnostic retrieval using full problem descriptions',
                    'query_type': 'language_agnostic',
                    'text_type': 'full',
                    'language_constraint': 'none',
                    'count': 84
                },
                't2c_full_retro_python': {
                    'name': 'Text2Code Full Retro Python',
                    'description': 'Python-specific retrieval using full problem descriptions',
                    'query_type': 'language_specific',
                    'text_type': 'full',
                    'language_constraint': 'python',
                    'count': 73
                },
                't2c_full_retro_java': {
                    'name': 'Text2Code Full Retro Java',
                    'description': 'Java-specific retrieval using full problem descriptions',
                    'query_type': 'language_specific',
                    'text_type': 'full',
                    'language_constraint': 'java',
                    'count': 55
                },
                't2c_full_retro_cpp': {
                    'name': 'Text2Code Full Retro C++',
                    'description': 'C++-specific retrieval using full problem descriptions',
                    'query_type': 'language_specific',
                    'text_type': 'full',
                    'language_constraint': 'cpp',
                    'count': 48
                },
                't2c_full_retro_ruby': {
                    'name': 'Text2Code Full Retro Ruby',
                    'description': 'Ruby-specific retrieval using full problem descriptions',
                    'query_type': 'language_specific',
                    'text_type': 'full',
                    'language_constraint': 'ruby',
                    'count': 43
                },
                't2c_full_retro_go': {
                    'name': 'Text2Code Full Retro Go',
                    'description': 'Go-specific retrieval using full problem descriptions',
                    'query_type': 'language_specific',
                    'text_type': 'full',
                    'language_constraint': 'go',
                    'count': 39
                },
                # Search (LLM-generated) subtasks
                't2c_search_any': {
                    'name': 'Text2Code Search Any',
                    'description': 'Language-agnostic retrieval using LLM-generated search queries',
                    'query_type': 'language_agnostic',
                    'text_type': 'search',
                    'language_constraint': 'none',
                    'count': 100
                },
                't2c_search_python': {
                    'name': 'Text2Code Search Python',
                    'description': 'Python-specific retrieval using LLM-generated search queries',
                    'query_type': 'language_specific',
                    'text_type': 'search',
                    'language_constraint': 'python',
                    'count': 100
                },
                't2c_search_java': {
                    'name': 'Text2Code Search Java',
                    'description': 'Java-specific retrieval using LLM-generated search queries',
                    'query_type': 'language_specific',
                    'text_type': 'search',
                    'language_constraint': 'java',
                    'count': 100
                },
                't2c_search_cpp': {
                    'name': 'Text2Code Search C++',
                    'description': 'C++-specific retrieval using LLM-generated search queries',
                    'query_type': 'language_specific',
                    'text_type': 'search',
                    'language_constraint': 'cpp',
                    'count': 100
                },
                't2c_search_ruby': {
                    'name': 'Text2Code Search Ruby',
                    'description': 'Ruby-specific retrieval using LLM-generated search queries',
                    'query_type': 'language_specific',
                    'text_type': 'search',
                    'language_constraint': 'ruby',
                    'count': 100
                },
                't2c_search_go': {
                    'name': 'Text2Code Search Go',
                    'description': 'Go-specific retrieval using LLM-generated search queries',
                    'query_type': 'language_specific',
                    'text_type': 'search',
                    'language_constraint': 'go',
                    'count': 100
                }
            },
            'descriptive_stats': {
                'n_samples': {'test': 1284},  # Total: 18 subtasks with varying counts
                'avg_character_length': {'test': {'queries': 600, 'corpus': 150}},
                'subtask_counts': {
                    'canonical_retro': 342,  # 6 subtasks: any(84) + python(73) + java(55) + cpp(48) + ruby(43) + go(39)
                    'full_retro': 342,       # 6 subtasks: any(84) + python(73) + java(55) + cpp(48) + ruby(43) + go(39)
                    'search': 600            # 6 subtasks: any(100) + python(100) + java(100) + cpp(100) + ruby(100) + go(100)
                }
            }
        },
        'code2code': {
            'name': 'CoREB-Code2Code',
            'description': 'CoREB: Find semantically similar code implementations within the same language',
            'type': 'Retrieval',
            'category': 'code',
            'modalities': ['code'],
            'eval_splits': ['test'],
            'eval_langs': ['python'],
            'main_score': 'ndcg_at_10',
            'domains': ['Programming'],
            'task_subtypes': ['Code Similarity', 'Monolingual Code Matching'],
            'license': 'apache-2.0',
            'annotations_creators': 'expert-generated',
            'dialect': [],
            'sample_creation': 'found',
            'bibtex_citation': '',
            'subtasks': {
                'c2c_mono_lang': {
                    'name': 'Code2Code Monolingual',
                    'description': 'Find similar code implementations within the same programming language',
                    'anchor_language': 'python',
                    'anchor_model': 'o1-mini',
                    'similarity_type': 'semantic',
                    'count': 47
                },
                'c2c_cross_lang': {
                    'name': 'Code2Code Cross-Language',
                    'description': 'Find similar code implementations across different programming languages',
                    'anchor_language': 'python',
                    'anchor_model': 'o1-mini',
                    'similarity_type': 'cross_language',
                    'count': 169
                },
                'c2c_cross_model': {
                    'name': 'Code2Code Cross-Model',
                    'description': 'Find similar code implementations from different models',
                    'anchor_language': 'python',
                    'anchor_model': 'o1-mini',
                    'similarity_type': 'cross_model',
                    'count': 48
                }
            },
            'descriptive_stats': {
                'n_samples': {'test': 264},  # Total: 3 subtasks
                'avg_character_length': {'test': {'queries': 150, 'corpus': 150}},
                'subtask_counts': {
                    'mono_lang': 47,      # Same language similarity
                    'cross_lang': 169,    # Cross-language similarity  
                    'cross_model': 48     # Cross-model similarity
                }
            }
        },
        'code2text': {
            'name': 'CoREB-Code2Text',
            'description': 'CoREB: Retrieve full problem descriptions from code implementations across multiple programming languages',
            'type': 'Retrieval',
            'category': 'code',
            'modalities': ['code', 'text'],
            'eval_splits': ['test'],
            'eval_langs': ['python', 'java', 'cpp', 'ruby'],
            'main_score': 'ndcg_at_10',
            'domains': ['Programming', 'Algorithm'],
            'task_subtypes': ['Code Understanding', 'Cross-Language Retrieval'],
            'license': 'apache-2.0',
            'annotations_creators': 'expert-generated',
            'dialect': [],
            'sample_creation': 'found',
            'bibtex_citation': '',
            'subtasks': {
                # Canonical Retro (abbreviated) subtasks
                'c2t_canonical_retro_any': {
                    'name': 'Code2Text Canonical Retro Any',
                    'description': 'Language-agnostic retrieval of canonical (abbreviated) problem descriptions from code',
                    'task_type': 'description_retrieval',
                    'text_type': 'canonical',
                    'language_constraint': 'none',
                    'count': 84
                },
                'c2t_canonical_retro_python': {
                    'name': 'Code2Text Canonical Retro Python',
                    'description': 'Python-specific retrieval of canonical problem descriptions from code',
                    'task_type': 'description_retrieval',
                    'text_type': 'canonical',
                    'language_constraint': 'python',
                    'count': 55
                },
                'c2t_canonical_retro_java': {
                    'name': 'Code2Text Canonical Retro Java',
                    'description': 'Java-specific retrieval of canonical problem descriptions from code',
                    'task_type': 'description_retrieval',
                    'text_type': 'canonical',
                    'language_constraint': 'java',
                    'count': 52
                },
                'c2t_canonical_retro_cpp': {
                    'name': 'Code2Text Canonical Retro C++',
                    'description': 'C++-specific retrieval of canonical problem descriptions from code',
                    'task_type': 'description_retrieval',
                    'text_type': 'canonical',
                    'language_constraint': 'cpp',
                    'count': 47
                },
                'c2t_canonical_retro_ruby': {
                    'name': 'Code2Text Canonical Retro Ruby',
                    'description': 'Ruby-specific retrieval of canonical problem descriptions from code',
                    'task_type': 'description_retrieval',
                    'text_type': 'canonical',
                    'language_constraint': 'ruby',
                    'count': 42
                },
                'c2t_canonical_retro_go': {
                    'name': 'Code2Text Canonical Retro Go',
                    'description': 'Go-specific retrieval of canonical problem descriptions from code',
                    'task_type': 'description_retrieval',
                    'text_type': 'canonical',
                    'language_constraint': 'go',
                    'count': 38
                },
                # Full Retro (complete) subtasks
                'c2t_full_retro_any': {
                    'name': 'Code2Text Full Retro Any',
                    'description': 'Language-agnostic retrieval of full problem descriptions from code',
                    'task_type': 'description_retrieval',
                    'text_type': 'full_description',
                    'language_constraint': 'none',
                    'count': 84
                },
                'c2t_full_retro_python': {
                    'name': 'Code2Text Full Retro Python',
                    'description': 'Python-specific retrieval of full problem descriptions from code',
                    'task_type': 'description_retrieval',
                    'text_type': 'full_description',
                    'language_constraint': 'python',
                    'count': 55
                },
                'c2t_full_retro_java': {
                    'name': 'Code2Text Full Retro Java',
                    'description': 'Java-specific retrieval of full problem descriptions from code',
                    'task_type': 'description_retrieval',
                    'text_type': 'full_description',
                    'language_constraint': 'java',
                    'count': 52
                },
                'c2t_full_retro_cpp': {
                    'name': 'Code2Text Full Retro C++',
                    'description': 'C++-specific retrieval of full problem descriptions from code',
                    'task_type': 'description_retrieval',
                    'text_type': 'full_description',
                    'language_constraint': 'cpp',
                    'count': 47
                },
                'c2t_full_retro_ruby': {
                    'name': 'Code2Text Full Retro Ruby',
                    'description': 'Ruby-specific retrieval of full problem descriptions from code',
                    'task_type': 'description_retrieval',
                    'text_type': 'full_description',
                    'language_constraint': 'ruby',
                    'count': 42
                },
                'c2t_full_retro_go': {
                    'name': 'Code2Text Full Retro Go',
                    'description': 'Go-specific retrieval of full problem descriptions from code',
                    'task_type': 'description_retrieval',
                    'text_type': 'full_description',
                    'language_constraint': 'go',
                    'count': 38
                },
                # Binary Matching subtask
                'c2t_match': {
                    'name': 'Code2Text Binary Matching',
                    'description': 'Binary classification task: predict if code and text match',
                    'task_type': 'binary_classification',
                    'text_type': 'mixed',
                    'language_constraint': 'none',
                    'count': 336
                }
            },
            'descriptive_stats': {
                'n_samples': {'test': 972},  # Total: 13 subtasks
                'avg_character_length': {'test': {'queries': 150, 'corpus': 600}},
                'subtask_counts': {
                    'canonical_retro': 318,  # 6 subtasks: any(84) + python(55) + java(52) + cpp(47) + ruby(42) + go(38)
                    'full_retro': 318,       # 6 subtasks: any(84) + python(55) + java(52) + cpp(47) + ruby(42) + go(38)
                    'binary_match': 336      # 1 subtask: c2t_match(336)
                }
            }
        }
    }
    
    def __init__(self, data_dir: str = './scripts/preprocess/ddb_storage'):
        """
        Initialize task definitions.
        
        Args:
            data_dir: Directory containing the benchmark data files
        """
        self.data_dir = Path(data_dir)
        
        # Initialize without CoIR dependency check
    
    def create_task(
        self,
        task_name: str,
        corpus_file: str,
        queries_file: str,
        qrels_file: str,
        subtasks: Optional[List[str]] = None
    ):
        """
        Create a CoIR/MTEB compatible task object.
        
        Args:
            task_name: Name of the task (text2code, code2code, code2text)
            corpus_file: Path to corpus JSONL file
            queries_file: Path to queries JSONL file
            qrels_file: Path to qrels JSONL file
            subtasks: Optional list of subtasks to filter
        
        Returns:
            Task object compatible with CoIR/MTEB
        """
        from coreb_runner.benchmark.data_loader import (
            load_custom_dataset,
            filter_queries_by_subtask,
            load_jsonl,
            convert_queries_to_coir_format,
            convert_qrels_to_coir_format
        )
        
        # Load data
        corpus, queries, qrels = load_custom_dataset(corpus_file, queries_file, qrels_file)
        
        # Filter by subtasks if specified
        if subtasks:
            logger.info(f"Filtering {task_name} to subtasks: {subtasks}")
            queries_raw = load_jsonl(queries_file)
            qrels_raw = load_jsonl(qrels_file)
            
            filtered_queries, filtered_qrels = filter_queries_by_subtask(
                queries_raw, qrels_raw, subtasks
            )
            
            queries = convert_queries_to_coir_format(filtered_queries)
            qrels = convert_qrels_to_coir_format(filtered_qrels)
        
        # Get metadata
        metadata = self.TASK_METADATA.get(task_name, {})
        task_display_name = metadata.get('name', f'CoREB-{task_name}')
        
        # Create Task object
        task = Task(
            name=task_display_name,
            corpus=corpus,
            queries=queries,
            qrels=qrels
        )
        
        # Add metadata attributes (for MTEB compatibility)
        for key, value in metadata.items():
            setattr(task, key, value)
        
        logger.info(f"Created task: {task_display_name}")
        logger.info(f"  • Corpus: {len(corpus)} documents")
        logger.info(f"  • Queries: {len(queries)} queries")
        logger.info(f"  • Qrels: {len(qrels)} query-doc pairs")
        
        return task
    
    def get_text2code_task(self, subtasks: Optional[List[str]] = None):
        """
        Get Text2Code retrieval task.
        
        Args:
            subtasks: Optional list of subtasks to include
                Examples: ['t2c_search_any', 't2c_canonical_retro_python']
        
        Returns:
            Task object
        """
        return self.create_task(
            task_name='text2code',
            corpus_file=str(self.data_dir / 'code_corpus_v2.jsonl'),
            queries_file=str(self.data_dir / 'text2code_queries.jsonl'),
            qrels_file=str(self.data_dir / 'text2code_qrels.jsonl'),
            subtasks=subtasks
        )
    
    def get_code2code_task(self, subtasks: Optional[List[str]] = None):
        """
        Get Code2Code similarity task.

        Each C2C query is generated from an anchor code item that is also present
        in the shared retrieval corpus.  When ranking, every model will place this
        anchor at rank 1 (cosine similarity = 1.0) because the query *is* the
        anchor.  However, the anchor is NOT marked as a positive in the qrels
        (only cross-language translations are positives), so rank 1 is always
        structurally "wasted".

        To produce fair metrics we attach a ``c2c_anchor_map`` dict
        (query_id -> anchor_code_id) to the returned Task.  Evaluation runners
        should filter out the anchor from each query's ranked list before
        computing nDCG / Recall / MRR.

        Args:
            subtasks: Optional list of subtasks to include
                Examples: ['c2c_mono_lang', 'c2c_cross_lang']

        Returns:
            Task object with an extra attribute ``c2c_anchor_map``
        """
        from coreb_runner.benchmark.data_loader import load_jsonl

        task = self.create_task(
            task_name='code2code',
            corpus_file=str(self.data_dir / 'code_corpus_v2.jsonl'),
            queries_file=str(self.data_dir / 'code2code_queries.jsonl'),
            qrels_file=str(self.data_dir / 'code2code_qrels.jsonl'),
            subtasks=subtasks
        )

        # Build query_id -> anchor_code_id map for post-hoc anchor exclusion.
        queries_raw = load_jsonl(str(self.data_dir / 'code2code_queries.jsonl'))
        anchor_map: Dict[str, str] = {}
        for q in queries_raw:
            meta = q.get('meta', {})
            if isinstance(meta, str):
                # meta stored as string repr — parse safely
                import re as _re
                m = _re.search(r"'anchor_code_id':\s*'([^']+)'", meta)
                anchor_id = m.group(1) if m else ''
            else:
                anchor_id = meta.get('anchor_code_id', '')
            if anchor_id:
                anchor_map[q['query_id']] = anchor_id

        task.c2c_anchor_map = anchor_map
        logger.info(
            f"  • C2C anchor map built for {len(anchor_map)} queries "
            f"(anchor will be excluded from ranked lists before metric computation)"
        )
        return task
    
    def get_code2text_task(self, subtasks: Optional[List[str]] = None):
        """
        Get Code2Text retrieval task.
        
        Args:
            subtasks: Optional list of subtasks to include
                Examples: ['c2t_canonical_retro_any', 'c2t_match']
        
        Returns:
            Task object
        """
        return self.create_task(
            task_name='code2text',
            corpus_file=str(self.data_dir / 'text_corpus_v2.jsonl'),
            queries_file=str(self.data_dir / 'code2text_queries.jsonl'),
            qrels_file=str(self.data_dir / 'code2text_qrels.jsonl'),
            subtasks=subtasks
        )
    
    def get_all_tasks(self):
        """
        Get all benchmark tasks.
        
        Returns:
            List of all task objects
        """
        return [
            self.get_text2code_task(),
            self.get_code2code_task(),
            self.get_code2text_task()
        ]
    
    def get_available_subtasks(self, task_name: str) -> List[str]:
        """
        Get available subtasks for a given task.
        
        Args:
            task_name: Name of the task ('text2code', 'code2code', 'code2text')
        
        Returns:
            List of available subtask names
        """
        metadata = self.TASK_METADATA.get(task_name, {})
        subtasks = metadata.get('subtasks', {})
        return list(subtasks.keys())
    
    def get_subtask_info(self, task_name: str, subtask_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific subtask.
        
        Args:
            task_name: Name of the task
            subtask_name: Name of the subtask
        
        Returns:
            Dictionary with subtask information
        """
        metadata = self.TASK_METADATA.get(task_name, {})
        subtasks = metadata.get('subtasks', {})
        return subtasks.get(subtask_name, {})
    
    def get_tasks_by_names(self, task_names: List[str]):
        """
        Get specific tasks by name.
        
        Args:
            task_names: List of task names ('text2code', 'code2code', 'code2text')
        
        Returns:
            List of task objects
        """
        tasks = []
        for name in task_names:
            if name == 'text2code':
                tasks.append(self.get_text2code_task())
            elif name == 'code2code':
                tasks.append(self.get_code2code_task())
            elif name == 'code2text':
                tasks.append(self.get_code2text_task())
            else:
                logger.warning(f"Unknown task name: {name}")
        
        return tasks


def get_coreb_tasks(
    task_names: Optional[List[str]] = None,
    data_dir: str = './scripts/preprocess/ddb_storage',
    subtasks: Optional[Dict[str, List[str]]] = None
):
    """
    Get CoREB (Code Retrieval Embedding Benchmark) tasks using CoIR framework.
    
    This function integrates CoREB tasks with the CoIR evaluation framework.
    
    Args:
        task_names: List of CoREB task names to load. If None, loads all tasks.
            Options: ['text2code', 'code2code', 'code2text']
        data_dir: Directory containing benchmark data
        subtasks: Optional dict mapping task names to subtask lists
            Example: {'text2code': ['t2c_canonical_retro_any'], 'code2code': ['c2c_mono_lang']}
    
    Returns:
        List of CoIR Task objects for evaluation
    
    Example:
        >>> # Get all CoREB tasks
        >>> tasks = get_coreb_tasks()
        >>> 
        >>> # Get specific tasks
        >>> tasks = get_coreb_tasks(task_names=['text2code', 'code2code'])
        >>> 
        >>> # Get tasks with subtask filtering
        >>> tasks = get_coreb_tasks(
        ...     task_names=['text2code'],
        ...     subtasks={'text2code': ['t2c_canonical_retro_any']}
        ... )
        >>> 
        >>> # Use with CoREB evaluation
        >>> from coreb_runner.benchmark.evaluation import CoREBEvaluation
        >>> evaluation = CoREBEvaluation(tasks=tasks, batch_size=128)
        >>> results = evaluation.run(model, output_folder="results")
    """
    logger.warning(
        "get_coreb_tasks() is deprecated. Use CoREBTasks directly to build "
        "task dicts from your own corpus/queries/qrels files."
    )
    return {}


def print_available_subtasks():
    """
    Print all available CoREB subtasks with their descriptions.
    """
    task_loader = CoREBTasks()
    
    print("🎯 CoREB (Code Retrieval Embedding Benchmark) Available Subtasks")
    print("=" * 60)
    
    for task_name in ['text2code', 'code2code', 'code2text']:
        metadata = task_loader.TASK_METADATA.get(task_name, {})
        print(f"\n📋 {metadata.get('name', f'CoREB-{task_name}')}")
        print(f"   {metadata.get('description', '')}")
        
        subtasks = metadata.get('subtasks', {})
        if subtasks:
            print("   Available subtasks:")
            for subtask_name, subtask_info in subtasks.items():
                print(f"   • {subtask_name}: {subtask_info.get('description', '')}")
        else:
            print("   No subtasks defined")
    
    print("\n" + "=" * 60)
    print("💡 Usage Examples:")
    print("   # Get all tasks")
    print("   tasks = get_coleb_tasks()")
    print("   ")
    print("   # Get specific tasks")
    print("   tasks = get_coleb_tasks(task_names=['text2code', 'code2code'])")
    print("   ")
    print("   # Filter by subtasks")
    print("   tasks = get_coleb_tasks(")
    print("       task_names=['text2code'],")
    print("       subtasks={'text2code': ['t2c_canonical_retro_any']}")
    print("   )")


# Backward compatibility aliases
get_code_emb_tasks = get_coreb_tasks
CodeEmbBenchmarkTasks = CoREBTasks

