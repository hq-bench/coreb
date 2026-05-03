"""
Unified Dataset Maker

Creates HuggingFace-compatible datasets for code embedding benchmarks:
1. text2code: Text queries to code retrieval
2. code2code: Code to code similarity
3. code2text: Code to text retrieval

Data Structure:
- Flexible, HuggingFace-compatible schema
- Language agnostic/specific query types
- Optional negatives for different evaluation scenarios
"""

import json
import os
import random
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

from easyllm_kit.utils import get_logger, read_json, save_json
from coreb_runner.runners.base_runner import Runner
from coreb_runner.utils.code_utils import load_jsonl

logger = get_logger('query_corpus_dataset_maker')


@Runner.register("query_corpus_dataset_maker")
class QueryCorpusDatasetMaker(Runner):
    """Unified dataset maker for code embedding benchmarks."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.data_config = config.get('data', {})
        self.dataset_config = config.get('dataset', {})
        
        # Input data paths
        self.queries_dir = self.data_config.get('queries_dir')
        self.code_corpus_dir = self.data_config.get('code_corpus_dir')
        self.text_corpus_dir = self.data_config.get('text_corpus_dir')
        self.abbrev_text_dir = self.data_config.get('abbrev_text_dir')
        
        # Output paths
        self.output_dir = self.data_config.get('output_dir')
        
        # Dataset configuration
        self.max_queries_per_type = self.dataset_config.get('max_queries_per_type', 100)
        self.random_seed = self.dataset_config.get('random_seed', 42)
        self.enabled_languages = self.dataset_config.get('enabled_languages', None)  # Subset of languages to use
        
        # Task generation flags (control which tasks to generate)
        self.generate_flags = self.dataset_config.get('generate_flags', {})
        self.generate_text2code = self.generate_flags.get('text2code', True)
        self.generate_text2code_retro = self.generate_flags.get('text2code_retro', True)
        self.generate_text2code_search = self.generate_flags.get('text2code_search', True)
        self.generate_code2code = self.generate_flags.get('code2code', True)
        self.generate_code2code_mono = self.generate_flags.get('code2code_mono_lang', True)
        self.generate_code2code_cross_lang = self.generate_flags.get('code2code_cross_lang', True)
        self.generate_code2code_cross_model = self.generate_flags.get('code2code_cross_model_cross_lang', True)
        self.generate_code2text = self.generate_flags.get('code2text', False)
        self.generate_code2text_any = self.generate_flags.get('code2text_any', True)
        self.generate_code2text_lang_specific = self.generate_flags.get('code2text_language_specific', True)
        self.generate_code2text_match = self.generate_flags.get('code2text_match', False)
        self.generate_code2text_cross_model = self.generate_flags.get('code2text_cross_model', False)
        self.generate_code2text_snippet = self.generate_flags.get('code2text_snippet', False)
        self.generate_code2text_multi = self.generate_flags.get('code2text_multi_solution', False)
        
        # Set random seed for reproducibility
        random.seed(self.random_seed)
        
        # Load data
        self._load_data()
        
        logger.info(f"QueryCorpusDatasetMaker initialized with {len(self.queries)} queries, "
                   f"{len(self.code_corpus)} code entries, {len(self.text_corpus)} text entries")

    def _load_data(self):
        """Load all required data files."""
        logger.info("Loading data files...")
        
        # Load queries (with preprocessing if needed)
        if self.queries_dir:
            logger.info(f"Loading queries from {self.queries_dir}")
            self.queries = read_json(self.queries_dir)
            logger.info(f"Preprocessed {len(self.queries)} queries")
            
        # Load code corpus
        if self.code_corpus_dir:
            logger.info(f"Loading code corpus from {self.code_corpus_dir}")
            self.code_corpus = load_jsonl(self.code_corpus_dir)
            
        # Load text corpus
        if self.text_corpus_dir:
            logger.info(f"Loading text corpus from {self.text_corpus_dir}")
            self.text_corpus = load_jsonl(self.text_corpus_dir)
            
        # Load abbreviated texts
        if self.abbrev_text_dir:
            logger.info(f"Loading abbreviated texts from {self.abbrev_text_dir}")
            self.abbrev_texts = read_json(self.abbrev_text_dir)
        
        # Build reusable indices for efficient lookups
        self._build_code_indices()
        self._build_text_indices()

    def _build_code_indices(self):
        """
        Build reusable indices from code corpus for efficient lookups.
        
        Creates:
        1. correct_code_by_problem: All correct solutions grouped by problem (nested by model and language)
        2. all_code_by_problem: All solutions (correct + incorrect) grouped by problem
        3. correct_code_flat_by_problem: Flat list of correct solutions by problem
        4. failed_code_by_problem: Failed solutions grouped by problem (nested by model and language)
        5. available_languages: Set of all languages in corpus
        """
        logger.info("📊 Building code indices for efficient lookups...")
        
        # 1. Correct code grouped by problem → model → language (for code2code tasks)
        self.correct_code_by_problem = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        
        # 2. All code grouped by problem (for text2code - includes both correct and incorrect)
        self.all_code_by_problem = defaultdict(list)
        
        # 3. Flat list of correct code by problem (for quick validation checks)
        self.correct_code_flat_by_problem = defaultdict(list)

        # 4. Failed code grouped by problem → model → language (for hard negatives)
        self.failed_code_by_problem = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        self.failed_code_flat_by_problem = defaultdict(list)

        # 5. Track available languages
        languages_set = set()

        correct_count = 0
        failed_count = 0
        for code_entry in self.code_corpus:
            problem_id = code_entry['meta']['source_problem_id']
            model = code_entry['model']
            language = code_entry['language']
            is_correct = code_entry['meta'].get('passed', False)
            
            # Add to all_code index
            self.all_code_by_problem[problem_id].append(code_entry)
            
            # Add to correct_code / failed_code indices
            if is_correct:
                self.correct_code_by_problem[problem_id][model][language].append(code_entry)
                self.correct_code_flat_by_problem[problem_id].append(code_entry)
                correct_count += 1
            else:
                self.failed_code_by_problem[problem_id][model][language].append(code_entry)
                self.failed_code_flat_by_problem[problem_id].append(code_entry)
                failed_count += 1

            # Track languages
            languages_set.add(language)
        
        # Apply language filtering if specified
        if self.enabled_languages:
            self.available_languages = sorted([lang for lang in languages_set if lang in self.enabled_languages])
            logger.info(f"  📌 Language filtering enabled: {self.enabled_languages}")
            logger.info(f"     Filtered from {len(languages_set)} to {len(self.available_languages)} languages")
        else:
            self.available_languages = sorted(languages_set)
        
        logger.info(f"  ✅ Built indices:")
        logger.info(f"     - {len(self.all_code_by_problem)} problems total")
        logger.info(f"     - {len(self.correct_code_flat_by_problem)} problems with correct solutions")
        logger.info(f"     - {correct_count} correct solutions")
        logger.info(f"     - {failed_count} failed solutions (available for hard negatives)")
        logger.info(f"     - {len(self.failed_code_flat_by_problem)} problems with failed solutions")
        logger.info(f"     - {len(self.available_languages)} languages (enabled): {self.available_languages}")

    def _build_text_indices(self):
        """Build noise text index: original_problem_id → [noise text entries]."""
        self.noise_text_by_problem = defaultdict(list)
        for text_entry in self.text_corpus:
            if text_entry['meta'].get('is_noise_sample'):
                orig_pid = text_entry['meta'].get('original_problem_id')
                if orig_pid:
                    self.noise_text_by_problem[orig_pid].append(text_entry)
        logger.info(f"  📊 Text indices: {len(self.noise_text_by_problem)} problems with noise texts, "
                    f"{sum(len(v) for v in self.noise_text_by_problem.values())} noise entries total")

    def run(self):
        """Generate unified datasets for all tasks based on configuration flags."""
        logger.info("🚀 Starting unified dataset generation")
        logger.info(f"  Generation flags: text2code={self.generate_text2code}, "
                   f"code2code={self.generate_code2code}, code2text={self.generate_code2text}")
        
        # Generate text2code datasets
        if self.generate_text2code:
            self._generate_text2code_datasets()
        else:
            logger.info("⏭️  Skipping text2code generation (disabled)")
        
        # Generate code2code datasets
        if self.generate_code2code:
            self._generate_code2code_datasets()
        else:
            logger.info("⏭️  Skipping code2code generation (disabled)")
        
        # Generate code2text datasets
        if self.generate_code2text:
            self._generate_code2text_datasets()
        else:
            logger.info("⏭️  Skipping code2text generation (disabled)")

        # Generate unified HuggingFace dataset
        self._generate_unified_huggingface_dataset()
        
        logger.info("✅ Dataset generation completed")

    def _generate_text2code_datasets(self):
        """
        Generate text2code datasets with separate queries and qrels files.

        Includes three subtasks:
        1. Canonical Retro (canonical_retro_any, canonical_retro_<lang>): Short canonical text → Code
        2. Full Retro (full_retro_any, full_retro_<lang>): Full detailed text → Code  
        3. Code Search (search_any, search_<lang>): Natural language diverse queries → Code
        """
        logger.info("📝 Generating text2code datasets...")
        
        all_queries = []
        all_qrels = []
        query_id_counter = 1

        # === PART 1: Canonical Retro Queries (from abbreviated/canonical texts) ===
        if self.generate_text2code_retro:
            logger.info("  📚 Part 1: Canonical Retro (canonical text → code)...")

            if self.abbrev_texts:
                canonical_queries, canonical_qrels = self._generate_canonical_retro_queries(query_id_counter)
                all_queries.extend(canonical_queries)
                all_qrels.extend(canonical_qrels)
                query_id_counter += len(canonical_queries) + 1
                logger.info(f"    Added {len(canonical_queries)} canonical_retro queries with {len(canonical_qrels)} qrels")
            else:
                logger.warning("    ⚠️  No abbreviated texts found, skipping Canonical Retro")
        else:
            logger.info("  ⏭️  Skipping Canonical Retro (disabled)")

        # === PART 2: Full Retro Queries (from full problem descriptions) ===
        if self.generate_text2code_retro:  # Use same flag as canonical_retro
            logger.info("  📖 Part 2: Full Retro (full text → code)...")
            
            full_queries, full_qrels = self._generate_full_retro_queries(query_id_counter)
            all_queries.extend(full_queries)
            all_qrels.extend(full_qrels)
            query_id_counter += len(full_queries) + 1
            logger.info(f"    Added {len(full_queries)} full_retro queries with {len(full_qrels)} qrels")
        else:
            logger.info("  ⏭️  Skipping Full Retro (disabled)")

        # === PART 3: Code Search Queries (from retro_queries.json) ===
        if self.generate_text2code_search:
            logger.info("  🔍 Part 3: Code Search (natural language queries → code)...")
            query_groups = self._parse_queries()
        
        for query_type, queries in query_groups.items():
            if not queries:
                continue
                
                logger.info(f"    Processing {len(queries)} {query_type} queries")
            
            # Limit queries per type
            if len(queries) > self.max_queries_per_type:
                queries = random.sample(queries, self.max_queries_per_type)
            
                # Generate query and qrel entries
            for query in queries:
                    query_entry, qrel_entries = self._create_text2code_entries(query, query_type, query_id_counter)
                    if query_entry and qrel_entries:
                        all_queries.append(query_entry)
                        all_qrels.extend(qrel_entries)
                        query_id_counter += 1
        else:
            logger.info("  ⏭️  Skipping Code Search queries (disabled)")

        # Save separate files with task prefix
        if all_queries:
            self._save_queries(all_queries, "text2code")
            logger.info(f"  ✅ Saved {len(all_queries)} total text2code queries")
        if all_qrels:
            self._save_qrels(all_qrels, "text2code")
            logger.info(f"  ✅ Saved {len(all_qrels)} total text2code qrels")

        # Also save HuggingFace-compatible format
        if all_queries and all_qrels:
            self._save_huggingface_format({
                "text2code": all_queries,
                "text2code_qrels": all_qrels
            }, "text2code_hf")

    def _generate_code2code_datasets(self):
        """
        Generate code2code similarity datasets with three subtasks.

        Subtasks:
        1. mono_lang_python: Same language (Python), different models (o1-mini vs claude-sonnet)
        2. cross_lang_claude: Different languages, same model (claude-sonnet only)
        3. cross_model_cross_lang: Different models AND different languages (o1-mini Python vs claude-sonnet non-Python)
        """
        logger.info("🔗 Generating code2code datasets...")
        
        # Use prebuilt index of correct solutions
        code_by_problem = self.correct_code_by_problem

        all_queries = []
        all_qrels = []
        query_id_counter = 1

        # === TASK 1: Mono-Language Similarity (Python only, cross-model) ===
        if self.generate_code2code_mono:
            logger.info("  📚 Task 1: Mono-Language Similarity (Python, o1-mini vs claude-sonnet)...")
            task1_queries, task1_qrels = self._generate_mono_lang_similarity(
                code_by_problem, query_id_counter
            )
            all_queries.extend(task1_queries)
            all_qrels.extend(task1_qrels)
            query_id_counter += len(task1_queries) + 1
            logger.info(f"    Added {len(task1_queries)} queries with {len(task1_qrels)} qrels")
        else:
            logger.info("  ⏭️  Skipping Mono-Language Similarity (disabled)")

        # === TASK 2: Cross-Language Similarity (Claude-sonnet only) ===
        if self.generate_code2code_cross_lang:
            logger.info("  🌐 Task 2: Cross-Language Similarity (claude-sonnet, Python vs others)...")
            task2_queries, task2_qrels = self._generate_cross_lang_similarity(
                code_by_problem, query_id_counter
            )
            all_queries.extend(task2_queries)
            all_qrels.extend(task2_qrels)
            query_id_counter += len(task2_queries) + 1
            logger.info(f"    Added {len(task2_queries)} queries with {len(task2_qrels)} qrels")
        else:
            logger.info("  ⏭️  Skipping Cross-Language Similarity (disabled)")

        # === TASK 3: Multi-Model Cross-Language Similarity ===
        if self.generate_code2code_cross_model:
            logger.info("  🔀 Task 3: Multi-Model Cross-Language (o1-mini Python vs claude-sonnet non-Python)...")
            task3_queries, task3_qrels = self._generate_cross_model_cross_lang_similarity(
                code_by_problem, query_id_counter
            )
            all_queries.extend(task3_queries)
            all_qrels.extend(task3_qrels)
            logger.info(f"    Added {len(task3_queries)} queries with {len(task3_qrels)} qrels")
        else:
            logger.info("  ⏭️  Skipping Multi-Model Cross-Language Similarity (disabled)")

        # Log relevance distribution
        from collections import Counter
        rel_dist = Counter(q['relevance'] for q in all_qrels)
        logger.info(f"  📊 C2C qrel breakdown: {rel_dist.get(2, 0)} positives (rel=2), "
                    f"{rel_dist.get(1, 0)} hard negatives (rel=1), {rel_dist.get(0, 0)} unrelated (rel=0)")

        # Save queries and qrels
        if all_queries:
            self._save_queries(all_queries, "code2code")
            logger.info(f"  ✅ Saved {len(all_queries)} total code2code queries")
        if all_qrels:
            self._save_qrels(all_qrels, "code2code")
            logger.info(f"  ✅ Saved {len(all_qrels)} total code2code qrels")

        # Also save HuggingFace-compatible format
        if all_queries and all_qrels:
            self._save_huggingface_format({
                "code2code": all_queries,
                "code2code_qrels": all_qrels
            }, "code2code_hf")

    def _generate_code2text_datasets(self):
        """
        Generate code2text datasets (code → text description retrieval).

        Five main subtasks:
        
        1. c2t_full_retro_any: Code → Full description (language-agnostic, 1 query per problem)
        2. c2t_full_retro_<lang>: Code → Full description (language-specific)
        3. c2t_canonical_retro_any: Code → Canonical description (language-agnostic)
        4. c2t_canonical_retro_<lang>: Code → Canonical description (language-specific)
        5. c2t_match: Binary classification - (Code, Text) → match / not match
        """
        logger.info("📄 Generating code2text datasets (code → text retrieval)...")

        # Use prebuilt index of correct solutions (flat list)
        code_by_problem = self.correct_code_flat_by_problem

        # Create text lookup (text_id by problem_id) for FULL descriptions
        text_by_problem = {}
        for text_entry in self.text_corpus:
            problem_id = text_entry['meta']['source_problem_id']
            text_by_problem[problem_id] = text_entry['text_id']

        # Create text lookup for CANONICAL (abbreviated) descriptions
        abbrev_text_by_problem = {}
        if self.abbrev_texts:
            for problem_id, abbrev_data in self.abbrev_texts.items():
                if isinstance(abbrev_data, dict):
                    abbrev_text_by_problem[problem_id] = abbrev_data.get('abbreviated', '')
                else:
                    abbrev_text_by_problem[problem_id] = abbrev_data

        logger.info(f"  Found {len(code_by_problem)} problems with correct code solutions")
        logger.info(f"  Found {len(text_by_problem)} full text descriptions")
        logger.info(f"  Found {len(abbrev_text_by_problem)} canonical (abbreviated) texts")

        all_queries = []
        all_qrels = []
        query_id_counter = 1

        # === PART 1: Full Retro (Code → Full Description) ===
        if self.generate_code2text_any:
            logger.info("  📖 Part 1: Full Retro (code → full description)...")
            
            # 1a. Language-agnostic
            full_any_queries, full_any_qrels = self._generate_c2t_full_retro_any(
                code_by_problem, text_by_problem, query_id_counter
            )
            all_queries.extend(full_any_queries)
            all_qrels.extend(full_any_qrels)
            query_id_counter += len(full_any_queries) + 1
            logger.info(f"    Generated {len(full_any_queries)} c2t_full_retro_any queries")
            
            # 1b. Language-specific
            full_lang_queries, full_lang_qrels = self._generate_c2t_full_retro_lang_specific(
                code_by_problem, text_by_problem, query_id_counter
            )
            all_queries.extend(full_lang_queries)
            all_qrels.extend(full_lang_qrels)
            query_id_counter += len(full_lang_queries) + 1
            logger.info(f"    Generated {len(full_lang_queries)} c2t_full_retro_<lang> queries")
        else:
            logger.info("  ⏭️  Skipping Full Retro (disabled)")

        # === PART 2: Canonical Retro (Code → Canonical/Short Description) ===
        if self.generate_code2text_lang_specific and abbrev_text_by_problem:
            logger.info("  📚 Part 2: Canonical Retro (code → canonical description)...")
            
            # 2a. Language-agnostic
            canonical_any_queries, canonical_any_qrels = self._generate_c2t_canonical_retro_any(
                code_by_problem, text_by_problem, abbrev_text_by_problem, query_id_counter
            )
            all_queries.extend(canonical_any_queries)
            all_qrels.extend(canonical_any_qrels)
            query_id_counter += len(canonical_any_queries) + 1
            logger.info(f"    Generated {len(canonical_any_queries)} c2t_canonical_retro_any queries")
            
            # 2b. Language-specific
            canonical_lang_queries, canonical_lang_qrels = self._generate_c2t_canonical_retro_lang_specific(
                code_by_problem, text_by_problem, abbrev_text_by_problem, query_id_counter
            )
            all_queries.extend(canonical_lang_queries)
            all_qrels.extend(canonical_lang_qrels)
            query_id_counter += len(canonical_lang_queries) + 1
            logger.info(f"    Generated {len(canonical_lang_queries)} c2t_canonical_retro_<lang> queries")
        else:
            if not abbrev_text_by_problem:
                logger.warning("  ⚠️  No canonical texts found, skipping Canonical Retro")
            else:
                logger.info("  ⏭️  Skipping Canonical Retro (disabled)")

        # === PART 3: Binary Matching (Code-Text Pairs → match / not match) ===
        if self.generate_code2text_match:
            logger.info("  🎯 Part 3: Binary Matching (code-text pair classification)...")
            
            match_queries, match_qrels = self._generate_c2t_match(
                code_by_problem, text_by_problem, abbrev_text_by_problem, query_id_counter
            )
            all_queries.extend(match_queries)
            all_qrels.extend(match_qrels)
            query_id_counter += len(match_queries) + 1
            logger.info(f"    Generated {len(match_queries)} c2t_match queries")
        else:
            logger.info("  ⏭️  Skipping Binary Matching (disabled)")

        # === PART 4: Cross-Model Variation (Code from different models → same text) ===
        if self.generate_code2text_cross_model:
            logger.info("  🤖 Part 4: Cross-Model Variation (different models' code → same text)...")
            
            cross_model_queries, cross_model_qrels = self._generate_c2t_cross_model(
                code_by_problem, text_by_problem, abbrev_text_by_problem, query_id_counter
            )
            all_queries.extend(cross_model_queries)
            all_qrels.extend(cross_model_qrels)
            query_id_counter += len(cross_model_queries) + 1
            logger.info(f"    Generated {len(cross_model_queries)} c2t_cross_model queries")
        else:
            logger.info("  ⏭️  Skipping Cross-Model Variation (disabled)")

        # Save queries and qrels
        if all_queries:
            self._save_queries(all_queries, "code2text")
            logger.info(f"  ✅ Saved {len(all_queries)} total code2text queries")
        if all_qrels:
            self._save_qrels(all_qrels, "code2text")
            logger.info(f"  ✅ Saved {len(all_qrels)} total code2text qrels")

        # Save HuggingFace format
        if all_queries and all_qrels:
            self._save_huggingface_format({
                "code2text": all_queries,
                "code2text_qrels": all_qrels
            }, "code2text_hf")

    def _generate_mono_lang_similarity(self, code_by_problem: Dict, start_id: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Generate mono-language similarity queries (Python only, cross-model).

        Anchor: Python from o1-mini
        Positives: Python from claude-sonnet (same problem)
        """
        queries = []
        qrels = []
        query_id_counter = start_id

        for problem_id, models in code_by_problem.items():
            # Check if we have Python solutions from both models
            o1_python = models.get('gemini-3-flash', {}).get('python', [])
            claude_python = models.get('claude-sonnet-4-5', {}).get('python', [])

            if not o1_python or not claude_python:
                continue
                
            # Use first o1-mini Python solution as anchor
            anchor = o1_python[0]

            query_id = f"q_c2c_mono_lang_{query_id_counter:04d}"
            query_id_counter += 1

            query = {
                'query_id': query_id,
                'query': anchor['code'],
                'split': 'code2code',
                'subtask': 'c2c_mono_lang',
                'anchor_language': 'python',
                    'anchor_model': anchor['model'],
                'meta': {
                    'source_problem_id': problem_id,
                    'anchor_code_id': anchor['code_id'],
                    'created_at': datetime.utcnow().isoformat() + 'Z',
                    'generator_version': 'v1.0'
                }
            }
            queries.append(query)

            # Create qrels for all claude Python solutions (true positives)
            positive_ids = set()
            for positive in claude_python:
                qrel = {
                    'query_id': query_id,
                    'doc_id': positive['code_id'],
                    'relevance': 2
                }
                qrels.append(qrel)
                positive_ids.add(positive['code_id'])

            # Add hard negatives
            hard_negs = self._select_c2c_hard_negatives(
                problem_id=problem_id,
                anchor_code_id=anchor['code_id'],
                anchor_language='python',
                anchor_model=anchor['model'],
                subtask='c2c_mono_lang',
                positive_doc_ids=positive_ids,
                target_count=3,
            )
            for neg in hard_negs:
                qrels.append({
                    'query_id': query_id,
                    'doc_id': neg['doc_id'],
                    'relevance': neg['relevance'],
                })

        return queries, qrels

    def _generate_cross_lang_similarity(self, code_by_problem: Dict, start_id: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Generate cross-language similarity queries (claude-sonnet only).

        Create unique pairs of correct solutions in different languages.
        Each pair (Solution A, Solution B) is represented once, avoiding duplicates like (B, A).
        
        Example: Problem with Python sol 1, Python sol 2, Java sol 1:
        - Query 1: Python sol 1 → Python sol 2 is SKIPPED (same language)
        - Query 2: Python sol 1 → Java sol 1 (cross-language) ✅
        - Query 3: Python sol 2 → Java sol 1 is SKIPPED (duplicate of Query 2 reverse) ❌
        
        Strategy: Only create query if anchor code_id < target code_id (lexicographically)
        """
        queries = []
        qrels = []
        query_id_counter = start_id

        for problem_id, models in code_by_problem.items():
            # Get all claude solutions (flattened list with metadata)
            all_solutions = []
            for model_name in ['claude-sonnet-4-5']:
                if model_name in models:
                    for lang, codes in models[model_name].items():
                        for code in codes:
                            all_solutions.append({
                                'code': code['code'],
                            'code_id': code['code_id'],
                                'language': lang,
                            'model': code['model']
                            })

            # Need at least 2 solutions with different languages
            if len(all_solutions) < 2:
                continue
            
            # Check if we have multiple languages
            languages = set(sol['language'] for sol in all_solutions)
            if len(languages) < 2:
                continue

            # Generate unique cross-language pairs
            # For each solution, pair it with all other solutions in different languages
            # Use lexicographic ordering to avoid duplicates
            for i, anchor in enumerate(all_solutions):
                anchor_lang = anchor['language']
                anchor_id = anchor['code_id']
                
                # Find all target solutions in different languages that come after this one
                # (to avoid creating both A→B and B→A)
                targets = []
                for j, target in enumerate(all_solutions):
                    # Skip same language
                    if target['language'] == anchor_lang:
                        continue
                    
                    # To avoid duplicates, only create pair if anchor_id < target_id
                    # This ensures we only get one direction per pair
                    if anchor_id < target['code_id']:
                        targets.append(target)
                
                if not targets:
                    continue
                
                # Create query for this anchor with its valid targets
                query_id = f"q_c2c_cross_lang_{query_id_counter:04d}"
                query_id_counter += 1
                
                target_languages = sorted(set(t['language'] for t in targets))
                
                query = {
                    'query_id': query_id,
                    'query': anchor['code'],
                    'split': 'code2code',
                    'subtask': 'c2c_cross_lang',
                    'anchor_language': anchor_lang,
                    'anchor_model': anchor['model'],
                    'meta': {
                        'source_problem_id': problem_id,
                        'anchor_code_id': anchor_id,
                        'target_languages': target_languages,
                        'num_target_solutions': len(targets),
                        'created_at': datetime.utcnow().isoformat() + 'Z',
                        'generator_version': 'v1.0'
                    }
                }
                queries.append(query)
                
                # Create qrels for all target solutions (true positives)
                positive_ids = set()
                for target in targets:
                    qrel = {
                        'query_id': query_id,
                        'doc_id': target['code_id'],
                        'relevance': 2
                    }
                    qrels.append(qrel)
                    positive_ids.add(target['code_id'])

                # Add hard negatives
                hard_negs = self._select_c2c_hard_negatives(
                    problem_id=problem_id,
                    anchor_code_id=anchor_id,
                    anchor_language=anchor_lang,
                    anchor_model=anchor['model'],
                    subtask='c2c_cross_lang',
                    positive_doc_ids=positive_ids,
                    target_count=3,
                )
                for neg in hard_negs:
                    qrels.append({
                        'query_id': query_id,
                        'doc_id': neg['doc_id'],
                        'relevance': neg['relevance'],
                    })

        return queries, qrels

    def _generate_cross_model_cross_lang_similarity(self, code_by_problem: Dict, start_id: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Generate multi-model cross-language similarity queries.

        For each o1-mini Python solution, create a query that pairs it with
        ALL claude-sonnet solutions in non-Python languages.
        
        Example: Problem with 2 o1-mini Python solutions, 1 Claude Java, 1 Claude C++:
        - Query 1: o1-mini Python sol 1 → Claude Java, Claude C++
        - Query 2: o1-mini Python sol 2 → Claude Java, Claude C++
        
        This tests cross-model AND cross-language similarity simultaneously.
        """
        queries = []
        qrels = []
        query_id_counter = start_id

        for problem_id, models in code_by_problem.items():
            # Get ALL gemini-3-flash Python solutions
            o1_python = models.get('gemini-3-flash', {}).get('python', [])
            if not o1_python:
                continue

            # Get all non-Python claude solutions
            claude_non_python = []
            for model_name in ['claude-sonnet-4-5']:
                if model_name in models:
                    for lang, codes in models[model_name].items():
                        if lang != 'python':
                            claude_non_python.extend(codes)

            if not claude_non_python:
                continue

            # Create one query per o1-mini Python solution
            for anchor in o1_python:
                query_id = f"q_c2c_cross_model_{query_id_counter:04d}"
                query_id_counter += 1

                # Get target languages
                target_langs = sorted(set(code['language'] for code in claude_non_python))

                query = {
                    'query_id': query_id,
                    'query': anchor['code'],
                    'split': 'code2code',
                    'subtask': 'c2c_cross_model',
                    'anchor_language': 'python',
                    'anchor_model': anchor['model'],
                    'meta': {
                        'source_problem_id': problem_id,
                        'anchor_code_id': anchor['code_id'],
                        'target_languages': target_langs,
                        'num_target_solutions': len(claude_non_python),
                        'created_at': datetime.utcnow().isoformat() + 'Z',
                        'generator_version': 'v1.0'
                    }
                }
                queries.append(query)

                # Create qrels for all non-Python claude solutions (true positives)
                positive_ids = set()
                for positive in claude_non_python:
                    qrel = {
                        'query_id': query_id,
                        'doc_id': positive['code_id'],
                        'relevance': 2
                    }
                    qrels.append(qrel)
                    positive_ids.add(positive['code_id'])

                # Add hard negatives
                hard_negs = self._select_c2c_hard_negatives(
                    problem_id=problem_id,
                    anchor_code_id=anchor['code_id'],
                    anchor_language='python',
                    anchor_model=anchor['model'],
                    subtask='c2c_cross_model',
                    positive_doc_ids=positive_ids,
                    target_count=3,
                )
                for neg in hard_negs:
                    qrels.append({
                        'query_id': query_id,
                        'doc_id': neg['doc_id'],
                        'relevance': neg['relevance'],
                    })

        return queries, qrels

    def _select_c2c_hard_negatives(
        self,
        problem_id: str,
        anchor_code_id: str,
        anchor_language: str,
        anchor_model: str,
        subtask: str,
        positive_doc_ids: set,
        target_count: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Select hard negative qrels for a C2C query.

        Returns a list of dicts with 'doc_id' and 'relevance':
        - relevance=1: same problem, hard negative (failed solution OR correct but
          not matching positive criteria, e.g. wrong model/language for the subtask)

        Selection priority:
        1. Constraint-matching failed solutions (failed, same problem, matches subtask filter)
        2. Any failed solution for the same problem (relaxed constraints)
        3. Correct solutions for the same problem that are NOT positives (confusable)
        """
        exclude_ids = {anchor_code_id} | positive_doc_ids
        failed_for_problem = self.failed_code_by_problem.get(problem_id, {})
        correct_for_problem = self.correct_code_by_problem.get(problem_id, {})

        # Priority 1: constraint-matching failed solutions
        candidates = []
        if subtask == 'c2c_cross_lang':
            for model_name, langs in failed_for_problem.items():
                for lang, codes in langs.items():
                    if lang != anchor_language:
                        candidates.extend(c for c in codes if c['code_id'] not in exclude_ids)
        elif subtask == 'c2c_mono_lang':
            for model_name, langs in failed_for_problem.items():
                if model_name != anchor_model:
                    for lang, codes in langs.items():
                        if lang == anchor_language:
                            candidates.extend(c for c in codes if c['code_id'] not in exclude_ids)
        elif subtask == 'c2c_cross_model':
            for model_name, langs in failed_for_problem.items():
                if model_name != anchor_model:
                    for lang, codes in langs.items():
                        if lang != anchor_language:
                            candidates.extend(c for c in codes if c['code_id'] not in exclude_ids)

        # Priority 2: any failed solution for the same problem
        if len(candidates) < target_count:
            seen = {c['code_id'] for c in candidates}
            all_failed = self.failed_code_flat_by_problem.get(problem_id, [])
            for c in all_failed:
                if c['code_id'] not in exclude_ids and c['code_id'] not in seen:
                    candidates.append(c)
                    seen.add(c['code_id'])

        # Priority 3: correct solutions for the same problem that are NOT positives
        # These are confusable: same problem, correct code, but wrong model/language for the subtask
        if len(candidates) < target_count:
            seen = {c['code_id'] for c in candidates}
            for model_name, langs in correct_for_problem.items():
                for lang, codes in langs.items():
                    for c in codes:
                        if c['code_id'] not in exclude_ids and c['code_id'] not in seen:
                            candidates.append(c)
                            seen.add(c['code_id'])

        # Sample hard negatives (relevance=1)
        if len(candidates) > target_count:
            candidates = random.sample(candidates, target_count)

        return [{'doc_id': c['code_id'], 'relevance': 1} for c in candidates]

    def _select_t2c_hard_negatives(
        self,
        problem_id: str,
        positive_doc_ids: set,
        language_constraint: Optional[str] = None,
        target_count: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Select hard negative code entries for a T2C query.

        Returns a list of dicts with 'doc_id' and 'relevance':
        - relevance=1: same problem, failed code solution (hard negative)

        Priority:
        1. Failed solutions matching the language constraint (if any)
        2. Any failed solution for the same problem
        3. Correct solutions for the same problem not in the positive set
        """
        exclude_ids = set(positive_doc_ids)
        failed_for_problem = self.failed_code_by_problem.get(problem_id, {})

        # Priority 1: constraint-matching failed solutions
        candidates = []
        if language_constraint:
            for model_name, langs in failed_for_problem.items():
                for lang, codes in langs.items():
                    if lang == language_constraint:
                        candidates.extend(c for c in codes if c['code_id'] not in exclude_ids)
        else:
            for model_name, langs in failed_for_problem.items():
                for lang, codes in langs.items():
                    candidates.extend(c for c in codes if c['code_id'] not in exclude_ids)

        # Priority 2: any failed solution for the same problem (relaxed constraint)
        if len(candidates) < target_count and language_constraint:
            seen = {c['code_id'] for c in candidates}
            all_failed = self.failed_code_flat_by_problem.get(problem_id, [])
            for c in all_failed:
                if c['code_id'] not in exclude_ids and c['code_id'] not in seen:
                    candidates.append(c)
                    seen.add(c['code_id'])

        # Priority 3: correct solutions not in positive set
        if len(candidates) < target_count:
            seen = {c['code_id'] for c in candidates}
            correct_for_problem = self.correct_code_by_problem.get(problem_id, {})
            for model_name, langs in correct_for_problem.items():
                for lang, codes in langs.items():
                    for c in codes:
                        if c['code_id'] not in exclude_ids and c['code_id'] not in seen:
                            candidates.append(c)
                            seen.add(c['code_id'])

        if len(candidates) > target_count:
            candidates = random.sample(candidates, target_count)

        return [{'doc_id': c['code_id'], 'relevance': 1} for c in candidates]

    def _select_c2t_hard_negatives(
        self,
        problem_id: str,
        positive_doc_ids: set,
        target_count: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Select hard negative text entries for a C2T query.

        Returns a list of dicts with 'doc_id' and 'relevance':
        - relevance=1: noise text for the same problem (hard negative)

        Priority:
        1. Noise text entries for the same problem (4 available per problem)
        2. Noise text entries from other problems
        """
        exclude_ids = set(positive_doc_ids)

        # Priority 1: noise texts for the same problem
        candidates = [
            t for t in self.noise_text_by_problem.get(problem_id, [])
            if t['text_id'] not in exclude_ids
        ]

        # Priority 2: noise texts from other problems
        if len(candidates) < target_count:
            seen = {c['text_id'] for c in candidates}
            other_noise = []
            for pid, texts in self.noise_text_by_problem.items():
                if pid != problem_id:
                    other_noise.extend(t for t in texts if t['text_id'] not in exclude_ids and t['text_id'] not in seen)
            if other_noise:
                needed = target_count - len(candidates)
                candidates.extend(random.sample(other_noise, min(needed, len(other_noise))))

        if len(candidates) > target_count:
            candidates = random.sample(candidates, target_count)

        return [{'doc_id': c['text_id'], 'relevance': 1} for c in candidates]

    def _generate_c2t_full_retro_any(self, code_by_problem: Dict, text_by_problem: Dict, start_id: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        c2t_full_retro_any: Code → Full Description (Language-Agnostic)
        
        Goal: Given a code snippet, retrieve the full/complete problem description.
        Setup: 1 query per problem (use best quality code available)
        
        Note: Uses FULL text description (comprehensive, ~1500 chars).
        Counterpart to c2t_canonical_retro_any (which uses canonical/abbreviated text, ~600 chars).
        
        Selection strategy: Prefer Python from Claude-Sonnet for consistency.
        Direction: Code (query) → Full Text Description (document in text corpus)
        """
        queries = []
        qrels = []
        query_id_counter = start_id

        for problem_id, codes in code_by_problem.items():
            if problem_id not in text_by_problem:
                continue
                
            # Select anchor code: prefer Python from claude-sonnet
            anchor = self._select_anchor_code(codes, prefer_lang='python', prefer_model='claude')
            if not anchor:
                continue

            query_id = f"q_c2t_full_retro_any_{query_id_counter:04d}"
            query_id_counter += 1

            query = {
                'query_id': query_id,
                'query': anchor['code'],  # Code as query
                'split': 'code2text',
                'subtask': 'c2t_full_retro_any',
                'anchor_language': anchor['language'],
                'anchor_model': anchor['model'],
                'meta': {
                    'source_problem_id': problem_id,
                    'anchor_code_id': anchor['code_id'],
                    'task_type': 'description_retrieval',
                    'text_type': 'full_description',
                    'created_at': datetime.utcnow().isoformat() + 'Z',
                    'generator_version': 'v1.0'
                }
            }
            queries.append(query)

            # Create qrel - text as document (true positive)
            pos_doc_id = text_by_problem[problem_id]
            qrel = {
                'query_id': query_id,
                'doc_id': pos_doc_id,
                'relevance': 2
            }
            qrels.append(qrel)

            # Add hard negatives (noise texts for the same problem)
            hard_negs = self._select_c2t_hard_negatives(
                problem_id=problem_id,
                positive_doc_ids={pos_doc_id},
                target_count=3,
            )
            for neg in hard_negs:
                qrels.append({
                    'query_id': query_id,
                    'doc_id': neg['doc_id'],
                    'relevance': neg['relevance'],
                })

        return queries, qrels

    def _generate_c2t_full_retro_lang_specific(self, code_by_problem: Dict, text_by_problem: Dict, start_id: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        c2t_full_retro_<lang>: Code → Full Description (Language-Specific)
        
        Goal: Test if embeddings are language-invariant - code in different languages
              for the same problem should all retrieve the same text description.
        
        Setup: For each problem with multiple language solutions, create 1 query
               per language. All should map to the same full text description.
               
        Direction: Code (query) → Full Text (document in text corpus)
        """
        queries = []
        qrels = []
        query_id_counter = start_id

        for problem_id, codes in code_by_problem.items():
            if problem_id not in text_by_problem:
                continue

            # Group codes by language for this problem
            by_lang = {}
            for code in codes:
                lang = code['language']
                if lang not in by_lang:
                    by_lang[lang] = []
                by_lang[lang].append(code)

            # Only create cross-lang queries if we have multiple languages
            if len(by_lang) < 2:
                continue

            # Create one query per language
            for lang, lang_codes in by_lang.items():
                # Use first solution in this language
                anchor = lang_codes[0]

                query_id = f"q_c2t_full_retro_{lang}_{query_id_counter:04d}"
                query_id_counter += 1

                query = {
                    'query_id': query_id,
                    'query': anchor['code'],  # Code as query
                    'split': 'code2text',
                    'subtask': f'c2t_full_retro_{lang}',
                    'anchor_language': lang,
                    'anchor_model': anchor['model'],
                'meta': {
                        'source_problem_id': problem_id,
                        'anchor_code_id': anchor['code_id'],
                        'task_type': 'cross_language_invariance',
                        'text_type': 'full_description',
                        'available_languages': sorted(by_lang.keys()),
                    'created_at': datetime.utcnow().isoformat() + 'Z',
                    'generator_version': 'v1.0'
                }
            }
                queries.append(query)

                # Create qrel - same text for all languages (true positive)
                pos_doc_id = text_by_problem[problem_id]
                qrel = {
                    'query_id': query_id,
                    'doc_id': pos_doc_id,
                    'relevance': 2
                }
                qrels.append(qrel)

                # Add hard negatives
                hard_negs = self._select_c2t_hard_negatives(
                    problem_id=problem_id,
                    positive_doc_ids={pos_doc_id},
                    target_count=3,
                )
                for neg in hard_negs:
                    qrels.append({
                        'query_id': query_id,
                        'doc_id': neg['doc_id'],
                        'relevance': neg['relevance'],
                    })

        return queries, qrels

    def _generate_c2t_canonical_retro_any(self, code_by_problem: Dict, text_by_problem: Dict, abbrev_text_by_problem: Dict, start_id: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        c2t_canonical_retro_any: Code → Canonical Description (Language-Agnostic)
        
        Goal: Given a code snippet, retrieve the canonical/abbreviated problem description.
        Setup: 1 query per problem (use best quality code available)
        
        Note: Uses CANONICAL/ABBREVIATED text (short, ~600 chars).
        Counterpart to c2t_full_retro_any (which uses full description, ~1500 chars).
        
        Selection strategy: Prefer Python from Claude-Sonnet for consistency.
        Direction: Code (query) → Canonical Text (document - abbreviated text stored in text corpus)
        
        Fix: Use actual text_id from text corpus instead of virtual canonical_{problem_id} ID.
        Canonical texts are stored as abbreviated_text field in text corpus entries.
        """
        queries = []
        qrels = []
        query_id_counter = start_id

        for problem_id, codes in code_by_problem.items():
            if problem_id not in abbrev_text_by_problem:
                continue
            
            # Must have both abbreviated text AND text corpus entry
            if problem_id not in text_by_problem:
                continue
                
            canonical_text = abbrev_text_by_problem[problem_id]
            if not canonical_text:
                continue
                
            # Select anchor code: prefer Python from claude-sonnet
            anchor = self._select_anchor_code(codes, prefer_lang='python', prefer_model='claude')
            if not anchor:
                continue

            query_id = f"q_c2t_canonical_retro_any_{query_id_counter:04d}"
            query_id_counter += 1

            query = {
                'query_id': query_id,
                'query': anchor['code'],  # Code as query
                'split': 'code2text',
                'subtask': 'c2t_canonical_retro_any',
                'anchor_language': anchor['language'],
                'anchor_model': anchor['model'],
                'meta': {
                    'source_problem_id': problem_id,
                    'anchor_code_id': anchor['code_id'],
                    'task_type': 'description_retrieval',
                    'text_type': 'canonical_description',
                    'created_at': datetime.utcnow().isoformat() + 'Z',
                    'generator_version': 'v1.0'
                }
            }
            queries.append(query)

            # Create qrel - use actual text_id from text corpus (true positive)
            pos_doc_id = text_by_problem[problem_id]
            qrel = {
                'query_id': query_id,
                'doc_id': pos_doc_id,
                'relevance': 2
            }
            qrels.append(qrel)

            # Add hard negatives
            hard_negs = self._select_c2t_hard_negatives(
                problem_id=problem_id,
                positive_doc_ids={pos_doc_id},
                target_count=3,
            )
            for neg in hard_negs:
                qrels.append({
                    'query_id': query_id,
                    'doc_id': neg['doc_id'],
                    'relevance': neg['relevance'],
                })

        return queries, qrels

    def _generate_c2t_canonical_retro_lang_specific(self, code_by_problem: Dict, text_by_problem: Dict, abbrev_text_by_problem: Dict, start_id: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        c2t_canonical_retro_<lang>: Code → Canonical Description (Language-Specific)
        
        Goal: Test if embeddings are language-invariant - code in different languages
              for the same problem should all retrieve the same canonical description.
        
        Setup: For each problem with multiple language solutions, create 1 query
               per language. All should map to the same canonical text.
               
        Direction: Code (query) → Canonical Text (document - abbreviated text stored in text corpus)
        
        Fix: Use actual text_id from text corpus instead of virtual canonical_{problem_id} ID.
        Canonical texts are stored as abbreviated_text field in text corpus entries.
        """
        queries = []
        qrels = []
        query_id_counter = start_id

        for problem_id, codes in code_by_problem.items():
            if problem_id not in abbrev_text_by_problem:
                continue
            
            # Must have both abbreviated text AND text corpus entry
            if problem_id not in text_by_problem:
                continue
                
            canonical_text = abbrev_text_by_problem[problem_id]
            if not canonical_text:
                continue

            # Group codes by language for this problem
            by_lang = {}
            for code in codes:
                lang = code['language']
                if lang not in by_lang:
                    by_lang[lang] = []
                by_lang[lang].append(code)

            # Only create cross-lang queries if we have multiple languages
            if len(by_lang) < 2:
                continue

            # Get text_id for this problem (same for all language variants)
            text_id = text_by_problem[problem_id]

            # Create one query per language
            for lang, lang_codes in by_lang.items():
                # Use first solution in this language
                anchor = lang_codes[0]

                query_id = f"q_c2t_canonical_retro_{lang}_{query_id_counter:04d}"
                query_id_counter += 1

                query = {
                    'query_id': query_id,
                    'query': anchor['code'],  # Code as query
                    'split': 'code2text',
                    'subtask': f'c2t_canonical_retro_{lang}',
                    'anchor_language': lang,
                    'anchor_model': anchor['model'],
                    'meta': {
                        'source_problem_id': problem_id,
                        'anchor_code_id': anchor['code_id'],
                        'task_type': 'cross_language_invariance',
                        'text_type': 'canonical_description',
                        'available_languages': sorted(by_lang.keys()),
                        'created_at': datetime.utcnow().isoformat() + 'Z',
                        'generator_version': 'v1.0'
                    }
                }
                queries.append(query)

                # Create qrel (true positive)
                qrel = {
                    'query_id': query_id,
                    'doc_id': text_id,
                    'relevance': 2
                }
                qrels.append(qrel)

                # Add hard negatives
                hard_negs = self._select_c2t_hard_negatives(
                    problem_id=problem_id,
                    positive_doc_ids={text_id},
                    target_count=3,
                )
                for neg in hard_negs:
                    qrels.append({
                        'query_id': query_id,
                        'doc_id': neg['doc_id'],
                        'relevance': neg['relevance'],
                    })

        return queries, qrels

    def _generate_c2t_match(self, code_by_problem: Dict, text_by_problem: Dict, 
                           abbrev_text_by_problem: Dict, start_id: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        c2t_match: Binary Classification Task - (Code, Text) → match / not match
        
        Goal: Train models to classify if a (code, text) pair correctly matches.
        Setup: For each problem, create positive and negative pairs.
        
        Positive pairs: Code from problem A → Text from problem A (label=1)
        Negative pairs: Code from problem A → Text from problem B (label=0)
        
        Strategies for hard negatives:
        - Similar problems (same difficulty, similar algorithms)
        - Random mismatches
        
        Evaluated with: Accuracy, F1, Precision, Recall, ROC-AUC
        """
        queries = []
        qrels = []
        query_id_counter = start_id
        
        problems = list(code_by_problem.keys())
        
        # Create positive and negative pairs
        for problem_id in problems:
            if problem_id not in text_by_problem:
                continue
                
            codes = code_by_problem[problem_id]
            if not codes:
                continue
            
            # Use both full and canonical text if available
            text_variants = []
            if problem_id in text_by_problem:
                text_variants.append(('full', text_by_problem[problem_id], None))
            # Canonical text uses same text_id as full text (canonical is stored as abbreviated_text field)
            if problem_id in abbrev_text_by_problem and problem_id in text_by_problem:
                canonical_text = abbrev_text_by_problem[problem_id]
                if canonical_text:
                    # Use actual text_id from text corpus (canonical text is in abbreviated_text field)
                    text_variants.append(('canonical', text_by_problem[problem_id], canonical_text))
            
            if not text_variants:
                continue
            
            # Select a representative code (prefer Python from Claude)
            anchor = self._select_anchor_code(codes, prefer_lang='python', prefer_model='claude')
            if not anchor:
                anchor = codes[0]
            
            # Create positive pairs (1 per text variant)
            for text_type, text_id, text_content in text_variants:
                query_id = f"q_c2t_match_{query_id_counter:04d}"
                query_id_counter += 1
                
                query = {
                    'query_id': query_id,
                    'code': anchor['code'],
                    'text_id': text_id,
                    'split': 'code2text',
                    'subtask': 'c2t_match',
                    'label': 1,  # Positive match
                    'text_type': text_type,
                    'anchor_language': anchor['language'],
                    'anchor_model': anchor['model'],
                    'meta': {
                        'source_problem_id': problem_id,
                        'anchor_code_id': anchor['code_id'],
                        'pair_type': 'positive',
                        'created_at': datetime.utcnow().isoformat() + 'Z',
                        'generator_version': 'v1.0'
                    }
                }
                
                # Include text directly for canonical
                if text_content:
                    query['text'] = text_content
                
                queries.append(query)
                
                # Create qrel (label as relevance)
                qrel = {
                    'query_id': query_id,
                    'doc_id': text_id,
                    'relevance': 1  # Positive label
                }
                qrels.append(qrel)
            
            # Create negative pairs (hard negatives from other problems)
            # Strategy 1: Random mismatch (1 negative per text variant)
            other_problems = [p for p in problems if p != problem_id and p in text_by_problem]
            if other_problems:
                for text_type, _, _ in text_variants:
                    # Pick a random different problem
                    negative_problem = random.choice(other_problems)
                    
                    # Get text from the negative problem
                    if text_type == 'full' and negative_problem in text_by_problem:
                        neg_text_id = text_by_problem[negative_problem]
                        neg_text_content = None
                    elif text_type == 'canonical' and negative_problem in abbrev_text_by_problem and negative_problem in text_by_problem:
                        neg_text_content = abbrev_text_by_problem[negative_problem]
                        if not neg_text_content:
                            continue
                        # Use actual text_id from text corpus (canonical text is in abbreviated_text field)
                        neg_text_id = text_by_problem[negative_problem]
                    else:
                        continue
                    
                    query_id = f"q_c2t_match_{query_id_counter:04d}"
                    query_id_counter += 1
                    
                    query = {
                        'query_id': query_id,
                        'code': anchor['code'],
                        'text_id': neg_text_id,
                        'split': 'code2text',
                        'subtask': 'c2t_match',
                        'label': 0,  # Negative match
                        'text_type': text_type,
                        'anchor_language': anchor['language'],
                        'anchor_model': anchor['model'],
                        'meta': {
                            'source_problem_id': problem_id,
                            'anchor_code_id': anchor['code_id'],
                            'negative_problem_id': negative_problem,
                            'pair_type': 'negative_random',
                            'created_at': datetime.utcnow().isoformat() + 'Z',
                            'generator_version': 'v1.0'
                        }
                    }
                    
                    # Include text directly for canonical
                    if neg_text_content:
                        query['text'] = neg_text_content
                    
                    queries.append(query)
                    
                    # Create qrel (label as relevance)
                    qrel = {
                        'query_id': query_id,
                        'doc_id': neg_text_id,
                        'relevance': 0  # Negative label
                    }
                    qrels.append(qrel)
        
        logger.info(f"    Created {len(queries)} code-text pairs:")
        positive_count = sum(1 for q in queries if q['label'] == 1)
        negative_count = sum(1 for q in queries if q['label'] == 0)
        logger.info(f"      - Positive pairs: {positive_count}")
        logger.info(f"      - Negative pairs: {negative_count}")
        
        return queries, qrels

    def _generate_c2t_cross_model(self, code_by_problem: Dict, text_by_problem: Dict, 
                                   abbrev_text_by_problem: Dict, start_id: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        c2t_cross_model: Cross-Model Variation (Code from different models → same text)
        
        Goal: Test if embeddings are model-invariant - code from different models
              for the same problem should all retrieve the same text description.
        
        Setup: For each problem with correct solutions from multiple models (e.g., Claude + Gemini),
               create 1 query per model. All should map to the same text description.
               
        Direction: Code (query) → Text (document in text corpus)
        
        Example for a problem with both Claude and Gemini solutions:
        - Query 1: Claude Python code → Full text (score=1)
        - Query 2: Gemini Python code → Same full text (score=1)
        
        This complements code2code cross-model task by testing text retrieval across models.
        """
        queries = []
        qrels = []
        query_id_counter = start_id

        for problem_id, codes in code_by_problem.items():
            # Check if problem has both full and canonical text
            has_full_text = problem_id in text_by_problem
            has_canonical_text = problem_id in abbrev_text_by_problem and abbrev_text_by_problem[problem_id]
            
            if not (has_full_text or has_canonical_text):
                continue

            # Group codes by model for this problem
            by_model = {}
            for code in codes:
                model = code['model']
                if model not in by_model:
                    by_model[model] = []
                by_model[model].append(code)

            # Only create cross-model queries if we have multiple models
            if len(by_model) < 2:
                continue

            # Get available models (sorted for consistency)
            available_models = sorted(by_model.keys())

            # === Full Text Variant ===
            if has_full_text:
                # Create one query per model (using full text)
                for model, model_codes in by_model.items():
                    # Select best code from this model (prefer Python)
                    anchor = self._select_best_code_for_model(model_codes, prefer_lang='python')
                    if not anchor:
                        continue

                    query_id = f"q_c2t_cross_model_full_{query_id_counter:04d}"
                    query_id_counter += 1

                    query = {
                        'query_id': query_id,
                        'query': anchor['code'],  # Code as query
                        'split': 'code2text',
                        'subtask': 'c2t_cross_model_full',
                        'anchor_language': anchor['language'],
                        'anchor_model': model,
                        'meta': {
                            'source_problem_id': problem_id,
                            'anchor_code_id': anchor['code_id'],
                            'task_type': 'cross_model_invariance',
                            'text_type': 'full_description',
                            'available_models': available_models,
                            'num_models': len(available_models),
                            'created_at': datetime.utcnow().isoformat() + 'Z',
                            'generator_version': 'v1.0'
                        }
                    }
                    queries.append(query)

                    # Create qrel - same text for all models
                    qrel = {
                        'query_id': query_id,
                        'doc_id': text_by_problem[problem_id],  # text_v202601_* ID
                        'relevance': 1
                    }
                    qrels.append(qrel)

            # === Canonical Text Variant ===
            if has_canonical_text:
                canonical_text = abbrev_text_by_problem[problem_id]
                
                # Create one query per model (using canonical text)
                for model, model_codes in by_model.items():
                    # Select best code from this model (prefer Python)
                    anchor = self._select_best_code_for_model(model_codes, prefer_lang='python')
                    if not anchor:
                        continue

                    query_id = f"q_c2t_cross_model_canonical_{query_id_counter:04d}"
                    query_id_counter += 1

                    query = {
                        'query_id': query_id,
                        'query': anchor['code'],  # Code as query
                        'split': 'code2text',
                        'subtask': 'c2t_cross_model_canonical',
                        'anchor_language': anchor['language'],
                        'anchor_model': model,
                        'meta': {
                            'source_problem_id': problem_id,
                            'anchor_code_id': anchor['code_id'],
                            'task_type': 'cross_model_invariance',
                            'text_type': 'canonical_description',
                            'available_models': available_models,
                            'num_models': len(available_models),
                            'created_at': datetime.utcnow().isoformat() + 'Z',
                            'generator_version': 'v1.0'
                        }
                    }
                    queries.append(query)

                    # Create qrel - use actual text_id from text corpus
                    # Canonical text is stored as abbreviated_text field in the text corpus entry
                    qrel = {
                        'query_id': query_id,
                        'doc_id': text_by_problem[problem_id],  # Actual text corpus ID (text_v202601_XXXXX)
                        'relevance': 1
                        # Note: Canonical text is available in text corpus entry's abbreviated_text field
                    }
                    qrels.append(qrel)

        # Log statistics
        query_by_subtask = defaultdict(int)
        for query in queries:
            query_by_subtask[query['subtask']] += 1

        logger.info("    📊 Cross-Model query distribution:")
        for subtask in sorted(query_by_subtask.keys()):
            logger.info(f"      - {subtask}: {query_by_subtask[subtask]} queries")

        return queries, qrels

    def _select_best_code_for_model(self, codes: List[Dict], prefer_lang: str = 'python') -> Optional[Dict]:
        """
        Select the best code from a single model's solutions.
        
        Strategy:
        1. Prefer specified language (e.g., Python)
        2. Fall back to first available code
        """
        # Try preferred language
        for code in codes:
            if code['language'] == prefer_lang:
                return code
        
        # Return first available
        return codes[0] if codes else None

    def _select_anchor_code(self, codes: List[Dict], prefer_lang: str = 'python', prefer_model: str = 'claude') -> Optional[Dict]:
        """
        Select an anchor code from a list of solutions.

        Strategy:
        1. Prefer specified language and model
        2. Fall back to specified language, any model
        3. Fall back to first available code
        """
        # Try preferred language and model
        for code in codes:
            if (code['language'] == prefer_lang and
                prefer_model in code['model'].lower()):
                return code

        # Try preferred language, any model
        for code in codes:
            if code['language'] == prefer_lang:
                return code

        # Return first available
        return codes[0] if codes else None

    def _generate_canonical_retro_queries(self, start_id: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Generate Canonical Retro queries from abbreviated/canonical texts (short, ~600 chars).

        For each problem with abbreviated text, creates:
        1. One t2c_canonical_retro_any query (language-agnostic)
        2. N t2c_canonical_retro_<lang> queries (language-specific), one per available language

        Args:
            start_id: Starting query ID counter

        Returns:
            Tuple of (queries, qrels)
        """
        all_queries = []
        all_qrels = []
        query_id_counter = start_id

        # Use prebuilt available languages
        available_languages = self.available_languages
        logger.info(f"    Available languages for retro: {available_languages}")

        # For each problem with abbreviated text
        for problem_id, abbrev_data in self.abbrev_texts.items():
            # Extract abbreviated text
            if isinstance(abbrev_data, dict):
                abbrev_text = abbrev_data.get('abbreviated', '')
            else:
                abbrev_text = abbrev_data

            if not abbrev_text:
                continue

            # Get all correct code solutions for this problem from prebuilt index
            all_correct_codes = self.correct_code_flat_by_problem.get(problem_id, [])

            if not all_correct_codes:
                continue

            # 1. Create t2c_canonical_retro_any query (language-agnostic)
            query_id_any = f"q_t2c_canonical_retro_any_{query_id_counter:04d}"
            query_id_counter += 1

            query_any = {
                'query_id': query_id_any,
                'query': abbrev_text,
                'split': 'text2code',
                'subtask': 't2c_canonical_retro_any',
                'query_type': 'language_agnostic',
                'sub_query_type': 'abbreviated',
                'language_constraint': 'none',
                'meta': {
                    'source_problem_id': problem_id,
                    'original_query_type': 'abbreviated',
                    'created_at': datetime.utcnow().isoformat() + 'Z',
                    'generator_version': 'v1.0'
                }
            }
            all_queries.append(query_any)

            # Create qrels for retro_any (all correct solutions as true positives)
            positive_ids = set()
            for code in all_correct_codes:
                qrel = {
                    'query_id': query_id_any,
                    'doc_id': code['code_id'],
                    'relevance': 2
                }
                all_qrels.append(qrel)
                positive_ids.add(code['code_id'])

            # Add hard negatives (failed code for the same problem)
            hard_negs = self._select_t2c_hard_negatives(
                problem_id=problem_id,
                positive_doc_ids=positive_ids,
                language_constraint=None,
                target_count=3,
            )
            for neg in hard_negs:
                all_qrels.append({
                    'query_id': query_id_any,
                    'doc_id': neg['doc_id'],
                    'relevance': neg['relevance'],
                })

            # 2. Create language-specific queries (retro_<lang>)
            for lang in available_languages:
                # Find correct solutions in this language
                lang_correct_codes = [
                    code for code in all_correct_codes
                    if code['language'] == lang
                ]

                if not lang_correct_codes:
                    # Skip if no correct solutions in this language
                    continue

                query_id_lang = f"q_t2c_canonical_retro_{lang}_{query_id_counter:04d}"
                query_id_counter += 1

                query_lang = {
                    'query_id': query_id_lang,
                    'query': abbrev_text,  # Same query text!
                    'split': 'text2code',
                    'subtask': f't2c_canonical_retro_{lang}',
                    'query_type': 'language_specific',
                    'sub_query_type': 'abbreviated',
                    'language_constraint': lang,
                    'meta': {
                        'source_problem_id': problem_id,
                        'original_query_type': 'abbreviated',
                        'created_at': datetime.utcnow().isoformat() + 'Z',
                        'generator_version': 'v1.0'
                    }
                }
                all_queries.append(query_lang)

                # Create qrels for retro_<lang> (only that language, true positives)
                lang_positive_ids = set()
                for code in lang_correct_codes:
                    qrel = {
                        'query_id': query_id_lang,
                        'doc_id': code['code_id'],
                        'relevance': 2
                    }
                    all_qrels.append(qrel)
                    lang_positive_ids.add(code['code_id'])

                # Add hard negatives (prefer same language failed code)
                hard_negs = self._select_t2c_hard_negatives(
                    problem_id=problem_id,
                    positive_doc_ids=lang_positive_ids,
                    language_constraint=lang,
                    target_count=3,
                )
                for neg in hard_negs:
                    all_qrels.append({
                        'query_id': query_id_lang,
                        'doc_id': neg['doc_id'],
                        'relevance': neg['relevance'],
                    })

        # Log statistics
        query_by_subtask = defaultdict(int)
        for query in all_queries:
            query_by_subtask[query['subtask']] += 1

        logger.info("    📊 Canonical Retro query distribution:")
        for subtask in sorted(query_by_subtask.keys()):
            logger.info(f"      - {subtask}: {query_by_subtask[subtask]} queries")

        return all_queries, all_qrels

    def _generate_full_retro_queries(self, start_id: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Generate Full Retro queries from full problem descriptions (long, ~1500 chars).

        For each problem with full description, creates:
        1. One t2c_full_retro_any query (language-agnostic)
        2. N t2c_full_retro_<lang> queries (language-specific), one per available language

        Args:
            start_id: Starting query ID counter

        Returns:
            Tuple of (queries, qrels)
        """
        all_queries = []
        all_qrels = []
        query_id_counter = start_id

        # Use prebuilt available languages
        available_languages = self.available_languages
        logger.info(f"    Available languages for full retro: {available_languages}")

        # For each problem in text corpus (full descriptions)
        for text_entry in self.text_corpus:
            problem_id = text_entry['meta']['source_problem_id']
            full_text = text_entry['text']

            if not full_text:
                continue

            # Get all correct code solutions for this problem from prebuilt index
            all_correct_codes = self.correct_code_flat_by_problem.get(problem_id, [])

            if not all_correct_codes:
                continue

            # 1. Create t2c_full_retro_any query (language-agnostic)
            query_id_any = f"q_t2c_full_retro_any_{query_id_counter:04d}"
            query_id_counter += 1

            query_any = {
                'query_id': query_id_any,
                'query': full_text,
                'split': 'text2code',
                'subtask': 't2c_full_retro_any',
                'query_type': 'language_agnostic',
                'sub_query_type': 'full_description',
                'language_constraint': 'none',
                'meta': {
                    'source_problem_id': problem_id,
                    'original_query_type': 'full_description',
                    'created_at': datetime.utcnow().isoformat() + 'Z',
                    'generator_version': 'v1.0'
                }
            }
            all_queries.append(query_any)

            # Create qrels for full_retro_any (all correct solutions as true positives)
            positive_ids = set()
            for code in all_correct_codes:
                qrel = {
                    'query_id': query_id_any,
                    'doc_id': code['code_id'],
                    'relevance': 2
                }
                all_qrels.append(qrel)
                positive_ids.add(code['code_id'])

            # Add hard negatives
            hard_negs = self._select_t2c_hard_negatives(
                problem_id=problem_id,
                positive_doc_ids=positive_ids,
                language_constraint=None,
                target_count=3,
            )
            for neg in hard_negs:
                all_qrels.append({
                    'query_id': query_id_any,
                    'doc_id': neg['doc_id'],
                    'relevance': neg['relevance'],
                })

            # 2. Create language-specific queries (full_retro_<lang>)
            for lang in available_languages:
                # Find correct solutions in this language
                lang_correct_codes = [
                    code for code in all_correct_codes
                    if code['language'] == lang
                ]

                if not lang_correct_codes:
                    # Skip if no correct solutions in this language
                    continue

                query_id_lang = f"q_t2c_full_retro_{lang}_{query_id_counter:04d}"
                query_id_counter += 1

                query_lang = {
                    'query_id': query_id_lang,
                    'query': full_text,  # Same query text!
                    'split': 'text2code',
                    'subtask': f't2c_full_retro_{lang}',
                    'query_type': 'language_specific',
                    'sub_query_type': 'full_description',
                    'language_constraint': lang,
                    'meta': {
                        'source_problem_id': problem_id,
                        'original_query_type': 'full_description',
                        'created_at': datetime.utcnow().isoformat() + 'Z',
                        'generator_version': 'v1.0'
                    }
                }
                all_queries.append(query_lang)

                # Create qrels for full_retro_<lang> (true positives)
                lang_positive_ids = set()
                for code in lang_correct_codes:
                    qrel = {
                        'query_id': query_id_lang,
                        'doc_id': code['code_id'],
                        'relevance': 2
                    }
                    all_qrels.append(qrel)
                    lang_positive_ids.add(code['code_id'])

                # Add hard negatives
                hard_negs = self._select_t2c_hard_negatives(
                    problem_id=problem_id,
                    positive_doc_ids=lang_positive_ids,
                    language_constraint=lang,
                    target_count=3,
                )
                for neg in hard_negs:
                    all_qrels.append({
                        'query_id': query_id_lang,
                        'doc_id': neg['doc_id'],
                        'relevance': neg['relevance'],
                    })

        # Log statistics
        query_by_subtask = defaultdict(int)
        for query in all_queries:
            query_by_subtask[query['subtask']] += 1

        logger.info("    📊 Full Retro query distribution:")
        for subtask in sorted(query_by_subtask.keys()):
            logger.info(f"      - {subtask}: {query_by_subtask[subtask]} queries")

        return all_queries, all_qrels

    def _parse_queries(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Parse queries from retro_queries.json and expand into language variants for Code Search.

        For each base query (e.g., algorithm_search), creates:
        1. One language-agnostic version (t2c_search_any)
        2. N language-specific versions (t2c_search_<lang>), one per available language

        The query text stays the same; only metadata changes.
        """
        query_groups = defaultdict(list)
        
        for key, queries in self.queries.items():
            # Parse key format: {problem_id}_{query_type}
            if '_' not in key:
                continue
                
            # Extract problem_id and query_type
            parts = key.split('_')
            if len(parts) < 2:
                continue
                
            # Extract problem_id
            problem_id = '_'.join(parts[:-2])
            query_type = '_'.join(parts[-2:])

            if query_type == 'cross_language': # we ignore this type
                continue
                
            # Get available languages for this problem from code corpus
            available_languages = self._get_available_languages_for_problem(problem_id)
            
            # Process each query
            for query_data in queries:
                if not isinstance(query_data, dict) or 'query' not in query_data:
                    continue

                query_text = query_data['query']

                # 1. Create language-agnostic version (t2c_search_any)
                best_match = query_data.get('best_match')
                best_match_with_prefix = ('lcb_' + best_match) if best_match else None
                
                query_groups['t2c_search_any'].append({
                    'query': query_text,
                    'problem_id': problem_id,
                        'original_type': query_type,
                    'language': None,
                    'best_match': best_match_with_prefix,  # Add 'lcb_' prefix if exists
                        'best_reason': query_data.get('best_reason')
                    })
        
                # 2. Create language-specific versions (t2c_search_<lang>) for each available language
                # IMPORTANT: Only create language-specific queries if correct solutions exist in that language
                for lang in available_languages:
                    # Validate that correct solutions exist in this language
                    if self._has_correct_solutions_in_language(problem_id, lang):
                        query_groups[f't2c_search_{lang}'].append({
                            'query': query_text,  # Same query text!
                            'problem_id': problem_id,
                            'original_type': query_type,
                            'language': lang,  # Different language constraint
                            'best_match': best_match_with_prefix,  # Add 'lcb_' prefix if exists
                            'best_reason': query_data.get('best_reason')
                        })
                    else:
                        logger.warning(f"Skipping search_{lang} for {problem_id}: no correct solutions in {lang}")

        # Log statistics about query generation
        total_queries = sum(len(queries) for queries in query_groups.values())
        logger.info(f"✅ Query expansion complete: {total_queries} total queries across {len(query_groups)} subtasks")
        for subtask, queries in sorted(query_groups.items()):
            logger.info(f"  - {subtask}: {len(queries)} queries")
        
        return dict(query_groups)

    def _get_available_languages_for_problem(self, problem_id: str) -> List[str]:
        """Get list of available languages for a problem from prebuilt index."""
        languages = set()
        for code in self.all_code_by_problem.get(problem_id, []):
            lang = code.get('language')
            if lang:
                languages.add(lang)
        return sorted(list(languages))

    def _has_correct_solutions_in_language(self, problem_id: str, language: str) -> bool:
        """
        Validate that the problem has at least one correct solution in the specified language.
        Uses prebuilt index for efficient lookup.

        Args:
            problem_id: Problem identifier (e.g., 'lcb_3587')
            language: Target language (e.g., 'python', 'cpp')

        Returns:
            True if at least one correct solution exists in that language, False otherwise
        """
        # Check prebuilt correct code index
        for code in self.correct_code_flat_by_problem.get(problem_id, []):
            if code.get('language') == language:
                return True
        return False

    def _get_language_constraint(self, subtask: str, original_type: str, language: Optional[str] = None) -> str:
        """
        Get language constraint for the subtask.

        Returns:
            'none' for language-agnostic subtasks (t2c_canonical_retro_any, t2c_search_any)
            Language code for language-specific subtasks (e.g., 'python', 'cpp')
        """
        if subtask in ('t2c_canonical_retro_any', 't2c_search_any'):
            return 'none'  # Language agnostic
        elif subtask.startswith('t2c_canonical_retro_') or subtask.startswith('t2c_search_'):
            # Extract language from subtask name (e.g., 't2c_canonical_retro_python' -> 'python')
            return subtask.split('_')[-1]  # Get last part after final '_'
        return 'none'

    def _map_query_types(self, original_type: str, language: Optional[str] = None) -> Tuple[str, str]:
        """
        Map original query types to query_type and sub_query_type.

        Returns:
            Tuple of (query_type, sub_query_type)
            - query_type: 'language_agnostic' or 'language_specific'
            - sub_query_type: The original query type (e.g., 'algorithm_search', 'description_search')
        """
        # Determine query_type based on language constraint
        if language:
            query_type = 'language_specific'
        else:
            query_type = 'language_agnostic'
        
        # Map to sub_query_type (keep the original query type)
        sub_query_mapping = {
            'algorithm_search': 'algorithm_search',
            'description_search': 'description_search',
            'language_agnostic': 'language_agnostic'
        }
        sub_query_type = sub_query_mapping.get(original_type, 'algorithm_search')
        
        return query_type, sub_query_type
    
    def _create_text2code_entries(self, query: Dict[str, Any], subtask: str, query_id: int) -> Tuple[
        Optional[Dict[str, Any]], List[Dict[str, Any]]]:
        """Create separate query and qrel entries for text2code dataset."""
        problem_id = query['problem_id']
        original_type = query.get('original_type')

        # Get all code entries for this problem from prebuilt index
        all_matching_codes = self.all_code_by_problem.get(problem_id, [])

        if not all_matching_codes:
            return None, []

        # Apply language constraint to filter positives based on subtask policy
        language = query.get('language')
        language_constraint = self._get_language_constraint(subtask, original_type, language)
        if language_constraint == 'none':
            # t2c_canonical_retro_any/t2c_search_any (language-agnostic): Score 1 for every correct solution in any language
            matching_codes = all_matching_codes
        else:
            # t2c_canonical_retro_<lang>/t2c_search_<lang> (language-specific): Score 1 only for correct solutions in target language
            # Don't list other languages at all in qrels (they're implicitly non-relevant)
            matching_codes = [code for code in all_matching_codes
                              if code['language'] == language_constraint]
        
        if not matching_codes:
            return None, []
        
        # Determine query type and sub_query_type
        query_type, sub_query_type = self._map_query_types(original_type, language)

        # Create query entry with retro_any/retro_<lang> format
        query_id_formatted = f"q_{subtask}_{query_id:04d}"

        # Track overlap metadata: how many models have correct solutions for this problem
        # This helps analyze performance on single-model vs multi-model problems
        models_in_relevant = list(set(code['model'] for code in matching_codes))
        num_relevant = len(matching_codes)
        is_multi_model = len(models_in_relevant) > 1

        query_entry = {
            'query_id': query_id_formatted,
            'query': query['query'],
            'split': 'text2code',  # Task name as split
            'subtask': subtask,  # Subtask name as separate column
            'query_type': query_type,
            'sub_query_type': sub_query_type,
            'language_constraint': language_constraint,
            'meta': {
                'source_problem_id': problem_id,
                'original_query_type': original_type,
                'created_at': datetime.utcnow().isoformat() + 'Z',
                'generator_version': 'v1.0',
                # Overlap tracking: multiple models may have correct solutions for same problem
                # All correct solutions are labeled as relevant (score=1 in qrels)
                # This metadata allows post-hoc filtering/analysis by overlap type
                'num_relevant_docs': num_relevant,  # Total correct solutions (across all models)
                'models_in_relevant_docs': models_in_relevant,  # List of models with correct solutions
                'is_multi_model': is_multi_model  # True if multiple models solved this problem
            }
        }

        # Create qrel entries with graded relevance
        # True positives (correct solutions) get relevance=2
        qrel_entries = []
        positive_ids = set()
        for code in matching_codes:
            qrel_entry = {
                'query_id': query_id_formatted,
                'doc_id': code['code_id'],
                'relevance': 2
            }
            qrel_entries.append(qrel_entry)
            positive_ids.add(code['code_id'])

        # Add hard negatives (failed code for the same problem)
        lang_constraint = language_constraint if language_constraint != 'none' else None
        hard_negs = self._select_t2c_hard_negatives(
            problem_id=problem_id,
            positive_doc_ids=positive_ids,
            language_constraint=lang_constraint,
            target_count=3,
        )
        for neg in hard_negs:
            qrel_entries.append({
                'query_id': query_id_formatted,
                'doc_id': neg['doc_id'],
                'relevance': neg['relevance'],
            })

        return query_entry, qrel_entries

    def _calculate_relevance_score(self, code: Dict[str, Any], language_constraint: str, subtask: str) -> int:
        """
        Calculate relevance score following binary relevance rules.
        
        Overlap handling policy:
        - ALL correct solutions from ANY model are labeled as relevant (relevance=1)
        - If both Claude and Gemini solve a problem, BOTH get relevance=1
        - This follows standard IR practice and allows models to retrieve any correct solution
        - Use query metadata (is_multi_model) for post-hoc filtering if needed
        """
        # Binary relevance: relevance = 1 for relevant, omit non-relevant
        return 1

    def _save_dataset(self, dataset_entries: List[Dict[str, Any]], dataset_name: str):
        """Save dataset to JSONL file."""
        output_path = f"{self.output_dir}/{dataset_name}.jsonl"
        logger.info(f"Saving {len(dataset_entries)} entries to {output_path}")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for entry in dataset_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        # Save metadata
        metadata = {
            'dataset_name': dataset_name,
            'num_entries': len(dataset_entries),
            'created_at': datetime.utcnow().isoformat() + 'Z',
            'config': {
                'max_queries_per_type': self.max_queries_per_type,
                'random_seed': self.random_seed
            }
        }
        
        metadata_path = f"{self.output_dir}/{dataset_name}_metadata.json"
        save_json(metadata, metadata_path)
        
        logger.info(f"✅ Saved {dataset_name} dataset with {len(dataset_entries)} entries")

    def _save_queries(self, query_entries: List[Dict[str, Any]], dataset_name: str):
        """Save queries to JSONL file."""
        output_path = f"{self.output_dir}/{dataset_name}_queries.jsonl"
        logger.info(f"Saving {len(query_entries)} queries to {output_path}")

        with open(output_path, 'w', encoding='utf-8') as f:
            for entry in query_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')

        logger.info(f"✅ Saved {dataset_name}_queries with {len(query_entries)} queries")

    def _save_qrels(self, qrel_entries: List[Dict[str, Any]], dataset_name: str):
        """Save qrels to JSONL file."""
        output_path = f"{self.output_dir}/{dataset_name}_qrels.jsonl"
        logger.info(f"Saving {len(qrel_entries)} qrels to {output_path}")

        with open(output_path, 'w', encoding='utf-8') as f:
            for entry in qrel_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')

        logger.info(f"✅ Saved {dataset_name}_qrels with {len(qrel_entries)} qrels")

    def _save_huggingface_format(self, data_dict: Dict[str, List[Dict[str, Any]]], dataset_name: str):
        """Save data in HuggingFace-compatible format with split names as keys."""
        output_path = f"{self.output_dir}/{dataset_name}.json"
        logger.info(f"Saving HuggingFace format to {output_path}")

        # Create the HuggingFace format: {split_name: [entries]}
        hf_data = {}
        for split_name, entries in data_dict.items():
            hf_data[split_name] = entries
            logger.info(f"  Split '{split_name}': {len(entries)} entries")

        # Save as JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(hf_data, f, ensure_ascii=False, indent=2)

        logger.info(f"✅ Saved HuggingFace format {dataset_name} with {len(data_dict)} splits")

    def _generate_unified_huggingface_dataset(self):
        """Generate a unified HuggingFace dataset with all tasks."""
        logger.info("🌐 Generating unified HuggingFace dataset...")

        # Load all generated files
        unified_data = {}

        # Load text2code data
        text2code_queries_path = f"{self.output_dir}/text2code_queries.jsonl"
        text2code_qrels_path = f"{self.output_dir}/text2code_qrels.jsonl"

        if os.path.exists(text2code_queries_path):
            text2code_queries = load_jsonl(text2code_queries_path)
            unified_data["text2code"] = text2code_queries
            logger.info(f"  Loaded {len(text2code_queries)} text2code queries")

        if os.path.exists(text2code_qrels_path):
            text2code_qrels = load_jsonl(text2code_qrels_path)
            unified_data["text2code_qrels"] = text2code_qrels
            logger.info(f"  Loaded {len(text2code_qrels)} text2code qrels")

        # Load code2code data
        code2code_queries_path = f"{self.output_dir}/code2code_queries.jsonl"
        code2code_qrels_path = f"{self.output_dir}/code2code_qrels.jsonl"

        if os.path.exists(code2code_queries_path):
            code2code_queries = load_jsonl(code2code_queries_path)
            unified_data["code2code"] = code2code_queries
            logger.info(f"  Loaded {len(code2code_queries)} code2code queries")

        if os.path.exists(code2code_qrels_path):
            code2code_qrels = load_jsonl(code2code_qrels_path)
            unified_data["code2code_qrels"] = code2code_qrels
            logger.info(f"  Loaded {len(code2code_qrels)} code2code qrels")

        # Load code2text data
        code2text_queries_path = f"{self.output_dir}/code2text_queries.jsonl"
        code2text_qrels_path = f"{self.output_dir}/code2text_qrels.jsonl"

        if os.path.exists(code2text_queries_path):
            code2text_queries = load_jsonl(code2text_queries_path)
            unified_data["code2text"] = code2text_queries
            logger.info(f"  Loaded {len(code2text_queries)} code2text queries")

        if os.path.exists(code2text_qrels_path):
            code2text_qrels = load_jsonl(code2text_qrels_path)
            unified_data["code2text_qrels"] = code2text_qrels
            logger.info(f"  Loaded {len(code2text_qrels)} code2text qrels")

        # Save unified dataset
        if unified_data:
            self._save_huggingface_format(unified_data, "code_embedding_benchmark")
            logger.info("✅ Generated unified HuggingFace dataset")
        else:
            logger.warning("⚠️ No data found to create unified dataset")
