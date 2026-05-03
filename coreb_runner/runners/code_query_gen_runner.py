"""
Code Query Generation Runner

This runner focuses specifically on generating diverse queries for code retrieval evaluation:
1. Generate diverse queries using LLM with structured prompts
2. Support multiple query types (title, description, algorithm, cross-language, etc.)
3. Generate ranked candidate matches for each query
4. Save queries to database for later retrieval evaluation
"""

import os
import random
import numpy as np
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from collections import defaultdict
from sentence_transformers import SentenceTransformer

from easyllm_kit.utils.io_utils import initialize_database, write_to_database
from easyllm_kit.utils import get_logger, read_json, extract_json_from_text
from easyllm_kit.models import LLM
from easyllm_kit.configs.llm_base_config import GenerationArguments

from coreb_runner.runners.base_runner import Runner
from coreb_runner.prompts.query_gen_prompt import (
    QueryGenPrompt, AlgorithmFocusedQueryGenPrompt, CrossLanguageQueryGenPrompt
)

logger = get_logger('code_query_gen_runner', 'code_query_gen_runner.log')


@dataclass
class RetrievalQuery:
    """Represents a retrieval query with metadata."""
    query_id: str
    query_text: str
    problem_id: str
    query_type: str
    target_solutions: List[str]
    metadata: Dict[str, Any]


@Runner.register("code_query_gen")
class CodeQueryGenRunner(Runner):
    """
    Runner for code query generation.
    
    This runner generates diverse queries for code retrieval evaluation.
    
    For each query type (title_search, description_search, algorithm_search),
    it generates BOTH:
    - Base language-agnostic version (retro_any - Retrieval, any language)
    - Language-specific variants for each available language (retro_<lang> - Retrieval, specific language)
    
    Example:
    - algorithm_search → retro_any (language-agnostic)
    - algorithm_search_python → retro_python (Python-specific)
    - algorithm_search_cpp → retro_cpp (C++-specific)
    - algorithm_search_java → retro_java (Java-specific)
    - algorithm_search_go → retro_go (Go-specific)
    - algorithm_search_ruby → retro_ruby (Ruby-specific)
    """
    
    # Supported languages for language-specific queries
    SUPPORTED_LANGUAGES = ['python', 'cpp', 'java', 'go', 'ruby']
    
    # Query type to subtask mapping (base/language-agnostic versions)
    QUERY_TYPE_SUBTASK_MAP = {
        'title_search': 'retro_any',
        'algorithm_search': 'retro_any',
        'description_search': 'retro_any'
    }

    def __init__(self, config: Dict[str, Any]):
        super().__init__()

        # Get configurations with defaults
        self.data_config = config.get('data', {})
        self.query_config = config.get('query_generation', {})
        self.advanced_config = config.get('advanced', {})
        
        # Model selection configuration
        self.source_models = self.data_config.get('source_models', ['bedrock-claude'])
        self.problem_selection_strategy = self.data_config.get('problem_selection_strategy', 'all')
        
        logger.info(f"Source models for query generation: {self.source_models}")
        logger.info(f"  Note: For overlap problems, Claude Sonnet is used as reference")
        logger.info(f"  Overlap is tracked in metadata for post-hoc filtering/weighting")
        logger.info(f"Problem selection strategy: {self.problem_selection_strategy}")

        # Initialize LLM for query generation
        self.llm_model_config = self.query_config.get("model", {})
        self.llm_generation_config = GenerationArguments(**self.query_config.get("generation", {}))
        self.llm = self.setup_model(self.llm_model_config, self.llm_generation_config)
        self.llm_name = self.llm.model_config.model_full_name

        # Check if semantic similarity is enabled
        similarity_model = self.query_config.get('similarity_model', None)
        logger.info(f"🔄 Initializing sentence transformer: {similarity_model}")

        # Initialize with CPU device to avoid GPU memory issues
        self.sentence_transformer = SentenceTransformer(similarity_model, device='cpu')

        # Initialize database
        self.output_db_name = self.data_config.get('output_db_name')
        self.db = initialize_database(self.output_db_name)

        # Load data with error handling
        self.question_content_data = read_json(self.data_config.get('data_dir'))
        
        # Load clean merged evaluation data
        # Expected format: {problem_id: {model_language: True/False}}
        eval_data_path = self.data_config.get('eval_data_dir')
        logger.info(f"Loading evaluation data from: {eval_data_path}")
        self.eval_data = read_json(eval_data_path)
        logger.info(f"✅ Loaded {len(self.eval_data)} problems")

        # Preprocess and clean evaluation data for efficient lookup
        logger.info("🧹 Preprocessing evaluation data for efficient lookup...")
        self.clean_eval_data = self._preprocess_evaluation_data()
        logger.info(f"✅ Preprocessed {len(self.clean_eval_data)} problems with passed solutions")
        
        # Initialize context cache to avoid recomputing similarity for same problem across variants
        # Each problem has ~18 variants that all use the same context (target + similar problems)
        # Caching saves ~95% of similarity computations and ~92% of input tokens
        self.context_cache = {}
        logger.info("💾 Initialized context cache for smart context reuse")

        # Initialize data structures
        self.retrieval_queries = []

        # Statistics
        self.stats = {
            'total_queries': 0,
            'languages': set(),
            'models': set(),
            'query_types': set(),
            'problems': set(),
            'errors': 0,
            'fallback_used': 0
        }

        logger.info(f"CodeQueryGenRunner initialized with {len(self.question_content_data)} problems")
        logger.info(f"LLM Model: {self.llm_name}")

    def _preprocess_evaluation_data(self) -> Dict[str, Dict[str, Any]]:
        """
        Preprocess evaluation data from clean merged format.
        
        Expected format: {problem_id: {model_language: True/False}}
        Example: {"lcb_3487": {"bedrock-claude-4-sonnet_python": True, ...}}
        """
        clean_data = {}

        logger.info(f"Preprocessing {len(self.eval_data)} problems from evaluation data")
        logger.info(f"Using source models: {self.source_models}")

        for problem_id, model_lang_results in self.eval_data.items():
            if not isinstance(model_lang_results, dict):
                continue

            problem_clean_data = {
                'available_languages': [],
                'language_solutions': {},
                'model_language_solutions': {},
                'total_passed_solutions': 0,
                'has_passed_solutions': False
            }

            # Process each model-language result
            for model_lang_key, passed in model_lang_results.items():
                if not passed:  # Skip failed solutions
                    continue

                # Extract model and language from key
                # Format: model_name_language (e.g., 'bedrock-claude-4-sonnet_python')
                parts = model_lang_key.split('_')
                if len(parts) < 2:
                    continue
                
                language = parts[-1]  # Last part is the language
                model_name = '_'.join(parts[:-1])  # Everything except last part is the model
                
                # Check if this model is in our source_models list
                is_source_model = any(model_name.startswith(source) for source in self.source_models)
                if not is_source_model:
                    continue
                
                # Track by language (aggregated across models)
                if language not in problem_clean_data['available_languages']:
                    problem_clean_data['available_languages'].append(language)
                
                # Track by language (aggregated)
                if language not in problem_clean_data['language_solutions']:
                    problem_clean_data['language_solutions'][language] = {
                        'passed_solutions': [0],  # Dummy solution index (not used for query gen)
                        'total_solutions': 0,
                        'models': []
                    }
                
                problem_clean_data['language_solutions'][language]['total_solutions'] += 1
                if model_name not in problem_clean_data['language_solutions'][language]['models']:
                    problem_clean_data['language_solutions'][language]['models'].append(model_name)
                
                # Track by model and language (for balanced sampling)
                problem_clean_data['model_language_solutions'][model_lang_key] = {
                    'model': model_name,
                    'language': language,
                    'passed_solutions': [0],  # Dummy solution index
                    'total_solutions': 1,
                    'passed': True
                }
                
                problem_clean_data['total_passed_solutions'] += 1
                problem_clean_data['has_passed_solutions'] = True

            # Only include problems that have passed solutions
            if problem_clean_data['has_passed_solutions']:
                # Pre-compute formatted text representation for this problem
                if problem_id in self.question_content_data:
                    problem_data = self.question_content_data[problem_id]
                    problem_id_no_prefix = problem_id.replace('lcb_', '')
                    title = problem_data.get('ANNOTATED_TITLE')
                    description = problem_data.get('ANNOTATED_QUESTION_CONTENT')
                    difficulty = problem_data.get('lcb_difficulty')
                    
                    # Format problem for prompt template
                    problem_text = f"""
**Problem:**
ID: {problem_id_no_prefix}
Title: {title}
Description: {description}
Difficulty: {difficulty}
Available Languages: {', '.join(problem_clean_data['available_languages'])}
"""
                    problem_clean_data['formatted_text'] = problem_text.strip()
                
                clean_data[problem_id] = problem_clean_data

        # Log statistics by model
        model_stats = defaultdict(lambda: {'problems': 0, 'solutions': 0})
        for problem_id, problem_data in clean_data.items():
            for model_lang_key, sol_data in problem_data['model_language_solutions'].items():
                model = sol_data['model']
                model_stats[model]['problems'] += 1
                model_stats[model]['solutions'] += sol_data['total_solutions']
        
        # Analyze overlap between models
        overlap_stats = self._analyze_model_overlap(clean_data)
        
        logger.info(f"Preprocessing completed: {len(clean_data)} problems with passed solutions")
        logger.info(f"Model statistics:")
        for model, stats in model_stats.items():
            logger.info(f"  - {model}: {stats['problems']} problems, {stats['solutions']} solutions")
        
        logger.info(f"Overlap statistics:")
        logger.info(f"  - Problem-language pairs in overlap: {overlap_stats['overlap_pairs']}")
        logger.info(f"  - Overlap percentage: {overlap_stats['overlap_percentage']:.1f}%")
        
        return clean_data
    
    def _analyze_model_overlap(self, clean_data: Dict) -> Dict[str, Any]:
        """Analyze overlap of solved problem-language pairs between models."""
        if len(self.source_models) <= 1:
            return {'overlap_pairs': 0, 'overlap_percentage': 100.0}
        
        # Track which problem-language pairs each model solved
        model_pairs = {model: set() for model in self.source_models}
        
        for problem_id, problem_data in clean_data.items():
            for model_lang_key, sol_data in problem_data['model_language_solutions'].items():
                model = sol_data['model']
                language = sol_data['language']
                # Find which source model this belongs to
                for source_model in self.source_models:
                    if model.startswith(source_model):
                        model_pairs[source_model].add((problem_id, language))
                        break
        
        # Calculate intersection (overlap)
        if len(model_pairs) > 0:
            overlap = set.intersection(*model_pairs.values()) if len(model_pairs) > 1 else list(model_pairs.values())[0]
            total_unique = set.union(*model_pairs.values())
            overlap_pct = (len(overlap) / len(total_unique) * 100) if total_unique else 0
        else:
            overlap = set()
            overlap_pct = 0
        
        return {
            'overlap_pairs': len(overlap),
            'overlap_percentage': overlap_pct,
            'overlap_set': overlap,
            'model_pairs': model_pairs
        }

    def _get_available_languages(self, problem_id: str) -> List[str]:
        """
        Get available languages for a problem based on problem selection strategy.
        
        Filters languages based on whether we want:
        - all: All languages solved by any model
        - intersection: Only languages solved by ALL models
        - union_with_preference: All languages, but mark overlap for weighting
        """
        if problem_id not in self.clean_eval_data:
            return []

        problem_data = self.clean_eval_data[problem_id]
        all_languages = problem_data['available_languages'].copy()
        
        if self.problem_selection_strategy == 'intersection' and len(self.source_models) > 1:
            # Only include languages solved by ALL source models
            language_models = defaultdict(set)
            for model_lang_key, sol_data in problem_data['model_language_solutions'].items():
                language = sol_data['language']
                model = sol_data['model']
                # Find which source model this belongs to
                for source_model in self.source_models:
                    if model.startswith(source_model):
                        language_models[language].add(source_model)
                        break
            
            # Filter to languages solved by all models
            overlap_languages = [lang for lang, models in language_models.items() 
                                if len(models) == len(self.source_models)]
            return overlap_languages
        
        return all_languages
    
    def _is_overlap_problem_language(self, problem_id: str, language: str) -> bool:
        """Check if a problem-language pair is solved by all source models."""
        if problem_id not in self.clean_eval_data or len(self.source_models) <= 1:
            return True  # Treat as overlap if only one model
        
        problem_data = self.clean_eval_data[problem_id]
        
        # Check which models solved this problem-language pair
        models_for_lang = set()
        for model_lang_key, sol_data in problem_data['model_language_solutions'].items():
            if sol_data['language'] == language:
                model = sol_data['model']
                # Find which source model this belongs to
                for source_model in self.source_models:
                    if model.startswith(source_model):
                        models_for_lang.add(source_model)
                        break
        
        # It's overlap if all source models solved it
        return len(models_for_lang) == len(self.source_models)
    
    def _get_balanced_model_language_combinations(self, problem_id: str) -> List[Tuple[str, str, Dict]]:
        """
        Get balanced model-language combinations for query generation.
        
        For overlap problems (solved by multiple models), prefers Claude Sonnet as reference.
        
        Returns a list of (model, language, metadata) tuples that ensures balanced representation
        across models and languages based on the configured sampling strategy.
        
        Args:
            problem_id: Problem identifier
            
        Returns:
            List of (model_name, language, metadata) tuples where metadata includes:
                - is_overlap: bool (whether multiple models solved this)
                - reference_model: str (which model's solution is used as reference)
                - available_models: List[str] (all models that solved this)
        """
        if problem_id not in self.clean_eval_data:
            return []
        
        problem_data = self.clean_eval_data[problem_id]
        model_lang_solutions = problem_data.get('model_language_solutions', {})
        
        if not model_lang_solutions:
            return []

        # Group by language to identify overlaps
        lang_models = defaultdict(list)
        for model_lang_key, sol_data in model_lang_solutions.items():
            language = sol_data['language']
            lang_models[language].append((sol_data['model'], model_lang_key, sol_data))
        
        combinations = []
        
        for language, models_for_lang in lang_models.items():
            is_overlap = len(models_for_lang) > 1
            available_models = [m[0] for m in models_for_lang]
            
            if is_overlap:
                # For overlap: prefer Claude Sonnet as reference
                claude_model = None
                for model, key, sol_data in models_for_lang:
                    if model.startswith('bedrock-claude'):
                        claude_model = (model, key, sol_data)
                        break
                
                if claude_model:
                    reference_model = claude_model[0]
                    metadata = {
                        'is_overlap': True,
                        'reference_model': reference_model,
                        'available_models': available_models,
                        'preference_reason': 'claude_sonnet_reference'
                    }
                    combinations.append((reference_model, language, metadata))
                else:
                    # No Claude, use first available
                    reference_model = models_for_lang[0][0]
                    metadata = {
                        'is_overlap': True,
                        'reference_model': reference_model,
                        'available_models': available_models,
                        'preference_reason': 'first_available'
                    }
                    combinations.append((reference_model, language, metadata))
            else:
                # Non-overlap: use the only available model
                reference_model = models_for_lang[0][0]
                metadata = {
                    'is_overlap': False,
                    'reference_model': reference_model,
                    'available_models': available_models,
                    'preference_reason': 'unique_solution'
                }
                combinations.append((reference_model, language, metadata))
        
        return combinations


    def _get_solution_stats(self, problem_id: str) -> Dict[str, Any]:
        """Get statistics about solutions for a problem using preprocessed data."""
        if problem_id in self.clean_eval_data:
            problem_data = self.clean_eval_data[problem_id]
            return {
                'total_solutions': problem_data['total_passed_solutions'],
                'passed_solutions': problem_data['total_passed_solutions'],
                'languages': {lang: {'total': lang_data['total_solutions'], 'passed': lang_data['total_solutions']}
                              for lang, lang_data in problem_data['language_solutions'].items()},
                'has_passed_solutions': problem_data['has_passed_solutions']
            }

        return {
            'total_solutions': 0,
            'passed_solutions': 0,
            'languages': {},
            'has_passed_solutions': False
        }


    def setup_model(self, model_config, generation_config):
        # Build the LLM model
        llm_config = {'model_config': model_config,
                      'generation_config': generation_config}

        llm = LLM.build_from_config(llm_config)
        return llm

    def run(self):
        """Run the query generation process."""
        logger.info("🚀 Starting Code Query Generation")
        
        # Generate queries using LLM
        self._generate_queries_with_llm()
        
        # Print summary
        self._print_summary()
        
        logger.info("✅ Query generation completed successfully")

    def _generate_queries_with_llm(self):
        """Generate diverse queries using LLM."""
        logger.info("🤖 Generating queries with LLM...")

        max_queries_per_problem = self.query_config.get('max_queries_per_problem', 10)
        query_types = self.query_config.get('query_types')

        # Get advanced configuration
        max_retries = self.advanced_config.get('max_retries', 3)
        batch_size = self.advanced_config.get('batch_size', 10)

        logger.info(f"Configuration: max_queries_per_problem={max_queries_per_problem}, "
                    f"query_types={len(query_types)}, max_retries={max_retries}")

        # Process problems in batches
        problems = list(self.question_content_data.items())
        total_problems = len(problems)
        processed_count = 0
        problems_with_queries = 0

        logger.info(f"Starting to process {total_problems} problems in batches of {batch_size}")

        for i in range(0, total_problems, batch_size):
            batch = problems[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (total_problems + batch_size - 1) // batch_size

            logger.info(f"🔄 Processing batch {batch_num}/{total_batches} ({len(batch)} problems)")
            logger.info(f"   Batch problems: {[pid for pid, _ in batch]}")

            for problem_id, problem_data in batch:
                logger.debug(f"🔍 Processing problem {problem_id}")

                # Get available languages from evaluation data
                available_languages = self._get_available_languages(problem_id)

                if not available_languages:
                    logger.warning(
                        f"❌ No available languages with passed solutions for problem {problem_id}, skipping")
                    continue

                logger.info(
                    f"✅ Problem {problem_id}: Available languages with passed solutions: {available_languages}")

                for query_type in query_types:
                    # Get all variants for this query type
                    query_variants = self._get_query_variants(query_type, available_languages)
                    
                    # Check if any variants need generation (skip if all exist)
                    variants_to_generate = {}
                    for variant_type, variant_info in query_variants.items():
                        key = f'{problem_id}_{variant_type}'
                        if self.db.get(key) is None:
                            variants_to_generate[variant_type] = variant_info
                        else:
                            logger.info(f"⏭️  Skipping {key} (already exists)")
                    
                    # If all variants exist, skip this query type
                    if not variants_to_generate:
                        logger.info(f"✓ All {query_type} variants already exist for {problem_id}")
                        continue

                    # Generate ALL variants for this query type in ONE API call
                    logger.info(f"🔄 Generating {len(variants_to_generate)} {query_type} variants for {problem_id} in batch")

                    batched_queries = self._generate_queries_for_type_batched(
                        problem_id, query_type, available_languages, variants_to_generate
                    )

                    # Save each variant separately
                    for variant_type, queries in batched_queries.items():
                        if queries:
                            key = f'{problem_id}_{variant_type}'
                            variant_info = query_variants[variant_type]
                            
                            # Preprocess queries to add language constraints and subtask info
                            preprocessed_queries = self._preprocess_queries(
                                queries, problem_id, variant_type, available_languages, variant_info
                            )

                            # Save to database
                            write_to_database(self.output_db_name, key, preprocessed_queries)

                            self.retrieval_queries.extend(preprocessed_queries)
                            logger.debug(f"   ✓ Saved {len(preprocessed_queries)} {variant_type} queries")
                    problems_with_queries += 1

                processed_count += 1

            logger.info(f"✅ Completed batch {batch_num}/{total_batches}")
            logger.info(f"   Progress: {processed_count}/{total_problems} problems processed")
            logger.info(f"   Generated queries so far: {len(self.retrieval_queries)}")
            logger.info(f"   Problems with queries: {problems_with_queries}")

        logger.info(f"🏁 Finished processing all {total_batches} batches")

        # Limit total queries
        if len(self.retrieval_queries) > max_queries_per_problem * len(self.question_content_data):
            self.retrieval_queries = random.sample(
                self.retrieval_queries,
                max_queries_per_problem * len(self.question_content_data)
            )

        self.stats['total_queries'] = len(self.retrieval_queries)

        logger.info(f"✅ Generated {len(self.retrieval_queries)} queries")

    def _generate_queries_for_type_batched(self, problem_id: str, query_type: str,
                                           available_languages: List[str], 
                                           variants_to_generate: Dict[str, Dict]) -> Dict[str, List]:
        """
        Generate ALL language variants for a query type in ONE API call.
        
        Args:
            problem_id: Problem identifier
            query_type: Base query type (e.g., 'title_search')
            available_languages: Available languages for this problem
            variants_to_generate: Dict of {variant_type: variant_info} to generate
            
        Returns:
            Dict mapping variant_type to list of queries
        """
        # Use cached context
        if problem_id not in self.context_cache:
            target_problem, similar_problem_list = self._format_smart_context(problem_id)
            self.context_cache[problem_id] = (target_problem, similar_problem_list)
            logger.debug(f"💾 Cached context for {problem_id}")
        else:
            target_problem, similar_problem_list = self.context_cache[problem_id]
            logger.debug(f"⚡ Using cached context for {problem_id}")
        
        # Build prompt requesting all variants at once
        context = {
            'target_problem': target_problem,
            'similar_problem_list': similar_problem_list,
            'query_type': query_type,
            'available_languages': available_languages,
            'variants_to_generate': variants_to_generate,
            'max_queries_per_problem': self.query_config.get('max_queries_per_problem', 2)
        }
        
        result = {}
        
        try:
            prompt = self._create_batched_query_prompt(context)
            response = self.llm.generate(prompt)
            logger.debug(f"      Batched LLM response received for {problem_id} {query_type}: {len(str(response))} chars")
            
            # Parse batched response
            parsed_response = extract_json_from_text(response)
            
            # Expected format: { 'retro_any': [...], 'retro_python': [...], ... }
            # or { 'queries': { 'retro_any': [...], ... } }
            if 'queries' in parsed_response and isinstance(parsed_response['queries'], dict):
                queries_dict = parsed_response['queries']
            else:
                # Direct format
                queries_dict = parsed_response
            
            # Map subtasks back to variant types
            for variant_type in variants_to_generate.keys():
                variant_info = variants_to_generate[variant_type]
                subtask = variant_info['subtask']
                
                # Get queries for this subtask
                variant_queries = queries_dict.get(subtask, [])
                if variant_queries:
                    result[variant_type] = variant_queries
                    logger.debug(f"      ✓ {variant_type}: {len(variant_queries)} queries")
                else:
                    logger.warning(f"      ⚠️  No queries generated for {variant_type} ({subtask})")
                    result[variant_type] = []
            
            logger.info(f"   ✅ Batched generation complete: {sum(len(q) for q in result.values())} total queries")
            
        except Exception as e:
            logger.error(f"Error in batched generation for {problem_id} {query_type}: {str(e)}")
            logger.error(f"   Falling back to empty results")
            self.stats['errors'] += 1
            # Return empty results for all variants
            for variant_type in variants_to_generate.keys():
                result[variant_type] = []
        
        return result

    def _generate_queries_for_type(self, problem_id: str, query_type: str,
                                   available_languages: List[str], variant_info: Dict = None) -> List[RetrievalQuery]:
        """Generate queries for a specific type using LLM (legacy method, kept for compatibility)."""
        key = f'{problem_id}_{query_type}'

        queries = []

        # Use cached context if available (saves ~95% of similarity computations)
        # All variants of the same problem share the same context (target + similar problems)
        if problem_id not in self.context_cache:
            target_problem, similar_problem_list = self._format_smart_context(problem_id)
            self.context_cache[problem_id] = (target_problem, similar_problem_list)
            logger.debug(f"💾 Cached context for {problem_id}")
        else:
            target_problem, similar_problem_list = self.context_cache[problem_id]
            logger.debug(f"⚡ Using cached context for {problem_id}")

        # Prepare context for LLM
        context = {
            'target_problem': target_problem,
            'similar_problem_list': similar_problem_list,
            'query_type': query_type,
            'available_languages': available_languages,
            'variant_info': variant_info
        }

        # Generate queries using LLM if enabled
        try:
            prompt = self._create_query_generation_prompt(context)
            response = self.llm.generate(prompt)
            logger.debug(
                f"      LLM response received for {problem_id} {query_type}: {len(str(response))} chars")

            # Parse LLM response
            parsed_response = extract_json_from_text(response)
            generated_queries = parsed_response.get('queries', [])
            queries.extend(generated_queries)
            logger.debug(f"      Parsed {len(generated_queries)} queries from LLM response")

            # Note: Queries will be saved as preprocessed queries in the main loop

        except Exception as e:
            logger.error(f"Error generating queries for {problem_id} {query_type}: {str(e)}")
            self.stats['errors'] += 1

        return queries

    def _get_query_variants(self, query_type: str, available_languages: List[str]) -> Dict[str, Dict]:
        """
        Generate both language-agnostic and language-specific variants for a query type.
        
        For all query types (title_search, description_search, algorithm_search):
        1. One base language-agnostic version (retro_any)
        2. N language-specific versions (retro_<lang>), one for each available language
        
        Args:
            query_type: Base query type (e.g., 'algorithm_search', 'description_search', 'title_search')
            available_languages: List of available languages for the problem (e.g., ['python', 'cpp', 'java'])
            
        Returns:
            Dictionary mapping variant type to variant info. Example:
            {
                'algorithm_search': {'subtask': 'retro_any', 'language_constraint': 'none', 'target_language': None},
                'algorithm_search_python': {'subtask': 'retro_python', 'language_constraint': 'python', 'target_language': 'python'},
                'algorithm_search_cpp': {'subtask': 'retro_cpp', 'language_constraint': 'cpp', 'target_language': 'cpp'},
            }
        """
        variants = {}
        
        # 1. Generate the base language-agnostic version (retro_any - Retrieval, any language)
        variants[query_type] = {
            'subtask': 'retro_any',
            'language_constraint': 'none',
            'target_language': None
        }
        
        # 2. Generate language-specific versions for each available language (retro_<lang> - Retrieval, specific language)
        for language in available_languages:
            language_variant = f"{query_type}_{language}"
            variants[language_variant] = {
                'subtask': f'retro_{language}',
                'language_constraint': language,
                'target_language': language
            }
        
        return variants

    def _preprocess_queries(self, queries: List[Dict], problem_id: str, query_type: str, 
                           available_languages: List[str], variant_info: Dict = None) -> List[Dict]:
        """
        Preprocess queries to add language constraints, subtask information, and model overlap metadata.
        
        For overlap problems (solved by multiple models), marks Claude Sonnet as the reference model.
        
        Args:
            queries: Raw queries from LLM generation
            problem_id: Problem identifier
            query_type: Query type variant (e.g., 'algorithm_search' or 'algorithm_search_python')
            available_languages: Available languages for the problem
            variant_info: Pre-computed variant information (subtask, language_constraint, target_language)
            
        Returns:
            List of preprocessed query dictionaries with full metadata including overlap information
        """
        if not queries:
            return []
            
        # Extract metadata from variant_info or determine from query_type
        if variant_info:
            subtask = variant_info['subtask']
            language_constraint = variant_info['language_constraint']
            target_language = variant_info.get('target_language')
        else:
            # Fallback: determine from query type (backward compatibility)
            subtask, language_constraint = self._determine_subtask_and_constraint(query_type, available_languages)
            target_language = None
        
        # Get model overlap information for this problem-language
        overlap_info = self._get_overlap_info_for_problem(problem_id, target_language)
        
        preprocessed = []
        for i, query_data in enumerate(queries):
            # Extract query text (handle both dict and string formats)
            query_text = query_data.get('query', '') if isinstance(query_data, dict) else str(query_data)
            
            if not query_text.strip():
                logger.warning(f"Empty query text for {problem_id}_{query_type}_{i:03d}, skipping")
                continue
            
            preprocessed_query = {
                'problem_id': problem_id,
                'original_query_type': query_type,
                'subtask': subtask,
                'language_constraint': language_constraint,
                'query': query_text,
                'query_id': f"{problem_id}_{query_type}_{i:03d}",
                'available_languages': available_languages,
                'metadata': {
                    'generated_by': self.llm_name,
                    'query_type': query_type,
                    'subtask': subtask,
                    'language_constraint': language_constraint,
                    'target_language': target_language,
                    # Overlap information
                    'is_overlap': overlap_info['is_overlap'],
                    'reference_model': overlap_info['reference_model'],
                    'available_models': overlap_info['available_models'],
                    'overlap_note': overlap_info.get('note', '')
                }
            }
            preprocessed.append(preprocessed_query)
        
        return preprocessed
    
    def _get_overlap_info_for_problem(self, problem_id: str, language: str = None) -> Dict[str, Any]:
        """
        Get overlap information for a problem-language pair.
        
        Determines if multiple models solved this problem-language pair and which model
        is used as the reference (preferring Claude Sonnet for overlap cases).
        
        Args:
            problem_id: Problem identifier
            language: Specific language (None for language-agnostic queries)
            
        Returns:
            Dictionary with overlap metadata:
                - is_overlap: bool
                - reference_model: str (model used as reference)
                - available_models: List[str] (all models that solved this)
                - note: str (explanation)
        """
        if problem_id not in self.clean_eval_data:
            return {
                'is_overlap': False,
                'reference_model': 'unknown',
                'available_models': [],
                'note': 'Problem not found in evaluation data'
            }
        
        problem_data = self.clean_eval_data[problem_id]
        model_lang_solutions = problem_data.get('model_language_solutions', {})
        
        if language:
            # Language-specific query: check overlap for this specific language
            models_for_lang = []
            for model_lang_key, sol_data in model_lang_solutions.items():
                if sol_data['language'] == language:
                    models_for_lang.append(sol_data['model'])
            
            is_overlap = len(models_for_lang) > 1
            
            # Prefer Claude Sonnet as reference for overlap
            reference_model = None
            for model in models_for_lang:
                if model.startswith('bedrock-claude'):
                    reference_model = model
                    break
            
            if not reference_model and models_for_lang:
                reference_model = models_for_lang[0]
            
            return {
                'is_overlap': is_overlap,
                'reference_model': reference_model or 'unknown',
                'available_models': models_for_lang,
                'note': 'Claude Sonnet used as reference' if is_overlap and reference_model and reference_model.startswith('bedrock-claude') else ''
            }
        else:
            # Language-agnostic query: check overall overlap across all languages
            all_models = set()
            for sol_data in model_lang_solutions.values():
                all_models.add(sol_data['model'])
            
            is_overlap = len(all_models) > 1
            
            # For language-agnostic, indicate presence of multiple models
            reference_model = 'mixed'  # No single reference for language-agnostic
            
            return {
                'is_overlap': is_overlap,
                'reference_model': reference_model,
                'available_models': list(all_models),
                'note': f'Multiple models available across languages' if is_overlap else ''
            }

    def _determine_subtask_and_constraint(self, query_type: str, available_languages: List[str]) -> Tuple[str, str]:
        """
        Determine subtask and language constraint from query type.
        
        This is a fallback method for backward compatibility. New code should use variant_info.
        
        Args:
            query_type: Query type string (may include language suffix like 'algorithm_search_python')
            available_languages: Available languages for the problem
            
        Returns:
            Tuple of (subtask, language_constraint)
        """
        # Check if query_type has a language suffix
        parts = query_type.split('_')
        
        if len(parts) > 1 and parts[-1] in self.SUPPORTED_LANGUAGES:
            # Language-specific variant (e.g., 'algorithm_search_python')
            base_type = '_'.join(parts[:-1])
            language = parts[-1]
            return f'retro_{language}', language
        
        # Default to language-agnostic
        base_subtask = self.QUERY_TYPE_SUBTASK_MAP.get(query_type, 'retro_any')
        return base_subtask, 'none'

    def _create_batched_query_prompt(self, context: Dict) -> str:
        """
        Create prompt for generating ALL language variants in ONE call.
        
        Args:
            context: Context dictionary containing:
                - target_problem: Target problem description
                - similar_problem_list: Similar problems for context
                - query_type: Base query type (e.g., 'title_search')
                - available_languages: List of available languages
                - variants_to_generate: Dict of variant_type -> variant_info
                
        Returns:
            Formatted prompt string requesting all variants
        """
        query_type = context['query_type']
        target_problem = context['target_problem']
        similar_problem_list = context['similar_problem_list']
        available_languages = context['available_languages']
        variants = context['variants_to_generate']
        
        # Build list of language variants to generate
        variant_list = []
        variant_list.append("- **retro_any**: Language-agnostic queries (work for any programming language)")
        
        for lang in available_languages:
            lang_name = {
                'python': 'Python',
                'cpp': 'C++',
                'java': 'Java',
                'go': 'Go',
                'ruby': 'Ruby'
            }.get(lang, lang.capitalize())
            variant_list.append(f"- **retro_{lang}**: {lang_name}-specific queries (mention {lang_name} explicitly)")
        
        variants_text = "\n".join(variant_list)
        
        # Create batched prompt
        prompt = f"""# Batch Code Retrieval Query Generation

## Task
Generate diverse search queries for the **{query_type}** query type across MULTIPLE language variants in ONE response.

## Target Problem
{target_problem}

## Similar Problems (for context)
{similar_problem_list}

## Query Variants to Generate
Generate {context.get('max_queries_per_problem', 2)} queries for EACH of the following variants:

{variants_text}

## Guidelines
- **Natural language:** Write queries like real developers searching
- **Query length:** 15-40 words per query
- **Target-specific:** Focus on aspects UNIQUE to the target problem
- **Differentiation:** Avoid terms shared with similar problems
- **Language-specific variants:** Explicitly mention the programming language
- **Language-agnostic variant:** Focus on algorithmic concepts without mentioning specific languages

## Output Format
Return a JSON object with queries grouped by subtask:

```json
{{
  "retro_any": [
    {{"query": "algorithmic approach for...", "best_match": "problem_id", "best_reason": "..."}},
    {{"query": "solution pattern for...", "best_match": "problem_id", "best_reason": "..."}}
  ],
  "retro_python": [
    {{"query": "python implementation using...", "best_match": "problem_id", "best_reason": "..."}},
    {{"query": "python code for...", "best_match": "problem_id", "best_reason": "..."}}
  ],
  "retro_cpp": [
    {{"query": "c++ solution with...", "best_match": "problem_id", "best_reason": "..."}},
    {{"query": "cpp implementation of...", "best_match": "problem_id", "best_reason": "..."}}
  ]
  // ... (one entry per available language)
}}
```

Generate {context.get('max_queries_per_problem', 2)} queries for EACH variant."""
        
        return prompt

    def _create_query_generation_prompt(self, context: Dict) -> str:
        """
        Create prompt for LLM query generation using refined prompts.
        
        Args:
            context: Context dictionary containing:
                - target_problem: Target problem description
                - similar_problem_list: Similar problems for context
                - query_type: Type of query to generate
                - variant_info: Optional variant information (for language-specific queries)
                
        Returns:
            Formatted prompt string for LLM
        """
        query_type = context['query_type']
        variant_info = context.get('variant_info', {})
        target_problem = context['target_problem']
        similar_problem_list = context['similar_problem_list']

        # Check if this is a language-specific variant
        target_language = variant_info.get('target_language') if variant_info else None
        
        if target_language:
            # Language-specific query generation
            return self._create_language_specific_prompt(
                query_type, target_language, target_problem, similar_problem_list
            )
        else:
            # Language-agnostic query generation
            return self._create_language_agnostic_prompt(
                query_type, target_problem, similar_problem_list
            )

    def _create_language_specific_prompt(self, query_type: str, target_language: str, 
                                        target_problem: str, similar_problem_list: str) -> str:
        """Create prompt for language-specific query generation using CrossLanguageQueryGenPrompt."""
        # Extract base query type (remove language suffix if present)
        base_type = query_type.rsplit('_', 1)[0] if '_' in query_type else query_type
        
        # Use CrossLanguageQueryGenPrompt which explicitly instructs to mention language naturally
        prompt_template = CrossLanguageQueryGenPrompt.init()
        
        # Format with language-specific context
        return prompt_template.format(
            target_problem=target_problem,
            similar_problem_list=similar_problem_list,
            query_type=base_type,
            target_language=target_language
        )

    def _create_language_agnostic_prompt(self, query_type: str, target_problem: str, 
                                        similar_problem_list: str) -> str:
        """Create prompt for language-agnostic query generation."""
        # Select prompt template based on query type
        if query_type == 'algorithm_search':
            prompt_template = AlgorithmFocusedQueryGenPrompt.init()
        else:
            prompt_template = QueryGenPrompt.init()
        
        return prompt_template.format(
            target_problem=target_problem,
            similar_problem_list=similar_problem_list,
            query_type=query_type
        )


    def _calculate_semantic_similarity(self, target_problem_id: str, other_problem_id: str) -> float:
        """Calculate semantic similarity between two problems using sentence transformer."""
        
        # Get formatted_text from clean_eval_data where it's pre-computed
        target_text = None
        other_text = None

        if target_problem_id in self.clean_eval_data:
            target_text = self.clean_eval_data[target_problem_id].get('formatted_text')
        if other_problem_id in self.clean_eval_data:
            other_text = self.clean_eval_data[other_problem_id].get('formatted_text')

        # Skip if texts are not available or too short
        if not target_text or not other_text or len(target_text.strip()) < 10 or len(other_text.strip()) < 10:
            return 0.0

        # Get embeddings
        target_embedding = self.sentence_transformer.encode(
            [target_text],
            convert_to_tensor=False,
            show_progress_bar=False,
            batch_size=1
        )
        other_embedding = self.sentence_transformer.encode(
            [other_text],
            convert_to_tensor=False,
            show_progress_bar=False,
            batch_size=1
        )

        # Ensure embeddings are numpy arrays
        if hasattr(target_embedding, 'cpu'):
            target_embedding = target_embedding.cpu().numpy()
        if hasattr(other_embedding, 'cpu'):
            other_embedding = other_embedding.cpu().numpy()

        # Calculate cosine similarity
        target_vec = target_embedding[0]
        other_vec = other_embedding[0]

        target_norm = np.linalg.norm(target_vec)
        other_norm = np.linalg.norm(other_vec)

        if target_norm == 0 or other_norm == 0:
            return 0.0

        similarity = np.dot(target_vec, other_vec) / (target_norm * other_norm)

        if np.isnan(similarity) or np.isinf(similarity):
            return 0.0

        return float(similarity)

    def _format_smart_context(self, target_problem_id) -> Tuple[str]:
        """Format context with target problem + top 5 similar problems based on question content similarity."""
        
        # Calculate similarity scores for all problems (excluding target)
        problem_scores = []
        
        for problem_id, problem_data in self.question_content_data.items():
            if problem_id == target_problem_id:
                continue
                
            # Skip if no passed solutions (not in clean_eval_data)
            if problem_id not in self.clean_eval_data:
                continue
                
            # Calculate semantic similarity
            similarity_score = self._calculate_semantic_similarity(target_problem_id, problem_id)
            
            if similarity_score > 0.1:  # Only include problems with meaningful similarity
                problem_scores.append((similarity_score, problem_id))
        
        # Sort by similarity score and take top 5 problems
        problem_scores.sort(key=lambda x: x[0], reverse=True)
        top_problems = problem_scores[:5]  # Top 5 similar problems
        
        logger.debug(f"Selected {len(top_problems)} similar problems for {target_problem_id}")
        
        # Build final formatted output using pre-computed formatted text
        similar_problem_list = []
        
        # Add target problem first
        if target_problem_id in self.clean_eval_data:
            target_problem = self.clean_eval_data[target_problem_id]['formatted_text']
        
        # Add top similar problems
        for score, problem_id in top_problems:
            # Take first 1000 characters of the formatted text, otherwise it's too long
            similar_problem_list.append(self.clean_eval_data[problem_id]['formatted_text'][:500])
            logger.debug(f"  {problem_id}: similarity {score:.3f}")
        
        return target_problem, "\n".join(similar_problem_list)

    def _print_summary(self):
        """Print summary using logger."""
        logger.info("\n" + "=" * 60)
        logger.info("🎯 CODE QUERY GENERATION SUMMARY")
        logger.info("=" * 60)

        logger.info(f"📊 Generation Statistics:")
        logger.info(f"   • Total Queries Generated: {len(self.retrieval_queries)}")
        logger.info(f"   • Total Problems Processed: {len(self.question_content_data)}")
        logger.info(f"   • Languages: {', '.join(self.stats['languages'])}")
        logger.info(f"   • Models: {', '.join(self.stats['models'])}")
        logger.info(f"   • Errors Encountered: {self.stats['errors']}")
        logger.info(f"   • Fallback Queries Used: {self.stats['fallback_used']}")

        # Add solution statistics
        total_problems_with_solutions = 0
        total_passed_solutions = 0
        for problem_id in self.question_content_data.keys():
            stats = self._get_solution_stats(problem_id)
            if stats['has_passed_solutions']:
                total_problems_with_solutions += 1
                total_passed_solutions += stats['passed_solutions']

        logger.info(f"   • Problems with Passed Solutions: {total_problems_with_solutions}")
        logger.info(f"   • Total Passed Solutions: {total_passed_solutions}")

        # Query type breakdown
        query_type_counts = defaultdict(int)
        for query in self.retrieval_queries:
            # Query is a dict, access query_type from metadata or original_query_type
            query_type = query.get('original_query_type') or query.get('metadata', {}).get('query_type', 'unknown')
            query_type_counts[query_type] += 1

        logger.info(f"\n📈 Query Type Distribution:")
        for query_type, count in sorted(query_type_counts.items()):
            logger.info(f"   • {query_type}: {count}")

        logger.info("=" * 60)
