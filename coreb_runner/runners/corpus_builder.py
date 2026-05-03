"""
Corpus Builder

Builds clean, HuggingFace-ready code and text corpora from evaluation results and annotated data.

Output:
- code_corpus: All code solutions with metadata
- text_corpus: Problem descriptions (title and title+description)
"""

import json
from datetime import datetime
from typing import Dict, Any, List
from collections import defaultdict

from coreb_runner.runners.base_runner import Runner
from coreb_runner.utils.code_utils import extract_code, parse_solution_key
from easyllm_kit.utils import get_logger, read_json, save_json


logger = get_logger('corpus_builder')


@Runner.register("corpus_builder")
class CorpusBuilder(Runner):
    """Build code and text corpora from evaluation and annotation data."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.data_config = config.get('data', {})
        self.corpus_config = config.get('corpus', {})
        
        # Input paths - support both old (single file) and new (multiple files) formats
        self.eval_data_files = self.data_config.get('eval_data_files') or [self.data_config.get('eval_data_dir')]
        self.source_data_files = self.data_config.get('source_data_files') or [self.data_config.get('source_data_dir')]
        self.annotated_data_dir = self.data_config.get('annotated_data_dir')
        self.abbrev_data_dir = self.data_config.get('abbrev_data_dir')  # Optional
        
        # Output paths
        self.code_corpus_output_dir = self.data_config.get('code_corpus_output_dir')
        self.text_corpus_output_dir = self.data_config.get('text_corpus_output_dir')
        
        # Validate required parameters
        if not self.code_corpus_output_dir:
            raise ValueError("Missing required parameter: data.code_corpus_output_dir")
        if not self.text_corpus_output_dir:
            raise ValueError("Missing required parameter: data.text_corpus_output_dir")
        
        # Corpus settings
        self.corpus_version = self.corpus_config.get('version')

        # Model filtering: specify which models to include in corpus
        # Format: list of model name prefixes or exact names
        # Examples: ['claude-4-sonnet', 'gemini-3-flash-preview'] or ['bedrock-claude', 'vertex_ai/gemini']
        # If None or empty, all models are included
        self.allowed_models = self.corpus_config.get('allowed_models')
        if self.allowed_models:
            logger.info(f"Model filter enabled: {self.allowed_models}")
            logger.info("  Only solutions from these models will be included in corpus")
        else:
            logger.info("No model filter specified - all models will be included")
        
        # Model name mapping: rename models in corpus output
        # Format: dict mapping original_name -> new_name
        # Applied after cleaning (removing bedrock- prefix) and filtering
        # Example: {'claude-4-sonnet': 'claude-4-sonnet', 'vertex_ai/gemini-3-flash-preview': 'gemini-3-flash'}
        self.model_name_mapping = self.corpus_config.get('model_name_mapping', {})
        if self.model_name_mapping:
            logger.info(f"Model name mapping enabled: {self.model_name_mapping}")
            logger.info("  Models will be renamed in corpus output")
        else:
            logger.info("No model name mapping specified - using original names")

        # Load and merge evaluation data from multiple files
        self.eval_data = self._load_and_merge_data(self.eval_data_files, "evaluation")
        
        # Load and merge source data (solutions) from multiple files
        self.source_data = self._load_and_merge_data(self.source_data_files, "source")
        
        # Load annotated data for problem descriptions (optional, falls back to source_data)
        if self.annotated_data_dir:
            logger.info(f"Loading annotated problem data from {self.annotated_data_dir}")
            self.annotated_data = read_json(self.annotated_data_dir)
            logger.info(f"Loaded {len(self.annotated_data)} annotated problems")
        else:
            logger.info("No annotated data provided, using source data for problem descriptions")
            self.annotated_data = self.source_data
        
        # Load abbreviation data (optional)
        self.abbrev_data = {}
        if self.abbrev_data_dir:
            logger.info(f"Loading abbreviation data from {self.abbrev_data_dir}")
            self.abbrev_data = read_json(self.abbrev_data_dir)
            logger.info(f"Loaded {len(self.abbrev_data)} abbreviated texts")
        else:
            logger.info("No abbreviation data provided, skipping")
        
        logger.info(f"Total: {len(self.eval_data)} eval entries, {len(self.source_data)} source problems")
    
    def _load_and_merge_data(self, file_paths: List[str], data_type: str) -> Dict[str, Any]:
        """Load and merge data from multiple files."""
        if not file_paths or not file_paths[0]:
            raise ValueError(f"No {data_type} files specified")
        
        merged_data = {}
        
        for file_path in file_paths:
            logger.info(f"Loading {data_type} data from {file_path}")
            data = read_json(file_path)
            logger.info(f"  Loaded {len(data)} entries")
            
            # Merge data (later files override earlier ones for same problem_id)
            for problem_id, problem_data in data.items():
                if problem_id not in merged_data:
                    merged_data[problem_id] = problem_data
                else:
                    # Merge: add new keys, keep existing keys
                    merged_data[problem_id].update(problem_data)
        
        logger.info(f"Merged {data_type} data: {len(merged_data)} total problems")
        return merged_data
    
    def _find_latest_round_by_name(self, rounds: Dict[str, Any]) -> str:
        """Find the latest round by analyzing round names when latest_round is not specified."""
        try:
            # Common round naming patterns in order of preference
            round_priority = [
                'round_4_ruby_evaluation',
                'round_3_non_python_fixes',
                'round_2_timeout_fix',
                'round_1',
                'round_2',
                'round_3'
            ]

            # First, try to find rounds in priority order
            for round_name in round_priority:
                if round_name in rounds:
                    return round_name

            # If no priority rounds found, get all round names and sort them
            round_names = list(rounds.keys())
            if not round_names:
                return None

            # Sort round names and return the last one (assuming lexicographic order works)
            round_names.sort()
            return round_names[-1]

        except Exception as e:
            logger.warning(f"Error finding latest round by name: {e}")
            return None
    
    def run(self):
        """Build both code and text corpora."""
        logger.info("🚀 Building code and text corpora")
        
        # Build code corpus
        code_corpus, code_stats = self._build_code_corpus()
        logger.info(f"Built code corpus with {len(code_corpus)} entries")
        
        # Build text corpus
        text_corpus, text_stats = self._build_text_corpus()
        logger.info(f"Built text corpus with {len(text_corpus)} entries")
        
        # Export code corpus
        self._export_corpus(code_corpus, self.code_corpus_output_dir, code_stats, corpus_type='code')
        
        # Export text corpus
        self._export_corpus(text_corpus, self.text_corpus_output_dir, text_stats, corpus_type='text')
        
        logger.info("✅ Corpus building complete")
    
    def _build_code_corpus(self) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Build code corpus from all solutions (correct + incorrect).
        
        Inclusion policy:
        - ALL solutions are included regardless of pass/fail status
        - Each entry has meta.passed flag (True/False) for filtering downstream
        - This allows dataset_maker to use incorrect solutions as hard negatives
        
        Overlap handling:
        - Multiple models may have solutions for same problem-language pair
        - Each solution gets a unique code_id
        - Model name is stored in 'model' field for overlap detection downstream
        - Dataset_maker uses problem_id + model to track overlap (see OVERLAP_HANDLING.md)
        """
        corpus = []
        stats = {
            'by_language': defaultdict(lambda: {'count': 0, 'total_pass_rate': 0.0}),
            'by_model': defaultdict(lambda: {'count': 0, 'total_pass_rate': 0.0})
        }
        code_id_counter = 1
        
        for problem_id, eval_content in self.eval_data.items():
            # Get source problem data
            if problem_id not in self.source_data:
                logger.warning(f"Problem {problem_id} not found in source data, skipping")
                continue
            
            source_problem = self.source_data[problem_id]
            
            # Process each model_language evaluation
            for key, eval_result in eval_content.items():
                if not key.endswith('_evaluation'):
                    continue
                
                # Parse model and language
                key_without_eval = key.replace('_evaluation', '')
                model, language = parse_solution_key(key_without_eval)
                
                # Clean model name for output (remove 'bedrock-' prefix)
                clean_model = model.replace('bedrock-', '') if model.startswith('bedrock-') else model
                
                # Filter by allowed models if specified
                if self.allowed_models:
                    # Check if this model matches any allowed model prefix/name
                    model_matches = any(
                        clean_model.startswith(allowed) or 
                        clean_model == allowed or
                        model.startswith(allowed) or
                        model == allowed
                        for allowed in self.allowed_models
                    )
                    if not model_matches:
                        logger.debug(f"Skipping {clean_model} (not in allowed_models)")
                        continue
                
                # Apply model name mapping if specified
                # Map from cleaned name to desired output name
                final_model_name = self.model_name_mapping.get(clean_model, clean_model)
                if final_model_name != clean_model:
                    logger.debug(f"Renaming model: {clean_model} -> {final_model_name}")
                
                if not isinstance(eval_result, dict):
                    continue
                
                # Get evaluation metrics from latest round
                passed = False
                pass_rate = 0.0
                test_passed = 0
                test_total = 0
                
                # Check if solution passed (look in rounds or evaluations)
                if 'rounds' in eval_result:
                    latest_round_id = eval_result.get('latest_round', '')
                    
                    # If no latest_round specified, find the latest round by name
                    if not latest_round_id:
                        latest_round_id = self._find_latest_round_by_name(eval_result['rounds'])
                    
                    if latest_round_id and latest_round_id in eval_result['rounds']:
                        round_data = eval_result['rounds'][latest_round_id]
                        evaluations = round_data.get('evaluations', [])
                        
                        # Count passed solutions
                        total_solutions = len(evaluations)
                        passed_count = sum(1 for eval_item in evaluations 
                                         if isinstance(eval_item, dict) and eval_item.get('overall_result') == 'passed')
                        
                        # Calculate metrics
                        passed = passed_count > 0
                        pass_rate = passed_count / total_solutions if total_solutions > 0 else 0.0
                        test_passed = passed_count
                        test_total = total_solutions
                
                # Get code from source data using the original key format
                solution_key = key_without_eval  # This is the original key without '_evaluation'
                if solution_key not in source_problem:
                    logger.debug(f"Solution key {solution_key} not found in {problem_id}")
                    continue
                
                solution_data = source_problem[solution_key]
                if not isinstance(solution_data, list) or not solution_data:
                    continue
                
                code = extract_code(solution_data[0])
                if not code:
                    continue
                
                # Compute code metrics
                code_length = len(code)
                
                # Create corpus entry (minimal, no information leakage)
                entry = {
                    'code_id': f"code_v202601_{code_id_counter:05d}",  # Unique ID for retrieval
                    'code': code,
                    'language': language,
                    'model': final_model_name,  # Mapped model name (used for overlap detection in dataset_maker)
                    'code_length': code_length,
                    'meta': {
                        'source_problem_id': problem_id,  # Links to queries and text corpus
                        'solution_key': solution_key,
                        'passed': passed,  # True/False - used to filter positives/negatives
                        'pass_rate': pass_rate,
                        'test_passed': test_passed,
                        'test_total': test_total,
                        'created_at': datetime.utcnow().isoformat() + 'Z',
                        'corpus_version': self.corpus_version
                    }
                }
                
                corpus.append(entry)
                code_id_counter += 1
                
                # Update stats
                stats['by_language'][language]['count'] += 1
                stats['by_language'][language]['total_pass_rate'] += pass_rate
                stats['by_model'][clean_model]['count'] += 1
                stats['by_model'][clean_model]['total_pass_rate'] += pass_rate
        
        # Compute averages
        summary_stats = self._compute_corpus_stats(corpus, stats)
        
        return corpus, summary_stats
    
    def _build_text_corpus(self) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Build text corpus from problem descriptions and abbreviations.
        
        Structure:
        - One entry per problem (not per model or language)
        - Contains full text (title + description)
        - Optional abbreviated_text field if available from step_6
        - Links to code corpus via meta.source_problem_id
        
        Note: Abbreviated texts are stored as optional field, not separate entries.
        This keeps the corpus simple while supporting multiple text granularities.
        """
        corpus = []
        stats = defaultdict(int)
        text_id_counter = 1
        
        for problem_id, source_problem in self.annotated_data.items():
            # Support multiple field name formats:
            # - ANNOTATED_TITLE/ANNOTATED_QUESTION_CONTENT (from solution files)
            # - annotate_question_title/annotate_question_content (from annotated files)
            title = source_problem.get('ANNOTATED_TITLE') or source_problem.get('annotate_question_title')
            description = source_problem.get('ANNOTATED_QUESTION_CONTENT') or source_problem.get('annotate_question_content')
            
            # Get abbreviated text if available
            abbreviated_text = None
            if problem_id in self.abbrev_data:
                abbrev_entry = self.abbrev_data[problem_id]
                if isinstance(abbrev_entry, dict):
                    abbreviated_text = abbrev_entry.get('abbreviated')
                else:
                    abbreviated_text = abbrev_entry
            
            # Create single text entry per problem (title + description)
            if title and description:
                text_content = f"{title}\n\n{description}"
                text_style = 'title_plus_description'
            elif title:
                text_content = title
                text_style = 'title_only'
            else:
                continue  # Skip problems without title
            
            # Create corpus entry
            entry = {
                'text_id': f"text_v202601_{text_id_counter:05d}",  # Unique ID for retrieval
                'text': text_content,  # Full text (title + description or title only)
                'text_style': text_style,  # 'title_plus_description' or 'title_only'
                'text_length': len(text_content),
                'meta': {
                    'source_problem_id': problem_id,  # Links to code corpus and queries
                    'created_at': datetime.utcnow().isoformat() + 'Z',
                    'corpus_version': self.corpus_version
                }
            }
            
            # Add abbreviated text if available (from step_6: question_abbrev_runner)
            # Stored as optional field, not separate entry - supports canonical_retro tasks
            if abbreviated_text:
                entry['abbreviated_text'] = abbreviated_text
                entry['abbreviated_text_length'] = len(abbreviated_text)
                stats['with_abbreviation'] += 1
            
            corpus.append(entry)
            text_id_counter += 1
            
            stats[f"style_{text_style}"] += 1
        
        summary_stats = {
            'total_entries': len(corpus),
            'unique_problems': len(self.annotated_data),
            'with_abbreviation': stats.get('with_abbreviation', 0),
            'abbreviation_coverage': f"{stats.get('with_abbreviation', 0) / len(corpus) * 100:.1f}%" if corpus else "0%",
            'by_style': {k.replace('style_', ''): v for k, v in stats.items() if k.startswith('style_')}
        }
        
        return corpus, summary_stats
    
    def _compute_corpus_stats(self, corpus: List[Dict[str, Any]], stats: Dict) -> Dict[str, Any]:
        """Compute summary statistics for corpus."""
        summary = {
            'corpus_version': self.corpus_version,
            'created_at': datetime.utcnow().isoformat() + 'Z',
            'total_entries': len(corpus),
            'unique_problems': len(set(entry['meta']['source_problem_id'] for entry in corpus)),
            'by_language': {},
            'by_model': {},
            'code_metrics': {
                'avg_length': sum(e['code_length'] for e in corpus) / len(corpus) if corpus else 0,
                'total_passed': sum(1 for e in corpus if e['meta']['passed']),
                'total_failed': sum(1 for e in corpus if not e['meta']['passed'])
            }
        }
        
        # Compute averages for each dimension
        for dimension in ['by_language', 'by_model']:
            for key, data in stats[dimension].items():
                if data['count'] > 0:
                    summary[dimension][key] = {
                        'count': data['count'],
                        'avg_pass_rate': data['total_pass_rate'] / data['count']
                    }
        
        return summary
    
    def _export_corpus(self, corpus: List[Dict[str, Any]], output_path: str,
                      stats: Dict[str, Any], corpus_type: str):
        """Export corpus to multiple formats."""
        # Export to JSONL
        logger.info(f"Exporting {corpus_type} corpus to {output_path}")
        
        # Save as JSONL (one JSON object per line)
        with open(output_path, 'w', encoding='utf-8') as f:
            for entry in corpus:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        # Export stats (replace .jsonl with _stats.json)
        stats_path = output_path.replace('.jsonl', '_stats.json')
        logger.info(f"Exporting {corpus_type} corpus stats to {stats_path}")
        
        save_json(stats, stats_path)
        
        logger.info(f"✅ Exported {len(corpus)} {corpus_type} corpus entries")

