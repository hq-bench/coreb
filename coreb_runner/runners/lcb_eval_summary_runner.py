"""
LiveCodeBench Evaluation Summary Runner - Pipeline Implementation

This module implements a LiveCodeBench evaluation summary runner that generates comprehensive
summary statistics from evaluation results, aggregating performance metrics by 
models and by languages.

Key Features:
- Aggregates evaluation results by models across all languages
- Aggregates evaluation results by languages across all models
- Provides detailed model-language combination statistics
- Generates leaderboard rankings
- Saves comprehensive summary reports in JSON format
- Integrates with the existing LiveCodeBench evaluation pipeline

Usage Example:

1. Configuration file (config.yaml):
```yaml
runner_name: "lcb_eval_summary"
data:
  input_db: "./output/evaluation_results.json"
  output_summary: "./output/evaluation_summary.json"
eval_summary:
  include_overall_stats: true
  include_model_stats: true
  include_language_stats: true
  include_combinations: true
  include_leaderboard: true
  leaderboard_top_n: 10
  sort_by: "pass_at_1"  # Options: pass_at_1, pass_at_5, pass_at_10, solution_pass_rate
```

2. Running the summary generation:
```python
from omegaconf import OmegaConf
from coreb_runner.runners import Runner

config = OmegaConf.load("config.yaml")
runner = Runner.build_from_config(config)
runner.run()
```

3. Expected input format:
The runner expects evaluation results to be stored in the database with keys like:
- "{model_name}_{language}_evaluation": Detailed evaluation results

4. Output format:
The runner generates a comprehensive JSON summary with sections:
- Overall statistics
- Model-wise performance
- Language-wise performance
- Model-language combination performance
- Leaderboard rankings
"""

import json
import logging
from collections import defaultdict
from typing import Dict, List, Any, Tuple
import numpy as np
from datetime import datetime
from dataclasses import dataclass

from coreb_runner.runners import Runner
from easyllm_kit.utils import read_json, get_logger

logger = get_logger('lcb_eval_summary_runner', 'lcb_eval_summary_runner.log')


@dataclass
class SummaryMetrics:
    """Data class for summary metrics."""
    num_problems: int
    num_solutions: int
    solutions_passed: int
    total_tests: int
    passed_tests: int
    pass_at_1: float
    pass_at_5: float
    pass_at_10: float
    solution_pass_rate: float
    test_pass_rate: float
    compilation_errors: int
    runtime_errors: int
    timeout_errors: int
    total_errors: int
    total_execution_time: float
    avg_execution_time: float
    min_execution_time: float
    max_execution_time: float


@Runner.register("lcb_eval_summary_runner")
class LCBEvalSummaryRunner(Runner):
    """
    LiveCodeBench runner for generating comprehensive evaluation summary statistics.
    
    This runner processes LiveCodeBench evaluation results and generates detailed summaries
    organized by models, languages, and their combinations. It provides insights
    into model performance across different programming languages and creates
    leaderboards for easy comparison.
    
    Configuration options:
    - data: Configuration for input/output data
    - eval_summary: Additional configuration for summary generation
      - include_overall_stats: Whether to include overall statistics (default: True)
      - include_model_stats: Whether to include model-wise statistics (default: True)
      - include_language_stats: Whether to include language-wise statistics (default: True)
      - include_combinations: Whether to include model-language combinations (default: True)
      - include_leaderboard: Whether to include leaderboard (default: True)
      - leaderboard_top_n: Number of top combinations to show (default: 10)
      - sort_by: Metric to sort leaderboard by (default: "pass_at_1")
    
    Example configuration:
    {
        "data": {
            "input_db": "./output/evaluation_results.json",
            "output_summary": "evaluation_summary.json"
        },
        "lcb_eval_summary": {
            "include_overall_stats": true,
            "include_model_stats": true,
            "include_language_stats": true,
            "include_combinations": true,
            "include_leaderboard": true,
            "leaderboard_top_n": 10,
            "sort_by": "pass_at_1"
        }
    }
    """

    def __init__(self, config):
        super().__init__()
        self.data_config = config.data
        self.config = config
        
        # Get evaluation summary specific config - handle both naming conventions
        eval_summary_config = config.get("eval_summary", config.get("lcb_eval_summary", {}))
        self.include_overall_stats = eval_summary_config.get("include_overall_stats", True)
        self.include_model_stats = eval_summary_config.get("include_model_stats", True)
        self.include_language_stats = eval_summary_config.get("include_language_stats", True)
        self.include_combinations = eval_summary_config.get("include_combinations", True)
        self.include_leaderboard = eval_summary_config.get("include_leaderboard", True)
        self.leaderboard_top_n = eval_summary_config.get("leaderboard_top_n", 10)
        self.sort_by = eval_summary_config.get("sort_by", "pass_at_1")
        
        logger.info(f"LiveCodeBench evaluation summary configured with:")
        logger.info(f"  Input file: {self.data_config.input_db}")
        logger.info(f"  Output file: {self.data_config.output_summary}")
        logger.info(f"  Include overall stats: {self.include_overall_stats}")
        logger.info(f"  Include model stats: {self.include_model_stats}")
        logger.info(f"  Include language stats: {self.include_language_stats}")
        logger.info(f"  Include combinations: {self.include_combinations}")
        logger.info(f"  Include leaderboard: {self.include_leaderboard}")
        logger.info(f"  Leaderboard top N: {self.leaderboard_top_n}")
        logger.info(f"  Sort by: {self.sort_by}")
        
        # Load evaluation results
        self.evaluation_results = read_json(self.data_config.input_db)
        logger.info(f"Loaded evaluation results for {len(self.evaluation_results)} problems")
        
        # Count evaluation entries to validate data
        eval_count = 0
        for problem_data in self.evaluation_results.values():
            for key in problem_data.keys():
                if key.endswith('_evaluation'):
                    eval_count += 1
        logger.info(f"Found {eval_count} evaluation entries in the dataset")
        
        # Initialize summary data
        self.summary_data = {
            'by_model': defaultdict(dict),
            'by_language': defaultdict(dict),
            'overall': {},
            'model_language_combinations': defaultdict(dict)
        }

    def extract_model_language_from_key(self, key: str) -> Tuple[str, str]:
        """Extract model name and language from evaluation key."""
        if not key.endswith('_evaluation'):
            return None, None
        
        # Remove '_evaluation' suffix
        base_key = key[:-11]  # len('_evaluation') = 11
        
        # Split by underscore to find model and language
        parts = base_key.split('_')
        
        # Common programming languages
        languages = ['python', 'java', 'cpp', 'c', 'javascript', 'go', 'rust', 'typescript']
        
        # Find the language (usually the last part)
        language = None
        model_name = None
        
        for i, part in enumerate(parts):
            if part.lower() in languages:
                language = part.lower()
                model_name = '_'.join(parts[:i])
                break
        
        # If no language found, assume the last part is language
        if language is None and parts:
            language = parts[-1].lower()
            model_name = '_'.join(parts[:-1])
        
        return model_name, language

    def _get_best_round_results(self, rounds: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get the best round results from multiple evaluation rounds.
        
        Args:
            rounds: Dictionary of round_id -> round_data
            
        Returns:
            Dict containing the best performing round's data
        """
        if not rounds:
            return {}
        
        # Find the round with the best pass_at_1 performance
        best_pass_at_1 = -1.0
        best_solutions_passed = -1
        best_round_data = None
        best_round_id = None
        
        for round_id, round_data in rounds.items():
            current_pass_at_1 = round_data.get('pass_at_1', 0.0)
            current_solutions_passed = round_data.get('solutions_passed', 0)
            
            # Prioritize by pass_at_1, then by solutions_passed
            if (current_pass_at_1 > best_pass_at_1 or 
                (current_pass_at_1 == best_pass_at_1 and current_solutions_passed > best_solutions_passed)):
                best_pass_at_1 = current_pass_at_1
                best_solutions_passed = current_solutions_passed
                best_round_data = round_data.copy()
                best_round_id = round_id
        
        # Add metadata about the merging
        if best_round_data:
            best_round_data['merged_from_rounds'] = list(rounds.keys())
            best_round_data['best_performance_round'] = best_round_id
            best_round_data['total_rounds_available'] = len(rounds)
        
        return best_round_data or {}

    def calculate_metrics_for_evaluation(self, evaluation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive metrics for a single evaluation (handles multi-round structure)."""
        metrics = {}
        
        # Handle multi-round evaluation structure
        if 'rounds' in evaluation_data:
            # Use the best round results for multi-round evaluations
            round_data = self._get_best_round_results(evaluation_data['rounds'])
        else:
            # Legacy format - use the data directly
            round_data = evaluation_data
        
        # Basic statistics from round data
        metrics['num_solutions'] = round_data.get('num_solutions', 0)
        metrics['solutions_passed'] = round_data.get('solutions_passed', 0)
        
        # Calculate total tests and passed tests from evaluations
        evaluations = round_data.get('evaluations', [])
        total_tests = 0
        passed_tests = 0
        total_execution_time = 0.0
        
        # Error counts
        compilation_errors = 0
        runtime_errors = 0
        timeout_errors = 0
        
        for eval_item in evaluations:
            total_tests += eval_item.get('total_tests', 0)
            passed_tests += eval_item.get('passed_tests', 0)
            total_execution_time += eval_item.get('execution_time', 0.0)
            
            # Count errors from test results
            test_results = eval_item.get('test_results', [])
            for test_result in test_results:
                result = test_result.get('result', '')
                if result == 'compilation_error':
                    compilation_errors += 1
                elif result == 'runtime_error':
                    runtime_errors += 1
                elif result == 'timeout':
                    timeout_errors += 1
        
        metrics['total_tests'] = total_tests
        metrics['passed_tests'] = passed_tests
        metrics['total_execution_time'] = total_execution_time
        
        # Pass rates
        if metrics['num_solutions'] > 0:
            metrics['solution_pass_rate'] = metrics['solutions_passed'] / metrics['num_solutions']
        else:
            metrics['solution_pass_rate'] = 0.0
        
        if metrics['total_tests'] > 0:
            metrics['test_pass_rate'] = metrics['passed_tests'] / metrics['total_tests']
        else:
            metrics['test_pass_rate'] = 0.0
        
        # Pass@k metrics from round data
        for k in [1, 5, 10]:
            pass_at_k_key = f'pass_at_{k}'
            metrics[pass_at_k_key] = round_data.get(pass_at_k_key, 0.0)
        
        # Error analysis
        metrics['compilation_errors'] = compilation_errors
        metrics['runtime_errors'] = runtime_errors
        metrics['timeout_errors'] = timeout_errors
        metrics['total_errors'] = compilation_errors + runtime_errors + timeout_errors
        
        # Average execution time per test
        if metrics['total_tests'] > 0:
            metrics['avg_execution_time'] = total_execution_time / metrics['total_tests']
        else:
            metrics['avg_execution_time'] = 0.0
        
        # Store round information for debugging
        metrics['latest_round'] = evaluation_data.get('latest_round', 'N/A')
        metrics['total_rounds'] = evaluation_data.get('total_rounds', 1)
        
        return metrics

    def aggregate_metrics(self, metrics_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate metrics from multiple evaluations."""
        if not metrics_list:
            return {}
        
        aggregated = {}
        
        # Count metrics (sum across all evaluations)
        count_metrics = ['num_solutions', 'solutions_passed', 'total_tests', 'passed_tests',
                        'compilation_errors', 'runtime_errors', 'timeout_errors', 'total_errors']
        
        for metric in count_metrics:
            aggregated[metric] = sum(m.get(metric, 0) for m in metrics_list)
        
        # Average metrics (weighted by number of solutions/tests)
        avg_metrics = ['solution_pass_rate', 'test_pass_rate', 'pass_at_1', 'pass_at_5', 'pass_at_10']
        
        for metric in avg_metrics:
            values = [m.get(metric, 0.0) for m in metrics_list if m.get(metric, 0.0) is not None]
            if values:
                aggregated[metric] = np.mean(values)
            else:
                aggregated[metric] = 0.0
        
        # Time metrics - handle both individual and total execution times
        execution_times = []
        total_execution_times = []
        avg_execution_times = []
        
        for m in metrics_list:
            if 'total_execution_time' in m:
                total_execution_times.append(m['total_execution_time'])
            if 'avg_execution_time' in m:
                avg_execution_times.append(m['avg_execution_time'])
            if 'execution_time' in m:  # Legacy support
                execution_times.append(m['execution_time'])
        
        # Aggregate execution time statistics
        all_times = total_execution_times + execution_times
        if all_times:
            aggregated['total_execution_time'] = sum(total_execution_times) if total_execution_times else sum(execution_times)
            aggregated['avg_execution_time'] = np.mean(avg_execution_times) if avg_execution_times else np.mean(all_times)
            aggregated['min_execution_time'] = np.min(all_times)
            aggregated['max_execution_time'] = np.max(all_times)
        else:
            aggregated['total_execution_time'] = 0.0
            aggregated['avg_execution_time'] = 0.0
            aggregated['min_execution_time'] = 0.0
            aggregated['max_execution_time'] = 0.0
        
        # Calculate overall pass rates
        if aggregated['num_solutions'] > 0:
            aggregated['overall_solution_pass_rate'] = aggregated['solutions_passed'] / aggregated['num_solutions']
        else:
            aggregated['overall_solution_pass_rate'] = 0.0
        
        if aggregated['total_tests'] > 0:
            aggregated['overall_test_pass_rate'] = aggregated['passed_tests'] / aggregated['total_tests']
        else:
            aggregated['overall_test_pass_rate'] = 0.0
        
        # Add metadata about rounds
        total_rounds = sum(m.get('total_rounds', 1) for m in metrics_list)
        unique_rounds = set(m.get('latest_round', 'N/A') for m in metrics_list)
        aggregated['total_evaluation_rounds'] = total_rounds
        aggregated['unique_rounds'] = list(unique_rounds)
        
        # Calculate error rates
        if aggregated['total_tests'] > 0:
            aggregated['compilation_error_rate'] = aggregated['compilation_errors'] / aggregated['total_tests']
            aggregated['runtime_error_rate'] = aggregated['runtime_errors'] / aggregated['total_tests']
            aggregated['timeout_error_rate'] = aggregated['timeout_errors'] / aggregated['total_tests']
            aggregated['overall_error_rate'] = aggregated['total_errors'] / aggregated['total_tests']
        else:
            aggregated['compilation_error_rate'] = 0.0
            aggregated['runtime_error_rate'] = 0.0
            aggregated['timeout_error_rate'] = 0.0
            aggregated['overall_error_rate'] = 0.0
        
        return aggregated

    def _generate_pass_at_1_breakdown(self, model_language_evaluations: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Generate a structured Pass@1 breakdown by model and language."""
        breakdown = {
            'by_model': defaultdict(dict),
            'by_language': defaultdict(dict),
            'matrix': {},
            'summary_table': []
        }
        
        # Extract models and languages
        models = set()
        languages = set()
        
        for combo in model_language_evaluations.keys():
            if '_' in combo:
                # Split at the last underscore to handle model names with underscores
                parts = combo.rsplit('_', 1)
                if len(parts) == 2:
                    model, language = parts
                    models.add(model)
                    languages.add(language)
        
        models = sorted(models)
        languages = sorted(languages)
        
        # Create matrix structure
        breakdown['matrix'] = {model: {} for model in models}
        
        # Fill in the data
        for combo, evaluations in model_language_evaluations.items():
            if '_' in combo:
                parts = combo.rsplit('_', 1)
                if len(parts) == 2:
                    model, language = parts
                    
                    # Calculate metrics for this combination
                    combo_metrics = self.aggregate_metrics(evaluations)
                    pass_at_1 = combo_metrics.get('pass_at_1', 0.0)
                    num_problems = len(evaluations)
                    solutions_passed = combo_metrics.get('solutions_passed', 0)
                    total_solutions = combo_metrics.get('num_solutions', 0)
                    
                    # Store in matrix
                    breakdown['matrix'][model][language] = {
                        'pass_at_1': pass_at_1,
                        'pass_at_1_percent': pass_at_1 * 100,
                        'solutions_passed': solutions_passed,
                        'total_solutions': total_solutions,
                        'num_problems': num_problems
                    }
                    
                    # Store in by_model breakdown
                    breakdown['by_model'][model][language] = pass_at_1
                    
                    # Store in by_language breakdown
                    breakdown['by_language'][language][model] = pass_at_1
                    
                    # Add to summary table
                    breakdown['summary_table'].append({
                        'model': model,
                        'language': language,
                        'combination': combo,
                        'pass_at_1': pass_at_1,
                        'pass_at_1_percent': pass_at_1 * 100,
                        'solutions_passed': solutions_passed,
                        'total_solutions': total_solutions,
                        'num_problems': num_problems
                    })
        
        # Sort summary table by Pass@1 descending
        breakdown['summary_table'].sort(key=lambda x: x['pass_at_1'], reverse=True)
        
        # Add model and language lists for reference
        breakdown['models'] = models
        breakdown['languages'] = languages
        
        return breakdown

    def _format_pass_at_1_matrix(self) -> str:
        """Format Pass@1 breakdown as a readable matrix table."""
        if 'pass_at_1_breakdown' not in self.summary_data:
            return "Pass@1 breakdown not available"
        
        breakdown = self.summary_data['pass_at_1_breakdown']
        models = breakdown.get('models', [])
        languages = breakdown.get('languages', [])
        matrix = breakdown.get('matrix', {})
        
        if not models or not languages:
            return "No model-language combinations found"
        
        lines = []
        
        # Create header
        header = f"{'Model':<25}"
        for lang in languages:
            header += f"{lang.upper():<12}"
        lines.append(header)
        lines.append("-" * len(header))
        
        # Create rows for each model
        for model in models:
            row = f"{model:<25}"
            for lang in languages:
                if model in matrix and lang in matrix[model]:
                    pass_at_1_percent = matrix[model][lang]['pass_at_1_percent']
                    row += f"{pass_at_1_percent:>6.1f}%    "
                else:
                    row += f"{'N/A':<12}"
            lines.append(row)
        
        # Add summary table
        lines.append("\n" + "Detailed Breakdown:")
        lines.append("-" * 70)
        lines.append(f"{'Rank':<4} {'Model-Language':<30} {'Pass@1':<8} {'Passed/Total':<12} {'Problems':<8}")
        lines.append("-" * 70)
        
        summary_table = breakdown.get('summary_table', [])
        for i, entry in enumerate(summary_table, 1):
            lines.append(
                f"{i:<4} {entry['combination']:<30} "
                f"{entry['pass_at_1_percent']:>6.1f}% "
                f"{entry['solutions_passed']:>3}/{entry['total_solutions']:<6} "
                f"{entry['num_problems']:<8}"
            )
        
        return "\n".join(lines)

    def generate_summary_statistics(self):
        """Generate comprehensive summary statistics."""
        logger.info("Generating summary statistics...")
        
        # Collect all evaluations
        model_evaluations = defaultdict(list)
        language_evaluations = defaultdict(list)
        model_language_evaluations = defaultdict(list)
        all_evaluations = []
        
        for problem_id, problem_data in self.evaluation_results.items():
            for key, value in problem_data.items():
                if key.endswith('_evaluation') and isinstance(value, dict):
                    model_name, language = self.extract_model_language_from_key(key)
                    
                    if model_name and language:
                        metrics = self.calculate_metrics_for_evaluation(value)
                        
                        # Add problem context
                        metrics['problem_id'] = problem_id
                        metrics['difficulty'] = problem_data.get('lcb_difficulty', 'unknown')
                        
                        # Collect by model
                        model_evaluations[model_name].append(metrics)
                        
                        # Collect by language
                        language_evaluations[language].append(metrics)
                        
                        # Collect by model-language combination
                        model_language_key = f"{model_name}_{language}"
                        model_language_evaluations[model_language_key].append(metrics)
                        
                        # Collect all
                        all_evaluations.append(metrics)
        
        # Aggregate by model
        for model_name, evaluations in model_evaluations.items():
            self.summary_data['by_model'][model_name] = self.aggregate_metrics(evaluations)
            self.summary_data['by_model'][model_name]['num_problems'] = len(evaluations)
        
        # Aggregate by language
        for language, evaluations in language_evaluations.items():
            self.summary_data['by_language'][language] = self.aggregate_metrics(evaluations)
            self.summary_data['by_language'][language]['num_problems'] = len(evaluations)
        
        # Aggregate by model-language combination
        for combo, evaluations in model_language_evaluations.items():
            self.summary_data['model_language_combinations'][combo] = self.aggregate_metrics(evaluations)
            self.summary_data['model_language_combinations'][combo]['num_problems'] = len(evaluations)
        
        # Overall statistics
        self.summary_data['overall'] = self.aggregate_metrics(all_evaluations)
        self.summary_data['overall']['num_problems'] = len(set(m['problem_id'] for m in all_evaluations))
        self.summary_data['overall']['num_models'] = len(model_evaluations)
        self.summary_data['overall']['num_languages'] = len(language_evaluations)
        self.summary_data['overall']['num_combinations'] = len(model_language_evaluations)
        
        # Generate difficulty-based breakdown
        difficulty_breakdown = defaultdict(list)
        for evaluation in all_evaluations:
            difficulty = evaluation.get('difficulty', 'unknown')
            difficulty_breakdown[difficulty].append(evaluation)
        
        self.summary_data['by_difficulty'] = {}
        for difficulty, evaluations in difficulty_breakdown.items():
            self.summary_data['by_difficulty'][difficulty] = self.aggregate_metrics(evaluations)
            self.summary_data['by_difficulty'][difficulty]['num_problems'] = len(set(m['problem_id'] for m in evaluations))
        
        # Generate Pass@1 breakdown by model-language combinations
        self.summary_data['pass_at_1_breakdown'] = self._generate_pass_at_1_breakdown(model_language_evaluations)
        
        logger.info(f"Generated summaries for {len(model_evaluations)} models, {len(language_evaluations)} languages")
        logger.info(f"Difficulty breakdown: {dict((k, v['num_problems']) for k, v in self.summary_data['by_difficulty'].items())}")
        logger.info(f"Pass@1 breakdown generated for {len(model_language_evaluations)} model-language combinations")

    def format_metrics_table(self, metrics: Dict[str, Any], title: str) -> str:
        """Format metrics as a table."""
        lines = []
        lines.append(f"\n{title}")
        lines.append("=" * len(title))
        
        # Basic statistics
        lines.append(f"Number of Problems: {metrics.get('num_problems', 0)}")
        lines.append(f"Total Solutions: {metrics.get('num_solutions', 0)}")
        lines.append(f"Solutions Passed: {metrics.get('solutions_passed', 0)}")
        lines.append(f"Total Tests: {metrics.get('total_tests', 0)}")
        lines.append(f"Tests Passed: {metrics.get('passed_tests', 0)}")
        
        # Pass rates
        lines.append(f"\nPass Rates:")
        lines.append(f"  Solution Pass Rate: {metrics.get('overall_solution_pass_rate', 0.0):.4f} ({metrics.get('overall_solution_pass_rate', 0.0)*100:.2f}%)")
        lines.append(f"  Test Pass Rate: {metrics.get('overall_test_pass_rate', 0.0):.4f} ({metrics.get('overall_test_pass_rate', 0.0)*100:.2f}%)")
        
        # Pass@k metrics
        lines.append(f"\nPass@k Metrics:")
        for k in [1, 5, 10]:
            pass_at_k = metrics.get(f'pass_at_{k}', 0.0)
            lines.append(f"  Pass@{k}: {pass_at_k:.4f} ({pass_at_k*100:.2f}%)")
        
        # Error analysis
        lines.append(f"\nError Analysis:")
        lines.append(f"  Compilation Errors: {metrics.get('compilation_errors', 0)} ({metrics.get('compilation_error_rate', 0.0)*100:.2f}%)")
        lines.append(f"  Runtime Errors: {metrics.get('runtime_errors', 0)} ({metrics.get('runtime_error_rate', 0.0)*100:.2f}%)")
        lines.append(f"  Timeout Errors: {metrics.get('timeout_errors', 0)} ({metrics.get('timeout_error_rate', 0.0)*100:.2f}%)")
        lines.append(f"  Total Errors: {metrics.get('total_errors', 0)} ({metrics.get('overall_error_rate', 0.0)*100:.2f}%)")
        
        # Execution time
        if 'total_execution_time' in metrics:
            lines.append(f"\nExecution Time:")
            lines.append(f"  Total: {metrics.get('total_execution_time', 0.0):.2f}s")
            lines.append(f"  Average: {metrics.get('avg_execution_time', 0.0):.2f}s")
            lines.append(f"  Min: {metrics.get('min_execution_time', 0.0):.2f}s")
            lines.append(f"  Max: {metrics.get('max_execution_time', 0.0):.2f}s")
        
        return "\n".join(lines)

    def generate_summary_text(self) -> str:
        """Generate the complete summary text."""
        lines = []
        
        # Header
        lines.append("EVALUATION SUMMARY REPORT")
        lines.append("=" * 50)
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Input File: {self.data_config.input_db}")
        lines.append("")
        
        # Overall statistics
        if self.include_overall_stats:
            lines.append(self.format_metrics_table(self.summary_data['overall'], "OVERALL STATISTICS"))
        
        # Model-wise statistics
        if self.include_model_stats:
            lines.append("\n" + "=" * 80)
            lines.append("MODEL-WISE PERFORMANCE")
            lines.append("=" * 80)
            
            # Sort models by Pass@1 score
            sorted_models = sorted(
                self.summary_data['by_model'].items(),
                key=lambda x: x[1].get('pass_at_1', 0.0),
                reverse=True
            )
            
            for model_name, metrics in sorted_models:
                lines.append(self.format_metrics_table(metrics, f"MODEL: {model_name.upper()}"))
        
        # Language-wise statistics
        if self.include_language_stats:
            lines.append("\n" + "=" * 80)
            lines.append("LANGUAGE-WISE PERFORMANCE")
            lines.append("=" * 80)
            
            # Sort languages by Pass@1 score
            sorted_languages = sorted(
                self.summary_data['by_language'].items(),
                key=lambda x: x[1].get('pass_at_1', 0.0),
                reverse=True
            )
            
            for language, metrics in sorted_languages:
                lines.append(self.format_metrics_table(metrics, f"LANGUAGE: {language.upper()}"))
        
        # Model-Language combination statistics
        if self.include_combinations:
            lines.append("\n" + "=" * 80)
            lines.append("MODEL-LANGUAGE COMBINATION PERFORMANCE")
            lines.append("=" * 80)
            
            # Sort combinations by Pass@1 score
            sorted_combinations = sorted(
                self.summary_data['model_language_combinations'].items(),
                key=lambda x: x[1].get('pass_at_1', 0.0),
                reverse=True
            )
            
            for combo, metrics in sorted_combinations:
                lines.append(self.format_metrics_table(metrics, f"COMBINATION: {combo.upper()}"))
        
        # Pass@1 Breakdown Matrix
        if hasattr(self, 'summary_data') and 'pass_at_1_breakdown' in self.summary_data:
            lines.append("\n" + "=" * 80)
            lines.append("PASS@1 BREAKDOWN BY MODEL-LANGUAGE")
            lines.append("=" * 80)
            lines.append(self._format_pass_at_1_matrix())
        
        # Leaderboard
        if self.include_leaderboard:
            lines.append("\n" + "=" * 80)
            lines.append(f"LEADERBOARD (Top {self.leaderboard_top_n} by {self.sort_by.replace('_', '@')})")
            lines.append("=" * 80)
            
            # Create leaderboard entries
            leaderboard_entries = []
            for combo, metrics in self.summary_data['model_language_combinations'].items():
                leaderboard_entries.append({
                    'combination': combo,
                    'pass_at_1': metrics.get('pass_at_1', 0.0),
                    'pass_at_5': metrics.get('pass_at_5', 0.0),
                    'pass_at_10': metrics.get('pass_at_10', 0.0),
                    'num_problems': metrics.get('num_problems', 0),
                    'solution_pass_rate': metrics.get('overall_solution_pass_rate', 0.0)
                })
            
            # Sort by specified metric
            leaderboard_entries.sort(key=lambda x: x.get(self.sort_by, 0.0), reverse=True)
            
            # Format leaderboard
            lines.append(f"{'Rank':<4} {'Model-Language':<30} {'Pass@1':<8} {'Pass@5':<8} {'Pass@10':<8} {'Problems':<10} {'Sol. Pass Rate':<15}")
            lines.append("-" * 95)
            
            for i, entry in enumerate(leaderboard_entries[:self.leaderboard_top_n], 1):
                lines.append(
                    f"{i:<4} {entry['combination']:<30} "
                    f"{entry['pass_at_1']*100:<7.2f}% "
                    f"{entry['pass_at_5']*100:<7.2f}% "
                    f"{entry['pass_at_10']*100:<7.2f}% "
                    f"{entry['num_problems']:<10} "
                    f"{entry['solution_pass_rate']*100:<14.2f}%"
                )
        
        return "\n".join(lines)

    def generate_summary_json(self) -> Dict[str, Any]:
        """Generate the complete summary as JSON."""
        summary_json = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "input_file": self.data_config.input_db,
                "configuration": {
                    "include_overall_stats": self.include_overall_stats,
                    "include_model_stats": self.include_model_stats,
                    "include_language_stats": self.include_language_stats,
                    "include_combinations": self.include_combinations,
                    "include_leaderboard": self.include_leaderboard,
                    "leaderboard_top_n": self.leaderboard_top_n,
                    "sort_by": self.sort_by
                }
            }
        }
        
        # Overall statistics
        if self.include_overall_stats:
            summary_json["overall_statistics"] = self.summary_data['overall']
        
        # Model-wise statistics
        if self.include_model_stats:
            summary_json["model_statistics"] = {
                model: metrics for model, metrics in self.summary_data['by_model'].items()
            }
        
        # Language-wise statistics
        if self.include_language_stats:
            summary_json["language_statistics"] = {
                language: metrics for language, metrics in self.summary_data['by_language'].items()
            }
        
        # Model-Language combination statistics
        if self.include_combinations:
            summary_json["model_language_combinations"] = {
                combo: metrics for combo, metrics in self.summary_data['model_language_combinations'].items()
            }
        
        # Difficulty breakdown
        if hasattr(self, 'summary_data') and 'by_difficulty' in self.summary_data:
            summary_json["difficulty_statistics"] = {
                difficulty: metrics for difficulty, metrics in self.summary_data['by_difficulty'].items()
            }
        
        # Pass@1 breakdown by model-language combinations
        if hasattr(self, 'summary_data') and 'pass_at_1_breakdown' in self.summary_data:
            summary_json["pass_at_1_breakdown"] = self.summary_data['pass_at_1_breakdown']
        
        # Leaderboard
        if self.include_leaderboard:
            leaderboard_entries = []
            for combo, metrics in self.summary_data['model_language_combinations'].items():
                leaderboard_entries.append({
                    'combination': combo,
                    'pass_at_1': metrics.get('pass_at_1', 0.0),
                    'pass_at_5': metrics.get('pass_at_5', 0.0),
                    'pass_at_10': metrics.get('pass_at_10', 0.0),
                    'num_problems': metrics.get('num_problems', 0),
                    'solution_pass_rate': metrics.get('overall_solution_pass_rate', 0.0),
                    'compilation_error_rate': metrics.get('compilation_error_rate', 0.0),
                    'runtime_error_rate': metrics.get('runtime_error_rate', 0.0),
                    'timeout_error_rate': metrics.get('timeout_error_rate', 0.0)
                })
            
            # Sort by specified metric
            leaderboard_entries.sort(key=lambda x: x.get(self.sort_by, 0.0), reverse=True)
            summary_json["leaderboard"] = leaderboard_entries[:self.leaderboard_top_n]
        
        return summary_json

    def run(self):
        """Main execution method."""
        logger.info("Starting LiveCodeBench evaluation summary generation...")
        
        try:
            # Generate statistics
            self.generate_summary_statistics()
            
            # Generate summary JSON
            summary_json = self.generate_summary_json()
            
            # Save summary
            logger.info(f"Saving summary to {self.data_config.output_summary}")
            with open(self.data_config.output_summary, 'w', encoding='utf-8') as f:
                json.dump(summary_json, f, indent=2, ensure_ascii=False)
            
            logger.info("LiveCodeBench evaluation summary generation completed!")
            logger.info(f"Summary saved to: {self.data_config.output_summary}")
            
        except Exception as e:
            logger.error(f"Error during summary generation: {e}")
            raise
