"""
Evaluation metrics for code evaluation, compatible with LiveCodeBench methodology.

This module implements the standard Pass@k metric and other evaluation metrics
used in code evaluation benchmarks like LiveCodeBench, HumanEval, and MBPP.
"""

import math
from typing import List, Dict, Any, Union
from collections import defaultdict
import numpy as np


def pass_at_k(n: int, c: int, k: int) -> float:
    """
    Calculate the unbiased estimator for Pass@k metric.
    
    This is the standard Pass@k calculation used in LiveCodeBench, HumanEval, and other
    code evaluation benchmarks. It computes the probability that at least one of k 
    randomly sampled solutions passes all test cases.
    
    Args:
        n (int): Total number of generated solutions
        c (int): Number of correct solutions (that pass all tests)
        k (int): Number of solutions to sample
        
    Returns:
        float: Pass@k score (probability between 0 and 1)
        
    Formula:
        Pass@k = 1 - C(n-c, k) / C(n, k)
        
    Where C(n, k) is the binomial coefficient "n choose k".
    
    This formula computes the probability that at least one solution in a random
    sample of k solutions is correct, given that c out of n total solutions are correct.
    """
    if n - c < k:
        return 1.0
    
    if c == 0:
        return 0.0
        
    if k == 0:
        return 0.0
    
    # Calculate binomial coefficients
    # C(n-c, k) / C(n, k) = (n-c)! * k! * (n-k)! / (k! * (n-c-k)! * n!)
    # = (n-c)! * (n-k)! / ((n-c-k)! * n!)
    
    try:
        # Use math.comb for exact calculation (Python 3.8+)
        if hasattr(math, 'comb'):
            prob_all_wrong = math.comb(n - c, k) / math.comb(n, k)
        else:
            # Fallback for older Python versions
            prob_all_wrong = (
                math.factorial(n - c) * math.factorial(n - k) /
                (math.factorial(n - c - k) * math.factorial(n))
            )
        return 1.0 - prob_all_wrong
    except (ValueError, OverflowError):
        # Handle edge cases and overflow
        if n - c < k:
            return 1.0
        return 0.0


def calculate_pass_at_k_for_problems(results: List[Dict[str, Any]], k_values: List[int] = None) -> Dict[str, float]:
    """
    Calculate Pass@k metrics across multiple problems.
    
    Args:
        results: List of problem results, each containing:
            - 'total_solutions': Total number of generated solutions
            - 'solutions_passed': Number of solutions that passed all tests
        k_values: List of k values to calculate (default: [1, 5, 10])
        
    Returns:
        Dict mapping "pass_at_{k}" to the average Pass@k score across all problems
    """
    if k_values is None:
        k_values = [1, 5, 10]
    
    pass_at_k_scores = defaultdict(list)
    
    for result in results:
        n = result.get('total_solutions', 0)
        c = result.get('solutions_passed', 0)
        
        if n == 0:
            continue
            
        for k in k_values:
            if k <= n:  # Only calculate if we have enough solutions
                score = pass_at_k(n, c, k)
                pass_at_k_scores[f'pass_at_{k}'].append(score)
    
    # Calculate average scores
    avg_scores = {}
    for k_name, scores in pass_at_k_scores.items():
        if scores:
            avg_scores[k_name] = np.mean(scores)
        else:
            avg_scores[k_name] = 0.0
    
    return avg_scores


def calculate_test_pass_rate(evaluation_results: List[Dict[str, Any]]) -> float:
    """
    Calculate the overall test pass rate across all evaluations.
    
    Args:
        evaluation_results: List of evaluation results, each containing:
            - 'passed_tests': Number of tests passed
            - 'total_tests': Total number of tests
            
    Returns:
        float: Overall test pass rate (0.0 to 1.0)
    """
    total_tests = 0
    passed_tests = 0
    
    for result in evaluation_results:
        total_tests += result.get('total_tests', 0)
        passed_tests += result.get('passed_tests', 0)
    
    if total_tests == 0:
        return 0.0
    
    return passed_tests / total_tests


def calculate_error_distribution(evaluation_results: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    Calculate the distribution of different error types.
    
    Args:
        evaluation_results: List of evaluation results, each containing:
            - 'overall_result': The result type (e.g., 'passed', 'failed', 'timeout', etc.)
            
    Returns:
        Dict mapping error types to their counts
    """
    error_counts = defaultdict(int)
    
    for result in evaluation_results:
        result_type = result.get('overall_result', 'unknown')
        error_counts[result_type] += 1
    
    return dict(error_counts)


def calculate_execution_time_stats(evaluation_results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Calculate execution time statistics.
    
    Args:
        evaluation_results: List of evaluation results, each containing:
            - 'execution_time': Total execution time for the solution
            
    Returns:
        Dict with execution time statistics (mean, median, std, min, max)
    """
    execution_times = [
        result.get('execution_time', 0.0) 
        for result in evaluation_results 
        if result.get('execution_time', 0.0) > 0
    ]
    
    if not execution_times:
        return {
            'mean': 0.0,
            'median': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0
        }
    
    return {
        'mean': np.mean(execution_times),
        'median': np.median(execution_times),
        'std': np.std(execution_times),
        'min': np.min(execution_times),
        'max': np.max(execution_times)
    }


def calculate_difficulty_breakdown(results: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """
    Calculate Pass@1 breakdown by problem difficulty.
    
    Args:
        results: Dictionary with problem IDs as keys and problem data as values
        
    Returns:
        Dict mapping difficulty levels to their Pass@1 scores
    """
    difficulty_results = defaultdict(list)
    
    for problem_id, problem_data in results.items():
        difficulty = problem_data.get('lcb_difficulty', 'unknown')
        
        # Find evaluation results
        for key, value in problem_data.items():
            if key.endswith('_evaluation'):
                pass_at_1 = value.get('pass_at_1', 0.0)
                difficulty_results[difficulty].append(pass_at_1)
    
    # Calculate average Pass@1 for each difficulty
    difficulty_scores = {}
    for difficulty, scores in difficulty_results.items():
        if scores:
            difficulty_scores[difficulty] = {
                'pass_at_1': np.mean(scores),
                'count': len(scores)
            }
    
    return difficulty_scores


def _get_best_round_results(rounds: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge results from multiple rounds, taking the best performance for each solution.
    
    This creates a "virtual best round" that combines:
    - The best pass rate and solutions_passed across all rounds
    - Merged evaluations with best test results for each solution
    - Combined error statistics and execution times
    
    Args:
        rounds: Dictionary of round_id -> round_data
        
    Returns:
        Dict containing merged best results across all rounds
    """
    if not rounds:
        return {}

    # Filter out any corrupted entries (e.g. bool written during a crash)
    valid_rounds = {k: v for k, v in rounds.items() if isinstance(v, dict)}
    if not valid_rounds:
        return {}

    # Start with the latest round as base
    round_ids = sorted(valid_rounds.keys())
    latest_round = valid_rounds[round_ids[-1]]
    
    # Initialize merged result with latest round data
    merged_result = latest_round.copy()
    
    # Find the best pass_at_1 and solutions_passed across all rounds
    best_pass_at_1 = 0.0
    best_solutions_passed = 0
    best_round_id = round_ids[0]
    
    for round_id, round_data in valid_rounds.items():
        current_pass_at_1 = round_data.get('pass_at_1', 0.0)
        current_solutions_passed = round_data.get('solutions_passed', 0)
        
        if current_pass_at_1 > best_pass_at_1 or (current_pass_at_1 == best_pass_at_1 and current_solutions_passed > best_solutions_passed):
            best_pass_at_1 = current_pass_at_1
            best_solutions_passed = current_solutions_passed
            best_round_id = round_id
    
    # Use the best performing round's metrics
    best_round = valid_rounds[best_round_id]
    merged_result.update({
        'pass_at_1': best_pass_at_1,
        'solutions_passed': best_solutions_passed,
        'num_solutions': best_round.get('num_solutions', merged_result.get('num_solutions', 0)),
        'evaluations': best_round.get('evaluations', []),
        'round_id': f"best_from_{best_round_id}",
        'round_description': f"Best results merged from {len(valid_rounds)} rounds (best from {best_round_id})",
        'merged_from_rounds': list(valid_rounds.keys()),
        'best_performance_round': best_round_id
    })
    
    return merged_result


def calculate_comprehensive_metrics(
    results: Dict[str, Any], 
    k_values: List[int] = None
) -> Dict[str, Any]:
    """
    Calculate comprehensive evaluation metrics for a model evaluation run.
    
    Args:
        results: Complete evaluation results dictionary
        k_values: List of k values for Pass@k calculation
        
    Returns:
        Dict containing all calculated metrics
    """
    if k_values is None:
        k_values = [1, 5, 10]
    
    # Collect all evaluation data
    all_evaluations = []
    problem_results = []
    
    for problem_id, problem_data in results.items():
        for key, value in problem_data.items():
            if key.endswith('_evaluation'):
                # Handle multi-round evaluation structure
                if 'rounds' in value:
                    # Multi-round format - merge best results from all rounds
                    best_round_data = _get_best_round_results(value['rounds'])
                    round_data = best_round_data
                else:
                    # Legacy format - use the data directly
                    round_data = value
                
                # Extract problem-level results from the best round
                problem_results.append({
                    'problem_id': problem_id,
                    'total_solutions': round_data.get('num_solutions', 0),
                    'solutions_passed': round_data.get('solutions_passed', 0),
                    'pass_at_1': round_data.get('pass_at_1', 0.0)
                })
                
                # Extract solution-level evaluations from the best round
                evaluations = round_data.get('evaluations', [])
                all_evaluations.extend(evaluations)
    
    # Calculate comprehensive metrics
    metrics = {}
    
    # Pass@k metrics
    pass_at_k_metrics = calculate_pass_at_k_for_problems(problem_results, k_values)
    metrics.update(pass_at_k_metrics)
    
    # Test pass rate
    metrics['test_pass_rate'] = calculate_test_pass_rate(all_evaluations)
    
    # Error distribution
    metrics['error_distribution'] = calculate_error_distribution(all_evaluations)
    
    # Execution time statistics
    metrics['execution_time_stats'] = calculate_execution_time_stats(all_evaluations)
    
    # Difficulty breakdown
    metrics['difficulty_breakdown'] = calculate_difficulty_breakdown(results)
    
    # Summary statistics
    metrics['total_problems'] = len(problem_results)
    metrics['total_solutions'] = sum(r['total_solutions'] for r in problem_results)
    metrics['total_solutions_passed'] = sum(r['solutions_passed'] for r in problem_results)
    
    if metrics['total_solutions'] > 0:
        metrics['overall_success_rate'] = metrics['total_solutions_passed'] / metrics['total_solutions']
    else:
        metrics['overall_success_rate'] = 0.0
    
    return metrics


# Compatibility aliases for LiveCodeBench
def compute_pass_at_k(n: int, c: int, k: int) -> float:
    """Alias for pass_at_k for LiveCodeBench compatibility."""
    return pass_at_k(n, c, k)


def estimate_pass_at_k(num_samples: int, num_correct: int, k: int) -> float:
    """Alternative name for pass_at_k calculation."""
    return pass_at_k(num_samples, num_correct, k) 


# Retrieval Metrics Functions

def calculate_retrieval_metrics(retrieved_solutions: List[str], target_solutions: List[str], k_values: List[int] = [1, 5, 10, 20]) -> Dict[str, float]:
    """
    Calculate retrieval metrics for a single query.
    
    Args:
        retrieved_solutions: List of retrieved solution IDs in order of relevance
        target_solutions: List of target solution IDs (ground truth)
        k_values: List of k values for calculating metrics at different ranks
    
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Convert to sets for faster lookup
    target_set = set(target_solutions)
    
    # Calculate metrics for each k
    for k in k_values:
        # Get top-k retrieved solutions
        top_k_retrieved = retrieved_solutions[:k]
        top_k_set = set(top_k_retrieved)
        
        # Calculate basic metrics
        relevant_retrieved = len(top_k_set.intersection(target_set))
        precision = relevant_retrieved / k if k > 0 else 0.0
        recall = relevant_retrieved / len(target_set) if len(target_set) > 0 else 0.0
        
        # Calculate MRR (Mean Reciprocal Rank)
        mrr = 0.0
        for i, solution in enumerate(top_k_retrieved):
            if solution in target_set:
                mrr = 1.0 / (i + 1)
                break
        
        # Calculate NDCG (Normalized Discounted Cumulative Gain)
        ndcg = calculate_ndcg(top_k_retrieved, target_solutions, k)
        
        # Store metrics
        metrics[f'mrr_at_{k}'] = mrr
        metrics[f'recall_at_{k}'] = recall
        metrics[f'precision_at_{k}'] = precision
        metrics[f'ndcg_at_{k}'] = ndcg
    
    return metrics


def calculate_ndcg(retrieved_solutions: List[str], target_solutions: List[str], k: int) -> float:
    """Calculate Normalized Discounted Cumulative Gain."""
    target_set = set(target_solutions)
    
    # Calculate DCG
    dcg = 0.0
    for i, solution in enumerate(retrieved_solutions[:k]):
        if solution in target_set:
            dcg += 1.0 / math.log2(i + 2)  # i+2 because log2(1) = 0
    
    # Calculate IDCG (Ideal DCG)
    idcg = 0.0
    for i in range(min(k, len(target_solutions))):
        idcg += 1.0 / math.log2(i + 2)
    
    # Calculate NDCG
    ndcg = dcg / idcg if idcg > 0 else 0.0
    return ndcg


def calculate_cross_language_metrics(retrieval_results: List[Dict], code_samples: List[Dict]) -> Dict[str, Any]:
    """Calculate cross-language specific metrics."""
    cross_lang_metrics = {}
    
    # Group results by query type
    query_type_results = defaultdict(list)
    for result in retrieval_results:
        query_type = result.get('metadata', {}).get('query_type', 'unknown')
        query_type_results[query_type].append(result)
    
    # Calculate cross-language specific metrics
    for query_type, results in query_type_results.items():
        if query_type.startswith('cross_lang_'):
            # Extract target language
            target_lang = query_type.split('_')[-1]
            
            # Calculate language-specific metrics
            lang_metrics = []
            for result in results:
                # Get retrieved solutions
                retrieved = result.get('retrieved_solutions', [])
                
                # Count solutions in target language
                lang_count = 0
                for sol_id in retrieved[:10]:  # Top 10
                    # Find code sample
                    for sample in code_samples:
                        if sample.get('sample_id') == sol_id and sample.get('language') == target_lang:
                            lang_count += 1
                            break
                
                lang_metrics.append(lang_count / 10.0)  # Normalize by top-10
            
            cross_lang_metrics[f'{query_type}_accuracy'] = np.mean(lang_metrics) if lang_metrics else 0.0
    
    return cross_lang_metrics