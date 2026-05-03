"""
Code Evaluation Runner - LiveCodeBench Pipeline Implementation

This module implements a comprehensive code evaluation runner that mimics the LiveCodeBench 
evaluation pipeline for executing generated code solutions and running tests to evaluate 
their correctness.

Key Features:
- Executes code solutions in multiple programming languages (Python, Java, C++, C, JavaScript, Go, Ruby)
- Runs both public and private test cases
- Handles compilation errors, runtime errors, and timeouts
- Computes detailed evaluation metrics (pass rates, execution times, etc.)
- Supports multiple solutions per problem
- Provides comprehensive logging and error handling
- Properly handles functional vs stdin test types (LiveCodeBench compatible)
- Multi-round evaluation support for incremental evaluation with different timeout settings
- Preserves evaluation history across multiple rounds
- Supports targeted re-evaluation of timeout or failed cases

Usage Example:

1. Configuration file (config.yaml):
```yaml
runner_name: "code_eval"
data:
  output_db: "./output/evaluation_results.json"
  data_dir: "./data/problems.json"
  test_cases_dir: "./data/test_cases.json" # Optional: separate test cases file
  # test_cases_dir: null # Set to null to use test cases from data_dir
code_eval:
  languages: ["python", "java", "cpp", "javascript", "go", "ruby"]
  timeout: 30
  memory_limit: 1024
  use_private_tests: true
  overwrite: false
  models_to_evaluate: ["gpt-4", "claude-3"]
  evaluation_round:
    round_id: "round_1"
    round_description: "Initial evaluation with 30s timeout"
    timeout_extension: false
    target_cases: "all"  # Options: "all", "timeout", "failed", "specific_ids"
    specific_case_ids: []  # Used when target_cases is "specific_ids"
```

2. Running the evaluation:
```python
from omegaconf import OmegaConf
from coreb_runner.runners import Runner

config = OmegaConf.load("config.yaml")
runner = Runner.build_from_config(config)
runner.run()
```

3. Expected data format:
The runner expects generated solutions to be stored in the database with keys like:
- "{model_name}_{language}": List of generated code solutions
- "annotate_public_test_cases": List of public test cases
- "annotate_private_test_cases": List of private test cases (optional)

4. Output format:
The runner stores evaluation results with keys like:
- "{model_name}_{language}_evaluation": Multi-round evaluation results including:
  - model_name: Name of the model
  - language: Programming language
  - latest_round: ID of the most recent evaluation round
  - total_rounds: Total number of evaluation rounds
  - best_performance: Best performance metrics across all rounds
  - evaluation_rounds: Dictionary of round-specific results:
    - "{round_id}": Round-specific evaluation data including:
      - round_id: Unique identifier for this round
      - round_description: Description of this evaluation round
      - timeout_extension: Whether this round extended timeout from previous
      - timestamp: When this round was executed
      - timeout: Timeout setting used in this round
      - pass_at_1, pass_at_5, etc.: Pass@k metrics for this round
  - evaluations: Per-solution results with test-by-test details

Prerequisites:
- Python 3.7+
- Required compilers/interpreters for target languages:
  - Python: python3
  - Java: javac, java
  - C++: g++
  - C: gcc
  - JavaScript: node
  - Go: go
  - Ruby: ruby
"""

import subprocess
import tempfile
import os
import time
import multiprocessing as mp
from multiprocessing import Pool, Manager
import json
import ast
import sys
import signal
import queue
import warnings
import gc  # Added for garbage collection
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

# Suppress torchvision warnings that clutter multiprocessing output
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
warnings.filterwarnings("ignore", message=".*torchvision.*")
warnings.filterwarnings("ignore", message=".*Failed to load image Python extension.*")
warnings.filterwarnings("ignore", message=".*Symbol not found.*")
warnings.filterwarnings("ignore", message=".*Redirects are currently not supported.*")
warnings.filterwarnings("ignore", message=".*torchvision.datapoints.*")
warnings.filterwarnings("ignore", message=".*torchvision.transforms.v2.*")

from easyllm_kit.utils.io_utils import initialize_database, write_to_database
from easyllm_kit.utils import get_logger, read_json
from coreb_runner.runners.base_runner import Runner
from coreb_runner.utils.code_utils import extract_code
from coreb_runner.utils.metrics import pass_at_k, calculate_comprehensive_metrics

logger = get_logger('code_eval_runner', 'code_eval_runner.log')


class ExecutionResult(Enum):
    """Enum for execution results"""
    PASSED = "passed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    RUNTIME_ERROR = "runtime_error"
    COMPILATION_ERROR = "compilation_error"
    MEMORY_ERROR = "memory_error"


class TestType(Enum):
    """Enum for test types"""
    STDIN = "stdin"
    FUNCTIONAL = "functional"


@dataclass
class TestResult:
    """Result of a single test case execution"""
    test_index: int
    result: ExecutionResult
    expected_output: str
    actual_output: str
    error_message: str
    execution_time: float


@dataclass
class EvaluationResult:
    """Result of evaluating a single solution"""
    lcb_id: str
    language: str
    model_name: str
    solution_index: int
    passed_tests: int
    total_tests: int
    pass_rate: float
    test_results: List[TestResult]
    overall_result: ExecutionResult
    total_execution_time: float


@Runner.register("code_eval")
class CodeEvalRunner(Runner):
    """
    Runner for evaluating generated code solutions by executing them against test cases.
    
    This runner mimics the LiveCodeBench evaluation pipeline:
    1. Extracts code from generated solutions
    2. Executes code against test cases (both public and private)
    3. Computes pass rates and detailed evaluation metrics
    4. Supports multiple programming languages
    5. Handles timeouts, compilation errors, and runtime errors
    6. Properly handles functional vs stdin test types
    
    Configuration options:
    - data: Configuration for input/output data
    - code_eval: Additional configuration for code evaluation
      - languages: List of programming languages to evaluate (default: ['python'])
      - timeout: Maximum execution time per test case in seconds (default: 5)
      - memory_limit: Maximum memory usage in MB (default: 512)
      - use_private_tests: Whether to use private test cases (default: True)
      - overwrite: Whether to overwrite existing evaluation results (default: False)
      - models_to_evaluate: List of model names to evaluate (if None, evaluates all available)
      - k_values: Pass@k values to calculate (default: [1, 5, 10])
      - abs_tol: Absolute tolerance for floating-point comparison (default: 1e-6)
    
    Floating-Point Comparison:
    The abs_tol parameter controls how floating-point numbers in outputs are compared.
    Two numbers are considered equal if their absolute difference is within this tolerance.
    
    Examples:
    - Strict: abs_tol: 1e-9 (very precise, for exact calculations)
    - Standard: abs_tol: 1e-6 (default, good for most competitive programming problems)
    - Relaxed: abs_tol: 1e-3 (tolerant, for approximate solutions)
    
    Example configuration:
    {
        "data": { ... },
        "code_eval": {
            "languages": ["python", "java", "cpp"],
            "timeout": 10,
            "memory_limit": 1024,
            "use_private_tests": true,
            "overwrite": false,
            "models_to_evaluate": ["gpt-4", "claude-3"],
            "abs_tol": 1e-6
        }
    }
    """

    def __init__(self, config):
        super().__init__()
        
        # Some macOS sessions export malloc debug env vars (e.g. MallocStackLogging=0),
        # which causes every spawned child python process to print noisy warnings:
        # "can't turn off malloc stack logging because it was not enabled."
        # Remove these once at runner startup so subprocesses inherit a clean env.
        self._sanitize_macos_malloc_env()
        
        # Parse configuration
        self.data_config = config.data
        self.eval_config = config.code_eval
        
        # Core evaluation parameters
        self.languages = self.eval_config.get('languages', ['python'])
        self.timeout = self.eval_config.get('timeout', 30)
        self.memory_limit = self.eval_config.get('memory_limit', 1024)
        self.use_private_tests = self.eval_config.get('use_private_tests', True)
        self.overwrite = self.eval_config.get('overwrite', False)
        self.models_to_evaluate = self.eval_config.get('models_to_evaluate', None)
        self.k_values = self.eval_config.get('k_values', [1, 5, 10])
        self.abs_tolerance = self.eval_config.get('abs_tol', 1e-6)
        self.num_processes = self.eval_config.get('num_processes', 1)  # Default to 1 to avoid memory issues
        self.auto_add_imports = self.eval_config.get('auto_add_imports', True)
        # One problem at a time minimizes peak RAM (solutions + tests + subprocess I/O).
        self.batch_size = self.eval_config.get('batch_size', 1)
        self.enable_debug_logging = self.eval_config.get('enable_debug_logging', False)
        # Hard cap on combined stdout+stderr from any single subprocess call.
        # Prevents runaway solutions (e.g. infinite print loops) from filling RAM.
        self.max_subprocess_output_chars = int(
            self.eval_config.get('max_subprocess_output_chars', 131072)  # 128 KiB
        )
        # TestResult objects are kept in memory until JSON is written; trim huge I/O strings.
        self.max_stored_io_chars = int(self.eval_config.get('max_stored_io_chars', 4096))
        # After each problem, drop its row from the in-memory test-case map (huge for annotate JSON).
        self.drop_separate_test_cases_after_problem = self.eval_config.get(
            'drop_separate_test_cases_after_problem', True
        )
        # Drop generated solution blobs from the in-memory item once that problem is done.
        self.strip_solution_strings_after_problem = self.eval_config.get(
            'strip_solution_strings_after_problem', True
        )
        # Problems to skip entirely (mark all language evaluations as FAILED).
        # Useful for known-crashing problems. Accepts bare IDs or lcb_* prefixed IDs.
        raw_skip = self.eval_config.get('skip_problems', [])
        self.skip_problems: set = set()
        for p in raw_skip:
            self.skip_problems.add(p)
            if p.startswith('lcb_'):
                self.skip_problems.add(p[4:])
            else:
                self.skip_problems.add('lcb_' + p)
        if self.skip_problems:
            logger.info(f"  Skip list: {sorted(self.skip_problems)}")
        # Max wall-clock seconds allowed to evaluate ONE language for ONE problem.
        # With 42 tests × 60s timeout each the worst case is 42 min; this cap cuts that short.
        # Default: 5 × per-test timeout (e.g. 300s for timeout=60).
        self.per_language_timeout = int(
            self.eval_config.get('per_language_timeout', self.timeout * 5)
        )
        
        # Test case field names - set prefix based on config
        # Options: 'lcb' for original LCB format, 'annotate' for annotated format
        test_case_prefix = self.eval_config.get('test_case_field_prefix', 'lcb')
        if test_case_prefix == 'annotate':
            self.public_test_field = 'annotate_public_test_cases'
            self.private_test_field = 'annotate_private_test_cases'
        else:  # default to 'lcb'
            self.public_test_field = 'lcb_public_test_cases'
            self.private_test_field = 'lcb_private_test_cases'
        logger.info(f"  Test case fields: {self.public_test_field}, {self.private_test_field}")
        
        # Round configuration (simplified)
        round_config = self.eval_config.get('evaluation_round', {})
        self.round_id = round_config.get('round_id', 'default_round')
        self.round_description = round_config.get('round_description', 'Code evaluation round')
        self.target_cases = round_config.get('target_cases', 'all')  # 'all', 'timeout', 'failed'
        
        logger.info(f"Code evaluation configured:")
        logger.info(f"  Languages: {self.languages}")
        logger.info(f"  Timeout: {self.timeout}s, Memory: {self.memory_limit}MB")
        logger.info(f"  Round: {self.round_id} - {self.round_description}")
        logger.info(f"  Target cases: {self.target_cases}")
        logger.info(f"  Overwrite: {self.overwrite}")
        logger.info(f"  Multiprocessing: {self.num_processes} processes")
        logger.info(f"  Per-language wall-clock cap: {self.per_language_timeout}s")
        logger.info(
            f"  Memory helpers: batch_size={self.batch_size}, "
            f"max_subprocess_output_chars={self.max_subprocess_output_chars}, "
            f"max_stored_io_chars={self.max_stored_io_chars}, "
            f"drop_test_cases_after_problem={self.drop_separate_test_cases_after_problem}, "
            f"strip_solutions_after_problem={self.strip_solution_strings_after_problem}"
        )

        # Initialize database
        self.db = initialize_database(self.data_config.output_db)

        # Read data
        self.codebench_data = read_json(self.data_config.data_dir)
        
        # Test-cases file.
        # The annotation JSON can be 500 MB+ on disk with test inputs that are
        # themselves hundreds of MB. Loading it all into RAM causes OOM.
        # Instead we split it into per-problem files once and load only the
        # current problem's file on demand (~1-4 MB at a time).
        self.test_cases_data = {}          # unused when split dir exists
        self.test_cases_file_path = None
        self.test_cases_split_dir = None
        if hasattr(self.data_config, 'test_cases_dir') and self.data_config.test_cases_dir:
            self.test_cases_file_path = self.data_config.test_cases_dir
            self.test_cases_split_dir = self._build_test_cases_split(self.test_cases_file_path)

        # Language execution commands
        self.execution_commands = {
            "python": self._execute_python,
            "java": self._execute_java,
            "cpp": self._execute_cpp,
            "c": self._execute_c,
            "javascript": self._execute_javascript,
            "go": self._execute_go,
            "rust": self._execute_rust,
            "ruby": self._execute_ruby,
        }

    def _sanitize_macos_malloc_env(self):
        """Remove macOS malloc-debug env vars that spam subprocess stderr."""
        if sys.platform != "darwin":
            return
        for key in (
            "MallocStackLogging",
            "MallocStackLoggingNoCompact",
            "MallocScribble",
            "MallocGuardEdges",
            "MallocCheckHeapStart",
            "MallocCheckHeapEach",
        ):
            os.environ.pop(key, None)

    def _truncate_stored_io(self, s: str) -> str:
        if not s:
            return s
        lim = self.max_stored_io_chars
        if len(s) <= lim:
            return s
        return s[:lim] + "\n...[truncated]"

    def _compact_io_for_test_result(
        self,
        final_result: ExecutionResult,
        expected_output: str,
        actual_output: str,
        error_message: str,
    ) -> Tuple[str, str, str]:
        """Shrink strings retained on TestResult (pass/fail already decided)."""
        if final_result == ExecutionResult.PASSED:
            return "", "", self._truncate_stored_io(error_message) if error_message else ""
        return (
            self._truncate_stored_io(expected_output),
            self._truncate_stored_io(actual_output),
            self._truncate_stored_io(error_message),
        )

    def _finalize_problem_memory(self, lcb_id: str, item: Optional[Dict[str, Any]]) -> None:
        """Free per-problem solution blobs from the in-memory item.
        The test_cases_data trimmed dict stays resident (it's small, ~5-20 MB total)."""
        if self.strip_solution_strings_after_problem and item:
            suffixes = (
                "_python",
                "_java",
                "_cpp",
                "_c",
                "_javascript",
                "_go",
                "_rust",
                "_ruby",
            )
            for k in list(item.keys()):
                if any(k.endswith(sfx) for sfx in suffixes):
                    item.pop(k, None)
    
    @staticmethod
    def _subprocess_memory_limit():
        """preexec_fn: cap this child process's virtual address space to 3 GB."""
        try:
            import resource
            limit = 3 * 1024 * 1024 * 1024  # 3 GB
            resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
        except Exception:
            pass

    def _terminate_process_safely(self, process):
        """Terminate a subprocess, escalating to SIGKILL if needed."""
        try:
            process.terminate()
        except (ProcessLookupError, OSError):
            pass
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            try:
                process.kill()
            except (ProcessLookupError, OSError):
                pass

    def _communicate_limited(self, process, input_data=None, timeout=None):
        """Replacement for process.communicate() that kills the child if combined
        stdout+stderr exceeds self.max_subprocess_output_chars, preventing OOM from
        solutions that print gigabytes before the per-test timeout fires."""
        import threading

        max_chars = self.max_subprocess_output_chars
        stdout_buf: list = []
        stderr_buf: list = []
        total_read = [0]
        overflow = [False]

        def _read_limited(stream, buf):
            try:
                for line in stream:
                    # Suppress known macOS malloc-debug noise lines so they don't
                    # count toward max_subprocess_output_chars or appear in logs.
                    if "MallocStackLogging" in line:
                        continue
                    buf.append(line)
                    total_read[0] += len(line)
                    if total_read[0] > max_chars:
                        overflow[0] = True
                        self._terminate_process_safely(process)
                        break
            except Exception:
                pass

        # Write stdin in a thread to avoid deadlock: if the child's stdout pipe fills
        # up before it reads all of stdin, both sides block forever unless reads and
        # writes happen concurrently.
        def _write_stdin():
            if process.stdin:
                try:
                    if input_data:
                        process.stdin.write(input_data)
                        process.stdin.flush()
                    process.stdin.close()
                except (BrokenPipeError, OSError):
                    pass

        t_in  = threading.Thread(target=_write_stdin, daemon=True)
        t_out = threading.Thread(target=_read_limited, args=(process.stdout, stdout_buf), daemon=True)
        t_err = threading.Thread(target=_read_limited, args=(process.stderr, stderr_buf), daemon=True)
        t_in.start()
        t_out.start()
        t_err.start()

        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self._terminate_process_safely(process)
            t_in.join(timeout=2)
            t_out.join(timeout=2)
            t_err.join(timeout=2)
            return "", "Timeout", -1

        t_in.join(timeout=5)
        t_out.join(timeout=5)
        t_err.join(timeout=5)

        stdout = "".join(stdout_buf)
        stderr = "".join(stderr_buf)
        if overflow[0]:
            stderr = f"[Output truncated: exceeded {max_chars} chars]\n" + stderr[:500]
        return stdout.strip(), stderr.strip(), process.returncode

    def _evaluate_test_case_worker(self, args):
        """Worker function for multiprocessing test case evaluation"""
        # Suppress warnings in worker processes
        import warnings
        import gc
        warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
        warnings.filterwarnings("ignore", message=".*torchvision.*")
        warnings.filterwarnings("ignore", message=".*Failed to load image Python extension.*")
        warnings.filterwarnings("ignore", message=".*Symbol not found.*")
        warnings.filterwarnings("ignore", message=".*Redirects are currently not supported.*")
        warnings.filterwarnings("ignore", message=".*torchvision.datapoints.*")
        warnings.filterwarnings("ignore", message=".*torchvision.transforms.v2.*")
        
        test_index = -1
        expected_output = ""
        try:
            code, test_input, expected_output, test_index, language, test_type, timeout = args
            
            start_time = time.time()
            actual_output, error_message, result = self._execute_code_with_timeout(code, test_input, language, test_type, timeout)
            execution_time = time.time() - start_time
            
            # Normalize outputs for comparison
            expected_normalized = self._normalize_output(expected_output)
            actual_normalized = self._normalize_output(actual_output)
            
            # Check if outputs match
            if result == ExecutionResult.PASSED and self._compare_outputs(expected_normalized, actual_normalized):
                final_result = ExecutionResult.PASSED
            else:
                final_result = result if result != ExecutionResult.PASSED else ExecutionResult.FAILED
            
            ex, act, err = self._compact_io_for_test_result(
                final_result, expected_output, actual_output, error_message
            )
            test_result = TestResult(
                test_index=test_index,
                result=final_result,
                expected_output=ex,
                actual_output=act,
                error_message=err,
                execution_time=execution_time
            )
            return test_result
            
        except Exception as e:
            return TestResult(
                test_index=test_index,
                result=ExecutionResult.RUNTIME_ERROR,
                expected_output=self._truncate_stored_io(expected_output),
                actual_output="",
                error_message=self._truncate_stored_io(f"Worker error: {str(e)}"),
                execution_time=0.0
            )

    def _execute_code_with_timeout(self, code: str, test_input: str, language: str, test_type: TestType, timeout: int) -> Tuple[str, str, ExecutionResult]:
        """Execute code with specified timeout - simplified and clean"""
        if language not in self.execution_commands:
            return "", f"Unsupported language: {language}", ExecutionResult.COMPILATION_ERROR
        
        try:
            start_time = time.time()
            stdout, stderr, return_code = self.execution_commands[language](code, test_input, timeout, test_type)
            execution_time = time.time() - start_time
            
            # Categorize result based on return code and stderr
            if return_code == -1 and stderr == "Timeout":
                return "", "Execution timeout", ExecutionResult.TIMEOUT
            elif return_code == -1:
                return "", stderr, ExecutionResult.RUNTIME_ERROR
            elif return_code != 0:
                # Check if it's compilation or runtime error
                stderr_lower = stderr.lower()
                if any(keyword in stderr_lower for keyword in ["syntax", "indentation", "import"]):
                    return "", stderr, ExecutionResult.COMPILATION_ERROR
                else:
                    return "", stderr, ExecutionResult.RUNTIME_ERROR
            else:
                return stdout, stderr, ExecutionResult.PASSED
                
        except subprocess.TimeoutExpired:
            return "", "Subprocess timeout", ExecutionResult.TIMEOUT
        except Exception as e:
            return "", f"Execution error: {str(e)}", ExecutionResult.RUNTIME_ERROR

    def _parse_test_input(self, test_input: str, test_type: TestType) -> Any:
        """
        Parse test input based on test type.
        
        For functional tests, parse as Python literals (LiveCodeBench format).
        For stdin tests, return as-is for standard input processing.
        """
        if test_type == TestType.FUNCTIONAL:
            # Clean and validate input
            cleaned_input = test_input.strip()
            if not cleaned_input:
                logger.warning("Empty functional test input")
                return ""
            
            # Check if this is a multi-line input (multiple arguments) FIRST
            lines = cleaned_input.split('\n')
            if len(lines) > 1:
                # Multi-line functional test input - parse each line separately
                parsed_args = []
                for line in lines:
                    line = line.strip()
                    if line:
                        try:
                            parsed_args.append(ast.literal_eval(line))
                        except (ValueError, SyntaxError):
                            # If individual line parsing fails, add as string
                            parsed_args.append(line)
                return parsed_args
            
            # Single-line input - try different parsing approaches
            try:
                # Try Python literal parsing first
                parsed_input = ast.literal_eval(cleaned_input)
                return parsed_input
                
            except (ValueError, SyntaxError) as e:
                # Fallback 1: Try JSON parsing (handles true/false/null differences)
                try:
                    import json
                    parsed_input = json.loads(cleaned_input)
                    logger.info(f"Successfully parsed single-line input using JSON fallback: {type(parsed_input)}")
                    return parsed_input
                except json.JSONDecodeError:
                    pass
                
                # Fallback 2: Return as raw string for the functional test executor to handle
                logger.warning(f"Could not parse functional test input, using as raw string: '{test_input[:50]}...'")
                return cleaned_input
        else:
            # For stdin tests, return as-is for subprocess input
            return test_input

    def _execute_functional_test_python(self, code: str, test_input: Any, timeout: int) -> Tuple[str, str, int]:
        """Standard Python functional test execution (used by main run())"""
        # Extract the solution class/function
        try:
            solution_code = extract_code(code)
            if not solution_code:
                solution_code = code
            
            # Add common imports that match LiveCodeBench exactly
            common_imports = [
                "from string import *",
                "from re import *", 
                "from datetime import *",
                "from collections import *",
                "from heapq import *",
                "from bisect import *",
                "from copy import *",
                "from math import *",
                "from random import *",
                "from statistics import *",
                "from itertools import *",
                "from functools import *",
                "from operator import *",
                "from io import *",
                "from sys import *",
                "from json import *",
                "from builtins import *",
                "from typing import *",
                "import string",
                "import re",
                "import datetime", 
                "import collections",
                "import heapq",
                "import bisect",
                "import copy",
                "import math",
                "import random",
                "import statistics",
                "import itertools",
                "import functools",
                "import operator",
                "import io",
                "import sys",
                "import json",
                "sys.setrecursionlimit(50000)",
            ]
            
            if self.auto_add_imports:
                enhanced_code = "\n".join(common_imports) + "\n\n" + solution_code
            else:
                enhanced_code = solution_code
                
            # Create standard LiveCodeBench-style wrapper
            wrapper_code = f'''
import sys
import traceback

def reliability_guard(maximum_memory_bytes=None):
    """Basic reliability guard for standard execution"""
    import builtins
    builtins.exit = None
    builtins.quit = None
    
    import os
    os.environ['OMP_NUM_THREADS'] = '1'

# Apply basic reliability guard
try:
    reliability_guard()
except:
    pass

# Solution code
solution_code = {repr(enhanced_code)}

# Basic main() prevention
def prevent_main_execution(code):
    """Remove main() function calls to prevent input() hangs"""
    lines = code.split('\\n')
    modified_lines = []
    
    for line in lines:
        stripped = line.strip()
        if (stripped.startswith('main()') or 
            stripped.startswith('if __name__')):
            modified_lines.append('# ' + line + '  # Prevented execution')
        else:
            modified_lines.append(line)
    
    return '\\n'.join(modified_lines)

fixed_solution_code = prevent_main_execution(solution_code)

try:
    # Execute the solution code
    exec_globals = {{"__name__": "__main__"}}
    exec(fixed_solution_code, exec_globals)
    
    # Parse test input
    test_input = {repr(test_input)}
    
    # LiveCodeBench-style method detection and execution
    if 'Solution' in exec_globals:
        solution_class = exec_globals['Solution']
        solution_instance = solution_class()
        
        # Get all callable methods (excluding private ones)
        methods = []
        for name in dir(solution_class):
            if not name.startswith('_') and callable(getattr(solution_class, name)):
                methods.append(name)
        
        # Try to execute with each method until one succeeds
        for method_name in methods:
            try:
                method = getattr(solution_instance, method_name)
                
                # Try different argument patterns
                if isinstance(test_input, list) and len(test_input) > 0:
                    try:
                        result = method(*test_input)
                        print(result)
                        sys.exit(0)
                    except (TypeError, ValueError):
                        pass
                    
                    try:
                        result = method(test_input)
                        print(result)
                        sys.exit(0)
                    except (TypeError, ValueError):
                        pass
                else:
                    try:
                        result = method(test_input)
                        print(result)
                        sys.exit(0)
                    except (TypeError, ValueError):
                        pass
                        
            except Exception:
                continue
        
        print(f"No working method found. Available methods: {{methods}}", file=sys.stderr)
        sys.exit(1)
        
    else:
        print("No Solution class found", file=sys.stderr)
        sys.exit(1)
        
except Exception as e:
    print(f"Execution error: {{str(e)}}", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
'''
            
            # Execute with subprocess
            process = subprocess.Popen(
                ['python', '-c', wrapper_code],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                preexec_fn=self._subprocess_memory_limit,
            )
            return self._communicate_limited(process, timeout=timeout)
        except Exception as e:
            return "", f"Functional test execution error: {str(e)}", -1
    

    def _execute_python(self, code: str, test_input: str, timeout: int, test_type: TestType = TestType.STDIN) -> Tuple[str, str, int]:
        """Execute Python code with given input - simplified version"""
        
        # Handle functional tests differently
        if test_type == TestType.FUNCTIONAL:
            parsed_input = self._parse_test_input(test_input, test_type)
            return self._execute_functional_test_python(code, parsed_input, timeout)
        
        # Add comprehensive imports for competitive programming (LiveCodeBench style)
        common_imports = [
            "import sys", "import math", "import itertools", "import functools",
            "from collections import defaultdict, Counter, deque, namedtuple",
            "import heapq", "import bisect", "import re", "import string",
            "import random", "import statistics", "import copy", "import operator",
            "import io", "import json", "import datetime", "import os",
            "from typing import *", "from builtins import *",
            "sys.setrecursionlimit(50000)"
        ]
        
        # Add imports if auto_add_imports is enabled and code doesn't have them
        if self.auto_add_imports and not any(line.strip().startswith(('import ', 'from ')) for line in code.split('\n')):
            enhanced_code = "\n".join(common_imports) + "\n\n" + code
        else:
            enhanced_code = code
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=True) as f:
            f.write(enhanced_code)
            f.flush()
            
            try:
                process = subprocess.Popen(
                    ['python', f.name],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    preexec_fn=self._subprocess_memory_limit,
                )
                return self._communicate_limited(process, input_data=test_input, timeout=timeout)
            except Exception:
                return "", "Execution error", -1

    def _execute_java(self, code: str, test_input: str, timeout: int, test_type: TestType = TestType.STDIN) -> Tuple[str, str, int]:
        """Execute Java code with given input"""
        if self.enable_debug_logging:
            logger.debug(f"DEBUG: Executing Java code with timeout {timeout}s")
            logger.debug(f"DEBUG: Test input: {test_input[:100]}...")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Extract class name from code
            class_name = "Solution"
            if "public class " in code:
                for line in code.split('\n'):
                    if line.strip().startswith('public class '):
                        class_name = line.split('public class ')[1].split()[0]
                        break
            
            # Add comprehensive imports if auto_add_imports is enabled and code doesn't have them
            if self.auto_add_imports and not any(line.strip().startswith('import ') for line in code.split('\n')):
                imports = [
                    "import java.util.*;",
                    "import java.io.*;",
                    "import java.math.*;",
                    "import java.text.*;",
                    "import java.time.*;",
                    "import java.util.stream.*;",
                    "import java.util.concurrent.*;",
                    "import java.util.regex.*;",
                    "import java.nio.*;",
                    "import java.net.*;",
                    ""
                ]
                enhanced_code = "\n".join(imports) + code
            else:
                enhanced_code = code
            
            # Add common Java utility methods to help with input parsing
            if self.auto_add_imports and "parseInt" in enhanced_code:
                # Add helper methods INSIDE the class, not outside
                helper_methods = """
    // Helper methods for safer input parsing
    private static int safeParseInt(String s) {
        try {
            // Remove brackets and other non-numeric characters
            s = s.replaceAll("[\\[\\],\\s]", "");
            return Integer.parseInt(s);
        } catch (NumberFormatException e) {
            return 0; // Default value
        }
    }
    
    private static int[] parseArray(String s) {
        try {
            // Remove brackets and split by comma
            s = s.replaceAll("[\\[\\]\\s]", "");
            if (s.isEmpty()) return new int[0];
            String[] parts = s.split(",");
            int[] result = new int[parts.length];
            for (int i = 0; i < parts.length; i++) {
                result[i] = safeParseInt(parts[i]);
            }
            return result;
        } catch (Exception e) {
            return new int[0];
        }
    }
"""
                # Insert helper methods INSIDE the class, before the main method
                if "public static void main" in enhanced_code:
                    enhanced_code = enhanced_code.replace("public static void main", helper_methods + "\n    public static void main")
                elif "static void main" in enhanced_code:
                    enhanced_code = enhanced_code.replace("static void main", helper_methods + "\n    static void main")
            
            java_file = os.path.join(temp_dir, f"{class_name}.java")
            with open(java_file, 'w') as f:
                f.write(enhanced_code)
            
            try:
                # Set Java environment
                env = os.environ.copy()
                env['PATH'] = "/opt/homebrew/opt/openjdk@11/bin:" + env.get('PATH', '')
                
                # Compile
                compile_process = subprocess.run(
                    ['javac', java_file],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    env=env
                )
                
                if compile_process.returncode != 0:
                    if self.enable_debug_logging:
                        logger.debug(f"DEBUG: Java compilation failed with return code {compile_process.returncode}")
                        logger.debug(f"DEBUG: Compilation error: {compile_process.stderr}")
                        logger.debug(f"DEBUG: Enhanced code (first 1000 chars): {enhanced_code[:1000]}")
                    return "", compile_process.stderr, compile_process.returncode
                
                # Execute
                process = subprocess.Popen(
                    ['java', '-cp', temp_dir, class_name],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    env=env,
                    preexec_fn=self._subprocess_memory_limit,
                )
                
                return self._communicate_limited(process, input_data=test_input, timeout=timeout)
            except Exception:
                return "", "Execution error", -1

    def _execute_cpp(self, code: str, test_input: str, timeout: int, test_type: TestType = TestType.STDIN) -> Tuple[str, str, int]:
        """Execute C++ code with given input"""
        if self.enable_debug_logging:
            logger.debug(f"DEBUG: Executing C++ code with timeout {timeout}s")
            logger.debug(f"DEBUG: Test input: {test_input[:100]}...")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            cpp_file = os.path.join(temp_dir, "solution.cpp")
            exe_file = os.path.join(temp_dir, "solution")
            
            # Add essential includes if auto_add_imports is enabled and code doesn't have them
            if self.auto_add_imports and not any(line.strip().startswith('#include') for line in code.split('\n')):
                includes = [
                    "#include <iostream>",
                    "#include <vector>",
                    "#include <string>",
                    "#include <algorithm>",
                    "#include <map>",
                    "#include <set>",
                    "#include <unordered_map>",
                    "#include <unordered_set>",
                    "#include <queue>",
                    "#include <stack>",
                    "#include <deque>",
                    "#include <cmath>",
                    "#include <climits>",
                    "#include <cstring>",
                    "#include <cstdio>",
                    "#include <cstdlib>",
                    "#include <functional>",
                    "#include <numeric>",
                    "#include <utility>",
                    "#include <sstream>",
                    "#include <iomanip>",
                    "using namespace std;",
                    ""
                ]
                enhanced_code = "\n".join(includes) + "\n\n" + code
            else:
                enhanced_code = code
            
            # Clean up common C++ issues
            enhanced_code = enhanced_code.replace("main();", "int main() {")
            if "main()" in enhanced_code and "int main()" not in enhanced_code:
                enhanced_code = enhanced_code.replace("main()", "int main()")
            
            # Fix multiple main() function definitions
            main_count = enhanced_code.count("int main()")
            if main_count > 1:
                # Keep only the first main() function and remove duplicates
                lines = enhanced_code.split('\n')
                new_lines = []
                main_found = False
                brace_count = 0
                skip_until_brace_close = False
                
                for line in lines:
                    if "int main()" in line and not main_found:
                        main_found = True
                        new_lines.append(line)
                        if "{" in line:
                            brace_count = 1
                    elif "int main()" in line and main_found:
                        # Skip duplicate main() functions
                        skip_until_brace_close = True
                        continue
                    elif skip_until_brace_close:
                        # Count braces to know when to stop skipping
                        brace_count += line.count("{") - line.count("}")
                        if brace_count <= 0:
                            skip_until_brace_close = False
                        continue
                    else:
                        new_lines.append(line)
                
                enhanced_code = '\n'.join(new_lines)
            
            # Add safer vector operations to prevent std::length_error
            if self.auto_add_imports and "vector" in enhanced_code:
                # Add helper macros for safer vector operations AFTER includes but BEFORE main
                vector_helpers = """
// Helper macros for safer vector operations
#define SAFE_VECTOR_RESIZE(v, size) if (size >= 0 && size <= 1000000) v.resize(size)
#define SAFE_VECTOR_ACCESS(v, idx) (idx >= 0 && idx < v.size() ? v[idx] : 0)
#define SAFE_VECTOR_PUSH(v, val) if (v.size() < 1000000) v.push_back(val)
"""
                # Insert helper macros after includes but before main function
                if "using namespace std;" in enhanced_code:
                    enhanced_code = enhanced_code.replace("using namespace std;", "using namespace std;" + vector_helpers)
                elif "#include" in enhanced_code:
                    # Find the last include and add after it
                    lines = enhanced_code.split('\n')
                    new_lines = []
                    for i, line in enumerate(lines):
                        new_lines.append(line)
                        if line.strip().startswith('#include'):
                            # Check if this is the last include
                            remaining_lines = lines[i+1:]
                            if not any(l.strip().startswith('#include') for l in remaining_lines):
                                new_lines.append(vector_helpers)
                    enhanced_code = '\n'.join(new_lines)
            
            with open(cpp_file, 'w') as f:
                f.write(enhanced_code)
            
            try:
                # Compile
                compile_process = subprocess.run(
                    ['g++', '-o', exe_file, cpp_file, '-std=c++17'],
                    capture_output=True,
                    text=True,
                    timeout=timeout
                )
                
                if compile_process.returncode != 0:
                    if self.enable_debug_logging:
                        logger.debug(f"DEBUG: C++ compilation failed with return code {compile_process.returncode}")
                        logger.debug(f"DEBUG: Compilation error: {compile_process.stderr}")
                        logger.debug(f"DEBUG: Enhanced code (first 1000 chars): {enhanced_code[:1000]}")
                    return "", compile_process.stderr, compile_process.returncode
                
                # Execute
                process = subprocess.Popen(
                    [exe_file],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    preexec_fn=self._subprocess_memory_limit,
                )
                
                return self._communicate_limited(process, input_data=test_input, timeout=timeout)
            except Exception:
                return "", "Execution error", -1

    def _execute_c(self, code: str, test_input: str, timeout: int, test_type: TestType = TestType.STDIN) -> Tuple[str, str, int]:
        """Execute C code with given input"""
        with tempfile.TemporaryDirectory() as temp_dir:
            c_file = os.path.join(temp_dir, "solution.c")
            exe_file = os.path.join(temp_dir, "solution")
            
            # Add comprehensive includes if auto_add_imports is enabled and code doesn't have them
            if self.auto_add_imports and not any(line.strip().startswith('#include') for line in code.split('\n')):
                includes = [
                    "#include <stdio.h>",
                    "#include <stdlib.h>",
                    "#include <string.h>",
                    "#include <math.h>",
                    "#include <ctype.h>",
                    "#include <limits.h>",
                    "#include <float.h>",
                    "#include <stdbool.h>",
                    "#include <stdint.h>",
                    "#include <stddef.h>",
                    "#include <stdarg.h>",
                    "#include <setjmp.h>",
                    "#include <signal.h>",
                    "#include <time.h>",
                    "#include <locale.h>",
                    "#include <errno.h>",
                    "#include <assert.h>",
                    "#include <complex.h>",
                    "#include <fenv.h>",
                    "#include <inttypes.h>",
                    "#include <iso646.h>",
                    "#include <stdalign.h>",
                    "#include <stdatomic.h>",
                    "#include <stdnoreturn.h>",
                    "#include <threads.h>",
                    "#include <uchar.h>",
                    "#include <wchar.h>",
                    "#include <wctype.h>",
                    ""
                ]
                enhanced_code = "\n".join(includes) + "\n\n" + code
            else:
                enhanced_code = code
            
            with open(c_file, 'w') as f:
                f.write(enhanced_code)
            
            try:
                # Compile
                compile_process = subprocess.run(
                    ['gcc', '-o', exe_file, c_file, '-std=c99'],
                    capture_output=True,
                    text=True,
                    timeout=timeout
                )
                
                if compile_process.returncode != 0:
                    return "", compile_process.stderr, compile_process.returncode
                
                # Execute
                process = subprocess.Popen(
                    [exe_file],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    preexec_fn=self._subprocess_memory_limit,
                )
                
                return self._communicate_limited(process, input_data=test_input, timeout=timeout)
            except Exception:
                return "", "Execution error", -1

    def _execute_javascript(self, code: str, test_input: str, timeout: int, test_type: TestType = TestType.STDIN) -> Tuple[str, str, int]:
        """Execute JavaScript code with given input"""
        
        # Add comprehensive imports if auto_add_imports is enabled and code doesn't have them
        if self.auto_add_imports and not any(line.strip().startswith(('import ', 'const ', 'let ', 'var ')) for line in code.split('\n')):
            # JavaScript doesn't have traditional imports like other languages, but we can add common utilities
            common_utilities = [
                "// Common JavaScript utilities for competitive programming",
                "const readline = require('readline');",
                "const fs = require('fs');",
                "const path = require('path');",
                "const util = require('util');",
                "const crypto = require('crypto');",
                "const os = require('os');",
                "const process = require('process');",
                "const child_process = require('child_process');",
                "const events = require('events');",
                "const stream = require('stream');",
                "const buffer = require('buffer');",
                "const url = require('url');",
                "const querystring = require('querystring');",
                "const http = require('http');",
                "const https = require('https');",
                "const net = require('net');",
                "const dgram = require('dgram');",
                "const dns = require('dns');",
                "const zlib = require('zlib');",
                "const readline = require('readline');",
                "",
                "// Helper functions for competitive programming",
                "function readInt() { return parseInt(readline()); }",
                "function readFloat() { return parseFloat(readline()); }",
                "function readString() { return readline(); }",
                "function readArray() { return readline().split(' ').map(Number); }",
                "function readStringArray() { return readline().split(' '); }",
                "",
            ]
            enhanced_code = "\n".join(common_utilities) + "\n\n" + code
        else:
            enhanced_code = code
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=True) as f:
            f.write(enhanced_code)
            f.flush()
            
            try:
                process = subprocess.Popen(
                    ['node', f.name],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    preexec_fn=self._subprocess_memory_limit,
                )
                
                return self._communicate_limited(process, input_data=test_input, timeout=timeout)
            except Exception:
                return "", "Execution error", -1

    def _execute_go(self, code: str, test_input: str, timeout: int, test_type: TestType = TestType.STDIN) -> Tuple[str, str, int]:
        """Execute Go code with given input"""
        if self.enable_debug_logging:
            logger.debug(f"DEBUG: Executing Go code with timeout {timeout}s")
            logger.debug(f"DEBUG: Test input: {test_input[:100]}...")
        
        # Add common imports that are often needed for competitive programming
        common_imports = [
            "fmt",
            "strings",
            "strconv",
            "sort",
            "math",
            "bufio",
            "os",
            "io",
            "bytes",
            "unicode",
            "regexp",
            "time",
            "container/heap",
            "container/list",
        ]
        
        # Check if code already has imports
        has_imports = any(line.strip().startswith('import ') for line in code.split('\n'))
        
        # If no imports and auto_add_imports is enabled, add comprehensive ones
        if not has_imports and self.auto_add_imports:
            import_block = 'import (\n    "' + '"\n    "'.join(common_imports) + '"\n)'
            # Ensure proper Go structure: package main, imports, then code
            if not code.strip().startswith('package main'):
                enhanced_code = "package main\n\n" + import_block + "\n\n" + code
            else:
                # Code already has package main, just add imports
                lines = code.split('\n')
                new_lines = []
                for line in lines:
                    new_lines.append(line)
                    if line.strip() == 'package main':
                        new_lines.append('')
                        new_lines.append(import_block)
                        new_lines.append('')
                enhanced_code = '\n'.join(new_lines)
        else:
            enhanced_code = code
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.go', delete=True) as f:
            f.write(enhanced_code)
            f.flush()
            
            try:
                process = subprocess.Popen(
                    ['go', 'run', f.name],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    preexec_fn=self._subprocess_memory_limit,
                )
                
                stdout, stderr, return_code = self._communicate_limited(process, input_data=test_input, timeout=timeout)
                if return_code != 0 and self.enable_debug_logging:
                    logger.debug(f"DEBUG: Go execution failed with return code {return_code}")
                    logger.debug(f"DEBUG: Go error: {stderr}")
                    logger.debug(f"DEBUG: Enhanced code (first 1000 chars): {enhanced_code[:1000]}")
                
                return stdout, stderr, return_code
            except Exception:
                return "", "Execution error", -1

    def _execute_rust(self, code: str, test_input: str, timeout: int, test_type: TestType = TestType.STDIN) -> Tuple[str, str, int]:
        """Execute Rust code with given input"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Add comprehensive imports for competitive programming
            common_imports = [
                "use std::collections::*;",
                "use std::io::*;",
                "use std::cmp::*;",
                "use std::mem::*;",
                "use std::str::*;",
                "use std::fmt::*;",
                "use std::iter::*;",
                "use std::ops::*;",
                "use std::convert::*;",
                "use std::default::*;",
                "use std::hash::*;",
                "use std::marker::*;",
                "use std::num::*;",
                "use std::ptr::*;",
                "use std::rc::*;",
                "use std::sync::*;",
                "use std::thread::*;",
                "use std::time::*;",
                "use std::fs::*;",
                "use std::path::*;",
                "use std::env::*;",
                "use std::process::*;",
                "use std::os::*;",
                "use std::ffi::*;",
                "use std::net::*;",
                "use std::ascii::*;",
                "use std::borrow::*;",
                "use std::cell::*;",
                "use std::char::*;",
                "use std::clone::*;",
                "use std::error::*;",
                "use std::f32::*;",
                "use std::f64::*;",
                "use std::i8::*;",
                "use std::i16::*;",
                "use std::i32::*;",
                "use std::i64::*;",
                "use std::i128::*;",
                "use std::isize::*;",
                "use std::u8::*;",
                "use std::u16::*;",
                "use std::u32::*;",
                "use std::u64::*;",
                "use std::u128::*;",
                "use std::usize::*;",
            ]
            
            # Check if code already has imports
            has_imports = any(line.strip().startswith('use ') for line in code.split('\n'))
            
            # If no imports and auto_add_imports is enabled, add common ones
            if not has_imports and self.auto_add_imports:
                enhanced_code = "\n".join(common_imports) + "\n\n" + code
            else:
                enhanced_code = code
            
            rust_file = os.path.join(temp_dir, "solution.rs")
            exe_file = os.path.join(temp_dir, "solution")
            
            with open(rust_file, 'w') as f:
                f.write(enhanced_code)
            
            try:
                # Compile
                compile_process = subprocess.run(
                    ['rustc', '-o', exe_file, rust_file],
                    capture_output=True,
                    text=True,
                    timeout=timeout
                )
                
                if compile_process.returncode != 0:
                    return "", compile_process.stderr, compile_process.returncode
                
                # Execute
                process = subprocess.Popen(
                    [exe_file],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    )
                
                return self._communicate_limited(process, input_data=test_input, timeout=timeout)
            except Exception:
                return "", "Execution error", -1

    def _execute_ruby(self, code: str, test_input: str, timeout: int, test_type: TestType = TestType.STDIN) -> Tuple[str, str, int]:
        """Execute Ruby code with given input"""
        with tempfile.TemporaryDirectory() as temp_dir:
            ruby_file = os.path.join(temp_dir, "solution.rb")
            
            with open(ruby_file, 'w') as f:
                f.write(code)
            
            try:
                # Execute Ruby directly (no compilation needed)
                process = subprocess.Popen(
                    ['ruby', ruby_file],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    preexec_fn=self._subprocess_memory_limit,
                )
                
                return self._communicate_limited(process, input_data=test_input, timeout=timeout)
            except Exception:
                return "", "Execution error", -1

    def _normalize_output(self, output) -> str:
        """Normalize output for comparison"""
        # Convert to string if it's not already a string
        if not isinstance(output, str):
            output = str(output)
        return output.strip().replace('\r\n', '\n').replace('\r', '\n')

    def _is_number(self, s: str) -> bool:
        """Check if a string represents a number (int or float)"""
        try:
            float(s)
            return True
        except ValueError:
            return False

    def _numbers_are_close(self, num1: str, num2: str, abs_tol: float = 1e-6) -> bool:
        """
        Compare two numeric strings with absolute tolerance for floating-point precision.
        
        Uses absolute tolerance which is simpler and more intuitive than relative tolerance.
        Two numbers are considered equal if their absolute difference is within the tolerance.
        
        Args:
            num1, num2: String representations of numbers
            abs_tol: Absolute tolerance (default 1e-6, good for most competitive programming problems)
        """
        try:
            val1 = float(num1)
            val2 = float(num2)
            
            # Handle special cases
            if val1 == val2:
                return True
            
            # Check if both are infinite or both are NaN
            if (val1 == float('inf') and val2 == float('inf')) or \
               (val1 == float('-inf') and val2 == float('-inf')):
                return True
            
            if (val1 != val1) and (val2 != val2):  # Both NaN
                return False  # NaN != NaN by IEEE standards
            
            # Use absolute tolerance - much simpler!
            return abs(val1 - val2) <= abs_tol
            
        except (ValueError, OverflowError):
            return False

    def _compare_outputs(self, expected: str, actual: str) -> bool:
        """
        Compare two outputs with intelligent handling of floating-point numbers.
        
        This method implements a sophisticated comparison strategy:
        1. First try exact string comparison (fastest path)
        2. Normalize whitespace and try again
        3. Split into tokens and compare token by token
        4. For numeric tokens, use floating-point tolerant comparison
        5. For non-numeric tokens, use exact string comparison
        
        This approach handles mixed outputs (text + numbers) and provides
        the floating-point tolerance needed for competitive programming problems.
        """
        # Fast path: exact match
        if expected == actual:
            return True
        
        # Normalize basic whitespace differences
        expected_norm = self._normalize_output(expected)
        actual_norm = self._normalize_output(actual)
        
        if expected_norm == actual_norm:
            return True
        
        # Split into tokens for detailed comparison
        expected_tokens = expected_norm.split()
        actual_tokens = actual_norm.split()
        
        # Must have same number of tokens
        if len(expected_tokens) != len(actual_tokens):
            return False
        
        # Compare token by token
        for exp_token, act_token in zip(expected_tokens, actual_tokens):
            # If both tokens are numbers, use tolerant comparison
            if self._is_number(exp_token) and self._is_number(act_token):
                if not self._numbers_are_close(exp_token, act_token, abs_tol=self.abs_tolerance):
                    return False
            else:
                # For non-numeric tokens, require exact match
                if exp_token != act_token:
                    return False
        
        return True



    def _run_test_case(self, code: str, test_case: Dict[str, Any], test_index: int, language: str, test_type: TestType) -> TestResult:
        """Run a single test case with fixed timeout"""
        test_input = test_case.get('input', '')
        expected_output = test_case.get('output', '')
        
        start_time = time.time()
        actual_output, error_message, result = self._execute_code_with_timeout(code, test_input, language, test_type, self.timeout)
        execution_time = time.time() - start_time
        
        # Normalize outputs for comparison
        expected_normalized = self._normalize_output(expected_output)
        actual_normalized = self._normalize_output(actual_output)
        
        # Check if outputs match
        if result == ExecutionResult.PASSED and self._compare_outputs(expected_normalized, actual_normalized):
            final_result = ExecutionResult.PASSED
        else:
            final_result = result if result != ExecutionResult.PASSED else ExecutionResult.FAILED
        
        ex, act, err = self._compact_io_for_test_result(
            final_result, expected_output, actual_output, error_message
        )
        return TestResult(
            test_index=test_index,
            result=final_result,
            expected_output=ex,
            actual_output=act,
            error_message=err,
            execution_time=execution_time
        )

    def _evaluate_solution(self, lcb_id: str, language: str, model_name: str, 
                          solution: str, solution_index: int, test_cases: List[Dict[str, Any]]) -> EvaluationResult:
        """Evaluate a single solution against test cases"""
        logger.info(f"Evaluating {model_name} {language} solution {solution_index} for {lcb_id}")
        
        # Extract code from solution
        code = extract_code(solution)
        if not code:
            code = solution  # Use raw solution if no code blocks found
        
        # Debug logging for failed cases
        if self.enable_debug_logging:
            logger.debug(f"DEBUG: Raw solution for {lcb_id} {model_name} {language}:")
            logger.debug(f"DEBUG: {solution[:500]}...")  # First 500 chars
            logger.debug(f"DEBUG: Extracted code:")
            logger.debug(f"DEBUG: {code[:500]}...")  # First 500 chars
        
        # Prepare arguments for multiprocessing
        test_args = []
        for i, test_case in enumerate(test_cases):
            # Determine test type
            test_type = TestType.STDIN
            if 'testtype' in test_case:
                test_type_str = test_case['testtype'].lower()
                if test_type_str == 'functional':
                    test_type = TestType.FUNCTIONAL
            
            # Get test input and expected output
            test_input = test_case.get('input', '')
            expected_output = test_case.get('output', '')
            
            test_args.append((code, test_input, expected_output, i, language, test_type, self.timeout))
        
        # Run test cases sequentially with a per-language wall-clock cap so that a
        # single slow problem cannot exceed per_language_timeout seconds.
        test_results = []
        total_execution_time = 0
        lang_wall_start = time.time()

        if self.num_processes > 1 and len(test_cases) > 5:
            try:
                with Pool(processes=min(self.num_processes, 2), maxtasksperchild=10) as pool:
                    test_results = pool.map(self._evaluate_test_case_worker, test_args)
                    pool.close()
                    pool.join()
            except Exception as e:
                logger.warning(f"Multiprocessing failed ({e}), falling back to sequential evaluation")
                test_results = []
                for args in test_args:
                    elapsed = time.time() - lang_wall_start
                    if elapsed >= self.per_language_timeout:
                        remaining = len(test_args) - len(test_results)
                        logger.warning(
                            f"Per-language wall-clock cap ({self.per_language_timeout}s) hit "
                            f"after {elapsed:.1f}s for {lcb_id} {language}; "
                            f"marking {remaining} remaining test(s) as TIMEOUT"
                        )
                        test_index = args[3]
                        for j in range(len(test_results), len(test_args)):
                            test_results.append(TestResult(
                                test_index=test_args[j][3],
                                result=ExecutionResult.TIMEOUT,
                                expected_output=self._truncate_stored_io(test_args[j][2]),
                                actual_output="",
                                error_message="Skipped: per-language wall-clock cap exceeded",
                                execution_time=0.0,
                            ))
                        break
                    test_result = self._evaluate_test_case_worker(args)
                    test_results.append(test_result)
        else:
            for args in test_args:
                elapsed = time.time() - lang_wall_start
                if elapsed >= self.per_language_timeout:
                    remaining = len(test_args) - len(test_results)
                    logger.warning(
                        f"Per-language wall-clock cap ({self.per_language_timeout}s) hit "
                        f"after {elapsed:.1f}s for {lcb_id} {language}; "
                        f"marking {remaining} remaining test(s) as TIMEOUT"
                    )
                    for j in range(len(test_results), len(test_args)):
                        test_results.append(TestResult(
                            test_index=test_args[j][3],
                            result=ExecutionResult.TIMEOUT,
                            expected_output=self._truncate_stored_io(test_args[j][2]),
                            actual_output="",
                            error_message="Skipped: per-language wall-clock cap exceeded",
                            execution_time=0.0,
                        ))
                    break
                test_result = self._evaluate_test_case_worker(args)
                test_results.append(test_result)
                
        # Force garbage collection after test evaluation
        gc.collect()
        
        # Calculate total execution time
        total_execution_time = sum(tr.execution_time for tr in test_results)
        
        # Log progress
        passed_count = sum(1 for tr in test_results if tr.result == ExecutionResult.PASSED)
        logger.info(f"Completed {len(test_cases)} tests for {lcb_id} ({passed_count} passed)")
        
        # Calculate metrics
        passed_tests = sum(1 for result in test_results if result.result == ExecutionResult.PASSED)
        total_tests = len(test_results)
        pass_rate = passed_tests / total_tests if total_tests > 0 else 0.0
        
        # Determine overall result
        if passed_tests == total_tests:
            overall_result = ExecutionResult.PASSED
        elif passed_tests == 0:
            # Check if all failures are due to the same issue
            error_types = [result.result for result in test_results]
            if all(error_type == ExecutionResult.COMPILATION_ERROR for error_type in error_types):
                overall_result = ExecutionResult.COMPILATION_ERROR
            elif all(error_type == ExecutionResult.TIMEOUT for error_type in error_types):
                overall_result = ExecutionResult.TIMEOUT
            else:
                overall_result = ExecutionResult.FAILED
        else:
            overall_result = ExecutionResult.FAILED
        
        # Force garbage collection to free memory
        gc.collect()
        
        return EvaluationResult(
            lcb_id=lcb_id,
            language=language,
            model_name=model_name,
            solution_index=solution_index,
            passed_tests=passed_tests,
            total_tests=total_tests,
            pass_rate=pass_rate,
            test_results=test_results,
            overall_result=overall_result,
            total_execution_time=total_execution_time
        )

    def _build_test_cases_split(self, src_path: str) -> str:
        """Split the large test-cases JSON into one small file per problem.

        The split directory lives next to the source file with a ".split" suffix.
        The one-time build scans the source file with ijson (streaming) so peak
        memory during the build is O(one entry), not O(whole file).
        On subsequent runs the existing split directory is reused directly.
        """
        import json as _json, os as _os
        # Always extract all known test-case fields so the same split dir
        # can serve both "lcb" and "annotate" prefix evaluations.
        needed = {
            'lcb_public_test_cases', 'lcb_private_test_cases',
            'annotate_public_test_cases', 'annotate_private_test_cases',
        }
        split_dir = src_path + ".split"
        index_path = _os.path.join(split_dir, "_index.json")

        if _os.path.exists(index_path):
            with open(index_path) as fh:
                idx = _json.load(fh)
            if abs(idx.get("source_mtime", 0) - _os.path.getmtime(src_path)) < 2:
                count = idx.get("count", "?")
                logger.info(f"Using existing test-cases split ({count} files) in {split_dir}")
                return split_dir

        logger.info(f"Building per-problem test-case split from {src_path} (one-time ~seconds)…")
        _os.makedirs(split_dir, exist_ok=True)
        count = 0
        try:
            import ijson
            with open(src_path, "rb") as fh:
                for key, value in ijson.kvitems(fh, ""):
                    trimmed = {f: value[f] for f in needed if f in value}
                    safe_key = key.replace("/", "_").replace("\\", "_")
                    with open(_os.path.join(split_dir, safe_key + ".json"), "w") as of:
                        _json.dump(trimmed, of)
                    count += 1
            logger.info(f"ijson split complete: {count} files")
        except ImportError:
            logger.warning("ijson not installed; falling back to full json.load split (uses more RAM once)")
            with open(src_path) as fh:
                raw = _json.load(fh)
            for key, value in raw.items():
                trimmed = {f: value[f] for f in needed if f in value}
                safe_key = key.replace("/", "_").replace("\\", "_")
                with open(_os.path.join(split_dir, safe_key + ".json"), "w") as of:
                    _json.dump(trimmed, of)
                count += 1
            del raw
            gc.collect()

        with open(index_path, "w") as fh:
            _json.dump({"source_mtime": _os.path.getmtime(src_path), "count": count}, fh)
        logger.info(f"Split complete: {count} files in {split_dir}")
        return split_dir

    def _load_test_case_entry(self, lcb_id: str) -> Optional[Dict[str, Any]]:
        """Load one problem's test-case file from the split directory."""
        import json as _json, os as _os
        if not self.test_cases_split_dir:
            return None
        bare_id = lcb_id[4:] if lcb_id.startswith("lcb_") else lcb_id
        for candidate in (lcb_id, bare_id):
            safe = candidate.replace("/", "_").replace("\\", "_")
            path = _os.path.join(self.test_cases_split_dir, safe + ".json")
            if _os.path.exists(path):
                with open(path) as fh:
                    return _json.load(fh)
        return None

    def _get_test_cases(self, item: Dict[str, Any], lcb_id: str) -> List[Dict[str, Any]]:
        """Get test cases for evaluation"""
        test_cases = []
        test_case_item = None

        # Load from per-problem split file: O(one file) memory, no large dict in RAM.
        if self.test_cases_split_dir:
            test_case_item = self._load_test_case_entry(lcb_id)
            if test_case_item:
                logger.info(f"Found test case data for {lcb_id} in separate file")
        
        # Extract test cases from test_case_item if found
        if test_case_item:
            # Add public test cases from test cases file using configured field name
            public_tests = test_case_item.get(self.public_test_field)
            if public_tests:
                test_cases.extend(public_tests)
                logger.info(f"Found {len(public_tests)} public test cases from test cases file")
            
            # Add private test cases if enabled using configured field name
            if self.use_private_tests:
                private_tests = test_case_item.get(self.private_test_field)
                if private_tests:
                    test_cases.extend(private_tests)
                    logger.info(f"Found {len(private_tests)} private test cases from test cases file")
                
        # If no test cases found in separate file, try the original item
        if not test_cases:
            logger.info(f"No test cases found in separate file for {lcb_id}, trying original data")
            
            # Add public test cases using configured field name
            public_tests = item.get(self.public_test_field)
            if public_tests:
                test_cases.extend(public_tests)
                logger.info(f"Found {len(public_tests)} public test cases from original data")
            
            # Add private test cases if enabled using configured field name
            if self.use_private_tests:
                private_tests = item.get(self.private_test_field)
                if private_tests:
                    test_cases.extend(private_tests)
                    logger.info(f"Found {len(private_tests)} private test cases from original data")
        
        # Normalize test case format to ensure testtype field is present
        normalized_test_cases = []
        for test_case in test_cases:
            normalized_case = test_case.copy()
            # Ensure testtype field exists, default to stdin if not present
            if 'testtype' not in normalized_case:
                normalized_case['testtype'] = 'stdin'
            normalized_test_cases.append(normalized_case)
        
        return normalized_test_cases

    def _should_evaluate_model_language(self, lcb_id: str, model_name: str, language: str) -> bool:
        """Check if we should evaluate this model-language combination - enhanced logic with round checking"""
        eval_key = f"{model_name}_{language}_evaluation"
        
        # First check if we should skip based on overwrite setting and current round
        if not self.overwrite and lcb_id in self.db and eval_key in self.db[lcb_id]:
            eval_data = self.db[lcb_id][eval_key]
            
            # Check if current round already exists
            if 'rounds' in eval_data and self.round_id in eval_data['rounds']:
                logger.info(f"Skipping {model_name} {language} evaluation for {lcb_id} - round {self.round_id} already exists")
                return False
        
        # Then check if we should evaluate based on target_cases setting
        if self.target_cases == "timeout":
            # Only evaluate if previous results had timeouts
            if lcb_id in self.db and eval_key in self.db[lcb_id]:
                has_timeouts = self._has_timeouts_in_results(self.db[lcb_id][eval_key])
                if has_timeouts:
                    logger.info(f"Will evaluate {model_name} {language} for {lcb_id} - has timeouts")
                else:
                    logger.info(f"Skipping {model_name} {language} for {lcb_id} - no timeouts found")
                return has_timeouts
            logger.info(f"Skipping {model_name} {language} for {lcb_id} - no previous evaluation data")
            return False
        elif self.target_cases == "failed":
            # Only evaluate if previous results had failures
            if lcb_id in self.db and eval_key in self.db[lcb_id]:
                has_failures = self._has_failures_in_results(self.db[lcb_id][eval_key])
                if has_failures:
                    logger.info(f"Will evaluate {model_name} {language} for {lcb_id} - has failures")
                else:
                    logger.info(f"Skipping {model_name} {language} for {lcb_id} - no failures found")
                return has_failures
            logger.info(f"Skipping {model_name} {language} for {lcb_id} - no previous evaluation data")
            return False
        elif self.target_cases == "all":
            # Evaluate all cases
            logger.info(f"Will evaluate {model_name} {language} for {lcb_id} - target_cases=all")
            return True
        
        return True
    
    def _has_timeouts_in_results(self, eval_data: dict) -> bool:
        """Check if evaluation results contain timeouts"""
        # Handle both legacy and multi-round formats
        if 'rounds' in eval_data:
            # Multi-round format - check latest round
            latest_round_id = eval_data.get('latest_round')
            if latest_round_id and latest_round_id in eval_data['rounds']:
                latest_round = eval_data['rounds'][latest_round_id]
                return self._check_round_for_timeouts(latest_round)
        else:
            # Legacy format
            return self._check_round_for_timeouts(eval_data)
        return False
    
    def _has_failures_in_results(self, eval_data: dict) -> bool:
        """Check if evaluation results contain failures"""
        # Handle both legacy and multi-round formats
        if 'rounds' in eval_data:
            # Multi-round format - check latest round
            latest_round_id = eval_data.get('latest_round')
            if latest_round_id and latest_round_id in eval_data['rounds']:
                latest_round = eval_data['rounds'][latest_round_id]
                return self._check_round_for_failures(latest_round)
        else:
            # Legacy format
            return self._check_round_for_failures(eval_data)
        return False
    
    def _check_round_for_timeouts(self, round_data: dict) -> bool:
        """Check if a specific round has timeouts"""
        if 'evaluations' in round_data:
            for eval_result in round_data['evaluations']:
                if 'test_results' in eval_result:
                    for test_result in eval_result['test_results']:
                        if test_result.get('result') == 'timeout':
                            return True
        return False
    
    def _check_round_for_failures(self, round_data: dict) -> bool:
        """Check if a specific round has failures"""
        if 'evaluations' in round_data:
            for eval_result in round_data['evaluations']:
                if eval_result.get('overall_result') in ['failed', 'runtime_error', 'compilation_error']:
                    return True
        return False
    
    def _track_evaluation_fixes(self, eval_data: Dict[str, Any], current_round_id: str, current_round_data: Dict[str, Any]) -> None:
        """Track which issues were fixed in this round compared to previous rounds - Flexible fix tracking"""
        if 'fixes_tracker' not in eval_data:
            eval_data['fixes_tracker'] = {}
        
        # Get all previous rounds (excluding current)
        previous_rounds = {rid: rdata for rid, rdata in eval_data['rounds'].items() if rid != current_round_id}
        
        if not previous_rounds:
            # No previous rounds to compare with
            return
        
        # Get the most recent previous round for comparison
        latest_previous_round_id = max(previous_rounds.keys(), key=lambda x: previous_rounds[x].get('timestamp', 0))
        previous_round_data = previous_rounds[latest_previous_round_id]
        
        # Extract fix strategy from round description/config
        fix_strategy = self._identify_fix_strategy(current_round_id, current_round_data.get('round_description', ''))
        
        # Track solution-level fixes with flexible categorization
        for current_eval in current_round_data.get('evaluations', []):
            sol_idx = current_eval['solution_index']
            current_result = current_eval['overall_result']
            
            # Find corresponding evaluation in previous round
            previous_eval = None
            for prev_eval in previous_round_data.get('evaluations', []):
                if prev_eval['solution_index'] == sol_idx:
                    previous_eval = prev_eval
                    break
            
            if previous_eval:
                previous_result = previous_eval['overall_result']
                
                # Track any result change (not just to 'passed')
                if previous_result != current_result:
                    fix_info = self._analyze_fix_type(previous_result, current_result, fix_strategy, current_eval, previous_eval)
                    
                    if fix_info:
                        fix_key = f"solution_{sol_idx}_{fix_info['change_type']}"
                        eval_data['fixes_tracker'][fix_key] = {
                            'fixed_in_round': current_round_id,
                            'previous_round': latest_previous_round_id,
                            'previous_result': previous_result,
                            'current_result': current_result,
                            'fix_strategy': fix_strategy,
                            'fix_category': fix_info['category'],
                            'fix_type': fix_info['type'],
                            'improvement_level': fix_info['improvement_level'],
                            'confidence': fix_info['confidence'],
                            'details': fix_info['details'],
                            'timestamp': current_round_data['timestamp']
                        }
                        
                        # Log with appropriate emoji based on improvement level
                        emoji = fix_info['emoji']
                        logger.info(f"{emoji} Solution {sol_idx} {fix_info['description']} in round {current_round_id}")
        
        # Track test-level improvements for partial fixes
        self._track_test_level_improvements(eval_data, current_round_id, current_round_data, previous_round_data, fix_strategy)
        
        # Track overall improvement metrics with fix attribution
        self._track_overall_improvements(eval_data, current_round_id, current_round_data, previous_round_data, fix_strategy)
    
    def _update_best_performance_metrics(self, existing_eval: dict, new_round_data: dict):
        """Update best performance metrics across all rounds"""
        if 'best_performance' not in existing_eval:
            existing_eval['best_performance'] = {}
        
        best_perf = existing_eval['best_performance']
        
        # Update best pass_at_1
        new_pass_at_1 = new_round_data.get('pass_at_1', 0.0)
        if 'pass_at_1' not in best_perf or new_pass_at_1 > best_perf['pass_at_1']:
            best_perf['pass_at_1'] = new_pass_at_1
            best_perf['best_pass_at_1_round'] = new_round_data['round_id']
        
        # Update best solutions_passed
        new_solutions_passed = new_round_data.get('solutions_passed', 0)
        if 'solutions_passed' not in best_perf or new_solutions_passed > best_perf['solutions_passed']:
            best_perf['solutions_passed'] = new_solutions_passed
            best_perf['best_solutions_passed_round'] = new_round_data['round_id']
        
        # Update best pass rates for each k value
        for key, value in new_round_data.items():
            if key.startswith('pass_at_') and key != 'pass_at_1':
                if key not in best_perf or value > best_perf[key]:
                    best_perf[key] = value
                    best_perf[f'best_{key}_round'] = new_round_data['round_id']

    def _get_available_models(self, existing_data: Dict[str, Any]) -> List[str]:
        """Get list of available models from existing data"""
        models = set()
        # Look for keys that end with language names and extract model names
        language_suffixes = ['_python', '_java', '_cpp', '_c', '_javascript', '_go', '_rust']
        
        for key in existing_data.keys():
            # Skip evaluation results and metadata
            if key.endswith('_evaluation') or key.startswith(('lcb_', 'ANNOTATED_', 'annotate_')):
                continue
                
            # Check if this key represents a model-language combination
            for suffix in language_suffixes:
                if key.endswith(suffix):
                    model_name = key[:-len(suffix)]
                    models.add(model_name)
                    break
        
        return list(models)

    def _record_skipped_problem(self, lcb_id: str, item: Dict[str, Any]) -> None:
        """Write FAILED evaluation records for every model-language combo in item."""
        if lcb_id not in self.db:
            self.db[lcb_id] = {}
        existing_data = self.db.get(lcb_id, {})
        existing_data.setdefault('lcb_difficulty', item.get('lcb_difficulty', 'unknown'))
        existing_data.setdefault('lcb_question_id', lcb_id)

        available_models = self._get_available_models(item)
        if self.models_to_evaluate:
            available_models = [m for m in available_models if m in self.models_to_evaluate]

        for model_name in available_models:
            for language in self.languages:
                solution_key = f"{model_name}_{language}"
                if solution_key not in item:
                    continue
                if not self._should_evaluate_model_language(lcb_id, model_name, language):
                    continue
                solutions = item[solution_key]
                if not isinstance(solutions, list):
                    solutions = [solutions]
                eval_key = f"{model_name}_{language}_evaluation"
                failed_round = {
                    'round_id': self.round_id,
                    'round_description': self.round_description,
                    'timestamp': time.time(),
                    'timeout': self.timeout,
                    'num_solutions': len(solutions),
                    'solutions_passed': 0,
                    'pass_at_1': 0.0,
                    **{f'pass_at_{k}': 0.0 for k in self.k_values},
                    'evaluations': [
                        {
                            'solution_index': i,
                            'passed_tests': 0,
                            'total_tests': 0,
                            'pass_rate': 0.0,
                            'overall_result': ExecutionResult.RUNTIME_ERROR.value,
                            'execution_time': 0.0,
                            'test_results': [],
                        }
                        for i in range(len(solutions))
                    ],
                    'skipped': True,
                }
                if eval_key in existing_data and 'rounds' in existing_data[eval_key]:
                    existing_data[eval_key]['rounds'][self.round_id] = failed_round
                    existing_data[eval_key]['latest_round'] = self.round_id
                    existing_data[eval_key]['total_rounds'] = len(existing_data[eval_key]['rounds'])
                else:
                    existing_data[eval_key] = {
                        'model_name': model_name,
                        'language': language,
                        'latest_round': self.round_id,
                        'total_rounds': 1,
                        'rounds': {self.round_id: failed_round},
                        'fixes_tracker': {},
                    }
                write_to_database(self.data_config.output_db, lcb_id, existing_data)
                logger.info(f"Recorded FAILED (skipped) for {model_name} {language} on {lcb_id}")

    def run(self):
        """Main execution loop - processes normal evaluation cases with batch processing for memory efficiency"""
        processed_count = 0
        
        # Convert to list for batch processing, then release the original dict so we
        # don't keep two copies of all solution strings in memory simultaneously.
        items_list = list(self.codebench_data.items())
        total_items = len(items_list)
        del self.codebench_data
        gc.collect()
        
        # Process in batches to manage memory
        for batch_start in range(0, total_items, self.batch_size):
            batch_end = min(batch_start + self.batch_size, total_items)
            batch_items = items_list[batch_start:batch_end]
            
            logger.info(f"Processing batch {batch_start//self.batch_size + 1}/{(total_items-1)//self.batch_size + 1}: "
                       f"items {batch_start+1}-{batch_end} of {total_items}")
            
            for question_id, item in batch_items:
                if item is None:
                    continue
                # Handle different ID formats
                if 'lcb_question_id' in item:
                    lcb_id = item['lcb_question_id']
                elif isinstance(question_id, str) and question_id.startswith('lcb_'):
                    lcb_id = question_id
                else:
                    lcb_id = 'lcb_' + str(question_id)

                test_cases = None
                try:
                    logger.info(f"Processing {processed_count + 1}/{total_items}: {lcb_id}")

                    # Skip list: mark every model-language combo as FAILED and move on.
                    if lcb_id in self.skip_problems:
                        logger.warning(f"Skipping {lcb_id} (in skip_problems list) — recording as FAILED")
                        self._record_skipped_problem(lcb_id, item)
                        processed_count += 1
                        continue
                    
                    # Get test cases from the appropriate source (test_cases_dir or codebench_data)
                    test_cases = self._get_test_cases(item, lcb_id)
                    if not test_cases:
                        logger.warning(f"No test cases found for {lcb_id}, skipping evaluation")
                        continue
                    
                    logger.info(f"Found {len(test_cases)} total test cases for {lcb_id}")
                    
                    # Get available models from the generated code in codebench_data
                    available_models = self._get_available_models(item)
                    
                    # Filter models if specified
                    if self.models_to_evaluate:
                        available_models = [model for model in available_models if model in self.models_to_evaluate]
                    
                    if not available_models:
                        logger.warning(f"No models to evaluate for {lcb_id}")
                        continue
                    
                    # Initialize database entry if not exists
                    if lcb_id not in self.db:
                        self.db[lcb_id] = {}
                    
                    # Get existing data for this question (empty dict if question doesn't exist)
                    existing_data = self.db.get(lcb_id, {})
                    
                    # Store problem metadata if not already present
                    if 'lcb_difficulty' not in existing_data:
                        existing_data['lcb_difficulty'] = item.get('lcb_difficulty', 'unknown')
                    if 'lcb_question_id' not in existing_data:
                        existing_data['lcb_question_id'] = lcb_id
                    
                    # Evaluate each model-language combination
                    evaluations_performed = False
                    
                    for model_name in available_models:
                        for language in self.languages:
                            solution_key = f"{model_name}_{language}"
                            
                            # Check if generated code exists for this model-language combination
                            if solution_key not in item:
                                continue
                            
                            if not self._should_evaluate_model_language(lcb_id, model_name, language):
                                continue
                            
                            # Get solutions from codebench_data
                            solutions = item[solution_key]
                            if not isinstance(solutions, list):
                                solutions = [solutions]
                            
                            logger.info(f"Evaluating {len(solutions)} {model_name} {language} solutions for {lcb_id}")
                            
                            # Initialize evaluation results for this model-language combination
                            evaluation_results = []
                            solutions_passed = 0
                            pass_at_1 = 0.0
                            pass_at_k_results = {}
                            
                            try:
                                for i, solution in enumerate(solutions):
                                    eval_result = self._evaluate_solution(
                                        lcb_id, language, model_name, solution, i, test_cases
                                    )
                                    evaluation_results.append(eval_result)
                                    
                                    logger.info(f"Solution {i}: {eval_result.passed_tests}/{eval_result.total_tests} tests passed "
                                              f"(pass rate: {eval_result.pass_rate:.2%})")
                            except Exception as eval_ex:
                                logger.error(
                                    f"Evaluation crashed for {model_name} {language} on {lcb_id}: {eval_ex} — "
                                    f"recording partial results as FAILED and continuing"
                                )
                                # Fill in FAILED placeholders for any solutions not yet evaluated
                                for i in range(len(evaluation_results), len(solutions)):
                                    evaluation_results.append(EvaluationResult(
                                        lcb_id=lcb_id,
                                        language=language,
                                        model_name=model_name,
                                        solution_index=i,
                                        passed_tests=0,
                                        total_tests=0,
                                        pass_rate=0.0,
                                        test_results=[],
                                        overall_result=ExecutionResult.RUNTIME_ERROR,
                                        total_execution_time=0.0,
                                    ))
                                gc.collect()
                            
                            # Calculate overall statistics for this model-language combination
                            total_solutions = len(evaluation_results)
                            solutions_passed = sum(1 for result in evaluation_results if result.overall_result == ExecutionResult.PASSED)
                            
                            # Calculate Pass@k for all specified k values
                            for k in self.k_values:
                                if k <= total_solutions:
                                    pass_at_k_results[f'pass_at_{k}'] = pass_at_k(total_solutions, solutions_passed, k)
                                else:
                                    pass_at_k_results[f'pass_at_{k}'] = 0.0
                            
                            # Legacy pass_at_1 for backward compatibility
                            pass_at_1 = pass_at_k_results.get('pass_at_1', 0.0)
                            
                            # Store evaluation results with round tracking for fixes
                            eval_key = f"{model_name}_{language}_evaluation"
                            
                            # Create current round evaluation data
                            current_round_data = {
                                'round_id': self.round_id,
                                'round_description': self.round_description,
                                'timestamp': time.time(),
                                'timeout': self.timeout,
                                'num_solutions': len(evaluation_results),
                                'solutions_passed': solutions_passed,
                                'pass_at_1': pass_at_1,
                                **pass_at_k_results,  # Include all Pass@k metrics
                                'evaluations': [
                                    {
                                        'solution_index': result.solution_index,
                                        'passed_tests': result.passed_tests,
                                        'total_tests': result.total_tests,
                                        'pass_rate': result.pass_rate,
                                    'overall_result': result.overall_result.value,
                                    'execution_time': result.total_execution_time,
                                    'test_results': [
                                        {
                                            'test_index': tr.test_index,
                                            'result': tr.result.value,
                                            'execution_time': tr.execution_time,
                                            'error_message': tr.error_message if tr.result != ExecutionResult.PASSED else ""
                                        }
                                        for tr in result.test_results
                                    ]
                                }
                                for result in evaluation_results
                            ]
                            }
                            
                            # Handle multi-round storage and track fixes
                            if eval_key in existing_data:
                                # Get existing evaluation data
                                existing_eval = existing_data[eval_key]
                                
                                # Convert to multi-round format if needed
                                if 'rounds' not in existing_eval:
                                    # Legacy format - convert to multi-round
                                    first_round_data = existing_eval.copy()
                                    first_round_data['round_id'] = first_round_data.get('round_id', 'initial_round')
                                    
                                    existing_data[eval_key] = {
                                        'model_name': model_name,
                                        'language': language,
                                        'latest_round': self.round_id,
                                        'total_rounds': 2,
                                        'rounds': {
                                            first_round_data['round_id']: first_round_data,
                                            self.round_id: current_round_data
                                        },
                                        'fixes_tracker': {}
                                    }
                                else:
                                    # Already multi-round format - add new round
                                    existing_eval['rounds'][self.round_id] = current_round_data
                                    existing_eval['latest_round'] = self.round_id
                                    existing_eval['total_rounds'] = len(existing_eval['rounds'])
                                
                                # Track fixes: compare with previous round to see what was fixed
                                self._track_evaluation_fixes(existing_data[eval_key], self.round_id, current_round_data)
                                
                            else:
                                # First evaluation - create multi-round structure
                                existing_data[eval_key] = {
                                    'model_name': model_name,
                                    'language': language,
                                    'latest_round': self.round_id,
                                    'total_rounds': 1,
                                    'rounds': {
                                        self.round_id: current_round_data
                                    },
                                    'fixes_tracker': {}  # Track which rounds fixed which issues
                                }
                            
                            # Save incrementally after each model-language combination
                            write_to_database(self.data_config.output_db, lcb_id, existing_data)
                            logger.info(f"Incrementally saved evaluation results for {model_name} {language} on {lcb_id}")
                            
                            evaluations_performed = True
                            logger.info(f"Completed evaluation of {model_name} {language} solutions for {lcb_id}")
                            
                            # Log Pass@k results for all k values
                            pass_at_k_log = ", ".join([f"Pass@{k.split('_')[-1]}: {score:.2%}" 
                                                     for k, score in pass_at_k_results.items()])
                            logger.info(f"{pass_at_k_log} ({solutions_passed}/{total_solutions} solutions passed)")
                            del evaluation_results
                
                    # After writing to disk, slim down the in-memory self.db entry so
                    # the full test-result arrays don't accumulate across all 175 problems.
                    # _should_evaluate_model_language only needs the round keys to exist,
                    # not the full per-test evaluation payloads.
                    if lcb_id in self.db:
                        slim = {}
                        for k, v in self.db[lcb_id].items():
                            if k.endswith('_evaluation') and isinstance(v, dict):
                                slim[k] = {
                                    'rounds': {r: True for r in v.get('rounds', {}).keys()},
                                }
                            else:
                                slim[k] = v
                        self.db[lcb_id] = slim
    
                    processed_count += 1
                finally:
                    if test_cases is not None:
                        test_cases.clear()
                    self._finalize_problem_memory(lcb_id, item)
            
            # Release references to the processed batch items (solution strings can be
            # several KB each; nulling them out lets the GC reclaim the memory before
            # the next batch starts).
            for _i in range(batch_start, batch_end):
                items_list[_i] = None

            # Memory management after each batch
            logger.info(f"Completed batch {batch_start//self.batch_size + 1}, forcing garbage collection...")
            gc.collect()  # Force garbage collection after each batch
            
            # Note: Incremental saving is already handled inside the language loop
        
        logger.info(f"Completed evaluation of {processed_count} items")
        
        # Calculate and log comprehensive metrics
        self._log_comprehensive_metrics()
        
        return
    
    def _log_comprehensive_metrics(self):
        """Calculate and log comprehensive evaluation metrics"""
        try:
            # Get all evaluation results from the database
            all_results = {}
            for lcb_id, data in self.db.items():
                if any(key.endswith('_evaluation') for key in data.keys()):
                    all_results[lcb_id] = data
            
            if not all_results:
                logger.info("No evaluation results found for comprehensive metrics calculation")
                return
            
            # Calculate comprehensive metrics
            metrics = calculate_comprehensive_metrics(all_results, self.k_values)
            
            logger.info("="*60)
            logger.info("COMPREHENSIVE EVALUATION METRICS")
            logger.info("="*60)
            
            # Log Pass@k metrics
            logger.info("Pass@k Metrics:")
            for k in self.k_values:
                pass_at_k_key = f'pass_at_{k}'
                if pass_at_k_key in metrics:
                    logger.info(f"  Pass@{k}: {metrics[pass_at_k_key]:.2%}")
            
            # Log summary statistics
            logger.info(f"Total Problems: {metrics['total_problems']}")
            logger.info(f"Total Solutions: {metrics['total_solutions']}")
            logger.info(f"Solutions Passed: {metrics['total_solutions_passed']}")
            logger.info(f"Overall Success Rate: {metrics['overall_success_rate']:.2%}")
            logger.info(f"Test Pass Rate: {metrics['test_pass_rate']:.2%}")
            
            # Log error distribution
            if metrics['error_distribution']:
                logger.info("Error Distribution:")
                for error_type, count in metrics['error_distribution'].items():
                    logger.info(f"  {error_type}: {count}")
            
            # Log execution time statistics
            exec_stats = metrics['execution_time_stats']
            if exec_stats['mean'] > 0:
                logger.info(f"Execution Time - Mean: {exec_stats['mean']:.3f}s, "
                          f"Median: {exec_stats['median']:.3f}s, "
                          f"Std: {exec_stats['std']:.3f}s")
            
            # Log difficulty breakdown
            if metrics['difficulty_breakdown']:
                logger.info("Performance by Difficulty:")
                for difficulty, stats in metrics['difficulty_breakdown'].items():
                    logger.info(f"  {difficulty}: Pass@1 = {stats['pass_at_1']:.2%} ({stats['count']} problems)")
            
            logger.info("="*60)
            
        except Exception as e:
            logger.error(f"Failed to calculate comprehensive metrics: {e}")
            import traceback
            traceback.print_exc()

    def _identify_fix_strategy(self, round_id: str, round_description: str) -> Dict[str, Any]:
        """Identify the fix strategy used in this round"""
        strategy = {
            'type': 'unknown',
            'parameters': {},
            'description': round_description
        }
        
        # Analyze round ID and description to identify strategy
        round_lower = round_id.lower()
        desc_lower = round_description.lower()
        
        if 'timeout' in round_lower or 'timeout' in desc_lower:
            strategy['type'] = 'timeout_increase'
            # Extract timeout value if mentioned
            import re
            timeout_match = re.search(r'(\d+)s?\s*timeout', desc_lower)
            if timeout_match:
                strategy['parameters']['timeout'] = int(timeout_match.group(1))
        
        elif 'memory' in round_lower or 'memory' in desc_lower:
            strategy['type'] = 'memory_increase'
            memory_match = re.search(r'(\d+)\s*(mb|gb)', desc_lower)
            if memory_match:
                value = int(memory_match.group(1))
                unit = memory_match.group(2)
                strategy['parameters']['memory'] = value * (1024 if unit == 'gb' else 1)
        
        elif 'compilation' in round_lower or 'compile' in desc_lower:
            strategy['type'] = 'compilation_fix'
            strategy['parameters']['approach'] = 'dependency_resolution'
        
        elif 'import' in desc_lower or 'dependency' in desc_lower:
            strategy['type'] = 'import_fix'
            strategy['parameters']['approach'] = 'import_enhancement'
        
        
        elif 'retry' in round_lower or 'retry' in desc_lower:
            strategy['type'] = 'retry_with_changes'
            strategy['parameters']['approach'] = 'environmental_adjustment'
        
        else:
            strategy['type'] = 'general_improvement'
        
        return strategy
    
    def _analyze_fix_type(self, previous_result: str, current_result: str, fix_strategy: Dict[str, Any], 
                         current_eval: Dict[str, Any], previous_eval: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Analyze the type of fix that occurred between evaluations"""
        
        # Define improvement levels
        MAJOR_FIX = 'major'      # Failed/Error → Passed
        MINOR_FIX = 'minor'      # Partial improvement
        LATERAL_FIX = 'lateral'  # Different error type
        REGRESSION = 'regression' # Got worse
        
        # Analyze the change
        if previous_result in ['failed', 'timeout', 'compilation_error', 'runtime_error'] and current_result == 'passed':
            # Major fix - complete resolution
            return {
                'change_type': 'fixed',
                'category': 'complete_resolution',
                'type': f"{previous_result}_to_passed",
                'improvement_level': MAJOR_FIX,
                'confidence': 0.95,
                'emoji': '✅',
                'description': f"FIXED: {previous_result} → {current_result}",
                'details': {
                    'test_improvement': current_eval.get('passed_tests', 0) - previous_eval.get('passed_tests', 0),
                    'execution_time_change': current_eval.get('execution_time', 0) - previous_eval.get('execution_time', 0),
                    'fix_strategy_effective': fix_strategy['type']
                }
            }
        
        elif previous_result != 'passed' and current_result != 'passed':
            # Check for partial improvements
            prev_passed = previous_eval.get('passed_tests', 0)
            curr_passed = current_eval.get('passed_tests', 0)
            
            if curr_passed > prev_passed:
                # Partial improvement
                return {
                    'change_type': 'improved',
                    'category': 'partial_improvement',
                    'type': f"{previous_result}_to_{current_result}_with_progress",
                    'improvement_level': MINOR_FIX,
                    'confidence': 0.7,
                    'emoji': '🔧',
                    'description': f"IMPROVED: {previous_result} → {current_result} (+{curr_passed - prev_passed} tests)",
                    'details': {
                        'test_improvement': curr_passed - prev_passed,
                        'total_tests': current_eval.get('total_tests', 0),
                        'progress_ratio': (curr_passed - prev_passed) / max(current_eval.get('total_tests', 1), 1),
                        'fix_strategy_effective': fix_strategy['type']
                    }
                }
            
            elif previous_result != current_result:
                # Lateral change (different error type)
                return {
                    'change_type': 'changed',
                    'category': 'error_type_change',
                    'type': f"{previous_result}_to_{current_result}",
                    'improvement_level': LATERAL_FIX,
                    'confidence': 0.5,
                    'emoji': '🔄',
                    'description': f"CHANGED: {previous_result} → {current_result}",
                    'details': {
                        'error_evolution': f"{previous_result} → {current_result}",
                        'might_indicate_progress': True,
                        'fix_strategy_attempted': fix_strategy['type']
                    }
                }
        
        elif previous_result == 'passed' and current_result != 'passed':
            # Regression
            return {
                'change_type': 'regressed',
                'category': 'performance_regression',
                'type': f"passed_to_{current_result}",
                'improvement_level': REGRESSION,
                'confidence': 0.9,
                'emoji': '⚠️',
                'description': f"REGRESSED: passed → {current_result}",
                'details': {
                    'regression_type': current_result,
                    'possible_cause': fix_strategy['type'],
                    'needs_investigation': True
                }
            }
        
        return None
    
    def _track_test_level_improvements(self, eval_data: Dict[str, Any], current_round_id: str, 
                                     current_round_data: Dict[str, Any], previous_round_data: Dict[str, Any],
                                     fix_strategy: Dict[str, Any]) -> None:
        """Track improvements at the test case level for detailed analysis"""
        test_improvements = []
        
        for current_eval in current_round_data.get('evaluations', []):
            sol_idx = current_eval['solution_index']
            
            # Find corresponding previous evaluation
            previous_eval = None
            for prev_eval in previous_round_data.get('evaluations', []):
                if prev_eval['solution_index'] == sol_idx:
                    previous_eval = prev_eval
                    break
            
            if previous_eval:
                curr_tests = current_eval.get('test_results', [])
                prev_tests = previous_eval.get('test_results', [])
                
                # Compare test by test
                for i, (curr_test, prev_test) in enumerate(zip(curr_tests, prev_tests)):
                    if prev_test.get('result') != 'passed' and curr_test.get('result') == 'passed':
                        test_improvements.append({
                            'solution_index': sol_idx,
                            'test_index': i,
                            'previous_result': prev_test.get('result'),
                            'current_result': curr_test.get('result'),
                            'execution_time_improvement': prev_test.get('execution_time', 0) - curr_test.get('execution_time', 0)
                        })
        
        if test_improvements:
            eval_data['fixes_tracker'][f"test_level_improvements_{current_round_id}"] = {
                'round': current_round_id,
                'fix_strategy': fix_strategy,
                'test_fixes': test_improvements,
                'total_test_fixes': len(test_improvements),
                'timestamp': current_round_data['timestamp']
            }
            logger.info(f"🔬 Test-level improvements: {len(test_improvements)} individual tests fixed")
    
    def _track_overall_improvements(self, eval_data: Dict[str, Any], current_round_id: str,
                                  current_round_data: Dict[str, Any], previous_round_data: Dict[str, Any],
                                  fix_strategy: Dict[str, Any]) -> None:
        """Track overall improvements with fix strategy attribution"""
        current_solutions_passed = current_round_data.get('solutions_passed', 0)
        previous_solutions_passed = previous_round_data.get('solutions_passed', 0)
        
        current_pass_at_1 = current_round_data.get('pass_at_1', 0.0)
        previous_pass_at_1 = previous_round_data.get('pass_at_1', 0.0)
        
        if current_solutions_passed != previous_solutions_passed or abs(current_pass_at_1 - previous_pass_at_1) > 0.01:
            improvement_key = f"round_{current_round_id}_overall_change"
            
            # Determine improvement category
            if current_solutions_passed > previous_solutions_passed:
                improvement_type = 'significant_improvement'
                emoji = '📈'
            elif current_solutions_passed < previous_solutions_passed:
                improvement_type = 'performance_regression'
                emoji = '📉'
            else:
                improvement_type = 'metric_fluctuation'
                emoji = '📊'
            
            eval_data['fixes_tracker'][improvement_key] = {
                'round': current_round_id,
                'previous_round': max(eval_data['rounds'].keys(), key=lambda x: eval_data['rounds'][x].get('timestamp', 0) if x != current_round_id else 0),
                'fix_strategy': fix_strategy,
                'improvement_type': improvement_type,
                'solutions_passed_before': previous_solutions_passed,
                'solutions_passed_after': current_solutions_passed,
                'solutions_change': current_solutions_passed - previous_solutions_passed,
                'pass_at_1_before': previous_pass_at_1,
                'pass_at_1_after': current_pass_at_1,
                'pass_at_1_change': current_pass_at_1 - previous_pass_at_1,
                'strategy_effectiveness': self._assess_strategy_effectiveness(fix_strategy, current_solutions_passed - previous_solutions_passed),
                'timestamp': current_round_data['timestamp']
            }
            
            logger.info(f"{emoji} Overall change: {previous_solutions_passed} → {current_solutions_passed} solutions passed "
                       f"(Pass@1: {previous_pass_at_1:.1%} → {current_pass_at_1:.1%})")
    
    def _assess_strategy_effectiveness(self, fix_strategy: Dict[str, Any], solutions_improvement: int) -> Dict[str, Any]:
        """Assess how effective the fix strategy was"""
        effectiveness = {
            'score': 0.0,
            'rating': 'ineffective',
            'confidence': 0.5
        }
        
        if solutions_improvement > 0:
            if fix_strategy['type'] == 'timeout_increase' and solutions_improvement >= 2:
                effectiveness = {'score': 0.8, 'rating': 'highly_effective', 'confidence': 0.9}
            elif fix_strategy['type'] == 'memory_increase' and solutions_improvement >= 1:
                effectiveness = {'score': 0.7, 'rating': 'effective', 'confidence': 0.8}
            elif fix_strategy['type'] == 'compilation_fix' and solutions_improvement >= 1:
                effectiveness = {'score': 0.9, 'rating': 'highly_effective', 'confidence': 0.95}
            elif fix_strategy['type'] == 'import_fix' and solutions_improvement >= 1:
                effectiveness = {'score': 0.85, 'rating': 'highly_effective', 'confidence': 0.9}
            elif solutions_improvement >= 1:
                effectiveness = {'score': 0.6, 'rating': 'moderately_effective', 'confidence': 0.7}
        elif solutions_improvement == 0:
            effectiveness = {'score': 0.3, 'rating': 'neutral', 'confidence': 0.6}
        else:
            effectiveness = {'score': 0.1, 'rating': 'counterproductive', 'confidence': 0.8}
        
        return effectiveness

    # Flexible fix tracking system - supports various fix strategies and detailed analysis 
