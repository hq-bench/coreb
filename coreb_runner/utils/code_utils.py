import json
import random
import re
import string
from typing import List, Dict, Any


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load JSONL file into list of dictionaries."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


# Matches a single line that contains only numeric tokens (integers, floats, scientific notation).
# Examples that match:  "5",  "-3 14",  "3.14 2.71",  "1e-6",  "-1.5e+3 0"
# Examples that fail:   "YES",  "Alice",  "3 apples"
_NUM_TOKEN = r'[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?'
line_pattern = re.compile(r'^' + _NUM_TOKEN + r'(?:\s+' + _NUM_TOKEN + r')*$')

# Standard competitive-programming output tokens that are context-independent and never
# need rewriting regardless of how the problem domain is changed.
_CP_STANDARD_OUTPUT = re.compile(
    r'^(Yes|No|YES|NO|yes|no|inf|Inf|INF|nan|NaN|Infinity|-Infinity|impossible|Impossible|'
    r'IMPOSSIBLE|possible|Possible|POSSIBLE|none|None|NONE|'
    r'true|false|True|False|TRUE|FALSE)$'
)

# Matches a line consisting entirely of standard grid/map characters used in competitive
# programming (e.g. maze, grid, board problems).  These are structural symbols with no
# domain-specific meaning and never need rewriting.
# Conservative set: . # * @ + (wall/empty/player markers) and S G E T X (Start/Goal/End)
_GRID_LINE = re.compile(r'^[.#*@+SGETXo ]+$')


def build_char_bijection(seed: str) -> dict:
    """
    Build a deterministic bijection over ASCII letters seeded by `seed` (e.g. problem_id).
    Uppercase letters are permuted among themselves; lowercase letters are permuted among
    themselves; all other characters are passed through unchanged.
    Used to programmatically annotate oversized string test cases without an LLM call.
    """
    rng = random.Random(seed)
    uppers = list(string.ascii_uppercase)
    lowers = list(string.ascii_lowercase)
    shuffled_uppers = uppers[:]
    shuffled_lowers = lowers[:]
    rng.shuffle(shuffled_uppers)
    rng.shuffle(shuffled_lowers)
    table = str.maketrans(
        ''.join(uppers) + ''.join(lowers),
        ''.join(shuffled_uppers) + ''.join(shuffled_lowers)
    )
    return table


def annotate_test_case_programmatic(tc: dict, char_table) -> dict:
    """
    Annotate a single test-case dict programmatically (no LLM) by applying a character
    bijection to both 'input' and 'output'.

    Strategy:
      - If the output is context-independent (numeric / grid / CP keyword), we CANNOT
        permute the input letters because the numeric output was computed from the original
        character set; permuting would make the stored output wrong.  Return tc unchanged.
      - Otherwise (output is also string-based), apply the same bijection to both input
        and output — this preserves all algorithmic relationships while changing the
        surface form enough to prevent memorisation.

    Known cases handled by this function:
      abc394_c (Debug): input and output are both long uppercase-letter strings (~3800 chars).
        The bijection is applied to both — the "debug" relationship between input and output
        is structural (positional edits), not character-value-dependent, so the bijection
        is safe.
      abc394_e (Palindromic Shortest Path): input is a character-labelled graph adjacency
        matrix (~2000 chars), output is a numeric distance matrix.  Output is CI → returned
        unchanged.  The random graph structure is effectively impossible to memorise.
    """
    out = tc.get('output', '')
    out_ci = _is_context_independent_output(out)
    if out_ci:
        # Numeric/grid output: input letters are structural graph/sequence labels whose
        # meaning is tied to the pre-computed numeric output.  Keeping as-is is safe
        # because large random graphs/sequences are impossible to memorise.
        return tc
    result = dict(tc)
    result['input']  = tc['input'].translate(char_table)
    result['output'] = tc['output'].translate(char_table)
    return result


def extract_code(model_output: str):
    """
    Extract code from model output that may contain either:
    1. Markdown code blocks with triple backticks: ```code```
    2. XML-like code tags: <code>code</code>
    3. Incomplete code blocks (missing closing tags) - will be fixed with warning
    
    Returns the extracted code string with comments removed, or the original string if no code blocks are found.
    """
    import logging
    logger = logging.getLogger(__name__)

    if model_output is None:
        return ""
    if not isinstance(model_output, str):
        model_output = str(model_output)

    # First, try to extract from complete <code></code> tags
    code_tag_pattern = re.compile(r'<code>(.*?)</code>', re.DOTALL)
    code_match = code_tag_pattern.search(model_output)
    if code_match:
        code_content = code_match.group(1).strip()
        return _remove_comments(code_content)
    
    # Try to extract from incomplete <code> tags (missing closing tag)
    incomplete_code_pattern = re.compile(r'<code>(.*?)$', re.DOTALL)
    incomplete_match = incomplete_code_pattern.search(model_output)
    if incomplete_match:
        code_content = incomplete_match.group(1).strip()
        logger.warning(f"Found incomplete <code> tag (missing </code>). Code length: {len(code_content)} chars")
        return _remove_comments(code_content)
    
    # Fall back to markdown code blocks with triple backticks
    outputlines = model_output.split("\n")
    indexlines = [i for i, line in enumerate(outputlines) if "```" in line]
    if len(indexlines) >= 2:
        code_content = "\n".join(outputlines[indexlines[0] + 1: indexlines[1]])
        return _remove_comments(code_content)
    elif len(indexlines) == 1:
        # Handle truncated code block (missing closing backticks)
        code_content = "\n".join(outputlines[indexlines[0] + 1:])
        logger.warning(f"Found truncated markdown code block (missing closing ```). Code length: {len(code_content)} chars")
        return _remove_comments(code_content)
    
    # If no code blocks found, return the original string with comments removed
    return _remove_comments(model_output.strip())


def _remove_comments(code: str) -> str:
    """
    Remove comments from code to avoid information leakage.
    Supports:
    - Python/Ruby: # comments
    - C++/Java/Go: // and /* */ comments
    - JavaScript: // and /* */ comments
    """
    if not code:
        return code
    
    lines = code.split('\n')
    cleaned_lines = []
    
    for line in lines:
        # Remove single-line comments
        # Python: # comments
        if '#' in line:
            # Check if # is inside a string (basic check)
            in_string = False
            quote_char = None
            i = 0
            while i < len(line):
                char = line[i]
                if char in ['"', "'"] and (i == 0 or line[i-1] != '\\'):
                    if not in_string:
                        in_string = True
                        quote_char = char
                    elif char == quote_char:
                        in_string = False
                        quote_char = None
                elif char == '#' and not in_string:
                    line = line[:i].rstrip()
                    break
                i += 1
        
        # C++/Java/Go/JavaScript: // comments
        if '//' in line:
            # Check if // is inside a string (basic check)
            in_string = False
            quote_char = None
            i = 0
            while i < len(line) - 1:
                if line[i] in ['"', "'"] and (i == 0 or line[i-1] != '\\'):
                    if not in_string:
                        in_string = True
                        quote_char = line[i]
                    elif line[i] == quote_char:
                        in_string = False
                        quote_char = None
                elif line[i:i+2] == '//' and not in_string:
                    line = line[:i].rstrip()
                    break
                i += 1
        
        cleaned_lines.append(line)
    
    # Join lines and remove multi-line comments /* */
    code_without_single_comments = '\n'.join(cleaned_lines)
    
    # Remove multi-line comments /* */
    # This is a simple approach - might not handle all edge cases
    multiline_pattern = re.compile(r'/\*.*?\*/', re.DOTALL)
    code_without_comments = multiline_pattern.sub('', code_without_single_comments)
    
    return code_without_comments.strip()


def extract_code_from_solution(solution: str) -> str:
    """
    Extract clean code from solution string (removes markdown formatting).
    This is a consolidated version used across the repository.
    
    Args:
        solution: Raw solution string that may contain markdown formatting
        
    Returns:
        Clean code string with markdown formatting removed
    """
    # Remove markdown code blocks
    code = re.sub(r'```\w*\n', '', solution)
    code = re.sub(r'```\n?', '', code)
    code = re.sub(r'<code>\n?', '', code)
    code = re.sub(r'</code>\n?', '', code)
    return code.strip()


def _is_context_independent_line(line: str) -> bool:
    """Return True if a single stripped line is context-independent for both input and output."""
    return bool(line_pattern.match(line) or _CP_STANDARD_OUTPUT.match(line) or _GRID_LINE.match(line))


def is_purely_numeric_input(input_str: str) -> bool:
    """
    Returns True if every non-empty line of input_str is context-independent:
    numeric tokens, standard CP keywords, or pure grid/map characters.
    """
    lines = [line.strip() for line in input_str.splitlines() if line.strip()]
    return all(_is_context_independent_line(line) for line in lines)


def _is_context_independent_output(output_str: str) -> bool:
    """
    Return True if every non-empty line of output_str is context-independent:
    numeric tokens, standard CP keywords, or pure grid/map characters.
    """
    lines = [l.strip() for l in output_str.splitlines() if l.strip()]
    return all(_is_context_independent_line(line) for line in lines)


def check_test_cases(test_cases_list):
    """
    Takes a list of test-case dicts (each with 'input' and 'output') and returns a list
    of booleans indicating whether each test case needs NO annotation.

    A test case can be kept unchanged when:
    - Its input is purely numeric (no domain-specific strings to replace), AND
    - Its output is either purely numeric OR a context-independent keyword like Yes/No/inf
      (these never change regardless of how the problem domain is rewritten).

    A test case MUST be annotated when its input contains domain-specific strings (names,
    objects, etc.) that need to be replaced to match the rewritten problem context.
    """
    results = []
    for case in test_cases_list:
        input_ok = is_purely_numeric_input(str(case.get('input', '')))
        output_ok = _is_context_independent_output(str(case.get('output', '')))
        results.append(input_ok and output_ok)
    return results


def check_all_test_cases(test_cases_decoded: dict):
    """
    Expects a dict with 'public_test_cases_decoded' and 'private_test_cases_decoded'
    (just like in your screenshot). Returns a dict of results for each.
    """
    return {
        'public': check_test_cases(test_cases_decoded['public_test_cases_decoded']),
        'private': check_test_cases(test_cases_decoded['private_test_cases_decoded'])
    }


# --------------------
# Example Usage
# --------------------

if __name__ == "__main__":
    # Simulate the structure from your screenshot
    test_cases_decoded = {
        "public_test_cases_decoded": [
            {"input": "2\n1 2\n-1 0\n", "output": "6.06...", "testtype": "stdin"},
            {"input": "7\n-14142 13562\n...", "output": "6.06...", "testtype": "stdin"},
            {"input": "5\n-100000 100000\n-100000\n...", "output": "384694.57...", "testtype": "stdin"},
        ],
        "private_test_cases_decoded": [
            {"input": "26619\n-472795666 103641861\n-944064245 740775144\n...", "testtype": "stdin"},
            # More items ...
        ]
    }

    results = check_all_test_cases(test_cases_decoded)
    print("Public test-case inputs purely numeric? ", results['public'])
    print("Private test-case inputs purely numeric?", results['private'])


# --------------------
# Shared solution key parsing utilities
# --------------------

def parse_solution_key(model_lang_key: str):
    """
    Parse a solution key of the form '{model}_{language}' into (model, language).
    Examples:
      - 'bedrock-claude-4-sonnet_python' -> ('bedrock-claude-4-sonnet', 'python')
      - 'o1-mini_python' -> ('o1-mini', 'python')
      - 'bedrock-claude-4-sonnet_cpp' -> ('bedrock-claude-4-sonnet', 'cpp')
    If the pattern does not match, returns (key, '').
    """
    if '_' not in model_lang_key:
        return model_lang_key, ''
    model, language = model_lang_key.rsplit('_', 1)
    return model, language


def create_solution_key(model: str, problem_id: str, language: str) -> str:
    """Create standardized solution key: '{model}_{problem_id}_{language}'."""
    return f"{model}_{problem_id}_{language}"
