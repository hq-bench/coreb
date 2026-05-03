"""
Question Abbreviation Runner

Rewrites each problem's natural-language description into a concise, retrieval-friendly
abbreviation. Uses lightweight heuristics by default; optionally supports an LLM pass.
"""

import re
from typing import Dict, Any

from coreb_runner.runners.base_runner import Runner
from coreb_runner.prompts.query_gen_prompt import QuestionContentAbbreviatePrompt
from easyllm_kit.utils import get_logger, read_json, extract_json_from_text
from easyllm_kit.utils.io_utils import initialize_database, write_to_database
from easyllm_kit.models import LLM
from easyllm_kit.configs.llm_base_config import GenerationArguments

logger = get_logger('question_abbrev_runner', 'question_abbrev_runner.log')


def _extract_first_sentence(text: str) -> str:
    if not text:
        return ''
    # Split on sentence terminators, fallback to first 160 chars if no period
    m = re.split(r'[.!?]\s+', text.strip(), maxsplit=1)
    return (m[0] if m else text.strip())[:280]


def _remove_examples(text: str) -> str:
    if not text:
        return ''
    # Heuristic: drop sections starting with "Example" or "Examples" until a blank line
    lines = text.splitlines()
    out = []
    skip = False
    for line in lines:
        if re.match(r'^\s*Examples?:', line, flags=re.IGNORECASE):
            skip = True
            continue
        if skip:
            if line.strip() == '':
                skip = False
            continue
        out.append(line)
    return '\n'.join(out)


def _extract_constraints(text: str) -> str:
    if not text:
        return ''
    # Extract lines under a "Constraints" header if present
    constraints = []
    lines = text.splitlines()
    in_constraints = False
    for line in lines:
        if re.match(r'^\s*Constraints?:', line, flags=re.IGNORECASE):
            in_constraints = True
            continue
        if in_constraints:
            if line.strip() == '':
                break
            constraints.append(line.strip())
    if constraints:
        return 'Constraints: ' + ' '.join(constraints)
    return ''


def _abbreviate_problem(title: str, description: str, max_len: int) -> str:
    """Produce a compact abstraction: one-sentence goal + constraints."""
    if not description:
        description = ''
    # Remove example sections for brevity
    desc_no_ex = _remove_examples(description)
    # First sentence as task goal
    goal = _extract_first_sentence(desc_no_ex)
    # Add constraints if available
    constraints = _extract_constraints(description)
    pieces = []
    if title:
        pieces.append(f"Title: {title}")
    if goal:
        pieces.append(f"Goal: {goal}")
    if constraints:
        pieces.append(constraints)
    abbreviated = ' | '.join(pieces).strip()
    if len(abbreviated) > max_len:
        abbreviated = abbreviated[:max_len].rstrip() + '…'
    return abbreviated


def _abbreviate_with_llm(title: str, description: str, llm, prompt_template: QuestionContentAbbreviatePrompt) -> str:
    """Use LLM to create a refined abbreviation."""
    try:
        prompt = prompt_template.format(question_title=title, question_content=description)
        response = llm.generate(prompt)
        parsed_response = extract_json_from_text(response)
        return parsed_response.get('abbreviated_content', description)
    except Exception as e:
        logger.warning(f"LLM abbreviation failed: {e}, falling back to heuristic")
        return _abbreviate_problem(title, description, 600)


@Runner.register("question_abbrev")
class QuestionAbbrevRunner(Runner):
    """Generate abbreviated problem statements for efficient retrieval."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.data_config = config.get('data', {})
        self.model_config = config.get('model', {})
        self.generation_config = GenerationArguments(**config.get('generation', {}))

        # Params
        self.llm = self.setup_model()
        self.llm_name = self.llm.model_config.model_full_name

        # init database
        self.output_db_name = self.data_config.get('output_db_name')
        self.db = initialize_database(self.output_db_name)

        # read codebench data
        self.annotated_codebench_data = read_json(self.data_config.get('data_dir'))

        # LLM setup
        self.prompt_template = QuestionContentAbbreviatePrompt.init()

    def setup_model(self):
        # Build the LLM model
        llm_config = {'model_config': self.model_config,
                      'generation_config': self.generation_config}

        llm = LLM.build_from_config(llm_config)

        return llm

    def run(self):
        logger.info(f"🚀 Generating abbreviated questions from {self.data_config.get('data_dir')}")
        logger.info(f"Loaded {len(self.annotated_codebench_data)} problems from source")

        total = len(self.annotated_codebench_data)
        skipped = 0
        generated = 0

        for problem_id, obj in self.annotated_codebench_data.items():
            # Check if already exists
            if self.db.get(problem_id) is not None:
                logger.debug(f"⏭️  Skipping {problem_id} (already exists)")
                skipped += 1
                continue

            title = obj.get('ANNOTATED_TITLE') or obj.get('annotate_question_title')
            description = obj.get('ANNOTATED_QUESTION_CONTENT') or obj.get('annotate_question_content')

            if not title or not description:
                logger.warning(f"⚠️  Skipping {problem_id} - missing title or description")
                continue

            # Use LLM to create abbreviation
            abbreviated = _abbreviate_with_llm(title, description, self.llm, self.prompt_template)

            result = {
                'problem_id': problem_id,
                'title': title,
                'abbreviated': abbreviated,
                'method': self.llm_name,
            }

            write_to_database(self.output_db_name, problem_id, result)
            generated += 1
            logger.info(f"✅ Generated abbreviation for {problem_id} ({generated}/{total - skipped} new)")

        logger.info(f"🏁 Completed: {generated} generated, {skipped} skipped, {total} total")