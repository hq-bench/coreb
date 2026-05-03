import json
import pickle
import zlib
import base64

from easyllm_kit.utils.io_utils import initialize_database, write_to_database
from easyllm_kit.utils import get_logger, read_json
from easyllm_kit.models import LLM
from easyllm_kit.configs.llm_base_config import GenerationArguments
from coreb_runner.runners.base_runner import Runner
from coreb_runner.utils import DatasetColumns as DC, generate_response_from_llm, safe_parse_response, check_all_test_cases
from coreb_runner.utils.code_utils import build_char_bijection, annotate_test_case_programmatic
from coreb_runner.prompts import AnnotatePrompt, TestCaseAnnotatePrompt

logger = get_logger('annotate_runner')


@Runner.register("annotation")
class AnnotateRunner(Runner):
    """
    Runner for annotating coding problems.
    
    Configuration options:
    - model: Configuration for the language model
    - generation: Configuration for generation parameters
    - data: Configuration for input/output data
    - annotation: Additional configuration for annotation process
      - max_test_case_length: Maximum length of test cases in characters (default: 1000)
        Test cases exceeding this length will be dropped.
      - overwrite: Whether to overwrite existing annotations (default: False)
    
    Example configuration:
    {
        "model": { ... },
        "generation": { ... },
        "data": { ... },
        "annotation": {
            "max_test_case_length": 1500,
            "overwrite": true
        }
    }
    """
    def __init__(self, config):
        # Initialize Runner without passing config to super().__init__
        # since the base Runner class doesn't have parameters in its __init__
        
        self.model_config = config["model"]
        self.generation_config = GenerationArguments(**config.get('generation', {}))
        self.data_config = config["data"]
        self.config = config

        # Get annotation configuration
        annotation_config = config.get("annotation", {})
        self.max_test_case_length = annotation_config.get("max_test_case_length", 5000)
        self.overwrite = annotation_config.get("overwrite", False)
        self.max_problems = annotation_config.get("max_problems", None)  # None = all

        logger.info(f"Maximum test case length configured to: {self.max_test_case_length} characters")
        logger.info(f"Overwrite mode: {self.overwrite}")
        if self.max_problems:
            logger.info(f"Dry-run mode: processing at most {self.max_problems} problems")
        
        self.llm = self.setup_model()
        self.llm_name = self.llm.model_config.model_full_name

        # init database
        self.db = initialize_database(self.data_config.output_db)

        # read codebench data
        self.codebench_data = read_json(self.data_config.data_dir)

    def setup_model(self):
        # Build the LLM model
        llm_config = {'model_config': self.model_config,
                      'generation_config': self.generation_config}

        llm = LLM.build_from_config(llm_config)

        return llm

    def filter_test_cases_by_length(self, test_cases, max_length=None):
        """Filter out test cases that are too long"""
        # Use configured max length if none is provided
        if max_length is None:
            max_length = self.max_test_case_length
            
        filtered_cases = []
        dropped_count = 0
        
        for idx, test_case in enumerate(test_cases):
            # Convert test case to string to measure its length
            test_case_str = str(test_case)
            
            if len(test_case_str) <= max_length:
                filtered_cases.append(test_case)
            else:
                dropped_count += 1
                logger.warning(f"Dropping test case {idx} due to excessive length: {len(test_case_str)} characters (max: {max_length})")
                
        if dropped_count > 0:
            logger.info(f"Dropped {dropped_count} test cases out of {len(test_cases)} due to length constraints (max: {max_length})")
            
        return filtered_cases

    def _classify_test_cases(self, test_cases: list) -> list:
        """
        Return a list of booleans (one per test case): True means the test case is
        context-independent (numerical input + numeric/Yes/No output) and needs no
        LLM annotation; False means it contains domain-specific strings and must be
        rewritten.  Pure regex — no LLM call.
        """
        if not test_cases:
            return []
        result = check_all_test_cases({
            'public_test_cases_decoded': test_cases,
            'private_test_cases_decoded': []
        })
        return result['public']  # list[bool], True = can skip

    def _annotate_test_cases_batched(
        self, test_cases: list, flags: list, sample: dict, annotated: dict,
        batch_size: int, is_public: bool
    ) -> list:
        """
        Annotate test cases that need rewriting (flag=False), keep the rest unchanged.
        `flags` is the pre-computed output of _classify_test_cases — passed in to avoid
        recomputing the regex check a second time.
        Returns the full list in original order — no test cases are ever dropped.

        A test case is skipped (kept as-is) only when it is context-independent
        (numerical / grid / standard CP keyword).

        Raises ValueError if a non-skippable test case exceeds max_test_case_length —
        the problem must be investigated manually before annotation can proceed.
        """
        total = len(test_cases)
        label = "public" if is_public else "private"

        # Decide per-case: needs LLM annotation, skip (CI), or annotate programmatically.
        # Test cases that exceed max_test_case_length are annotated without LLM using a
        # deterministic character bijection (see annotate_test_case_programmatic):
        #   - If output is context-independent (numeric/grid): keep input+output unchanged.
        #     Large random graphs/sequences are impossible to memorise, so the annotation
        #     benefit is minimal and permuting input letters would invalidate the output.
        #   - If output is also string-based: apply a letter bijection to both input and
        #     output, preserving all algorithmic relationships while changing surface form.
        # Start with originals; overwritten in-place for programmatic and LLM annotations
        result = list(test_cases)

        skip_numerical   = sum(1 for ok in flags if ok)
        prog_annotated   = 0
        to_annotate      = []
        char_table       = None  # built lazily if needed
        for i, (tc, ok) in enumerate(zip(test_cases, flags)):
            if ok:
                continue  # context-independent, skip
            tc_len = len(str(tc))
            if tc_len > self.max_test_case_length:
                if char_table is None:
                    char_table = build_char_bijection(sample.get('question_id', str(i)))
                result[i] = annotate_test_case_programmatic(tc, char_table)
                prog_annotated += 1
                logger.info(f"Programmatically annotated {label} test case {i} "
                            f"({tc_len} chars, too long for LLM)")
                continue
            to_annotate.append((i, tc))

        if skip_numerical or prog_annotated:
            logger.info(f"{label.capitalize()} test cases: {skip_numerical}/{total} skipped "
                        f"(numerical/CI), {prog_annotated} programmatically annotated, "
                        f"{len(to_annotate)} sent to LLM")

        if not to_annotate:
            return result

        # Batch only the cases that need annotation
        indices, cases_to_annotate = zip(*to_annotate)
        cases_to_annotate = list(cases_to_annotate)

        for start in range(0, len(cases_to_annotate), batch_size):
            batch = cases_to_annotate[start:start + batch_size]
            batch_indices = indices[start:start + batch_size]
            end = start + len(batch)
            logger.info(f"Annotating {label} test cases batch [{start}:{end}] of {len(cases_to_annotate)}")
            try:
                prompt_batch = TestCaseAnnotatePrompt.init().format(
                    original_title=sample['question_title'],
                    transformed_title=annotated['annotate_question_title'],
                    transformed_content=annotated['annotate_question_content'],
                    public_test_cases=str(batch) if is_public else '',
                    private_test_cases='' if is_public else str(batch)
                )
                resp = generate_response_from_llm(self.llm, prompt_batch)
                parsed = safe_parse_response(resp)
                key = 'annotate_public_test_cases' if is_public else 'annotate_private_test_cases'
                annotated_batch = parsed.get(key, batch)
                for orig_idx, ann_tc in zip(batch_indices, annotated_batch):
                    orig_tc = result[orig_idx]
                    # Restore trailing \n on output/input if the original had one.
                    # LLMs silently strip trailing newlines during rewriting.
                    if isinstance(ann_tc, dict) and isinstance(orig_tc, dict):
                        for field in ('input', 'output'):
                            orig_val = orig_tc.get(field, '')
                            ann_val  = ann_tc.get(field, '')
                            if isinstance(orig_val, str) and isinstance(ann_val, str):
                                if orig_val.endswith('\n') and not ann_val.endswith('\n'):
                                    ann_tc[field] = ann_val + '\n'
                    result[orig_idx] = ann_tc
            except Exception as e:
                error_msg = f"Error annotating {label} test cases batch [{start}:{end}]: {e}"
                logger.warning(error_msg)
                for orig_idx, tc in zip(batch_indices, batch):
                    tc_copy = tc.copy() if isinstance(tc, dict) else tc
                    if isinstance(tc_copy, dict):
                        tc_copy['annotation_error'] = error_msg
                    result[orig_idx] = tc_copy

        return result

    def annotate_one_sample(self, sample):
        try:
            pub_tc = sample['public_test_cases_decoded']
            priv_tc = sample['private_test_cases_decoded']

            logger.info(f"Processing sample: question_id={sample.get('question_id', 'unknown')}, "
                        f"title={sample.get('question_title', 'unknown')[:50]}..., "
                        f"public={len(pub_tc)}, private={len(priv_tc)}")

            # Classify all test cases once with pure regex — no LLM.
            # True = context-independent (skip), False = needs rewriting.
            pub_flags = self._classify_test_cases(pub_tc)
            priv_flags = self._classify_test_cases(priv_tc)

            pub_all_skip = all(pub_flags)
            priv_all_skip = all(priv_flags)
            logger.info(f"Test case classification — "
                        f"public: {sum(pub_flags)}/{len(pub_flags)} skip, "
                        f"private: {sum(priv_flags)}/{len(priv_flags)} skip")

            # Include a few non-skippable public test cases in the content prompt so the
            # LLM sees which domain-specific terms need to be transformed.
            pub_examples = [tc for tc, ok in zip(pub_tc, pub_flags) if not ok][:3]
            pub_for_prompt = '' if pub_all_skip else str(pub_examples)

            prompt = AnnotatePrompt.init().format(
                question_title=sample['question_title'],
                question_content=sample['question_content'],
                starter_code=sample['starter_code'],
                public_test_cases=pub_for_prompt,
                private_test_cases=''
            )
            response = generate_response_from_llm(self.llm, prompt)
            annotated = safe_parse_response(response)

            required_fields = ['annotate_question_title', 'annotate_question_content', 'annotate_starter_code']
            missing = [f for f in required_fields if f not in annotated]
            if missing:
                raise ValueError(f"LLM response missing required fields: {missing}. Raw response: {response[:200]}")

            # If original starter code is empty, force annotated to empty too.
            # o3-mini (and other reasoning models) tend to hallucinate placeholder
            # code when the source problem has no starter code.
            if not sample.get('starter_code', '').strip():
                annotated['annotate_starter_code'] = ''

            # Default: keep originals; overwrite only positions that need annotation.
            annotated['annotate_public_test_cases'] = pub_tc
            annotated['annotate_private_test_cases'] = priv_tc

            if not pub_all_skip and pub_tc:
                try:
                    annotated['annotate_public_test_cases'] = self._annotate_test_cases_batched(
                        pub_tc, pub_flags, sample, annotated, batch_size=10, is_public=True)
                except Exception as e:
                    logger.error(f"Failed to annotate public test cases: {e}")

            if not priv_all_skip and priv_tc:
                try:
                    annotated['annotate_private_test_cases'] = self._annotate_test_cases_batched(
                        priv_tc, priv_flags, sample, annotated, batch_size=5, is_public=False)
                except Exception as e:
                    logger.error(f"Failed to annotate private test cases: {e}")

            return annotated

        except Exception as e:
            error_msg = f"Failed to annotate sample: {str(e)}"
            logger.error(error_msg)
            return {
                'annotate_question_title': sample['question_title'],
                'annotate_question_content': sample['question_content'] + f"\n\n[ANNOTATION ERROR: {error_msg}]",
                'annotate_starter_code': sample['starter_code'],
                'annotate_public_test_cases': sample['public_test_cases_decoded'],
                'annotate_private_test_cases': sample['private_test_cases_decoded'],
                'annotation_error': error_msg
            }

    def run(self):
        processed = 0
        for _, item in self.codebench_data.items():
            if self.max_problems and processed >= self.max_problems:
                logger.info(f"Reached max_problems={self.max_problems}, stopping.")
                break
            # Check if this item is already in the database
            lcb_id = 'lcb_' + item['question_id']
            existing_data = self.db.get(lcb_id, {})
            
            if existing_data and not self.overwrite:
                logger.info(f"Skipping item {lcb_id} as it already exists and overwrite is False.")
                continue

            item['public_test_cases_decoded'] = json.loads(item['public_test_cases'])
            item['private_test_cases_decoded'] = json.loads(
                pickle.loads(
                    zlib.decompress(
                        base64.b64decode(item['private_test_cases'].encode("utf-8"))  # type: ignore
                    )
                )
            )

            # Create response using the helper method
            response = DC.create_dict(
                LCB_ID=lcb_id,
                LCB_DIFFICULTY=item['difficulty'],
                LCB_TITLE=item['question_title'],
                LCB_QUESTION_CONTENT=item['question_content'],
                LCB_CONTEST_DATE=item['contest_date'],
                LCB_STARTER_CODE=item['starter_code'],
                LCB_PUBLIC_TEST_CASES=item['public_test_cases_decoded'],
                LCB_PRIVATE_TEST_CASES=item['private_test_cases_decoded'],
                ANNOTATE_QUESTION_TITLE=item['question_title'],
                ANNOTATE_QUESTION_CONTENT=item['question_content']
            )
            
            # Generate annotated versions
            annotate_response = self.annotate_one_sample(item)
            
            # Update with annotated content
            response.update({
                DC.ANNOTATE_QUESTION_TITLE.value: annotate_response['annotate_question_title'],
                DC.ANNOTATE_QUESTION_CONTENT.value: annotate_response['annotate_question_content'],
                DC.ANNOTATE_STARTER_CODE.value: annotate_response['annotate_starter_code'],
                DC.ANNOTATE_PUBLIC_TEST_CASES.value: annotate_response['annotate_public_test_cases'],
                DC.ANNOTATE_PRIVATE_TEST_CASES.value: annotate_response['annotate_private_test_cases']
            })

            # Write to database
            write_to_database(self.data_config.output_db, lcb_id, response)
            processed += 1

        return
