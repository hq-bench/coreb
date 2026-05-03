from easyllm_kit.utils.io_utils import initialize_database, write_to_database
from easyllm_kit.utils import get_logger, read_json
from easyllm_kit.models import LLM
from easyllm_kit.configs.llm_base_config import GenerationArguments
from coreb_runner.runners.base_runner import Runner
from coreb_runner.utils import DatasetColumns as DC
from coreb_runner.prompts import CodeGenPrompt

logger = get_logger('code_gen_runner', 'code_gen_runner.log')


@Runner.register("code_gen")
class CodeGenRunner(Runner):
    """
    Runner for generating code solutions to programming problems.
    
    Configuration options:
    - model: Configuration for the language model
    - generation: Configuration for generation parameters
    - data: Configuration for input/output data
    - code_gen: Additional configuration for code generation
      - languages: List of programming languages to generate code for (default: ['python'])
      - num_samples: Number of code samples to generate per problem (default: 1)
      - overwrite: Whether to overwrite existing solutions (default: False)
    
    Example configuration:
    {
        "model": { ... },
        "generation": { ... },
        "data": { ... },
        "code_gen": {
            "languages": ["python", "java", "cpp"],
            "num_samples": 3,
            "overwrite": true
        }
    }
    """

    def __init__(self, config):
        super().__init__()
        self.model_config = config["model"]

        # --- Handle generation config ------------------------------------
        raw_generation_cfg = config.get("generation", {})

        # Extract keys that are NOT part of GenerationArguments (e.g. ``language``)
        # ``language`` is frequently included in some configs but is not accepted by
        # GenerationArguments.  We remove it here and instead let the ``code_gen``
        # section (or a fallback) decide the languages for code generation.
        extra_language = raw_generation_cfg.pop("language", None)

        # Create GenerationArguments only with the supported keys
        self.generation_config = GenerationArguments(**raw_generation_cfg)

        # ------------------------------------------------------------------
        self.data_config = config["data"]
        self.config = config

        # Get code generation specific config
        code_gen_config = config.get("code_gen", {})
        self.languages = code_gen_config.get("languages")

        # If languages were not specified under ``code_gen`` we can fall back to
        # whatever we extracted from the generation config.
        if not self.languages:
            if extra_language is not None:
                # Accept either a single string or list provided in config
                self.languages = [extra_language] if isinstance(extra_language, str) else list(extra_language)
            else:
                self.languages = ["python"]

        # number of samples (default 1)
        self.num_samples = code_gen_config.get("num_samples", 1)

        logger.info(f"Code generation configured for languages: {self.languages}")
        logger.info(f"Will generate {self.num_samples} samples per problem")

        self.llm = self.setup_model()
        self.llm_name = self.llm.model_config.model_full_name

        # Initialize database
        self.db = initialize_database(self.data_config.output_db)

        # read codebench data
        self.codebench_data = read_json(self.data_config.data_dir)

    def initialize_db(self):
        """Return the current DB snapshot for skip-checking."""
        import dictdatabase as DDB
        return DDB.at(self.data_config.output_db).read() or {}

    def setup_model(self):
        # Build the LLM model
        llm_config = {'model_config': self.model_config,
                      'generation_config': self.generation_config}

        llm = LLM.build_from_config(llm_config)
        return llm

    def generate_code_for_language(self, sample, language):
        """Generate code solution for a specific programming language"""
        try:
            # Helper function to try multiple possible key names
            # Priority: ANNOTATED (uppercase) > annotated (lowercase) > original (no prefix)
            def get_field(sample, *keys):
                for key in keys:
                    if key in sample and sample[key]:
                        return sample[key]
                return ''
            
            # Create prompt for code generation
            # Tries: 1) Annotated fields first, 2) Original LCB fields as fallback
            # Note: Following LiveCodeBench methodology - NO public test cases in prompt
            prompt = CodeGenPrompt.init().format(
                question_title=get_field(sample, 'ANNOTATED_TITLE', 'annotate_question_title', 'question_title'),
                question_content=get_field(sample, 'ANNOTATED_QUESTION_CONTENT', 'annotate_question_content', 'question_content'),
                starter_code=get_field(sample, 'ANNOTATED_STARTER_CODE', 'annotate_starter_code', 'starter_code'),
                language=language
            )

            # Generate code samples
            generated_codes = []
            for _ in range(self.num_samples):
                response = self.llm.generate(prompt)
                generated_codes.append(response)

            return generated_codes

        except Exception as e:
            error_msg = f"Failed to generate {language} code: {str(e)}"
            logger.error(error_msg)
            return [f"# Error generating code: {error_msg}"]

    def process_one_sample(self, sample, languages_to_generate=None):
        """Process one sample and generate code solutions for specified languages"""
        try:
            # Log meta information
            question_id = sample.get('lcb_question_id', sample.get('question_id', 'unknown'))
            title = sample.get('annotate_question_title', sample.get('question_title', 'unknown'))
            logger.info(f"Processing sample: question_id={question_id}, "
                        f"title={title[:50]}...")

            # Use provided languages or fall back to all configured languages
            if languages_to_generate is None:
                languages_to_generate = self.languages

            # Generate code for each specified language
            generated_solutions = {}
            for language in languages_to_generate:
                logger.info(f"Generating {language} code...")
                solutions = self.generate_code_for_language(sample, language)
                generated_solutions[language] = solutions

            return generated_solutions

        except Exception as e:
            error_msg = f"Failed to process sample: {str(e)}"
            logger.error(error_msg)
            return {lang: [f"# Error: {error_msg}"] for lang in languages_to_generate or self.languages}

    def run(self):
        """Main execution loop - generates fresh solutions for all problems"""
        total_count = len(self.codebench_data)
        overwrite = self.config.get("code_gen", {}).get("overwrite", False)

        existing_db = self.initialize_db()

        for idx, (question_id, item) in enumerate(self.codebench_data.items(), 1):
            # Extract the real question_id from the item if available
            real_question_id = item.get('question_id', question_id)
            lcb_id = real_question_id if real_question_id.startswith('lcb_') else 'lcb_' + real_question_id
            logger.info(f"Processing {idx}/{total_count}: {lcb_id} (original: {real_question_id})")

            if not overwrite and lcb_id in existing_db:
                logger.info(f"Skipping {lcb_id} — already exists in DB (overwrite=False)")
                continue

            # Helper to get field from multiple possible key names
            def get_field(item, *keys):
                for key in keys:
                    if key in item and item[key]:
                        return item[key]
                return ''
            
            # Create response with problem metadata
            response = DC.create_dict(
                LCB_ID=lcb_id,
                LCB_DIFFICULTY=item.get('lcb_difficulty', item.get('difficulty', '')),
                ANNOTATED_TITLE=get_field(item, 'ANNOTATED_TITLE', 'annotate_question_title', 'question_title'),
                ANNOTATED_QUESTION_CONTENT=get_field(item, 'ANNOTATED_QUESTION_CONTENT', 'annotate_question_content', 'question_content'),
                ANNOTATED_STARTER_CODE=get_field(item, 'ANNOTATED_STARTER_CODE', 'annotate_starter_code', 'starter_code'))

            # Generate code solutions for all languages
            generated_solutions = self.process_one_sample(item, self.languages)

            # Add generated solutions to response
            for lang, solutions in generated_solutions.items():
                key = f"{self.llm_name}_{lang}"
                response[key] = solutions

            # Write to database
            write_to_database(self.data_config.output_db, lcb_id, response)
            logger.info(f"Successfully saved {lcb_id}")

        logger.info(f"Completed processing {total_count} items")
        return
