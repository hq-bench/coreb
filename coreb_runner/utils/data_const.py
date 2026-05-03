from enum import Enum
from typing import Dict
from datasets import Features, Value
from dataclasses import dataclass


class LCBReleases(str, Enum):
    """
    Enum class for LiveCodeBench release versions.
    Maps release versions to their corresponding file names.
    """
    RELEASE_V1 = 'release_v1'
    RELEASE_V2 = 'release_v2'
    RELEASE_V3 = 'release_v3'
    RELEASE_V4 = 'release_v4'
    RELEASE_V5 = 'release_v5'
    RELEASE_V1_V2 = 'release_v1_v2'
    RELEASE_V2_V3 = 'release_v2_v3'
    RELEASE_V3_V4 = 'release_v3_v4'
    RELEASE_V4_V5 = 'release_v4_v5'  # Combined release

    @classmethod 
    def get_url(cls, release: str) -> str:
        """
        Get the URL for a specific release version.
        
        Args:
            release: The release version
            
        Returns:
            The URL for the release
        """
        mapping = {
            cls.RELEASE_V1: "https://huggingface.co/datasets/livecodebench/code_generation_lite/blob/main/test1.jsonl",
            cls.RELEASE_V1_V2: "https://huggingface.co/datasets/livecodebench/code_generation_lite/blob/main/test2.jsonl",
            cls.RELEASE_V2_V3: "https://huggingface.co/datasets/livecodebench/code_generation_lite/blob/main/test3.jsonl",
            cls.RELEASE_V3_V4: "https://huggingface.co/datasets/livecodebench/code_generation_lite/blob/main/test4.jsonl",
            cls.RELEASE_V4_V5: "https://huggingface.co/datasets/livecodebench/code_generation_lite/blob/main/test5.jsonl",
        }
        return mapping.get(release, "")
    
    @classmethod
    def get_file_name(cls, release: str) -> str:
        """
        Get the file name for a specific release version.
        
        Args:
            release: The release version
            
        Returns:
            The file name for the release
        """
        mapping = {
            cls.RELEASE_V1: "test1.jsonl",
            cls.RELEASE_V1_V2: "test2.jsonl",
            cls.RELEASE_V2_V3: "test3.jsonl",
            cls.RELEASE_V3_V4: "test4.jsonl",
            cls.RELEASE_V4_V5: "test5.jsonl",  # Points to the same file as RELEASE_V5
        }
        return mapping.get(release, "")

class ReleaseVersion(str, Enum):
    """
    Maps release versions to their short names.
    """
    RELEASE_V1 = 'release_v2501'
    RELEASE_V2 = 'release_v2505'
    RELEASE_V3 = 'release_v2510'
    
    @classmethod
    def to_short_name(cls, release: str) -> str:
        """Convert release version to short name"""
        mapping = {
            cls.RELEASE_V1: 'r1',
            cls.RELEASE_V2: 'r2',
            cls.RELEASE_V3: 'r3',
        }
        return mapping.get(release, release)
    
    @classmethod
    def from_short_name(cls, short_name: str) -> str:
        """Convert short name back to release version"""
        mapping = {
            'r1': cls.RELEASE_V1,
            'r2': cls.RELEASE_V2,
            'r3': cls.RELEASE_V3,
        }
        return mapping.get(short_name, short_name)

class DatasetColumns(str, Enum):
    """
    Enum class for dataset column names.
    Inherits from str to allow direct string comparison and usage.
    """
    INDEX = 'idx'
    QUESTION_ID = 'question_id'
    ANSWER = 'answer'
    LANGUAGE = 'language'
    RELEASE = 'release'
    ANNOTATE_QUESTION_TITLE = 'annotate_question_title'
    ANNOTATE_QUESTION_CONTENT = 'annotate_question_content'
    ANNOTATE_STARTER_CODE = 'annotate_starter_code'
    ANNOTATE_PUBLIC_TEST_CASES = 'annotate_public_test_cases'
    ANNOTATE_PRIVATE_TEST_CASES = 'annotate_private_test_cases'
    IS_CORRECT = 'is_correct'
    LCB_RELEASE = 'lcb_release'
    LCB_ID = 'lcb_question_id' # id in livecodebench
    LCB_DIFFICULTY = 'lcb_difficulty' # difficulty in livecodebench
    LCB_CONTEST_DATE = 'lcb_contest_date' # contest date in livecodebench
    LCB_QUESTION_TITLE = 'lcb_question_title' # question title in livecodebench
    LCB_QUESTION_CONTENT = 'lcb_question_content' # question content in livecodebench
    LCB_STARTER_CODE = 'lcb_starter_code' # starter code in livecodebench
    LCB_PUBLIC_TEST_CASES = 'lcb_public_test_cases' # public test cases in livecodebench
    LCB_PRIVATE_TEST_CASES = 'lcb_private_test_cases' # private test cases in livecodebench
    LCB_METADATA = 'lcb_metadata' # metadata in livecodebench
    LCB_TITLE = 'lcb_title' # title in livecodebench

    @classmethod
    def all_columns(cls) -> list[str]:
        """Returns list of all column names"""
        return [member.value for member in cls]

    @classmethod
    def create_dict(cls, **kwargs) -> dict:
        """
        Create a dictionary with string keys instead of enum objects.
        
        Example:
            DatasetColumns.create_dict(
                LCB_ID='value1', 
                ANSWER='value2'
            )
            
        Returns:
            {'lcb_question_id': 'value1', 'answer': 'value2'}
        """
        result = {}
        for key, value in kwargs.items():
            if hasattr(cls, key):
                enum_key = getattr(cls, key)
                result[enum_key.value] = value
            else:
                # Fall back to using the key directly if it's not an enum
                result[key] = value
        return result

    @classmethod
    def get_features(cls) -> Features:
        """
        Returns the features dictionary compatible with Hugging Face datasets.
        """
        features = {
            cls.INDEX: Value("int64"),
            cls.QUESTION_ID: Value("string"),
            cls.ANSWER: Value("string"),
            cls.LANGUAGE: Value("string"),
            cls.RELEASE: Value("string"),
            cls.ANNOTATE_QUESTION_TITLE: Value("string"),
            cls.ANNOTATE_QUESTION_CONTENT: Value("string"), 
            cls.ANNOTATE_STARTER_CODE: Value("string"),
            cls.ANNOTATE_PUBLIC_TEST_CASES: Value("string"),
            cls.ANNOTATE_PRIVATE_TEST_CASES: Value("string"),
            cls.IS_CORRECT: Value("bool"),
            cls.LCB_ID: Value("string"),
            cls.LCB_TITLE: Value("string"),
            cls.LCB_DIFFICULTY: Value("string"),
            cls.LCB_CONTEST_DATE: Value("string"),
            cls.LCB_QUESTION_TITLE: Value("string"),
            cls.LCB_QUESTION_CONTENT: Value("string"),
            cls.LCB_STARTER_CODE: Value("string"),
            cls.LCB_PUBLIC_TEST_CASES: Value("string"),
            cls.LCB_PRIVATE_TEST_CASES: Value("string"),
            cls.LCB_METADATA: Value("string")
        }
        return Features(features)

    @classmethod
    def validate_sample(cls, sample: Dict) -> bool:
        """
        Validates if a sample matches the expected schema.
        Returns True if valid, raises ValueError if invalid.
        """
        required_keys = {
            cls.INDEX, 
            cls.QUESTION_ID, 
            cls.ANSWER,
            cls.LANGUAGE,
            cls.RELEASE,
            cls.LCB_ID, 
            cls.LCB_TITLE, 
            cls.LCB_QUESTION_CONTENT,
            cls.LCB_CONTEST_DATE, 
            cls.LCB_STARTER_CODE, 
            cls.LCB_PUBLIC_TEST_CASES, 
            cls.LCB_PRIVATE_TEST_CASES,
            cls.ANNOTATE_QUESTION_TITLE, 
            cls.ANNOTATE_QUESTION_CONTENT,
            cls.ANNOTATE_STARTER_CODE
        }
        
        missing_keys = required_keys - set(sample.keys())
        if missing_keys:
            raise ValueError(f"Missing required keys: {missing_keys}")
            
        return True


# Define the desired language order
LANGUAGE_ORDER = {'python': 0, 
                  'java': 1, 
                  'cpp': 2,
                  'c': 3,
                  'javascript': 4,
                  'go': 5,
                  'ruby': 6,
                  'php': 7}


# ref: https://github.com/LiveCodeBench/LiveCodeBench/blob/998c52d394b836f15fff3b9a29866191108ff81b/lcb_runner/benchmarks/code_generation.py#L53
class LCBTestType(Enum):
    STDIN = "stdin"
    FUNCTIONAL = "functional"


@dataclass
class LCBTest:
    input: str
    output: str
    testtype: LCBTestType

    def __post_init__(self):
        self.testtype = LCBTestType(self.testtype)
