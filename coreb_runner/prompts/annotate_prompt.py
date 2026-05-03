from easyllm_kit.utils import PromptTemplate


class ValidateTestCasesPrompt(PromptTemplate):
    @classmethod
    def init(cls):
        _template = """# Test Case Validation Task

        ## Task Description
        You are tasked with analyzing test cases to determine if they are purely numerical. A test case is considered purely numerical if it contains only:
        - Numbers (integers, floats)
        - Basic operators (+, -, *, /, etc.)
        - Data structures (arrays, lists, etc.)
        - No domain-specific terms or text

        ## Examples

        Purely Numerical Test Cases:
        1. "[1, 2, 3] → 6"
        2. "5 + 10 = 15"
        3. "[[1,2], [3,4]] → [[4,6], [6,8]]"
        4. "nums=[1,2,3,4,5], target=3 → 2"

        Non-Numerical Test Cases:
        1. "books=['novel','textbook'] → 2"
        2. "students=['Alice','Bob'] → 2"
        3. "items=['hammer','wrench'] → 2"
        4. "words=['hello','world'] → 2"

        ## Your Task
        Analyze the following test cases and determine if they are purely numerical:

        Public Test Cases:
        ```
        {{public_test_cases}}
        ```

        Private Test Cases:
        ```
        {{private_test_cases}}
        ```

        Please provide your analysis in JSON format with the following structure:
        ```json
        {
          "is_public_purely_numerical": true/false,
          "is_private_purely_numerical": true/false,
        }
        ```

        Note: Be conservative in your assessment. If there's any doubt about whether a test case is purely numerical, mark it as false.
        """

        return cls(
            template=_template,
            input_variables=["public_test_cases", "private_test_cases"]
        )


class TestCaseAnnotatePrompt(PromptTemplate):
    @classmethod
    def init(cls):
        _template = """# Test Case Annotation Task

        ## Task Description
        You are tasked with transforming test cases for a coding problem while maintaining their functional equivalence.
        The original problem has already been transformed, and you need to adapt the test cases to match.

        ## Original Problem
        Title: {{original_title}}
        
        ## Transformed Problem
        Title: {{transformed_title}}
        Content: {{transformed_content}}
        
        ## Important Guidelines
        1. Replace domain-specific terms in the test cases to match the transformed problem's context
        2. PRESERVE the exact same algorithmic structure and complexity
        3. Maintain the same input/output patterns and edge cases
        4. Example: If you changed "count books on shelf" to "count tools in box", then
           "books=['novel','textbook'] → 2" becomes "tools=['hammer','wrench'] → 2"

        Test cases must:
        - Remain syntactically correct in the target language
        - Test exactly the same edge cases and functionality
        - Have the same expected outputs for equivalent inputs

        {% if public_test_cases %}
        ## Public Test Cases to Transform
        ```
        {{public_test_cases}}
        ```
        {% endif %}

        {% if private_test_cases %}
        ## Private Test Cases to Transform
        ```
        {{private_test_cases}}
        ```
        {% endif %}
        
        ## Your Response
        Please provide your transformed test cases in JSON format:
        
        ```json
        {
          {% if public_test_cases %}
          "annotate_public_test_cases": <transformed_public_test_cases>
          {% endif %}
          {% if private_test_cases and public_test_cases %}
          ,
          {% endif %}
          {% if private_test_cases %}
          "annotate_private_test_cases": <transformed_private_test_cases>
          {% endif %}
        }
        ```
        """

        return cls(
            template=_template,
            input_variables=["original_title", "transformed_title", "transformed_content", 
                            "public_test_cases", "private_test_cases"]
        )


class AnnotatePrompt(PromptTemplate):
    @classmethod
    def init(cls):
        _template = """# Code Problem Annotation Task

        ## Task Description
        You are tasked with transforming a coding problem into a counterfactual version to avoid data contamination. Your goal is to preserve the core algorithmic challenge without changing the output while changing superficial details.

        ## Transformation Instructions
        Please create a counterfactual version of this problem by applying these transformations:

        1. **Named Entity Replacement**: 
        - Replace all proper nouns, character names, company names, etc.
        - Example: "Alice wants to sort her books" → "Marcus needs to organize his collection"

        2. **Domain/Context Shifting**:
        - Change the problem domain while keeping the algorithmic challenge identical
        - Example: "Calculate profit from stock trades" → "Determine score changes in a game tournament"

        3. **Noun Phrase Substitution**:
        - Replace key objects/items with different but functionally equivalent ones
        - Example: "array of integers" can stay the same, but "list of books" → "array of products"

        4. **Synonym Replacement**:
        - Replace verbs and adjectives with synonyms
        - Example: "maximize profit" → "optimize earnings"

        5. **Variable/Function Name Changes**:
        - If example code is provided, rename variables and functions
        - Example: `calculateSum()` → `computeTotal()`

        ## Important Guidelines
        - Preserve the exact same algorithmic challenge and difficulty
        - Maintain the same input/output structure and constraints
        - Keep the same time/space complexity requirements
        - Ensure the transformed problem requires the same solution approach
        - The starter code should remain syntactically correct and functionally equivalent

        ## Test Case Guidelines
        When handling test cases, follow these strict rules:
        1. For purely numerical test cases (containing only numbers, basic operators, and data structures):
           - DO NOT MODIFY them at all - keep them exactly as they are
           - Example: Leave "[1, 2, 3] → 6" or "5 + 10 = 15" unchanged
        
        2. For non-numerical test cases (containing domain-specific terms):
           - Make MINIMAL changes necessary to match your transformed problem context
           - PRESERVE the exact same algorithmic structure and complexity
           - Maintain the same input/output patterns and edge cases
           - Example: If you changed "count books on shelf" to "count tools in box", then
             "books=['novel','textbook'] → 2" becomes "tools=['hammer','wrench'] → 2"

        3. ALL test cases must:
           - Remain syntactically correct in the target language
           - Test exactly the same edge cases and functionality
           - Have the same expected outputs for equivalent inputs

        ## Your Counterfactual Version
          
        Given the original problem:
        Title: {{question_title}}
        
        Content:
        {{question_content}}
        
        Starter Code:
        ```
        {{starter_code}}
        ```

        {% if public_test_cases %}
        {% if public_test_cases is string and public_test_cases.strip() %}
        Public Test Cases:
        ```
        {{public_test_cases}}
        ```
        {% elif public_test_cases is not string and public_test_cases %}
        Public Test Cases:
        ```
        {{public_test_cases}}
        ```
        {% endif %}
        {% endif %}

        {% if private_test_cases %}
        {% if private_test_cases is string and private_test_cases.strip() %}
        Private Test Cases:
        ```
        {{private_test_cases}}
        ```
        {% elif private_test_cases is not string and private_test_cases %}
        Private Test Cases:
        ```
        {{private_test_cases}}
        ```
        {% endif %}
        {% endif %}

        Please provide your transformed version below in JSON format with the following structure:
        ```json
        {
          "annotate_question_title": "<transformed_title>", 
          "annotate_question_content": "<transformed_content>", 
          "annotate_starter_code": "<transformed_starter_code>"
          {% if public_test_cases %}
          {% if public_test_cases is string and public_test_cases.strip() or public_test_cases is not string and public_test_cases %}
          ,"annotate_public_test_cases": "<transformed_public_test_cases>"
          {% else %}
          ,"annotate_public_test_cases": ""
          {% endif %}
          {% else %}
          ,"annotate_public_test_cases": ""
          {% endif %}
          {% if private_test_cases %}
          {% if private_test_cases is string and private_test_cases.strip() or private_test_cases is not string and private_test_cases %}
          ,"annotate_private_test_cases": "<transformed_private_test_cases>"
          {% else %}
          ,"annotate_private_test_cases": ""
          {% endif %}
          {% else %}
          ,"annotate_private_test_cases": ""
          {% endif %}
        }
        ```

        Note: Ensure that your transformed version preserves all the algorithmic details while changing the superficial context. The code should remain valid and compilable.
        """

        return cls(
                template=_template,
                input_variables=["question_title", "question_content", "starter_code",
                                 "public_test_cases", "private_test_cases"]
            )
    