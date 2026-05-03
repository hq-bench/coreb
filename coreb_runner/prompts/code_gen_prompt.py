from easyllm_kit.utils import PromptTemplate

class CodeGenPrompt(PromptTemplate):
    @classmethod
    def init(cls):
        _template = """
You are an expert competitive-programming assistant.

Task
----
Write an *entire*, *runnable* program in **{{language}}** that solves the problem below:

Title : {{question_title}}
Statement : {{question_content}}

Requirements
------------
1. Provide **ONLY executable source code**:
   • NO comments of any kind (no //, no #, no /* */, no docstrings)
   • NO explanations
   • NO markdown formatting except the <code> tags
   • Just pure, clean, runnable code
2. Implement a `main()` function that:
   • reads all input from **stdin**,  
   • computes the answer,  
   • writes the answer to **stdout**.
3. Add any helper functions you need (without comments).
4. **Call `main()` at the bottom** of the file, otherwise the solution will not be executed.
5. Do not modify the supplied starter skeleton:

{{starter_code}}

Output Format
-------------
Surround the final code with

<code>
…your code…
</code>

CRITICAL: Generate the COMPLETE solution. Do NOT stop mid-function. Ensure all functions are complete and the program is fully executable.

Now produce the final program.                  
"""
        return cls(
            template=_template,
            input_variables=["language", "question_title", "question_content", "starter_code"]
        )