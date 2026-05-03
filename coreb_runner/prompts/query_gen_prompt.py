from easyllm_kit.utils import PromptTemplate


class QueryGenPrompt(PromptTemplate):
    @classmethod
    def init(cls):
        _template = """# Code Retrieval Query Generation Task

## Task Description
You are an expert at generating developer-style search queries for a code embedding retrieval system.

Given a target coding problem and a list of similar but distinct problems (retrieved via cosine similarity), your goal is to create queries that best match the target problem while avoiding ambiguity with the similar ones.

Input
Target Problem: {{target_problem}}
Similar Problems: {{similar_problem_list}}

## Core Principles
Target-Centric Queries
- Generate queries that clearly and uniquely map to the target problem.
- Use precise, problem-specific terminology, inputs, outputs, and constraints.
Differentiation Awareness
- Analyze differences between the target and each similar problem.
- Explicitly avoid shared terms or overlapping constraints that could confuse retrieval.
Ambiguity Avoidance
- Queries must exclude vague phrasing that could apply to multiple problems.
- Do not rely on unspecified variable names like m, t, n. Always describe them explicitly (e.g., "array size up to 200").
- Emphasize unique requirements or constraints that only the target problem contains.

## Query Generation Guidelines
Generate diverse search queries for the specified query type: **{{query_type}}**
- Natural Language Style: Write queries like real developers searching online or in documentation.
- Length: 15–40 words per query.
- Target-Specific Focus: Emphasize unique aspects that ONLY apply to the target problem.
- Exclusion Strategy: Avoid terms or concepts shared with similar problems.
- Precision Over Generality: Use narrow, context-rich descriptions instead of broad concepts.
- Unique Constraint Emphasis: Highlight constraints unique to the target problem (e.g., input size, optimization goal, special conditions).
- Distinctive Terminology: Use domain-specific language that distinguishes the target from its distractors.

### Query Type Specifications:
**title_search:** Focus on core problem names and key concepts UNIQUE to the target problem
**description_search:** Extract requirements and functionality SPECIFIC to the target problem
**algorithm_search:** Focus on algorithmic techniques SPECIFIC to the target problem's solution
**cross_language:** Target specific languages with implementation details UNIQUE to the target problem
**language_agnostic:** Focus on algorithmic concepts SPECIFIC to the target problem without language specificity

## Requirements
1. **Natural Language:** Use realistic, conversational queries that developers would actually type
2. **Target-Specific Focus:** Include ONLY technical terms and concepts UNIQUE to the target problem
3. **Detailed Context:** Create comprehensive queries (15-40 words) that capture the target problem's complexity
4. **Maximum Precision:** Generate queries that clearly distinguish the target problem from similar ones
5. **Exclusion Strategy:** Avoid concepts, operations, or approaches shared with similar problems
6. **Unique Constraint Emphasis:** Highlight constraints, requirements, or conditions unique to the target problem
7. **Avoid Placeholder Variables:** Never use symbols like m, t, n without describing their meaning.
8. **Distinctive Terminology:** Use domain-specific language that distinguishes the target problem

## Output Format
Generate queries in JSON format with the following structure:

```json
{
  "queries": [
    {
      "query": "query text here",
      "best_match": "problem_id",
      "best_reason": "brief explanation",
      "second_best": "problem_id", 
      "second_reason": "brief explanation",
      "third_best": "problem_id",
      "third_reason": "brief explanation"
    }
  ]
}
```

"""
        return cls(
            template=_template,
            input_variables=["target_problem", "similar_problem_list", "query_type"]
        )


class CrossLanguageQueryGenPrompt(PromptTemplate):
    @classmethod
    def init(cls):
        _template = """# Cross-Language Code Retrieval Query Generation

## Task Description
You are an expert at generating developer-style search queries for a code embedding retrieval system.

Given a target coding problem and a list of similar but distinct problems, your goal is to create queries that best match the target problem while avoiding ambiguity with the similar ones, specifically requesting solutions in **{{target_language}}**.

Input
Target Problem: {{target_problem}}
Similar Problems: {{similar_problem_list}}
Target Language: {{target_language}}

## Core Principles
Target-Centric Queries
- Generate queries that clearly and uniquely map to the target problem.
- Use precise, problem-specific terminology, inputs, outputs, and constraints.
- Always mention the target language explicitly.
Differentiation Awareness
- Analyze differences between the target and each similar problem.
- Explicitly avoid shared terms or overlapping constraints that could confuse retrieval.
Ambiguity Avoidance
- Queries must exclude vague phrasing that could apply to multiple problems.
- Do not rely on unspecified variable names like m, t, n. Always describe them explicitly (e.g., "array size up to 200").
- Emphasize unique requirements or constraints that only the target problem contains.

## Query Generation Guidelines
Generate diverse search queries for the specified query type: **{{query_type}}**

- Natural Language Style: Write queries like real developers searching online or in documentation.
- Length: 15–40 words per query.
- Target-Specific Focus: Emphasize unique aspects that ONLY apply to the target problem.
- Exclusion Strategy: Avoid terms or concepts shared with similar problems.
- Precision Over Generality: Use narrow, context-rich descriptions instead of broad concepts.
- Unique Constraint Emphasis: Highlight constraints unique to the target problem (e.g., input size, optimization goal, special conditions).
- Distinctive Terminology: Use domain-specific language that distinguishes the target from its distractors.
- Language Integration: Seamlessly integrate **{{target_language}}** requirements with problem-specific details.

### Query Type Specifications:
**title_search:** Focus on core problem names and key concepts UNIQUE to the target problem
**description_search:** Extract requirements and functionality SPECIFIC to the target problem
**algorithm_search:** Focus on algorithmic techniques SPECIFIC to the target problem's solution
**cross_language:** Target specific languages with implementation details UNIQUE to the target problem
**language_agnostic:** Focus on algorithmic concepts SPECIFIC to the target problem without language specificity

## Requirements
1. **Natural Language:** Use realistic, conversational queries that developers would actually type
2. **Target-Specific Focus:** Include ONLY technical terms and concepts UNIQUE to the target problem
3. **Detailed Context:** Create comprehensive queries (15-40 words) that capture the target problem's complexity
4. **Maximum Precision:** Generate queries that clearly distinguish the target problem from similar ones
5. **Exclusion Strategy:** Avoid concepts, operations, or approaches shared with similar problems
6. **Unique Constraint Emphasis:** Highlight constraints, requirements, or conditions unique to the target problem
7. **Avoid Placeholder Variables:** Never use symbols like m, t, n without describing their meaning.
8. **Distinctive Terminology:** Use domain-specific language that distinguishes the target problem
9. **Language Specificity:** Always mention **{{target_language}}** explicitly in queries

## Output Format
Generate queries in JSON format with the following structure:

```json
{
  "queries": [
    {
      "query": "query text here",
      "best_match": "problem_id",
      "best_reason": "brief explanation",
      "second_best": "problem_id", 
      "second_reason": "brief explanation",
      "third_best": "problem_id",
      "third_reason": "brief explanation"
    }
  ]
}
```
"""
        return cls(
            template=_template,
            input_variables=["target_problem", "similar_problem_list", "target_language", "query_type"]
        )


class AlgorithmFocusedQueryGenPrompt(PromptTemplate):
    @classmethod
    def init(cls):
        _template = """# Algorithm-Focused Query Generation

## Task Description
You are an expert at generating developer-style search queries for a code embedding retrieval system.

Given a target coding problem and a list of similar but distinct problems, your goal is to create queries that best match the target problem while avoiding ambiguity with the similar ones, focusing on specific algorithms and data structures.

Input
Target Problem: {{target_problem}}
Similar Problems: {{similar_problem_list}}

## Core Principles
Target-Centric Queries
- Generate queries that clearly and uniquely map to the target problem.
- Use precise, algorithm-specific terminology, data structures, and optimization techniques.
Differentiation Awareness
- Analyze differences between the target and each similar problem.
- Explicitly avoid shared algorithmic approaches that could confuse retrieval.
Ambiguity Avoidance
- Queries must exclude vague phrasing that could apply to multiple problems.
- Do not rely on unspecified variable names like m, t, n. Always describe them explicitly (e.g., "array size up to 200").
- Emphasize unique algorithmic requirements or constraints that only the target problem contains.

## Query Generation Guidelines
Generate diverse search queries for the specified query type: **{{query_type}}**

- Natural Language Style: Write queries like real developers searching online or in documentation.
- Length: 15–40 words per query.
- Target-Specific Focus: Emphasize unique aspects that ONLY apply to the target problem.
- Exclusion Strategy: Avoid terms or concepts shared with similar problems.
- Precision Over Generality: Use narrow, context-rich descriptions instead of broad concepts.
- Unique Constraint Emphasis: Highlight constraints unique to the target problem (e.g., input size, optimization goal, special conditions).
- Distinctive Terminology: Use domain-specific language that distinguishes the target from its distractors.
- Algorithm-Specific Focus: Emphasize specific algorithms, data structures, and optimization techniques.

### Query Type Specifications:
**title_search:** Focus on core problem names and key concepts UNIQUE to the target problem
**description_search:** Extract requirements and functionality SPECIFIC to the target problem
**algorithm_search:** Focus on algorithmic techniques SPECIFIC to the target problem's solution
**cross_language:** Target specific languages with implementation details UNIQUE to the target problem
**language_agnostic:** Focus on algorithmic concepts SPECIFIC to the target problem without language specificity

## Requirements
1. **Natural Language:** Use realistic, conversational queries that developers would actually type
2. **Target-Specific Focus:** Include ONLY technical terms and concepts UNIQUE to the target problem
3. **Detailed Context:** Create comprehensive queries (15-40 words) that capture the target problem's complexity
4. **Maximum Precision:** Generate queries that clearly distinguish the target problem from similar ones
5. **Exclusion Strategy:** Avoid concepts, operations, or approaches shared with similar problems
6. **Unique Constraint Emphasis:** Highlight constraints, requirements, or conditions unique to the target problem
7. **Avoid Placeholder Variables:** Never use symbols like m, t, n without describing their meaning.
8. **Distinctive Terminology:** Use domain-specific language that distinguishes the target problem
9. **Algorithm Specificity:** Focus on specific algorithms, data structures, and optimization techniques unique to the target problem

## Output Format
Generate queries in JSON format with the following structure:

```json
{
  "queries": [
    {
      "query": "query text here",
      "best_match": "problem_id",
      "best_reason": "brief explanation",
      "second_best": "problem_id", 
      "second_reason": "brief explanation",
      "third_best": "problem_id",
      "third_reason": "brief explanation"
    }
  ]
}
```
"""
        return cls(
            template=_template,
            input_variables=["target_problem", "similar_problem_list", "query_type"]
        )


class QuestionContentAbbreviatePrompt(PromptTemplate):
    @classmethod
    def init(cls):
        _template = """# Question Content Abbreviation Task

## Task Description
You are an expert at creating concise, retrieval-optimized summaries of coding problem descriptions. Your goal is to distill the essential information from a problem statement into a compact format that preserves key details while removing redundancy.

## Input
Question title: {{question_title}}
Content: {{question_content}}

## Core Principles
**Essential Information Preservation**
- Retain the core problem goal, key constraints, and unique requirements
- Preserve specific numerical limits, data types, and optimization objectives
- Keep distinctive algorithmic concepts or problem-specific terminology

**Conciseness Optimization**
- Remove verbose explanations, examples, and repetitive content
- Eliminate unnecessary background information or motivational text
- Compress similar concepts into unified statements

**Retrieval-Friendly Format**
- Use clear, structured language that matches how developers search
- Include specific technical terms and constraints that distinguish this problem
- Maintain logical flow from problem statement to requirements

## Abbreviation Guidelines
1. **Problem Goal**: Start with a clear, concise statement of what needs to be accomplished
2. **Key Constraints**: Include specific limits, ranges, and requirements (e.g., "array size ≤ 10^5", "time complexity O(n log n)")
3. **Input/Output Format**: Specify data types and formats when critical to the problem
4. **Unique Requirements**: Highlight distinctive aspects that differentiate this problem from similar ones
5. **Algorithmic Hints**: Include key algorithmic concepts if they're central to the solution approach

## Output Requirements
- **Length**: 50-150 words maximum
- **Structure**: Use clear, declarative sentences
- **Precision**: Include specific numerical constraints and technical details
- **Clarity**: Avoid ambiguous pronouns or references
- **Completeness**: Ensure all essential problem-solving information is preserved


Please provide your abbreviated version in JSON format:
```json
{
  "abbreviated_content": "<your_concise_abbreviation>"
}
```"""

        return cls(
            template=_template,
            input_variables=["question_title", "question_content"]
        )
