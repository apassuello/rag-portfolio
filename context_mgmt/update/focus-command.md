# Minimal Context Loading

Load only essential context for specific work area to preserve conversation space.

## Instructions

1. **Parse focus area parameter**
   - Valid areas: hybrid-search, query-enhancement, answer-generation, testing, architecture
   - Default to current task if no area specified

2. **Read current task from state**
   - Read @current_plan.md for current_task and specific focus
   - Identify specific component/function being worked on

3. **Load MINIMAL context only**
   - Current task description (2-3 lines max)
   - Specific code file and function/class (not entire file)
   - Relevant test case (specific test method)
   - NO comprehensive documentation
   - NO architecture files unless specifically needed

4. **Update focus tracking**
   - Update current_focus in @current_plan.md
   - Add timestamp for focus session
   - Note specific work area

5. **Display loaded context with token count**
   - Show exactly what was loaded
   - Display token count to confirm minimal
   - Provide specific work guidance

## Areas

### `hybrid-search`
- Focus: BM25 + dense retrieval combination
- Files: src/retrievers/hybrid_retriever.py, tests/test_hybrid_retriever.py
- Context: Current implementation status, failing tests

### `query-enhancement`  
- Focus: Query expansion and refinement
- Files: src/query_processors/enhancer.py, tests/test_query_enhancement.py
- Context: Enhancement strategies, current gaps

### `answer-generation`
- Focus: LLM integration and response generation
- Files: src/generators/answer_generator.py, tests/test_answer_generation.py  
- Context: Generation pipeline, quality metrics

### `testing`
- Focus: Test development and validation
- Files: Current test file, implementation being tested
- Context: Coverage gaps, test strategies

### `architecture`
- Focus: System design and boundaries
- Files: Specific component interface, design docs
- Context: Design decisions needed

## Output Format

```
🎯 FOCUSING: [area]

Loading minimal context...
📄 Loaded ([X] tokens):
   1. Current task: "[specific task description]"
   2. [filename] (lines X-Y)
   3. [test_file]::test_method

📝 Updated: current_plan.md
   - Current focus: "[area]"
   - Focus timestamp: "[ISO timestamp]"

Ready to work on:
- [Specific immediate action]
- [Key detail to remember]
- [Test to satisfy]

Token usage: [X]/100000 (minimal - no compaction risk)
```

## Examples

### Example 1: Focus on current bug
```
/focus hybrid-search

🎯 FOCUSING: hybrid-search

Loading minimal context...
📄 Loaded (387 tokens):
   1. Current task: "Fix BM25 IDF calculation"
   2. src/retrievers/hybrid_retriever.py (lines 45-67)
   3. tests/test_hybrid_retriever.py::test_bm25_scoring

📝 Updated: current_plan.md
   - Current focus: "hybrid-search"
   - Focus timestamp: "2024-01-22T10:30:00Z"

Ready to work on:
- Fix IDF formula missing log term (line 52)
- Test expects: log(N/df) not N/df
- Run test with: pytest tests/test_hybrid_retriever.py::test_bm25_scoring -xvs

Token usage: 387/100000 (minimal - no compaction risk)
```

### Example 2: Switch to different area
```
/focus testing

🎯 FOCUSING: testing

Loading minimal context...
📄 Loaded (295 tokens):
   1. Current task: "Add edge case tests for BM25"
   2. tests/test_hybrid_retriever.py (lines 89-102)
   3. Test coverage: 87% (missing edge cases)

📝 Updated: current_plan.md
   - Current focus: "testing"
   - Focus timestamp: "2024-01-22T14:15:00Z"

Ready to work on:
- Add test for zero-frequency terms
- Add test for empty documents
- Current coverage missing: error handling paths

Token usage: 295/100000 (minimal - no compaction risk)
```

## Critical Rules

1. **Never load more than 500 tokens**
2. **Load specific functions, not entire files**
3. **Include line numbers for precise context**
4. **Always show token count**
5. **Update focus tracking for continuity**

This command enables productive work without triggering conversation compaction.