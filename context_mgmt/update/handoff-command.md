# Create Session Handoff

Generate ready-to-use prompt for seamless session continuation.

## Instructions

1. **Analyze current verified state**
   - Read @current_plan.md for accurate current state
   - Read @sessions/recent-work.md for today's progress
   - Check @sessions/validation-results.md for test status

2. **Identify concrete next steps**
   - Determine immediate next action from current state
   - Identify specific file and function to work on
   - Note any blockers or context needed

3. **Create self-contained continuation prompt**
   - Include verified current state (not assumptions)
   - Specify exact commands to run first
   - Provide specific work focus
   - Include relevant test commands

4. **Save handoff document**
   - Create @sessions/handoff-[date].md with full context
   - Include session summary and next steps
   - Add timestamp and state snapshot

5. **Generate copy-paste ready prompt**
   - Make it completely self-contained
   - No references to "previous session" or "yesterday"
   - Include specific line numbers and test names

## Output Format

```
🤝 CREATING HANDOFF - Session Continuity

Analyzing current state...
✓ Today's progress: [start%] → [end%] ([what actually completed])
✓ Current task: [specific task name]
✓ Tests passing: [X]/[Y]
✓ Next logical step: [specific action]

Creating handoff document...
📄 Created: sessions/handoff-[YYYY-MM-DD].md

=== READY-TO-USE PROMPT FOR NEXT SESSION ===

I'm continuing the [specific feature] implementation. 

Current state:
- [Specific component] is working ([what's verified working])
- [Current component] needs [specific fix/feature]
- Tests passing: [X]/[Y]
- Specific failure: [test name] expects [expected behavior]

Please:
1. Run /status to verify current state
2. Run /focus [area] to load context
3. [Specific implementation instruction]
4. [Where to find the code]: [filename] line [X]
5. Run [specific test command] after implementation

The main work is in [specific method/function] that currently [current state].

=== END PROMPT ===

Handoff complete. Copy the prompt above to start next session.
```

## Example Handoffs

### Example 1: Mid-feature handoff
```
=== READY-TO-USE PROMPT FOR NEXT SESSION ===

I'm continuing the hybrid search implementation.

Current state:
- BM25 tokenizer is working (test_bm25_tokenizer passes)
- BM25 scoring needs IDF calculation fix
- Tests passing: 42/47
- Specific failure: test_bm25_scoring expects log(N/df) in IDF calculation

Please:
1. Run /status to verify current state
2. Run /focus hybrid-search to load context  
3. Fix the IDF calculation in calculate_bm25_score method
4. The bug is in src/retrievers/hybrid_retriever.py line 52
5. Run pytest tests/test_hybrid_retriever.py::test_bm25_scoring -xvs

The IDF calculation is currently using N/df but should use log(N/df).

=== END PROMPT ===
```

### Example 2: Debug session handoff
```
=== READY-TO-USE PROMPT FOR NEXT SESSION ===

I'm investigating a test inconsistency in the answer generation module.

Current state:
- Test claims to pass but generated answers are truncated
- Issue appears to be in response streaming
- All unit tests pass but integration test shows truncation
- Debug session identified potential buffer size issue

Please:
1. Run /status to verify current state
2. Run /debug to activate critical analysis mode
3. Check ResponseStreamer.buffer_size in src/generators/answer_generator.py
4. Run integration test with debug output: pytest tests/integration/test_answer_generation.py -xvs --capture=no
5. Add logging to track buffer flushes

The truncation happens after exactly 512 tokens, suggesting buffer limit.

=== END PROMPT ===
```

## Critical Success Factors

1. **Self-contained prompt** - No external references needed
2. **Specific actions** - Exact commands and line numbers
3. **Verified state** - Only include confirmed information
4. **Clear next step** - One specific task to accomplish
5. **Test command** - How to verify success

This command ensures zero context loss between sessions by creating a perfect starting point.