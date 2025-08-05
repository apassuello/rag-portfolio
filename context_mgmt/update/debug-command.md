# Debug Mode - Critical Analysis

Activate extreme skepticism and systematic debugging approach.

## Instructions

1. **Read current problem context**
   - Read @current_plan.md for current blockers
   - Identify the specific issue to investigate
   - Load ONLY the problematic code section

2. **Activate debug mental model**
   - Switch to extreme skepticism mode
   - Question every assumption
   - Trust nothing, verify everything
   - Focus on systematic isolation

3. **Load minimal debug context**
   - The specific failing test
   - The exact function/method in question  
   - Recent debug observations from @sessions/debug-log.md
   - NO comprehensive files

4. **Update debug tracking**
   - Create/update @sessions/debug-log.md
   - Record investigation start time
   - Note specific issue being investigated

5. **Provide debug approach guidance**
   - List systematic steps to isolate issue
   - Suggest specific logging/print statements
   - Recommend isolation techniques
   - Challenge common assumptions

## Output Format

```
🔍 DEBUG MODE ACTIVATED - Trust Nothing

Loading debug context...
📄 Loaded ([X] tokens):
   - Current issue: "[specific problem description]"
   - [filename] ([specific function/lines])
   - [test_file]::[failing_test]
   - Last debug note: "[previous observation if any]"

📝 Updated: sessions/debug-log.md
   - Debug session started: [timestamp]
   - Target: [specific issue]

⚠️  Debug Reflexes Active:
✓ "Test passed" messages are LIES until proven
✓ Read EVERY character of error messages
✓ Question ALL assumptions
✓ Verify ACTUAL vs EXPECTED character by character
✓ Check for silent failures and swallowed exceptions
✓ State dependencies may be corrupted
✓ The bug is probably where you're NOT looking

Critical Questions:
1. [Specific question about the issue]
2. [Another critical question]
3. [Third probing question]

Systematic Investigation Steps:
1. [First concrete action with specific code]
2. [Second isolation step]
3. [Third verification step]

Debug Commands:
- Add print: `print(f"DEBUG: variable={variable!r}")`
- Run isolated: `pytest [test] -xvs --capture=no -k specific_test`
- Check state: `print(f"State before: {locals()}")`

Remember: The bug is real. Your assumptions are wrong.
```

## Debug Patterns

### Pattern 1: Test passes but behavior wrong
```
⚠️  SUSPICIOUS: Test passes but behavior incorrect

Critical Questions:
1. Is the test actually testing the right thing?
2. Are we comparing the expected values?
3. Is the test data realistic?

Investigation:
1. Add assertion to verify test preconditions
2. Print actual vs expected with repr()
3. Run test with production-like data
```

### Pattern 2: Intermittent failures
```
⚠️  FLAKY: Intermittent test failures

Critical Questions:
1. Is there shared state between tests?
2. Are we depending on timing/ordering?
3. Is there a race condition?

Investigation:
1. Run test in isolation vs suite
2. Add timestamps to all operations  
3. Check for class-level state pollution
```

### Pattern 3: "Should work" but doesn't
```
⚠️  ASSUMPTION: Code "should work" but fails

Critical Questions:
1. What are we assuming about input format?
2. Are we handling all edge cases?
3. Is the error actually somewhere else?

Investigation:
1. Log ALL inputs with type and repr()
2. Check every function call return value
3. Trace execution path step by step
```

## Example Debug Sessions

### Example 1: Silent failure
```
🔍 DEBUG MODE ACTIVATED - Trust Nothing

Loading debug context...
📄 Loaded (423 tokens):
   - Current issue: "BM25 scores always 0.0"
   - src/retrievers/hybrid_retriever.py (calculate_bm25_score method)
   - tests/test_hybrid_retriever.py::test_bm25_scoring
   - Last debug note: "IDF calculation returns value"

⚠️  Debug Reflexes Active:
✓ Zero values suggest silent math failure
✓ Check for division by zero
✓ Verify data types in calculations

Critical Questions:
1. Are document frequencies actually populated?
2. Is the term actually in the document?
3. Are we using integer division by accident?

Systematic Investigation Steps:
1. Add: `print(f"DEBUG: tf={tf}, df={df}, N={N}")`
2. Check: `print(f"Type check: {type(tf)}, {type(df)}")`
3. Verify: `assert df > 0, f"df is {df}"`

The bug is probably in data initialization, not the formula.
```

## Debug Principles

1. **Trust nothing** - Verify every claim
2. **Read carefully** - Every character matters
3. **Question assumptions** - They're probably wrong
4. **Isolate systematically** - One variable at a time
5. **Log everything** - repr() is your friend
6. **The bug is real** - It's not going away

This command activates a different mental model essential for solving difficult problems.