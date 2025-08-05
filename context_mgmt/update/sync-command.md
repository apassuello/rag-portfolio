# Deep Reality Reconciliation

Reconcile all state files with git history and comprehensive test results.

## Instructions

1. **Analyze git history for actual work**
   - Run `git log --oneline --since="48 hours ago"`
   - Identify commits not reflected in state files
   - Extract actual changes made from git diff

2. **Run comprehensive validation suite**
   - Execute ALL validation commands from current_plan.md
   - Run `pytest tests/ -v` for complete test coverage
   - Capture detailed results and failure reasons

3. **Compare state files vs reality**
   - Check every claim in current_plan.md against test results
   - Verify recent-work.md matches git commits
   - Identify all discrepancies and state drift

4. **Fix ALL discrepancies found**
   - Remove unverified claims from state files
   - Add missed accomplishments from git
   - Update progress percentages based on actual test results
   - Correct task descriptions to match current reality

5. **Show complete reconciliation report**
   - List every discrepancy found
   - Show all fixes applied
   - Confirm final state accuracy

## Output Format

```
🔄 REALITY SYNC - Deep State Reconciliation

Analyzing git history...
$ git log --oneline --since="48 hours ago"
[Show actual commits]
✓ Found: [X] commits not in state files

Running comprehensive validation...
$ pytest tests/ -v --tb=short
[Show test summary]
=============== [X] passed, [Y] failed ===============

Reconciling state files with reality...

❌ State Drift Detected:
1. [Specific discrepancy description]
2. [Another discrepancy]
[... list all found]

📝 Fixed: current_plan.md
   - Removed: "[unverified claim]"
   - Corrected: "Progress: [old%] → [real%]"
   - Updated: "[other corrections]"

📝 Fixed: sessions/recent-work.md
   - Added from git: "[missed accomplishment]"
   - Removed: "[unverified work]"
   - Updated: "[status corrections]"

📝 Fixed: sessions/validation-results.md
   - Updated with current test results
   - [X]/[Y] tests passing ([percentage]%)
   - Critical failures: [list key failures]

✅ Sync Complete:
- Fixed [N] state files
- Corrected [M] discrepancies
- State now matches reality
- Ready for accurate development
```

## When to Use

- After conversation compaction
- When state feels "off" or inaccurate  
- After resuming work from break
- When tests pass but state shows failure (or vice versa)
- As periodic reality check (weekly)

## Example Execution

The assistant should:

1. Show actual git commits found
2. Run real validation and show results
3. List EVERY discrepancy explicitly
4. Update all affected files
5. Confirm state accuracy restored

This command is the "nuclear option" for state recovery - it fixes everything to match reality.