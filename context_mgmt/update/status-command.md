# Status Check + Auto-Update

Verify current project reality and update state files to match.

## Instructions

1. **Read current state files**
   - Read @current_plan.md for claimed progress and current task
   - Read @sessions/recent-work.md for recent activity
   - Note last update timestamps

2. **Execute validation to verify reality**
   - Run validation commands from current_plan.md
   - Focus on tests for current task area
   - Capture actual pass/fail results

3. **Compare claimed vs actual state**
   - Check if claimed progress matches test results
   - Identify any discrepancies between claims and reality
   - Calculate actual completion percentage

4. **UPDATE state files to match reality**
   - Update progress in @current_plan.md based on test results
   - Update current task description if needed
   - Update blockers based on failing tests
   - Update @sessions/recent-work.md with actual status

5. **Show explicit changes made**
   - Display each file updated with before/after values
   - Show specific discrepancies found and fixed
   - Provide clear next action based on reality

## Output Format

```
🔍 STATUS CHECK - Verifying Project Reality

Reading state files...
✓ current_plan.md (last updated: [time ago])
✓ sessions/recent-work.md (last session: [when])

Running validation...
$ [validation command]
[Show actual test output summary]

Comparing claimed vs actual:
[✓ or ❌] [Comparison of each major claim]

📝 Updated: current_plan.md
   - Progress: [old%] → [new%] (based on actual tests)
   - Current task: "[updated task description]"
   - Blockers: [updated based on failures]

📝 Updated: sessions/recent-work.md
   - Status: "[actual current status]"
   - Next: "[specific next action]"

Current State:
- Task: [current task with context]
- Actual Progress: [real%] ([x]/[total] components working)
- Blocker: [most critical blocker]
- Next Action: [specific actionable step]
```

## Example Execution

When user types `/status`, the assistant should:

1. Actually read the current files
2. Show the validation being run
3. Update files based on test results
4. Display the updates clearly
5. Provide actionable next steps

Remember: This command exists to ensure state files ALWAYS reflect reality, not hopes or partial work.