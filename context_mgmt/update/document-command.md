# Document Session Work

Record verified accomplishments and update progress tracking.

## Instructions

1. **Analyze actual work done**
   - Run `git diff --stat` to see changed files
   - Run `git log -1 --oneline` for latest commit
   - Compare against planned work from @current_plan.md

2. **Verify accomplishments with tests**
   - Run relevant validation commands
   - Only document work that has passing tests
   - Note any regressions or new failures

3. **Create accurate session record**
   - Document what was planned vs accomplished
   - Record key decisions and solutions
   - Note any blockers discovered
   - Include time spent if relevant

4. **Update progress tracking**
   - Update progress percentage in @current_plan.md
   - Update current task if completed
   - Add new blockers if discovered
   - Update @sessions/recent-work.md with verified work

5. **Save session documentation**
   - Create @sessions/session-[date].md
   - Include all verified accomplishments
   - Add next steps based on current state

## Output Format

```
📝 DOCUMENTING SESSION - Verified Reality

Analyzing actual work done...
$ git diff --stat
[Show file changes]

$ git log -1 --oneline
[Show latest commit]

Running validation to verify claims...
$ [validation command]
[Show test results]

Session Record Created:
📄 sessions/session-[YYYY-MM-DD].md

Session Summary:
- Planned: [what was intended]
- Accomplished: 
  ✓ [verified accomplishment 1]
  ✓ [verified accomplishment 2]
  ✗ [attempted but not completed]
- Decisions: [key decisions made]
- Blockers: [any new blockers found]
- Validation: [test results summary]
- Time: [duration if tracked]

📝 Updated: current_plan.md
   - Progress: [old%] → [new%]
   - Current task: [updated if needed]
   - Blockers: [updated list]

📝 Updated: sessions/recent-work.md
   - Added: [today's verified work]
   - Status: [current state]

Next Steps:
1. [Immediate next action]
2. [Following action]
```

## Example Documentation

### Example 1: Feature completion
```
📝 DOCUMENTING SESSION - Verified Reality

Analyzing actual work done...
$ git diff --stat
 src/retrievers/hybrid_retriever.py | 23 +++++++++++++++++------
 tests/test_hybrid_retriever.py     |  5 ++++-
 2 files changed, 21 insertions(+), 7 deletions(-)

$ git log -1 --oneline
a5f2c31 Fix BM25 IDF calculation with log term

Running validation to verify claims...
$ pytest tests/test_hybrid_retriever.py::test_bm25_scoring -xvs
PASSED

Session Record Created:
📄 sessions/session-2024-01-22.md

Session Summary:
- Planned: Fix BM25 scoring algorithm
- Accomplished:
  ✓ Fixed IDF calculation (added log term)
  ✓ Handled edge case for zero document frequency
  ✓ All BM25 tests now passing
- Decisions: Use natural log for IDF calculation
- Blockers: None removed, scoring now works
- Validation: test_bm25_scoring PASSED
- Time: 2 hours

📝 Updated: current_plan.md
   - Progress: 22% → 35%
   - Current task: "Implement score combination logic"
   - Blockers: []

📝 Updated: sessions/recent-work.md
   - Added: "BM25 scoring complete with IDF fix"
   - Status: "Ready for hybrid combination"

Next Steps:
1. Implement combine_scores method
2. Test with different weight combinations
```

### Example 2: Partial progress
```
📝 DOCUMENTING SESSION - Verified Reality

Analyzing actual work done...
$ git diff --stat
 src/query_processors/enhancer.py | 45 ++++++++++++++++++++++
 1 file changed, 45 insertions(+)

Running validation to verify claims...
$ pytest tests/test_query_enhancement.py -xvs
3 passed, 2 failed

Session Summary:
- Planned: Implement query enhancement
- Accomplished:
  ✓ Basic synonym expansion working
  ✓ Query structure parsing implemented
  ✗ Semantic expansion not complete (embedding integration pending)
- Decisions: Use WordNet for synonyms initially
- Blockers: Need embedding model for semantic expansion
- Validation: 3/5 tests passing

📝 Updated: current_plan.md
   - Progress: 35% → 42%
   - Blockers: ["Embedding model integration needed"]

Next Steps:
1. Integrate embedding model for semantic expansion
2. Complete remaining enhancement strategies
```

## Key Principles

1. **Only document verified work** - Must have test/commit evidence
2. **Show git evidence** - Real commits, real changes
3. **Update actual progress** - Based on tests passing
4. **Create useful record** - Future you will thank you
5. **Be honest about incomplete work** - ✗ for attempted but not done

This command creates an accurate historical record for future reference and progress tracking.