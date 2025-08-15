# RAG Portfolio Command System v2.0

## Executive Summary

The RAG Portfolio Command System provides automated context management for AI-assisted development sessions. This system solves the critical problem of context loss between conversations by ensuring state files always reflect reality, not aspirations.

**Core Innovation**: Commands that verify and update state based on actual project reality, not claimed progress.

## System Architecture

### Design Principles

1. **Reality-Based State Management**
   - Every command verifies actual state through tests and git
   - State files updated to match what ACTUALLY works
   - No tracking of "started but not finished" work

2. **Minimal Context Loading**
   - Commands load <500 tokens typically
   - Prevents conversation compaction
   - Preserves space for actual development

3. **Clear State Updates**
   - Every command shows explicit file updates
   - Users see exactly what changed and why
   - Format: `📝 Updated: filename (what changed)`

4. **Natural Workflow Integration**
   - Commands match actual development patterns
   - Mental models for different types of work
   - No artificial ceremony or overhead

## Command Reference

### Phase 1: Critical Commands (Essential Daily Use)

#### `/status` - Reality Check + Auto-Update
**Purpose**: Verify what's actually done and update state to match reality

**Workflow**:
```
1. Read current state files
2. Run validation commands (pytest)
3. Compare claimed vs actual progress
4. UPDATE state files to match reality
5. Show explicit changes made
```

**Example Usage**:
```bash
/status

# Output:
🔍 STATUS CHECK - Verifying Project Reality
✓ Ran: pytest tests/test_hybrid_retriever.py
❌ Claimed: 30% complete | Actual: 20% complete
📝 Updated: current_plan.md (progress: 30% → 20%)
📝 Updated: sessions/recent-work.md (status: BM25 scoring incomplete)
Current Task: Fix BM25 scoring algorithm
Blocker: IDF calculation missing log term
```

#### `/sync` - Deep Reality Reconciliation
**Purpose**: Reconcile all state files with git history and test results

**Workflow**:
```
1. Analyze git commits (last 24-48h)
2. Run comprehensive test suite
3. Compare all state claims vs reality
4. Fix ALL discrepancies
5. Show complete reconciliation
```

**Example Usage**:
```bash
/sync

# Output:
🔄 REALITY SYNC - Deep State Reconciliation
✓ Found: 5 commits not in state files
❌ State drift: 7 discrepancies found
📝 Fixed: current_plan.md (removed unverified claims)
📝 Fixed: sessions/recent-work.md (added missed commits)
✅ Sync complete: State now matches reality
```

#### `/focus [area]` - Minimal Context Loading
**Purpose**: Load only essential context for specific work area

**Areas**:
- `hybrid-search` - BM25 + dense retrieval
- `query-enhancement` - Query expansion
- `answer-generation` - LLM integration
- `testing` - Test development
- `architecture` - System design

**Example Usage**:
```bash
/focus hybrid-search

# Output:
🎯 FOCUSING: hybrid-search
📄 Loaded (387 tokens):
   - Current task: Implement BM25 scoring
   - src/retrievers/hybrid_retriever.py (lines 45-67)
   - tests/test_hybrid_retriever.py::test_bm25_scoring
📝 Updated: current_plan.md (focus: hybrid-search)
Ready to work on: Fix IDF calculation
```

### Phase 2: Workflow Commands (Session Management)

#### `/handoff` - Create Next Session Prompt
**Purpose**: Generate ready-to-use prompt for next session

**Workflow**:
```
1. Analyze current verified state
2. Identify next logical steps
3. Create self-contained prompt
4. Include specific context needs
```

**Example Usage**:
```bash
/handoff

# Output:
🤝 CREATING HANDOFF
📄 Created: sessions/handoff-2024-01-22.md

=== COPY THIS PROMPT FOR NEXT SESSION ===
I'm continuing hybrid search implementation.
Current: BM25 scoring fixed, need combination logic.
Run: /status then /focus hybrid-search
Task: Implement combine_scores method
Test: tests/test_hybrid_retriever.py::test_hybrid_combination
===
```

#### `/document` - Record Verified Session Work
**Purpose**: Create accurate record of what was actually accomplished

**Workflow**:
```
1. Analyze git diff
2. Run tests to verify claims
3. Create session record
4. Update progress tracking
```

#### `/validate` - Run Test Suite
**Purpose**: Execute validation commands and interpret results

**Workflow**:
```
1. Read validation commands from current_plan.md
2. Execute each command
3. Interpret results
4. Update validation-results.md
```

### Phase 3: Mental Model Commands (Specialized Work)

#### `/debug` - Critical Analysis Mode
**Purpose**: Activate extreme skepticism for difficult problems

**Mental Model**:
- "Test passed" messages are lies
- Read every character carefully
- Question all assumptions
- Isolate and reproduce systematically

**Example Usage**:
```bash
/debug

# Output:
🔍 DEBUG MODE ACTIVATED - Trust Nothing
⚠️ Debug Reflexes Active:
✓ Question EVERYTHING
✓ Verify character by character
✓ Check for silent failures
📝 Updated: sessions/debug-log.md
First steps: Add print statements, run in isolation
```

#### Role Commands (When Truly Different Work)
- `/architect` - System design, boundaries, specifications
- `/implementer` - Code implementation, optimization
- `/validator` - Testing, quality assurance

## File Structure

### State Management Files
```
current_plan.md              # Single source of truth
├── current_task            # What we're working on
├── progress                # Verified completion %
├── blockers                # Current obstacles
└── validation_commands     # How to verify state

sessions/
├── recent-work.md          # Rolling accomplishment log
├── validation-results.md   # Latest test results
├── debug-log.md           # Investigation findings
└── handoff-*.md           # Session continuity files
```

### Command Files
```
.claude/commands/
├── status.md              # Reality check
├── sync.md                # Deep reconciliation
├── focus.md               # Minimal context
├── handoff.md             # Session continuity
├── document.md            # Progress recording
├── validate.md            # Test execution
├── debug.md               # Critical analysis
├── architect.md           # Design mode
├── implementer.md         # Coding mode
└── validator.md           # Testing mode
```

## Implementation Guidelines

### Command Development Rules

1. **Always Update State**
   ```markdown
   ## Output Format
   📝 Updated: current_plan.md
      - Progress: 20% → 25%
      - Current task: Updated based on reality
   ```

2. **Verify Before Claiming**
   ```markdown
   1. Run actual tests
   2. Check git status
   3. Update only verified progress
   ```

3. **Minimal Context Loading**
   ```markdown
   📄 Loaded (387 tokens):
      - Only essential files
      - Specific line ranges
      - Token count displayed
   ```

4. **Clear Action Items**
   ```markdown
   Next Action: Fix IDF calculation in line 52
   Blocker: Missing log term in formula
   Test: test_bm25_scoring expects log(N/df)
   ```

## Usage Patterns

### Daily Development Flow
```bash
# Morning startup
/status          # What's real state?
/focus [area]    # Load minimal context

# After context loss
/sync            # Reconcile everything

# End of session  
/document        # Record actual work
/handoff         # Prep for tomorrow
```

### Debugging Flow
```bash
/debug           # Activate skepticism
# ... systematic investigation ...
/document        # Record findings
```

### Design Flow
```bash
/architect       # Design perspective
# ... design work ...
/status          # Update reality
```

## Success Metrics

1. **State Accuracy**: 100% - files always match reality
2. **Context Size**: <500 tokens per command
3. **Update Clarity**: Every change explicitly shown
4. **Workflow Speed**: <5 seconds per command
5. **User Confidence**: Complete trust in state accuracy

## Anti-Patterns to Avoid

1. ❌ **Claiming Unverified Progress**
   ```bash
   # Bad: "Started implementation"
   # Good: "Tests passing: 4/10"
   ```

2. ❌ **Loading Comprehensive Context**
   ```bash
   # Bad: Load entire architecture doc
   # Good: Load 20 lines of current function
   ```

3. ❌ **Display Without Update**
   ```bash
   # Bad: Show status without fixing
   # Good: Show status AND update files
   ```

4. ❌ **Complex Command Chains**
   ```bash
   # Bad: /context then /role then /validate then...
   # Good: /status (does everything needed)
   ```

## Troubleshooting

### "Context was compacted"
- Run `/sync` to reconcile state
- Use `/focus` for minimal context
- Check token counts in output

### "State seems wrong"
- Run `/sync` for deep reconciliation
- Check git log manually if needed
- Verify tests are actually running

### "Not sure what to do next"
- Run `/status` for current reality
- Check blockers in output
- Use `/handoff` for clear next steps

## Migration Guide

From old system → new system:

1. **First Time Setup**
   ```bash
   /sync  # Reconcile everything with reality
   ```

2. **Replace Old Commands**
   - `/context` → `/focus [area]`
   - `/optimizer` → removed (use architect/implementer)
   - Manual status checks → `/status`

3. **New Workflow**
   - Start sessions with `/status`
   - End sessions with `/handoff`
   - Trust state files (they're verified now)

## Command Implementation Priority

### Week 1: Core Commands
1. Implement `/status` - Daily essential
2. Implement `/sync` - Recovery essential
3. Implement `/focus` - Work essential
4. Test with real workflow

### Week 2: Workflow & Refinement
1. Implement `/handoff` - Continuity
2. Implement `/document` - Progress tracking
3. Implement `/validate` - Quick checks
4. Add role commands if needed

## Final Notes

This system solves the fundamental problem of context management in AI-assisted development by ensuring state files always reflect reality. The commands are designed to be fast, focused, and trustworthy.

Remember: The goal is not perfect tracking of everything, but accurate tracking of what actually works. This enables confident development across multiple sessions without fear of context loss.