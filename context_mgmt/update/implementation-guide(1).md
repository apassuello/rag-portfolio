# RAG Portfolio Command System - Implementation Guide

## Quick Start Implementation

### Step 1: Create Command Directory Structure

```bash
# From your project root
cd project-1-technical-rag/.claude
mkdir -p commands
mkdir -p sessions

# Create empty state files
touch current_plan.md
touch sessions/recent-work.md
touch sessions/validation-results.md
touch sessions/debug-log.md
```

### Step 2: Create current_plan.md Template

```yaml
# RAG Portfolio Project 1 - Current State

## Project Overview
**Project**: Technical Documentation RAG System
**Current Phase**: Week 2 - Advanced Features
**Overall Progress**: 22%  # Updated by /status command

## Current Task
**Task**: Implement BM25 scoring algorithm
**Focus Area**: hybrid-search
**Task Progress**: 60%  # Based on actual tests
**Blockers**: 
  - IDF calculation missing log term

## Validation Commands
validation_commands:
  - "pytest tests/test_hybrid_retriever.py -xvs"
  - "pytest tests/test_query_enhancement.py"
  - "python tests/run_comprehensive_tests.py"

## State Tracking
**Last Updated**: 2024-01-22T10:30:00Z
**Last Sync**: 2024-01-22T09:00:00Z
**Current Focus**: hybrid-search
**Focus Since**: 2024-01-22T10:30:00Z
```

### Step 3: Install Commands

Place each command file in `.claude/commands/`:

1. `status.md` - Reality check (Phase 1)
2. `sync.md` - Deep reconciliation (Phase 1)
3. `focus.md` - Minimal context (Phase 1)
4. `handoff.md` - Session continuity (Phase 2)
5. `document.md` - Progress recording (Phase 2)
6. `debug.md` - Critical analysis (Phase 2)

### Step 4: First Time Setup

```bash
# In Claude Code, run:
/sync  # Reconcile everything with reality

# This will:
# - Analyze your git history
# - Run all tests
# - Create accurate state files
# - Show you the real project state
```

## Daily Workflow Examples

### Morning Startup
```bash
# 1. Check reality
/status

# Output shows:
# - What tests actually pass
# - Real progress percentage  
# - Current blockers
# - Next action

# 2. Load minimal context
/focus hybrid-search

# Now you're ready to work with <500 tokens loaded
```

### After Conversation Compaction
```bash
# Lost context? No problem:
/sync

# This will:
# - Check git for what you actually did
# - Run tests to verify current state
# - Fix all state files
# - Show you exactly where you are
```

### Debugging Session
```bash
# Something's broken but unclear why:
/debug

# Activates:
# - Extreme skepticism  
# - Systematic debugging
# - Trust nothing mindset
# - Specific investigation steps
```

### End of Day
```bash
# 1. Document what you actually did
/document

# 2. Create handoff for tomorrow
/handoff

# Copy the generated prompt for perfect continuity
```

## State File Examples

### sessions/recent-work.md
```markdown
# Recent Work Log

## 2024-01-22
- Fixed BM25 IDF calculation (added log term)
- All BM25 tests now passing
- Started hybrid score combination
- Status: BM25 complete, combination in progress

## 2024-01-21  
- Implemented BM25 tokenizer
- Added stopword filtering
- Status: Tokenizer working, scoring had bugs

## 2024-01-20
- Set up hybrid retriever structure
- Created test suite
- Status: Structure ready, implementation started
```

### sessions/validation-results.md
```markdown
# Validation Results

## Last Run: 2024-01-22T15:30:00Z

### Test Summary
- Total Tests: 47
- Passing: 44
- Failing: 3
- Coverage: 87%

### Failing Tests
1. `test_hybrid_combination` - combine_scores not implemented
2. `test_query_enhancement_semantic` - embedding integration needed
3. `test_answer_generation_streaming` - buffer size issue

### Performance Metrics
- BM25 Retrieval: 45ms average
- Dense Retrieval: 23ms average  
- Hybrid Retrieval: Not yet measured

### Architecture Compliance
- Component Boundaries: ✓ 100%
- Interface Compliance: ✓ 100%
- Performance Targets: ⚠️ 2/5 met
```

## Troubleshooting

### "Command not found"
- Ensure file is in `.claude/commands/` directory
- Filename must match command (e.g., `status.md` for `/status`)

### "State file not found"  
- Run `/sync` to create all state files
- Check file paths in command match your structure

### "Tests timing out"
- Add timeout to validation commands
- Use subset of tests for quick checks: `pytest -k "quick"`

### "Too much context loaded"
- Check `/focus` is using line ranges
- Reduce loaded content in command
- Verify token count is displayed

## Advanced Usage

### Custom Focus Areas

Add to `/focus` command:
```markdown
### `ml-pipeline`
- Focus: ML model integration
- Files: src/ml/pipeline.py, tests/test_ml_pipeline.py
- Context: Model loading, inference optimization
```

### Project-Specific Validation

Update `current_plan.md`:
```yaml
validation_commands:
  - "pytest tests/unit/ -xvs"  # Quick unit tests
  - "pytest tests/integration/ -x"  # Integration suite
  - "python validate_performance.py"  # Custom metrics
  - "python check_memory_usage.py"  # Resource validation
```

### Team Handoffs

Enhanced `/handoff` for team collaboration:
```markdown
=== TEAM HANDOFF ===
For: @teammate
Branch: feature/hybrid-search
PR: #123 (draft)

Current state: [details]
Next steps: [specific tasks]
Context: /focus hybrid-search
===
```

## Best Practices

1. **Run `/status` first thing** - Start with reality
2. **Use `/sync` weekly** - Prevent drift accumulation
3. **Keep `/focus` minimal** - Under 500 tokens always
4. **Document before leaving** - Future you will thank you
5. **Trust the system** - State files are verified reality

## Success Metrics

Track these to ensure system is working:

- **State Accuracy**: Files match git/tests 100%
- **Context Size**: Average <400 tokens per `/focus`
- **Handoff Success**: Can resume work immediately
- **Debug Efficiency**: Issues found faster with `/debug`
- **No Context Loss**: Never lose work between sessions

## Next Steps

1. **Week 1**: Implement core commands (`/status`, `/sync`, `/focus`)
2. **Week 2**: Add workflow commands as needed
3. **Ongoing**: Refine based on actual usage
4. **Monthly**: Review and optimize commands

Remember: The goal is accurate state tracking that gives you complete confidence in your project status, enabling seamless work across multiple sessions without fear of context loss.