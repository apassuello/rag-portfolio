# RAG Demo Testing Guide

Quick validation guide for testing the RAG demo system after Ollama setup.

## Prerequisites

✅ **Already Complete**:
- 2,538 documents indexed in FAISS
- Embedder ready (sentence-transformers model)
- Query filtering implemented (threshold: 0.5)
- Demo scripts ready (`demo_rag.py`, `app.py`)

⏳ **Needs Setup**:
- Ollama installed and running
- Model downloaded (llama3.2:3b recommended)

## Quick Ollama Setup

```bash
# Install Ollama
brew install ollama

# Start service
brew services start ollama

# Download model
ollama pull llama3.2:3b

# Verify
ollama list
curl http://localhost:11434/api/tags
```

---

## Test 1: CLI Demo - Domain Queries (Expected: PASS)

Test queries that SHOULD get high relevance scores and good answers.

```bash
cd project-1-technical-rag

# Test 1: RISC-V vector instructions
python scripts/demo_rag.py --query "What are RISC-V vector instructions?" --show-sources

# Expected:
# ✓ Top relevance score: 0.65-0.75
# ✓ LLM generates answer with citations
# ✓ Sources shown with high scores
# ✓ Total time: <5s
```

```bash
# Test 2: Privilege levels
python scripts/demo_rag.py --query "Explain privilege levels in RISC-V"

# Expected:
# ✓ Top relevance score: 0.65-0.75
# ✓ Answer references privilege modes (M, S, U)
# ✓ Citations point to correct PDFs
```

```bash
# Test 3: CSR registers
python scripts/demo_rag.py --query "What are CSR control status registers?"

# Expected:
# ✓ Top relevance score: 0.65-0.75
# ✓ Answer explains CSR purpose and usage
```

---

## Test 2: CLI Demo - Out-of-Domain Queries (Expected: FILTERED)

Test queries that SHOULD be filtered out (relevance < 0.5).

```bash
# Test 4: Cooking (should be filtered)
python scripts/demo_rag.py --query "How do I cook pasta?"

# Expected:
# ⚠️ Query filtered: Top relevance score (0.15-0.25) below threshold (0.5)
# ✓ Answer: "I don't have information about this topic..."
# ✓ No LLM call (generation time: 0ms)
# ✓ No sources displayed
```

```bash
# Test 5: Sports (should be filtered)
python scripts/demo_rag.py --query "Who won the world cup?"

# Expected:
# ⚠️ Query filtered
# ✓ Same filtering behavior
```

```bash
# Test 6: Geography (should be filtered)
python scripts/demo_rag.py --query "What is the capital of France?"

# Expected:
# ⚠️ Query filtered
# ✓ Same filtering behavior
```

---

## Test 3: Interactive Mode

Test the conversation-style interface.

```bash
python scripts/demo_rag.py --interactive
```

**Test Sequence**:
1. Type: `What are RISC-V vector instructions?`
   - Expected: High score (0.65-0.75), good answer
2. Type: `How does memory management work?`
   - Expected: High score, good answer
3. Type: `sources` (toggle source display)
   - Expected: Sources now shown in output
4. Type: `How do I cook pasta?`
   - Expected: Filtered (score 0.15-0.25), no LLM call
5. Type: `quit`
   - Expected: Exit cleanly

---

## Test 4: Streamlit Web App

Test the web UI.

```bash
streamlit run app.py
```

**Manual Test Steps**:
1. **App Loads**:
   - Check sidebar shows: 2,538 documents indexed
   - Check LLM status (should show Ollama model name)

2. **Domain Query**:
   - Click sample query: "What are RISC-V vector instructions?"
   - Expected:
     - Answer appears in conversation
     - Sources shown with scores (0.65-0.75)
     - Performance metrics displayed
     - Answer has citations [1], [2], etc.

3. **Out-of-Domain Query**:
   - Type: "How do I cook pasta?"
   - Expected:
     - Filtered warning appears
     - Top score shown (0.15-0.25)
     - No sources displayed
     - Generation time: 0ms

4. **Source Toggle**:
   - Uncheck "Show source documents"
   - Submit query
   - Expected: Sources hidden but answer still shown

5. **Session History**:
   - Submit multiple queries
   - Expected: All queries saved in collapsible expanders
   - Latest query expanded by default

---

## Test 5: Custom Threshold Testing

Test configurable relevance threshold.

```bash
# Lower threshold (more permissive)
python scripts/demo_rag.py --query "How do I cook pasta?" --relevance-threshold 0.2

# Expected: Query NOT filtered (passes with score ~0.21)

# Higher threshold (more strict)
python scripts/demo_rag.py --query "What are RISC-V vector instructions?" --relevance-threshold 0.8

# Expected: Query filtered (score 0.71 < 0.8)
```

---

## Test 6: Fallback Modes

### Test Without Ollama (Mock Mode)

```bash
# Stop Ollama
brew services stop ollama

# Run demo
python scripts/demo_rag.py --query "What are RISC-V vector instructions?"

# Expected:
# ⚠️ Ollama not available
# ℹ️ Falling back to mock LLM
# ✓ Retrieval works (shows scores)
# ✓ Mock answer displayed
# ✓ Instructions to install Ollama shown
```

### Test With OpenAI (if API key available)

```bash
export OPENAI_API_KEY="sk-..."
python scripts/demo_rag.py --query "What are RISC-V vector instructions?" --use-openai

# Expected:
# ✓ OpenAI ready (model: gpt-3.5-turbo)
# ✓ Real AI answer generated
# ✓ Citations included
```

---

## Test 7: Performance Validation

Expected performance benchmarks:

| Operation | Expected Time |
|-----------|---------------|
| Retrieval (5 docs) | <5ms |
| Embedding generation | <50ms |
| LLM generation (Ollama) | 2-5s |
| Total (domain query) | <6s |
| Total (filtered query) | <100ms |

---

## Validation Checklist

After testing, verify:

- [ ] **Domain queries work**: 3 domain queries with 0.65-0.75 scores
- [ ] **Out-of-domain filtering works**: 3 queries correctly filtered
- [ ] **LLM integration works**: Ollama generates answers with citations
- [ ] **Interactive mode works**: Can ask multiple queries in conversation
- [ ] **Streamlit app works**: Web UI loads and processes queries
- [ ] **Query filtering works**: Threshold correctly rejects irrelevant queries
- [ ] **Performance acceptable**: Total time <6s for domain queries
- [ ] **Fallback works**: Mock LLM works when Ollama unavailable
- [ ] **Sources accurate**: Citations reference correct PDFs

---

## Troubleshooting

### Issue: "Ollama not available"
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# If not, start it
brew services start ollama

# Check status
brew services list | grep ollama
```

### Issue: "No models found"
```bash
# List models
ollama list

# If empty, download
ollama pull llama3.2:3b
```

### Issue: "FileNotFoundError: documents.pkl"
```bash
# Check indices exist
ls -lh project-1-technical-rag/data/indices/

# If missing, rebuild
python scripts/build_indices.py
```

### Issue: Slow LLM generation
- Try smaller model: `ollama pull llama3.2:1b`
- Or use OpenAI: `--use-openai`

### Issue: Import errors
```bash
# Reinstall dependencies
pip install -r requirements-streamlit.txt
```

---

## Success Criteria

**Minimum for Phase 1 Complete**:
- ✅ Domain queries: 3/3 high relevance (0.65-0.75)
- ✅ Out-of-domain: 3/3 correctly filtered (<0.5)
- ✅ LLM generates answers with citations
- ✅ Total query time <6s
- ✅ Streamlit app functional

**Ready for Phase 2** (Demo video & cloud deployment):
- ✅ All Phase 1 criteria met
- ✅ Demo script polished
- ✅ Screenshots captured
- ✅ Performance benchmarks documented

---

## Next Steps After Validation

1. **Document Results**: Save test outputs and screenshots
2. **Create Demo Video**: Record 5-10 minute walkthrough
3. **Prepare Cloud Deployment**: HuggingFace Spaces setup
4. **Update Portfolio**: Add demo links and metrics

---

## Quick Test Script

Run all tests automatically:

```bash
#!/bin/bash
# quick_test.sh

echo "=== RAG Demo Quick Test ==="

echo -e "\n1. Domain Query Test..."
python scripts/demo_rag.py --query "What are RISC-V vector instructions?"

echo -e "\n2. Filtered Query Test..."
python scripts/demo_rag.py --query "How do I cook pasta?"

echo -e "\n3. Performance Test..."
time python scripts/demo_rag.py --query "Explain privilege levels in RISC-V"

echo -e "\n=== All Tests Complete ==="
```

Make executable: `chmod +x quick_test.sh`
Run: `./quick_test.sh`
