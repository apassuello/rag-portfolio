# Full Deployment Demo Implementation Plan

## Objective
Create a complete, working RAG system demonstration that showcases the last 3 weeks of work, ready for portfolio presentation and deployment.

---

## Phase 1: Local RAG Demo (1-2 hours)

### 1.1 LLM Integration Setup (30 min)
**Goal**: Get Ollama running with a local LLM

**Tasks**:
- [ ] Install Ollama on Mac
  ```bash
  brew install ollama
  brew services start ollama
  ```
- [ ] Download recommended model
  ```bash
  ollama pull llama3.2:3b
  ```
- [ ] Verify model works
  ```bash
  ollama run llama3.2:3b "What is RISC-V?"
  ```
- [ ] Test API endpoint
  ```bash
  curl http://localhost:11434/api/tags
  ```

**Success Criteria**:
- ✓ Ollama service running
- ✓ Model downloaded and responds
- ✓ API accessible

### 1.2 Test Demo Script (15 min)
**Goal**: Validate demo_rag.py works end-to-end

**Tasks**:
- [ ] Run demo with built indices
  ```bash
  cd project-1-technical-rag
  python scripts/demo_rag.py --query "What are RISC-V vector instructions?"
  ```
- [ ] Verify output shows:
  - Query text
  - Retrieved documents (5 sources)
  - Relevance scores (should be 0.6-0.7)
  - Generated answer with citations
  - Performance metrics

**Success Criteria**:
- ✓ Loads 2,538 indexed documents (not re-processing)
- ✓ Retrieval shows good scores (0.6-0.7)
- ✓ LLM generates coherent answer
- ✓ Citations reference correct sources
- ✓ No errors or crashes

### 1.3 Interactive Demo Testing (30 min)
**Goal**: Test multiple queries interactively

**Tasks**:
- [ ] Run interactive mode
  ```bash
  python scripts/demo_rag.py --interactive
  ```
- [ ] Test domain queries (should score 0.6-0.7):
  - "What are RISC-V vector instructions?"
  - "Explain privilege levels in RISC-V"
  - "How does memory management work?"
  - "What are CSR control status registers?"
  - "How do interrupts work in RISC-V?"

- [ ] Test out-of-domain queries (should score 0.1-0.2):
  - "How do I cook pasta?"
  - "Who won the world cup?"
  - "What is the capital of France?"

- [ ] Document results in testing log

**Success Criteria**:
- ✓ Domain queries get relevant answers
- ✓ Out-of-domain queries handled gracefully (low scores)
- ✓ Citations are accurate
- ✓ Performance consistent (<5s per query)

### 1.4 Add Query Filtering (15 min)
**Goal**: Implement relevance threshold to reject irrelevant queries

**Tasks**:
- [ ] Add filtering logic to demo_rag.py:
  ```python
  # After retrieval, check top score
  if retrieved_docs and retrieved_docs[0][1] < 0.5:
      return {
          'answer': "I don't have information about this topic in my knowledge base.",
          'sources': [],
          'filtered': True,
          'top_score': retrieved_docs[0][1]
      }
  ```
- [ ] Test filtering works:
  - Domain query: proceeds to LLM
  - Random query: filtered out before LLM

**Success Criteria**:
- ✓ Irrelevant queries rejected (score < 0.5)
- ✓ No LLM calls for filtered queries
- ✓ Clear message to user

---

## Phase 2: Streamlit UI Demo (2-3 hours)

### 2.1 Install Dependencies (15 min)
**Goal**: Set up Streamlit environment

**Tasks**:
- [ ] Install Streamlit dependencies
  ```bash
  pip install -r requirements-streamlit.txt
  ```
- [ ] Verify Streamlit works
  ```bash
  streamlit hello
  ```

**Success Criteria**:
- ✓ All dependencies installed
- ✓ Streamlit runs

### 2.2 Update app.py for Built Indices (30 min)
**Goal**: Ensure app.py uses your 2,538 indexed documents

**Tasks**:
- [ ] Verify app.py loads from `data/indices/`
- [ ] Add query filtering threshold (0.5)
- [ ] Add session state for query history
- [ ] Add performance metrics display
- [ ] Test locally:
  ```bash
  streamlit run app.py
  ```

**Success Criteria**:
- ✓ Loads indices on startup
- ✓ Shows 2,538 documents in sidebar
- ✓ Query filtering works
- ✓ UI is responsive

### 2.3 UI Testing (45 min)
**Goal**: Comprehensive UI validation

**Tasks**:
- [ ] Test all sample queries
- [ ] Test custom queries
- [ ] Test query filtering display
- [ ] Test source display
- [ ] Test performance metrics
- [ ] Test session history
- [ ] Take screenshots for portfolio

**Success Criteria**:
- ✓ All features work
- ✓ UI is polished
- ✓ No errors in console
- ✓ Screenshots captured

### 2.4 Polish & Documentation (30 min)
**Goal**: Production-ready local app

**Tasks**:
- [ ] Add demo instructions to README
- [ ] Create quick-start guide
- [ ] Document sample queries
- [ ] Add troubleshooting section

**Success Criteria**:
- ✓ Anyone can run the demo
- ✓ Clear documentation

---

## Phase 3: Demonstration Scenarios (1 hour)

### 3.1 Create Demo Script (30 min)
**Goal**: Scripted demonstration for interviews/presentations

**Demo Flow**:
```
1. Show System Status
   - "This RAG system has 2,538 indexed technical documents"
   - "Built using FAISS for vector search, Sentence Transformers for embeddings"
   - Show stats in Streamlit sidebar

2. Demonstrate Domain Queries
   - "Let me ask: What are RISC-V vector instructions?"
   - Show: High relevance (0.71), good sources, accurate answer
   - Show: Citations reference correct PDFs
   - "Let me ask: Explain privilege levels in RISC-V"
   - Show: Again high relevance (0.68), accurate answer

3. Demonstrate Query Filtering
   - "Let me ask: How do I cook pasta?"
   - Show: Low relevance (0.21), query rejected
   - "The system correctly identifies this as out-of-domain"

4. Show Performance
   - "Notice the performance metrics"
   - Retrieval: <1ms
   - Generation: ~2-3s
   - Total: <5s

5. Show Sources
   - Click on a source
   - Show actual document content
   - "System maintains transparency with source citations"

6. Highlight Recent Work
   - "This represents 3 weeks of work:"
   - "2,538 documents indexed (data pipeline execution)"
   - "0.65-0.75 retrieval quality (validated)"
   - "Query filtering (smart rejection)"
   - "95.7% type hint coverage (code quality)"
   - "Zero security vulnerabilities (hardened)"
```

**Tasks**:
- [ ] Write demo script
- [ ] Practice demo flow
- [ ] Time demo (should be 5-10 minutes)

**Success Criteria**:
- ✓ Demonstrates key features
- ✓ Shows recent work
- ✓ Polished presentation

### 3.2 Record Demo Video (30 min)
**Goal**: Portfolio artifact

**Tasks**:
- [ ] Record screen while running demo
- [ ] Narrate key points
- [ ] Show multiple queries
- [ ] Show source display
- [ ] Show performance metrics
- [ ] Export video (MP4)

**Success Criteria**:
- ✓ 5-10 minute video
- ✓ Clear audio
- ✓ Shows all features
- ✓ Ready for portfolio

---

## Phase 4: Cloud Deployment (Optional, 2-3 hours)

### 4.1 HuggingFace Spaces Setup (1 hour)
**Goal**: Deploy to cloud for public demo

**Tasks**:
- [ ] Create HuggingFace account
- [ ] Create new Space (Streamlit)
- [ ] Configure Space settings
- [ ] Prepare deployment files:
  - app.py
  - requirements.txt
  - README.md
  - Pre-built indices (upload)

**Success Criteria**:
- ✓ Space created
- ✓ Files ready

### 4.2 Deploy & Test (1 hour)
**Goal**: Working cloud demo

**Tasks**:
- [ ] Push code to HuggingFace
- [ ] Upload indices to Space
- [ ] Configure secrets (if using OpenAI)
- [ ] Test deployed app
- [ ] Fix any deployment issues

**Success Criteria**:
- ✓ App deploys successfully
- ✓ Indices load correctly
- ✓ Queries work
- ✓ Public URL accessible

### 4.3 Optimization (30 min - if needed)
**Goal**: Improve cloud performance

**Tasks**:
- [ ] Optimize loading time
- [ ] Add caching
- [ ] Reduce cold start
- [ ] Monitor resource usage

**Success Criteria**:
- ✓ App loads in <30s
- ✓ Queries work smoothly
- ✓ No timeout issues

---

## Success Metrics

### Technical Metrics
- [ ] Retrieval precision@5 > 0.8 (validated at 0.65-0.75)
- [ ] Query filtering accuracy > 90%
- [ ] End-to-end latency < 5s
- [ ] System handles 10+ consecutive queries

### Portfolio Metrics
- [ ] Live local demo works
- [ ] Demo video recorded
- [ ] Documentation complete
- [ ] Optional: Public cloud URL

---

## Deliverables

### Must Have (Phase 1-2)
1. Working local demo (demo_rag.py)
2. Streamlit UI (app.py)
3. Demo script for presentations
4. Screenshots/video

### Nice to Have (Phase 3-4)
5. Demo video (5-10 min)
6. Cloud deployment URL
7. Performance benchmarks

---

## Timeline

**Optimistic** (everything works first time):
- Phase 1: 1.5 hours
- Phase 2: 2 hours
- Phase 3: 1 hour
- Phase 4: 2 hours (optional)
- **Total**: 4.5-6.5 hours

**Realistic** (with debugging):
- Phase 1: 2 hours
- Phase 2: 3 hours
- Phase 3: 1 hour
- Phase 4: 3 hours (optional)
- **Total**: 6-9 hours

**Recommended for today**: Phase 1 + Phase 2 (4-5 hours)
**Leave for later**: Phase 3-4 (when ready to present)

---

## Risk Mitigation

### If Ollama doesn't work:
- Fallback to OpenAI API (requires API key)
- Or use mock LLM (shows retrieval only)

### If Streamlit has issues:
- Use CLI demo only (demo_rag.py)
- Create simple Gradio UI instead

### If deployment fails:
- Demo locally via screen share
- Record video instead of live URL

---

## Next Immediate Action

**Start with Phase 1.1** (30 minutes):
```bash
# Install Ollama
brew install ollama
brew services start ollama
ollama pull llama3.2:3b

# Test it works
ollama run llama3.2:3b "Hello, how are you?"
```

Then run:
```bash
cd project-1-technical-rag
python scripts/demo_rag.py --query "What are RISC-V vector instructions?"
```

If that works, proceed to Phase 1.2.

---

## Portfolio Impact

**Before deployment**: 85/100
**After Phase 1-2**: 87/100 (working demo)
**After Phase 3**: 90/100 (polished demo + video)
**After Phase 4**: 93/100 (cloud deployed)

The key value: **Demonstrates your actual work** (2,538 indexed docs, retrieval quality, query filtering) in a professional, shareable format.
