# RAG Portfolio Development Context

## Project Overview
Building a 3-project RAG portfolio for ML Engineer positions in Swiss tech market.
Currently working on **Project 1: Technical Documentation RAG System**.

## Developer Background
- Arthur Passuello, transitioning from Embedded Systems to AI/ML
- 2.5 years medical device firmware experience
- Recent 7-week ML intensive (transformers from scratch, multimodal systems)
- Strong optimization and production mindset from embedded background
- Using M4-Pro Apple Silicon Mac with MPS acceleration

## Current Development Environment
- Python 3.11 in conda environment `rag-portfolio`
- PyTorch with Metal/MPS support
- Key libraries: transformers, sentence-transformers, langchain, faiss-cpu
- IDE: Cursor with AI assistant
- Git: SSH-based workflow

## Project 1 Technical Stack
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2 (local inference)
- **Vector Store**: FAISS (local development, migrate to Pinecone later)
- **LLM**: Start local, add API fallback
- **Deployment Target**: Streamlit on HuggingFace Spaces
- **Evaluation**: RAGAS framework

## Development Philosophy
- **Production-first**: Every component deployment-ready
- **Modular design**: Small, testable, single-purpose functions
- **Swiss market aligned**: Quality, reliability, thorough documentation
- **Optimization mindset**: Leverage embedded systems background

## Current Status: Week 1 Foundation Complete ✅

### Implemented Modules (All Tested & Working)
1. **PDF Parser** (`shared_utils/document_processing/pdf_parser.py`)
   - Function: `extract_text_with_metadata(pdf_path: Path) -> Dict[str, Any]`
   - Performance: 470KB text in 0.282s
   - Features: Page-by-page extraction, metadata preservation, error handling
   - Test: Validates RISC-V PDF processing

2. **Text Chunker** (`shared_utils/document_processing/chunker.py`)
   - Function: `chunk_technical_text(text: str, chunk_size: int, overlap: int) -> List[Dict]`
   - Features: Sentence-aware boundaries, metadata tracking, content hashing
   - Quality: Preserves technical context, configurable overlap
   - Test: Validates chunking integrity and sentence completeness

3. **Embedding Generator** (`shared_utils/embeddings/generator.py`)
   - Function: `generate_embeddings(texts: List[str], use_mps: bool) -> np.ndarray`
   - Performance: 129.6 texts/second with Apple Silicon MPS
   - Features: Model caching, batch processing, content-based caching
   - Test: Validates performance, consistency, and caching behavior

4. **Basic RAG System** (`project-1-technical-rag/src/basic_rag.py`)
   - Class: `BasicRAG` with methods `index_document()` and `query()`
   - Features: FAISS integration, document storage, similarity search
   - Integration: Combines all previous modules into working pipeline
   - Test: End-to-end validation with RISC-V documentation

### Architecture Implemented
```
📄 PDF Document
    ↓ (extract_text_with_metadata)
📝 Structured Text + Metadata
    ↓ (chunk_technical_text)  
🧩 Semantic Chunks with Overlap
    ↓ (generate_embeddings)
🔢 384-dim Embeddings (MPS accelerated)
    ↓ (FAISS IndexFlatIP)
🔍 Searchable Vector Index
    ↓ (cosine similarity search)
📊 Ranked Results with Metadata
```

### Performance Metrics Achieved
- **PDF Processing**: 238 pages → 470KB text in 0.282s
- **Embedding Generation**: 129.6 texts/second (target: 50+)
- **End-to-End Pipeline**: <10 seconds for full document processing
- **Memory Efficiency**: <500MB for complete pipeline
- **Test Coverage**: 100% pass rate on all modules

## Current Week 2 Focus: Advanced RAG Implementation

### Next Implementation: Hybrid Search (Day 8-9)
**Target**: Dense + Sparse retrieval with Reciprocal Rank Fusion
**Timeline**: 2-3 hours for complete implementation
**Location**: `shared-utils/retrieval/hybrid_search.py`

**Requirements from Original Specification**:
- Dense retrieval: Sentence transformer embeddings (reuse existing)
- Sparse retrieval: BM25 algorithm (rank-bm25 library)
- Fusion: Reciprocal Rank Fusion (RRF) with weights 0.7 dense, 0.3 sparse
- Integration: Extend BasicRAG with hybrid_query() method
- Testing: Demonstrate improved retrieval vs pure semantic search

### Upcoming Week 2 Tasks
- **Day 10-11**: Query Enhancement (expansion, synonyms, acronyms)
- **Day 12-14**: Answer Generation (local LLM, streaming, citations)

## Implementation Quality Standards
- **Type hints** for all functions
- **Comprehensive error handling** with informative messages
- **Clear docstrings** with examples and performance notes
- **Modular design** with single-purpose functions
- **Apple Silicon optimizations** where applicable
- **Test-driven development** with real-world validation
- **Performance benchmarking** with quantified metrics

## Test Strategy
- **Unit tests** for individual module functionality
- **Integration tests** for end-to-end pipeline validation
- **Performance tests** with Apple Silicon benchmarks
- **Real-world validation** using RISC-V technical documentation
- **Error condition testing** for robustness

## Success Criteria for Week 2
- **Hybrid Search**: Outperform basic semantic search on technical queries
- **Query Enhancement**: Demonstrate improved question understanding
- **Answer Generation**: Complete RAG system with local LLM integration
- **Evaluation**: RAGAS metrics showing system quality
- **Deployment**: Working Streamlit demo

## Repository Structure Status
```
rag-portfolio/
├── project-1-technical-rag/
│   ├── src/
│   │   └── basic_rag.py ✅
│   ├── tests/
│   │   ├── test_pdf_parser.py ✅
│   │   ├── test_chunker.py ✅  
│   │   ├── test_embeddings.py ✅
│   │   ├── test_basic_rag.py ✅
│   │   └── test_integration.py ✅
│   ├── data/test/
│   │   └── riscv-base-instructions.pdf ✅
│   └── demo_basic_rag.py ✅
├── shared_utils/
│   ├── document_processing/
│   │   ├── pdf_parser.py ✅
│   │   └── chunker.py ✅
│   ├── embeddings/
│   │   └── generator.py ✅
│   └── retrieval/ (to be created)
│       └── hybrid_search.py (next task)
└── CLAUDE.md ✅
```

## Code Style Preferences
- **Maximum 50 lines per function** for focused implementation
- **Comprehensive docstrings** with Args, Returns, Raises, Performance notes
- **Error handling** that provides actionable information
- **Apple Silicon optimization** using MPS where applicable
- **Content-based caching** for performance where appropriate
- **Modular composition** over inheritance for flexibility

## Current Session Context
- **Week**: 1 complete, starting Week 2
- **Current Task**: Hybrid Search implementation (Day 8-9)
- **Previous Success**: All foundation modules working and tested
- **Next Goal**: Advanced retrieval outperforming basic semantic search
- **Timeline**: Following original 3-4 week Project 1 specification
- **Quality Standard**: Production-ready code suitable for Swiss tech portfolio