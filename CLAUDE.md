# RAG Portfolio Development Context

## Project Overview
Building a 3-project RAG portfolio for ML Engineer positions in Swiss tech market.
Currently working on **Project 1: Technical Documentation RAG System**.

## Developer Background
- Arthur Passuello, transitioning from Embedded Systems to AI/ML
- 2.5 years medical device firmware experience
- Recent 7-week ML intensive (transformers, optimization from scratch)
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

## Current Session Goals
Building document processing foundation with 3 focused modules:
1. PDF text extraction with metadata
2. Technical document chunking strategy  
3. Embedding generation with caching

## Code Style Preferences
- Type hints for all functions
- Comprehensive error handling
- Clear docstrings with examples
- Maximum 50 lines per function
- Prefer composition over inheritance
- Apple Silicon optimizations where applicable

## Test Strategy
- Write tests first (TDD approach)
- Real-world data validation
- Performance benchmarks
- Error condition coverage

## Success Criteria for Today
Each module must have:
- Clear specification
- Working implementation 
- Passing test with real data
- Performance measurement