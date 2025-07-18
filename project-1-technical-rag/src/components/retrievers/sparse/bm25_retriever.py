"""
BM25 Sparse Retriever implementation for Modular Retriever Architecture.

This module provides a direct implementation of BM25 sparse retrieval
extracted from the existing sparse retrieval system for improved modularity.
"""

import logging
import re
import time
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from rank_bm25 import BM25Okapi

from src.core.interfaces import Document
from .base import SparseRetriever

logger = logging.getLogger(__name__)


class BM25Retriever(SparseRetriever):
    """
    BM25-based sparse retrieval implementation.
    
    This is a direct implementation that handles BM25 keyword search
    without external adapters. It provides efficient sparse retrieval
    for technical documentation with optimized tokenization.
    
    Features:
    - Technical term preservation (handles RISC-V, ARM Cortex-M, etc.)
    - Configurable BM25 parameters (k1, b)
    - Normalized scoring for fusion compatibility
    - Efficient preprocessing and indexing
    - Performance monitoring
    
    Example:
        config = {
            "k1": 1.2,
            "b": 0.75,
            "lowercase": True,
            "preserve_technical_terms": True
        }
        retriever = BM25Retriever(config)
        retriever.index_documents(documents)
        results = retriever.search("RISC-V processor", k=5)
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize BM25 sparse retriever.
        
        Args:
            config: Configuration dictionary with:
                - k1: Term frequency saturation parameter (default: 1.2)
                - b: Document length normalization factor (default: 0.75)
                - lowercase: Whether to lowercase text (default: True)
                - preserve_technical_terms: Whether to preserve technical terms (default: True)
        """
        self.config = config
        self.k1 = config.get("k1", 1.2)
        self.b = config.get("b", 0.75)
        self.lowercase = config.get("lowercase", True)
        self.preserve_technical_terms = config.get("preserve_technical_terms", True)
        
        # Validation
        if self.k1 <= 0:
            raise ValueError("k1 must be positive")
        if not 0 <= self.b <= 1:
            raise ValueError("b must be between 0 and 1")
        
        # BM25 components
        self.bm25: Optional[BM25Okapi] = None
        self.documents: List[Document] = []
        self.tokenized_corpus: List[List[str]] = []
        self.chunk_mapping: List[int] = []
        
        # Compile regex patterns for technical term preservation
        if self.preserve_technical_terms:
            self._tech_pattern = re.compile(r'[a-zA-Z0-9][\w\-_.]*[a-zA-Z0-9]|[a-zA-Z0-9]')
            self._punctuation_pattern = re.compile(r'[^\w\s\-_.]')
        else:
            self._tech_pattern = re.compile(r'\b\w+\b')
            self._punctuation_pattern = re.compile(r'[^\w\s]')
        
        logger.info(f"BM25Retriever initialized with k1={self.k1}, b={self.b}")
    
    def index_documents(self, documents: List[Document]) -> None:
        """
        Index documents for BM25 sparse retrieval.
        
        Args:
            documents: List of documents to index
        """
        if not documents:
            raise ValueError("Cannot index empty document list")
        
        start_time = time.time()
        
        # Store documents
        self.documents = documents.copy()
        
        # Extract and preprocess texts
        texts = [doc.content for doc in documents]
        self.tokenized_corpus = [self._preprocess_text(text) for text in texts]
        
        # Filter out empty tokenized texts and track mapping
        valid_corpus = []
        self.chunk_mapping = []
        
        for i, tokens in enumerate(self.tokenized_corpus):
            if tokens:  # Only include non-empty tokenized texts
                valid_corpus.append(tokens)
                self.chunk_mapping.append(i)
        
        if not valid_corpus:
            raise ValueError("No valid text content found in documents")
        
        # Create BM25 index
        self.bm25 = BM25Okapi(valid_corpus, k1=self.k1, b=self.b)
        
        elapsed = time.time() - start_time
        total_tokens = sum(len(tokens) for tokens in self.tokenized_corpus)
        tokens_per_sec = total_tokens / elapsed if elapsed > 0 else 0
        
        logger.info(f"Indexed {len(documents)} documents ({len(valid_corpus)} valid) in {elapsed:.3f}s")
        logger.debug(f"Processing rate: {tokens_per_sec:.1f} tokens/second")
    
    def search(self, query: str, k: int = 5) -> List[Tuple[int, float]]:
        """
        Search for documents using BM25 sparse retrieval.
        
        Args:
            query: Search query string
            k: Number of results to return
            
        Returns:
            List of (document_index, score) tuples sorted by relevance
        """
        if self.bm25 is None:
            raise ValueError("Must call index_documents() before searching")
        
        if not query or not query.strip():
            return []
        
        if k <= 0:
            raise ValueError("k must be positive")
        
        # Preprocess query using same method as documents
        query_tokens = self._preprocess_text(query)
        if not query_tokens:
            return []
        
        # Get BM25 scores for all documents
        scores = self.bm25.get_scores(query_tokens)
        
        if len(scores) == 0:
            return []
        
        # Normalize scores to [0,1] range for fusion compatibility
        max_score = np.max(scores)
        if max_score > 0:
            normalized_scores = scores / max_score
        else:
            normalized_scores = scores
        
        # Create results with original document indices
        results = [
            (self.chunk_mapping[i], float(normalized_scores[i]))
            for i in range(len(scores))
        ]
        
        # Sort by score (descending) and return top_k
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:k]
    
    def get_document_count(self) -> int:
        """Get the number of indexed documents."""
        return len(self.documents)
    
    def clear(self) -> None:
        """Clear all indexed documents."""
        self.documents.clear()
        self.tokenized_corpus.clear()
        self.chunk_mapping.clear()
        self.bm25 = None
        logger.info("BM25 index cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the BM25 retriever.
        
        Returns:
            Dictionary with retriever statistics
        """
        stats = {
            "k1": self.k1,
            "b": self.b,
            "lowercase": self.lowercase,
            "preserve_technical_terms": self.preserve_technical_terms,
            "total_documents": len(self.documents),
            "valid_documents": len(self.chunk_mapping),
            "is_indexed": self.bm25 is not None
        }
        
        if self.tokenized_corpus:
            total_tokens = sum(len(tokens) for tokens in self.tokenized_corpus)
            stats.update({
                "total_tokens": total_tokens,
                "avg_tokens_per_doc": total_tokens / len(self.tokenized_corpus) if self.tokenized_corpus else 0
            })
        
        return stats
    
    def _preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess text preserving technical terms and acronyms.
        
        Args:
            text: Raw text to tokenize
            
        Returns:
            List of preprocessed tokens
        """
        if not text or not text.strip():
            return []
        
        # Convert to lowercase while preserving structure
        if self.lowercase:
            text = text.lower()
        
        # Remove punctuation except hyphens, underscores, periods in technical terms
        text = self._punctuation_pattern.sub(' ', text)
        
        # Extract tokens using appropriate pattern
        tokens = self._tech_pattern.findall(text)
        
        # Filter out single characters and empty strings
        tokens = [token for token in tokens if len(token) > 1]
        
        return tokens
    
    def get_query_tokens(self, query: str) -> List[str]:
        """
        Get preprocessed tokens for a query (useful for debugging).
        
        Args:
            query: Query string
            
        Returns:
            List of preprocessed tokens
        """
        return self._preprocess_text(query)
    
    def get_document_tokens(self, doc_index: int) -> List[str]:
        """
        Get preprocessed tokens for a document (useful for debugging).
        
        Args:
            doc_index: Document index
            
        Returns:
            List of preprocessed tokens
        """
        if 0 <= doc_index < len(self.tokenized_corpus):
            return self.tokenized_corpus[doc_index]
        else:
            raise IndexError(f"Document index {doc_index} out of range")
    
    def get_bm25_scores(self, query: str) -> List[float]:
        """
        Get raw BM25 scores for all documents (useful for debugging).
        
        Args:
            query: Query string
            
        Returns:
            List of BM25 scores (not normalized)
        """
        if self.bm25 is None:
            raise ValueError("Must call index_documents() before getting scores")
        
        query_tokens = self._preprocess_text(query)
        if not query_tokens:
            return []
        
        scores = self.bm25.get_scores(query_tokens)
        return scores.tolist()
    
    def get_term_frequencies(self, query: str) -> Dict[str, int]:
        """
        Get term frequencies for a query (useful for analysis).
        
        Args:
            query: Query string
            
        Returns:
            Dictionary mapping terms to frequencies
        """
        query_tokens = self._preprocess_text(query)
        term_freqs = {}
        for token in query_tokens:
            term_freqs[token] = term_freqs.get(token, 0) + 1
        return term_freqs