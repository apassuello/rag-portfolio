"""
Graph-Enhanced Fusion Strategy for Architecture-Compliant Graph Integration.

This module provides a fusion strategy that properly integrates graph-based 
retrieval signals with dense and sparse retrieval results, following the 
proper sub-component architecture patterns.

This replaces the misplaced graph/ component with proper fusion sub-component 
enhancement.
"""

import logging
import time
from typing import List, Dict, Any, Tuple, Optional, Union
from collections import defaultdict
import numpy as np

from .base import FusionStrategy
from .rrf_fusion import RRFFusion
from src.core.interfaces import Document, RetrievalResult

# Import spaCy for entity extraction
try:
    import spacy
    SPACY_AVAILABLE = True
    # Try to load the English model
    try:
        _nlp = spacy.load("en_core_web_sm")
        NLP_MODEL_AVAILABLE = True
    except IOError:
        NLP_MODEL_AVAILABLE = False
        _nlp = None
except ImportError:
    SPACY_AVAILABLE = False
    NLP_MODEL_AVAILABLE = False
    _nlp = None

logger = logging.getLogger(__name__)


class GraphEnhancedFusionError(Exception):
    """Raised when graph-enhanced fusion operations fail."""
    pass


class GraphEnhancedRRFFusion(FusionStrategy):
    """
    Graph-enhanced RRF fusion strategy with sophisticated capabilities.
    
    This fusion strategy extends standard RRF to incorporate graph-based 
    retrieval signals as a third input stream, providing enhanced relevance
    through document relationship analysis.
    
    The implementation follows proper architecture patterns by enhancing
    the existing fusion sub-component rather than creating a separate
    graph component.
    
    Features:
    - ✅ Standard RRF fusion (dense + sparse)
    - ✅ Graph signal integration (third stream)
    - ✅ Configurable fusion weights
    - ✅ Entity-based document scoring
    - ✅ Relationship-aware relevance boosting
    - ✅ Performance optimization with caching
    - ✅ Graceful degradation without graph signals
    
    Architecture Compliance:
    - Properly located in fusion/ sub-component ✅
    - Extends existing FusionStrategy interface ✅  
    - Direct implementation (no external APIs) ✅
    - Backward compatible with existing fusion ✅
    
    Example:
        config = {
            "base_fusion": {
                "k": 60,
                "weights": {"dense": 0.6, "sparse": 0.3}
            },
            "graph_enhancement": {
                "enabled": True,
                "graph_weight": 0.1,
                "entity_boost": 0.15,
                "relationship_boost": 0.1
            }
        }
        fusion = GraphEnhancedRRFFusion(config)
        results = fusion.fuse_results(dense_results, sparse_results)
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize graph-enhanced RRF fusion strategy.
        
        Args:
            config: Configuration dictionary for graph-enhanced fusion
        """
        self.config = config
        
        # Initialize base RRF fusion
        base_config = config.get("base_fusion", {
            "k": 60,
            "weights": {"dense": 0.7, "sparse": 0.3}
        })
        self.base_fusion = RRFFusion(base_config)
        
        # Graph enhancement configuration
        self.graph_config = config.get("graph_enhancement", {
            "enabled": True,
            "graph_weight": 0.1,
            "entity_boost": 0.15,
            "relationship_boost": 0.1,
            "similarity_threshold": 0.7,
            "max_graph_hops": 3
        })
        
        # Performance tracking
        self.stats = {
            "total_fusions": 0,
            "graph_enhanced_fusions": 0,
            "entity_boosts_applied": 0,
            "relationship_boosts_applied": 0,
            "avg_graph_latency_ms": 0.0,
            "total_graph_latency_ms": 0.0,
            "fallback_activations": 0
        }
        
        # Graph analysis components (lightweight, self-contained)
        self.entity_cache = {}
        self.relationship_cache = {}
        
        # Document store for entity/relationship analysis
        self.documents = []
        self.query_cache = {}
        
        # Entity extraction setup
        self.nlp = _nlp if NLP_MODEL_AVAILABLE else None
        if not NLP_MODEL_AVAILABLE and self.graph_config.get("enabled", True):
            logger.warning("spaCy model not available, falling back to keyword matching for entity extraction")
        
        logger.info(f"GraphEnhancedRRFFusion initialized with graph_enabled={self.graph_config['enabled']}")
    
    def set_documents_and_query(self, documents: List[Document], query: str) -> None:
        """
        Set the documents and current query for entity/relationship analysis.
        
        Args:
            documents: List of documents being processed
            query: Current query string
        """
        self.documents = documents
        self.current_query = query
        
        # Clear query-specific caches
        self.query_cache = {}
    
    def fuse_results(
        self, 
        dense_results: List[Tuple[int, float]], 
        sparse_results: List[Tuple[int, float]]
    ) -> List[Tuple[int, float]]:
        """
        Fuse dense and sparse results with graph enhancement.
        
        This method maintains backward compatibility with the standard
        FusionStrategy interface while adding graph signal support
        when available.
        
        Args:
            dense_results: List of (document_index, score) from dense retrieval
            sparse_results: List of (document_index, score) from sparse retrieval
            
        Returns:
            List of (document_index, fused_score) tuples sorted by score
        """
        start_time = time.time()
        self.stats["total_fusions"] += 1
        
        try:
            # Step 1: Apply base RRF fusion (dense + sparse)
            base_fused = self.base_fusion.fuse_results(dense_results, sparse_results)
            
            # Step 2: Apply graph enhancement if enabled
            if self.graph_config.get("enabled", True):
                enhanced_results = self._apply_graph_enhancement(
                    base_fused, dense_results, sparse_results
                )
                self.stats["graph_enhanced_fusions"] += 1
            else:
                enhanced_results = base_fused
            
            # Step 3: Update performance statistics
            graph_latency_ms = (time.time() - start_time) * 1000
            self._update_stats(graph_latency_ms)
            
            return enhanced_results
            
        except Exception as e:
            logger.error(f"Graph-enhanced fusion failed: {e}")
            self.stats["fallback_activations"] += 1
            # Fallback to base fusion
            return self.base_fusion.fuse_results(dense_results, sparse_results)
    
    def _apply_graph_enhancement(
        self,
        base_results: List[Tuple[int, float]],
        dense_results: List[Tuple[int, float]], 
        sparse_results: List[Tuple[int, float]]
    ) -> List[Tuple[int, float]]:
        """
        Apply graph-based enhancement to fusion results.
        
        Args:
            base_results: Base RRF fusion results
            dense_results: Original dense retrieval results  
            sparse_results: Original sparse retrieval results
            
        Returns:
            Graph-enhanced fusion results
        """
        try:
            # Create a score map for efficient updates
            enhanced_scores = {}
            for doc_idx, score in base_results:
                enhanced_scores[doc_idx] = score
            
            # Extract document indices from all result sets
            all_doc_indices = set()
            for doc_idx, _ in base_results:
                all_doc_indices.add(doc_idx)
            for doc_idx, _ in dense_results:
                all_doc_indices.add(doc_idx)
            for doc_idx, _ in sparse_results:
                all_doc_indices.add(doc_idx)
            
            # Apply entity-based scoring enhancement
            entity_boosts = self._calculate_entity_boosts(list(all_doc_indices))
            
            # Apply relationship-based scoring enhancement  
            relationship_boosts = self._calculate_relationship_boosts(list(all_doc_indices))
            
            # Combine enhancements with base scores
            graph_weight = self.graph_config.get("graph_weight", 0.1)
            
            for doc_idx in enhanced_scores:
                entity_boost = entity_boosts.get(doc_idx, 0.0)
                relationship_boost = relationship_boosts.get(doc_idx, 0.0)
                
                # Apply graph enhancement with configurable weight
                graph_enhancement = (entity_boost + relationship_boost) * graph_weight
                enhanced_scores[doc_idx] += graph_enhancement
                
                # Track statistics
                if entity_boost > 0:
                    self.stats["entity_boosts_applied"] += 1
                if relationship_boost > 0:
                    self.stats["relationship_boosts_applied"] += 1
            
            # Sort by enhanced scores and return
            enhanced_results = sorted(enhanced_scores.items(), key=lambda x: x[1], reverse=True)
            
            return enhanced_results
            
        except Exception as e:
            logger.error(f"Graph enhancement failed: {e}")
            return base_results
    
    def _calculate_entity_boosts(self, doc_indices: List[int]) -> Dict[int, float]:
        """
        Calculate entity-based scoring boosts for documents using real entity extraction.
        
        Uses spaCy NLP to extract entities from query and documents, then calculates
        overlap-based boost scores. Falls back to keyword matching if spaCy unavailable.
        
        Args:
            doc_indices: List of document indices to analyze
            
        Returns:
            Dictionary mapping doc_index to entity boost score
        """
        entity_boosts = {}
        
        try:
            entity_boost_value = self.graph_config.get("entity_boost", 0.15)
            
            # Extract query entities once per query
            query_cache_key = f"query_entities:{getattr(self, 'current_query', '')}"
            if query_cache_key in self.query_cache:
                query_entities = self.query_cache[query_cache_key]
            else:
                query_entities = self._extract_entities(getattr(self, 'current_query', ''))
                self.query_cache[query_cache_key] = query_entities
            
            # Skip if no query entities found
            if not query_entities:
                return {doc_idx: 0.0 for doc_idx in doc_indices}
            
            for doc_idx in doc_indices:
                # Check cache first
                cache_key = f"entity:{doc_idx}:{hash(frozenset(query_entities))}"
                if cache_key in self.entity_cache:
                    entity_boosts[doc_idx] = self.entity_cache[cache_key]
                    continue
                
                # Get document content
                if doc_idx < len(self.documents):
                    doc_content = self.documents[doc_idx].content
                    
                    # Extract document entities
                    doc_entities = self._extract_entities(doc_content)
                    
                    # Calculate entity overlap score
                    if query_entities and doc_entities:
                        overlap = len(query_entities & doc_entities)
                        overlap_ratio = overlap / len(query_entities)
                        boost = overlap_ratio * entity_boost_value
                    else:
                        boost = 0.0
                else:
                    boost = 0.0
                
                # Cache the result
                self.entity_cache[cache_key] = boost
                entity_boosts[doc_idx] = boost
            
            return entity_boosts
            
        except Exception as e:
            logger.warning(f"Entity boost calculation failed: {e}")
            return {doc_idx: 0.0 for doc_idx in doc_indices}
    
    def _extract_entities(self, text: str) -> set:
        """
        Extract entities from text using spaCy or fallback to keyword matching.
        
        Args:
            text: Text to extract entities from
            
        Returns:
            Set of entity strings (normalized to lowercase)
        """
        if not text:
            return set()
        
        entities = set()
        
        try:
            if self.nlp and NLP_MODEL_AVAILABLE:
                # Use spaCy for real entity extraction
                doc = self.nlp(text)
                for ent in doc.ents:
                    # Focus on relevant entity types for technical documents
                    if ent.label_ in ['ORG', 'PRODUCT', 'PERSON', 'GPE', 'MONEY', 'CARDINAL']:
                        entities.add(ent.text.lower().strip())
                
                # Also extract technical terms (capitalized words, acronyms, etc.)
                for token in doc:
                    # Technical terms: all caps (>=2 chars), camelCase, or specific patterns
                    if (token.text.isupper() and len(token.text) >= 2) or \
                       (token.text[0].isupper() and any(c.isupper() for c in token.text[1:])) or \
                       any(tech_pattern in token.text.lower() for tech_pattern in 
                           ['risc', 'cisc', 'cpu', 'gpu', 'arm', 'x86', 'isa', 'api']):
                        entities.add(token.text.lower().strip())
            else:
                # Fallback: extract technical keywords and patterns
                import re
                
                # Technical acronyms and terms
                tech_patterns = [
                    r'\b[A-Z]{2,}\b',  # All caps 2+ chars (RISC, CISC, ARM, x86)
                    r'\b[A-Z][a-z]*[A-Z][A-Za-z]*\b',  # CamelCase
                    r'\bRV\d+[A-Z]*\b',  # RISC-V variants (RV32I, RV64I)
                    r'\b[Aa]rm[vV]\d+\b',  # ARM versions
                    r'\b[Xx]86\b',  # x86 variants
                ]
                
                for pattern in tech_patterns:
                    matches = re.findall(pattern, text)
                    entities.update(match.lower().strip() for match in matches)
                
                # Common technical terms
                tech_terms = ['risc', 'cisc', 'arm', 'intel', 'amd', 'qualcomm', 'apple', 
                             'samsung', 'berkeley', 'processor', 'cpu', 'gpu', 'architecture',
                             'instruction', 'set', 'pipelining', 'cache', 'memory']
                
                words = text.lower().split()
                entities.update(term for term in tech_terms if term in words)
            
        except Exception as e:
            logger.warning(f"Entity extraction failed: {e}")
        
        return entities
    
    def _calculate_relationship_boosts(self, doc_indices: List[int]) -> Dict[int, float]:
        """
        Calculate relationship-based scoring boosts using semantic similarity analysis.
        
        Uses document embeddings to calculate centrality scores in the semantic
        similarity graph, boosting documents that are central to the result set.
        
        Args:
            doc_indices: List of document indices to analyze
            
        Returns:
            Dictionary mapping doc_index to relationship boost score
        """
        relationship_boosts = {}
        
        try:
            relationship_boost_value = self.graph_config.get("relationship_boost", 0.1)
            similarity_threshold = self.graph_config.get("similarity_threshold", 0.7)
            
            # Need at least 2 documents for relationship analysis
            if len(doc_indices) < 2:
                return {doc_idx: 0.0 for doc_idx in doc_indices}
            
            # Get document embeddings for similarity calculation
            doc_embeddings = []
            valid_indices = []
            
            for doc_idx in doc_indices:
                if doc_idx < len(self.documents) and hasattr(self.documents[doc_idx], 'embedding'):
                    doc_embeddings.append(self.documents[doc_idx].embedding)
                    valid_indices.append(doc_idx)
            
            # Skip if we don't have enough embeddings
            if len(doc_embeddings) < 2:
                return {doc_idx: 0.0 for doc_idx in doc_indices}
            
            # Calculate similarity matrix
            embeddings_array = np.array(doc_embeddings)
            if embeddings_array.ndim == 1:
                embeddings_array = embeddings_array.reshape(1, -1)
            
            # Normalize embeddings for cosine similarity
            norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            normalized_embeddings = embeddings_array / norms
            
            # Calculate cosine similarity matrix
            similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)
            
            # Calculate centrality scores (sum of similarities above threshold)
            centrality_scores = []
            for i in range(len(similarity_matrix)):
                # Count strong connections (similarity above threshold)
                strong_connections = np.sum(similarity_matrix[i] > similarity_threshold)
                # Weight by average similarity to other documents
                avg_similarity = np.mean(similarity_matrix[i])
                centrality_score = (strong_connections * 0.6) + (avg_similarity * 0.4)
                centrality_scores.append(centrality_score)
            
            # Normalize centrality scores
            if centrality_scores:
                max_centrality = max(centrality_scores)
                if max_centrality > 0:
                    centrality_scores = [score / max_centrality for score in centrality_scores]
            
            # Apply relationship boosts
            for i, doc_idx in enumerate(valid_indices):
                # Check cache first
                cache_key = f"relationship:{doc_idx}:{len(valid_indices)}"
                if cache_key in self.relationship_cache:
                    relationship_boosts[doc_idx] = self.relationship_cache[cache_key]
                    continue
                
                centrality_score = centrality_scores[i] if i < len(centrality_scores) else 0.0
                boost = centrality_score * relationship_boost_value
                
                # Cache the result
                self.relationship_cache[cache_key] = boost
                relationship_boosts[doc_idx] = boost
            
            # Fill in zero boosts for documents without embeddings
            for doc_idx in doc_indices:
                if doc_idx not in relationship_boosts:
                    relationship_boosts[doc_idx] = 0.0
            
            return relationship_boosts
            
        except Exception as e:
            logger.warning(f"Relationship boost calculation failed: {e}")
            return {doc_idx: 0.0 for doc_idx in doc_indices}
    
    def _update_stats(self, graph_latency_ms: float) -> None:
        """Update performance statistics."""
        self.stats["total_graph_latency_ms"] += graph_latency_ms
        
        if self.stats["graph_enhanced_fusions"] > 0:
            self.stats["avg_graph_latency_ms"] = (
                self.stats["total_graph_latency_ms"] / self.stats["graph_enhanced_fusions"]
            )
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """
        Get information about the graph-enhanced fusion strategy.
        
        Returns:
            Dictionary with strategy configuration and statistics
        """
        base_info = self.base_fusion.get_strategy_info()
        
        enhanced_info = {
            "type": "graph_enhanced_rrf",
            "base_strategy": base_info,
            "graph_enabled": self.graph_config.get("enabled", True),
            "graph_weight": self.graph_config.get("graph_weight", 0.1),
            "entity_boost": self.graph_config.get("entity_boost", 0.15),
            "relationship_boost": self.graph_config.get("relationship_boost", 0.1),
            "statistics": self.stats.copy()
        }
        
        # Add performance metrics
        if self.stats["total_fusions"] > 0:
            enhanced_info["graph_enhancement_rate"] = (
                self.stats["graph_enhanced_fusions"] / self.stats["total_fusions"]
            )
        
        if self.stats["graph_enhanced_fusions"] > 0:
            enhanced_info["avg_graph_latency_ms"] = self.stats["avg_graph_latency_ms"]
        
        return enhanced_info
    
    def enable_graph_enhancement(self) -> None:
        """Enable graph enhancement features."""
        self.graph_config["enabled"] = True
        logger.info("Graph enhancement enabled")
    
    def disable_graph_enhancement(self) -> None:
        """Disable graph enhancement features."""
        self.graph_config["enabled"] = False
        logger.info("Graph enhancement disabled")
    
    def clear_caches(self) -> None:
        """Clear entity and relationship caches."""
        self.entity_cache.clear()
        self.relationship_cache.clear()
        logger.info("Graph enhancement caches cleared")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get detailed performance statistics.
        
        Returns:
            Dictionary with performance metrics
        """
        return {
            **self.stats,
            "cache_sizes": {
                "entity_cache": len(self.entity_cache),
                "relationship_cache": len(self.relationship_cache)
            },
            "base_fusion_stats": self.base_fusion.get_strategy_info()
        }