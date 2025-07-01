"""
Content analysis module for RAG quality evaluation.

Provides objective metrics to measure result quality:
- Term coverage analysis
- Semantic coherence measurement  
- Information diversity calculation
- Technical relevance assessment
"""

from typing import List, Dict, Any, Set, Tuple
import re
import numpy as np
from collections import Counter
from pathlib import Path
import sys

# Add paths for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from shared_utils.embeddings.generator import generate_embeddings


class ContentAnalyzer:
    """
    Analyzes content quality and relevance of RAG retrieval results.
    
    Provides multiple objective metrics to validate whether enhanced
    retrieval actually produces better quality results.
    """
    
    def __init__(self):
        """Initialize content analyzer."""
        # Technical term patterns
        self.technical_patterns = [
            r'\b[A-Z]{2,}\b',  # Acronyms (CPU, RAM, etc.)
            r'\b\w+[\-_]\w+\b',  # Hyphenated terms (real-time, etc.)
            r'\b\d+[a-zA-Z]+\b',  # Alphanumeric (RV32I, etc.)
        ]
        
        # Citation/reference patterns
        self.citation_patterns = [
            r'\[[0-9]+\]',  # [1], [2], etc.
            r'Figure \d+',   # Figure references
            r'Section \d+\.?\d*',  # Section references
            r'Chapter \d+',  # Chapter references
            r'Table \d+',    # Table references
        ]
        
        # Stop words for content analysis
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'shall', 'this', 'that', 'these',
            'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him',
            'her', 'us', 'them'
        }
    
    def extract_key_terms(self, text: str) -> Set[str]:
        """
        Extract meaningful terms from text.
        
        Args:
            text: Input text
            
        Returns:
            Set of key terms (lowercased, no stop words)
        """
        # Basic tokenization
        words = re.findall(r'\b[a-zA-Z][a-zA-Z0-9\-_]*\b', text.lower())
        
        # Filter stop words and short terms
        key_terms = {
            word for word in words 
            if word not in self.stop_words and len(word) > 2
        }
        
        return key_terms
    
    def extract_technical_terms(self, text: str) -> Set[str]:
        """
        Extract technical terms from text.
        
        Args:
            text: Input text
            
        Returns:
            Set of technical terms
        """
        technical_terms = set()
        
        for pattern in self.technical_patterns:
            matches = re.findall(pattern, text)
            technical_terms.update(match.lower() for match in matches)
        
        return technical_terms
    
    def calculate_term_coverage(self, query: str, results: List[Dict]) -> Dict[str, float]:
        """
        Calculate how well results cover query terms and concepts.
        
        Args:
            query: Original search query
            results: List of result chunks
            
        Returns:
            Dictionary with coverage metrics
        """
        query_terms = self.extract_key_terms(query)
        query_technical = self.extract_technical_terms(query)
        all_query_terms = query_terms.union(query_technical)
        
        if not all_query_terms:
            return {'avg_coverage': 0.0, 'max_coverage': 0.0, 'total_coverage': 0.0}
        
        coverage_scores = []
        all_covered_terms = set()
        
        for chunk in results:
            chunk_text = chunk.get('text', '')
            chunk_terms = self.extract_key_terms(chunk_text)
            chunk_technical = self.extract_technical_terms(chunk_text)
            all_chunk_terms = chunk_terms.union(chunk_technical)
            
            # Calculate coverage for this chunk
            covered_terms = all_query_terms.intersection(all_chunk_terms)
            coverage_score = len(covered_terms) / len(all_query_terms)
            coverage_scores.append(coverage_score)
            
            # Track all covered terms across results
            all_covered_terms.update(covered_terms)
        
        return {
            'avg_coverage': np.mean(coverage_scores) if coverage_scores else 0.0,
            'max_coverage': max(coverage_scores) if coverage_scores else 0.0,
            'total_coverage': len(all_covered_terms) / len(all_query_terms),
            'covered_terms': list(all_covered_terms),
            'missing_terms': list(all_query_terms - all_covered_terms)
        }
    
    def calculate_semantic_coherence(self, query: str, results: List[Dict]) -> Dict[str, float]:
        """
        Measure semantic coherence between query and results.
        
        Args:
            query: Original search query
            results: List of result chunks
            
        Returns:
            Dictionary with coherence metrics
        """
        if not results:
            return {'avg_coherence': 0.0, 'min_coherence': 0.0, 'coherence_variance': 0.0}
        
        try:
            # Generate embeddings
            query_embedding = generate_embeddings([query])[0]
            result_texts = [chunk.get('text', '') for chunk in results]
            result_embeddings = generate_embeddings(result_texts)
            
            # Calculate cosine similarities
            coherence_scores = []
            for result_emb in result_embeddings:
                # Cosine similarity
                dot_product = np.dot(query_embedding, result_emb)
                norm_product = np.linalg.norm(query_embedding) * np.linalg.norm(result_emb)
                
                if norm_product > 0:
                    similarity = dot_product / norm_product
                    coherence_scores.append(similarity)
                else:
                    coherence_scores.append(0.0)
            
            return {
                'avg_coherence': np.mean(coherence_scores),
                'min_coherence': min(coherence_scores),
                'max_coherence': max(coherence_scores),
                'coherence_variance': np.var(coherence_scores),
                'coherence_std': np.std(coherence_scores)
            }
            
        except Exception as e:
            print(f"Error calculating semantic coherence: {e}")
            return {'avg_coherence': 0.0, 'min_coherence': 0.0, 'coherence_variance': 0.0}
    
    def calculate_information_diversity(self, results: List[Dict]) -> Dict[str, float]:
        """
        Measure diversity of information in results.
        
        Args:
            results: List of result chunks
            
        Returns:
            Dictionary with diversity metrics
        """
        if len(results) < 2:
            return {'avg_diversity': 1.0, 'min_diversity': 1.0, 'pairwise_diversities': []}
        
        diversity_scores = []
        result_texts = [chunk.get('text', '') for chunk in results]
        
        # Calculate pairwise text similarities
        for i in range(len(result_texts)):
            for j in range(i + 1, len(result_texts)):
                similarity = self._calculate_text_similarity(result_texts[i], result_texts[j])
                diversity = 1.0 - similarity  # Diversity = 1 - similarity
                diversity_scores.append(diversity)
        
        return {
            'avg_diversity': np.mean(diversity_scores) if diversity_scores else 1.0,
            'min_diversity': min(diversity_scores) if diversity_scores else 1.0,
            'max_diversity': max(diversity_scores) if diversity_scores else 1.0,
            'pairwise_diversities': diversity_scores
        }
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts using term overlap."""
        terms1 = self.extract_key_terms(text1)
        terms2 = self.extract_key_terms(text2)
        
        if not terms1 and not terms2:
            return 1.0  # Both empty
        if not terms1 or not terms2:
            return 0.0  # One empty
        
        # Jaccard similarity
        intersection = len(terms1.intersection(terms2))
        union = len(terms1.union(terms2))
        
        return intersection / union if union > 0 else 0.0
    
    def analyze_technical_relevance(self, query: str, results: List[Dict]) -> Dict[str, Any]:
        """
        Analyze technical relevance of results for domain-specific queries.
        
        Args:
            query: Original search query
            results: List of result chunks
            
        Returns:
            Dictionary with technical relevance metrics
        """
        query_technical_terms = self.extract_technical_terms(query)
        
        technical_metrics = []
        
        for chunk in results:
            chunk_text = chunk.get('text', '')
            chunk_technical_terms = self.extract_technical_terms(chunk_text)
            chunk_citations = self._count_citations(chunk_text)
            
            # Technical term overlap
            technical_overlap = len(query_technical_terms.intersection(chunk_technical_terms))
            
            # Technical term density
            all_terms = len(self.extract_key_terms(chunk_text))
            technical_density = len(chunk_technical_terms) / all_terms if all_terms > 0 else 0.0
            
            # Citation density (indicates academic/technical quality)
            word_count = len(chunk_text.split())
            citation_density = chunk_citations / word_count if word_count > 0 else 0.0
            
            technical_metrics.append({
                'technical_overlap': technical_overlap,
                'technical_density': technical_density,
                'citation_density': citation_density,
                'technical_terms_found': list(chunk_technical_terms),
                'citation_count': chunk_citations
            })
        
        # Aggregate metrics
        avg_technical_overlap = np.mean([m['technical_overlap'] for m in technical_metrics])
        avg_technical_density = np.mean([m['technical_density'] for m in technical_metrics])
        avg_citation_density = np.mean([m['citation_density'] for m in technical_metrics])
        
        return {
            'avg_technical_overlap': avg_technical_overlap,
            'avg_technical_density': avg_technical_density,
            'avg_citation_density': avg_citation_density,
            'individual_metrics': technical_metrics,
            'query_technical_terms': list(query_technical_terms)
        }
    
    def _count_citations(self, text: str) -> int:
        """Count citation/reference patterns in text."""
        citation_count = 0
        
        for pattern in self.citation_patterns:
            matches = re.findall(pattern, text)
            citation_count += len(matches)
        
        return citation_count
    
    def comprehensive_analysis(self, query: str, results: List[Dict]) -> Dict[str, Any]:
        """
        Perform comprehensive content analysis.
        
        Args:
            query: Original search query
            results: List of result chunks
            
        Returns:
            Complete analysis with all metrics
        """
        analysis = {
            'query': query,
            'result_count': len(results),
            'term_coverage': self.calculate_term_coverage(query, results),
            'semantic_coherence': self.calculate_semantic_coherence(query, results),
            'information_diversity': self.calculate_information_diversity(results),
            'technical_relevance': self.analyze_technical_relevance(query, results)
        }
        
        # Calculate overall quality score
        coverage_score = analysis['term_coverage']['avg_coverage']
        coherence_score = analysis['semantic_coherence']['avg_coherence']
        diversity_score = analysis['information_diversity']['avg_diversity']
        technical_score = analysis['technical_relevance']['avg_technical_density']
        
        # Weighted average (adjust weights based on use case)
        overall_quality = (
            0.3 * coverage_score +
            0.3 * coherence_score +
            0.2 * diversity_score +
            0.2 * technical_score
        )
        
        analysis['overall_quality_score'] = overall_quality
        
        return analysis
    
    def compare_result_sets(self, query: str, 
                           basic_results: List[Dict], 
                           enhanced_results: List[Dict]) -> Dict[str, Any]:
        """
        Compare two sets of results for the same query.
        
        Args:
            query: Search query
            basic_results: Results from basic method
            enhanced_results: Results from enhanced method
            
        Returns:
            Comparative analysis
        """
        basic_analysis = self.comprehensive_analysis(query, basic_results)
        enhanced_analysis = self.comprehensive_analysis(query, enhanced_results)
        
        # Calculate improvements
        improvements = {}
        
        # Coverage improvement
        basic_coverage = basic_analysis['term_coverage']['avg_coverage']
        enhanced_coverage = enhanced_analysis['term_coverage']['avg_coverage']
        improvements['coverage_improvement'] = enhanced_coverage - basic_coverage
        improvements['coverage_improvement_pct'] = (
            (enhanced_coverage - basic_coverage) / basic_coverage * 100 
            if basic_coverage > 0 else 0
        )
        
        # Coherence improvement
        basic_coherence = basic_analysis['semantic_coherence']['avg_coherence']
        enhanced_coherence = enhanced_analysis['semantic_coherence']['avg_coherence']
        improvements['coherence_improvement'] = enhanced_coherence - basic_coherence
        improvements['coherence_improvement_pct'] = (
            (enhanced_coherence - basic_coherence) / basic_coherence * 100
            if basic_coherence > 0 else 0
        )
        
        # Overall quality improvement
        basic_quality = basic_analysis['overall_quality_score']
        enhanced_quality = enhanced_analysis['overall_quality_score']
        improvements['overall_improvement'] = enhanced_quality - basic_quality
        improvements['overall_improvement_pct'] = (
            (enhanced_quality - basic_quality) / basic_quality * 100
            if basic_quality > 0 else 0
        )
        
        return {
            'query': query,
            'basic_analysis': basic_analysis,
            'enhanced_analysis': enhanced_analysis,
            'improvements': improvements,
            'is_enhanced_better': enhanced_quality > basic_quality,
            'improvement_summary': {
                'coverage': f"{improvements['coverage_improvement_pct']:+.1f}%",
                'coherence': f"{improvements['coherence_improvement_pct']:+.1f}%",
                'overall': f"{improvements['overall_improvement_pct']:+.1f}%"
            }
        }