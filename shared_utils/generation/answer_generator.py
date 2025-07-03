"""
Answer generation module using Ollama for local LLM inference.

This module provides answer generation with citation support for RAG systems,
optimized for technical documentation Q&A on Apple Silicon.
"""

import json
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Generator, Tuple
import ollama
from datetime import datetime
import re

logger = logging.getLogger(__name__)


@dataclass
class Citation:
    """Represents a citation to a source document chunk."""
    chunk_id: str
    page_number: int
    source_file: str
    relevance_score: float
    text_snippet: str


@dataclass
class GeneratedAnswer:
    """Represents a generated answer with citations."""
    answer: str
    citations: List[Citation]
    confidence_score: float
    generation_time: float
    model_used: str
    context_used: List[Dict[str, Any]]


class AnswerGenerator:
    """
    Generates answers using local LLMs via Ollama with citation support.
    
    Optimized for technical documentation Q&A with:
    - Streaming response support
    - Citation extraction and formatting
    - Confidence scoring
    - Fallback model support
    """
    
    def __init__(
        self,
        primary_model: str = "llama3.2:3b",
        fallback_model: str = "mistral:latest",
        temperature: float = 0.3,
        max_tokens: int = 1024,
        stream: bool = True
    ):
        """
        Initialize the answer generator.
        
        Args:
            primary_model: Primary Ollama model to use
            fallback_model: Fallback model for complex queries
            temperature: Generation temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            stream: Whether to stream responses
        """
        self.primary_model = primary_model
        self.fallback_model = fallback_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.stream = stream
        self.client = ollama.Client()
        
        # Verify models are available
        self._verify_models()
        
    def _verify_models(self) -> None:
        """Verify that required models are available."""
        try:
            model_list = self.client.list()
            available_models = []
            
            # Handle Ollama's ListResponse object
            if hasattr(model_list, 'models'):
                for model in model_list.models:
                    if hasattr(model, 'model'):
                        available_models.append(model.model)
                    elif isinstance(model, dict) and 'model' in model:
                        available_models.append(model['model'])
            
            if self.primary_model not in available_models:
                logger.warning(f"Primary model {self.primary_model} not found. Available models: {available_models}")
                raise ValueError(f"Model {self.primary_model} not available. Please run: ollama pull {self.primary_model}")
                
            if self.fallback_model not in available_models:
                logger.warning(f"Fallback model {self.fallback_model} not found in: {available_models}")
                
        except Exception as e:
            logger.error(f"Error verifying models: {e}")
            raise
    
    def _create_system_prompt(self) -> str:
        """Create system prompt for technical documentation Q&A."""
        return """You are a technical documentation assistant that provides accurate answers based on the provided context.

CRITICAL RULES:
1. NEVER add technical details not explicitly stated in the context
   - Do NOT specify numbers, measurements, or specifications unless in context
   - Do NOT use pre-trained knowledge to "complete" partial information
   - Do NOT make educated guesses or inferences
   
2. CONTEXT ADHERENCE:
   - Every technical claim MUST be directly traceable to the context
   - If context mentions "RISC-V has instruction formats" WITHOUT listing them, 
     say "The context mentions RISC-V has instruction formats but doesn't specify which ones"
   - For partial information, state ONLY what's given and explicitly note what's missing

3. RESPONSE APPROACH:
   - Complete context → Provide detailed, cited answer
   - Partial context → Answer what's available, explicitly note gaps
   - Irrelevant context → Brief refusal: "The context doesn't contain relevant information"
   - Suspicious context → Flag concerns and refuse to use

4. CITATION REQUIREMENTS:
   - EVERY fact must have [chunk_X] citation
   - No citation = information not used

VERIFICATION STEP:
Before finalizing your answer, ALWAYS verify:
1. Is every technical detail explicitly stated in the context?
2. Am I adding any numbers, sizes, or specifications from my training?
3. If context is incomplete, am I stating what's missing rather than guessing?

TECHNICAL DOCUMENTATION RULES:
- Technical specifications must be quoted verbatim from context
- Never interpolate between data points
- Never provide "typical" values for unspecified parameters
- When in doubt, request more specific documentation"""

    def _format_context(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Format retrieved chunks into context for the LLM.
        
        Args:
            chunks: List of retrieved chunks with metadata
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for i, chunk in enumerate(chunks):
            chunk_text = chunk.get('content', chunk.get('text', ''))
            page_num = chunk.get('metadata', {}).get('page_number', 'unknown')
            source = chunk.get('metadata', {}).get('source', 'unknown')
            
            context_parts.append(
                f"[chunk_{i+1}] (Page {page_num} from {source}):\n{chunk_text}\n"
            )
        
        return "\n---\n".join(context_parts)
    
    def _extract_citations(self, answer: str, chunks: List[Dict[str, Any]]) -> Tuple[str, List[Citation]]:
        """
        Extract citations from the generated answer.
        
        Args:
            answer: Generated answer with [chunk_X] citations
            chunks: Original chunks used for context
            
        Returns:
            Tuple of (clean_answer, citations)
        """
        citations = []
        # Only look for proper [chunk_X] format citations
        citation_pattern = r'\[chunk_(\d+)\]'
        
        cited_chunks = set()
        
        # Find [chunk_X] citations
        matches = re.finditer(citation_pattern, answer)
        for match in matches:
            chunk_idx = int(match.group(1)) - 1  # Convert to 0-based index
            if 0 <= chunk_idx < len(chunks):
                cited_chunks.add(chunk_idx)
        
        # Create Citation objects for each cited chunk
        for idx in cited_chunks:
            chunk = chunks[idx]
            citation = Citation(
                chunk_id=chunk.get('id', f'chunk_{idx}'),
                page_number=chunk.get('metadata', {}).get('page_number', 0),
                source_file=chunk.get('metadata', {}).get('source', 'unknown'),
                relevance_score=chunk.get('score', 0.0),
                text_snippet=chunk.get('content', chunk.get('text', ''))[:200] + '...'
            )
            citations.append(citation)
        
        # Clean only [chunk_X] citations from answer for display
        clean_answer = re.sub(citation_pattern, '', answer).strip()
        
        return clean_answer, citations
    
    def _calculate_confidence(self, answer: str, citations: List[Citation], chunks: List[Dict[str, Any]]) -> float:
        """
        Calculate confidence score for the generated answer with improved calibration.
        
        Args:
            answer: Generated answer
            citations: Extracted citations
            chunks: Retrieved chunks
            
        Returns:
            Confidence score (0.0-1.0)
        """
        # Start with minimal base confidence
        confidence = 0.05  # Much lower starting point
        
        # Expanded uncertainty and refusal indicators
        uncertainty_phrases = [
            "does not contain sufficient information",
            "context does not provide",
            "unclear",
            "conflicting", 
            "insufficient information",
            "cannot determine",
            "not specified",
            "questionable",
            "refuse to answer",  # New
            "not contained",     # New
            "no mention",        # New
            "no relevant",       # New
            "cannot answer",     # New
            "does not contain relevant",  # New
            "missing",           # New
            "not explicitly"     # New
        ]
        
        # Check for uncertainty/refusal indicators - these should drastically reduce confidence
        if any(phrase in answer.lower() for phrase in uncertainty_phrases):
            # For refusal/uncertainty answers, confidence should be very low regardless of citations
            return min(0.1, confidence * 2)  # Max 10% for uncertain/refusal answers
        
        # Check if no chunks were provided
        if not chunks:
            # No context = very low confidence, regardless of answer content
            return 0.05
        
        # Factor 1: Query-Context Relevance Assessment (NEW - CRITICAL)
        if chunks:
            scores = [chunk.get('score', 0) for chunk in chunks]
            max_relevance = max(scores) if scores else 0
            avg_relevance = sum(scores) / len(scores) if scores else 0
            
            # If all chunks have very low relevance scores, this is likely an irrelevant query
            if max_relevance < 0.4:
                # Very low relevance = very low confidence regardless of other factors
                return min(0.08, confidence)  # Max 8% for low relevance context
            elif max_relevance < 0.6:
                # Moderate relevance = cap confidence at lower level
                confidence = min(confidence, 0.15)
        
        # Factor 2: Citation coverage (only boost if relevance is good)
        if citations and chunks:
            citation_ratio = len(citations) / min(len(chunks), 3)
            # Only boost confidence if chunks are actually relevant (score > 0.6)
            relevant_chunks = [c for c in chunks if c.get('score', 0) > 0.6]
            if relevant_chunks:
                confidence += 0.3 * citation_ratio
            else:
                # Citations to irrelevant chunks don't boost confidence
                confidence += 0.1 * citation_ratio
        else:
            # No citations = suspicious, reduce confidence
            confidence *= 0.6
        
        # Factor 3: Average relevance score of cited chunks
        if citations:
            avg_relevance = sum(c.relevance_score for c in citations) / len(citations)
            # Only boost confidence if relevance is genuinely high
            if avg_relevance > 0.7:
                confidence += 0.2 * avg_relevance
            elif avg_relevance < 0.4:  # Lowered threshold
                # Low relevance chunks = significantly reduce confidence
                confidence *= 0.4  # Stronger penalty
        
        # Factor 4: Context quality assessment
        if chunks:
            # Check for very short or potentially low-quality chunks
            avg_chunk_length = sum(len(chunk.get('content', chunk.get('text', ''))) for chunk in chunks) / len(chunks)
            if avg_chunk_length < 100:  # Very short chunks
                confidence *= 0.8
            
            # Check if all chunks have very low similarity scores (redundant but kept for compatibility)
            scores = [chunk.get('score', 0) for chunk in chunks]
            if scores and max(scores) < 0.5:
                confidence *= 0.5  # Stronger penalty
        
        # Factor 4: Answer characteristics and content quality
        answer_words = len(answer.split())
        if answer_words < 10:
            # Very short answers suggest uncertainty
            confidence *= 0.8
        elif answer_words > 15:
            # Reasonable length suggests proper processing
            confidence += 0.1
        
        # Factor 5: Boost confidence for good context usage
        if citations and chunks and not any(phrase in answer.lower() for phrase in uncertainty_phrases):
            # If we have good citations and no uncertainty markers, boost confidence
            if len(citations) >= len(chunks) * 0.5:  # Cited at least half the chunks
                confidence += 0.2
            
            # Extra boost for high-quality context with specific information
            avg_chunk_length = sum(len(chunk.get('content', chunk.get('text', ''))) for chunk in chunks) / len(chunks)
            avg_relevance = sum(c.relevance_score for c in citations) / len(citations) if citations else 0
            if avg_chunk_length > 200 and avg_relevance > 0.8:
                confidence += 0.1
            
        return min(confidence, 1.0)
    
    def generate(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        use_fallback: bool = False
    ) -> GeneratedAnswer:
        """
        Generate an answer based on the query and retrieved chunks.
        
        Args:
            query: User's question
            chunks: Retrieved document chunks
            use_fallback: Whether to use fallback model
            
        Returns:
            GeneratedAnswer object with answer, citations, and metadata
        """
        start_time = datetime.now()
        model = self.fallback_model if use_fallback else self.primary_model
        
        # Check for no-context or very poor context situation
        if not chunks or all(len(chunk.get('content', chunk.get('text', ''))) < 20 for chunk in chunks):
            # Handle no-context situation explicitly
            user_prompt = f"""Context: [NO RELEVANT CONTEXT FOUND]

Question: {query}

Since no relevant context was provided, you must respond: "I cannot answer this question because no relevant context was found in the available documents. To get an accurate answer, please ensure the relevant documents are properly indexed and contain information about [topic from query]." """
        else:
            # Format context from chunks
            context = self._format_context(chunks)
            
            # Create the full prompt with explicit citation instructions
            user_prompt = f"""Context:
{context}

Question: {query}

INSTRUCTIONS:
1. Read the context carefully and determine if it contains relevant information to answer the question
2. If the context contains relevant information, answer the question using ONLY that information
3. You MUST cite every piece of information using [chunk_1], [chunk_2], etc. format
4. Example citation: "According to [chunk_1], RISC-V is an open-source architecture."
5. If context is insufficient, state clearly what information is missing

Answer the question now with proper [chunk_X] citations for every factual claim:"""
        
        try:
            # Generate response
            response = self.client.chat(
                model=model,
                messages=[
                    {"role": "system", "content": self._create_system_prompt()},
                    {"role": "user", "content": user_prompt}
                ],
                options={
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens
                },
                stream=False  # Get complete response for processing
            )
            
            # Extract answer
            answer_with_citations = response['message']['content']
            
            # Extract and clean citations
            clean_answer, citations = self._extract_citations(answer_with_citations, chunks)
            
            # Calculate confidence
            confidence = self._calculate_confidence(clean_answer, citations, chunks)
            
            # Calculate generation time
            generation_time = (datetime.now() - start_time).total_seconds()
            
            return GeneratedAnswer(
                answer=clean_answer,
                citations=citations,
                confidence_score=confidence,
                generation_time=generation_time,
                model_used=model,
                context_used=chunks
            )
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            # Return a fallback response
            return GeneratedAnswer(
                answer="I apologize, but I encountered an error while generating the answer. Please try again.",
                citations=[],
                confidence_score=0.0,
                generation_time=0.0,
                model_used=model,
                context_used=chunks
            )
    
    def generate_stream(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        use_fallback: bool = False
    ) -> Generator[str, None, GeneratedAnswer]:
        """
        Generate an answer with streaming support.
        
        Args:
            query: User's question
            chunks: Retrieved document chunks
            use_fallback: Whether to use fallback model
            
        Yields:
            Partial answer strings
            
        Returns:
            Final GeneratedAnswer object
        """
        start_time = datetime.now()
        model = self.fallback_model if use_fallback else self.primary_model
        
        # Check for no-context or very poor context situation
        if not chunks or all(len(chunk.get('content', chunk.get('text', ''))) < 20 for chunk in chunks):
            # Handle no-context situation explicitly
            user_prompt = f"""Context: [NO RELEVANT CONTEXT FOUND]

Question: {query}

Since no relevant context was provided, you must respond: "I cannot answer this question because no relevant context was found in the available documents. To get an accurate answer, please ensure the relevant documents are properly indexed and contain information about [topic from query]." """
        else:
            # Format context from chunks
            context = self._format_context(chunks)
            
            # Create the full prompt with explicit citation instructions
            user_prompt = f"""Context:
{context}

Question: {query}

INSTRUCTIONS:
1. Read the context carefully and determine if it contains relevant information to answer the question
2. If the context contains relevant information, answer the question using ONLY that information
3. You MUST cite every piece of information using [chunk_1], [chunk_2], etc. format
4. Example citation: "According to [chunk_1], RISC-V is an open-source architecture."
5. If context is insufficient, state clearly what information is missing

Answer the question now with proper [chunk_X] citations for every factual claim:"""
        
        try:
            # Generate streaming response
            stream = self.client.chat(
                model=model,
                messages=[
                    {"role": "system", "content": self._create_system_prompt()},
                    {"role": "user", "content": user_prompt}
                ],
                options={
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens
                },
                stream=True
            )
            
            # Collect full answer while streaming
            full_answer = ""
            for chunk in stream:
                if 'message' in chunk and 'content' in chunk['message']:
                    partial = chunk['message']['content']
                    full_answer += partial
                    yield partial
            
            # Process complete answer
            clean_answer, citations = self._extract_citations(full_answer, chunks)
            confidence = self._calculate_confidence(clean_answer, citations, chunks)
            generation_time = (datetime.now() - start_time).total_seconds()
            
            return GeneratedAnswer(
                answer=clean_answer,
                citations=citations,
                confidence_score=confidence,
                generation_time=generation_time,
                model_used=model,
                context_used=chunks
            )
            
        except Exception as e:
            logger.error(f"Error in streaming generation: {e}")
            yield "I apologize, but I encountered an error while generating the answer."
            
            return GeneratedAnswer(
                answer="Error occurred during generation.",
                citations=[],
                confidence_score=0.0,
                generation_time=0.0,
                model_used=model,
                context_used=chunks
            )
    
    def format_answer_with_citations(self, generated_answer: GeneratedAnswer) -> str:
        """
        Format the generated answer with citations for display.
        
        Args:
            generated_answer: GeneratedAnswer object
            
        Returns:
            Formatted string with answer and citations
        """
        formatted = f"{generated_answer.answer}\n\n"
        
        if generated_answer.citations:
            formatted += "**Sources:**\n"
            for i, citation in enumerate(generated_answer.citations, 1):
                formatted += f"{i}. {citation.source_file} (Page {citation.page_number})\n"
        
        formatted += f"\n*Confidence: {generated_answer.confidence_score:.1%} | "
        formatted += f"Model: {generated_answer.model_used} | "
        formatted += f"Time: {generated_answer.generation_time:.2f}s*"
        
        return formatted


if __name__ == "__main__":
    # Example usage
    generator = AnswerGenerator()
    
    # Example chunks (would come from retrieval system)
    example_chunks = [
        {
            "id": "chunk_1",
            "content": "RISC-V is an open-source instruction set architecture (ISA) based on reduced instruction set computer (RISC) principles.",
            "metadata": {"page_number": 1, "source": "riscv-spec.pdf"},
            "score": 0.95
        },
        {
            "id": "chunk_2", 
            "content": "The RISC-V ISA is designed to support a wide range of implementations including 32-bit, 64-bit, and 128-bit variants.",
            "metadata": {"page_number": 2, "source": "riscv-spec.pdf"},
            "score": 0.89
        }
    ]
    
    # Generate answer
    result = generator.generate(
        query="What is RISC-V?",
        chunks=example_chunks
    )
    
    # Display formatted result
    print(generator.format_answer_with_citations(result))