from typing import List, Dict
import re
import hashlib


def chunk_technical_text(
    text: str, chunk_size: int = 512, overlap: int = 50
) -> List[Dict]:
    """
    Chunk technical documentation preserving sentence boundaries.

    Args:
        text: Input text to chunk
        chunk_size: Target chunk size in characters
        overlap: Overlap between chunks in characters

    Returns:
        List of chunks, each containing:
        {
            "text": str,           # Chunk text content
            "start_char": int,     # Start position in original
            "end_char": int,       # End position in original
            "chunk_id": str,       # Unique identifier
            "word_count": int,     # Words in chunk
            "sentence_complete": bool  # Ends with complete sentence
        }

    Example:
        >>> text = "First sentence. Second sentence."
        >>> chunks = chunk_technical_text(text, chunk_size=20)
        >>> len(chunks) >= 1
        True
    """
    if not text.strip():
        return []
    
    # Clean and normalize text
    text = text.strip()
    chunks = []
    start_pos = 0
    
    while start_pos < len(text):
        # Calculate target end position
        target_end = min(start_pos + chunk_size, len(text))
        
        # Find sentence boundaries using regex
        sentence_pattern = r'[.!?:;](?:\s|$)'
        
        # Look for sentence boundary near target position
        search_start = max(start_pos, target_end - 100)
        search_text = text[search_start:target_end + 50]
        
        sentence_matches = list(re.finditer(sentence_pattern, search_text))
        
        if sentence_matches and target_end < len(text):
            # Find best sentence boundary
            best_match = sentence_matches[-1]
            chunk_end = search_start + best_match.end()
            sentence_complete = True
        else:
            # No good boundary found or at end of text
            chunk_end = target_end
            sentence_complete = text[chunk_end-1:chunk_end] in '.!?:;'
        
        # Extract chunk text
        chunk_text = text[start_pos:chunk_end].strip()
        
        if chunk_text:
            # Generate unique chunk ID
            chunk_hash = hashlib.md5(chunk_text.encode()).hexdigest()[:8]
            chunk_id = f"chunk_{chunk_hash}"
            
            # Count words
            word_count = len(chunk_text.split())
            
            chunks.append({
                "text": chunk_text,
                "start_char": start_pos,
                "end_char": chunk_end,
                "chunk_id": chunk_id,
                "word_count": word_count,
                "sentence_complete": sentence_complete
            })
        
        # Calculate next start position with overlap
        if chunk_end >= len(text):
            break
            
        overlap_start = max(chunk_end - overlap, start_pos + 1)
        start_pos = overlap_start
    
    return chunks
