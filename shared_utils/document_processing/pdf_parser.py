from typing import Dict, List, Any
from pathlib import Path
import time
import fitz

def extract_text_with_metadata(pdf_path: Path) -> Dict[str, Any]:
    """
    Extract text and metadata from technical PDF.
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        {
            "text": str,           # Complete text content
            "pages": List[Dict],   # Per-page text and metadata
            "metadata": Dict,      # Document-level metadata
            "page_count": int,     # Total pages
            "extraction_time": float  # Processing time in seconds
        }
        
    Raises:
        FileNotFoundError: If PDF doesn't exist
        ValueError: If PDF is corrupted or unreadable
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    start_time = time.perf_counter()
    
    try:
        doc = fitz.open(str(pdf_path))
        
        # Extract document metadata
        metadata = doc.metadata or {}
        page_count = len(doc)
        
        # Extract text from all pages
        pages = []
        all_text = []
        
        for page_num in range(page_count):
            page = doc[page_num]
            page_text = page.get_text()
            
            pages.append({
                "page_number": page_num + 1,
                "text": page_text,
                "char_count": len(page_text)
            })
            all_text.append(page_text)
        
        doc.close()
        extraction_time = time.perf_counter() - start_time
        
        return {
            "text": "\n".join(all_text),
            "pages": pages,
            "metadata": metadata,
            "page_count": page_count,
            "extraction_time": extraction_time
        }
        
    except Exception as e:
        raise ValueError(f"Failed to process PDF: {e}")