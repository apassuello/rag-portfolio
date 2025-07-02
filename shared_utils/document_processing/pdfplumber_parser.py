#!/usr/bin/env python3
"""
PDFPlumber-based Parser

Advanced PDF parsing using pdfplumber for better structure detection
and cleaner text extraction.

Author: Arthur Passuello
"""

import re
import pdfplumber
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any


class PDFPlumberParser:
    """Advanced PDF parser using pdfplumber for structure-aware extraction."""
    
    def __init__(self, target_chunk_size: int = 1400, min_chunk_size: int = 800,
                 max_chunk_size: int = 2000):
        """Initialize PDFPlumber parser."""
        self.target_chunk_size = target_chunk_size
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        
        # Trash content patterns
        self.trash_patterns = [
            r'Creative Commons.*?License',
            r'International License.*?authors',
            r'RISC-V International',
            r'Visit.*?for further',
            r'editors to suggest.*?corrections',
            r'released under.*?license',
            r'\.{5,}',  # Long dots (TOC artifacts)
            r'^\d+\s*$',  # Page numbers alone
        ]
        
    def extract_with_structure(self, pdf_path: Path) -> List[Dict]:
        """Extract PDF content with structure awareness using pdfplumber."""
        chunks = []
        
        with pdfplumber.open(pdf_path) as pdf:
            current_section = None
            current_text = []
            
            for page_num, page in enumerate(pdf.pages):
                # Extract text with formatting info
                page_content = self._extract_page_content(page, page_num + 1)
                
                for element in page_content:
                    if element['type'] == 'header':
                        # Save previous section if exists
                        if current_text:
                            chunk_text = '\n\n'.join(current_text)
                            if self._is_valid_chunk(chunk_text):
                                chunks.extend(self._create_chunks(
                                    chunk_text, 
                                    current_section or "Document",
                                    page_num
                                ))
                        
                        # Start new section
                        current_section = element['text']
                        current_text = []
                        
                    elif element['type'] == 'content':
                        # Add to current section
                        if self._is_valid_content(element['text']):
                            current_text.append(element['text'])
            
            # Don't forget last section
            if current_text:
                chunk_text = '\n\n'.join(current_text)
                if self._is_valid_chunk(chunk_text):
                    chunks.extend(self._create_chunks(
                        chunk_text,
                        current_section or "Document",
                        len(pdf.pages)
                    ))
        
        return chunks
    
    def _extract_page_content(self, page: Any, page_num: int) -> List[Dict]:
        """Extract structured content from a page."""
        content = []
        
        # Get all text with positioning
        chars = page.chars
        if not chars:
            return content
        
        # Group by lines
        lines = []
        current_line = []
        current_y = None
        
        for char in sorted(chars, key=lambda x: (x['top'], x['x0'])):
            if current_y is None or abs(char['top'] - current_y) < 2:
                current_line.append(char)
                current_y = char['top']
            else:
                if current_line:
                    lines.append(current_line)
                current_line = [char]
                current_y = char['top']
        
        if current_line:
            lines.append(current_line)
        
        # Analyze each line
        for line in lines:
            line_text = ''.join(char['text'] for char in line).strip()
            
            if not line_text:
                continue
            
            # Detect headers by font size
            avg_font_size = sum(char.get('size', 12) for char in line) / len(line)
            is_bold = any(char.get('fontname', '').lower().count('bold') > 0 for char in line)
            
            # Classify content
            if avg_font_size > 14 or is_bold:
                # Likely a header
                if self._is_valid_header(line_text):
                    content.append({
                        'type': 'header',
                        'text': line_text,
                        'font_size': avg_font_size,
                        'page': page_num
                    })
            else:
                # Regular content
                content.append({
                    'type': 'content',
                    'text': line_text,
                    'font_size': avg_font_size,
                    'page': page_num
                })
        
        return content
    
    def _is_valid_header(self, text: str) -> bool:
        """Check if text is a valid header."""
        # Skip if too short or too long
        if len(text) < 3 or len(text) > 200:
            return False
        
        # Skip if matches trash patterns
        for pattern in self.trash_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return False
        
        # Valid if starts with number or capital letter
        if re.match(r'^(\d+\.?\d*\s+|[A-Z])', text):
            return True
        
        # Valid if contains keywords
        keywords = ['chapter', 'section', 'introduction', 'conclusion', 'appendix']
        return any(keyword in text.lower() for keyword in keywords)
    
    def _is_valid_content(self, text: str) -> bool:
        """Check if text is valid content (not trash)."""
        # Skip very short text
        if len(text.strip()) < 10:
            return False
        
        # Skip trash patterns
        for pattern in self.trash_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return False
        
        return True
    
    def _is_valid_chunk(self, text: str) -> bool:
        """Check if chunk text is valid."""
        # Must have minimum length
        if len(text.strip()) < self.min_chunk_size // 2:
            return False
        
        # Must have some alphabetic content
        alpha_chars = sum(1 for c in text if c.isalpha())
        if alpha_chars < len(text) * 0.5:
            return False
        
        return True
    
    def _create_chunks(self, text: str, title: str, page: int) -> List[Dict]:
        """Create chunks from text."""
        chunks = []
        
        # Clean text
        text = self._clean_text(text)
        
        if len(text) <= self.max_chunk_size:
            # Single chunk
            chunks.append({
                'text': text,
                'title': title,
                'page': page,
                'metadata': {
                    'parsing_method': 'pdfplumber',
                    'quality_score': self._calculate_quality_score(text)
                }
            })
        else:
            # Split into chunks
            text_chunks = self._split_text_into_chunks(text)
            for i, chunk_text in enumerate(text_chunks):
                chunks.append({
                    'text': chunk_text,
                    'title': f"{title} (Part {i+1})",
                    'page': page,
                    'metadata': {
                        'parsing_method': 'pdfplumber',
                        'part_number': i + 1,
                        'total_parts': len(text_chunks),
                        'quality_score': self._calculate_quality_score(chunk_text)
                    }
                })
        
        return chunks
    
    def _clean_text(self, text: str) -> str:
        """Clean text from artifacts."""
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers at start/end
        text = re.sub(r'^\d+\s+', '', text)
        text = re.sub(r'\s+\d+$', '', text)
        
        # Remove excessive dots
        text = re.sub(r'\.{3,}', '', text)
        
        return text.strip()
    
    def _split_text_into_chunks(self, text: str) -> List[str]:
        """Split text into chunks at sentence boundaries."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            if current_size + sentence_size > self.target_chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_size = sentence_size
            else:
                current_chunk.append(sentence)
                current_size += sentence_size + 1
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def _calculate_quality_score(self, text: str) -> float:
        """Calculate quality score for chunk."""
        score = 1.0
        
        # Penalize very short or very long
        if len(text) < self.min_chunk_size:
            score *= 0.8
        elif len(text) > self.max_chunk_size:
            score *= 0.9
        
        # Reward complete sentences
        if text.strip().endswith(('.', '!', '?')):
            score *= 1.1
        
        # Reward technical content
        technical_terms = ['risc', 'instruction', 'register', 'memory', 'processor']
        term_count = sum(1 for term in technical_terms if term in text.lower())
        score *= (1 + term_count * 0.05)
        
        return min(score, 1.0)


def parse_pdf_with_pdfplumber(pdf_path: Path, **kwargs) -> List[Dict]:
    """Main entry point for PDFPlumber parsing."""
    parser = PDFPlumberParser(**kwargs)
    chunks = parser.extract_with_structure(pdf_path)
    
    print(f"PDFPlumber extracted {len(chunks)} chunks")
    
    return chunks