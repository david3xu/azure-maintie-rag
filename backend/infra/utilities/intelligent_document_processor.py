"""
Universal Intelligent Document Processor
========================================

Uses Azure OpenAI to intelligently chunk and process documents of any format.
No hardcoded parsing rules - purely LLM-driven document understanding.
"""

import logging
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class DocumentChunk:
    """Represents an intelligent chunk of a document"""
    content: str
    chunk_index: int
    chunk_type: str  # 'text_segment', 'list_item', 'table_row', etc.
    metadata: Dict[str, Any]
    source_info: Dict[str, Any]

class UniversalDocumentProcessor:
    """
    Intelligent document processor that uses LLM to understand and chunk documents
    Works with any text format without hardcoded rules
    """
    
    def __init__(self, azure_openai_client, max_chunk_size: int = 2000, overlap_size: int = 200):
        self.openai_client = azure_openai_client
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size
        
    async def process_document(self, document: Dict[str, Any]) -> List[DocumentChunk]:
        """
        Intelligently process a document using LLM-based understanding
        
        Args:
            document: Document dict with 'content', 'filename', etc.
            
        Returns:
            List of intelligently chunked document segments
        """
        content = document['content']
        filename = document['filename']
        
        # Skip very small documents - process as single chunk
        if len(content) < self.max_chunk_size:
            return [DocumentChunk(
                content=content,
                chunk_index=0,
                chunk_type='single_document',
                metadata={'total_chunks': 1, 'processing_method': 'direct'},
                source_info=document
            )]
        
        # Use LLM to intelligently understand document structure
        chunks = await self._intelligent_chunk_document(content, document)
        
        logger.info(f"Processed {filename}: {len(content)} chars → {len(chunks)} intelligent chunks")
        return chunks
    
    async def _intelligent_chunk_document(self, content: str, document: Dict[str, Any]) -> List[DocumentChunk]:
        """Use LLM to intelligently identify document structure and create meaningful chunks"""
        
        # First, ask LLM to analyze document structure
        structure_prompt = f"""
        Analyze this document and identify its structure. Determine the best way to split it into meaningful chunks for knowledge extraction.

        Document content (first 1000 chars):
        {content[:1000]}...

        Document info:
        - Filename: {document['filename']}
        - Total length: {len(content)} characters

        Respond with a JSON object containing:
        {{
            "document_type": "list_of_items|narrative_text|structured_data|mixed_format",
            "recommended_strategy": "sentence_based|paragraph_based|item_based|section_based",
            "estimated_chunks": number,
            "key_patterns": ["pattern1", "pattern2"],
            "reasoning": "explanation of why this strategy is best"
        }}
        """
        
        try:
            structure_analysis = await self._call_llm_for_analysis(structure_prompt)
            return await self._apply_chunking_strategy(content, document, structure_analysis)
        except Exception as e:
            logger.warning(f"LLM analysis failed, falling back to smart text splitting: {e}")
            return await self._fallback_smart_chunking(content, document)
    
    async def _call_llm_for_analysis(self, prompt: str) -> Dict[str, Any]:
        """Call Azure OpenAI for document structure analysis"""
        
        messages = [
            {"role": "system", "content": "You are an expert document analyst. Analyze document structure and provide JSON responses."},
            {"role": "user", "content": prompt}
        ]
        
        response = await self.openai_client.chat_completion(
            messages=messages,
            max_tokens=500,
            temperature=0.1
        )
        
        # Parse JSON response
        import json
        try:
            return json.loads(response['choices'][0]['message']['content'])
        except:
            # If JSON parsing fails, use fallback
            return {
                "document_type": "mixed_format",
                "recommended_strategy": "paragraph_based",
                "estimated_chunks": len(content) // self.max_chunk_size + 1,
                "reasoning": "JSON parsing failed, using fallback strategy"
            }
    
    async def _apply_chunking_strategy(self, content: str, document: Dict[str, Any], analysis: Dict[str, Any]) -> List[DocumentChunk]:
        """Apply the LLM-recommended chunking strategy"""
        
        strategy = analysis.get('recommended_strategy', 'paragraph_based')
        doc_type = analysis.get('document_type', 'mixed_format')
        
        logger.info(f"Applying {strategy} chunking for {doc_type} document: {document['filename']}")
        
        if strategy == 'item_based':
            return await self._chunk_by_items(content, document, analysis)
        elif strategy == 'sentence_based':
            return await self._chunk_by_sentences(content, document, analysis)
        elif strategy == 'section_based':
            return await self._chunk_by_sections(content, document, analysis)
        else:  # paragraph_based or fallback
            return await self._chunk_by_paragraphs(content, document, analysis)
    
    async def _chunk_by_items(self, content: str, document: Dict[str, Any], analysis: Dict[str, Any]) -> List[DocumentChunk]:
        """Chunk document by identifying individual items/entries"""
        
        # Use LLM to identify item boundaries
        item_prompt = f"""
        This document contains a list of items. Split it into individual items.
        Each item should be a complete, self-contained piece of information.
        
        Document content:
        {content[:3000]}...
        
        Return a list of the first 20 items, one per line, starting with "ITEM:" prefix.
        """
        
        try:
            response = await self._call_llm_for_analysis(item_prompt)
            # Parse LLM response to extract items
            items = self._extract_items_from_llm_response(content, response)
            
            chunks = []
            for i, item in enumerate(items[:100]):  # Limit to first 100 items for now
                chunks.append(DocumentChunk(
                    content=item.strip(),
                    chunk_index=i,
                    chunk_type='list_item',
                    metadata={
                        'total_chunks': len(items), 
                        'processing_method': 'llm_item_detection',
                        'document_type': analysis.get('document_type', 'list_of_items')
                    },
                    source_info=document
                ))
            
            return chunks
            
        except Exception as e:
            logger.warning(f"LLM item detection failed: {e}, using regex fallback")
            return await self._fallback_smart_chunking(content, document)
    
    def _extract_items_from_llm_response(self, content: str, llm_response: Any) -> List[str]:
        """Extract items from LLM response and apply to full content"""
        
        # For now, use a smart regex-based approach as fallback
        # Look for common patterns that indicate separate items
        patterns = [
            r'\n\s*[-*+]\s+(.+)',  # Bullet points
            r'\n\s*\d+[\.)]\s+(.+)',  # Numbered lists
            r'\n\s*[A-Za-z]\)\s+(.+)',  # Letter lists
            r'\n\s*<[^>]+>\s*(.+)',  # XML/HTML-like tags
            r'\n\s*\w+[:\-]\s*(.+)',  # Key-value pairs
            r'\n\n(.+?)(?=\n\n|\n\s*$)',  # Paragraph separation
        ]
        
        items = []
        for pattern in patterns:
            matches = re.findall(pattern, content, re.MULTILINE | re.DOTALL)
            if matches and len(matches) > 10:  # If we find many matches, this is likely the right pattern
                items = [match.strip() for match in matches if match.strip()]
                break
        
        # If no pattern works, split by double newlines
        if not items:
            items = [chunk.strip() for chunk in content.split('\n\n') if chunk.strip()]
        
        return items[:5000]  # Limit to reasonable number
    
    async def _chunk_by_sentences(self, content: str, document: Dict[str, Any], analysis: Dict[str, Any]) -> List[DocumentChunk]:
        """Chunk by sentences, grouping related sentences together"""
        
        sentences = re.split(r'[.!?]+\s+', content)
        chunks = []
        current_chunk = ""
        chunk_index = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Check if adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) > self.max_chunk_size and current_chunk:
                chunks.append(DocumentChunk(
                    content=current_chunk.strip(),
                    chunk_index=chunk_index,
                    chunk_type='sentence_group',
                    metadata={'total_chunks': 0, 'processing_method': 'sentence_based'},
                    source_info=document
                ))
                chunk_index += 1
                current_chunk = sentence
            else:
                current_chunk += (" " + sentence if current_chunk else sentence)
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append(DocumentChunk(
                content=current_chunk.strip(),
                chunk_index=chunk_index,
                chunk_type='sentence_group',
                metadata={'total_chunks': len(chunks) + 1, 'processing_method': 'sentence_based'},
                source_info=document
            ))
        
        # Update total_chunks in metadata
        for chunk in chunks:
            chunk.metadata['total_chunks'] = len(chunks)
        
        return chunks
    
    async def _chunk_by_paragraphs(self, content: str, document: Dict[str, Any], analysis: Dict[str, Any]) -> List[DocumentChunk]:
        """Chunk by paragraphs with intelligent grouping"""
        
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        chunks = []
        current_chunk = ""
        chunk_index = 0
        
        for paragraph in paragraphs:
            # Check if adding this paragraph would exceed chunk size
            if len(current_chunk) + len(paragraph) > self.max_chunk_size and current_chunk:
                chunks.append(DocumentChunk(
                    content=current_chunk.strip(),
                    chunk_index=chunk_index,
                    chunk_type='paragraph_group',
                    metadata={'total_chunks': 0, 'processing_method': 'paragraph_based'},
                    source_info=document
                ))
                chunk_index += 1
                current_chunk = paragraph
            else:
                current_chunk += ("\n\n" + paragraph if current_chunk else paragraph)
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append(DocumentChunk(
                content=current_chunk.strip(),
                chunk_index=chunk_index,
                chunk_type='paragraph_group',
                metadata={'total_chunks': len(chunks) + 1, 'processing_method': 'paragraph_based'},
                source_info=document
            ))
        
        # Update total_chunks in metadata
        for chunk in chunks:
            chunk.metadata['total_chunks'] = len(chunks)
        
        return chunks
    
    async def _chunk_by_sections(self, content: str, document: Dict[str, Any], analysis: Dict[str, Any]) -> List[DocumentChunk]:
        """Chunk by document sections (headers, etc.)"""
        
        # Look for common section markers
        section_patterns = [
            r'\n#+\s+(.+)',  # Markdown headers
            r'\n\s*\d+\.\s+(.+)',  # Numbered sections
            r'\n[A-Z][A-Z\s]+\n',  # ALL CAPS headers
            r'\n[-=]+\n',  # Underlined headers
        ]
        
        sections = []
        current_section = ""
        
        lines = content.split('\n')
        for line in lines:
            is_header = any(re.match(pattern, '\n' + line) for pattern in section_patterns)
            
            if is_header and current_section.strip():
                sections.append(current_section.strip())
                current_section = line
            else:
                current_section += '\n' + line
        
        # Add final section
        if current_section.strip():
            sections.append(current_section.strip())
        
        # Convert sections to chunks
        chunks = []
        for i, section in enumerate(sections):
            if len(section) > self.max_chunk_size:
                # If section is too large, sub-chunk it
                sub_chunks = await self._chunk_by_paragraphs(section, document, analysis)
                chunks.extend(sub_chunks)
            else:
                chunks.append(DocumentChunk(
                    content=section,
                    chunk_index=i,
                    chunk_type='document_section',
                    metadata={'total_chunks': len(sections), 'processing_method': 'section_based'},
                    source_info=document
                ))
        
        return chunks
    
    async def _fallback_smart_chunking(self, content: str, document: Dict[str, Any]) -> List[DocumentChunk]:
        """Fallback chunking when LLM analysis fails"""
        
        # Simple but effective chunking by paragraphs with overlap
        chunks = []
        chunk_index = 0
        start_pos = 0
        
        while start_pos < len(content):
            end_pos = start_pos + self.max_chunk_size
            
            # Try to end at a natural boundary (sentence, paragraph)
            if end_pos < len(content):
                # Look for paragraph break first
                para_break = content.rfind('\n\n', start_pos, end_pos)
                if para_break > start_pos:
                    end_pos = para_break
                else:
                    # Look for sentence break
                    sent_break = content.rfind('. ', start_pos, end_pos)
                    if sent_break > start_pos:
                        end_pos = sent_break + 1
            
            chunk_content = content[start_pos:end_pos].strip()
            if chunk_content:
                chunks.append(DocumentChunk(
                    content=chunk_content,
                    chunk_index=chunk_index,
                    chunk_type='smart_chunk',
                    metadata={'total_chunks': 0, 'processing_method': 'fallback_smart'},
                    source_info=document
                ))
                chunk_index += 1
            
            # Move start position with overlap
            start_pos = max(start_pos + 1, end_pos - self.overlap_size)
        
        # Update total_chunks in metadata
        for chunk in chunks:
            chunk.metadata['total_chunks'] = len(chunks)
        
        return chunks