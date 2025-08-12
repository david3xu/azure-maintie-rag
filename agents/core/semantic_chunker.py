"""
Semantic Chunking for Large Document Processing
==============================================

Intelligent text chunking that preserves semantic boundaries and context.
Supports the Azure Universal RAG system's zero-hardcoded-values philosophy.
"""

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ChunkMetadata:
    """Metadata for each chunk."""

    chunk_id: int
    start_char: int
    end_char: int
    word_count: int
    sentence_count: int
    contains_headers: bool
    contains_code: bool
    overlap_with_previous: int
    overlap_with_next: int


@dataclass
class ChunkedContent:
    """Result of chunking operation."""

    chunks: List[str]
    metadata: List[ChunkMetadata]
    total_chunks: int
    total_chars: int
    average_chunk_size: float
    overlap_ratio: float


class SemanticChunker:
    """
    Semantic text chunker that preserves context boundaries.

    Features:
    - Sentence boundary preservation
    - Header/section boundary detection
    - Code block preservation
    - Adaptive overlap based on content complexity
    - Zero hardcoded domain assumptions
    """

    def __init__(self):
        # Sentence boundary patterns (universal, not domain-specific)
        self.sentence_endings = re.compile(r"[.!?]+(?:\s|$)")
        self.header_patterns = re.compile(r"^#{1,6}\s+.+$|^[A-Z][^a-z]*$", re.MULTILINE)
        self.code_block_patterns = re.compile(r"```[\s\S]*?```|`[^`]+`")
        self.paragraph_break = re.compile(r"\n\s*\n")

    def chunk_text(
        self,
        text: str,
        chunk_size: int = 1000,
        overlap_ratio: float = 0.2,
        preserve_sentences: bool = True,
        preserve_code_blocks: bool = True,
    ) -> ChunkedContent:
        """
        Chunk text intelligently preserving semantic boundaries.

        Args:
            text: Input text to chunk
            chunk_size: Target chunk size in characters
            overlap_ratio: Overlap between chunks (0.0 to 0.5)
            preserve_sentences: Avoid breaking sentences
            preserve_code_blocks: Keep code blocks intact

        Returns:
            ChunkedContent with chunks and metadata
        """
        if not text or len(text) <= chunk_size:
            # Single chunk case
            return ChunkedContent(
                chunks=[text],
                metadata=[
                    ChunkMetadata(
                        chunk_id=0,
                        start_char=0,
                        end_char=len(text),
                        word_count=len(text.split()),
                        sentence_count=len(self.sentence_endings.findall(text)),
                        contains_headers=bool(self.header_patterns.search(text)),
                        contains_code=bool(self.code_block_patterns.search(text)),
                        overlap_with_previous=0,
                        overlap_with_next=0,
                    )
                ],
                total_chunks=1,
                total_chars=len(text),
                average_chunk_size=len(text),
                overlap_ratio=0.0,
            )

        chunks = []
        metadata = []
        overlap_size = int(chunk_size * overlap_ratio)

        # Find semantic boundaries
        semantic_boundaries = self._find_semantic_boundaries(text)

        start_pos = 0
        chunk_id = 0

        while start_pos < len(text):
            # Calculate chunk end position
            end_pos = min(start_pos + chunk_size, len(text))

            # Adjust for semantic boundaries if not at text end
            if end_pos < len(text):
                end_pos = self._adjust_for_boundaries(
                    text,
                    start_pos,
                    end_pos,
                    semantic_boundaries,
                    preserve_sentences,
                    preserve_code_blocks,
                )

            # Extract chunk
            chunk_text = text[start_pos:end_pos]

            # Calculate overlap with previous chunk
            overlap_with_prev = 0
            if chunk_id > 0:
                prev_end = metadata[-1].end_char
                overlap_with_prev = max(0, prev_end - start_pos)

            # Create metadata
            chunk_meta = ChunkMetadata(
                chunk_id=chunk_id,
                start_char=start_pos,
                end_char=end_pos,
                word_count=len(chunk_text.split()),
                sentence_count=len(self.sentence_endings.findall(chunk_text)),
                contains_headers=bool(self.header_patterns.search(chunk_text)),
                contains_code=bool(self.code_block_patterns.search(chunk_text)),
                overlap_with_previous=overlap_with_prev,
                overlap_with_next=0,  # Will be updated for previous chunk
            )

            # Update overlap for previous chunk
            if metadata:
                metadata[-1].overlap_with_next = overlap_with_prev

            chunks.append(chunk_text)
            metadata.append(chunk_meta)

            # Move to next chunk position with overlap
            if end_pos >= len(text):
                break

            start_pos = max(start_pos + 1, end_pos - overlap_size)
            chunk_id += 1

        # Calculate statistics
        total_chars = sum(len(chunk) for chunk in chunks)
        average_chunk_size = total_chars / len(chunks) if chunks else 0

        return ChunkedContent(
            chunks=chunks,
            metadata=metadata,
            total_chunks=len(chunks),
            total_chars=len(text),
            average_chunk_size=average_chunk_size,
            overlap_ratio=overlap_ratio,
        )

    def _find_semantic_boundaries(self, text: str) -> List[int]:
        """Find positions where semantic breaks naturally occur."""
        boundaries = set()

        # Paragraph boundaries
        for match in self.paragraph_break.finditer(text):
            boundaries.add(match.end())

        # Header boundaries
        for match in self.header_patterns.finditer(text):
            boundaries.add(match.start())
            boundaries.add(match.end())

        # Sentence boundaries
        for match in self.sentence_endings.finditer(text):
            boundaries.add(match.end())

        return sorted(list(boundaries))

    def _adjust_for_boundaries(
        self,
        text: str,
        start_pos: int,
        end_pos: int,
        boundaries: List[int],
        preserve_sentences: bool,
        preserve_code_blocks: bool,
    ) -> int:
        """Adjust chunk end position to respect semantic boundaries."""

        # Check if we're in the middle of a code block
        if preserve_code_blocks:
            code_block_end = self._find_code_block_end(text, start_pos, end_pos)
            if code_block_end and code_block_end > end_pos:
                # Extend to include complete code block
                return min(code_block_end, len(text))

        # Find the best boundary near the target end position
        if preserve_sentences:
            # Look for sentence boundaries within a reasonable range
            search_range = min(
                200, end_pos - start_pos // 4
            )  # 25% of chunk size or 200 chars

            # Find boundaries within range
            candidates = [
                b
                for b in boundaries
                if end_pos - search_range <= b <= end_pos + search_range
            ]

            if candidates:
                # Choose boundary closest to target
                return min(candidates, key=lambda x: abs(x - end_pos))

        return end_pos

    def _find_code_block_end(self, text: str, start_pos: int, end_pos: int) -> int:
        """Find end of code block if chunk ends inside one."""
        chunk_text = text[start_pos:end_pos]

        # Count opening and closing code blocks
        opening_blocks = chunk_text.count("```")

        # If odd number, we're inside a code block
        if opening_blocks % 2 == 1:
            # Find the next closing ```
            remaining_text = text[end_pos:]
            next_close = remaining_text.find("```")
            if next_close != -1:
                return end_pos + next_close + 3

        return None

    def get_chunk_summary(self, chunked_content: ChunkedContent) -> Dict[str, Any]:
        """Get summary statistics about chunking operation."""
        metadata = chunked_content.metadata

        return {
            "total_chunks": chunked_content.total_chunks,
            "total_characters": chunked_content.total_chars,
            "average_chunk_size": chunked_content.average_chunk_size,
            "size_range": {
                "min_size": (
                    min(len(chunk) for chunk in chunked_content.chunks)
                    if chunked_content.chunks
                    else 0
                ),
                "max_size": (
                    max(len(chunk) for chunk in chunked_content.chunks)
                    if chunked_content.chunks
                    else 0
                ),
            },
            "content_analysis": {
                "chunks_with_headers": sum(1 for m in metadata if m.contains_headers),
                "chunks_with_code": sum(1 for m in metadata if m.contains_code),
                "total_sentences": sum(m.sentence_count for m in metadata),
                "total_words": sum(m.word_count for m in metadata),
            },
            "overlap_analysis": {
                "target_overlap_ratio": chunked_content.overlap_ratio,
                "average_overlap": (
                    sum(m.overlap_with_next for m in metadata) / len(metadata)
                    if metadata
                    else 0
                ),
                "chunks_with_overlap": sum(
                    1 for m in metadata if m.overlap_with_next > 0
                ),
            },
        }


# Convenience function for easy integration
def chunk_text_semantically(
    text: str, chunk_size: int = 1000, overlap_ratio: float = 0.2
) -> ChunkedContent:
    """Convenience function for semantic chunking."""
    chunker = SemanticChunker()
    return chunker.chunk_text(text, chunk_size, overlap_ratio)
