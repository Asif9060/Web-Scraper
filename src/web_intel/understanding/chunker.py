"""
Text chunking for processing large documents.

Provides strategies for splitting text into manageable chunks
while preserving semantic coherence.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterator

from web_intel.utils.logging import get_logger

logger = get_logger(__name__)


class ChunkingStrategy(str, Enum):
    """Strategies for splitting text into chunks."""

    FIXED_SIZE = "fixed_size"  # Fixed character/token count
    SENTENCE = "sentence"  # Split on sentence boundaries
    PARAGRAPH = "paragraph"  # Split on paragraph boundaries
    SEMANTIC = "semantic"  # Split on semantic boundaries (headings, topics)


@dataclass
class Chunk:
    """
    A chunk of text from a larger document.

    Maintains metadata about position and context.
    """

    text: str
    index: int  # Position in sequence of chunks
    start_char: int  # Start position in original text
    end_char: int  # End position in original text
    metadata: dict = field(default_factory=dict)

    @property
    def char_count(self) -> int:
        """Number of characters in chunk."""
        return len(self.text)

    @property
    def word_count(self) -> int:
        """Approximate word count."""
        return len(self.text.split())

    def __repr__(self) -> str:
        preview = self.text[:50] + "..." if len(self.text) > 50 else self.text
        return f"Chunk(index={self.index}, words={self.word_count}, text={preview!r})"


class TextChunker:
    """
    Splits text into chunks for processing.

    Supports multiple strategies with configurable overlap
    to maintain context across chunk boundaries.

    Example:
        >>> chunker = TextChunker(chunk_size=1000, overlap=100)
        >>> chunks = chunker.chunk(long_text)
        >>> for chunk in chunks:
        ...     process(chunk.text)
    """

    # Sentence-ending patterns
    SENTENCE_ENDINGS = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")

    # Paragraph separators
    PARAGRAPH_SEP = re.compile(r"\n\s*\n")

    # Heading patterns (Markdown-style or text patterns)
    HEADING_PATTERNS = re.compile(
        r"(?:^|\n)(?:#{1,6}\s+.+|[A-Z][A-Za-z\s]+:(?=\s*\n))",
        re.MULTILINE,
    )

    def __init__(
        self,
        chunk_size: int = 1000,
        overlap: int = 100,
        strategy: ChunkingStrategy = ChunkingStrategy.SENTENCE,
        min_chunk_size: int = 100,
    ) -> None:
        """
        Initialize text chunker.

        Args:
            chunk_size: Target chunk size in characters
            overlap: Overlap between chunks in characters
            strategy: Chunking strategy to use
            min_chunk_size: Minimum chunk size (smaller chunks merged)
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.strategy = strategy
        self.min_chunk_size = min_chunk_size

        if overlap >= chunk_size:
            raise ValueError("Overlap must be smaller than chunk_size")

    def chunk(self, text: str) -> list[Chunk]:
        """
        Split text into chunks.

        Args:
            text: Text to chunk

        Returns:
            List of Chunk objects
        """
        if not text or not text.strip():
            return []

        if len(text) <= self.chunk_size:
            return [Chunk(text=text, index=0, start_char=0, end_char=len(text))]

        if self.strategy == ChunkingStrategy.FIXED_SIZE:
            return self._chunk_fixed_size(text)
        elif self.strategy == ChunkingStrategy.SENTENCE:
            return self._chunk_by_sentence(text)
        elif self.strategy == ChunkingStrategy.PARAGRAPH:
            return self._chunk_by_paragraph(text)
        elif self.strategy == ChunkingStrategy.SEMANTIC:
            return self._chunk_semantic(text)
        else:
            return self._chunk_fixed_size(text)

    def chunk_iterator(self, text: str) -> Iterator[Chunk]:
        """
        Iterate over chunks without storing all in memory.

        Args:
            text: Text to chunk

        Yields:
            Chunk objects
        """
        yield from self.chunk(text)

    def _chunk_fixed_size(self, text: str) -> list[Chunk]:
        """Split text into fixed-size chunks with overlap."""
        chunks = []
        start = 0
        index = 0

        while start < len(text):
            # Calculate end position
            end = min(start + self.chunk_size, len(text))

            # Extract chunk
            chunk_text = text[start:end]

            chunks.append(
                Chunk(
                    text=chunk_text,
                    index=index,
                    start_char=start,
                    end_char=end,
                )
            )

            # Move start, accounting for overlap
            start = end - self.overlap if end < len(text) else end
            index += 1

        return chunks

    def _chunk_by_sentence(self, text: str) -> list[Chunk]:
        """Split text on sentence boundaries."""
        # Split into sentences
        sentences = self.SENTENCE_ENDINGS.split(text)

        return self._combine_segments(sentences, text)

    def _chunk_by_paragraph(self, text: str) -> list[Chunk]:
        """Split text on paragraph boundaries."""
        # Split into paragraphs
        paragraphs = self.PARAGRAPH_SEP.split(text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        return self._combine_segments(paragraphs, text)

    def _chunk_semantic(self, text: str) -> list[Chunk]:
        """Split text on semantic boundaries (headings, topics)."""
        # Find heading positions
        headings = list(self.HEADING_PATTERNS.finditer(text))

        if not headings:
            # Fall back to paragraph chunking
            return self._chunk_by_paragraph(text)

        # Split at heading positions
        segments = []
        prev_end = 0

        for match in headings:
            # Add content before this heading
            if match.start() > prev_end:
                segment = text[prev_end: match.start()].strip()
                if segment:
                    segments.append(segment)

            prev_end = match.start()

        # Add final segment
        if prev_end < len(text):
            segment = text[prev_end:].strip()
            if segment:
                segments.append(segment)

        return self._combine_segments(segments, text)

    def _combine_segments(self, segments: list[str], original_text: str) -> list[Chunk]:
        """Combine segments into chunks of appropriate size."""
        chunks = []
        current_text = ""
        current_start = 0
        index = 0

        for segment in segments:
            segment = segment.strip()
            if not segment:
                continue

            # Check if adding this segment would exceed chunk size
            if current_text and len(current_text) + len(segment) + 1 > self.chunk_size:
                # Save current chunk
                if len(current_text) >= self.min_chunk_size:
                    end_pos = current_start + len(current_text)
                    chunks.append(
                        Chunk(
                            text=current_text,
                            index=index,
                            start_char=current_start,
                            end_char=end_pos,
                        )
                    )
                    index += 1

                    # Start new chunk with overlap
                    if self.overlap > 0 and current_text:
                        # Get last part of current chunk for overlap
                        overlap_text = current_text[-self.overlap:]
                        current_text = overlap_text + " " + segment
                        current_start = end_pos - self.overlap
                    else:
                        current_text = segment
                        current_start = self._find_position(
                            segment, original_text, current_start)
                else:
                    # Chunk too small, keep building
                    current_text = (current_text + " " + segment).strip()
            else:
                # Add segment to current chunk
                if current_text:
                    current_text = current_text + " " + segment
                else:
                    current_text = segment
                    current_start = self._find_position(
                        segment, original_text, 0)

        # Add final chunk
        if current_text and len(current_text) >= self.min_chunk_size:
            chunks.append(
                Chunk(
                    text=current_text,
                    index=index,
                    start_char=current_start,
                    end_char=current_start + len(current_text),
                )
            )
        elif current_text and chunks:
            # Merge small final chunk with previous
            prev = chunks[-1]
            chunks[-1] = Chunk(
                text=prev.text + " " + current_text,
                index=prev.index,
                start_char=prev.start_char,
                end_char=prev.start_char +
                len(prev.text) + 1 + len(current_text),
            )

        return chunks

    def _find_position(self, segment: str, text: str, start_from: int) -> int:
        """Find position of segment in original text."""
        # Try exact match first
        pos = text.find(segment, start_from)
        if pos >= 0:
            return pos

        # Try finding first few words
        words = segment.split()[:3]
        if words:
            search_text = " ".join(words)
            pos = text.find(search_text, start_from)
            if pos >= 0:
                return pos

        return start_from

    def estimate_chunk_count(self, text: str) -> int:
        """Estimate number of chunks without actually chunking."""
        if not text:
            return 0

        text_len = len(text)
        if text_len <= self.chunk_size:
            return 1

        effective_chunk = self.chunk_size - self.overlap
        return max(1, (text_len + effective_chunk - 1) // effective_chunk)


class TokenAwareChunker(TextChunker):
    """
    Chunker that respects token limits.

    Uses a tokenizer to ensure chunks don't exceed token limits,
    which is important for LLM context windows.
    """

    def __init__(
        self,
        tokenizer_func,
        max_tokens: int = 512,
        overlap_tokens: int = 50,
        strategy: ChunkingStrategy = ChunkingStrategy.SENTENCE,
    ) -> None:
        """
        Initialize token-aware chunker.

        Args:
            tokenizer_func: Function that returns token count for text
            max_tokens: Maximum tokens per chunk
            overlap_tokens: Token overlap between chunks
            strategy: Chunking strategy
        """
        # Estimate chars per token (rough approximation)
        chars_per_token = 4
        super().__init__(
            chunk_size=max_tokens * chars_per_token,
            overlap=overlap_tokens * chars_per_token,
            strategy=strategy,
        )
        self.tokenizer_func = tokenizer_func
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    def chunk(self, text: str) -> list[Chunk]:
        """Split text respecting token limits."""
        # First pass with character-based chunking
        initial_chunks = super().chunk(text)

        # Verify and split any chunks that exceed token limit
        final_chunks = []
        for chunk in initial_chunks:
            token_count = self.tokenizer_func(chunk.text)

            if token_count <= self.max_tokens:
                final_chunks.append(chunk)
            else:
                # Split oversized chunk
                sub_chunks = self._split_oversized(chunk)
                final_chunks.extend(sub_chunks)

        # Re-index
        for i, chunk in enumerate(final_chunks):
            chunk.index = i

        return final_chunks

    def _split_oversized(self, chunk: Chunk) -> list[Chunk]:
        """Split a chunk that exceeds token limit."""
        text = chunk.text
        sub_chunks = []
        start = 0

        while start < len(text):
            # Binary search for maximum text that fits in token limit
            low, high = 0, len(text) - start
            best = min(100, high)  # Start with minimum

            while low < high:
                mid = (low + high + 1) // 2
                test_text = text[start: start + mid]
                tokens = self.tokenizer_func(test_text)

                if tokens <= self.max_tokens:
                    best = mid
                    low = mid
                else:
                    high = mid - 1

            # Extract chunk
            end = start + best
            chunk_text = text[start:end].strip()

            if chunk_text:
                sub_chunks.append(
                    Chunk(
                        text=chunk_text,
                        index=len(sub_chunks),
                        start_char=chunk.start_char + start,
                        end_char=chunk.start_char + end,
                    )
                )

            start = end

        return sub_chunks
