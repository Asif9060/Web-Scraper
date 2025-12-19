"""
LLM-powered page understanding.

Uses local LLM to extract semantic information from page content:
- Summaries
- Topics
- Named entities
- Key facts
- Relationships
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from web_intel.config import Settings
from web_intel.extraction import ExtractedContent
from web_intel.llm import LocalLLM, GenerationConfig, ConversationMessage
from web_intel.llm.local_llm import Role
from web_intel.llm.prompt_templates import ExtractionPrompts
from web_intel.understanding.chunker import TextChunker, ChunkingStrategy
from web_intel.utils.logging import get_logger

logger = get_logger(__name__)


class EntityType(str, Enum):
    """Types of named entities."""

    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    PRODUCT = "product"
    DATE = "date"
    MONEY = "money"
    EVENT = "event"
    TECHNOLOGY = "technology"
    OTHER = "other"


@dataclass
class ExtractedEntity:
    """A named entity extracted from content."""

    name: str
    entity_type: EntityType
    mentions: int = 1  # Number of times mentioned
    context: str = ""  # Surrounding text for disambiguation

    def __hash__(self) -> int:
        return hash((self.name.lower(), self.entity_type))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ExtractedEntity):
            return False
        return self.name.lower() == other.name.lower() and self.entity_type == other.entity_type


@dataclass
class ExtractedFact:
    """A key fact extracted from content."""

    statement: str
    confidence: float = 1.0  # 0-1 confidence score
    source_chunk: int = 0  # Which chunk it came from

    def __hash__(self) -> int:
        return hash(self.statement.lower())


@dataclass
class ExtractedRelationship:
    """A relationship between two entities."""

    subject: str
    predicate: str  # The relationship type
    object: str
    confidence: float = 1.0

    def __str__(self) -> str:
        return f"{self.subject} -> {self.predicate} -> {self.object}"


@dataclass
class PageSummary:
    """Summary of a page's content."""

    short_summary: str  # 1-2 sentences
    topics: list[str] = field(default_factory=list)
    key_points: list[str] = field(default_factory=list)


@dataclass
class UnderstandingResult:
    """
    Complete understanding of a page's content.

    Combines all extracted information into a unified structure.
    """

    url: str
    title: str
    summary: PageSummary
    entities: list[ExtractedEntity] = field(default_factory=list)
    facts: list[ExtractedFact] = field(default_factory=list)
    relationships: list[ExtractedRelationship] = field(default_factory=list)
    content_category: str = ""
    word_count: int = 0
    chunk_count: int = 0
    processing_errors: list[str] = field(default_factory=list)

    @property
    def entity_count(self) -> int:
        """Total number of unique entities."""
        return len(self.entities)

    @property
    def fact_count(self) -> int:
        """Total number of extracted facts."""
        return len(self.facts)

    def get_entities_by_type(self, entity_type: EntityType) -> list[ExtractedEntity]:
        """Get entities of a specific type."""
        return [e for e in self.entities if e.entity_type == entity_type]

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "url": self.url,
            "title": self.title,
            "summary": {
                "short": self.summary.short_summary,
                "topics": self.summary.topics,
                "key_points": self.summary.key_points,
            },
            "entities": [
                {"name": e.name, "type": e.entity_type.value, "mentions": e.mentions}
                for e in self.entities
            ],
            "facts": [{"statement": f.statement, "confidence": f.confidence} for f in self.facts],
            "relationships": [
                {"subject": r.subject, "predicate": r.predicate, "object": r.object}
                for r in self.relationships
            ],
            "category": self.content_category,
            "word_count": self.word_count,
        }


class PageUnderstanding:
    """
    LLM-powered page content understanding.

    Extracts semantic information from web pages using a local LLM.
    Handles long documents by chunking and aggregating results.

    Example:
        >>> llm = LocalLLM.from_settings(settings)
        >>> understanding = PageUnderstanding(llm)
        >>> result = understanding.understand(extracted_content)
        >>> print(result.summary.short_summary)
        >>> for entity in result.entities:
        ...     print(f"{entity.entity_type}: {entity.name}")
    """

    # Entity type mapping from LLM output
    ENTITY_TYPE_MAP = {
        "person": EntityType.PERSON,
        "people": EntityType.PERSON,
        "organization": EntityType.ORGANIZATION,
        "org": EntityType.ORGANIZATION,
        "company": EntityType.ORGANIZATION,
        "location": EntityType.LOCATION,
        "place": EntityType.LOCATION,
        "city": EntityType.LOCATION,
        "country": EntityType.LOCATION,
        "product": EntityType.PRODUCT,
        "date": EntityType.DATE,
        "time": EntityType.DATE,
        "money": EntityType.MONEY,
        "price": EntityType.MONEY,
        "event": EntityType.EVENT,
        "technology": EntityType.TECHNOLOGY,
        "tech": EntityType.TECHNOLOGY,
    }

    def __init__(
        self,
        llm: LocalLLM,
        chunk_size: int = 2000,
        chunk_overlap: int = 200,
        max_chunks_per_page: int = 10,
        generation_config: GenerationConfig | None = None,
    ) -> None:
        """
        Initialize page understanding.

        Args:
            llm: LocalLLM instance for inference
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between chunks
            max_chunks_per_page: Maximum chunks to process per page
            generation_config: Custom generation config
        """
        self.llm = llm
        self.chunker = TextChunker(
            chunk_size=chunk_size,
            overlap=chunk_overlap,
            strategy=ChunkingStrategy.PARAGRAPH,
        )
        self.max_chunks = max_chunks_per_page
        self.generation_config = generation_config or GenerationConfig(
            max_new_tokens=256,
            temperature=0.1,
            do_sample=False,
        )

    @classmethod
    def from_settings(cls, settings: Settings, llm: LocalLLM | None = None) -> "PageUnderstanding":
        """
        Create PageUnderstanding from settings.

        Args:
            settings: Application settings
            llm: Optional pre-configured LLM

        Returns:
            Configured PageUnderstanding instance
        """
        if llm is None:
            llm = LocalLLM.from_settings(settings)

        return cls(llm=llm)

    def understand(
        self,
        content: ExtractedContent,
        extract_entities: bool = True,
        extract_facts: bool = True,
        extract_relationships: bool = False,
    ) -> UnderstandingResult:
        """
        Extract understanding from page content.

        Args:
            content: ExtractedContent from content extractor
            extract_entities: Whether to extract named entities
            extract_facts: Whether to extract key facts
            extract_relationships: Whether to extract relationships

        Returns:
            UnderstandingResult with all extracted information
        """
        logger.info(f"Understanding page: {content.url}")

        errors = []
        text = content.main_text

        # Chunk the text
        chunks = self.chunker.chunk(text)
        if len(chunks) > self.max_chunks:
            logger.warning(
                f"Truncating from {len(chunks)} to {self.max_chunks} chunks"
            )
            chunks = chunks[: self.max_chunks]

        # Extract summary
        try:
            summary = self._extract_summary(text, chunks)
        except Exception as e:
            logger.error(f"Summary extraction failed: {e}")
            errors.append(f"Summary extraction failed: {e}")
            summary = PageSummary(short_summary=content.summary_text)

        # Extract entities
        entities = []
        if extract_entities:
            try:
                entities = self._extract_entities(chunks)
            except Exception as e:
                logger.error(f"Entity extraction failed: {e}")
                errors.append(f"Entity extraction failed: {e}")

        # Extract facts
        facts = []
        if extract_facts:
            try:
                facts = self._extract_facts(chunks)
            except Exception as e:
                logger.error(f"Fact extraction failed: {e}")
                errors.append(f"Fact extraction failed: {e}")

        # Extract relationships
        relationships = []
        if extract_relationships:
            try:
                relationships = self._extract_relationships(chunks)
            except Exception as e:
                logger.error(f"Relationship extraction failed: {e}")
                errors.append(f"Relationship extraction failed: {e}")

        # Classify content
        try:
            category = self._classify_content(text[:2000])
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            errors.append(f"Classification failed: {e}")
            category = content.content_type

        return UnderstandingResult(
            url=content.url,
            title=content.title,
            summary=summary,
            entities=entities,
            facts=facts,
            relationships=relationships,
            content_category=category,
            word_count=content.word_count,
            chunk_count=len(chunks),
            processing_errors=errors,
        )

    def _extract_summary(self, full_text: str, chunks: list) -> PageSummary:
        """Extract summary and topics from content."""
        # For short texts, summarize directly
        if len(chunks) <= 2:
            text_to_summarize = full_text[:3000]
        else:
            # Use first and last chunks for summary
            text_to_summarize = chunks[0].text + "\n\n" + chunks[-1].text

        # Generate summary
        prompt_data = ExtractionPrompts.SUMMARIZE.format(
            text=text_to_summarize)
        summary_result = self.llm.generate(
            prompt=prompt_data["user"],
            system_prompt=prompt_data["system"],
            config=self.generation_config,
        )
        short_summary = summary_result.text.strip()

        # Extract topics
        prompt_data = ExtractionPrompts.EXTRACT_TOPICS.format(
            text=text_to_summarize)
        topics_result = self.llm.generate(
            prompt=prompt_data["user"],
            system_prompt=prompt_data["system"],
            config=self.generation_config,
        )
        topics = self._parse_topics(topics_result.text)

        # Extract key points from first chunk
        key_points = self._extract_key_points(
            chunks[0].text if chunks else full_text[:2000])

        return PageSummary(
            short_summary=short_summary,
            topics=topics,
            key_points=key_points,
        )

    def _parse_topics(self, text: str) -> list[str]:
        """Parse comma-separated topics from LLM output."""
        topics = []
        for topic in text.split(","):
            topic = topic.strip().strip(".-•*")
            if topic and len(topic) > 1 and len(topic) < 100:
                topics.append(topic)
        return topics[:10]  # Limit to 10 topics

    def _extract_key_points(self, text: str) -> list[str]:
        """Extract key points from text."""
        prompt_data = ExtractionPrompts.EXTRACT_KEY_FACTS.format(
            text=text[:2000])
        result = self.llm.generate(
            prompt=prompt_data["user"],
            system_prompt=prompt_data["system"],
            config=self.generation_config,
        )

        points = []
        for line in result.text.split("\n"):
            line = line.strip().strip(".-•*123456789)")
            if line and len(line) > 10 and len(line) < 500:
                points.append(line)

        return points[:5]  # Limit to 5 key points

    def _extract_entities(self, chunks: list) -> list[ExtractedEntity]:
        """Extract named entities from all chunks."""
        entity_counts: dict[ExtractedEntity, int] = {}

        for chunk in chunks:
            prompt_data = ExtractionPrompts.EXTRACT_ENTITIES.format(
                text=chunk.text)
            result = self.llm.generate(
                prompt=prompt_data["user"],
                system_prompt=prompt_data["system"],
                config=self.generation_config,
            )

            for entity in self._parse_entities(result.text):
                if entity in entity_counts:
                    entity_counts[entity] += 1
                else:
                    entity_counts[entity] = 1

        # Create final list with mention counts
        entities = []
        for entity, count in entity_counts.items():
            entity.mentions = count
            entities.append(entity)

        # Sort by mention count
        entities.sort(key=lambda e: e.mentions, reverse=True)

        return entities

    def _parse_entities(self, text: str) -> list[ExtractedEntity]:
        """Parse entities from LLM output."""
        entities = []

        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue

            # Try to parse TYPE: name format
            match = re.match(r"^(\w+):\s*(.+)$", line)
            if match:
                type_str = match.group(1).lower()
                name = match.group(2).strip()

                entity_type = self.ENTITY_TYPE_MAP.get(
                    type_str, EntityType.OTHER)

                if name and len(name) > 1:
                    entities.append(
                        ExtractedEntity(
                            name=name,
                            entity_type=entity_type,
                        )
                    )

        return entities

    def _extract_facts(self, chunks: list) -> list[ExtractedFact]:
        """Extract key facts from all chunks."""
        all_facts = []
        seen_facts = set()

        for i, chunk in enumerate(chunks):
            prompt_data = ExtractionPrompts.EXTRACT_KEY_FACTS.format(
                text=chunk.text)
            result = self.llm.generate(
                prompt=prompt_data["user"],
                system_prompt=prompt_data["system"],
                config=self.generation_config,
            )

            for line in result.text.split("\n"):
                line = line.strip().strip(".-•*123456789)")
                if line and len(line) > 15 and len(line) < 500:
                    # Deduplicate similar facts
                    normalized = line.lower()
                    if normalized not in seen_facts:
                        seen_facts.add(normalized)
                        all_facts.append(
                            ExtractedFact(
                                statement=line,
                                source_chunk=i,
                            )
                        )

        return all_facts

    def _extract_relationships(self, chunks: list) -> list[ExtractedRelationship]:
        """Extract relationships between entities."""
        relationships = []

        # Only process first few chunks for relationships
        for chunk in chunks[:3]:
            prompt_data = ExtractionPrompts.EXTRACT_RELATIONSHIPS.format(
                text=chunk.text)
            result = self.llm.generate(
                prompt=prompt_data["user"],
                system_prompt=prompt_data["system"],
                config=self.generation_config,
            )

            for rel in self._parse_relationships(result.text):
                relationships.append(rel)

        return relationships

    def _parse_relationships(self, text: str) -> list[ExtractedRelationship]:
        """Parse relationships from LLM output."""
        relationships = []

        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue

            # Try to parse SUBJECT -> PREDICATE -> OBJECT format
            parts = re.split(r"\s*->\s*", line)
            if len(parts) == 3:
                subject, predicate, obj = parts
                if subject and predicate and obj:
                    relationships.append(
                        ExtractedRelationship(
                            subject=subject.strip(),
                            predicate=predicate.strip(),
                            object=obj.strip(),
                        )
                    )

        return relationships

    def _classify_content(self, text: str) -> str:
        """Classify the type of content."""
        categories = "article, blog post, product page, documentation, news, tutorial, FAQ, landing page, about page, contact page"

        prompt_data = ExtractionPrompts.CLASSIFY_CONTENT.format(
            text=text,
            categories=categories,
        )
        result = self.llm.generate(
            prompt=prompt_data["user"],
            system_prompt=prompt_data["system"],
            config=GenerationConfig(max_new_tokens=20, do_sample=False),
        )

        category = result.text.strip().lower()
        # Clean up response
        category = category.split("\n")[0].strip(".")

        return category if len(category) < 50 else "article"

    def summarize(self, text: str, max_sentences: int = 3) -> str:
        """
        Generate a summary of text.

        Convenience method for quick summarization.

        Args:
            text: Text to summarize
            max_sentences: Target number of sentences

        Returns:
            Summary text
        """
        prompt = ExtractionPrompts.SUMMARIZE.format(text=text[:4000])
        result = self.llm.generate(
            prompt=prompt["user"],
            system_prompt=prompt["system"],
            config=self.generation_config,
        )
        return result.text.strip()

    def extract_topics(self, text: str) -> list[str]:
        """
        Extract main topics from text.

        Args:
            text: Text to analyze

        Returns:
            List of topic strings
        """
        prompt = ExtractionPrompts.EXTRACT_TOPICS.format(text=text[:4000])
        result = self.llm.generate(
            prompt=prompt["user"],
            system_prompt=prompt["system"],
            config=self.generation_config,
        )
        return self._parse_topics(result.text)
