"""
Prompt templates for LLM operations.

Provides structured prompts for:
- Content extraction tasks
- Entity extraction
- Query interpretation
- Answer generation
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class PromptTemplate:
    """
    A reusable prompt template with variable substitution.

    Example:
        >>> template = PromptTemplate(
        ...     name="summarize",
        ...     system="You are a helpful assistant.",
        ...     user="Summarize this text:\\n\\n{text}",
        ... )
        >>> prompt = template.format(text="Long article here...")
    """

    name: str
    system: str
    user: str
    assistant_prefix: str = ""  # Optional prefix for assistant response

    def format(self, **kwargs: Any) -> dict[str, str]:
        """
        Format the template with provided variables.

        Args:
            **kwargs: Variables to substitute

        Returns:
            Dictionary with formatted system and user prompts
        """
        return {
            "system": self.system.format(**kwargs) if kwargs else self.system,
            "user": self.user.format(**kwargs) if kwargs else self.user,
            "assistant_prefix": self.assistant_prefix,
        }

    def format_user(self, **kwargs: Any) -> str:
        """Format just the user prompt."""
        return self.user.format(**kwargs) if kwargs else self.user


class ExtractionPrompts:
    """
    Prompt templates for content extraction tasks.

    Optimized for local LLM (Qwen2.5-1.5B) with clear, structured prompts.
    """

    SUMMARIZE = PromptTemplate(
        name="summarize",
        system=(
            "You are a precise summarization assistant. "
            "Create clear, concise summaries that capture the key information. "
            "Focus on facts and main points. Do not add opinions or information not in the source."
        ),
        user=(
            "Summarize the following text in 2-3 sentences:\n\n"
            "---\n{text}\n---\n\n"
            "Summary:"
        ),
    )

    EXTRACT_TOPICS = PromptTemplate(
        name="extract_topics",
        system=(
            "You are a topic extraction assistant. "
            "Identify the main topics and themes from text. "
            "Return only a comma-separated list of topics."
        ),
        user=(
            "Extract the main topics from this text:\n\n"
            "---\n{text}\n---\n\n"
            "Topics (comma-separated):"
        ),
    )

    EXTRACT_ENTITIES = PromptTemplate(
        name="extract_entities",
        system=(
            "You are an entity extraction assistant. "
            "Identify named entities (people, organizations, locations, products, dates). "
            "Format each entity as: TYPE: name"
        ),
        user=(
            "Extract named entities from this text:\n\n"
            "---\n{text}\n---\n\n"
            "Entities (one per line, format TYPE: name):"
        ),
    )

    EXTRACT_KEY_FACTS = PromptTemplate(
        name="extract_key_facts",
        system=(
            "You are a fact extraction assistant. "
            "Extract key factual statements from text. "
            "Each fact should be a single, verifiable statement."
        ),
        user=(
            "Extract key facts from this text:\n\n"
            "---\n{text}\n---\n\n"
            "Key facts (one per line):"
        ),
    )

    CLASSIFY_CONTENT = PromptTemplate(
        name="classify_content",
        system=(
            "You are a content classification assistant. "
            "Classify content into one of the provided categories. "
            "Return only the category name, nothing else."
        ),
        user=(
            "Classify this content into one of these categories: {categories}\n\n"
            "Content:\n---\n{text}\n---\n\n"
            "Category:"
        ),
    )

    EXTRACT_RELATIONSHIPS = PromptTemplate(
        name="extract_relationships",
        system=(
            "You are a relationship extraction assistant. "
            "Identify relationships between entities in text. "
            "Format: ENTITY1 -> RELATIONSHIP -> ENTITY2"
        ),
        user=(
            "Extract relationships from this text:\n\n"
            "---\n{text}\n---\n\n"
            "Relationships (one per line):"
        ),
    )

    CLEAN_TEXT = PromptTemplate(
        name="clean_text",
        system=(
            "You are a text cleaning assistant. "
            "Clean and format the text while preserving its meaning. "
            "Remove noise, fix formatting, but keep all important content."
        ),
        user=(
            "Clean and format this text:\n\n"
            "---\n{text}\n---\n\n"
            "Cleaned text:"
        ),
    )


class QueryPrompts:
    """
    Prompt templates for query processing and answer generation.

    Optimized for API LLM (Claude) with detailed instructions.
    """

    INTERPRET_QUERY = PromptTemplate(
        name="interpret_query",
        system=(
            "You are a query interpretation assistant for a website knowledge base. "
            "Analyze user questions and extract:\n"
            "1. The main intent (what they want to know)\n"
            "2. Key entities or topics mentioned\n"
            "3. Any constraints (time, location, etc.)\n"
            "4. The type of answer expected (fact, explanation, list, comparison)\n\n"
            "Format your response as structured data."
        ),
        user=(
            "Interpret this query about a website:\n\n"
            "Query: {query}\n\n"
            "Website context: {context}\n\n"
            "Interpretation:"
        ),
    )

    GENERATE_SEARCH_QUERIES = PromptTemplate(
        name="generate_search_queries",
        system=(
            "You are a search query expansion assistant. "
            "Generate multiple search queries that would help find relevant information. "
            "Include variations, synonyms, and related concepts."
        ),
        user=(
            "Generate 3-5 search queries to find information for:\n\n"
            "User question: {query}\n\n"
            "Search queries (one per line):"
        ),
    )

    ANSWER_QUESTION = PromptTemplate(
        name="answer_question",
        system=(
            "You are a knowledgeable assistant answering questions about a website. "
            "Use ONLY the provided context to answer. "
            "If the context doesn't contain enough information, say so clearly. "
            "Be accurate, helpful, and cite specific parts of the context when relevant. "
            "Do not make up information that isn't in the context."
        ),
        user=(
            "Answer this question using the provided context:\n\n"
            "Question: {question}\n\n"
            "Context from the website:\n"
            "---\n{context}\n---\n\n"
            "Answer:"
        ),
    )

    ANSWER_WITH_SOURCES = PromptTemplate(
        name="answer_with_sources",
        system=(
            "You are a research assistant answering questions about a website. "
            "Use ONLY the provided sources to answer. "
            "After your answer, list the sources you used. "
            "If sources don't contain enough information, say so. "
            "Be accurate and cite your sources."
        ),
        user=(
            "Answer this question using the provided sources:\n\n"
            "Question: {question}\n\n"
            "Sources:\n{sources}\n\n"
            "Provide your answer, then list sources used:\n\n"
            "Answer:"
        ),
    )

    REFINE_ANSWER = PromptTemplate(
        name="refine_answer",
        system=(
            "You are an answer refinement assistant. "
            "Improve the previous answer using additional context. "
            "Keep what's accurate, add new information, correct any errors."
        ),
        user=(
            "Question: {question}\n\n"
            "Previous answer: {previous_answer}\n\n"
            "Additional context:\n---\n{context}\n---\n\n"
            "Refined answer:"
        ),
    )

    COMPARE_PAGES = PromptTemplate(
        name="compare_pages",
        system=(
            "You are a comparison assistant. "
            "Compare information from different pages of a website. "
            "Identify similarities, differences, and any contradictions."
        ),
        user=(
            "Compare these pages from the website:\n\n"
            "Page 1 ({page1_title}):\n{page1_content}\n\n"
            "Page 2 ({page2_title}):\n{page2_content}\n\n"
            "Comparison:"
        ),
    )

    SYNTHESIZE_INFO = PromptTemplate(
        name="synthesize_info",
        system=(
            "You are an information synthesis assistant. "
            "Combine information from multiple sources into a coherent summary. "
            "Resolve any conflicts and present a unified view."
        ),
        user=(
            "Synthesize information about: {topic}\n\n"
            "Sources:\n{sources}\n\n"
            "Synthesized summary:"
        ),
    )

    CONVERSATIONAL_FOLLOW_UP = PromptTemplate(
        name="conversational_follow_up",
        system=(
            "You are a helpful assistant having a conversation about a website. "
            "Use the conversation history and new context to answer follow-up questions. "
            "Maintain context from the conversation while incorporating new information."
        ),
        user=(
            "Conversation history:\n{history}\n\n"
            "New context:\n{context}\n\n"
            "Follow-up question: {question}\n\n"
            "Response:"
        ),
    )


class ChunkingPrompts:
    """Prompts for intelligent text chunking decisions."""

    FIND_CHUNK_BOUNDARIES = PromptTemplate(
        name="find_chunk_boundaries",
        system=(
            "You are a text segmentation assistant. "
            "Identify natural break points in text where topics change. "
            "Return line numbers where new sections begin."
        ),
        user=(
            "Find topic boundaries in this text (return line numbers):\n\n"
            "{text}\n\n"
            "Break points (line numbers, comma-separated):"
        ),
    )

    SUMMARIZE_CHUNK = PromptTemplate(
        name="summarize_chunk",
        system=(
            "You are a chunk summarization assistant. "
            "Create a brief summary (1-2 sentences) capturing the main point of this text chunk. "
            "This summary will be used for retrieval, so include key terms."
        ),
        user=(
            "Summarize this chunk for retrieval:\n\n"
            "---\n{chunk}\n---\n\n"
            "Summary:"
        ),
    )
