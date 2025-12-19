"""
Query execution and answer generation.

Orchestrates retrieval, ranking, and LLM-based
answer generation for user queries.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

from web_intel.config import Settings
from web_intel.storage import Database, PageRepository, ChunkRepository
from web_intel.vector_store import VectorStore, VectorSearchResult, SearchFilter
from web_intel.query_parser import QueryParser, ParsedQuery, QueryExpander
from web_intel.query_executor.ranker import ResultRanker, RankedResult, FusionMethod
from web_intel.memory_store import MemoryStore, ContextManager, RetrievedContext
from web_intel.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class AnswerSource:
    """
    A source used in generating an answer.

    Tracks where information came from for citations.
    """

    page_id: int
    chunk_id: int
    url: str
    title: str
    relevance_score: float
    text_snippet: str = ""

    def to_citation(self, index: int) -> str:
        """Format as citation."""
        return f"[{index}] {self.title} ({self.url})"


@dataclass
class RetrievalResult:
    """
    Result of retrieval phase.

    Contains ranked results before answer generation.
    """

    query: str
    parsed_query: ParsedQuery
    results: list[RankedResult] = field(default_factory=list)
    vector_hits: int = 0
    keyword_hits: int = 0
    graph_hits: int = 0
    retrieval_time_ms: float = 0.0

    @property
    def total_hits(self) -> int:
        """Total unique results."""
        return len(self.results)

    def get_context_text(self, max_results: int = 5) -> str:
        """Get combined context text from top results."""
        texts = []
        for result in self.results[:max_results]:
            if result.source_title:
                texts.append(f"[{result.source_title}]\n{result.text}")
            else:
                texts.append(result.text)
        return "\n\n---\n\n".join(texts)


@dataclass
class QueryResult:
    """
    Complete result of query execution.

    Contains the answer, sources, and metadata.
    """

    query: str
    answer: str
    sources: list[AnswerSource] = field(default_factory=list)
    retrieval: RetrievalResult | None = None
    confidence: float = 0.0
    generated_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc))
    generation_time_ms: float = 0.0
    total_time_ms: float = 0.0

    @property
    def has_answer(self) -> bool:
        """Check if an answer was generated."""
        return bool(self.answer and self.answer.strip())

    def format_with_citations(self) -> str:
        """Format answer with numbered citations."""
        if not self.sources:
            return self.answer

        citations = "\n\nSources:\n"
        for i, source in enumerate(self.sources, 1):
            citations += f"{source.to_citation(i)}\n"

        return self.answer + citations


class QueryExecutor:
    """
    Executes queries against the knowledge base.

    Orchestrates the full query pipeline:
    1. Parse and expand query
    2. Retrieve from vector store and keyword search
    3. Rank and fuse results
    4. Generate answer with LLM

    Example:
        >>> executor = QueryExecutor.from_settings(settings)
        >>>
        >>> # Simple query
        >>> result = executor.execute("What products do you offer?")
        >>> print(result.answer)
        >>>
        >>> # With conversation context
        >>> result = executor.execute(
        ...     "Tell me more about the pricing",
        ...     session_id=session_id
        ... )
    """

    def __init__(
        self,
        database: Database,
        vector_store: VectorStore,
        memory_store: MemoryStore | None = None,
        answer_generator: Callable[[str, str], str] | None = None,
        ranker: ResultRanker | None = None,
        parser: QueryParser | None = None,
        expander: QueryExpander | None = None,
        top_k: int = 10,
        min_score: float = 0.3,
    ) -> None:
        """
        Initialize query executor.

        Args:
            database: Database instance
            vector_store: Vector store for semantic search
            memory_store: Optional memory store for conversations
            answer_generator: Function(question, context) -> answer
            ranker: Result ranker for fusion
            parser: Query parser
            expander: Query expander
            top_k: Number of results to retrieve
            min_score: Minimum relevance score threshold
        """
        self.db = database
        self.vector_store = vector_store
        self.memory_store = memory_store
        self.answer_generator = answer_generator
        self.ranker = ranker or ResultRanker(method=FusionMethod.RRF)
        self.parser = parser or QueryParser()
        self.expander = expander or QueryExpander()
        self.top_k = top_k
        self.min_score = min_score

        self._page_repo = PageRepository(database)
        self._chunk_repo = ChunkRepository(database)
        self._context_manager: ContextManager | None = None

        if memory_store:
            self._context_manager = ContextManager(
                memory_store=memory_store,
                max_tokens=4096,
            )

        logger.info("QueryExecutor initialized")

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        database: Database | None = None,
        vector_store: VectorStore | None = None,
        memory_store: MemoryStore | None = None,
        answer_generator: Callable[[str, str], str] | None = None,
    ) -> "QueryExecutor":
        """
        Create QueryExecutor from settings.

        Args:
            settings: Application settings
            database: Optional pre-configured database
            vector_store: Optional pre-configured vector store
            memory_store: Optional memory store
            answer_generator: Optional answer generation function

        Returns:
            Configured QueryExecutor instance
        """
        if database is None:
            database = Database.from_settings(settings)

        if vector_store is None:
            vector_store = VectorStore.from_settings(
                settings, database=database)

        return cls(
            database=database,
            vector_store=vector_store,
            memory_store=memory_store,
            answer_generator=answer_generator,
        )

    def execute(
        self,
        query: str,
        session_id: str | None = None,
        generate_answer: bool = True,
        crawl_id: int | None = None,
    ) -> QueryResult:
        """
        Execute a query and generate an answer.

        Args:
            query: User's question
            session_id: Optional conversation session ID
            generate_answer: Whether to generate LLM answer
            crawl_id: Optional filter to specific crawl

        Returns:
            QueryResult with answer and sources
        """
        import time

        start_time = time.perf_counter()

        # Parse query
        parsed = self.parser.parse(query)
        logger.debug(
            f"Parsed query: type={parsed.query_type.value}, terms={parsed.key_terms}")

        # Handle follow-up questions
        if session_id and parsed.is_follow_up and self._context_manager:
            return self._execute_follow_up(query, parsed, session_id, crawl_id)

        # Retrieve relevant content
        retrieval = self._retrieve(query, parsed, crawl_id)

        # Generate answer if requested and generator available
        answer = ""
        generation_time = 0.0

        if generate_answer and self.answer_generator and retrieval.results:
            gen_start = time.perf_counter()
            context = retrieval.get_context_text(max_results=5)
            answer = self.answer_generator(query, context)
            generation_time = (time.perf_counter() - gen_start) * 1000

        # Build sources
        sources = self._build_sources(retrieval.results[:5])

        # Calculate confidence
        confidence = self._calculate_confidence(retrieval, answer)

        # Store in memory if session provided
        if session_id and self._context_manager:
            self._context_manager.add_query_to_memory(session_id, query)
            if answer:
                self._context_manager.add_response_to_memory(
                    session_id,
                    answer,
                    sources=[s.url for s in sources],
                )

        total_time = (time.perf_counter() - start_time) * 1000

        return QueryResult(
            query=query,
            answer=answer,
            sources=sources,
            retrieval=retrieval,
            confidence=confidence,
            generation_time_ms=generation_time,
            total_time_ms=total_time,
        )

    def _retrieve(
        self,
        query: str,
        parsed: ParsedQuery,
        crawl_id: int | None = None,
    ) -> RetrievalResult:
        """Perform hybrid retrieval."""
        import time

        start_time = time.perf_counter()

        # Expand query for better retrieval
        expanded = self.expander.expand(parsed)

        # Vector search
        vector_results = self._vector_search(
            queries=expanded.all_queries[:3],
            crawl_id=crawl_id,
        )

        # Keyword search
        keyword_results = self._keyword_search(
            queries=self.expander.expand_for_keyword_search(parsed),
            crawl_id=crawl_id,
        )

        # Fuse results
        ranked = self.ranker.fuse(
            vector_results=vector_results,
            keyword_results=keyword_results,
        )

        # Apply diversity re-ranking
        ranked = self.ranker.rerank_with_diversity(ranked)

        # Filter by minimum score
        ranked = [r for r in ranked if r.final_score >= self.min_score]

        retrieval_time = (time.perf_counter() - start_time) * 1000

        return RetrievalResult(
            query=query,
            parsed_query=parsed,
            results=ranked[:self.top_k],
            vector_hits=len(vector_results),
            keyword_hits=len(keyword_results),
            retrieval_time_ms=retrieval_time,
        )

    def _vector_search(
        self,
        queries: list[str],
        crawl_id: int | None = None,
    ) -> list[dict]:
        """Perform vector similarity search."""
        results = []
        seen_chunks = set()

        filters = SearchFilter(
            crawl_id=crawl_id,
            min_score=self.min_score,
        )

        for query in queries:
            try:
                hits = self.vector_store.search_text(
                    query,
                    top_k=self.top_k,
                    filters=filters,
                )

                for hit in hits:
                    if hit.chunk_id in seen_chunks:
                        continue
                    seen_chunks.add(hit.chunk_id)

                    # Get page info
                    page = self._page_repo.get_by_id(hit.page_id)

                    results.append({
                        "chunk_id": hit.chunk_id,
                        "page_id": hit.page_id,
                        "text": hit.text,
                        "score": hit.score,
                        "source_url": page.url if page else "",
                        "source_title": page.title if page else "",
                    })

            except Exception as e:
                logger.warning(f"Vector search failed for query: {e}")

        return results

    def _keyword_search(
        self,
        queries: list[str],
        crawl_id: int | None = None,
    ) -> list[dict]:
        """Perform full-text keyword search."""
        results = []
        seen_pages = set()

        for query in queries:
            try:
                # Search pages using FTS
                pages = self._page_repo.search_fts(query, limit=self.top_k)

                for page in pages:
                    if crawl_id and page.crawl_id != crawl_id:
                        continue
                    if page.id in seen_pages:
                        continue
                    seen_pages.add(page.id)

                    # Get chunks for this page
                    chunks = self._chunk_repo.get_by_page(page.id, limit=3)

                    for chunk in chunks:
                        results.append({
                            "chunk_id": chunk.id,
                            "page_id": page.id,
                            "text": chunk.text,
                            "score": 1.0,  # FTS doesn't provide scores easily
                            "source_url": page.url,
                            "source_title": page.title,
                        })

            except Exception as e:
                logger.warning(f"Keyword search failed for query: {e}")

        return results

    def _execute_follow_up(
        self,
        query: str,
        parsed: ParsedQuery,
        session_id: str,
        crawl_id: int | None = None,
    ) -> QueryResult:
        """Execute a follow-up query with conversation context."""
        import time

        start_time = time.perf_counter()

        # Get conversation history
        conversation_context = self._context_manager.get_conversation_context(
            session_id, max_turns=3
        )

        # Retrieve with expanded context
        retrieval = self._retrieve(query, parsed, crawl_id)

        # Generate answer with conversation context
        answer = ""
        generation_time = 0.0

        if self.answer_generator and retrieval.results:
            gen_start = time.perf_counter()

            # Combine conversation history with new context
            new_context = retrieval.get_context_text(max_results=3)
            full_context = (
                f"Previous conversation:\n{conversation_context}\n\n"
                f"New context:\n{new_context}"
            )

            answer = self.answer_generator(query, full_context)
            generation_time = (time.perf_counter() - gen_start) * 1000

        # Build sources
        sources = self._build_sources(retrieval.results[:5])

        # Store in memory
        self._context_manager.add_query_to_memory(session_id, query)
        if answer:
            self._context_manager.add_response_to_memory(
                session_id,
                answer,
                sources=[s.url for s in sources],
            )

        total_time = (time.perf_counter() - start_time) * 1000

        return QueryResult(
            query=query,
            answer=answer,
            sources=sources,
            retrieval=retrieval,
            confidence=self._calculate_confidence(retrieval, answer),
            generation_time_ms=generation_time,
            total_time_ms=total_time,
        )

    def _build_sources(self, results: list[RankedResult]) -> list[AnswerSource]:
        """Build source citations from results."""
        sources = []
        seen_pages = set()

        for result in results:
            # Deduplicate by page
            if result.page_id in seen_pages:
                continue
            seen_pages.add(result.page_id)

            sources.append(
                AnswerSource(
                    page_id=result.page_id,
                    chunk_id=result.chunk_id,
                    url=result.source_url,
                    title=result.source_title,
                    relevance_score=result.final_score,
                    text_snippet=result.text[:200] if result.text else "",
                )
            )

        return sources

    def _calculate_confidence(
        self,
        retrieval: RetrievalResult,
        answer: str,
    ) -> float:
        """Calculate confidence score for the result."""
        if not retrieval.results:
            return 0.0

        # Base confidence on top result score
        top_score = retrieval.results[0].final_score if retrieval.results else 0.0

        # Boost if multiple sources agree
        source_count = len(set(r.page_id for r in retrieval.results[:5]))
        source_boost = min(0.2, source_count * 0.05)

        # Boost if answer was generated
        answer_boost = 0.1 if answer else 0.0

        confidence = top_score + source_boost + answer_boost
        return min(1.0, confidence)

    def retrieve_only(
        self,
        query: str,
        crawl_id: int | None = None,
    ) -> RetrievalResult:
        """
        Retrieve without generating an answer.

        Useful for getting raw retrieval results.

        Args:
            query: User's question
            crawl_id: Optional filter to specific crawl

        Returns:
            RetrievalResult with ranked chunks
        """
        parsed = self.parser.parse(query)
        return self._retrieve(query, parsed, crawl_id)

    def search_pages(
        self,
        query: str,
        limit: int = 10,
    ) -> list[dict]:
        """
        Search for relevant pages.

        Returns page-level results instead of chunks.

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            List of page info dictionaries
        """
        # Use FTS for page search
        pages = self._page_repo.search_fts(query, limit=limit)

        return [
            {
                "id": page.id,
                "url": page.url,
                "title": page.title,
                "summary": page.summary,
                "word_count": page.word_count,
            }
            for page in pages
        ]

    def get_page_context(self, page_id: int) -> str:
        """
        Get full context for a specific page.

        Args:
            page_id: Page ID

        Returns:
            Combined text from all chunks
        """
        chunks = self._chunk_repo.get_by_page(page_id)
        return "\n\n".join(chunk.text for chunk in chunks)

    def create_session(self) -> str:
        """
        Create a new conversation session.

        Returns:
            Session ID
        """
        if not self.memory_store:
            raise ValueError("Memory store not configured")
        return self.memory_store.create_session()
