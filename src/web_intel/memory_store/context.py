"""
Context management for conversations.

Handles context window management, retrieval integration,
and token budgeting for LLM interactions.
"""

from dataclasses import dataclass, field
from typing import Callable

from web_intel.memory_store.store import (
    MemoryStore,
    ConversationMemory,
    MemoryEntry,
    MemoryType,
)
from web_intel.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RetrievedContext:
    """
    Context retrieved from the knowledge base.

    Contains relevant chunks and their sources.
    """

    chunk_id: int
    page_id: int
    text: str
    score: float
    source_url: str = ""
    source_title: str = ""
    token_count: int = 0

    def to_formatted_text(self, include_source: bool = True) -> str:
        """Format context for inclusion in prompt."""
        if include_source and self.source_title:
            return f"[Source: {self.source_title}]\n{self.text}"
        return self.text


@dataclass
class ContextWindow:
    """
    A context window with token budget management.

    Manages what content fits within the LLM's context limit.
    """

    max_tokens: int
    system_tokens: int = 0
    conversation_tokens: int = 0
    context_tokens: int = 0
    reserved_tokens: int = 512  # For response generation

    system_content: str = ""
    conversation_entries: list[MemoryEntry] = field(default_factory=list)
    retrieved_contexts: list[RetrievedContext] = field(default_factory=list)

    @property
    def used_tokens(self) -> int:
        """Total tokens used."""
        return self.system_tokens + self.conversation_tokens + self.context_tokens

    @property
    def available_tokens(self) -> int:
        """Tokens available for additional content."""
        return max(0, self.max_tokens - self.used_tokens - self.reserved_tokens)

    @property
    def is_full(self) -> bool:
        """Check if context window is at capacity."""
        return self.available_tokens <= 0

    def can_fit(self, tokens: int) -> bool:
        """Check if additional tokens can fit."""
        return tokens <= self.available_tokens

    def to_messages(self) -> list[dict]:
        """
        Convert context window to LLM message format.

        Returns:
            List of message dictionaries
        """
        messages = []

        # System message with context
        if self.system_content or self.retrieved_contexts:
            system_parts = []

            if self.system_content:
                system_parts.append(self.system_content)

            if self.retrieved_contexts:
                context_text = "\n\n---\n\n".join(
                    ctx.to_formatted_text() for ctx in self.retrieved_contexts
                )
                system_parts.append(f"\nRelevant context:\n{context_text}")

            messages.append({
                "role": "system",
                "content": "\n\n".join(system_parts),
            })

        # Conversation history
        for entry in self.conversation_entries:
            messages.append(entry.to_message_dict())

        return messages


class ContextManager:
    """
    Manages context for LLM interactions.

    Handles token budgeting, context retrieval integration,
    and conversation history management to maximize relevant
    information within context limits.

    Example:
        >>> manager = ContextManager(
        ...     memory_store=memory_store,
        ...     max_tokens=4096,
        ...     token_counter=llm.count_tokens
        ... )
        >>>
        >>> # Build context for a query
        >>> window = manager.build_context(
        ...     session_id=session_id,
        ...     query="What is this about?",
        ...     retrieved_contexts=search_results
        ... )
        >>>
        >>> # Get formatted messages
        >>> messages = window.to_messages()
    """

    def __init__(
        self,
        memory_store: MemoryStore,
        max_tokens: int = 4096,
        token_counter: Callable[[str], int] | None = None,
        system_prompt: str = "",
        context_budget_ratio: float = 0.4,
        history_budget_ratio: float = 0.3,
    ) -> None:
        """
        Initialize context manager.

        Args:
            memory_store: Memory store for conversation history
            max_tokens: Maximum context window size
            token_counter: Function to count tokens (defaults to word-based estimate)
            system_prompt: Default system prompt
            context_budget_ratio: Portion of budget for retrieved context (0-1)
            history_budget_ratio: Portion of budget for conversation history (0-1)
        """
        self.memory_store = memory_store
        self.max_tokens = max_tokens
        self.token_counter = token_counter or self._estimate_tokens
        self.system_prompt = system_prompt
        self.context_budget_ratio = context_budget_ratio
        self.history_budget_ratio = history_budget_ratio

        logger.info(
            f"ContextManager initialized (max_tokens={max_tokens}, "
            f"context_ratio={context_budget_ratio}, history_ratio={history_budget_ratio})"
        )

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (roughly 4 chars per token)."""
        return len(text) // 4 + 1

    def build_context(
        self,
        session_id: str,
        query: str,
        retrieved_contexts: list[RetrievedContext] | None = None,
        system_prompt: str | None = None,
        include_all_history: bool = False,
    ) -> ContextWindow:
        """
        Build a context window for a query.

        Args:
            session_id: Conversation session ID
            query: Current user query
            retrieved_contexts: Contexts from retrieval
            system_prompt: Override default system prompt
            include_all_history: Include all history (ignore budget)

        Returns:
            ContextWindow ready for LLM
        """
        system = system_prompt or self.system_prompt
        system_tokens = self.token_counter(system) if system else 0

        # Calculate budgets
        reserved = 512  # For response
        available = self.max_tokens - system_tokens - reserved

        context_budget = int(available * self.context_budget_ratio)
        history_budget = int(available * self.history_budget_ratio)
        query_budget = available - context_budget - history_budget

        window = ContextWindow(
            max_tokens=self.max_tokens,
            system_tokens=system_tokens,
            system_content=system,
            reserved_tokens=reserved,
        )

        # Add retrieved contexts (highest priority)
        if retrieved_contexts:
            window.retrieved_contexts = self._fit_contexts(
                retrieved_contexts, context_budget
            )
            window.context_tokens = sum(
                ctx.token_count for ctx in window.retrieved_contexts
            )

        # Add conversation history
        memory = self.memory_store.get_conversation(session_id)
        if memory and not memory.is_empty:
            if include_all_history:
                window.conversation_entries = list(memory.entries)
            else:
                window.conversation_entries = self._fit_history(
                    memory, history_budget
                )

            window.conversation_tokens = sum(
                e.token_count or self.token_counter(e.content)
                for e in window.conversation_entries
            )

        return window

    def _fit_contexts(
        self,
        contexts: list[RetrievedContext],
        budget: int,
    ) -> list[RetrievedContext]:
        """Fit contexts within token budget."""
        fitted = []
        used = 0

        # Contexts should already be sorted by score
        for ctx in contexts:
            if ctx.token_count == 0:
                ctx.token_count = self.token_counter(ctx.text)

            if used + ctx.token_count <= budget:
                fitted.append(ctx)
                used += ctx.token_count
            elif not fitted:
                # Include at least one truncated context
                fitted.append(ctx)
                break

        return fitted

    def _fit_history(
        self,
        memory: ConversationMemory,
        budget: int,
    ) -> list[MemoryEntry]:
        """Fit conversation history within token budget."""
        # Get conversation turns (not context entries)
        turns = [
            e for e in memory.entries
            if e.memory_type in (MemoryType.USER_QUERY, MemoryType.ASSISTANT_RESPONSE)
        ]

        if not turns:
            return []

        # Work backwards from most recent
        fitted = []
        used = 0

        for entry in reversed(turns):
            tokens = entry.token_count or self.token_counter(entry.content)

            if used + tokens <= budget:
                fitted.insert(0, entry)
                used += tokens
            else:
                # If we have nothing, include at least the last exchange
                if not fitted and len(turns) >= 2:
                    # Get last user query and response
                    last_entries = turns[-2:]
                    fitted = last_entries
                break

        return fitted

    def add_query_to_memory(
        self,
        session_id: str,
        query: str,
    ) -> str:
        """
        Add user query to conversation memory.

        Args:
            session_id: Session ID
            query: User query text

        Returns:
            Entry ID
        """
        token_count = self.token_counter(query)
        return self.memory_store.add_user_query(
            session_id, query, token_count
        )

    def add_response_to_memory(
        self,
        session_id: str,
        response: str,
        sources: list[str] | None = None,
    ) -> str:
        """
        Add assistant response to conversation memory.

        Args:
            session_id: Session ID
            response: Assistant response text
            sources: Source URLs used

        Returns:
            Entry ID
        """
        token_count = self.token_counter(response)
        metadata = {"sources": sources} if sources else None
        return self.memory_store.add_assistant_response(
            session_id, response, token_count, metadata
        )

    def summarize_if_needed(
        self,
        session_id: str,
        summarizer: Callable[[str], str] | None = None,
        threshold_tokens: int = 2000,
    ) -> bool:
        """
        Summarize conversation if it exceeds threshold.

        Args:
            session_id: Session to check
            summarizer: Function to generate summary
            threshold_tokens: Token threshold for summarization

        Returns:
            True if summarized
        """
        memory = self.memory_store.get_conversation(session_id)
        if not memory or memory.total_tokens < threshold_tokens:
            return False

        if summarizer is None:
            logger.debug("No summarizer provided, skipping summarization")
            return False

        # Build conversation text
        conversation_text = "\n".join(
            f"{e.role}: {e.content}"
            for e in memory.entries
            if e.memory_type in (MemoryType.USER_QUERY, MemoryType.ASSISTANT_RESPONSE)
        )

        # Generate summary
        summary = summarizer(conversation_text)
        self.memory_store.set_summary(session_id, summary)

        logger.info(f"Summarized conversation {session_id}")
        return True

    def get_conversation_context(
        self,
        session_id: str,
        max_turns: int = 5,
    ) -> str:
        """
        Get recent conversation as formatted context string.

        Args:
            session_id: Session ID
            max_turns: Maximum conversation turns

        Returns:
            Formatted conversation text
        """
        entries = self.memory_store.get_recent_entries(
            session_id,
            limit=max_turns * 2,
            memory_types=[MemoryType.USER_QUERY,
                          MemoryType.ASSISTANT_RESPONSE],
        )

        if not entries:
            return ""

        lines = []
        for entry in entries:
            role = "User" if entry.role == "user" else "Assistant"
            lines.append(f"{role}: {entry.content}")

        return "\n\n".join(lines)

    def create_follow_up_context(
        self,
        session_id: str,
        follow_up_query: str,
        new_contexts: list[RetrievedContext] | None = None,
    ) -> ContextWindow:
        """
        Create context for a follow-up question.

        Includes conversation history and optionally new retrieved context.

        Args:
            session_id: Session ID
            follow_up_query: The follow-up question
            new_contexts: New retrieved contexts

        Returns:
            ContextWindow for the follow-up
        """
        # Use follow-up specific system prompt
        follow_up_system = (
            f"{self.system_prompt}\n\n"
            "This is a follow-up question in an ongoing conversation. "
            "Use the conversation history to understand context and provide "
            "a coherent response that builds on previous exchanges."
        )

        return self.build_context(
            session_id=session_id,
            query=follow_up_query,
            retrieved_contexts=new_contexts,
            system_prompt=follow_up_system,
        )

    def estimate_response_tokens(self, window: ContextWindow) -> int:
        """
        Estimate available tokens for response.

        Args:
            window: Current context window

        Returns:
            Estimated available response tokens
        """
        return max(100, window.available_tokens)
