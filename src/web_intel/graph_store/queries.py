"""
Graph query utilities.

Provides structured queries and result types for
knowledge graph operations.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from web_intel.graph_store.store import (
    GraphStore,
    GraphNode,
    GraphEdge,
    RelationshipRecord,
)
from web_intel.utils.logging import get_logger

logger = get_logger(__name__)


class QueryType(str, Enum):
    """Types of graph queries."""

    NEIGHBORS = "neighbors"  # Direct connections
    PATH = "path"  # Path between entities
    SUBGRAPH = "subgraph"  # Neighborhood extraction
    PATTERN = "pattern"  # Pattern matching
    AGGREGATE = "aggregate"  # Aggregation queries


@dataclass
class PathResult:
    """
    Result of a path query between entities.

    Contains the found path and metadata about the search.
    """

    source: str
    target: str
    found: bool
    path: list[RelationshipRecord] = field(default_factory=list)
    path_length: int = 0
    entities_traversed: list[str] = field(default_factory=list)

    @property
    def path_description(self) -> str:
        """Human-readable path description."""
        if not self.found:
            return f"No path found from '{self.source}' to '{self.target}'"

        parts = [self.source]
        for rel in self.path:
            if rel.subject_name.lower() == parts[-1].lower():
                parts.append(f"-[{rel.predicate}]->")
                parts.append(rel.object_name)
            else:
                parts.append(f"<-[{rel.predicate}]-")
                parts.append(rel.subject_name)

        return " ".join(parts)


@dataclass
class SubgraphResult:
    """
    Result of a subgraph extraction query.

    Contains nodes and edges within the specified radius.
    """

    center: str
    radius: int
    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def edge_count(self) -> int:
        return len(self.edges)

    def get_node_by_name(self, name: str) -> GraphNode | None:
        """Find a node by name."""
        normalized = name.lower().strip()
        for node in self.nodes:
            if node.normalized_name == normalized:
                return node
        return None

    def get_edges_for_node(self, node_id: int) -> list[GraphEdge]:
        """Get all edges connected to a node."""
        return [
            e
            for e in self.edges
            if e.source_id == node_id or e.target_id == node_id
        ]

    def to_adjacency_list(self) -> dict[int, list[tuple[int, str]]]:
        """Convert to adjacency list representation."""
        adj: dict[int, list[tuple[int, str]]] = {n.id: [] for n in self.nodes}
        for edge in self.edges:
            adj[edge.source_id].append((edge.target_id, edge.predicate))
        return adj


@dataclass
class PatternMatch:
    """A single match result for a pattern query."""

    bindings: dict[str, GraphNode]  # Variable -> Node mapping
    relationships: list[RelationshipRecord]
    score: float = 1.0


class GraphQuery:
    """
    Query builder for knowledge graph operations.

    Provides a fluent interface for constructing and executing
    graph queries.

    Example:
        >>> query = GraphQuery(store)
        >>> results = (
        ...     query
        ...     .start_from("Alice")
        ...     .follow("works_at")
        ...     .follow("located_in")
        ...     .execute()
        ... )
        >>>
        >>> # Pattern matching
        >>> pattern = query.pattern(
        ...     "?person works_at ?company",
        ...     "?company located_in ?city"
        ... )
        >>> matches = pattern.find_all(limit=10)
    """

    def __init__(self, store: GraphStore) -> None:
        """
        Initialize query builder.

        Args:
            store: GraphStore instance
        """
        self.store = store
        self._start_entity: str | None = None
        self._predicates: list[str | None] = []
        self._filters: list[Callable[[RelationshipRecord], bool]] = []
        self._direction: str = "outgoing"
        self._max_depth: int = 3

    def start_from(self, entity: str) -> "GraphQuery":
        """Set the starting entity for traversal."""
        self._start_entity = entity
        return self

    def follow(self, predicate: str | None = None) -> "GraphQuery":
        """
        Add a traversal step.

        Args:
            predicate: Relationship type to follow (None = any)
        """
        self._predicates.append(predicate)
        return self

    def direction(self, direction: str) -> "GraphQuery":
        """Set traversal direction ('outgoing', 'incoming', 'both')."""
        self._direction = direction
        return self

    def max_depth(self, depth: int) -> "GraphQuery":
        """Set maximum traversal depth."""
        self._max_depth = depth
        return self

    def filter(self, predicate: Callable[[RelationshipRecord], bool]) -> "GraphQuery":
        """Add a filter predicate."""
        self._filters.append(predicate)
        return self

    def where_confidence(self, min_confidence: float) -> "GraphQuery":
        """Filter by minimum confidence score."""
        self._filters.append(lambda r: r.confidence >= min_confidence)
        return self

    def execute(self) -> list[RelationshipRecord]:
        """
        Execute the traversal query.

        Returns:
            List of relationships matching the query
        """
        if not self._start_entity:
            raise ValueError("No start entity specified")

        results = []
        current_entities = [self._start_entity]

        for step, predicate in enumerate(self._predicates):
            if step >= self._max_depth:
                break

            next_entities = []

            for entity in current_entities:
                rels = self.store.get_relationships_for_entity(
                    entity,
                    direction=self._direction,
                    predicate=predicate,
                )

                # Apply filters
                for rel in rels:
                    if all(f(rel) for f in self._filters):
                        results.append(rel)

                        # Determine next entity to traverse
                        if self._direction == "outgoing":
                            next_entities.append(rel.object_name)
                        elif self._direction == "incoming":
                            next_entities.append(rel.subject_name)
                        else:
                            if rel.subject_name.lower() == entity.lower():
                                next_entities.append(rel.object_name)
                            else:
                                next_entities.append(rel.subject_name)

            current_entities = next_entities

        return results

    def find_path_to(self, target: str) -> PathResult:
        """
        Find path from start entity to target.

        Args:
            target: Target entity name

        Returns:
            PathResult with path information
        """
        if not self._start_entity:
            raise ValueError("No start entity specified")

        paths = self.store.find_paths(
            self._start_entity,
            target,
            max_depth=self._max_depth,
        )

        if not paths:
            return PathResult(
                source=self._start_entity,
                target=target,
                found=False,
            )

        # Return shortest path
        shortest = min(paths, key=len)

        # Build entity list
        entities = [self._start_entity]
        for rel in shortest:
            if rel.subject_name.lower() == entities[-1].lower():
                entities.append(rel.object_name)
            else:
                entities.append(rel.subject_name)

        return PathResult(
            source=self._start_entity,
            target=target,
            found=True,
            path=shortest,
            path_length=len(shortest),
            entities_traversed=entities,
        )

    def get_subgraph(self, radius: int = 2, max_nodes: int = 50) -> SubgraphResult:
        """
        Extract subgraph around start entity.

        Args:
            radius: Number of hops from center
            max_nodes: Maximum nodes to include

        Returns:
            SubgraphResult with nodes and edges
        """
        if not self._start_entity:
            raise ValueError("No start entity specified")

        nodes, edges = self.store.get_subgraph(
            self._start_entity,
            radius=radius,
            max_nodes=max_nodes,
        )

        return SubgraphResult(
            center=self._start_entity,
            radius=radius,
            nodes=nodes,
            edges=edges,
        )

    def pattern(self, *patterns: str) -> "PatternQuery":
        """
        Create a pattern query.

        Patterns use ?variable syntax for placeholders.
        Example: "?person works_at ?company"

        Args:
            patterns: Pattern strings to match

        Returns:
            PatternQuery for execution
        """
        return PatternQuery(self.store, list(patterns))


class PatternQuery:
    """
    Pattern matching query for knowledge graph.

    Finds subgraphs matching specified patterns with variable bindings.

    Example:
        >>> pq = PatternQuery(store, [
        ...     "?person works_at ?company",
        ...     "?company located_in ?city"
        ... ])
        >>> matches = pq.find_all(limit=10)
        >>> for match in matches:
        ...     print(f"{match.bindings['?person'].name} works in {match.bindings['?city'].name}")
    """

    def __init__(self, store: GraphStore, patterns: list[str]) -> None:
        """
        Initialize pattern query.

        Args:
            store: GraphStore instance
            patterns: Pattern strings
        """
        self.store = store
        self.patterns = patterns
        self._parsed_patterns = [self._parse_pattern(p) for p in patterns]

    def _parse_pattern(self, pattern: str) -> tuple[str, str, str]:
        """
        Parse a pattern string into (subject, predicate, object).

        Pattern format: "?var1 predicate ?var2" or "EntityName predicate ?var"
        """
        parts = pattern.strip().split()
        if len(parts) != 3:
            raise ValueError(f"Invalid pattern format: {pattern}")

        return parts[0], parts[1], parts[2]

    def _is_variable(self, term: str) -> bool:
        """Check if a term is a variable."""
        return term.startswith("?")

    def find_all(self, limit: int = 100) -> list[PatternMatch]:
        """
        Find all matches for the pattern.

        Args:
            limit: Maximum matches to return

        Returns:
            List of PatternMatch results
        """
        if not self._parsed_patterns:
            return []

        # Start with first pattern
        first_pattern = self._parsed_patterns[0]
        initial_matches = self._match_single_pattern(first_pattern, {})

        # Join with subsequent patterns
        matches = initial_matches
        for pattern in self._parsed_patterns[1:]:
            matches = self._join_patterns(matches, pattern)
            if not matches:
                break

        # Limit results
        return matches[:limit]

    def _match_single_pattern(
        self,
        pattern: tuple[str, str, str],
        bindings: dict[str, GraphNode],
    ) -> list[PatternMatch]:
        """Match a single pattern with optional existing bindings."""
        subj, pred, obj = pattern
        matches = []

        # Get relationships matching the predicate
        rels = self.store.get_relationships_by_predicate(pred, limit=1000)

        for rel in rels:
            new_bindings = bindings.copy()
            valid = True

            # Check/bind subject
            if self._is_variable(subj):
                if subj in new_bindings:
                    if new_bindings[subj].name.lower() != rel.subject_name.lower():
                        valid = False
                else:
                    node = self.store.get_entity_node(rel.subject_name)
                    if node:
                        new_bindings[subj] = node
                    else:
                        # Create minimal node
                        new_bindings[subj] = GraphNode(
                            id=rel.subject_entity_id or 0,
                            name=rel.subject_name,
                            entity_type="unknown",
                        )
            else:
                if subj.lower() != rel.subject_name.lower():
                    valid = False

            # Check/bind object
            if valid:
                if self._is_variable(obj):
                    if obj in new_bindings:
                        if new_bindings[obj].name.lower() != rel.object_name.lower():
                            valid = False
                    else:
                        node = self.store.get_entity_node(rel.object_name)
                        if node:
                            new_bindings[obj] = node
                        else:
                            new_bindings[obj] = GraphNode(
                                id=rel.object_entity_id or 0,
                                name=rel.object_name,
                                entity_type="unknown",
                            )
                else:
                    if obj.lower() != rel.object_name.lower():
                        valid = False

            if valid:
                matches.append(
                    PatternMatch(
                        bindings=new_bindings,
                        relationships=[rel],
                        score=rel.confidence,
                    )
                )

        return matches

    def _join_patterns(
        self,
        existing_matches: list[PatternMatch],
        pattern: tuple[str, str, str],
    ) -> list[PatternMatch]:
        """Join existing matches with a new pattern."""
        joined = []

        for match in existing_matches:
            new_matches = self._match_single_pattern(pattern, match.bindings)

            for new_match in new_matches:
                joined.append(
                    PatternMatch(
                        bindings=new_match.bindings,
                        relationships=match.relationships + new_match.relationships,
                        score=match.score * new_match.score,
                    )
                )

        return joined

    def find_first(self) -> PatternMatch | None:
        """Find the first match."""
        matches = self.find_all(limit=1)
        return matches[0] if matches else None

    def count(self) -> int:
        """Count total matches."""
        return len(self.find_all(limit=10000))
