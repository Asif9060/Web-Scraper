"""
Knowledge graph store.

Manages entity relationships in SQLite with efficient
traversal and query capabilities.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Sequence

from web_intel.config import Settings
from web_intel.storage import Database
from web_intel.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RelationshipRecord:
    """
    Record of a relationship between two entities.

    Represents a directed edge in the knowledge graph.
    """

    id: int | None = None
    subject_entity_id: int | None = None
    subject_name: str = ""
    predicate: str = ""  # Relationship type (e.g., "works_at", "located_in")
    object_entity_id: int | None = None
    object_name: str = ""
    source_page_id: int | None = None
    confidence: float = 1.0
    evidence: str = ""  # Supporting text snippet
    created_at: datetime | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "subject_entity_id": self.subject_entity_id,
            "subject_name": self.subject_name,
            "predicate": self.predicate,
            "object_entity_id": self.object_entity_id,
            "object_name": self.object_name,
            "source_page_id": self.source_page_id,
            "confidence": self.confidence,
            "evidence": self.evidence,
        }

    @classmethod
    def from_row(cls, row: dict) -> "RelationshipRecord":
        """Create from database row."""
        created_at = None
        if row.get("created_at"):
            try:
                created_at = datetime.fromisoformat(row["created_at"])
            except (ValueError, TypeError):
                pass

        return cls(
            id=row.get("id"),
            subject_entity_id=row.get("subject_entity_id"),
            subject_name=row.get("subject_name", ""),
            predicate=row.get("predicate", ""),
            object_entity_id=row.get("object_entity_id"),
            object_name=row.get("object_name", ""),
            source_page_id=row.get("source_page_id"),
            confidence=row.get("confidence", 1.0),
            evidence=row.get("evidence", ""),
            created_at=created_at,
        )


@dataclass
class GraphNode:
    """
    Node in the knowledge graph representing an entity.

    Nodes have types, properties, and connections to other nodes.
    """

    id: int
    name: str
    entity_type: str
    normalized_name: str = ""
    page_ids: list[int] = field(default_factory=list)
    mention_count: int = 1
    properties: dict = field(default_factory=dict)

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GraphNode):
            return False
        return self.id == other.id


@dataclass
class GraphEdge:
    """
    Edge in the knowledge graph representing a relationship.

    Directed edge from source to target with a typed predicate.
    """

    source_id: int
    target_id: int
    predicate: str
    weight: float = 1.0  # Based on confidence and frequency
    page_ids: list[int] = field(default_factory=list)

    def __hash__(self) -> int:
        return hash((self.source_id, self.target_id, self.predicate))


class GraphStore:
    """
    Knowledge graph storage and query engine.

    Stores entity relationships and provides graph traversal,
    path finding, and subgraph extraction capabilities.

    Example:
        >>> store = GraphStore.from_settings(settings)
        >>>
        >>> # Add relationships
        >>> rel = RelationshipRecord(
        ...     subject_name="Alice",
        ...     predicate="works_at",
        ...     object_name="Acme Corp"
        ... )
        >>> store.add_relationship(rel)
        >>>
        >>> # Query relationships
        >>> rels = store.get_relationships_for_entity("Alice")
        >>> for r in rels:
        ...     print(f"{r.subject_name} {r.predicate} {r.object_name}")
        >>>
        >>> # Find paths
        >>> paths = store.find_paths("Alice", "Bob", max_depth=3)
    """

    def __init__(self, database: Database) -> None:
        """
        Initialize graph store.

        Args:
            database: Database instance for persistence
        """
        self.db = database
        self._ensure_tables()
        logger.info("GraphStore initialized")

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        database: Database | None = None,
    ) -> "GraphStore":
        """
        Create GraphStore from settings.

        Args:
            settings: Application settings
            database: Optional pre-configured database

        Returns:
            Configured GraphStore instance
        """
        if database is None:
            database = Database.from_settings(settings)

        return cls(database=database)

    def _ensure_tables(self) -> None:
        """Ensure relationship tables exist."""
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS relationships (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject_entity_id INTEGER REFERENCES entities(id) ON DELETE CASCADE,
                subject_name TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object_entity_id INTEGER REFERENCES entities(id) ON DELETE CASCADE,
                object_name TEXT NOT NULL,
                source_page_id INTEGER REFERENCES pages(id) ON DELETE SET NULL,
                confidence REAL DEFAULT 1.0,
                evidence TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(subject_name, predicate, object_name, source_page_id)
            )
        """)

        # Indexes for efficient traversal
        self.db.execute(
            "CREATE INDEX IF NOT EXISTS idx_rel_subject ON relationships(subject_name)"
        )
        self.db.execute(
            "CREATE INDEX IF NOT EXISTS idx_rel_object ON relationships(object_name)"
        )
        self.db.execute(
            "CREATE INDEX IF NOT EXISTS idx_rel_predicate ON relationships(predicate)"
        )
        self.db.execute(
            "CREATE INDEX IF NOT EXISTS idx_rel_subject_id ON relationships(subject_entity_id)"
        )
        self.db.execute(
            "CREATE INDEX IF NOT EXISTS idx_rel_object_id ON relationships(object_entity_id)"
        )

        self.db._get_connection().commit()

    def add_relationship(self, relationship: RelationshipRecord) -> int:
        """
        Add a relationship to the graph.

        Args:
            relationship: Relationship to add

        Returns:
            Relationship ID
        """
        data = relationship.to_dict()

        # Try to resolve entity IDs if not provided
        if relationship.subject_entity_id is None and relationship.subject_name:
            entity = self._find_entity_by_name(relationship.subject_name)
            if entity:
                data["subject_entity_id"] = entity["id"]

        if relationship.object_entity_id is None and relationship.object_name:
            entity = self._find_entity_by_name(relationship.object_name)
            if entity:
                data["object_entity_id"] = entity["id"]

        # Use INSERT OR REPLACE to handle duplicates
        try:
            rel_id = self.db.insert("relationships", data)
            logger.debug(
                f"Added relationship: {relationship.subject_name} "
                f"-[{relationship.predicate}]-> {relationship.object_name}"
            )
            return rel_id
        except Exception:
            # Update existing if duplicate
            existing = self.db.fetch_one(
                """
                SELECT id FROM relationships
                WHERE subject_name = ? AND predicate = ? AND object_name = ?
                AND (source_page_id = ? OR (source_page_id IS NULL AND ? IS NULL))
                """,
                (
                    relationship.subject_name,
                    relationship.predicate,
                    relationship.object_name,
                    relationship.source_page_id,
                    relationship.source_page_id,
                ),
            )
            if existing:
                return existing["id"]
            raise

    def add_relationships(self, relationships: Sequence[RelationshipRecord]) -> int:
        """
        Add multiple relationships.

        Args:
            relationships: Relationships to add

        Returns:
            Number of relationships added
        """
        count = 0
        for rel in relationships:
            try:
                self.add_relationship(rel)
                count += 1
            except Exception as e:
                logger.warning(f"Failed to add relationship: {e}")

        logger.debug(f"Added {count} relationships")
        return count

    def _find_entity_by_name(self, name: str) -> dict | None:
        """Find entity by normalized name."""
        normalized = name.lower().strip()
        return self.db.fetch_one(
            "SELECT * FROM entities WHERE normalized_name = ? LIMIT 1",
            (normalized,),
        )

    def get_relationship(self, rel_id: int) -> RelationshipRecord | None:
        """Get relationship by ID."""
        row = self.db.fetch_one(
            "SELECT * FROM relationships WHERE id = ?",
            (rel_id,),
        )
        return RelationshipRecord.from_row(row) if row else None

    def get_relationships_for_entity(
        self,
        entity_name: str,
        direction: str = "both",
        predicate: str | None = None,
    ) -> list[RelationshipRecord]:
        """
        Get all relationships involving an entity.

        Args:
            entity_name: Entity name to search
            direction: "outgoing", "incoming", or "both"
            predicate: Optional filter by relationship type

        Returns:
            List of relationships
        """
        normalized = entity_name.lower().strip()
        relationships = []

        if direction in ("outgoing", "both"):
            query = """
                SELECT * FROM relationships
                WHERE LOWER(subject_name) = ?
            """
            params = [normalized]

            if predicate:
                query += " AND predicate = ?"
                params.append(predicate)

            rows = self.db.fetch_all(query, tuple(params))
            relationships.extend(
                [RelationshipRecord.from_row(r) for r in rows])

        if direction in ("incoming", "both"):
            query = """
                SELECT * FROM relationships
                WHERE LOWER(object_name) = ?
            """
            params = [normalized]

            if predicate:
                query += " AND predicate = ?"
                params.append(predicate)

            rows = self.db.fetch_all(query, tuple(params))
            relationships.extend(
                [RelationshipRecord.from_row(r) for r in rows])

        return relationships

    def get_relationships_by_predicate(
        self,
        predicate: str,
        limit: int = 100,
    ) -> list[RelationshipRecord]:
        """Get all relationships of a given type."""
        rows = self.db.fetch_all(
            "SELECT * FROM relationships WHERE predicate = ? LIMIT ?",
            (predicate, limit),
        )
        return [RelationshipRecord.from_row(r) for r in rows]

    def get_relationships_by_page(
        self,
        page_id: int,
    ) -> list[RelationshipRecord]:
        """Get all relationships extracted from a specific page."""
        rows = self.db.fetch_all(
            "SELECT * FROM relationships WHERE source_page_id = ?",
            (page_id,),
        )
        return [RelationshipRecord.from_row(r) for r in rows]

    def get_related_entities(
        self,
        entity_name: str,
        max_depth: int = 1,
    ) -> list[tuple[str, str, int]]:
        """
        Get entities related to a given entity.

        Args:
            entity_name: Starting entity
            max_depth: How many hops to traverse

        Returns:
            List of (entity_name, relationship_path, depth) tuples
        """
        visited = {entity_name.lower().strip()}
        results = []
        current_level = [entity_name]

        for depth in range(1, max_depth + 1):
            next_level = []

            for entity in current_level:
                rels = self.get_relationships_for_entity(entity)

                for rel in rels:
                    # Outgoing: entity -> other
                    if rel.subject_name.lower() == entity.lower():
                        other = rel.object_name
                        path = rel.predicate
                    else:
                        other = rel.subject_name
                        path = f"inv_{rel.predicate}"

                    normalized_other = other.lower().strip()
                    if normalized_other not in visited:
                        visited.add(normalized_other)
                        results.append((other, path, depth))
                        next_level.append(other)

            current_level = next_level

        return results

    def find_paths(
        self,
        source_entity: str,
        target_entity: str,
        max_depth: int = 4,
    ) -> list[list[RelationshipRecord]]:
        """
        Find paths between two entities.

        Uses BFS to find all shortest paths up to max_depth.

        Args:
            source_entity: Starting entity name
            target_entity: Target entity name
            max_depth: Maximum path length

        Returns:
            List of paths (each path is a list of relationships)
        """
        source_norm = source_entity.lower().strip()
        target_norm = target_entity.lower().strip()

        if source_norm == target_norm:
            return [[]]

        # BFS with path tracking
        queue: list[tuple[str, list[RelationshipRecord]]] = [(source_norm, [])]
        visited = {source_norm}
        found_paths = []

        while queue:
            current, path = queue.pop(0)

            if len(path) >= max_depth:
                continue

            rels = self.get_relationships_for_entity(current)

            for rel in rels:
                # Determine the neighbor
                if rel.subject_name.lower().strip() == current:
                    neighbor = rel.object_name.lower().strip()
                else:
                    neighbor = rel.subject_name.lower().strip()

                new_path = path + [rel]

                if neighbor == target_norm:
                    found_paths.append(new_path)
                elif neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, new_path))

        return found_paths

    def get_entity_node(self, entity_name: str) -> GraphNode | None:
        """
        Get a graph node for an entity.

        Args:
            entity_name: Entity name

        Returns:
            GraphNode or None
        """
        normalized = entity_name.lower().strip()

        # Get entity info
        entity = self.db.fetch_one(
            """
            SELECT e.*, GROUP_CONCAT(DISTINCT e.page_id) as page_ids,
                   SUM(e.mentions) as total_mentions
            FROM entities e
            WHERE e.normalized_name = ?
            GROUP BY e.normalized_name
            """,
            (normalized,),
        )

        if not entity:
            return None

        page_ids = []
        if entity.get("page_ids"):
            page_ids = [int(p)
                        for p in str(entity["page_ids"]).split(",") if p]

        return GraphNode(
            id=entity["id"],
            name=entity["name"],
            entity_type=entity["entity_type"],
            normalized_name=entity["normalized_name"],
            page_ids=page_ids,
            mention_count=entity.get("total_mentions", 1),
        )

    def get_subgraph(
        self,
        center_entity: str,
        radius: int = 2,
        max_nodes: int = 50,
    ) -> tuple[list[GraphNode], list[GraphEdge]]:
        """
        Extract a subgraph centered on an entity.

        Args:
            center_entity: Center entity name
            radius: Number of hops from center
            max_nodes: Maximum nodes to include

        Returns:
            Tuple of (nodes, edges)
        """
        nodes_dict: dict[str, GraphNode] = {}
        edges_dict: dict[tuple[str, str, str], GraphEdge] = {}

        # Start with center node
        center_node = self.get_entity_node(center_entity)
        if center_node:
            nodes_dict[center_entity.lower()] = center_node

        # BFS to collect nodes and edges
        current_level = [center_entity]
        visited = {center_entity.lower().strip()}

        for _ in range(radius):
            if len(nodes_dict) >= max_nodes:
                break

            next_level = []

            for entity in current_level:
                rels = self.get_relationships_for_entity(entity)

                for rel in rels:
                    if len(nodes_dict) >= max_nodes:
                        break

                    # Get the other entity
                    if rel.subject_name.lower().strip() == entity.lower().strip():
                        other = rel.object_name
                        source = entity
                        target = other
                    else:
                        other = rel.subject_name
                        source = other
                        target = entity

                    # Add edge
                    edge_key = (source.lower(), target.lower(), rel.predicate)
                    if edge_key not in edges_dict:
                        source_node = self.get_entity_node(source)
                        target_node = self.get_entity_node(target)

                        edges_dict[edge_key] = GraphEdge(
                            source_id=source_node.id if source_node else 0,
                            target_id=target_node.id if target_node else 0,
                            predicate=rel.predicate,
                            weight=rel.confidence,
                            page_ids=[
                                rel.source_page_id] if rel.source_page_id else [],
                        )

                    # Add neighbor node
                    other_norm = other.lower().strip()
                    if other_norm not in visited:
                        visited.add(other_norm)
                        node = self.get_entity_node(other)
                        if node:
                            nodes_dict[other_norm] = node
                        next_level.append(other)

            current_level = next_level

        return list(nodes_dict.values()), list(edges_dict.values())

    def get_predicates(self) -> list[tuple[str, int]]:
        """
        Get all relationship types with counts.

        Returns:
            List of (predicate, count) tuples
        """
        rows = self.db.fetch_all(
            """
            SELECT predicate, COUNT(*) as count
            FROM relationships
            GROUP BY predicate
            ORDER BY count DESC
            """
        )
        return [(row["predicate"], row["count"]) for row in rows]

    def get_entity_types(self) -> list[tuple[str, int]]:
        """
        Get all entity types with counts.

        Returns:
            List of (entity_type, count) tuples
        """
        rows = self.db.fetch_all(
            """
            SELECT entity_type, COUNT(DISTINCT normalized_name) as count
            FROM entities
            GROUP BY entity_type
            ORDER BY count DESC
            """
        )
        return [(row["entity_type"], row["count"]) for row in rows]

    def search_entities(
        self,
        query: str,
        entity_type: str | None = None,
        limit: int = 20,
    ) -> list[GraphNode]:
        """
        Search entities by name pattern.

        Args:
            query: Search query (partial match)
            entity_type: Optional filter by type
            limit: Maximum results

        Returns:
            List of matching entities as GraphNodes
        """
        pattern = f"%{query}%"

        if entity_type:
            rows = self.db.fetch_all(
                """
                SELECT id, name, entity_type, normalized_name,
                       GROUP_CONCAT(DISTINCT page_id) as page_ids,
                       SUM(mentions) as total_mentions
                FROM entities
                WHERE name LIKE ? AND entity_type = ?
                GROUP BY normalized_name
                ORDER BY total_mentions DESC
                LIMIT ?
                """,
                (pattern, entity_type, limit),
            )
        else:
            rows = self.db.fetch_all(
                """
                SELECT id, name, entity_type, normalized_name,
                       GROUP_CONCAT(DISTINCT page_id) as page_ids,
                       SUM(mentions) as total_mentions
                FROM entities
                WHERE name LIKE ?
                GROUP BY normalized_name
                ORDER BY total_mentions DESC
                LIMIT ?
                """,
                (pattern, limit),
            )

        nodes = []
        for row in rows:
            page_ids = []
            if row.get("page_ids"):
                page_ids = [int(p)
                            for p in str(row["page_ids"]).split(",") if p]

            nodes.append(
                GraphNode(
                    id=row["id"],
                    name=row["name"],
                    entity_type=row["entity_type"],
                    normalized_name=row["normalized_name"],
                    page_ids=page_ids,
                    mention_count=row.get("total_mentions", 1),
                )
            )

        return nodes

    def delete_by_page(self, page_id: int) -> int:
        """Delete all relationships from a page."""
        affected = self.db.delete(
            "relationships",
            "source_page_id = ?",
            (page_id,),
        )
        logger.debug(f"Deleted {affected} relationships for page {page_id}")
        return affected

    def get_stats(self) -> dict:
        """Get graph statistics."""
        rel_count = self.db.fetch_one(
            "SELECT COUNT(*) as count FROM relationships"
        )
        entity_count = self.db.fetch_one(
            "SELECT COUNT(DISTINCT normalized_name) as count FROM entities"
        )
        predicate_count = self.db.fetch_one(
            "SELECT COUNT(DISTINCT predicate) as count FROM relationships"
        )

        return {
            "relationship_count": rel_count["count"] if rel_count else 0,
            "entity_count": entity_count["count"] if entity_count else 0,
            "predicate_count": predicate_count["count"] if predicate_count else 0,
        }

    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"GraphStore(entities={stats['entity_count']}, "
            f"relationships={stats['relationship_count']})"
        )
