"""
PiML Cross-Competition Persistent Memory Store.

SQLite-backed storage for competition learnings that persist across runs.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

from .persistent_memory import (
    CrossCompetitionMemory,
    DatasetFingerprint,
    StrategyRecommendation,
    WinningStrategy,
)

# Default database location
DEFAULT_DB_PATH = Path.home() / ".kaggle_agents" / "memory.db"


class PersistentMemoryStore:
    """
    SQLite-backed persistent memory store for cross-competition learning.

    Stores winning strategies, failed approaches, and dataset fingerprints
    to enable transfer learning across competitions.
    """

    def __init__(self, db_path: Optional[Union[Path, str]] = None):
        """
        Initialize the persistent memory store.

        Args:
            db_path: Path to SQLite database. Defaults to ~/.kaggle_agents/memory.db
        """
        if db_path is None:
            db_path = DEFAULT_DB_PATH

        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._init_database()

    def _init_database(self) -> None:
        """Initialize the database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS competition_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    competition_id TEXT UNIQUE NOT NULL,
                    competition_name TEXT NOT NULL,
                    domain TEXT NOT NULL,
                    fingerprint_json TEXT NOT NULL,
                    winning_strategy_json TEXT NOT NULL,
                    failed_approaches_json TEXT NOT NULL,
                    error_patterns_json TEXT NOT NULL,
                    final_score REAL NOT NULL,
                    medal TEXT,
                    iterations_used INTEGER NOT NULL,
                    total_time_seconds REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    notes TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Index for efficient domain-based queries
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_domain ON competition_memory(domain)
            """)

            # Index for efficient fingerprint queries
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp ON competition_memory(timestamp DESC)
            """)

            conn.commit()

    def save(self, memory: CrossCompetitionMemory) -> bool:
        """
        Save a competition memory record.

        Args:
            memory: CrossCompetitionMemory to save

        Returns:
            True if saved successfully
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO competition_memory (
                        competition_id, competition_name, domain,
                        fingerprint_json, winning_strategy_json,
                        failed_approaches_json, error_patterns_json,
                        final_score, medal, iterations_used, total_time_seconds,
                        timestamp, notes, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    """,
                    (
                        memory.competition_id,
                        memory.competition_name,
                        memory.domain,
                        json.dumps(memory.fingerprint.to_dict()),
                        json.dumps(memory.winning_strategy.to_dict()),
                        json.dumps(memory.failed_approaches),
                        json.dumps(memory.error_patterns),
                        memory.final_score,
                        memory.medal,
                        memory.iterations_used,
                        memory.total_time_seconds,
                        memory.timestamp.isoformat(),
                        memory.notes,
                    ),
                )
                conn.commit()
                return True
        except Exception as e:
            print(f"[PersistentMemoryStore] Error saving: {e}")
            return False

    def load(self, competition_id: str) -> Optional[CrossCompetitionMemory]:
        """
        Load a competition memory record by ID.

        Args:
            competition_id: Competition identifier

        Returns:
            CrossCompetitionMemory if found, None otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    "SELECT * FROM competition_memory WHERE competition_id = ?",
                    (competition_id,),
                )
                row = cursor.fetchone()
                if row:
                    return self._row_to_memory(row)
                return None
        except Exception as e:
            print(f"[PersistentMemoryStore] Error loading: {e}")
            return None

    def _row_to_memory(self, row: sqlite3.Row) -> CrossCompetitionMemory:
        """Convert a database row to CrossCompetitionMemory."""
        return CrossCompetitionMemory(
            competition_id=row["competition_id"],
            competition_name=row["competition_name"],
            domain=row["domain"],
            fingerprint=DatasetFingerprint.from_dict(json.loads(row["fingerprint_json"])),
            winning_strategy=WinningStrategy.from_dict(json.loads(row["winning_strategy_json"])),
            failed_approaches=json.loads(row["failed_approaches_json"]),
            error_patterns=json.loads(row["error_patterns_json"]),
            final_score=row["final_score"],
            medal=row["medal"],
            iterations_used=row["iterations_used"],
            total_time_seconds=row["total_time_seconds"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            notes=row["notes"] or "",
        )

    def find_similar(
        self,
        fingerprint: DatasetFingerprint,
        domain: Optional[str] = None,
        top_k: int = 3,
        min_similarity: float = 0.5,
    ) -> list[StrategyRecommendation]:
        """
        Find similar competitions and return strategy recommendations.

        Args:
            fingerprint: Dataset fingerprint to match against
            domain: Optional domain filter (e.g., "tabular", "cv")
            top_k: Maximum number of recommendations to return
            min_similarity: Minimum similarity threshold (0-1)

        Returns:
            List of StrategyRecommendation sorted by similarity (descending)
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row

                # Build query with optional domain filter
                query = "SELECT * FROM competition_memory"
                params: list[Any] = []

                if domain:
                    query += " WHERE domain = ?"
                    params.append(domain)

                query += " ORDER BY timestamp DESC LIMIT 100"

                cursor = conn.execute(query, params)
                rows = cursor.fetchall()

                # Compute similarity scores
                candidates: list[tuple[float, CrossCompetitionMemory]] = []
                for row in rows:
                    memory = self._row_to_memory(row)
                    similarity = fingerprint.similarity_score(memory.fingerprint)
                    if similarity >= min_similarity:
                        candidates.append((similarity, memory))

                # Sort by similarity (descending) and take top_k
                candidates.sort(key=lambda x: x[0], reverse=True)
                top_candidates = candidates[:top_k]

                # Convert to recommendations
                recommendations = []
                for similarity, memory in top_candidates:
                    rec = StrategyRecommendation(
                        source_competition=memory.competition_name,
                        similarity_score=similarity,
                        recommended_strategy=memory.winning_strategy,
                        approaches_to_avoid=memory.failed_approaches,
                        expected_score_range=(memory.final_score * 0.9, memory.final_score * 1.1),
                    )
                    recommendations.append(rec)

                return recommendations

        except Exception as e:
            print(f"[PersistentMemoryStore] Error finding similar: {e}")
            return []

    def find_by_domain(self, domain: str, limit: int = 10) -> list[CrossCompetitionMemory]:
        """
        Find competitions by domain.

        Args:
            domain: Domain to search for (e.g., "tabular", "cv", "nlp")
            limit: Maximum number of results

        Returns:
            List of CrossCompetitionMemory ordered by timestamp (most recent first)
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    """
                    SELECT * FROM competition_memory
                    WHERE domain = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                    """,
                    (domain, limit),
                )
                rows = cursor.fetchall()
                return [self._row_to_memory(row) for row in rows]
        except Exception as e:
            print(f"[PersistentMemoryStore] Error finding by domain: {e}")
            return []

    def get_failed_approaches(self, domain: Optional[str] = None) -> set[str]:
        """
        Get all failed approaches from historical competitions.

        Args:
            domain: Optional domain filter

        Returns:
            Set of approach names that have failed in the past
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                if domain:
                    cursor = conn.execute(
                        "SELECT failed_approaches_json FROM competition_memory WHERE domain = ?",
                        (domain,),
                    )
                else:
                    cursor = conn.execute("SELECT failed_approaches_json FROM competition_memory")

                failed = set()
                for row in cursor.fetchall():
                    approaches = json.loads(row[0])
                    failed.update(approaches)
                return failed
        except Exception as e:
            print(f"[PersistentMemoryStore] Error getting failed approaches: {e}")
            return set()

    def get_successful_strategies(
        self, domain: Optional[str] = None, medal_only: bool = False
    ) -> list[WinningStrategy]:
        """
        Get successful strategies from historical competitions.

        Args:
            domain: Optional domain filter
            medal_only: If True, only return strategies that achieved medals

        Returns:
            List of WinningStrategy sorted by score (best first)
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row

                query = "SELECT winning_strategy_json, final_score, medal FROM competition_memory"
                conditions = []
                params: list[Any] = []

                if domain:
                    conditions.append("domain = ?")
                    params.append(domain)

                if medal_only:
                    conditions.append("medal IS NOT NULL")

                if conditions:
                    query += " WHERE " + " AND ".join(conditions)

                query += " ORDER BY final_score DESC"

                cursor = conn.execute(query, params)
                strategies = []
                for row in cursor.fetchall():
                    strategy = WinningStrategy.from_dict(json.loads(row["winning_strategy_json"]))
                    strategies.append(strategy)
                return strategies
        except Exception as e:
            print(f"[PersistentMemoryStore] Error getting successful strategies: {e}")
            return []

    def count(self, domain: Optional[str] = None) -> int:
        """
        Count the number of stored competition memories.

        Args:
            domain: Optional domain filter

        Returns:
            Number of records
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                if domain:
                    cursor = conn.execute(
                        "SELECT COUNT(*) FROM competition_memory WHERE domain = ?",
                        (domain,),
                    )
                else:
                    cursor = conn.execute("SELECT COUNT(*) FROM competition_memory")
                return cursor.fetchone()[0]
        except Exception as e:
            print(f"[PersistentMemoryStore] Error counting: {e}")
            return 0

    def delete(self, competition_id: str) -> bool:
        """
        Delete a competition memory record.

        Args:
            competition_id: Competition identifier to delete

        Returns:
            True if deleted successfully
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "DELETE FROM competition_memory WHERE competition_id = ?",
                    (competition_id,),
                )
                conn.commit()
                return True
        except Exception as e:
            print(f"[PersistentMemoryStore] Error deleting: {e}")
            return False

    def clear(self) -> bool:
        """
        Clear all competition memory records.

        Returns:
            True if cleared successfully
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM competition_memory")
                conn.commit()
                return True
        except Exception as e:
            print(f"[PersistentMemoryStore] Error clearing: {e}")
            return False


# Singleton instance for global access
_global_store: Optional[PersistentMemoryStore] = None


def get_persistent_store(db_path: Optional[Union[Path, str]] = None) -> PersistentMemoryStore:
    """
    Get the global persistent memory store instance.

    Args:
        db_path: Optional path to database. Only used on first call.

    Returns:
        PersistentMemoryStore instance
    """
    global _global_store
    if _global_store is None:
        _global_store = PersistentMemoryStore(db_path)
    return _global_store
