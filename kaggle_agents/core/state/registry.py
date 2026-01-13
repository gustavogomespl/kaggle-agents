"""
Code Block Registry for Kaggle Agents.

Provides structured tracking of code blocks with:
- Stable IDs (hash-based)
- Type classification (data_loading, preprocessing, model, etc.)
- Dependency tracking
- Lineage tracking for learning across micro-edits

MLE-STAR uses code block tracking to understand which parts of
the solution pipeline are contributing to performance.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal


CodeBlockType = Literal[
    "data_loading",
    "preprocessing",
    "feature_engineering",
    "model",
    "cv",
    "metrics",
    "inference",
    "submission",
    "utility",
]


@dataclass
class CodeBlock:
    """Registered code block with stable ID and lineage tracking.

    Attributes:
        block_id: Hash of code content (stable identifier)
        block_type: Type of code block (data_loading, model, etc.)
        code: The actual code content
        start_line: Starting line number in the script
        end_line: Ending line number in the script
        depends_on: List of block_ids this block depends on
        created_at: ISO format timestamp
        component_name: Which ablation component created this block
        parent_block_id: Block this was refined from (for lineage)
        lineage_id: Stable ID for "family" of related blocks
    """

    block_id: str
    block_type: CodeBlockType
    code: str
    start_line: int
    end_line: int
    depends_on: list[str]
    created_at: str
    component_name: str

    # Lineage tracking (for learning across micro-edits)
    parent_block_id: str | None = None
    lineage_id: str | None = None

    def __repr__(self) -> str:
        """Return string representation for debugging."""
        lines = self.end_line - self.start_line + 1
        return (
            f"CodeBlock({self.block_type}, id={self.block_id[:8]}..., "
            f"lines={self.start_line}-{self.end_line} ({lines} lines))"
        )

    def to_dict(self) -> dict:
        """Serialize for checkpointing."""
        return {
            "block_id": self.block_id,
            "block_type": self.block_type,
            "code": self.code,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "depends_on": self.depends_on,
            "created_at": self.created_at,
            "component_name": self.component_name,
            "parent_block_id": self.parent_block_id,
            "lineage_id": self.lineage_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> CodeBlock:
        """Deserialize from checkpoint."""
        return cls(
            block_id=data["block_id"],
            block_type=data["block_type"],
            code=data["code"],
            start_line=data["start_line"],
            end_line=data["end_line"],
            depends_on=data["depends_on"],
            created_at=data["created_at"],
            component_name=data["component_name"],
            parent_block_id=data.get("parent_block_id"),
            lineage_id=data.get("lineage_id"),
        )


@dataclass
class CodeBlockRegistry:
    """Registry of all code blocks in the solution pipeline.

    Provides:
    - Registration of new blocks with automatic ID generation
    - Lookup by type, component, or lineage
    - Lineage tracking for understanding code evolution
    """

    blocks: dict[str, CodeBlock] = field(default_factory=dict)

    def __repr__(self) -> str:
        """Return string representation for debugging."""
        return f"CodeBlockRegistry({len(self.blocks)} blocks)"

    def register(self, block: CodeBlock) -> str:
        """Register a code block.

        Args:
            block: The CodeBlock to register

        Returns:
            The block_id of the registered block
        """
        self.blocks[block.block_id] = block
        return block.block_id

    def create_and_register(
        self,
        code: str,
        block_type: CodeBlockType,
        component_name: str,
        start_line: int = 0,
        end_line: int = 0,
        depends_on: list[str] | None = None,
        parent_block_id: str | None = None,
    ) -> CodeBlock:
        """Create a new code block and register it.

        Args:
            code: The code content
            block_type: Type of code block
            component_name: Which component created this
            start_line: Starting line number
            end_line: Ending line number
            depends_on: List of dependency block_ids
            parent_block_id: Parent block for lineage tracking

        Returns:
            The created and registered CodeBlock
        """
        # Generate stable block_id from code hash
        block_id = hashlib.md5(code.encode()).hexdigest()[:12]

        # Inherit lineage_id from parent, or create new one
        if parent_block_id and parent_block_id in self.blocks:
            parent = self.blocks[parent_block_id]
            lineage_id = parent.lineage_id or parent.block_id
        else:
            lineage_id = block_id  # New lineage starts with this block

        block = CodeBlock(
            block_id=block_id,
            block_type=block_type,
            code=code,
            start_line=start_line,
            end_line=end_line,
            depends_on=depends_on or [],
            created_at=datetime.now().isoformat(),
            component_name=component_name,
            parent_block_id=parent_block_id,
            lineage_id=lineage_id,
        )

        self.register(block)
        return block

    def get(self, block_id: str) -> CodeBlock | None:
        """Get a block by ID."""
        return self.blocks.get(block_id)

    def get_by_type(self, block_type: CodeBlockType) -> list[CodeBlock]:
        """Get all blocks of a specific type."""
        return [b for b in self.blocks.values() if b.block_type == block_type]

    def get_by_component(self, component_name: str) -> list[CodeBlock]:
        """Get all blocks created by a specific component."""
        return [b for b in self.blocks.values() if b.component_name == component_name]

    def get_lineage(self, lineage_id: str) -> list[CodeBlock]:
        """Get all blocks in a lineage family, sorted by creation time."""
        blocks = [b for b in self.blocks.values() if b.lineage_id == lineage_id]
        return sorted(blocks, key=lambda b: b.created_at)

    def get_latest_by_type(self, block_type: CodeBlockType) -> CodeBlock | None:
        """Get the most recently created block of a specific type."""
        blocks = self.get_by_type(block_type)
        if not blocks:
            return None
        return max(blocks, key=lambda b: b.created_at)

    def get_dependencies(self, block_id: str) -> list[CodeBlock]:
        """Get all blocks that this block depends on."""
        block = self.get(block_id)
        if not block:
            return []
        return [self.blocks[dep_id] for dep_id in block.depends_on if dep_id in self.blocks]

    def get_dependents(self, block_id: str) -> list[CodeBlock]:
        """Get all blocks that depend on this block."""
        return [b for b in self.blocks.values() if block_id in b.depends_on]

    def to_dict(self) -> dict:
        """Serialize for checkpointing."""
        return {"blocks": {k: v.to_dict() for k, v in self.blocks.items()}}

    @classmethod
    def from_dict(cls, data: dict) -> CodeBlockRegistry:
        """Deserialize from checkpoint."""
        registry = cls()
        for k, v in data.get("blocks", {}).items():
            registry.blocks[k] = CodeBlock.from_dict(v)
        return registry
