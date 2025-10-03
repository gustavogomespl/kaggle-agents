"""Tool retrieval system using ChromaDB vector database."""

import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import chromadb
from chromadb.config import Settings

from .embeddings import OpenaiEmbeddings

logger = logging.getLogger(__name__)


class RetrieveTool:
    """Retrieve relevant ML tools using vector similarity search."""

    def __init__(
        self,
        embeddings: OpenaiEmbeddings,
        doc_path: str,
        collection_name: str = "ml_tools",
        persist_directory: Optional[str] = None
    ):
        """Initialize tool retrieval system.

        Args:
            embeddings: Embeddings instance for vectorization
            doc_path: Path to tool documentation directory
            collection_name: Name for the ChromaDB collection
            persist_directory: Directory to persist the database
        """
        self.embeddings = embeddings
        self.doc_path = Path(doc_path)
        self.collection_name = collection_name

        # Set up ChromaDB
        if persist_directory is None:
            persist_directory = str(Path.cwd() / ".chromadb")

        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "ML tool documentation"}
        )

    def create_db_tools(self):
        """Index tool documentation in ChromaDB.

        Reads markdown files from doc_path and creates embeddings.
        """
        if not self.doc_path.exists():
            logger.warning(f"Tool documentation path does not exist: {self.doc_path}")
            return

        # Find all markdown files
        md_files = list(self.doc_path.glob("**/*.md"))

        if not md_files:
            logger.warning(f"No markdown files found in {self.doc_path}")
            return

        logger.info(f"Indexing {len(md_files)} tool documentation files")

        documents = []
        metadatas = []
        ids = []

        for idx, md_file in enumerate(md_files):
            try:
                with open(md_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Extract tool name from filename or content
                tool_name = md_file.stem

                documents.append(content)
                metadatas.append({
                    "tool_name": tool_name,
                    "file_path": str(md_file),
                    "file_name": md_file.name
                })
                ids.append(f"tool_{idx}_{tool_name}")

            except Exception as e:
                logger.error(f"Error reading {md_file}: {e}")

        if documents:
            # Generate embeddings
            try:
                embeddings_list = self.embeddings.embed_documents(documents)

                # Add to ChromaDB
                self.collection.add(
                    documents=documents,
                    embeddings=embeddings_list,
                    metadatas=metadatas,
                    ids=ids
                )

                logger.info(f"Successfully indexed {len(documents)} tools")

            except Exception as e:
                logger.error(f"Error indexing tools: {e}")

    def query_tools(
        self,
        query: str,
        state_name: Optional[str] = None,
        n_results: int = 3
    ) -> str:
        """Query for relevant tools based on natural language query.

        Args:
            query: Natural language query about needed tools
            state_name: Optional state/phase name for context
            n_results: Number of results to return

        Returns:
            Formatted string with tool documentation
        """
        try:
            # Generate query embedding
            query_embedding = self.embeddings.embed_query(query)

            # Query ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )

            if not results['documents'] or not results['documents'][0]:
                logger.warning(f"No tools found for query: {query}")
                return "No relevant tools found."

            # Format results
            formatted_tools = []
            for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
                tool_name = metadata.get('tool_name', 'Unknown')
                formatted_tools.append(f"\n{'='*60}\n")
                formatted_tools.append(f"TOOL: {tool_name}\n")
                formatted_tools.append(f"{'='*60}\n")
                formatted_tools.append(doc)
                formatted_tools.append("\n")

            result_str = "".join(formatted_tools)
            logger.info(f"Retrieved {len(results['documents'][0])} tools for query")

            return result_str

        except Exception as e:
            logger.error(f"Error querying tools: {e}")
            return f"Error retrieving tools: {str(e)}"

    def get_tool_by_name(self, tool_name: str) -> Optional[str]:
        """Get specific tool documentation by name.

        Args:
            tool_name: Name of the tool

        Returns:
            Tool documentation or None if not found
        """
        try:
            results = self.collection.get(
                where={"tool_name": tool_name}
            )

            if results['documents']:
                return results['documents'][0]
            else:
                logger.warning(f"Tool not found: {tool_name}")
                return None

        except Exception as e:
            logger.error(f"Error retrieving tool {tool_name}: {e}")
            return None

    def get_tools_by_names(self, tool_names: List[str]) -> str:
        """Get multiple tools by their names.

        Args:
            tool_names: List of tool names

        Returns:
            Formatted string with tool documentation
        """
        formatted_tools = []

        for tool_name in tool_names:
            doc = self.get_tool_by_name(tool_name)
            if doc:
                formatted_tools.append(f"\n{'='*60}\n")
                formatted_tools.append(f"TOOL: {tool_name}\n")
                formatted_tools.append(f"{'='*60}\n")
                formatted_tools.append(doc)
                formatted_tools.append("\n")

        if formatted_tools:
            return "".join(formatted_tools)
        else:
            return "No tools found with the specified names."

    def list_all_tools(self) -> List[str]:
        """List all available tool names.

        Returns:
            List of tool names
        """
        try:
            results = self.collection.get()
            tool_names = [meta['tool_name'] for meta in results['metadatas']]
            return list(set(tool_names))  # Remove duplicates
        except Exception as e:
            logger.error(f"Error listing tools: {e}")
            return []

    def clear_database(self):
        """Clear all tools from the database."""
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name
            )
            logger.info("Tool database cleared")
        except Exception as e:
            logger.error(f"Error clearing database: {e}")


if __name__ == '__main__':
    # Test tool retrieval
    import sys
    from ..api_handler import load_api_config

    try:
        api_key, base_url = load_api_config()
        embeddings = OpenaiEmbeddings(api_key=api_key, base_url=base_url)

        # Create retrieval tool
        tool_retriever = RetrieveTool(
            embeddings=embeddings,
            doc_path="kaggle_agents/tools/ml_tools_doc",
            collection_name="test_tools"
        )

        # Index tools (if documentation exists)
        tool_retriever.create_db_tools()

        # Test query
        query = "How to handle missing values in data?"
        results = tool_retriever.query_tools(query)
        print("Query Results:")
        print(results)

        # List all tools
        all_tools = tool_retriever.list_all_tools()
        print(f"\nAll available tools: {all_tools}")

    except Exception as e:
        print(f"Error: {e}")
        print("Make sure OPENAI_API_KEY is set and tool documentation exists")
