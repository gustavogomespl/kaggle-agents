"""OpenAI embeddings wrapper for tool retrieval."""

import logging
from typing import List
from openai import OpenAI
import httpx

logger = logging.getLogger(__name__)


class OpenaiEmbeddings:
    """Wrapper for OpenAI embeddings API."""

    def __init__(
        self,
        api_key: str,
        base_url: str = None,
        model: str = "text-embedding-3-small",
        verify_ssl: bool = True
    ):
        """Initialize OpenAI embeddings.

        Args:
            api_key: OpenAI API key
            base_url: Optional base URL for API
            model: Embedding model to use
            verify_ssl: Whether to verify SSL certificates
        """
        self.model = model

        # Configure HTTP client
        http_client = httpx.Client(verify=verify_ssl) if not verify_ssl else None

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=http_client
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors
        """
        try:
            response = self.client.embeddings.create(
                input=texts,
                model=self.model
            )
            embeddings = [item.embedding for item in response.data]
            logger.info(f"Successfully embedded {len(texts)} documents")
            return embeddings
        except Exception as e:
            logger.error(f"Error embedding documents: {e}")
            raise

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text.

        Args:
            text: Query text to embed

        Returns:
            Embedding vector
        """
        try:
            response = self.client.embeddings.create(
                input=[text],
                model=self.model
            )
            embedding = response.data[0].embedding
            logger.info("Successfully embedded query")
            return embedding
        except Exception as e:
            logger.error(f"Error embedding query: {e}")
            raise


if __name__ == '__main__':
    # Test embeddings
    import os

    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        embeddings = OpenaiEmbeddings(api_key=api_key)

        # Test single query
        query_embedding = embeddings.embed_query("How to handle missing values?")
        print(f"Query embedding dimension: {len(query_embedding)}")

        # Test multiple documents
        docs = [
            "Fill missing values with mean",
            "Remove rows with missing data",
            "Use forward fill for time series"
        ]
        doc_embeddings = embeddings.embed_documents(docs)
        print(f"Embedded {len(doc_embeddings)} documents")
    else:
        print("Set OPENAI_API_KEY to test embeddings")
