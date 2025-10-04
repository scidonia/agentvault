"""OpenAI client for embeddings and chat completions."""

import os
import logging
from typing import List, Optional, Dict, Any
import numpy as np

try:
    from openai import OpenAI

    HAS_OPENAI_SUPPORT = True
except ImportError:
    print("⚠️  OpenAI not available. Install with: uv add openai")
    HAS_OPENAI_SUPPORT = False
    OpenAI = None

from .config import OPENAI_API_KEY, OPENAI_EMBEDDING_MODEL, OPENAI_CHAT_MODEL

logger = logging.getLogger(__name__)


class OpenAIClient:
    """OpenAI client for embeddings and chat completions."""

    def __init__(self, api_key: Optional[str] = None):
        if not HAS_OPENAI_SUPPORT:
            raise ImportError(
                "OpenAI library not available. Install with: uv add openai"
            )

        self.api_key = api_key or OPENAI_API_KEY
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable."
            )

        self.client = OpenAI(api_key=self.api_key)
        self.embedding_model = OPENAI_EMBEDDING_MODEL
        self.chat_model = OPENAI_CHAT_MODEL

        logger.info(
            f"OpenAI client initialized with embedding model: {self.embedding_model}"
        )

    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text."""
        try:
            response = self.client.embeddings.create(
                model=self.embedding_model, input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            raise

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts."""
        try:
            response = self.client.embeddings.create(
                model=self.embedding_model, input=texts
            )
            return [data.embedding for data in response.data]
        except Exception as e:
            logger.error(f"Error getting embeddings: {e}")
            raise

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Get chat completion from OpenAI."""
        try:
            response = self.client.chat.completions.create(
                model=model or self.chat_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error getting chat completion: {e}")
            raise

    def chat_completion_stream(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ):
        """Get streaming chat completion from OpenAI."""
        try:
            stream = self.client.chat.completions.create(
                model=model or self.chat_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )

            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error(f"Error getting streaming chat completion: {e}")
            raise


def get_openai_client() -> OpenAIClient:
    """Get a configured OpenAI client instance."""
    return OpenAIClient()
