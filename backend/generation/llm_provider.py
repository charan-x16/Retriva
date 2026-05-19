"""LLM provider interface and OpenAI implementation."""

import os
from abc import ABC, abstractmethod

from dotenv import load_dotenv
from openai import OpenAI


class LLMProvider(ABC):
    """Abstract interface for text generation providers."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate text for a prompt."""


class OpenAIProvider(LLMProvider):
    """OpenAI chat-completions provider."""

    def __init__(self):
        load_dotenv()
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.client = OpenAI()

    def generate(self, prompt: str) -> str:
        """Generate an answer with the configured OpenAI model."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "Answer only from the supplied context and keep citations inline.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        return response.choices[0].message.content or ""


def get_llm_provider() -> LLMProvider:
    """Create the configured LLM provider."""

    load_dotenv()
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    if provider == "openai":
        return OpenAIProvider()
    raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")

