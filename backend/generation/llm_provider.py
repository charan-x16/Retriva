"""LLM provider interface plus OpenAI and OpenRouter implementations."""

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


class OpenRouterProvider(LLMProvider):
    """OpenRouter provider using its OpenAI-compatible API."""

    def __init__(self):
        load_dotenv()
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is required for OpenRouter.")

        self.model = os.getenv(
            "OPENROUTER_MODEL",
            "nvidia/nemotron-3-super-120b-a12b:free",
        )
        self.client = OpenAI(
            api_key=api_key,
            base_url=os.getenv(
                "OPENROUTER_BASE_URL",
                "https://openrouter.ai/api/v1",
            ),
        )

    def generate(self, prompt: str) -> str:
        """Generate an answer with the configured OpenRouter model."""

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
    provider = os.getenv("LLM_PROVIDER", "")
    if not provider:
        provider = "openrouter" if os.getenv("OPENROUTER_API_KEY") else "openai"
    provider = provider.lower()

    if provider == "openai":
        return OpenAIProvider()
    if provider == "openrouter":
        return OpenRouterProvider()
    raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")
