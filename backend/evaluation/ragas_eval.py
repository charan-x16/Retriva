"""Ragas metric computation for logged Retriva queries."""

import logging
import os
import re

from dotenv import load_dotenv

from backend.generation.prompt import (
    RAGAS_JUDGE_SYSTEM_PROMPT,
    build_ragas_metric_prompt,
)

logger = logging.getLogger(__name__)


def compute_ragas(
    question,
    answer,
    contexts,
    ground_truth=None,
    embed_model=None,
) -> dict:
    """Compute Ragas scores for one query, returning None scores on failure."""

    empty_scores = {
        "faithfulness": None,
        "answer_relevancy": None,
        "context_precision": None,
        "context_recall": None,
    }
    load_dotenv(override=True)

    if os.getenv("RAGAS_EVAL_MODE", "ragas").lower() in {
        "openrouter",
        "openrouter_judge",
        "simple",
    }:
        return _fill_missing_scores(
            empty_scores,
            question,
            answer,
            contexts,
            ground_truth or answer,
        )

    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

        evaluator_llm = _build_openrouter_llm()
        evaluator_embeddings = _build_local_embeddings(embed_model)

        ground_truth_value = ground_truth or answer
        data = {
            "user_input": [question],
            "response": [answer],
            "retrieved_contexts": [contexts],
            "reference": [ground_truth_value],
        }
        dataset = Dataset.from_dict(data)
        result = evaluate(
            dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ],
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
            raise_exceptions=False,
            show_progress=False,
        )
        scores = _normalize_scores(result, empty_scores)
        return _fill_missing_scores(scores, question, answer, contexts, ground_truth_value)
    except Exception as exc:
        logger.warning("Ragas evaluation failed: %s", exc)
        return _fill_missing_scores(
            empty_scores,
            question,
            answer,
            contexts,
            ground_truth or answer,
        )


def _build_openrouter_llm():
    """Create the OpenRouter-backed judge LLM used by Ragas."""

    from openai import OpenAI
    from ragas.llms import llm_factory

    api_key = os.getenv("RAGAS_OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is required for Ragas evaluation.")

    model = (
        os.getenv("RAGAS_OPENROUTER_MODEL")
        or os.getenv("OPENROUTER_MODEL")
        or "gpt-4o-mini"
    )
    base_url = os.getenv(
        "RAGAS_OPENROUTER_BASE_URL",
        "https://openrouter.ai/api/v1",
    )

    client = OpenAI(api_key=api_key, base_url=base_url)
    return llm_factory(
        model,
        provider="openai",
        client=client,
        adapter=os.getenv("RAGAS_LLM_ADAPTER", "instructor"),
        temperature=0,
    )


def _build_openrouter_chat_client():
    """Create the plain OpenRouter chat client used for fallback scoring."""

    from openai import OpenAI

    api_key = os.getenv("RAGAS_OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is required for fallback scoring.")

    base_url = os.getenv(
        "RAGAS_OPENROUTER_BASE_URL",
        "https://openrouter.ai/api/v1",
    )
    return OpenAI(api_key=api_key, base_url=base_url)


def _build_local_embeddings(embed_model=None):
    """Create the local embedding model Ragas needs for relevancy scoring."""

    if embed_model is not None:
        return _SentenceTransformerRagasEmbeddings(embed_model)

    from ragas.embeddings import HuggingFaceEmbeddings

    model = (
        os.getenv("RAGAS_EMBEDDING_MODEL")
        or os.getenv("EMBEDDING_MODEL")
        or "BAAI/bge-base-en-v1.5"
    )
    cache_dir = os.getenv("MODEL_CACHE_DIR")
    device = os.getenv("RAGAS_EMBEDDING_DEVICE")
    model_kwargs = {}
    if cache_dir:
        model_kwargs["cache_folder"] = cache_dir

    return HuggingFaceEmbeddings(
        model=model,
        device=device,
        normalize_embeddings=True,
        **model_kwargs,
    )


class _SentenceTransformerRagasEmbeddings:
    """Small Ragas embedding adapter around an already-loaded SentenceTransformer."""

    def __init__(self, model):
        from ragas.run_config import RunConfig

        self.model = model
        self.run_config = RunConfig()

    def set_run_config(self, run_config):
        """Allow Ragas to update retry and timeout settings."""

        self.run_config = run_config

    def embed_query(self, text: str) -> list[float]:
        """Embed one text string."""

        return self.embed_documents([text])[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple text strings."""

        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()

    async def aembed_query(self, text: str) -> list[float]:
        """Asynchronously embed one text string."""

        return self.embed_query(text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """Asynchronously embed multiple text strings."""

        return self.embed_documents(texts)


def _fill_missing_scores(scores, question, answer, contexts, reference) -> dict:
    """Fill failed Ragas metrics with simple OpenRouter numeric judge scores."""

    filled_scores = dict(scores)
    missing_metrics = [
        metric
        for metric, value in filled_scores.items()
        if value is None
    ]
    if not missing_metrics:
        return filled_scores

    try:
        client = _build_openrouter_chat_client()
        model = (
            os.getenv("RAGAS_OPENROUTER_MODEL")
            or os.getenv("OPENROUTER_MODEL")
            or "gpt-4o-mini"
        )
        context_text = "\n\n".join(contexts[:5])
        for metric in missing_metrics:
            filled_scores[metric] = _score_metric_with_llm(
                client,
                model,
                metric,
                question,
                answer,
                context_text,
                reference,
            )
    except Exception as exc:
        logger.warning("Fallback scoring failed: %s", exc)

    return filled_scores


def _score_metric_with_llm(client, model, metric, question, answer, context, reference):
    """Ask the evaluator model for one numeric metric score."""

    prompt = build_ragas_metric_prompt(metric, question, answer, context, reference)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": RAGAS_JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    content = response.choices[0].message.content or ""
    return _to_float_or_none(content)


def _normalize_scores(result, default_scores) -> dict:
    """Convert a Ragas result object into the expected score dictionary."""

    scores = dict(default_scores)

    if hasattr(result, "to_pandas"):
        row = result.to_pandas().iloc[0].to_dict()
    elif isinstance(result, dict):
        row = result
    else:
        row = dict(result)

    for metric in scores:
        value = row.get(metric)
        if isinstance(value, list):
            value = value[0] if value else None
        scores[metric] = _to_float_or_none(value)

    return scores


def _to_float_or_none(value):
    """Convert numeric-like values to float, preserving None."""

    if value is None:
        return None
    try:
        match = re.search(r"(?<![\d.])(?:0(?:\.\d+)?|1(?:\.0+)?)(?![\d.])", str(value))
        if not match:
            return None
        number = float(match.group(0))
        if 0.0 <= number <= 1.0:
            return number
        return None
    except (TypeError, ValueError):
        return None
