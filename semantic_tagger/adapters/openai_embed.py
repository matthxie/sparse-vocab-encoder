import hashlib
import os
from typing import Optional

from semantic_tagger.adapters.base import AbstractLLMAdapter
from semantic_tagger.types import ContentItem, TextContent, ImageContent, LinkContent, ScoredOutput


class OpenAIEmbeddingAdapter(AbstractLLMAdapter):
    """
    Scores vocabulary concepts against text/link content using OpenAI embedding dot-products.
    Requires: pip install semantic-tagger[openai]

    Does NOT support ImageContent — return it to OpenAIAdapter for vision-based scoring.

    Vocab embeddings are computed once per unique vocabulary and cached in memory for the
    lifetime of the adapter instance. On a cold start (first encode call) one extra API
    request is made to embed the full vocabulary; subsequent calls only embed the input.

    Normalization:
        Raw cosine similarities (dot products on L2-normalized vectors) are clipped at
        `threshold` (default 0.15). Terms below the threshold are marked absent (0x00).
        Terms at or above the threshold are linearly scaled so the highest-scoring term
        in the response maps to 1.0. This gives relative importance across the vocabulary.

        For absolute scores (recommended in production), supply calibration data via
        the `calibration` parameter: {term: (floor_sim, ceiling_sim)} derived from the
        10th/90th percentile of observed scores for each term over a representative corpus.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = 'text-embedding-3-small',
        threshold: float = 0.15,
        calibration: Optional[dict[str, tuple[float, float]]] = None,
    ):
        self._api_key = api_key or os.environ.get('OPENAI_API_KEY')
        self._model = model
        self._threshold = threshold
        self._calibration = calibration or {}
        # Cache: vocab_hash → list of per-term embedding vectors
        self._vocab_cache: dict[str, list[list[float]]] = {}

    def _vocab_hash(self, vocabulary: list[str]) -> str:
        return hashlib.md5('\x00'.join(vocabulary).encode()).hexdigest()

    async def _get_vocab_embeddings(
        self, vocabulary: list[str], client
    ) -> list[list[float]]:
        key = self._vocab_hash(vocabulary)
        if key not in self._vocab_cache:
            resp = await client.embeddings.create(model=self._model, input=vocabulary)
            ordered = sorted(resp.data, key=lambda x: x.index)
            self._vocab_cache[key] = [item.embedding for item in ordered]
        return self._vocab_cache[key]

    @staticmethod
    def _dot(a: list[float], b: list[float]) -> float:
        # OpenAI embeddings are L2-normalized, so dot product == cosine similarity
        return sum(x * y for x, y in zip(a, b))

    def _normalize(self, sims: list[float], vocabulary: list[str]) -> dict[str, float]:
        if self._calibration:
            scores: dict[str, float] = {}
            for term, sim in zip(vocabulary, sims):
                if term in self._calibration:
                    floor, ceiling = self._calibration[term]
                    if sim <= floor:
                        continue
                    scores[term] = min((sim - floor) / (ceiling - floor), 1.0)
                else:
                    if sim > self._threshold:
                        scores[term] = float(min(sim, 1.0))
            return scores

        max_sim = max(sims) if sims else 0.0
        if max_sim <= self._threshold:
            return {}
        scores = {}
        for term, sim in zip(vocabulary, sims):
            if sim > self._threshold:
                scores[term] = (sim - self._threshold) / (max_sim - self._threshold)
        return scores

    async def rank(
        self,
        content: ContentItem,
        vocabulary: list[str],
    ) -> ScoredOutput:
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError(
                "openai package is required for OpenAIEmbeddingAdapter. "
                "Install with: pip install semantic-tagger[openai]"
            )

        if isinstance(content, ImageContent):
            # Embedding route does not support images — caller should route images to OpenAIAdapter
            return ScoredOutput(scores={}, content_type='IMAGE')

        if isinstance(content, TextContent):
            input_text = content.body
            content_type: str = 'TEXT'
        else:
            parts: list[str] = []
            if content.title:
                parts.append(content.title)
            if content.description:
                parts.append(content.description)
            parts.append(content.url)
            input_text = ' '.join(parts)
            content_type = 'LINK'

        client = AsyncOpenAI(api_key=self._api_key)

        input_resp = await client.embeddings.create(model=self._model, input=[input_text])
        input_vec = input_resp.data[0].embedding
        vocab_vecs = await self._get_vocab_embeddings(vocabulary, client)

        sims = [self._dot(input_vec, vocab_vec) for vocab_vec in vocab_vecs]
        scores = self._normalize(sims, vocabulary)

        return ScoredOutput(scores=scores, content_type=content_type)  # type: ignore[arg-type]
