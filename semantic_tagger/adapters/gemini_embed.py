import asyncio
import hashlib
import logging
import math
import os
from typing import Optional

from semantic_tagger.adapters.base import AbstractLLMAdapter
from semantic_tagger.types import (
    AudioContent, ContentItem, ImageContent, LinkContent,
    ScoredOutput, TextContent, VideoContent,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL = 'gemini-embedding-2'
_VOCAB_CONCURRENCY = 20  # max parallel embed_content calls when embedding vocabulary terms


class GeminiEmbeddingAdapter(AbstractLLMAdapter):
    """
    Scores vocabulary concepts using Google Gemini multimodal embeddings.
    Requires: pip install semantic-tagger[gemini]

    Embeds content and vocabulary terms in Gemini's shared multimodal embedding
    space, then computes cosine similarity to produce sparse vocab scores.

    Supports all content types (TextContent, ImageContent, LinkContent,
    VideoContent, AudioContent). Image/video/audio require a multimodal-capable
    model such as gemini-embedding-2.

    Normalization: all terms with positive cosine similarity are stored, scaled
    so the highest-scoring term maps to 1.0. No threshold is applied here —
    filtering by minimum similarity belongs on the search side.

    Vocab embeddings are computed once per unique vocabulary and cached in memory
    for the lifetime of the adapter instance.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
    ):
        self._api_key = api_key or os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
        self._model = model
        self._vocab_cache: dict[str, list[list[float]]] = {}

    def _vocab_hash(self, vocabulary: list[str]) -> str:
        return hashlib.md5('\x00'.join(vocabulary).encode()).hexdigest()

    async def _get_vocab_embeddings(self, vocabulary: list[str], client) -> list[list[float]]:
        key = self._vocab_hash(vocabulary)
        if key in self._vocab_cache:
            return self._vocab_cache[key]

        sem = asyncio.Semaphore(_VOCAB_CONCURRENCY)

        async def _embed_term(term: str) -> list[float]:
            async with sem:
                resp = await client.aio.models.embed_content(
                    model=self._model,
                    contents=[term],
                )
                return list(resp.embeddings[0].values)

        all_vecs = list(await asyncio.gather(*[_embed_term(t) for t in vocabulary]))
        self._vocab_cache[key] = all_vecs
        return all_vecs

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        mag_a = math.sqrt(sum(x * x for x in a))
        mag_b = math.sqrt(sum(x * x for x in b))
        if mag_a == 0.0 or mag_b == 0.0:
            return 0.0
        return dot / (mag_a * mag_b)

    @staticmethod
    def _normalize(sims: list[float], vocabulary: list[str]) -> dict[str, float]:
        max_sim = max(sims) if sims else 0.0
        if max_sim <= 0.0:
            return {}
        return {
            term: sim / max_sim
            for term, sim in zip(vocabulary, sims)
            if sim > 0.0
        }

    async def _embed_content(self, content: ContentItem, client) -> list[float]:
        from google.genai import types as gtypes

        if isinstance(content, TextContent):
            resp = await client.aio.models.embed_content(
                model=self._model,
                contents=[content.body],
            )
            return resp.embeddings[0].values

        if isinstance(content, LinkContent):
            parts = [p for p in [content.title, content.description, content.url] if p]
            resp = await client.aio.models.embed_content(
                model=self._model,
                contents=[' '.join(parts)],
            )
            return resp.embeddings[0].values

        if isinstance(content, (ImageContent, VideoContent, AudioContent)):
            if content.data:
                part = gtypes.Part.from_bytes(data=content.data, mime_type=content.media_type)
            elif content.url:
                part = gtypes.Part.from_uri(uri=content.url, mime_type=content.media_type)
            else:
                raise ValueError(f'{type(content).__name__} requires data or url')
            resp = await client.aio.models.embed_content(
                model=self._model,
                contents=[part],
            )
            return resp.embeddings[0].values

        raise TypeError(f'Unsupported content type: {type(content).__name__}')

    async def rank(self, content: ContentItem, vocabulary: list[str]) -> ScoredOutput:
        try:
            from google import genai
        except ImportError:
            raise ImportError(
                'google-genai package is required for GeminiEmbeddingAdapter. '
                'Install with: pip install semantic-tagger[gemini]'
            )

        if isinstance(content, TextContent):
            content_type = 'TEXT'
        elif isinstance(content, ImageContent):
            content_type = 'IMAGE'
        elif isinstance(content, VideoContent):
            content_type = 'VIDEO'
        elif isinstance(content, AudioContent):
            content_type = 'AUDIO'
        else:
            content_type = 'LINK'

        try:
            client = genai.Client(api_key=self._api_key)
            vocab_vecs = await self._get_vocab_embeddings(vocabulary, client)
            content_vec = await self._embed_content(content, client)
            sims = [self._cosine(content_vec, v) for v in vocab_vecs]
            scores = self._normalize(sims, vocabulary)
        except Exception as exc:
            logger.exception(
                'GeminiEmbeddingAdapter.rank failed for content_type=%s: %s', content_type, exc
            )
            scores = {}

        return ScoredOutput(scores=scores, content_type=content_type)  # type: ignore[arg-type]
