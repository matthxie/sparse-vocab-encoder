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

DEFAULT_MODEL = 'gemini-embedding-exp-03-07'
_BATCH_SIZE = 100  # items per embed_content call


class GeminiEmbeddingAdapter(AbstractLLMAdapter):
    """
    Scores vocabulary concepts using Google Gemini multimodal embeddings.
    Requires: pip install semantic-tagger[gemini]

    Embeds content and vocabulary terms in Gemini's shared multimodal embedding
    space, then computes cosine similarity to produce sparse vocab scores.

    Supports all content types (TextContent, ImageContent, LinkContent,
    VideoContent, AudioContent). Image/video/audio require a multimodal-capable
    model such as gemini-embedding-exp-03-07.

    Vocab embeddings are computed once per unique vocabulary and cached in memory
    for the lifetime of the adapter instance.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        threshold: float = 0.15,
        task_type: str = 'SEMANTIC_SIMILARITY',
        calibration: Optional[dict[str, tuple[float, float]]] = None,
    ):
        self._api_key = api_key or os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
        self._model = model
        self._threshold = threshold
        self._task_type = task_type
        self._calibration = calibration or {}
        self._vocab_cache: dict[str, list[list[float]]] = {}

    def _vocab_hash(self, vocabulary: list[str]) -> str:
        return hashlib.md5('\x00'.join(vocabulary).encode()).hexdigest()

    async def _get_vocab_embeddings(self, vocabulary: list[str], client) -> list[list[float]]:
        key = self._vocab_hash(vocabulary)
        if key in self._vocab_cache:
            return self._vocab_cache[key]

        from google.genai import types as gtypes
        all_vecs: list[list[float]] = []
        for i in range(0, len(vocabulary), _BATCH_SIZE):
            chunk = vocabulary[i:i + _BATCH_SIZE]
            resp = await client.aio.models.embed_content(
                model=self._model,
                contents=chunk,
                config=gtypes.EmbedContentConfig(task_type=self._task_type),
            )
            all_vecs.extend(e.values for e in resp.embeddings)

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
        return {
            term: (sim - self._threshold) / (max_sim - self._threshold)
            for term, sim in zip(vocabulary, sims)
            if sim > self._threshold
        }

    async def _embed_content(self, content: ContentItem, client) -> list[float]:
        from google.genai import types as gtypes

        if isinstance(content, TextContent):
            resp = await client.aio.models.embed_content(
                model=self._model,
                contents=[content.body],
                config=gtypes.EmbedContentConfig(task_type=self._task_type),
            )
            return resp.embeddings[0].values

        if isinstance(content, LinkContent):
            parts = [p for p in [content.title, content.description, content.url] if p]
            resp = await client.aio.models.embed_content(
                model=self._model,
                contents=[' '.join(parts)],
                config=gtypes.EmbedContentConfig(task_type=self._task_type),
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
