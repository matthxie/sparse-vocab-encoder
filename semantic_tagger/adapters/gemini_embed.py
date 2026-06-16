import hashlib
import logging
import os
from typing import Optional

from semantic_tagger.adapters.base import AbstractLLMAdapter
from semantic_tagger.types import (
    AudioContent, ContentItem, ImageContent, LinkContent,
    ScoredOutput, TextContent, VideoContent,
)

logger = logging.getLogger(__name__)

DEFAULT_VISION_MODEL = 'gemini-2.5-flash'
DEFAULT_EMBED_MODEL = 'text-embedding-3-small'

_DESCRIBE_PROMPT = (
    'Analyze this content deeply. Describe its physical composition, but focus heavily on its '
    'artistic tone, lighting atmosphere, emotional vibe, textures, implicit narrative, and '
    'cross-modal sensory qualities (e.g., implied temperature, weight, fragrance, loudness). '
    'Be highly descriptive and evocative.'
)


class GeminiVisionAdapter(AbstractLLMAdapter):
    """
    Scores vocabulary concepts for image, video, and audio content.
    Requires: pip install semantic-tagger[gemini,openai]

    Pipeline for media (ImageContent, VideoContent, AudioContent):
        1. Gemini 2.5 Flash generates a detailed evocative text description
        2. OpenAI text-embedding-3-small embeds the description
        3. Dot product against vocab embeddings (also OpenAI) produces sparse scores

    Text and link content skip step 1 and are embedded directly with OpenAI,
    making this adapter a drop-in replacement for OpenAIEmbeddingAdapter when
    media support is needed.

    Both content and vocab embeddings share OpenAI's embedding space, so
    similarity scores are meaningful across all content types.

    Normalization: all terms with positive similarity are stored, scaled so the
    highest-scoring term maps to 1.0. No threshold — filtering belongs on the search side.

    Vocab embeddings are computed once per unique vocabulary and cached in memory.
    """

    def __init__(
        self,
        gemini_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        vision_model: str = DEFAULT_VISION_MODEL,
        embed_model: str = DEFAULT_EMBED_MODEL,
    ):
        self._gemini_key = gemini_api_key or os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
        self._openai_key = openai_api_key or os.environ.get('OPENAI_API_KEY')
        self._vision_model = vision_model
        self._embed_model = embed_model
        self._vocab_cache: dict[str, list[list[float]]] = {}

    def _vocab_hash(self, vocabulary: list[str]) -> str:
        return hashlib.md5('\x00'.join(vocabulary).encode()).hexdigest()

    async def _get_vocab_embeddings(self, vocabulary: list[str], openai_client) -> list[list[float]]:
        key = self._vocab_hash(vocabulary)
        if key not in self._vocab_cache:
            resp = await openai_client.embeddings.create(model=self._embed_model, input=vocabulary)
            ordered = sorted(resp.data, key=lambda x: x.index)
            self._vocab_cache[key] = [item.embedding for item in ordered]
        return self._vocab_cache[key]

    @staticmethod
    def _dot(a: list[float], b: list[float]) -> float:
        return sum(x * y for x, y in zip(a, b))

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

    async def _describe_media(self, content: ContentItem, gemini_client) -> str:
        from google.genai import types as gtypes

        if content.data:  # type: ignore[union-attr]
            part = gtypes.Part.from_bytes(data=content.data, mime_type=content.media_type)  # type: ignore[union-attr]
        elif content.url:  # type: ignore[union-attr]
            part = gtypes.Part.from_uri(uri=content.url, mime_type=content.media_type)  # type: ignore[union-attr]
        else:
            raise ValueError(f'{type(content).__name__} requires data or url')

        resp = await gemini_client.aio.models.generate_content(
            model=self._vision_model,
            contents=[part, _DESCRIBE_PROMPT],
        )
        return resp.text

    async def rank(self, content: ContentItem, vocabulary: list[str]) -> ScoredOutput:
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError(
                'openai package is required for GeminiVisionAdapter. '
                'Install with: pip install semantic-tagger[openai]'
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
            openai_client = AsyncOpenAI(api_key=self._openai_key)

            if isinstance(content, (ImageContent, VideoContent, AudioContent)):
                try:
                    from google import genai as google_genai
                except ImportError:
                    raise ImportError(
                        'google-genai package is required for media content. '
                        'Install with: pip install semantic-tagger[gemini]'
                    )
                gemini_client = google_genai.Client(api_key=self._gemini_key)
                description = await self._describe_media(content, gemini_client)
                logger.info(
                    'GeminiVisionAdapter description (%s, %d chars): %s…',
                    content_type, len(description), description[:120],
                )
                input_text = description
            elif isinstance(content, TextContent):
                input_text = content.body
            else:
                parts = [p for p in [content.title, content.description, content.url] if p]  # type: ignore[union-attr]
                input_text = ' '.join(parts)

            input_resp = await openai_client.embeddings.create(model=self._embed_model, input=[input_text])
            input_vec = input_resp.data[0].embedding
            vocab_vecs = await self._get_vocab_embeddings(vocabulary, openai_client)
            sims = [self._dot(input_vec, v) for v in vocab_vecs]
            scores = self._normalize(sims, vocabulary)

        except Exception as exc:
            logger.exception(
                'GeminiVisionAdapter.rank failed for content_type=%s: %s', content_type, exc
            )
            scores = {}

        return ScoredOutput(scores=scores, content_type=content_type)  # type: ignore[arg-type]
