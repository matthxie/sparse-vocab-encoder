import json
import os
from typing import Optional

from semantic_tagger.adapters.base import AbstractLLMAdapter
from semantic_tagger.types import ContentItem, TextContent, ImageContent, LinkContent, ScoredOutput

DEFAULT_MODEL = 'gpt-4o-mini'

SYSTEM_PROMPT = """You are a semantic tagging system. Given content and a vocabulary list, \
rate each term's presence using this strict tier system (scale 0–127):

- 0: entirely absent, antithetical, or completely unobservable
- 1–40 (Low): tangentially present or exists as background noise
- 41–84 (Mid): explicitly present and clearly recognizable, balanced with other themes
- 85–126 (High): highly dominant, acts as a primary driving feature or focal point
- 127: pure textbook definition of this concept, to the exclusion of almost everything else

Return ONLY a JSON object. Omit terms with score 0.
{"term": score, ...}"""


class OpenAIAdapter(AbstractLLMAdapter):
    """
    Scores vocabulary concepts using a structured OpenAI chat/vision prompt.
    Requires: pip install semantic-tagger[openai]

    Supports all content types including images (via GPT-4o vision).
    Uses the 0–127 tier rubric for consistent, calibration-free scoring.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 512,
    ):
        self._api_key = api_key or os.environ.get('OPENAI_API_KEY')
        self._model = model
        self._max_tokens = max_tokens

    def _build_vocab_prompt(self, vocabulary: list[str]) -> str:
        return f'Vocabulary: {json.dumps(vocabulary)}'

    async def rank(
        self,
        content: ContentItem,
        vocabulary: list[str],
    ) -> ScoredOutput:
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError(
                "openai package is required for OpenAIAdapter. "
                "Install with: pip install semantic-tagger[openai]"
            )

        client = AsyncOpenAI(api_key=self._api_key)
        vocab_prompt = self._build_vocab_prompt(vocabulary)

        if isinstance(content, TextContent):
            content_type = 'TEXT'
            user_content: list = [
                {
                    "type": "text",
                    "text": (
                        f"Content: {content.body}\n\n"
                        f"{vocab_prompt}\n\n"
                        "Rate each vocabulary term's presence in this content."
                    ),
                }
            ]
        elif isinstance(content, ImageContent):
            content_type = 'IMAGE'
            if content.url is not None:
                image_block: dict = {
                    "type": "image_url",
                    "image_url": {"url": content.url},
                }
            else:
                import base64
                encoded = base64.standard_b64encode(content.data).decode('ascii')
                image_block = {
                    "type": "image_url",
                    "image_url": {"url": f"data:{content.media_type};base64,{encoded}"},
                }
            user_content = [
                image_block,
                {
                    "type": "text",
                    "text": (
                        f"{vocab_prompt}\n\n"
                        "Rate each vocabulary term's presence in this image."
                    ),
                },
            ]
        else:
            content_type = 'LINK'
            parts: list[str] = []
            if content.title:
                parts.append(f"Title: {content.title}")
            if content.description:
                parts.append(f"Description: {content.description}")
            parts.append(f"URL: {content.url}")
            user_content = [
                {
                    "type": "text",
                    "text": (
                        "\n".join(parts) + "\n\n"
                        f"{vocab_prompt}\n\n"
                        "Rate each vocabulary term's presence in this link."
                    ),
                }
            ]

        try:
            response = await client.chat.completions.create(
                model=self._model,
                max_tokens=self._max_tokens,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
            )
            raw = response.choices[0].message.content.strip()
            parsed = json.loads(raw)
            if not isinstance(parsed, dict):
                parsed = {}
            vocab_set = set(vocabulary)
            scores: dict[str, float] = {}
            for term, score in parsed.items():
                if term not in vocab_set:
                    continue
                try:
                    int_score = int(score)
                except (TypeError, ValueError):
                    continue
                if int_score <= 0:
                    continue
                scores[term] = min(int_score, 127) / 127.0
        except Exception as exc:
            import logging
            logging.getLogger(__name__).exception(
                "OpenAIAdapter.rank failed for content_type=%s: %s", content_type, exc
            )
            scores = {}

        return ScoredOutput(scores=scores, content_type=content_type)  # type: ignore[arg-type]
