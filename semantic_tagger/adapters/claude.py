import json
import os
from typing import Optional

from semantic_tagger.adapters.base import AbstractLLMAdapter
from semantic_tagger.types import ContentItem, TextContent, ImageContent, LinkContent, RankedOutput

DEFAULT_MODEL = 'claude-haiku-4-5-20251001'

SYSTEM_PROMPT = """You are a semantic tagging system. Given content and a vocabulary list,
return a JSON array of vocabulary terms ordered from most to least semantically relevant.
Only include terms from the provided vocabulary. Omit terms that are clearly irrelevant.
Return ONLY the JSON array, no explanation."""


class ClaudeAdapter(AbstractLLMAdapter):
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 512,
    ):
        self._api_key = api_key or os.environ.get('ANTHROPIC_API_KEY')
        self._model = model
        self._max_tokens = max_tokens

    async def rank(
        self,
        content: ContentItem,
        vocabulary: list[str],
    ) -> RankedOutput:
        try:
            from anthropic import AsyncAnthropic
        except ImportError:
            raise ImportError(
                "anthropic package is required for ClaudeAdapter. "
                "Install it with: pip install semantic-tagger[claude]"
            )

        client = AsyncAnthropic(api_key=self._api_key)
        vocab_json = json.dumps(vocabulary)

        if isinstance(content, TextContent):
            content_type: str = 'TEXT'
            user_message: list = [
                {
                    "type": "text",
                    "text": (
                        f"Content: {content.body}\n\n"
                        f"Vocabulary: {vocab_json}\n\n"
                        "Rank vocabulary terms by relevance to this content."
                    ),
                }
            ]
        elif isinstance(content, ImageContent):
            content_type = 'IMAGE'
            if content.url is not None:
                image_block = {
                    "type": "image",
                    "source": {"type": "url", "url": content.url},
                }
            else:
                import base64
                encoded = base64.standard_b64encode(content.data).decode('ascii')
                image_block = {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": content.media_type,
                        "data": encoded,
                    },
                }
            user_message = [
                image_block,
                {
                    "type": "text",
                    "text": (
                        f"Vocabulary: {vocab_json}\n\n"
                        "Rank vocabulary terms by relevance to this image."
                    ),
                },
            ]
        else:
            content_type = 'LINK'
            parts = []
            if content.title:
                parts.append(f"Title: {content.title}")
            if content.description:
                parts.append(f"Description: {content.description}")
            parts.append(f"URL: {content.url}")
            user_message = [
                {
                    "type": "text",
                    "text": (
                        "\n".join(parts) + "\n\n"
                        f"Vocabulary: {vocab_json}\n\n"
                        "Rank vocabulary terms by relevance to this link."
                    ),
                }
            ]

        try:
            response = await client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )
            raw = response.content[0].text.strip()
            ranked = json.loads(raw)
            if not isinstance(ranked, list):
                ranked = []
            ranked = [term for term in ranked if isinstance(term, str)]
        except Exception:
            ranked = []

        return RankedOutput(ranked_concepts=ranked, content_type=content_type)  # type: ignore[arg-type]
