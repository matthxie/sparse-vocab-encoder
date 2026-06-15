import asyncio
from semantic_tagger.adapters.base import AbstractLLMAdapter
from semantic_tagger.adapters.claude import ClaudeAdapter
from semantic_tagger.encoder import pack_ranked_list
from semantic_tagger.types import ContentItem, TagResult


class SemanticTagger:
    def __init__(
        self,
        vocabulary: list[str],
        adapter: AbstractLLMAdapter | None = None,
    ):
        """
        vocabulary: ordered list of concept strings. Index position is stable —
            do not reorder vocabulary after creating a tagger or stored vectors
            will become misaligned.
        adapter: LLM adapter. If None, instantiates ClaudeAdapter with defaults.
        """
        self.vocabulary = vocabulary
        self.vocab_size = len(vocabulary)
        self.adapter = adapter or ClaudeAdapter()

    async def encode(self, content: ContentItem) -> TagResult:
        ranked_output = await self.adapter.rank(content, self.vocabulary)
        vector = pack_ranked_list(ranked_output.ranked_concepts, self.vocabulary)
        return TagResult(
            vector=vector,
            ranked_concepts=ranked_output.ranked_concepts,
            content_type=ranked_output.content_type,
        )

    async def encode_batch(
        self,
        items: list[ContentItem],
        concurrency: int = 5,
    ) -> list[TagResult]:
        semaphore = asyncio.Semaphore(concurrency)

        async def encode_one(item: ContentItem) -> TagResult:
            async with semaphore:
                return await self.encode(item)

        return await asyncio.gather(*[encode_one(item) for item in items])
