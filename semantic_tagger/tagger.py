import asyncio
from semantic_tagger.adapters.base import AbstractLLMAdapter
from semantic_tagger.encoder import pack_scored_concepts
from semantic_tagger.types import ContentItem, TextContent, ImageContent, VideoContent, AudioContent, TagResult

Route = tuple[AbstractLLMAdapter, list[str]]


class SemanticTagger:
    def __init__(
        self,
        vocabulary: list[str],
        adapter: AbstractLLMAdapter | None = None,
        routes: list[Route] | None = None,
    ):
        """
        vocabulary: full ordered concept list. Byte position in output vector == term index.
            Do not reorder after vectors have been stored.

        adapter: single adapter for all vocabulary terms. Ignored when routes is set.
            Defaults to OpenAIAdapter() if neither adapter nor routes is provided.

        routes: hybrid split — list of (adapter, vocab_subset) pairs. Each adapter handles
            its assigned subset; results are merged into one vector. Overlapping terms are
            averaged. Preferred: build via Vocabulary.to_tagger() instead of setting manually.
        """
        self.vocabulary = vocabulary
        self.vocab_size = len(vocabulary)

        if routes is not None:
            self._routes: list[Route] = routes
            self._single_adapter: AbstractLLMAdapter | None = None
        else:
            self._routes = []
            if adapter is not None:
                self._single_adapter = adapter
            else:
                from semantic_tagger.adapters.openai_chat import OpenAIAdapter
                self._single_adapter = OpenAIAdapter()

    @staticmethod
    def _content_type(content: ContentItem) -> str:
        if isinstance(content, TextContent): return 'TEXT'
        if isinstance(content, ImageContent): return 'IMAGE'
        if isinstance(content, VideoContent): return 'VIDEO'
        if isinstance(content, AudioContent): return 'AUDIO'
        return 'LINK'

    async def encode(self, content: ContentItem) -> TagResult:
        content_type = self._content_type(content)

        if self._routes:
            scores = await self._encode_routed(content)
        else:
            output = await self._single_adapter.rank(content, self.vocabulary)  # type: ignore[union-attr]
            scores = output.scores

        vector = pack_scored_concepts(scores, self.vocabulary)
        return TagResult(vector=vector, scores=scores, content_type=content_type)  # type: ignore[arg-type]

    async def _encode_routed(self, content: ContentItem) -> dict[str, float]:
        tasks = [adapter.rank(content, sub_vocab) for adapter, sub_vocab in self._routes]
        outputs = await asyncio.gather(*tasks)

        score_accumulator: dict[str, list[float]] = {}
        for output in outputs:
            for term, score in output.scores.items():
                score_accumulator.setdefault(term, []).append(score)

        return {term: sum(vals) / len(vals) for term, vals in score_accumulator.items()}

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
