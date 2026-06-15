import asyncio
import pytest
from semantic_tagger.adapters.base import AbstractLLMAdapter
from semantic_tagger.types import TextContent, ImageContent, RankedOutput
from semantic_tagger.tagger import SemanticTagger


class MockAdapter(AbstractLLMAdapter):
    def __init__(self, ranked: list[str]):
        self._ranked = ranked

    async def rank(self, content, vocabulary) -> RankedOutput:
        ct = 'TEXT' if isinstance(content, TextContent) else 'IMAGE'
        return RankedOutput(ranked_concepts=self._ranked, content_type=ct)


VOCAB = ["astrophotography", "urban", "food", "nature"]


async def test_encode_returns_tag_result():
    adapter = MockAdapter(["astrophotography", "urban"])
    tagger = SemanticTagger(vocabulary=VOCAB, adapter=adapter)
    result = await tagger.encode(TextContent(body="night sky"))
    assert result.content_type == 'TEXT'
    assert result.ranked_concepts == ["astrophotography", "urban"]
    assert len(result.vector) == len(VOCAB)


async def test_encode_correct_vector_bytes():
    # ranked = ["astrophotography", "urban"], vocab has 4 terms
    # rank 0: astro  → weight 1.0 - 0/2 = 1.0 → 0xFF
    # rank 1: urban  → weight 1.0 - 1/2 = 0.5 → 0x80 | round(63.5) = 0xC0
    # food absent → 0x00, nature absent → 0x00
    adapter = MockAdapter(["astrophotography", "urban"])
    tagger = SemanticTagger(vocabulary=VOCAB, adapter=adapter)
    result = await tagger.encode(TextContent(body="test"))
    assert result.vector[0] == 0xFF   # astrophotography
    assert result.vector[1] == 0xC0   # urban
    assert result.vector[2] == 0x00   # food
    assert result.vector[3] == 0x00   # nature


async def test_encode_batch_order_preserved():
    adapter = MockAdapter(["astrophotography"])
    tagger = SemanticTagger(vocabulary=VOCAB, adapter=adapter)
    items = [TextContent(body=str(i)) for i in range(5)]
    results = await tagger.encode_batch(items, concurrency=2)
    assert len(results) == 5
    for r in results:
        assert r.content_type == 'TEXT'


async def test_encode_batch_concurrency_limit():
    concurrent_count = 0
    max_concurrent = 0

    class CountingAdapter(AbstractLLMAdapter):
        async def rank(self, content, vocabulary) -> RankedOutput:
            nonlocal concurrent_count, max_concurrent
            concurrent_count += 1
            max_concurrent = max(max_concurrent, concurrent_count)
            await asyncio.sleep(0.01)
            concurrent_count -= 1
            return RankedOutput(ranked_concepts=[], content_type='TEXT')

    tagger = SemanticTagger(vocabulary=VOCAB, adapter=CountingAdapter())
    items = [TextContent(body=str(i)) for i in range(10)]
    await tagger.encode_batch(items, concurrency=3)
    assert max_concurrent <= 3


async def test_vocab_term_not_in_ranked_is_zero():
    adapter = MockAdapter(["astrophotography"])
    tagger = SemanticTagger(vocabulary=VOCAB, adapter=adapter)
    result = await tagger.encode(TextContent(body="test"))
    # urban, food, nature not in ranked → 0x00
    assert result.vector[1] == 0x00
    assert result.vector[2] == 0x00
    assert result.vector[3] == 0x00


async def test_ranked_term_not_in_vocab_ignored():
    # "galaxy" is not in VOCAB — should not raise
    adapter = MockAdapter(["galaxy", "astrophotography"])
    tagger = SemanticTagger(vocabulary=VOCAB, adapter=adapter)
    result = await tagger.encode(TextContent(body="test"))
    # astrophotography is rank 1 (after galaxy), weight = 1 - 1/2 = 0.5
    assert result.vector[0] == 0x80 | round(0.5 * 127)
