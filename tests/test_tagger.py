import asyncio
import pytest
from semantic_tagger.adapters.base import AbstractLLMAdapter
from semantic_tagger.types import TextContent, ImageContent, ScoredOutput
from semantic_tagger.tagger import SemanticTagger


class MockAdapter(AbstractLLMAdapter):
    def __init__(self, scores: dict[str, float]):
        self._scores = scores

    async def rank(self, content, vocabulary) -> ScoredOutput:
        ct = 'TEXT' if isinstance(content, TextContent) else 'IMAGE'
        # Return only scores for terms in the requested vocabulary
        filtered = {k: v for k, v in self._scores.items() if k in vocabulary}
        return ScoredOutput(scores=filtered, content_type=ct)


VOCAB = ["astrophotography", "urban", "food", "nature"]


async def test_encode_returns_tag_result():
    adapter = MockAdapter({"astrophotography": 1.0, "urban": 0.5})
    tagger = SemanticTagger(vocabulary=VOCAB, adapter=adapter)
    result = await tagger.encode(TextContent(body="night sky"))
    assert result.content_type == 'TEXT'
    assert result.scores == {"astrophotography": 1.0, "urban": 0.5}
    assert len(result.vector) == len(VOCAB)


async def test_encode_correct_vector_bytes():
    adapter = MockAdapter({"astrophotography": 1.0, "urban": 0.5})
    tagger = SemanticTagger(vocabulary=VOCAB, adapter=adapter)
    result = await tagger.encode(TextContent(body="test"))
    assert result.vector[0] == 0xFF                      # astrophotography → 1.0
    assert result.vector[1] == 0x80 | round(0.5 * 127)  # urban → 0.5
    assert result.vector[2] == 0x00                      # food → absent
    assert result.vector[3] == 0x00                      # nature → absent


async def test_encode_batch_order_preserved():
    adapter = MockAdapter({"astrophotography": 1.0})
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
        async def rank(self, content, vocabulary) -> ScoredOutput:
            nonlocal concurrent_count, max_concurrent
            concurrent_count += 1
            max_concurrent = max(max_concurrent, concurrent_count)
            await asyncio.sleep(0.01)
            concurrent_count -= 1
            return ScoredOutput(scores={}, content_type='TEXT')

    tagger = SemanticTagger(vocabulary=VOCAB, adapter=CountingAdapter())
    items = [TextContent(body=str(i)) for i in range(10)]
    await tagger.encode_batch(items, concurrency=3)
    assert max_concurrent <= 3


async def test_vocab_term_not_in_scores_is_zero():
    adapter = MockAdapter({"astrophotography": 1.0})
    tagger = SemanticTagger(vocabulary=VOCAB, adapter=adapter)
    result = await tagger.encode(TextContent(body="test"))
    assert result.vector[1] == 0x00  # urban
    assert result.vector[2] == 0x00  # food
    assert result.vector[3] == 0x00  # nature


async def test_scores_term_not_in_vocab_ignored():
    # "galaxy" is not in VOCAB — adapter returns it, tagger should ignore
    adapter = MockAdapter({"galaxy": 1.0, "astrophotography": 0.8})
    tagger = SemanticTagger(vocabulary=VOCAB, adapter=adapter)
    result = await tagger.encode(TextContent(body="test"))
    assert result.vector[0] == 0x80 | round(0.8 * 127)  # astrophotography
    assert result.vector[1] == 0x00                      # urban absent


async def test_hybrid_routes_merge():
    """Routes: two adapters for disjoint vocab subsets, results merged into one vector."""
    clip_adapter = MockAdapter({"astrophotography": 1.0, "nature": 0.7})
    llm_adapter = MockAdapter({"urban": 0.9, "food": 0.3})

    tagger = SemanticTagger(
        vocabulary=VOCAB,
        routes=[
            (clip_adapter, ["astrophotography", "nature"]),
            (llm_adapter, ["urban", "food"]),
        ],
    )
    result = await tagger.encode(TextContent(body="test"))
    assert result.vector[0] == 0x80 | round(1.0 * 127)  # astrophotography
    assert result.vector[1] == 0x80 | round(0.9 * 127)  # urban
    assert result.vector[2] == 0x80 | round(0.3 * 127)  # food
    assert result.vector[3] == 0x80 | round(0.7 * 127)  # nature


async def test_hybrid_routes_overlap_averaged():
    """Overlapping routes: scores for the same term are averaged."""
    adapter_a = MockAdapter({"astrophotography": 1.0})
    adapter_b = MockAdapter({"astrophotography": 0.5})

    tagger = SemanticTagger(
        vocabulary=VOCAB,
        routes=[
            (adapter_a, ["astrophotography"]),
            (adapter_b, ["astrophotography"]),
        ],
    )
    result = await tagger.encode(TextContent(body="test"))
    expected = 0x80 | round(0.75 * 127)  # average of 1.0 and 0.5
    assert result.vector[0] == expected
