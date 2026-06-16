import json
import pytest
import tempfile
import os
from semantic_tagger.vocab import Vocabulary, VocabTerm


def test_from_list_defaults_to_embedding():
    vocab = Vocabulary.from_list(["a", "b", "c"])
    assert vocab.all_terms == ["a", "b", "c"]
    assert vocab.embedding_terms == ["a", "b", "c"]
    assert vocab.llm_terms == []


def test_from_list_llm_route():
    vocab = Vocabulary.from_list(["a", "b"], route="llm")
    assert vocab.llm_terms == ["a", "b"]
    assert vocab.embedding_terms == []


def test_mixed_routes():
    vocab = Vocabulary([
        VocabTerm("minimalism"),
        VocabTerm("chaos"),
        VocabTerm("is_ai_generated", route="llm"),
    ])
    assert vocab.all_terms == ["minimalism", "chaos", "is_ai_generated"]
    assert vocab.embedding_terms == ["minimalism", "chaos"]
    assert vocab.llm_terms == ["is_ai_generated"]


def test_from_dict_plain_strings():
    vocab = Vocabulary.from_dict({
        "embedding": ["minimalism", "chaos"],
        "llm": ["is_ai_generated"],
    })
    assert vocab.all_terms == ["minimalism", "chaos", "is_ai_generated"]
    assert vocab.embedding_terms == ["minimalism", "chaos"]
    assert vocab.llm_terms == ["is_ai_generated"]


def test_from_dict_with_descriptions():
    vocab = Vocabulary.from_dict({
        "embedding": ["minimalism"],
        "llm": [{"name": "contains_pii", "description": "faces or IDs visible"}],
    })
    pii_term = next(t for t in vocab.terms if t.name == "contains_pii")
    assert pii_term.description == "faces or IDs visible"
    assert pii_term.route == "llm"


def test_plain_strings_in_constructor():
    vocab = Vocabulary(["a", "b", "c"])
    assert vocab.all_terms == ["a", "b", "c"]
    for t in vocab.terms:
        assert t.route == "embedding"


def test_to_dict_round_trip():
    original = Vocabulary([
        VocabTerm("minimalism"),
        VocabTerm("is_ai_generated", route="llm", description="AI artifacts"),
    ])
    data = original.to_dict()
    restored = Vocabulary.from_dict(data)
    assert restored.all_terms == original.all_terms
    assert restored.embedding_terms == original.embedding_terms
    assert restored.llm_terms == original.llm_terms


def test_from_json_round_trip():
    vocab = Vocabulary([
        VocabTerm("chaos"),
        VocabTerm("contains_pii", route="llm", description="faces visible"),
    ])
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        path = f.name
    try:
        vocab.to_json(path)
        restored = Vocabulary.from_json(path)
        assert restored.all_terms == vocab.all_terms
    finally:
        os.unlink(path)


def test_llm_terms_with_descriptions():
    vocab = Vocabulary([
        VocabTerm("minimalism"),
        VocabTerm("is_ai_generated", route="llm"),
        VocabTerm("contains_pii", route="llm", description="faces or IDs"),
    ])
    pairs = vocab.llm_terms_with_descriptions
    assert pairs == [("is_ai_generated", None), ("contains_pii", "faces or IDs")]


def test_len():
    vocab = Vocabulary.from_list(["a", "b", "c"])
    assert len(vocab) == 3


def test_repr():
    vocab = Vocabulary([VocabTerm("a"), VocabTerm("b", route="llm")])
    assert "2 terms" in repr(vocab)
    assert "1 embedding" in repr(vocab)
    assert "1 llm" in repr(vocab)


async def test_to_tagger_single_adapter():
    from semantic_tagger.adapters.base import AbstractLLMAdapter
    from semantic_tagger.types import ScoredOutput, TextContent

    class StubAdapter(AbstractLLMAdapter):
        async def rank(self, content, vocabulary):
            return ScoredOutput(scores={vocabulary[0]: 1.0}, content_type='TEXT')

    vocab = Vocabulary.from_list(["a", "b"])
    tagger = vocab.to_tagger(llm_adapter=StubAdapter())
    result = await tagger.encode(TextContent(body="test"))
    assert result.vector[0] == 0xFF
    assert result.vector[1] == 0x00


async def test_to_tagger_hybrid_routes():
    from semantic_tagger.adapters.base import AbstractLLMAdapter
    from semantic_tagger.types import ScoredOutput, TextContent

    class EmbedStub(AbstractLLMAdapter):
        async def rank(self, content, vocabulary):
            return ScoredOutput(scores={t: 1.0 for t in vocabulary}, content_type='TEXT')

    class LLMStub(AbstractLLMAdapter):
        async def rank(self, content, vocabulary):
            return ScoredOutput(scores={t: 0.5 for t in vocabulary}, content_type='TEXT')

    vocab = Vocabulary([
        VocabTerm("minimalism"),
        VocabTerm("chaos"),
        VocabTerm("is_ai_generated", route="llm"),
    ])
    tagger = vocab.to_tagger(embedding_adapter=EmbedStub(), llm_adapter=LLMStub())

    result = await tagger.encode(TextContent(body="test"))
    assert result.vector[0] == 0xFF   # minimalism → embed route → 1.0
    assert result.vector[1] == 0xFF   # chaos → embed route → 1.0
    assert result.vector[2] == 0x80 | round(0.5 * 127)  # is_ai_generated → llm route → 0.5
