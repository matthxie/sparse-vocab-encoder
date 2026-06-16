import json
from dataclasses import dataclass, field
from typing import Literal, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from semantic_tagger.tagger import SemanticTagger
    from semantic_tagger.adapters.base import AbstractLLMAdapter


@dataclass
class VocabTerm:
    """
    A single vocabulary concept.

    route:
        'embedding' — scored via embedding dot-product (fast, good for aesthetic/tone concepts).
        'llm'       — scored via a structured VLM prompt (slower, better for analytical/structural
                      concepts that embeddings struggle with, e.g. 'is_ai_generated', 'contains_pii').

    description: optional hint included in the LLM prompt when route='llm', to give the model
        more precise guidance on what the term means.
    """
    name: str
    route: Literal['embedding', 'llm'] = 'embedding'
    description: Optional[str] = None


class Vocabulary:
    """
    Ordered concept list that drives a SemanticTagger.

    The position of each term is stable and maps 1:1 to byte positions in the output vector.
    Do not reorder terms after vectors have been stored.

    Usage — from a list of strings (all default to embedding route):
        vocab = Vocabulary.from_list(["minimalism", "chaos", "melancholy"])

    Usage — with explicit route annotations:
        vocab = Vocabulary([
            VocabTerm("minimalism"),
            VocabTerm("chaos"),
            VocabTerm("is_ai_generated", route="llm"),
            VocabTerm("contains_pii", route="llm", description="faces, names, ID numbers visible"),
        ])

    Usage — from a JSON config file:
        vocab = Vocabulary.from_json("vocab.json")

        where vocab.json looks like:
        {
            "embedding": ["minimalism", "chaos", "melancholy"],
            "llm": [
                {"name": "is_ai_generated"},
                {"name": "contains_pii", "description": "faces, names, ID numbers visible"}
            ]
        }

    Then create a tagger:
        tagger = vocab.to_tagger(
            embedding_adapter=OpenAIEmbeddingAdapter(),
            llm_adapter=OpenAIAdapter(),
        )
    """

    def __init__(self, terms: list["VocabTerm | str"]):
        self._terms: list[VocabTerm] = [
            VocabTerm(t) if isinstance(t, str) else t
            for t in terms
        ]

    @property
    def terms(self) -> list[VocabTerm]:
        return list(self._terms)

    @property
    def all_terms(self) -> list[str]:
        """Full ordered term list. Use this as the vocabulary for vector storage."""
        return [t.name for t in self._terms]

    @property
    def embedding_terms(self) -> list[str]:
        """Subset of terms routed to the embedding adapter."""
        return [t.name for t in self._terms if t.route == 'embedding']

    @property
    def llm_terms(self) -> list[str]:
        """Subset of terms routed to the LLM adapter."""
        return [t.name for t in self._terms if t.route == 'llm']

    @property
    def llm_terms_with_descriptions(self) -> list[tuple[str, Optional[str]]]:
        """LLM terms with their optional descriptions, for richer prompt construction."""
        return [(t.name, t.description) for t in self._terms if t.route == 'llm']

    @classmethod
    def from_list(
        cls,
        names: list[str],
        route: Literal['embedding', 'llm'] = 'embedding',
    ) -> 'Vocabulary':
        """Create a vocabulary from plain strings, all assigned to the same route."""
        return cls([VocabTerm(name, route=route) for name in names])

    @classmethod
    def from_dict(cls, data: dict) -> 'Vocabulary':
        """
        Load from a dict. Supports both plain strings and dicts with metadata:
        {
            "embedding": ["minimalism", "chaos"],
            "llm": [
                "is_ai_generated",
                {"name": "contains_pii", "description": "faces or IDs visible"}
            ]
        }
        Embedding terms appear first in the vocabulary (preserving insertion order within each group).
        """
        terms: list[VocabTerm] = []
        for entry in data.get('embedding', []):
            if isinstance(entry, str):
                terms.append(VocabTerm(entry, route='embedding'))
            else:
                terms.append(VocabTerm(
                    entry['name'],
                    route='embedding',
                    description=entry.get('description'),
                ))
        for entry in data.get('llm', []):
            if isinstance(entry, str):
                terms.append(VocabTerm(entry, route='llm'))
            else:
                terms.append(VocabTerm(
                    entry['name'],
                    route='llm',
                    description=entry.get('description'),
                ))
        return cls(terms)

    @classmethod
    def from_json(cls, path: str) -> 'Vocabulary':
        """Load from a JSON file using the from_dict format."""
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def to_dict(self) -> dict:
        """Serialize to the from_dict format (round-trip safe)."""
        result: dict = {}
        for route_key in ('embedding', 'llm'):
            group = [
                t for t in self._terms if t.route == route_key
            ]
            if group:
                result[route_key] = [
                    t.name if t.description is None
                    else {'name': t.name, 'description': t.description}
                    for t in group
                ]
        return result

    def to_json(self, path: str) -> None:
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    def to_tagger(
        self,
        embedding_adapter: Optional['AbstractLLMAdapter'] = None,
        llm_adapter: Optional['AbstractLLMAdapter'] = None,
    ) -> 'SemanticTagger':
        """
        Build a SemanticTagger from this vocabulary.

        - If both adapters are provided, terms are routed automatically based on their
          route annotation: embedding_terms → embedding_adapter, llm_terms → llm_adapter.
        - If only one adapter is provided, all terms go to that adapter.
        - If neither is provided, defaults to OpenAIAdapter() for all terms.
        """
        from semantic_tagger.tagger import SemanticTagger

        routes = []
        if embedding_adapter is not None and self.embedding_terms:
            routes.append((embedding_adapter, self.embedding_terms))
        if llm_adapter is not None and self.llm_terms:
            routes.append((llm_adapter, self.llm_terms))

        if routes:
            return SemanticTagger(vocabulary=self.all_terms, routes=routes)

        single = llm_adapter or embedding_adapter
        if single is None:
            from semantic_tagger.adapters.openai_chat import OpenAIAdapter
            single = OpenAIAdapter()
        return SemanticTagger(vocabulary=self.all_terms, adapter=single)

    def __len__(self) -> int:
        return len(self._terms)

    def __repr__(self) -> str:
        n_embed = len(self.embedding_terms)
        n_llm = len(self.llm_terms)
        return f"Vocabulary({len(self._terms)} terms: {n_embed} embedding, {n_llm} llm)"
