from abc import ABC, abstractmethod
from semantic_tagger.types import ContentItem, RankedOutput


class AbstractLLMAdapter(ABC):
    """
    Contract: take content + vocabulary, return a ranked list of matching concepts.
    The adapter must NOT do any byte packing — that is the encoder's job.
    """

    @abstractmethod
    async def rank(
        self,
        content: ContentItem,
        vocabulary: list[str],
    ) -> RankedOutput:
        """
        Returns RankedOutput with ranked_concepts ordered most-relevant-first.
        Only return concepts from vocabulary (do not invent new terms).
        May return a subset if most terms are clearly irrelevant.
        """
