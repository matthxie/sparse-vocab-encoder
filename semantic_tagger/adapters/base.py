from abc import ABC, abstractmethod
from semantic_tagger.types import ContentItem, ScoredOutput


class AbstractLLMAdapter(ABC):
    """
    Contract: take content + vocabulary, return a scored dict of matching concepts.
    The adapter must NOT do any byte packing — that is the encoder's job.
    """

    @abstractmethod
    async def rank(
        self,
        content: ContentItem,
        vocabulary: list[str],
    ) -> ScoredOutput:
        """
        Returns ScoredOutput with scores as a sparse dict: concept → float in (0, 1].
        Absent or irrelevant concepts are omitted entirely (not scored 0.0).
        Only return concepts from vocabulary (do not invent new terms).
        """
