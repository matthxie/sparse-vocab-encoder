from dataclasses import dataclass, field
from typing import Literal, Optional

ContentType = Literal['TEXT', 'IMAGE', 'LINK']


@dataclass
class TextContent:
    body: str


@dataclass
class ImageContent:
    """Provide exactly one of: url (publicly accessible) or data (raw image bytes)."""
    url: Optional[str] = None
    data: Optional[bytes] = None
    media_type: str = 'image/jpeg'


@dataclass
class LinkContent:
    url: str
    title: Optional[str] = None
    description: Optional[str] = None


ContentItem = TextContent | ImageContent | LinkContent


@dataclass
class ScoredOutput:
    """
    Direct per-concept scores from an adapter.
    scores: sparse dict — absent concepts are simply omitted (not scored 0.0).
    Values are floats in (0.0, 1.0].
    """
    scores: dict[str, float]
    content_type: ContentType


@dataclass
class TagResult:
    """Final result returned by SemanticTagger.encode()."""
    vector: bytes
    scores: dict[str, float]       # the raw scored dict (for inspection/logging)
    content_type: ContentType
