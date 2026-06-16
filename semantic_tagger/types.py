from dataclasses import dataclass, field
from typing import Literal, Optional

ContentType = Literal['TEXT', 'IMAGE', 'LINK', 'VIDEO', 'AUDIO']


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


@dataclass
class VideoContent:
    """Provide one of: data (raw video bytes) or url (publicly accessible)."""
    data: Optional[bytes] = None
    url: Optional[str] = None
    media_type: str = 'video/mp4'


@dataclass
class AudioContent:
    """Provide one of: data (raw audio bytes) or url (publicly accessible)."""
    data: Optional[bytes] = None
    url: Optional[str] = None
    media_type: str = 'audio/mpeg'


ContentItem = TextContent | ImageContent | LinkContent | VideoContent | AudioContent


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
