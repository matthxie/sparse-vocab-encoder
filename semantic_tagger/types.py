from dataclasses import dataclass
from typing import Literal, Optional

ContentType = Literal['TEXT', 'IMAGE', 'LINK']


@dataclass
class TextContent:
    body: str


@dataclass
class ImageContent:
    """Provide exactly one of: url (publicly accessible) or data (base64-encoded bytes)."""
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
class RankedOutput:
    """Raw output from the LLM adapter before encoding."""
    ranked_concepts: list[str]
    content_type: ContentType


@dataclass
class TagResult:
    """Final result returned by SemanticTagger.encode()."""
    vector: bytes
    ranked_concepts: list[str]
    content_type: ContentType
