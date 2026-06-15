# PRODUCT SPECIFICATION: `semantic-tagger`

A lightweight Python library for LLM-based semantic vocabulary scoring. Accepts multimodal content (text, images, links), calls an LLM to rank vocabulary terms by relevance, and returns packed byte vectors compatible with the `sparse-vocab-index` byte format. Designed as the insert-time tagging backend for FastAPI services.

---

## 1. DESIGN CONSTRAINTS

- **Language:** Python 3.11+. Typed throughout (PEP 695 style annotations are fine, but `typing` module preferred for 3.11 compatibility).
- **Async-first:** All I/O-bound operations are `async`. Callers must be in an async context (FastAPI, asyncio).
- **Adapter pattern:** The LLM provider is injected. No hard dependency on any single SDK at the package level.
- **Default adapter:** `ClaudeAdapter` ships in the package and depends on `anthropic>=0.25`. It is the only optional dependency. All other code is pure Python stdlib.
- **Published as:** pip package. Name: `semantic-tagger`. Entry: `semantic_tagger/`.
- **Output format:** Packed `bytes` matching the `sparse-vocab-index` 7-bit quantization scheme (Section 2 of that spec), so vectors can be stored directly in Redis/Supabase and loaded into a `VocabIndex` without any conversion.

---

## 2. BYTE FORMAT CONTRACT

The output of `SemanticTagger.encode()` is a `bytes` object of length `vocab_size`, using the same scheme as `sparse-vocab-index`:

```
Bit 7 (MSB): Presence flag — 1 = term scored, 0 = absent (full byte is 0x00)
Bits 6–0:    Quantized score = round(score * 127), clamped to [0, 127]
```

This library is responsible for producing the byte vector. It does **not** handle serialization into the full binary blob (that is `sparse-vocab-index`'s job).

---

## 3. REPOSITORY STRUCTURE

```
semantic-tagger/
├── semantic_tagger/
│   ├── __init__.py              # Public API re-exports
│   ├── types.py                 # ContentItem, TagResult, RankedOutput
│   ├── encoder.py               # Float→byte packing (mirrors VectorEncoder.ts)
│   ├── tagger.py                # SemanticTagger main class
│   └── adapters/
│       ├── __init__.py
│       ├── base.py              # AbstractLLMAdapter ABC
│       └── claude.py            # ClaudeAdapter (default, requires anthropic SDK)
├── tests/
│   ├── test_encoder.py
│   ├── test_tagger.py
│   └── test_claude_adapter.py   # Requires ANTHROPIC_API_KEY
├── pyproject.toml
└── pyproject.toml
```

---

## 4. TYPE DEFINITIONS (`semantic_tagger/types.py`)

```python
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
    data: Optional[bytes] = None        # raw image bytes, any common format
    media_type: str = 'image/jpeg'      # used only when data is set

@dataclass
class LinkContent:
    url: str
    title: Optional[str] = None
    description: Optional[str] = None

ContentItem = TextContent | ImageContent | LinkContent

@dataclass
class RankedOutput:
    """Raw output from the LLM adapter before encoding."""
    ranked_concepts: list[str]          # ordered most-relevant-first
    content_type: ContentType

@dataclass
class TagResult:
    """Final result returned by SemanticTagger.encode()."""
    vector: bytes                       # packed byte vector, length == vocab_size
    ranked_concepts: list[str]          # the raw ranked list (for inspection/logging)
    content_type: ContentType
```

---

## 5. ENCODER (`semantic_tagger/encoder.py`)

Python mirror of `VectorEncoder.ts`. No dependencies beyond stdlib.

```python
def pack_floats_to_bytes(scores: list[float | None]) -> bytes:
    """
    scores[i] = None  → 0x00 (absent)
    scores[i] = float → 0x80 | round(clamp(val, 0.0, 1.0) * 127)
    """

def unpack_bytes_to_floats(data: bytes) -> list[float | None]:
    """
    0x00          → None (absent)
    byte with MSB → (byte & 0x7F) / 127.0
    """

def pack_sparse_map(scores: dict[int, float], vocab_size: int) -> bytes:
    """
    scores: { vocab_index: float_score }
    All unspecified indices → 0x00.
    Output length == vocab_size.
    """

def pack_ranked_list(
    ranked_concepts: list[str],
    vocabulary: list[str],
) -> bytes:
    """
    Rank-decay weight assignment (mirrors RankAdapter.ts):
        weight(rank) = 1.0 - (rank / len(ranked_concepts))
        rank 0 → 1.0, last rank → approaches 0.0

    Concepts absent from vocabulary are silently ignored.
    Vocabulary terms absent from ranked_concepts → 0x00.
    Output length == len(vocabulary).
    """
```

### Weight decay formula details

Given `ranked_concepts = ["a", "b", "c"]` and `vocabulary = ["a", "b", "c", "d"]`:

| rank | concept | weight formula         | weight |
|------|---------|------------------------|--------|
| 0    | a       | 1.0 − (0/3) = 1.0      | 1.0    |
| 1    | b       | 1.0 − (1/3) ≈ 0.667    | 0.667  |
| 2    | c       | 1.0 − (2/3) ≈ 0.333    | 0.333  |
| —    | d       | absent                 | 0x00   |

Edge case: `ranked_concepts` is empty → all bytes are 0x00.

---

## 6. ADAPTER INTERFACE (`semantic_tagger/adapters/base.py`)

```python
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
```

---

## 7. CLAUDE ADAPTER (`semantic_tagger/adapters/claude.py`)

Default adapter. Requires `anthropic` to be installed. If `import anthropic` fails at call time, raises `ImportError` with a message pointing to `pip install semantic-tagger[claude]`.

```python
import os
from anthropic import AsyncAnthropic
from semantic_tagger.adapters.base import AbstractLLMAdapter
from semantic_tagger.types import ContentItem, TextContent, ImageContent, LinkContent, RankedOutput

DEFAULT_MODEL = 'claude-haiku-4-5-20251001'

SYSTEM_PROMPT = """You are a semantic tagging system. Given content and a vocabulary list,
return a JSON array of vocabulary terms ordered from most to least semantically relevant.
Only include terms from the provided vocabulary. Omit terms that are clearly irrelevant.
Return ONLY the JSON array, no explanation."""

class ClaudeAdapter(AbstractLLMAdapter):
    def __init__(
        self,
        api_key: str | None = None,     # defaults to ANTHROPIC_API_KEY env var
        model: str = DEFAULT_MODEL,
        max_tokens: int = 512,
    ):
        ...

    async def rank(
        self,
        content: ContentItem,
        vocabulary: list[str],
    ) -> RankedOutput:
        """
        Builds a message with:
        - System prompt (above)
        - User message: content rendered appropriately (see below) + vocabulary list

        For TEXT: user message contains the body text.
        For IMAGE: user message contains an image block (base64 or URL) + prompt asking for ranking.
        For LINK: user message contains title + description + URL as text.

        Parses the JSON array response. Falls back to empty list if parsing fails
        (never raises on model output errors — caller gets a zero vector instead).
        """
```

### Prompt format

The user message sent to Claude for each content type:

**TEXT:**
```
Content: <body text>

Vocabulary: ["concept1", "concept2", ...]

Rank vocabulary terms by relevance to this content.
```

**IMAGE:**
```
[image block: base64 or URL]

Vocabulary: ["concept1", "concept2", ...]

Rank vocabulary terms by relevance to this image.
```

**LINK:**
```
Title: <title>
Description: <description>
URL: <url>

Vocabulary: ["concept1", "concept2", ...]

Rank vocabulary terms by relevance to this link.
```

Expected model response (parsed as JSON):
```json
["concept3", "concept1", "concept7"]
```

---

## 8. MAIN CLASS (`semantic_tagger/tagger.py`)

```python
from semantic_tagger.adapters.base import AbstractLLMAdapter
from semantic_tagger.adapters.claude import ClaudeAdapter
from semantic_tagger.encoder import pack_ranked_list
from semantic_tagger.types import ContentItem, TagResult

class SemanticTagger:
    def __init__(
        self,
        vocabulary: list[str],
        adapter: AbstractLLMAdapter | None = None,  # defaults to ClaudeAdapter()
    ):
        """
        vocabulary: ordered list of concept strings. Index position is stable —
            do not reorder vocabulary after creating a tagger or stored vectors
            will become misaligned.
        adapter: LLM adapter. If None, instantiates ClaudeAdapter with defaults.
        """
        self.vocabulary = vocabulary
        self.vocab_size = len(vocabulary)
        self.adapter = adapter or ClaudeAdapter()

    async def encode(self, content: ContentItem) -> TagResult:
        """
        1. Call adapter.rank(content, vocabulary) → RankedOutput
        2. Call pack_ranked_list(ranked_concepts, vocabulary) → bytes
        3. Return TagResult(vector=bytes, ranked_concepts=..., content_type=...)
        """

    async def encode_batch(
        self,
        items: list[ContentItem],
        concurrency: int = 5,
    ) -> list[TagResult]:
        """
        Encode multiple items concurrently.
        Uses asyncio.Semaphore(concurrency) to limit parallel LLM calls.
        Results are returned in the same order as input.
        """
```

---

## 9. PUBLIC API (`semantic_tagger/__init__.py`)

```python
from semantic_tagger.tagger import SemanticTagger
from semantic_tagger.types import (
    TextContent,
    ImageContent,
    LinkContent,
    ContentItem,
    TagResult,
    RankedOutput,
)
from semantic_tagger.encoder import (
    pack_floats_to_bytes,
    unpack_bytes_to_floats,
    pack_sparse_map,
    pack_ranked_list,
)
from semantic_tagger.adapters.base import AbstractLLMAdapter

__all__ = [
    'SemanticTagger',
    'TextContent', 'ImageContent', 'LinkContent', 'ContentItem',
    'TagResult', 'RankedOutput',
    'pack_floats_to_bytes', 'unpack_bytes_to_floats', 'pack_sparse_map', 'pack_ranked_list',
    'AbstractLLMAdapter',
]
```

---

## 10. TESTS

### `test_encoder.py`

- `pack_floats_to_bytes([0.0])` → `b'\x80'` (present, score 0)
- `pack_floats_to_bytes([1.0])` → `b'\xff'` (present, score 127)
- `pack_floats_to_bytes([None])` → `b'\x00'` (absent)
- `pack_floats_to_bytes([-0.5])` → `b'\x80'` (clamps to 0.0)
- `pack_floats_to_bytes([1.5])` → `b'\xff'` (clamps to 1.0)
- Round-trip: `unpack_bytes_to_floats(pack_floats_to_bytes([0.5]))[0]` ≈ `0.5` within `1/127` error
- `pack_sparse_map({2: 1.0}, vocab_size=4)` → `b'\x00\x00\xff\x00'`
- `pack_ranked_list(["a", "b"], ["a", "b", "c"])`:
  - index 0 ("a"): weight = 1.0 − 0/2 = 1.0 → `0xFF`
  - index 1 ("b"): weight = 1.0 − 1/2 = 0.5 → `0x80 | round(0.5 * 127)` = `0x80 | 64` = `0xC0`
  - index 2 ("c"): absent → `0x00`
  - result: `b'\xff\xc0\x00'`
- `pack_ranked_list([], ["a", "b"])` → `b'\x00\x00'` (empty ranked list → all absent)

### `test_tagger.py`

Uses a mock adapter that returns a fixed `RankedOutput` without any LLM call.

```python
class MockAdapter(AbstractLLMAdapter):
    def __init__(self, ranked: list[str]):
        self._ranked = ranked
    async def rank(self, content, vocabulary) -> RankedOutput:
        ct = 'TEXT' if isinstance(content, TextContent) else 'IMAGE'
        return RankedOutput(ranked_concepts=self._ranked, content_type=ct)
```

Tests:
- `encode(TextContent(body="..."))` with mock adapter → `TagResult` with correct vector bytes
- `encode_batch([...], concurrency=2)` returns results in input order
- `encode_batch` with 10 items and `concurrency=3` only runs 3 concurrent calls at a time (verified via a counter)
- Vocabulary term not in `ranked_concepts` → corresponding byte is `0x00`
- `ranked_concepts` term not in vocabulary → silently ignored, no error

### `test_claude_adapter.py`

Marked `pytest.mark.integration` — skipped unless `ANTHROPIC_API_KEY` is set and `RUN_INTEGRATION=1`.

- `ClaudeAdapter().rank(TextContent(body="night sky long exposure"), vocabulary=["astrophotography", "urban", "food"])` → `ranked_concepts[0] == "astrophotography"`
- `ClaudeAdapter().rank(LinkContent(url="...", title="Tokyo street photography at night"), vocabulary=["architecture", "street", "nature", "food"])` → `"street"` appears before `"nature"`
- Malformed JSON response from model → returns `RankedOutput(ranked_concepts=[], ...)` without raising

---

## 11. `pyproject.toml`

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "semantic-tagger"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = []           # zero required runtime dependencies

[project.optional-dependencies]
claude = ["anthropic>=0.25"]
dev = ["pytest", "pytest-asyncio", "anthropic>=0.25"]

[tool.pytest.ini_options]
asyncio_mode = "auto"

[tool.hatch.build.targets.wheel]
packages = ["semantic_tagger"]
```

---

## 12. INTEGRATION EXAMPLE (FastAPI insert pipeline)

```python
from fastapi import APIRouter
from semantic_tagger import SemanticTagger, TextContent, ImageContent, LinkContent

VOCABULARY = [
    'astrophotography', 'architecture', 'street', 'nature', 'urban',
    'portrait', 'abstract', 'typography', 'color', 'light', 'shadow',
    # ... up to ~512 terms
]

tagger = SemanticTagger(vocabulary=VOCABULARY)

router = APIRouter()

@router.post("/posts")
async def create_post(data: PostCreate, session: AsyncSession = Depends(get_session)):
    # ... create post in DB as before ...
    post = await session.get(Post, new_post.id)

    # Tag asynchronously — convert content to the right ContentItem
    if post.type == 'TEXT':
        content_item = TextContent(body=post.content.get('body', ''))
    elif post.type == 'IMAGE':
        content_item = ImageContent(url=post.content.get('storage_url'))
    elif post.type == 'LINK':
        content_item = LinkContent(
            url=post.content.get('url', ''),
            title=post.content.get('title'),
            description=post.content.get('description'),
        )

    result = await tagger.encode(content_item)

    # Store packed vector — compatible with sparse-vocab-index VocabIndex.add()
    post.vocab_vector = result.vector      # BYTEA column in Postgres
    post.tags = result.ranked_concepts[:8] # Top 8 as human-readable tags
    session.add(post)
    await session.commit()

    return post_to_read(post)
```

### Vocab vector column migration (Supabase SQL)

```sql
ALTER TABLE post
ADD COLUMN IF NOT EXISTS vocab_vector BYTEA;
```

---

## 13. CROSS-LIBRARY COMPATIBILITY NOTE

The `semantic-tagger` Python library and `sparse-vocab-index` TypeScript library share the same byte encoding contract. A vector produced by `semantic_tagger.encoder.pack_ranked_list()` in Python can be loaded directly into a TypeScript `VocabIndex` via `VocabIndex.add(id, new Uint8Array(vectorBytes))` with no conversion, as long as both sides use the same vocabulary array in the same order.

This is an intentional design choice: the vocabulary is the shared schema. Both libraries treat it as an immutable ordered list. Changing vocabulary order invalidates all stored vectors.
