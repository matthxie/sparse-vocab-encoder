# semantic-tagger

Encodes multimodal content (text, images, links) into a sparse vocabulary byte vector using a hybrid pipeline: OpenAI embeddings for aesthetic/tone concepts, GPT-4o-mini vision for structural/analytical ones.

Each concept in your vocabulary maps to one byte in the output. Byte format:
- `0x00` — absent
- `0x80–0xFF` — present, with score encoded in the lower 7 bits (`0x80 | round(score * 127)`)

Output is directly compatible with `sparse-vocab-index` for storage in Postgres/Redis.

## Install

```bash
pip install sparse-vocab-encoder
```

## How it works

You define a vocabulary of concept strings, annotated with a route:

- **`embedding`** — scored via `text-embedding-3-small` dot-product. Fast. Works for text and link content. Best for aesthetic/tone concepts (e.g. `minimalism`, `melancholy`, `saturation`).
- **`llm`** — scored via `gpt-4o-mini` using a strict 0–127 tier rubric. Supports all content types including images. Best for analytical concepts that embeddings struggle with (e.g. `is_ai_generated`, `contains_pii`).

The two routes run concurrently and their scores are merged into a single byte vector.

## Usage

```python
from semantic_tagger import Vocabulary, VocabTerm
from semantic_tagger.adapters.openai_embed import OpenAIEmbeddingAdapter
from semantic_tagger.adapters.openai_chat import OpenAIAdapter

vocab = Vocabulary([
    VocabTerm("minimalism"),
    VocabTerm("melancholy"),
    VocabTerm("saturation"),
    VocabTerm("is_ai_generated", route="llm"),
    VocabTerm("contains_pii", route="llm", description="faces or ID numbers visible"),
])

tagger = vocab.to_tagger(
    embedding_adapter=OpenAIEmbeddingAdapter(),
    llm_adapter=OpenAIAdapter(),
)

result = await tagger.encode(TextContent(body="a dimly lit portrait"))
# result.vector  → bytes, length == len(vocab)
# result.scores  → {"minimalism": 0.72, "melancholy": 0.91, ...}
```

Content types:

```python
from semantic_tagger import TextContent, ImageContent, LinkContent

TextContent(body="...")
ImageContent(url="https://...")          # or ImageContent(data=bytes, media_type="image/jpeg")
LinkContent(url="...", title="...", description="...")
```

Batch encoding with concurrency cap:

```python
results = await tagger.encode_batch(items, concurrency=5)
```

## Vocabulary from a config file

```json
{
  "embedding": ["minimalism", "chaos", "melancholy"],
  "llm": [
    "is_ai_generated",
    {"name": "contains_pii", "description": "faces or ID numbers visible"}
  ]
}
```

```python
vocab = Vocabulary.from_json("vocab.json")
```

## LLM scoring tiers (0–127)

The structured prompt instructs the model to follow rigid boundaries:

| Range | Meaning |
|-------|---------|
| 0 | Entirely absent — omitted from output |
| 1–40 | Tangentially present, background noise |
| 41–84 | Explicitly present, balanced with other themes |
| 85–126 | Highly dominant, primary feature |
| 127 | Pure textbook definition |

## Bring your own adapter

```python
from semantic_tagger.adapters.base import AbstractLLMAdapter
from semantic_tagger.types import ScoredOutput

class MyAdapter(AbstractLLMAdapter):
    async def rank(self, content, vocabulary) -> ScoredOutput:
        # return ScoredOutput(scores={"concept": float_0_to_1}, content_type="TEXT")
        ...
```

## Environment

Set `OPENAI_API_KEY` in your environment, or pass `api_key=` to each adapter constructor.
