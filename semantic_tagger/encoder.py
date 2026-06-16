def pack_floats_to_bytes(scores: list[float | None]) -> bytes:
    """
    scores[i] = None  → 0x00 (absent)
    scores[i] = float → 0x80 | round(clamp(val, 0.0, 1.0) * 127)
    """
    out = bytearray(len(scores))
    for i, score in enumerate(scores):
        if score is None:
            out[i] = 0x00
        else:
            clamped = max(0.0, min(1.0, score))
            out[i] = 0x80 | round(clamped * 127)
    return bytes(out)


def unpack_bytes_to_floats(data: bytes) -> list[float | None]:
    """
    0x00          → None (absent)
    byte with MSB → (byte & 0x7F) / 127.0
    """
    result: list[float | None] = []
    for byte in data:
        if byte == 0x00:
            result.append(None)
        else:
            result.append((byte & 0x7F) / 127.0)
    return result


def pack_sparse_map(scores: dict[int, float], vocab_size: int) -> bytes:
    """
    scores: { vocab_index: float_score }
    All unspecified indices → 0x00.
    Output length == vocab_size.
    """
    floats: list[float | None] = [None] * vocab_size
    for idx, score in scores.items():
        floats[idx] = score
    return pack_floats_to_bytes(floats)


def pack_scored_concepts(scores: dict[str, float], vocabulary: list[str]) -> bytes:
    """
    scores: { concept_name: float_score } — sparse, absent concepts omitted.
    Maps concept names to their vocabulary index positions.
    Output length == len(vocabulary).
    """
    floats: list[float | None] = [scores.get(term) for term in vocabulary]
    return pack_floats_to_bytes(floats)


def pack_ranked_list(
    ranked_concepts: list[str],
    vocabulary: list[str],
) -> bytes:
    """
    Rank-decay weight assignment:
        weight(rank) = 1.0 - (rank / len(ranked_concepts))
        rank 0 → 1.0, last rank → approaches 0.0

    Concepts absent from vocabulary are silently ignored.
    Vocabulary terms absent from ranked_concepts → 0x00.
    Output length == len(vocabulary).
    """
    if not ranked_concepts:
        return bytes(len(vocabulary))

    n = len(ranked_concepts)
    concept_to_weight: dict[str, float] = {}
    for rank, concept in enumerate(ranked_concepts):
        concept_to_weight[concept] = 1.0 - (rank / n)

    floats: list[float | None] = []
    for term in vocabulary:
        weight = concept_to_weight.get(term)
        floats.append(weight)

    return pack_floats_to_bytes(floats)
