import pytest
from semantic_tagger.encoder import (
    pack_floats_to_bytes,
    unpack_bytes_to_floats,
    pack_sparse_map,
    pack_ranked_list,
)


def test_pack_zero():
    assert pack_floats_to_bytes([0.0]) == b'\x80'


def test_pack_one():
    assert pack_floats_to_bytes([1.0]) == b'\xff'


def test_pack_none():
    assert pack_floats_to_bytes([None]) == b'\x00'


def test_pack_clamp_negative():
    assert pack_floats_to_bytes([-0.5]) == b'\x80'


def test_pack_clamp_above_one():
    assert pack_floats_to_bytes([1.5]) == b'\xff'


def test_round_trip():
    result = unpack_bytes_to_floats(pack_floats_to_bytes([0.5]))
    assert result[0] is not None
    assert abs(result[0] - 0.5) <= 1 / 127


def test_pack_sparse_map():
    assert pack_sparse_map({2: 1.0}, vocab_size=4) == b'\x00\x00\xff\x00'


def test_pack_ranked_list_basic():
    result = pack_ranked_list(["a", "b"], ["a", "b", "c"])
    assert result == b'\xff\xc0\x00'


def test_pack_ranked_list_empty():
    assert pack_ranked_list([], ["a", "b"]) == b'\x00\x00'
