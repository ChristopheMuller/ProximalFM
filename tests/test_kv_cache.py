import torch
import pytest

from proximalfm._model.kv_cache import KVCacheEntry


def test_is_valid():
    entry_empty = KVCacheEntry()
    assert not entry_empty.is_valid()

    entry_partial_k = KVCacheEntry(key=torch.randn(2, 4))
    assert not entry_partial_k.is_valid()

    entry_partial_v = KVCacheEntry(value=torch.randn(2, 4))
    assert not entry_partial_v.is_valid()

    entry_valid = KVCacheEntry(key=torch.randn(2, 4), value=torch.randn(2, 4))
    assert entry_valid.is_valid()


def test_getitem():
    k = torch.arange(10).view(5, 2)
    v = torch.arange(10, 20).view(5, 2)
    entry = KVCacheEntry(key=k, value=v)

    sliced_entry = entry[1:3]
    assert sliced_entry.is_valid()
    assert torch.equal(sliced_entry.key, k[1:3])
    assert torch.equal(sliced_entry.value, v[1:3])

    empty_entry = KVCacheEntry()
    sliced_empty = empty_entry[1:3]
    assert not sliced_empty.is_valid()
    assert sliced_empty.key is None
    assert sliced_empty.value is None


def test_setitem():
    k = torch.zeros(5, 2)
    v = torch.zeros(5, 2)
    entry = KVCacheEntry(key=k, value=v)

    k_other = torch.ones(2, 2)
    v_other = torch.ones(2, 2) * 2
    other_entry = KVCacheEntry(key=k_other, value=v_other)

    entry[1:3] = other_entry

    expected_k = torch.zeros(5, 2)
    expected_k[1:3] = 1
    expected_v = torch.zeros(5, 2)
    expected_v[1:3] = 2

    assert torch.equal(entry.key, expected_k)
    assert torch.equal(entry.value, expected_v)


def test_to():
    k = torch.randn(2, 2, dtype=torch.float32)
    v = torch.randn(2, 2, dtype=torch.float32)
    entry = KVCacheEntry(key=k, value=v)

    new_entry = entry.to(device="cpu", dtype=torch.float16)

    assert new_entry.is_valid()
    assert new_entry.key.dtype == torch.float16
    assert new_entry.value.dtype == torch.float16
    assert new_entry.key.device.type == "cpu"

    empty_entry = KVCacheEntry()
    new_empty = empty_entry.to("cpu")
    assert not new_empty.is_valid()


def test_concat():
    k1 = torch.ones(2, 3)
    v1 = torch.ones(2, 3) * 2
    entry1 = KVCacheEntry(key=k1, value=v1)

    k2 = torch.ones(3, 3) * 3
    v2 = torch.ones(3, 3) * 4
    entry2 = KVCacheEntry(key=k2, value=v2)

    entry3 = KVCacheEntry()  # Invalid entry should be ignored

    concatenated = KVCacheEntry.concat([entry1, entry2, entry3], dim=0)

    assert concatenated.is_valid()
    assert concatenated.key.shape == (5, 3)
    assert concatenated.value.shape == (5, 3)

    expected_k = torch.cat([k1, k2], dim=0)
    expected_v = torch.cat([v1, v2], dim=0)

    assert torch.equal(concatenated.key, expected_k)
    assert torch.equal(concatenated.value, expected_v)

    # Empty concat
    empty_concat = KVCacheEntry.concat([entry3])
    assert not empty_concat.is_valid()
