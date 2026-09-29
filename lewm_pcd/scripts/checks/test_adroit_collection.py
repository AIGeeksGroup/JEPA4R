import pytest

from scripts.data.collect_adroit_full import shard_counts, shard_seed_ranges


def test_shard_counts_distribute_remainder_deterministically():
    assert shard_counts(10, 3) == [4, 3, 3]
    assert shard_counts(40, 40) == [1] * 40


def test_shard_seed_ranges_are_contiguous_and_disjoint():
    counts = [4, 3, 3]
    ranges = shard_seed_ranges(3072, counts)
    assert ranges == [(3072, 3076), (3076, 3079), (3079, 3082)]
    assert all(end > start for start, end in ranges)
    assert sum(end - start for start, end in ranges) == sum(counts)