"""Attention profiling grid: defaults unchanged, short chunks added on request."""

import unittest

from vidur.profiling.attention.grid import (
    chunked_prefill_points,
    prefill_chunk_sizes,
    seq_lengths,
)


def original_points(max_seq_len):
    """Vidur's grid before the options: chunks of 64 tokens or more, prior
    context at multiples of the chunk size."""
    points = []
    for chunk in prefill_chunk_sizes(max_seq_len):
        points.extend((chunk, i * chunk) for i in range(max_seq_len // chunk))
    return points


class ProfilingGridTests(unittest.TestCase):
    def test_default_grid_is_unchanged(self):
        for max_seq_len in (2048 + 64, 4096 + 64):
            self.assertEqual(chunked_prefill_points(max_seq_len), original_points(max_seq_len))
            self.assertEqual(min(prefill_chunk_sizes(max_seq_len)), 64)

    def test_short_chunks_on_the_context_grid(self):
        n = 4096 + 64
        points = set(chunked_prefill_points(n, short_chunks=True, context_grid=True))
        contexts = seq_lengths(n)
        for chunk in (1, 22, 31, 56):
            near = max(c for c in (1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48, 56) if c <= chunk)
            for kv in contexts:
                if near + kv <= n:
                    self.assertIn((near, kv), points)
        # Existing chunks keep their multiples and gain the context grid.
        self.assertIn((512, 1024), points)
        self.assertIn((512, 96), points)
        self.assertTrue(set(original_points(n)) <= points)

    def test_short_chunks_without_context_grid_stay_bounded(self):
        n = 4096 + 64
        points = chunked_prefill_points(n, short_chunks=True)
        ones = [kv for chunk, kv in points if chunk == 1]
        self.assertEqual(ones, list(dict.fromkeys(seq_lengths(n))))


if __name__ == "__main__":
    unittest.main()
