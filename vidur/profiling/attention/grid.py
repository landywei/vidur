"""Chunked-prefill points of the attention profile (no torch needed).

Vidur's default grid profiles chunks of 64 tokens or more, with the prior
context at multiples of the chunk size. A chunked-prefill scheduler gives a
prompt's last chunk whatever its token budget leaves, so a serving run can
ask for any chunk from one token up, at any prior context. Two options widen
the grid: `short_chunks` adds chunks of 1 to 63 tokens, and `context_grid`
also profiles every chunk up to CONTEXT_GRID_MAX_CHUNK tokens at the
sequence-length grid of prior contexts.
"""

from itertools import product

# Chunks above this size occur only as whole prompts (no prior context).
CONTEXT_GRID_MAX_CHUNK = 1024
SHORT_PREFILL_CHUNKS = (1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48, 56)


def prefill_chunk_sizes(max_seq_len, short_chunks=False):
    space = (
        list(range(64, 128 + 1, 16))
        + list(range(128, 1024 + 1, 32))
        + list(range(1024, 4 * 1024 + 1, 64))
        + list(range(4 * 1024, 16 * 1024 + 1, 128))
        + list(range(16 * 1024, 64 * 1024 + 1, 256))
    )
    sizes = []
    for size in space:
        if size > max_seq_len:
            break
        sizes.append(size)
    if short_chunks:
        sizes = [s for s in SHORT_PREFILL_CHUNKS if s <= max_seq_len] + sizes
    return sizes


def seq_lengths(max_seq_len):
    space = (
        list(range(0, 1024 + 1, 32))
        + list(range(1024, 4 * 1024 + 1, 64))
        + list(range(4 * 1024, 64 * 1024 + 1, 256))
    )
    lengths = []
    for length in space:
        if length >= max_seq_len:
            break
        lengths.append(length)
    return lengths


def chunked_prefill_points(max_seq_len, short_chunks=False, context_grid=False):
    """(chunk, prior context) pairs, before validity filtering, in order."""
    points = []
    contexts = seq_lengths(max_seq_len) if context_grid else []
    for chunk in prefill_chunk_sizes(max_seq_len, short_chunks):
        multiples = [i * chunk for i in range(max_seq_len // chunk)]
        if chunk < 64:
            # Multiples of a one-token chunk would be every context length.
            grid = set(seq_lengths(max_seq_len))
            multiples = [m for m in multiples if m in grid]
        extra = contexts if chunk <= CONTEXT_GRID_MAX_CHUNK else []
        kv = list(dict.fromkeys(multiples + extra))
        points.extend(product([chunk], kv))
    return points
