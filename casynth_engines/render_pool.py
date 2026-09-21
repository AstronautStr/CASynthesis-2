"""One worker pool for the engines that split a block (2026-09-21).

WHY.  On the prototype's own field (52x30 Random) the Laplace engines cost more
than the 7.98 ms block they have to fill -- 6.5 ms for FM, 24 ms for the wave
bank with Saw -- and the device starved.  The arithmetic is hundreds of thousands
of sines and table lookups per block; one core cannot make that cheaper, and
numpy drops the GIL inside sin / multiply / sum, so threads really do run at once.

WHAT IT IS NOT.  It is not a way to "parallelise the sound": an engine hands each
worker a range of SAMPLES of the block it is already building, and every sample is
computed by the same operations in the same order as on one thread -- the sum over
sources never crosses two workers.  So the sound is bit-for-bit what one thread
produced whatever the thread count is, and each engine pins that with a gate of
its own (test_laplace_fm.ThreadedRender, test_laplace_carriers.SlabRender).

The pool is shared because a host plays ONE engine at a time; it is built on
first use (a host that only draws a UI never starts a thread) and the workers are
daemons, so an exit never waits for them.
"""
import os
import threading

_CPUS = os.cpu_count() or 1
# Half the cores, at most four: the UI thread, the render thread and the machine's
# own work all need a core too, and beyond four the block is too short to divide
# (measured: 4 workers on a 352-sample block, 8 are slower).
THREADS = max(1, min(4, _CPUS // 2))
try:
    THREADS = max(1, int(os.environ.get('CASYNTH_RENDER_THREADS', THREADS)))
except ValueError:
    pass

_lock = threading.Lock()
_pool = None


def pool(threads=None):
    """The shared pool, or None when a single thread is asked for (then the caller
    renders the block where it stands -- no hand-off, no threads started)."""
    global _pool
    if (THREADS if threads is None else threads) <= 1:
        return None
    with _lock:
        if _pool is None:
            from concurrent.futures import ThreadPoolExecutor
            _pool = ThreadPoolExecutor(THREADS, thread_name_prefix='casynth-render')
        return _pool


def ranges(n, parts):
    """`parts` contiguous [start, stop) ranges covering 0..n, the split a worker
    gets.  Equal by construction, so the workers finish together."""
    edge = [round(i * n / parts) for i in range(parts + 1)]
    return [(a, b) for a, b in zip(edge, edge[1:]) if b > a]
