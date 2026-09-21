"""Which tests MEASURE TIME, and therefore may not share the machine.

A gate that asserts "this block fits in 7.98 ms" is not testing the code when
five other test processes are on the cores -- it is testing the scheduler.  That
is why the whole suite ran one process at a time on a twenty-core machine, and
why the FM budget test still failed now and then inside check.py while passing
alone (memory/current.md, 2026-09-21).

So they say so themselves:

    from timing_gate import timing_test

    @timing_test
    def test_block_budget_of_the_two_new_scenes(self):
        ...

check.py then runs everything else in PARALLEL with CASYNTH_SKIP_TIMING=1 (these
tests skip), and the measured ones afterwards, alone, with nothing else running.
Run a file by hand and nothing is skipped -- the variable is only set by the
parallel wave.
"""
import os
import unittest

SKIP = bool(os.environ.get('CASYNTH_SKIP_TIMING'))


def timing_test(obj):
    """Mark a test (or a whole TestCase) as one that measures wall-clock time."""
    return unittest.skipIf(SKIP, 'measures time: check.py runs it alone')(obj)
