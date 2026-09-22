"""The Events articulation of the unified Laplace engine, and the two voicings
that are new (REQ memory/req-unified-laplace-2026-09-21.md sections 4 and 6).

The ARTICULATION is object_resonators unchanged -- its tracker, its packets
(a = e/(e+2)), its detector disk, its Fixed decay (gamma = ln 1000 / T60), its
attack, its slots and its tails.  A cell here only says WHICH of those settings
the axis offers (objects_params) and reads the bank back:

    Sine        : b_s = sum_j w_j Re(z_j)               the law's own kernel, so
                                                        the cell IS `ca_object_resonators`
    Saw/Square  : b_s = sum_j w_j |z_j| W(theta_j)      _render_wave
    FM          : y_s = A_s sin(theta_c + sum_j beta_j sin(theta_j)),
                  beta_j = I w_j |z_j|,  A_s = sqrt(sum_j (w_j |z_j|)^2)
                                                        _amps + _fm_sum

theta_j = arg(z_j) + pi/2 is the mode's OWN phase, read out of the state and never
kept beside it.  Two things follow.  The h = 1 term of W is then |z| sin(theta) =
Re(z) exactly, so a wave is the sine line with harmonics on top and a blend
between the two never beats; and a mode the tracker moves into a tail slot carries
its phase inside its own z, so no voicing can fall out of step with the bank.
(The FM carrier has no state to be read from, so its phase is ATTACHED to the
slots -- object_resonators.attach_slot_state -- and travels with them.)

W(theta; f) = sum_h c_h b(h f) sin(h theta) is the wave law of
memory/req-laplace-carriers-2026-09-20.md section 3 unchanged: c_h = 1/h (Saw) or
1/h on odd h (Square), and the band limit b is taken at the frequency of EVERY
harmonic of EVERY wave, never at f0.  The waves are played from that engine's own
band-limited tables, built for the EXACT frequency that sounds.

The FM law of memory/req-laplace-fm-2026-09-20.md sections 2-4 is unchanged too:
beta is a phase deviation in RADIANS, divided by nothing -- not by the number of
modes, not by sum a, not by an RMS; one carrier per figure on f0; every modulator
in the SAME phase sum; no feedback and no cascades; the sum built at OVERSAMPLE x
sr and brought down through that engine's Kaiser FIR, with its fixed 5 Hz DC
blocker after the decimation.  What is NEW -- and what makes the cell a new law
rather than a consequence of an old one -- is that beta_j now DECAYS with the
mode: a struck figure is an FM percussion whose index falls with its amplitude.

    The trajectories are computed on the audio clock by _amps and read linearly
    between consecutive samples for the oversampled phase sum.  Running the
    resonator recurrence at the oversampled rate instead would silently change
    the decay constants of the Objects law, which is the one thing this engine
    must not do.  The modulator PHASE is not interpolated: it free-runs at
    OVERSAMPLE x sr from the value the state gave it, which is exact, because
    arg(z) advances by exactly omega per sample.

_render_wave and _amps are copies of object_resonators._render with the readout
replaced: every state update, every ramp and every rounding is the law's, and at
`tab = -1` a mode reads back exactly Re(z) -- which a gate checks bit for bit
against the original kernel.

The module lives apart from laplace_unified so that the Env cells -- and with
them the prototype's default tab -- never pull numba or the 2000-line resonator
bank just to play a Laplace line (the seam's A1 win).
"""
import math

import numpy as np

from casynth_config import TWO_PI, _RAMP
from . import object_resonators as orz
from . import laplace_carriers as lc
from . import laplace_fm as lfm
from . import event_network as en
from . import render_pool

try:
    import numba as _numba
    _jit = _numba.njit(cache=True, nogil=True)
except ImportError:                    # pragma: no cover
    def _jit(f):
        return f

ROLE_FREE, ROLE_ACTIVE = orz.ROLE_FREE, orz.ROLE_ACTIVE
I_K, I_R_LEFT, I_G_LEFT, I_Q_LEFT = orz.I_K, orz.I_R_LEFT, orz.I_G_LEFT, orz.I_Q_LEFT
R_CUR, R_TGT, R_INC = orz.R_CUR, orz.R_TGT, orz.R_INC
P_LCUR, P_LINC, P_LTGT, P_RCUR, P_RINC, P_RTGT = range(6)
H_PREV, H_MIX = 0, 1
HALF_PI = math.pi / 2.0
TAIL_FLOOR = orz.TAIL_FLOOR


# ── Saw / Square: the law's kernel with a branched readout ───────────────────
#
# THE BLOCK IS BUILT BY SLOT, NOT BY SAMPLE (2026-09-22).  Reading a mode back as
# a wave costs an atan2 and a sqrt PER MODE PER SAMPLE: on the prototype's own
# Random field that is 1040 live modes x 352 samples = 366 000 of each inside one
# 7.98 ms block, and the device starved the moment the player moved `artic` to
# Events -- 504 underruns in ten seconds, the case tests/test_live_budget.py
# carries.  Measured there: the recurrence itself is 1.5 ms and the readout 6.3.
#
# The SAMPLES cannot be divided -- every mode is a recurrence, sample t needs
# t - 1 -- which is why render_pool's own split (a range of samples, what FM and
# the wave bank use) does not apply here.  The SLOTS can: two slots share nothing
# but the ramps the whole block rides and the sum at the very end.  So the kernel
# became three passes:
#
#   _wave_ramps   the ramps the block shares -- r, g, q and the blend -- one cheap
#                 pass over the samples, the same arithmetic in the same order
#   _wave_slots   a RANGE OF SLOTS, each carried across the whole block, writing
#                 what it contributes to L and R; this is the pass workers share
#   _wave_mix     the sum over slots IN SLOT ORDER, the DC blocker, the output
#
# The bytes do not move.  Every slot does its own operations in its own order (a
# slot's state is its own), the ramps are what the sample loop computed, and L is
# still accumulated over slots in ascending slot order -- so the additions happen
# in the order they always did.  The gate that holds this is
# tests/test_laplace_unified.ThreadedRender, which runs the same live scene
# through one thread and through the pool and compares the blocks bit for bit.


@_jit
def _live_slots(role, slots):
    """The slots that are not free, in ascending order -- the order the single
    loop visited them in, which is the order their contributions are summed."""
    k = 0
    for s in range(role.shape[0]):
        if role[s] != ROLE_FREE:
            slots[k] = s
            k += 1
    return k


@_jit
def _wave_ramps(n, rr, gg, qq, ints, bramp, blending, rb, gb, qb, mb):
    """The per-sample ramps the whole block shares: the decay r, the gain g, the
    packet pole q and the waveform blend.  Lifted out of the sample loop so a
    worker never touches them -- the arithmetic, and the order of it, is the
    sample loop's own."""
    for t in range(n):
        k = ints[I_K]
        left = ints[I_R_LEFT]
        if left > 0:
            left -= 1
            if left == 0:
                rr[R_CUR] = rr[R_TGT]
            else:
                rr[R_CUR] = rr[R_CUR] + rr[R_INC]
            ints[I_R_LEFT] = left
        left = ints[I_G_LEFT]
        if left > 0:
            left -= 1
            if left == 0:
                gg[R_CUR] = gg[R_TGT]
            else:
                gg[R_CUR] = gg[R_CUR] + gg[R_INC]
            ints[I_G_LEFT] = left
        left = ints[I_Q_LEFT]
        if left > 0:
            left -= 1
            if left == 0:
                qq[R_CUR] = qq[R_TGT]
            else:
                qq[R_CUR] = qq[R_CUR] + qq[R_INC]
            ints[I_Q_LEFT] = left
        rb[t] = rr[R_CUR]
        gb[t] = gg[R_CUR]
        qb[t] = qq[R_CUR]
        mb[t] = bramp[t] if blending == 1 else 1.0
        ints[I_K] = k + 1


@_jit
def _wave_slots(lo, hi, n, slots, role, ndrive, npulse, nlive, zre, zim, cth, sth,
                wcur, winc, wtgt, wleft, zf, zs, zu, zfm, zsm, zum, pan, pleft,
                cst, level, slaw, gam_a, gam_t, ksm, sr,
                tab_a, tab_b, tabs, table_n, blending, rb, qb, mb, cl, cr):
    """Slots slots[lo:hi] across the whole block: `a_j |z_j| W(phi_j)` in place of
    Re(z_j), the readout of object_resonators._render.

    tab_a / tab_b [s, j] : the row of `tabs` a mode reads, -1 = read Re(z).
    `blending` (0/1) mixes the two readouts across the block on `bramp` -- the
    one-block glide a waveform change gets.  The phase is READ OUT of the state
    (arg z + pi/2), never kept beside it: a mode that the tracker moves to a tail
    slot carries its phase with its own z, and nothing can fall out of step.

    cl / cr [s, t] : what this slot adds to L and R, summed by _wave_mix."""
    qf = cst[en.C_QF]; qs = cst[en.C_QS]; strength = cst[en.C_STRENGTH]
    scale = table_n / TWO_PI
    for si in range(lo, hi):
        s = slots[si]
        nl = nlive[s]
        # neither the role nor the decay law changes inside a block, so what the
        # sample loop re-read every sample is read once here
        nd = ndrive[s] if role[s] == ROLE_ACTIVE else npulse[s]
        adaptive = slaw[s] == 1
        lev = 0.0
        for t in range(n):
            r = rb[t]
            q = qb[t]
            mixb = mb[t]
            wl = wleft[s]
            if wl > 0:
                wl -= 1
                if wl == 0:
                    for j in range(nl):
                        wcur[s, j] = wtgt[s, j]
                else:
                    for j in range(nl):
                        wcur[s, j] = wcur[s, j] + winc[s, j]
                wleft[s] = wl
            pl = pleft[s]
            if pl > 0:
                pl -= 1
                if pl == 0:
                    pan[s, P_LCUR] = pan[s, P_LTGT]
                    pan[s, P_RCUR] = pan[s, P_RTGT]
                else:
                    pan[s, P_LCUR] = pan[s, P_LCUR] + pan[s, P_LINC]
                    pan[s, P_RCUR] = pan[s, P_RCUR] + pan[s, P_RINC]
                pleft[s] = pl
            p = strength * ((1.0 - qf) * zf[s] - (1.0 - qs) * zs[s]) / (qs - qf)
            zf[s] = zf[s] * qf
            zs[s] = zs[s] * qs
            u = q * zu[s] + (1.0 - q) * p
            zu[s] = u
            p = u
            acc = 0.0
            for j in range(nl):
                re = zre[s, j]
                im = zim[s, j]
                c = cth[s, j]
                sn = sth[s, j]
                if adaptive:
                    ga = ksm * gam_a[s, j] + (1.0 - ksm) * gam_t[s, j]
                    gam_a[s, j] = ga
                    rj = math.exp(-ga / sr)
                else:
                    rj = r
                if j < nd:
                    pm = strength * ((1.0 - qf) * zfm[s, j] - (1.0 - qs) * zsm[s, j]) / (qs - qf)
                    zfm[s, j] = zfm[s, j] * qf
                    zsm[s, j] = zsm[s, j] * qs
                    um = q * zum[s, j] + (1.0 - q) * pm
                    zum[s, j] = um
                    nre = rj * (c * re - sn * im) + (p + um)
                else:
                    nre = rj * (c * re - sn * im)
                nim = rj * (sn * re + c * im)
                zre[s, j] = nre
                zim[s, j] = nim
                # the readout: Re(z) for a sine line, |z| W(theta) for a wave.
                # theta = arg(z) + pi/2 is the mode's OWN phase, so the h = 1 term
                # of W is |z| sin(theta) = Re(z) exactly -- a wave is the sine line
                # with harmonics on top, and a blend between them never beats.
                ta = tab_a[s, j]
                tb = tab_b[s, j]
                if ta < 0 and tb < 0:
                    val = nre
                else:
                    mag = math.sqrt(nre * nre + nim * nim)
                    th = math.atan2(nim, nre) + HALF_PI
                    if th < 0.0:
                        th += TWO_PI
                    elif th >= TWO_PI:
                        th -= TWO_PI
                    x = th * scale
                    i0 = int(x)
                    fr = x - i0
                    if i0 >= table_n:                    # pragma: no cover (theta is wrapped)
                        i0 = table_n - 1
                        fr = 0.0
                    i1 = i0 + 1
                    if i1 >= table_n:
                        i1 = 0
                    if blending == 1:
                        if ta < 0:
                            va = nre
                        else:
                            va = mag * (tabs[ta, i0] * (1.0 - fr) + tabs[ta, i1] * fr)
                        if tb < 0:
                            vb = nre
                        else:
                            vb = mag * (tabs[tb, i0] * (1.0 - fr) + tabs[tb, i1] * fr)
                        val = va + (vb - va) * mixb
                    elif tb < 0:
                        val = nre
                    else:
                        val = mag * (tabs[tb, i0] * (1.0 - fr) + tabs[tb, i1] * fr)
                acc += wcur[s, j] * val
            b = acc
            a = b if b >= 0.0 else -b
            if a > lev:
                lev = a
            cl[s, t] = pan[s, P_LCUR] * b
            cr[s, t] = pan[s, P_RCUR] * b
        level[s] = lev


@_jit
def _wave_mix(n, slots, n_slots, cl, cr, gb, hp_h, hp, out_scale, out):
    """The sum over slots -- in slot order, the order the sample loop added them
    in -- then the engine's DC blocker and the block's gain."""
    for t in range(n):
        L = 0.0
        R = 0.0
        for si in range(n_slots):
            s = slots[si]
            L += cl[s, t]
            R += cr[s, t]
        yL = hp_h * ((hp[0, H_PREV] + L) - hp[0, H_MIX])
        hp[0, H_PREV] = yL
        hp[0, H_MIX] = L
        yR = hp_h * ((hp[1, H_PREV] + R) - hp[1, H_MIX])
        hp[1, H_PREV] = yR
        hp[1, H_MIX] = R
        g = gb[t]
        out[t, 0] = yL * out_scale * g
        out[t, 1] = yR * out_scale * g


class WaveScratch:
    """The room one wave block needs: the shared ramps and the per-slot halves of
    L and R.  A cell keeps one -- allocating 2 x S x n floats 125 times a second
    is exactly the kind of cost this split is meant to remove."""

    __slots__ = ('n', 'rb', 'gb', 'qb', 'mb', 'cl', 'cr', 'slots')

    def __init__(self, slots_max, n):
        self.n = int(n)
        self.rb = np.zeros(n)
        self.gb = np.zeros(n)
        self.qb = np.zeros(n)
        self.mb = np.zeros(n)
        self.cl = np.zeros((slots_max, n))
        self.cr = np.zeros((slots_max, n))
        self.slots = np.zeros(slots_max, np.int64)


def render_wave(n, out, role, ndrive, npulse, nlive, zre, zim, cth, sth,
                wcur, winc, wtgt, wleft, zf, zs, zu, zfm, zsm, zum, pan, pleft,
                rr, gg, qq, ints, cst, hp_h, hp, out_scale, level,
                slaw, gam_a, gam_t, ksm, sr,
                tab_a, tab_b, tabs, table_n, blending, bramp,
                scratch=None, threads=None):
    """One block of the wave readout, the slots divided over the render pool.

    `threads=1` renders where it stands and is the byte reference; any other
    count changes nothing but who does which slot (see the module note)."""
    S = role.shape[0]
    if scratch is None or scratch.n != n or scratch.cl.shape[0] < S:
        scratch = WaveScratch(S, n)
    level[:] = 0.0                       # a free slot reports no level, as before
    n_slots = _live_slots(role, scratch.slots)
    _wave_ramps(n, rr, gg, qq, ints, bramp, blending,
                scratch.rb, scratch.gb, scratch.qb, scratch.mb)
    rest = (n, scratch.slots, role, ndrive, npulse, nlive, zre, zim, cth, sth,
            wcur, winc, wtgt, wleft, zf, zs, zu, zfm, zsm, zum, pan, pleft,
            cst, level, slaw, gam_a, gam_t, ksm, sr,
            tab_a, tab_b, tabs, table_n, blending,
            scratch.rb, scratch.qb, scratch.mb, scratch.cl, scratch.cr)
    parts = render_pool.THREADS if threads is None else int(threads)
    pool = render_pool.pool(threads)
    # a handful of slots is not worth a hand-off: the workers would spend more on
    # being started than on the slots they were given
    if pool is None or parts <= 1 or n_slots < 2 * parts:
        _wave_slots(0, n_slots, *rest)
    else:
        spans = render_pool.ranges(n_slots, parts)
        futures = [pool.submit(_wave_slots, a, b, *rest) for a, b in spans[1:]]
        _wave_slots(spans[0][0], spans[0][1], *rest)   # the caller takes a share
        for f in futures:
            f.result()
    _wave_mix(n, scratch.slots, n_slots, scratch.cl, scratch.cr, scratch.gb,
              hp_h, hp, out_scale, out)


def _render_wave(n, out, role, ndrive, npulse, nlive, zre, zim, cth, sth,
                 wcur, winc, wtgt, wleft, zf, zs, zu, zfm, zsm, zum, pan, pleft,
                 rr, gg, qq, ints, cst, hp_h, hp, out_scale, level,
                 slaw, gam_a, gam_t, ksm, sr,
                 tab_a, tab_b, tabs, table_n, blending, bramp):
    """render_wave on ONE thread -- the form the byte gates call."""
    render_wave(n, out, role, ndrive, npulse, nlive, zre, zim, cth, sth,
                wcur, winc, wtgt, wleft, zf, zs, zu, zfm, zsm, zum, pan, pleft,
                rr, gg, qq, ints, cst, hp_h, hp, out_scale, level,
                slaw, gam_a, gam_t, ksm, sr,
                tab_a, tab_b, tabs, table_n, blending, bramp, threads=1)


# ── FM: the same articulation read as amplitude trajectories ─────────────────

@_jit
def _amps(n, role, ndrive, npulse, nlive, zre, zim, cth, sth,
          wcur, winc, wtgt, wleft, zf, zs, zu, zfm, zsm, zum, pan, pleft,
          rr, gg, qq, ints, cst, level, slaw, gam_a, gam_t, ksm, sr,
          row_of, slot_of, amp, carr, panl, panr, gain):
    """One block of the Objects articulation WITHOUT a readout: every state and
    ramp advances exactly as _render advances it, and the amplitude trajectory
    a_j(t) = w_j |z_j(t)| of the selected modes is written out instead of being
    summed into a sine.

    row_of[s, j] : the row of `amp` a mode writes, -1 = not carried this block.
    slot_of[s]   : the row of `carr` / `panl` / `panr` a slot writes, -1 = silent.
    `gain` is the block's ramped master gain, sample by sample (the kernel owns
    that ramp in this engine, so the voicing must get it from here).

    THE SINGLE LOOP: render_amps below divides this over the pool the same way
    the wave readout divides its own, and falls back to this form when there is
    nothing to divide.  It is also what the threading gate compares against."""
    S = role.shape[0]
    qf = cst[en.C_QF]; qs = cst[en.C_QS]; strength = cst[en.C_STRENGTH]
    for s in range(S):
        level[s] = 0.0
    for t in range(n):
        k = ints[I_K]
        left = ints[I_R_LEFT]
        if left > 0:
            left -= 1
            if left == 0:
                rr[R_CUR] = rr[R_TGT]
            else:
                rr[R_CUR] = rr[R_CUR] + rr[R_INC]
            ints[I_R_LEFT] = left
        left = ints[I_G_LEFT]
        if left > 0:
            left -= 1
            if left == 0:
                gg[R_CUR] = gg[R_TGT]
            else:
                gg[R_CUR] = gg[R_CUR] + gg[R_INC]
            ints[I_G_LEFT] = left
        left = ints[I_Q_LEFT]
        if left > 0:
            left -= 1
            if left == 0:
                qq[R_CUR] = qq[R_TGT]
            else:
                qq[R_CUR] = qq[R_CUR] + qq[R_INC]
            ints[I_Q_LEFT] = left
        r = rr[R_CUR]
        q = qq[R_CUR]
        gain[t] = gg[R_CUR]
        for s in range(S):
            if role[s] == ROLE_FREE:
                continue
            nl = nlive[s]
            wl = wleft[s]
            if wl > 0:
                wl -= 1
                if wl == 0:
                    for j in range(nl):
                        wcur[s, j] = wtgt[s, j]
                else:
                    for j in range(nl):
                        wcur[s, j] = wcur[s, j] + winc[s, j]
                wleft[s] = wl
            pl = pleft[s]
            if pl > 0:
                pl -= 1
                if pl == 0:
                    pan[s, P_LCUR] = pan[s, P_LTGT]
                    pan[s, P_RCUR] = pan[s, P_RTGT]
                else:
                    pan[s, P_LCUR] = pan[s, P_LCUR] + pan[s, P_LINC]
                    pan[s, P_RCUR] = pan[s, P_RCUR] + pan[s, P_RINC]
                pleft[s] = pl
            p = strength * ((1.0 - qf) * zf[s] - (1.0 - qs) * zs[s]) / (qs - qf)
            zf[s] = zf[s] * qf
            zs[s] = zs[s] * qs
            u = q * zu[s] + (1.0 - q) * p
            zu[s] = u
            p = u
            nd = ndrive[s] if role[s] == ROLE_ACTIVE else npulse[s]
            adaptive = slaw[s] == 1
            ssq = 0.0
            for j in range(nl):
                re = zre[s, j]
                im = zim[s, j]
                c = cth[s, j]
                sn = sth[s, j]
                if adaptive:
                    ga = ksm * gam_a[s, j] + (1.0 - ksm) * gam_t[s, j]
                    gam_a[s, j] = ga
                    rj = math.exp(-ga / sr)
                else:
                    rj = r
                if j < nd:
                    pm = strength * ((1.0 - qf) * zfm[s, j] - (1.0 - qs) * zsm[s, j]) / (qs - qf)
                    zfm[s, j] = zfm[s, j] * qf
                    zsm[s, j] = zsm[s, j] * qs
                    um = q * zum[s, j] + (1.0 - q) * pm
                    zum[s, j] = um
                    nre = rj * (c * re - sn * im) + (p + um)
                else:
                    nre = rj * (c * re - sn * im)
                nim = rj * (sn * re + c * im)
                zre[s, j] = nre
                zim[s, j] = nim
                row = row_of[s, j]
                if row >= 0:
                    a = wcur[s, j] * math.sqrt(nre * nre + nim * nim)
                    amp[row, t] = a
                    ssq += a * a
            v = slot_of[s]
            if v >= 0:
                A = math.sqrt(ssq)
                carr[v, t] = A
                panl[v, t] = pan[s, P_LCUR]
                panr[v, t] = pan[s, P_RCUR]
                if A > level[s]:
                    level[s] = A
        ints[I_K] = k + 1


@_jit
def _amps_ramps(n, rr, gg, qq, ints, rb, qb, gain):
    """The per-sample ramps the whole block shares -- the decay r, the gain g and
    the packet pole q.  Lifted out of the sample loop so a worker never touches
    them; the arithmetic, and the order of it, is the sample loop's own."""
    for t in range(n):
        k = ints[I_K]
        left = ints[I_R_LEFT]
        if left > 0:
            left -= 1
            if left == 0:
                rr[R_CUR] = rr[R_TGT]
            else:
                rr[R_CUR] = rr[R_CUR] + rr[R_INC]
            ints[I_R_LEFT] = left
        left = ints[I_G_LEFT]
        if left > 0:
            left -= 1
            if left == 0:
                gg[R_CUR] = gg[R_TGT]
            else:
                gg[R_CUR] = gg[R_CUR] + gg[R_INC]
            ints[I_G_LEFT] = left
        left = ints[I_Q_LEFT]
        if left > 0:
            left -= 1
            if left == 0:
                qq[R_CUR] = qq[R_TGT]
            else:
                qq[R_CUR] = qq[R_CUR] + qq[R_INC]
            ints[I_Q_LEFT] = left
        rb[t] = rr[R_CUR]
        gain[t] = gg[R_CUR]
        qb[t] = qq[R_CUR]
        ints[I_K] = k + 1


@_jit
def _amps_slots(lo, hi, n, slots, role, ndrive, npulse, nlive, zre, zim, cth, sth,
                wcur, winc, wtgt, wleft, zf, zs, zu, zfm, zsm, zum, pan, pleft,
                cst, level, slaw, gam_a, gam_t, ksm, sr,
                row_of, slot_of, amp, carr, panl, panr, rb, qb):
    """Slots slots[lo:hi] across the whole block: the body of _amps with the two
    loops turned inside out.  A slot's state is its own and it writes its own
    rows of `amp` / `carr` / `panl` / `panr`, so two slots share nothing."""
    qf = cst[en.C_QF]
    qs = cst[en.C_QS]
    strength = cst[en.C_STRENGTH]
    for si in range(lo, hi):
        s = slots[si]
        lev = 0.0
        for t in range(n):
            r = rb[t]
            q = qb[t]
            nl = nlive[s]
            wl = wleft[s]
            if wl > 0:
                wl -= 1
                if wl == 0:
                    for j in range(nl):
                        wcur[s, j] = wtgt[s, j]
                else:
                    for j in range(nl):
                        wcur[s, j] = wcur[s, j] + winc[s, j]
                wleft[s] = wl
            pl = pleft[s]
            if pl > 0:
                pl -= 1
                if pl == 0:
                    pan[s, P_LCUR] = pan[s, P_LTGT]
                    pan[s, P_RCUR] = pan[s, P_RTGT]
                else:
                    pan[s, P_LCUR] = pan[s, P_LCUR] + pan[s, P_LINC]
                    pan[s, P_RCUR] = pan[s, P_RCUR] + pan[s, P_RINC]
                pleft[s] = pl
            p = strength * ((1.0 - qf) * zf[s] - (1.0 - qs) * zs[s]) / (qs - qf)
            zf[s] = zf[s] * qf
            zs[s] = zs[s] * qs
            u = q * zu[s] + (1.0 - q) * p
            zu[s] = u
            p = u
            nd = ndrive[s] if role[s] == ROLE_ACTIVE else npulse[s]
            adaptive = slaw[s] == 1
            ssq = 0.0
            for j in range(nl):
                re = zre[s, j]
                im = zim[s, j]
                c = cth[s, j]
                sn = sth[s, j]
                if adaptive:
                    ga = ksm * gam_a[s, j] + (1.0 - ksm) * gam_t[s, j]
                    gam_a[s, j] = ga
                    rj = math.exp(-ga / sr)
                else:
                    rj = r
                if j < nd:
                    pm = strength * ((1.0 - qf) * zfm[s, j] - (1.0 - qs) * zsm[s, j]) / (qs - qf)
                    zfm[s, j] = zfm[s, j] * qf
                    zsm[s, j] = zsm[s, j] * qs
                    um = q * zum[s, j] + (1.0 - q) * pm
                    zum[s, j] = um
                    nre = rj * (c * re - sn * im) + (p + um)
                else:
                    nre = rj * (c * re - sn * im)
                nim = rj * (sn * re + c * im)
                zre[s, j] = nre
                zim[s, j] = nim
                row = row_of[s, j]
                if row >= 0:
                    a = wcur[s, j] * math.sqrt(nre * nre + nim * nim)
                    amp[row, t] = a
                    ssq += a * a
            v = slot_of[s]
            if v >= 0:
                A = math.sqrt(ssq)
                carr[v, t] = A
                panl[v, t] = pan[s, P_LCUR]
                panr[v, t] = pan[s, P_RCUR]
                if A > lev:
                    lev = A
        level[s] = lev


def render_amps(n, role, ndrive, npulse, nlive, zre, zim, cth, sth,
                wcur, winc, wtgt, wleft, zf, zs, zu, zfm, zsm, zum, pan, pleft,
                rr, gg, qq, ints, cst, level, slaw, gam_a, gam_t, ksm, sr,
                row_of, slot_of, amp, carr, panl, panr, gain,
                scratch=None, threads=None):
    """One block of the trajectories, the slots divided over the render pool.

    With one thread -- or with too few slots to be worth a hand-off -- it IS
    _amps, the single loop above."""
    parts = render_pool.THREADS if threads is None else int(threads)
    pool = render_pool.pool(threads)
    head = (n, role, ndrive, npulse, nlive, zre, zim, cth, sth,
            wcur, winc, wtgt, wleft, zf, zs, zu, zfm, zsm, zum, pan, pleft)
    if pool is None or parts <= 1:
        _amps(*head, rr, gg, qq, ints, cst, level, slaw, gam_a, gam_t, ksm, sr,
              row_of, slot_of, amp, carr, panl, panr, gain)
        return
    if scratch is None or scratch.n != n or scratch.slots.shape[0] < role.shape[0]:
        scratch = FMScratch(role.shape[0], row_of.shape[1], n, n)
    n_slots = _live_slots(role, scratch.slots)
    if n_slots < 2 * parts:
        _amps(*head, rr, gg, qq, ints, cst, level, slaw, gam_a, gam_t, ksm, sr,
              row_of, slot_of, amp, carr, panl, panr, gain)
        return
    level[:] = 0.0                       # a free slot reports no level, as before
    _amps_ramps(n, rr, gg, qq, ints, scratch.rb, scratch.qb, gain)
    rest = (n, scratch.slots, role, ndrive, npulse, nlive, zre, zim, cth, sth,
            wcur, winc, wtgt, wleft, zf, zs, zu, zfm, zsm, zum, pan, pleft,
            cst, level, slaw, gam_a, gam_t, ksm, sr,
            row_of, slot_of, amp, carr, panl, panr, scratch.rb, scratch.qb)
    spans = render_pool.ranges(n_slots, parts)
    futures = [pool.submit(_amps_slots, a, b, *rest) for a, b in spans[1:]]
    _amps_slots(spans[0][0], spans[0][1], *rest)     # the caller takes a share
    for f in futures:
        f.result()


@_jit
def _fm_sum(n_os, oversample, index, amp, amp_prev, carr, carr_prev,
            panl, panl_prev, panr, panr_prev, row_start, inc_mod, th_mod,
            inc_c, th_c, out_l, out_r):
    """The phase sum at OVERSAMPLE x sr: one carrier per sounding slot, every mode
    of that slot a modulator of index beta_j = I a_j entering the SAME phase.

    The trajectories arrive on the audio clock and are read linearly between
    consecutive samples (sub-sample os-1 of a group IS that sample), so an index
    never steps.  `*_prev` is the value the last block ended on.

    THE SINGLE LOOP, kept as it was written: render_fm below is what the cell
    calls, and this is the reference its gate compares against (the wave bank
    keeps laplace_carriers.render_reference for the same reason).  The law tests
    of tests/test_laplace_unified read the law off this form."""
    V = carr.shape[0]
    for m in range(n_os):
        t = m // oversample
        fr = (m % oversample + 1) / oversample
        L = 0.0
        R = 0.0
        for v in range(V):
            acc = 0.0
            for k in range(row_start[v], row_start[v + 1]):
                a0 = amp_prev[k] if t == 0 else amp[k, t - 1]
                a = a0 + (amp[k, t] - a0) * fr
                acc += index * a * math.sin(th_mod[k])
                thn = th_mod[k] + inc_mod[k]
                if thn >= TWO_PI:
                    thn -= TWO_PI
                th_mod[k] = thn
            A0 = carr_prev[v] if t == 0 else carr[v, t - 1]
            A = A0 + (carr[v, t] - A0) * fr
            y = A * math.sin(th_c[v] + acc)
            thn = th_c[v] + inc_c
            if thn >= TWO_PI:
                thn -= TWO_PI
            th_c[v] = thn
            l0 = panl_prev[v] if t == 0 else panl[v, t - 1]
            r0 = panr_prev[v] if t == 0 else panr[v, t - 1]
            L += (l0 + (panl[v, t] - l0) * fr) * y
            R += (r0 + (panr[v, t] - r0) * fr) * y
        out_l[m] = L
        out_r[m] = R


# ── the same sum, built by CARRIER instead of by sample (2026-09-22) ──────────
#
# WHY.  On the prototype's own Random field this cell cost 25.6 ms of the 7.98 ms
# block -- three times real time, and the device starved the moment the player
# put the FM voicing on the Events articulation (the case
# tests/test_live_budget.EventsFMBudget carries).  The reason is the law itself:
# every mode of every figure is a modulator with a sine of its own, and the sum
# is built at OVERSAMPLE x sr, so that field asks for 1018 modulators and 113
# carriers = 3.2 million sines inside one block.  _fm_sum alone was 21.1 of those
# 25.6 ms, and it was the only kernel of this family still on one thread: the
# wave bank divides its slots, Env + FM divides its samples, this one divided
# nothing.
#
# The SAMPLES cannot be divided here either -- a modulator's phase free-runs from
# the value the state gave it, so sample m needs m - 1 -- but the CARRIERS can: a
# carrier shares nothing with its neighbour except the sum at the very end.  So
# the sum became two passes, the shape the wave readout already has:
#
#   _fm_slots   a RANGE OF CARRIERS, each carried across the whole oversampled
#               block, writing what that carrier sounds; this is the pass the
#               workers share
#   _fm_mix     the pan and the sum over carriers IN CARRIER ORDER
#
# The bytes do not move.  A carrier's modulators are its own rows of `amp` and of
# `th_mod` (the rows are grouped by slot), so each phase advances once per
# oversampled sample in the order it always did; the pan is applied and L is
# accumulated over carriers in ascending order, which is the order the single
# loop added them in.  The gate is tests/test_laplace_unified.ThreadedRender.


@_jit
def _fm_slots(v0, v1, n_os, oversample, index, amp, amp_prev, carr, carr_prev,
              row_start, inc_mod, th_mod, inc_c, th_c, yv):
    """The carriers [v0, v1) of one block, each with its own modulators -- the
    arithmetic of _fm_sum with the two loops turned inside out."""
    for v in range(v0, v1):
        k0 = row_start[v]
        k1 = row_start[v + 1]
        thc = th_c[v]
        for m in range(n_os):
            t = m // oversample
            fr = (m % oversample + 1) / oversample
            acc = 0.0
            for k in range(k0, k1):
                a0 = amp_prev[k] if t == 0 else amp[k, t - 1]
                a = a0 + (amp[k, t] - a0) * fr
                acc += index * a * math.sin(th_mod[k])
                thn = th_mod[k] + inc_mod[k]
                if thn >= TWO_PI:
                    thn -= TWO_PI
                th_mod[k] = thn
            A0 = carr_prev[v] if t == 0 else carr[v, t - 1]
            A = A0 + (carr[v, t] - A0) * fr
            yv[v, m] = A * math.sin(thc + acc)
            thn = thc + inc_c
            if thn >= TWO_PI:
                thn -= TWO_PI
            thc = thn
        th_c[v] = thc


@_jit
def _fm_mix(n_os, oversample, yv, n_v, panl, panl_prev, panr, panr_prev,
            out_l, out_r):
    """The pan and the sum over carriers -- in carrier order, the order the
    single loop added them in."""
    for m in range(n_os):
        t = m // oversample
        fr = (m % oversample + 1) / oversample
        L = 0.0
        R = 0.0
        for v in range(n_v):
            y = yv[v, m]
            l0 = panl_prev[v] if t == 0 else panl[v, t - 1]
            r0 = panr_prev[v] if t == 0 else panr[v, t - 1]
            L += (l0 + (panl[v, t] - l0) * fr) * y
            R += (r0 + (panr[v, t] - r0) * fr) * y
        out_l[m] = L
        out_r[m] = R


class FMScratch:
    """The room one FM block needs: the ramps the trajectories share, the live
    slots, the trajectories themselves, and what each carrier sounded sample by
    OVERSAMPLED sample.  A cell keeps one -- on the field above `yv` is 113 x 2816
    doubles and `amp` is 1018 x 352, and asking the OS for 5 MB 125 times a second
    is exactly the kind of cost these splits are meant to remove.

    Nothing here is cleared between blocks except the two index maps: the kernels
    write every element they read (every row of `amp` on every sample, every
    carrier of `carr` / `panl` / `panr`, every sample of `l_os` / `r_os`), and the
    one case that does not -- a block with no carrier at all -- clears the two
    oversampled buffers itself."""

    __slots__ = ('n', 'n_os', 'rb', 'qb', 'slots', 'yv', 'row_of', 'slot_of',
                 'l_os', 'r_os', '_amp', '_carr', '_panl', '_panr')

    GRAIN = 128                          # `amp` grows in whole handfuls of rows

    def __init__(self, slots_max, modes_max, n, n_os):
        slots_max = max(int(slots_max), 1)
        self.n = int(n)
        self.n_os = int(n_os)
        self.rb = np.zeros(n)
        self.qb = np.zeros(n)
        self.slots = np.zeros(slots_max, np.int64)
        self.yv = np.zeros((slots_max, int(n_os)))
        self.row_of = np.full((slots_max, max(int(modes_max), 1)), -1, np.int64)
        self.slot_of = np.full(slots_max, -1, np.int64)
        self.l_os = np.zeros(int(n_os))
        self.r_os = np.zeros(int(n_os))
        self._amp = np.zeros((self.GRAIN, n))
        self._carr = np.zeros((slots_max, n))
        self._panl = np.zeros((slots_max, n))
        self._panr = np.zeros((slots_max, n))

    def room(self, k, v):
        """`amp`, `carr`, `panl`, `panr` for a block of k modes and v carriers,
        and the two index maps emptied.  The buffers only ever grow."""
        k = max(int(k), 1)
        v = max(int(v), 1)
        if self._amp.shape[0] < k:
            rows = -(-k // self.GRAIN) * self.GRAIN
            self._amp = np.zeros((rows, self.n))
        self.row_of.fill(-1)
        self.slot_of.fill(-1)
        return (self._amp[:k], self._carr[:v], self._panl[:v], self._panr[:v])


def render_fm(n_os, oversample, index, amp, amp_prev, carr, carr_prev,
              panl, panl_prev, panr, panr_prev, row_start, inc_mod, th_mod,
              inc_c, th_c, out_l, out_r, scratch=None, threads=None):
    """One block of the FM readout, the carriers divided over the render pool.

    With one thread -- or with too few carriers to be worth a hand-off -- it IS
    _fm_sum, the single loop above."""
    n_v = carr.shape[0]
    parts = render_pool.THREADS if threads is None else int(threads)
    pool = render_pool.pool(threads)
    if pool is None or parts <= 1 or n_v < 2 * parts:
        _fm_sum(n_os, oversample, index, amp, amp_prev, carr, carr_prev,
                panl, panl_prev, panr, panr_prev, row_start, inc_mod, th_mod,
                inc_c, th_c, out_l, out_r)
        return
    if scratch is None or scratch.n_os != n_os or scratch.yv.shape[0] < n_v:
        scratch = FMScratch(n_v, 1, n_os // oversample, n_os)
    rest = (n_os, oversample, index, amp, amp_prev, carr, carr_prev, row_start,
            inc_mod, th_mod, inc_c, th_c, scratch.yv)
    spans = render_pool.ranges(n_v, parts)
    futures = [pool.submit(_fm_slots, a, b, *rest) for a, b in spans[1:]]
    _fm_slots(spans[0][0], spans[0][1], *rest)       # the caller takes a share
    for f in futures:
        f.result()
    _fm_mix(n_os, oversample, scratch.yv, n_v, panl, panl_prev, panr, panr_prev,
            out_l, out_r)


# -- band-limited wave tables, kept across blocks ------------------------------

class WaveTables:
    """The pool of band-limited tables this cell reads -- which is the ONE pool
    laplace_carriers keeps (2026-09-22).

    It used to be a second pool on top of that one: every table was built through
    lc.wavetable (so it landed in lc's buffer) and then COPIED into a buffer of
    its own, and when that one filled it threw its whole buffer away and started
    at 16 rows.  Rows already written into tab_a / tab_b for the block in flight
    then pointed past the new buffer, and the readout -- a compiled kernel with no
    bounds check -- walked off the end of memory.  That is how the user's
    instrument died on 2026-09-22: an access violation, no traceback, the window
    simply gone.  Dragging `harm` reached it in seconds, because it asks for a
    table at a frequency that has never sounded, on every mode, on every frame.

    So the cell asks lc.table_rows for a whole block at once: one pool, one LRU,
    rows that no second frequency can take while this block is reading them, and
    -1 for a frequency the pool cannot serve (the kernel reads that as Re z).
    This class is what is left -- the buffer and its length, for the kernel."""

    def __init__(self, cap=None, table_n=lc.TABLE_N):
        self.table_n = int(table_n)

    @property
    def buf(self):
        b = lc.table_buffer()
        return _EMPTY_TABLES if b is None else b

    def rows(self, waveform, freqs, sr, new_block=True):
        return lc.table_rows(waveform, freqs, sr, new_block=new_block)

    @property
    def index(self):                     # what display() counts
        return lc._TAB_ROW

    @property
    def rebuilds(self):
        return lc._TAB_DECLINED[0]


_EMPTY_TABLES = np.zeros((1, lc.TABLE_N))    # nothing has been built yet


def objects_params(params):
    """The Objects parameter set the Events articulation pins (REQ section 4):
    the Disk detector, the Laplace spectrum, a uniform packet, the Fixed decay --
    and the four knobs the axis does offer.  `frequency_scale` never acts under
    the Laplace law; ctx.f0 and the note's transpose play its part."""
    out = {k: params[k] for k in orz.SPECTRUM_KEYS}
    out.update(detector=orz.DET_DISK, spectrum=orz.SPEC_LAPLACE,
               excitation=orz.EXC_UNIFORM, birth_strength=1.0,
               decay_law=orz.LAW_FIXED, frequency_scale=220.0,
               events=int(params['events']),
               radius_mul=float(params['radius_mul']),
               decay_s=float(params['decay_s']),
               attack_ms=float(params['attack_ms']))
    return out


class _EventsCell:
    """What every Events voicing shares: the Objects engine underneath."""

    def __init__(self, ctx, params, grid, exc, gain, strike):
        self.ctx = ctx
        self.n = int(ctx.block)
        self.sr = float(ctx.sr)
        self.obj = orz.ObjectResonatorsEngine(ctx, objects_params(params))
        self.obj.init(grid, exc, gain)
        if not strike:
            # a switch of an axis is not an event: the articulation takes the
            # current field as its OWN previous field and waits for a change
            self.obj.prime_silent()

    def set_field(self, grid, exc):
        self.obj.update_field(grid, exc)

    def set_params(self, params):
        self.obj.set_params(objects_params(params))

    def display(self):
        return self.obj.display()

    # the live pairs of the bank: a mode inside n_live whose frequency sounds,
    # and (for the trajectories) one that is either driven or still audible
    def _selection(self, transpose, audible_only):
        obj = self.obj
        S, M = obj.role.shape[0], obj.zre.shape[1]
        j = np.arange(M)[None, :]
        live = (j < obj.nlive[:, None]) & (obj.role != orz.ROLE_FREE)[:, None]
        freq = obj.ffreq * float(transpose)
        sel = live & (freq > 0.0) & (freq < lc.BAND_ZERO * self.sr)
        if audible_only:
            fed = np.where(obj.role == orz.ROLE_ACTIVE, obj.ndrive, obj.npulse)
            driven = j < fed[:, None]
            mag = np.hypot(obj.zre, obj.zim)
            sel = sel & (driven | (mag > orz.TAIL_FLOOR))
        return sel, freq


class EventsBankCell(_EventsCell):
    """Events + Bank: the Objects articulation read back as a wave per mode.

    At Sine this cell IS the Objects engine -- it runs the law's own kernel, so
    the cell is byte for byte `ca_object_resonators` (REQ section 3).  A wave
    runs _render_wave on the same state; a change of the waveform blends the two
    readouts over one block, the glide the wave bank already uses."""

    def __init__(self, ctx, params, grid, exc, gain, strike, tables=None):
        super().__init__(ctx, params, grid, exc, gain, strike)
        S, M = self.obj.role.shape[0], self.obj.zre.shape[1]
        self.tab_a = np.full((S, M), -1, np.int64)
        self.tab_b = np.full((S, M), -1, np.int64)
        self.tables = tables if tables is not None else WaveTables()
        self.wave = int(params['waveform'])
        self.wave_prev = self.wave
        self._sig = None
        self._out = np.zeros((self.n, 2))
        # the room a wave block needs, kept for the life of the cell (see the
        # module note above render_wave)
        self._scratch = WaveScratch(S, self.n)

    def set_params(self, params):
        super().set_params(params)
        self.wave = int(params['waveform'])

    def _assign(self, transpose):
        """A table row per live mode, rebuilt only when the bank, the note or the
        waveform has moved -- and one table per DISTINCT frequency, not per mode:
        the figures of a field share their resonances, so the tables a scene needs
        are tens, not hundreds."""
        sel, freq = self._selection(transpose, audible_only=False)
        sig = (self.wave, self.wave_prev, float(transpose),
               sel.tobytes(), freq[sel].tobytes())
        if sig == self._sig:
            return
        self._sig = sig
        self.tab_a[:] = -1
        self.tab_b[:] = -1
        if self.wave == lc.WF_SINE and self.wave_prev == lc.WF_SINE:
            return
        uniq, inv = np.unique(freq[sel], return_inverse=True)
        first = True
        for wf, dst in ((self.wave_prev, self.tab_a), (self.wave, self.tab_b)):
            if wf == lc.WF_SINE:
                continue
            # one ask for the whole block: see WaveTables.  A -1 comes back for a
            # frequency the pool cannot serve, and the kernel reads that as Re z.
            # A blend asks twice for the SAME block, so the second ask must not
            # count as a new one or it would evict the rows of the first.
            dst[sel] = self.tables.rows(wf, uniq, self.sr, new_block=first)[inv]
            first = False

    def render_float(self, gain, gain_prev, transpose, env=None):
        obj = self.obj
        obj.begin_block(gain, transpose, gain_prev)
        self._assign(transpose)
        out = self._out
        if self.wave == lc.WF_SINE and self.wave_prev == lc.WF_SINE:
            obj._kernel(self.n, out)                    # the law's own kernel
        else:
            render_wave(self.n, out, obj.role, obj.ndrive, obj.npulse, obj.nlive,
                        obj.zre, obj.zim, obj.cth, obj.sth, obj.wcur, obj.winc,
                        obj.wtgt, obj.wleft, obj.zf, obj.zs, obj.zu, obj.zfm,
                        obj.zsm, obj.zum, obj.pan, obj.pleft, obj.rr, obj.gg,
                        obj.qq, obj.ints, obj.consts, obj.hp_h, obj.hp,
                        orz.OUT_SCALE, obj.level, obj.slaw, obj.gam_a, obj.gam_t,
                        obj.ksm, obj.sr, self.tab_a, self.tab_b,
                        self.tables.buf, self.tables.table_n,
                        1 if self.wave_prev != self.wave else 0, _RAMP,
                        scratch=self._scratch)
        obj.end_block()
        self.wave_prev = self.wave
        return out.copy()

    def display(self):
        d = super().display()
        d.update(voicing='bank', wave=int(self.wave),
                 wave_name=lc.WAVE_NAMES[int(self.wave)],
                 wave_tables=len(self.tables.index),
                 tables_declined=int(self.tables.rebuilds))
        return d


class EventsFMCell(_EventsCell):
    """Events + FM: the trajectories of the modes become the indices of ONE
    carrier per figure (REQ section 6 -- a NEW law, not a consequence of an old
    one: a struck figure is an FM percussion whose index decays with it).

    The phase sum is built at OVERSAMPLE x sr and brought down through the FM
    engine's own Kaiser FIR, and its fixed DC blocker runs after the decimation;
    the index is beta_j = I a_j in radians, divided by nothing.  At FM d = 0
    every sounding figure is a decaying sine on f0 scaled by its own A_object --
    not a return to the sum of modes."""

    def __init__(self, ctx, params, grid, exc, gain, strike, oversample=None):
        super().__init__(ctx, params, grid, exc, gain, strike)
        S, M = self.obj.role.shape[0], self.obj.zre.shape[1]
        self.oversample = int(lfm.OVERSAMPLE if oversample is None else oversample)
        self.kernel = lfm._kernel(self.oversample, self.sr)
        self.taps = len(self.kernel)
        self.dc_r = float(math.exp(-TWO_PI * lfm.DC_HZ / self.sr))
        self.index = float(params['fm_depth'])
        # The carrier is an oscillator of its own, so its phase is attached to the
        # bank's slots and travels with them; a MODULATOR needs no state at all --
        # arg(z) advances by exactly omega per sample, so the phase is read out of
        # the mode itself at every boundary and free-runs inside the block, which
        # is what lets the sum be built at OVERSAMPLE x sr without interpolating a
        # full-band signal.
        self.th_c = self.obj.attach_slot_state(np.zeros(S))
        self.amp_last = np.zeros((S, M))
        self.carr_last = np.zeros(S)
        self.panl_last = np.zeros(S)
        self.panr_last = np.zeros(S)
        self.fir_l = np.zeros(self.taps - 1)
        self.fir_r = np.zeros(self.taps - 1)
        self.dc_xl = self.dc_yl = 0.0
        self.dc_xr = self.dc_yr = 0.0
        self._gain = np.zeros(self.n)
        self._out = np.zeros((self.n, 2))
        # the room one block needs, kept for the life of the cell (see FMScratch)
        self._scratch = FMScratch(S, M, self.n, self.n * self.oversample)
        self._n_rows = 0
        self._n_carriers = 0

    def set_params(self, params):
        super().set_params(params)
        self.index = float(params['fm_depth'])

    def render_float(self, gain, gain_prev, transpose, env=None):
        obj = self.obj
        obj.begin_block(gain, transpose, gain_prev)
        n, os = self.n, self.oversample
        sel, freq = self._selection(transpose, audible_only=True)
        slots = np.nonzero(sel.any(axis=1))[0]
        rs, rj = np.nonzero(sel)
        V, K = len(slots), len(rs)
        self._n_rows, self._n_carriers = K, V
        sc = self._scratch
        row_of, slot_of = sc.row_of, sc.slot_of
        amp, carr, panl, panr = sc.room(K, V)
        row_of[rs, rj] = np.arange(K)
        slot_of[slots] = np.arange(V)
        zre0 = obj.zre[rs, rj].copy()            # the state the block STARTS from
        zim0 = obj.zim[rs, rj].copy()
        render_amps(n, obj.role, obj.ndrive, obj.npulse, obj.nlive, obj.zre,
                    obj.zim, obj.cth, obj.sth, obj.wcur, obj.winc, obj.wtgt,
                    obj.wleft, obj.zf, obj.zs, obj.zu, obj.zfm, obj.zsm, obj.zum,
                    obj.pan, obj.pleft, obj.rr, obj.gg, obj.qq, obj.ints,
                    obj.consts, obj.level, obj.slaw, obj.gam_a, obj.gam_t,
                    obj.ksm, obj.sr, row_of, slot_of, amp, carr, panl, panr,
                    self._gain, scratch=self._scratch)
        n_os = n * os
        l_os, r_os = sc.l_os, sc.r_os
        if not V:
            l_os[:] = 0.0          # nothing sounds, and nothing wrote the buffers
            r_os[:] = 0.0
        else:
            # rows are grouped by slot because np.nonzero walks s ascending
            row_start = np.append(np.searchsorted(rs, slots), K).astype(np.int64)
            th_c = self.th_c[slots].copy()
            inc_mod = TWO_PI * freq[rs, rj] / (self.sr * os)
            inc_c = TWO_PI * obj.f0 * float(transpose) / (self.sr * os)
            # + pi/2 so that beta_j sin(theta_j) is I w_j Re(z_j) on the audio
            # clock; + one sub-sample because sub-sample os-1 of a group IS the
            # audio sample the phase was read from
            th_mod = np.mod(np.arctan2(zim0, zre0) + HALF_PI + inc_mod, TWO_PI)
            render_fm(n_os, os, self.index, amp, self.amp_last[rs, rj],
                      carr, self.carr_last[slots], panl, self.panl_last[slots],
                      panr, self.panr_last[slots], row_start, inc_mod, th_mod,
                      inc_c, th_c, l_os, r_os, scratch=self._scratch)
            self.th_c[slots] = th_c
        self.amp_last[:] = 0.0
        self.carr_last[:] = 0.0
        self.panl_last[:] = 0.0
        self.panr_last[:] = 0.0
        if K:
            self.amp_last[rs, rj] = amp[:, -1]
        if V:
            self.carr_last[slots] = carr[:, -1]
            self.panl_last[slots] = panl[:, -1]
            self.panr_last[slots] = panr[:, -1]
        L, self.fir_l = lfm.decimate(l_os, self.kernel, os, self.fir_l)
        R, self.fir_r = lfm.decimate(r_os, self.kernel, os, self.fir_r)
        L, self.dc_xl, self.dc_yl = lfm.dc_block(L, self.dc_r, self.dc_xl, self.dc_yl)
        R, self.dc_xr, self.dc_yr = lfm.dc_block(R, self.dc_r, self.dc_xr, self.dc_yr)
        out = self._out
        out[:, 0] = L * orz.OUT_SCALE * self._gain
        out[:, 1] = R * orz.OUT_SCALE * self._gain
        obj.end_block()
        return out.copy()

    def display(self):
        d = super().display()
        d.update(voicing='fm', depth=float(self.index),
                 oversample=int(self.oversample), taps=int(self.taps),
                 delay_samples=int(lfm.FIR_DELAY_OUT),
                 modulators=int(self._n_rows), carriers=int(self._n_carriers))
        return d
