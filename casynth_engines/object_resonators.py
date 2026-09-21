"""N4 -- every connected figure of the field is a resonator bank tuned by its own
Laplacian and struck by the cell changes inside its own circular detector (engine
id `ca_object_resonators`, REQ memory/req-object-resonators-n4-2026-09-16.md; the
Laplace spectrum mode and the tail rules of v2: REQ
memory/req-objects-laplace-comparison-2026-09-17.md and the Researcher review
memory/research/object-resonators-n4-review-2026-09-16.md).

    connected figure -> the spectrum of its Laplacian -> its bank of decaying resonators
    cell changes inside its circle ---------------------> the pulse that strikes the bank
    centre of mass -------------------------------------> detector position and panning

Figures (casynth_lab.figures): 8-connected components across the torus seam, the
periodic centre (cy, cx), the radius R = max torus distance centre -> cell, and the
detector mask K: `detector` = Disk -> every cell within R of the centre (boundary
included, equal weight, nothing outside; a single cell has R = 0 and covers only
itself); Own -> exactly the figure's cells.  Circles are not normalised against
each other: one change may strike several banks.

Spectrum -- two laws, `spectrum` = Figure / Laplace (the slot remembers which law
tuned it: `smode`):
  Figure : f_j = frequency_scale * sqrt(lambda_j) for the first N_BANK (24) positive
           eigenvalues of L = D - A of the figure's full unweighted 8-connectivity
           graph (multiplicities kept, zero mode dropped, no normalisation of the
           lowest mode, no crop, no decimation), weights 1 / n_modes (the bank is
           mean(Re z_j)).  N cells -> at most N - 1 modes; a single cell has none.
  Laplace: the mathematics of the old synth's map_laplacian (casynth_core) on the
           figure's OWN component: casynth_core.laplacian_modes on the full torus
           graph of the component (`fullshape` = 1: no crop, no decimation -- the
           node ceiling of the old synth is not carried over) or map_laplacian on
           the 8 x 8 window around the figure's centroid (`fullshape` = 0, exactly
           the old extract()); the lowest selected mode is ctx.f0 of the scene; the
           seven settings n / spread / alpha / shape / harm / fullshape / dyn have
           the ranges, defaults and names of the old Laplace registry entry and
           the same meaning (mode selection, guard, harmonic pull, weight law).
           `dyn` blends the events field (bench `exc`, the standard events_field of
           the transition) sampled at the figure's own cells, in the node order of
           its graph, into the weights when shape > 0 -- it is a weight, never a
           strike.  Weights are the map_laplacian amplitudes times LAPLACE_GAIN (a
           constant calibration of this mode, measured; no division by the number
           of modes or figures), the bank is sum(w_j Re z_j).  `frequency_scale`
           does not act.  On one compact component away from the seam the
           frequencies and the weights / LAPLACE_GAIN equal map_laplacian on the
           object's bounding box with the same seven settings, f0 and excitation
           (gate: tests/test_objects_laplace.py).
  A change of `spectrum` or of the seven settings, or a new `exc` on the same
  field (dyn > 0 and shape > 0), retunes every sounding figure (frequencies at once,
  states kept, weights over 20 ms) and never strikes.

Block boundary (before the samples of the block), only when the pending field G
differs from the last rendered field G_prev (an unchanged field, a parameter
change, a Restore add nothing):
    components of G are matched to the tracked figures (figures.match: overlap,
    then centre displacement <= 1.5 cells; split / merge -> old figures to tails,
    new identities)
    births = G & ~G_prev ;  deaths = G_prev & ~G
    `events` (v4, 2026-09-17, REQ memory/req-objects-event-source-modal-2026-09-17.md)
    = Both / Births / Deaths selects the SOURCE of new packets (both kinds stay
    positive events; a death is never a negative pulse):
    continuing figure : Both   e = sum(births * K_cur) + sum(deaths * K_prev)
                        Births e = sum(births * K_cur)
                        Deaths e = sum(deaths * K_prev)
    new figure        : Both / Births e = sum(births * K_cur);  Deaths: no packet
                        (the appearance of a bank is not an event)
    figure that lost its identity (left the field, split, merge):
                        Both / Births: no packet, its bank rings out as a tail
                        (the previous life cycle);  Deaths: ONE last packet
                        e = sum(deaths * K_prev) on its previous resonances -- the
                        bank goes to its tail slot WITH its pulse states and the
                        strike, the tail feeds the modes it drove (`npulse`) until
                        that pulse has decayed below TAIL_FLOOR, never again
    a = e / (e + 2)  -> added to BOTH pulse states of the figure's slot
    (K_prev = the detector of the figure's previous geometry in the CURRENT mode).
    A fresh init compares the initial field with zeros once (one starting packet
    for Both / Births -- every cell is a birth; none for Deaths); a switch of
    `events` never makes a packet.  Both is bit for bit the previous path.
    `excitation` (v4) = Uniform / Birth position distributes ONE packet between
    the driven modes: Uniform adds a to the slot's pulse states (every mode gets
    the same pulse -- the previous path); Birth position is defined for the
    combination Events = Births, Detector = Own, Spectrum = Laplace, full = on
    (registry hint `inactive` shows the condition otherwise; a parameter set that
    still holds it outside the condition makes NO packet -- counted `unsupported`,
    never replaced by a uniform strike).  For the set B of births inside the
    figure (its new cells), the normalised eigenvectors phi of its full torus
    Laplacian in the node order of the graph, the m selected modes j and the
    degenerate group Q(j) = {k : |lambda_k - lambda_j| <= 1e-8 max(1, |lambda_j|)}
    of the WHOLE spectrum (before the part / spread selection):
        p_j = sum_{i in B} sum_{k in Q(j)} phi_k(i)^2 / |Q(j)|
        b_j = sqrt(m p_j / sum_l p_l)       (sum_j b_j^2 = m, like Uniform's b = 1)
        delta_j = a b_j  -> added to the pulse states OF MODE j (zfm / zsm)
    sum_l p_l <= 1e-12: no mode gets the packet (counted `zero_participation`).
    `birth_strength` (v5, 2026-09-18, REQ section 5) = s in 0..4 (default 1)
    reshapes the distribution of the NEXT packets only (Birth position only;
    stored but inactive with Uniform):
        s = 0      : the Uniform path itself (a into the slot's pulse states; no
                     spatial participation needed -- but the combination above
                     is still required, s = 0 does not bypass it)
        0 < s < 1  : v_j = (1 - s) + s b_j ;  c_j = sqrt(m) v_j / sqrt(sum v^2)
        s = 1      : c_j = b_j  (the original b, no arithmetic on it)
        1 < s <= 4 : v_j = b_j ** s ;         c_j = sqrt(m) v_j / sqrt(sum v^2)
        delta_j = a c_j  (sum c^2 = m for every packet that exists)
    with s > 0 a zero participation (sum p <= 1e-12) still makes no packet (the
    blend is never applied to a packet that does not exist); an exact zero b_j
    stays zero for s >= 1 and appears from the Uniform share for 0 < s < 1;
    equal b stay equal, one mode gives c = 1.  A change of s never makes a
    packet and never touches the accumulated pulse states of either path.
    The per-mode pulse states keep every past packet with its own coefficients
    (a new mode of a retune starts with none, a vanished mode's states leave
    with it); the output weights are untouched -- a packet never recolours a
    ringing tail.
    A continuing figure keeps its id, colour, pulse and resonator states; its
    frequencies are updated (mode j -> mode j, ascending frequency), NEW modes start
    from zero, modes that VANISHED leave the bank: they ring on undriven at their
    old frequency, weight and panning in a tail slot of their own (v2: a returning
    mode never inherits a ringing state, an old tail never gets a new pulse), and
    the weights of the driven modes ramp over 20 ms (no level step from the count
    change alone).  The panning target follows the centre with a 20 ms ramp.

Sample path (float64), per sample k, for every slot s in use (active figures,
tails, fading tails):
    p_s   = 0.75 ((1 - q_f) z_f - (1 - q_s) z_s) / (q_s - q_f)   the N2 pulse
    z_f  *= q_f ;  z_s *= q_s                                       tau 0.25 / 2 ms
    u_s   = q u_s + (1 - q) p_s                                     Attack (v3, 2026-09-17):
                                                                   one causal pole on the
                                                                   pulse before the bank
    for the modes j < n_pulse of the slot (v4): the same pulse + smoothing on the
        per-mode states (zfm, zsm, zum)[s, j] -> u_sj  (exactly 0 while they are 0)
    for the modes j < n_live of the slot:
        z_j' = r e^{i theta_j} z_j  (+ (u_s + u_sj) if j < n_pulse)  theta_j = 2 pi f_j / SR
        b_s += w_j Re z_j'                                           w_j = ramped weight
    n_pulse = n_driven for an active slot; for a tail `npulse` = 0 -- except a
    tail that took the last Deaths packet with it (its old n_driven until the
    pulse states are below TAIL_FLOOR at a block boundary).
    L += panL_s b_s ;  R += panR_s b_s                             p = cx / (cols - 1),
                                                                   L / R = cos / sin(pi p / 2)
    hp[k] = h (hp[k-1] + mix[k] - mix[k-1])  per channel, h = exp(-2 pi 20 / SR)
    out   = hp * OUT_SCALE * gain[k]           gain = bench gain, 20 ms linear ramp
    r = 10 ** (-3 / (SR decay_s)), a manual decay change ramps r over 20 ms.
    q = exp(-ln 9 / (SR attack_ms / 1000)) for attack_ms > 0, else 0 (REQ
    memory/req-objects-radius-attack-2026-09-17.md: attack_ms is the 10 -> 90 %
    rise time of the smoothing's step response, not a promise about the audible
    attack of a note); a manual Attack change ramps q linearly over 20 ms.  At
    q = 0 the formula gives u_s = p_s EXACTLY (0 u + 1 p), so Attack 0 -- and the
    end of a ramp back to 0 -- is bit for bit the previous sample path; the
    state u_s is per slot, zeroed with the pulse states when a bank becomes a
    tail (a tail gets no excitation), part of the snapshot.  No level
    compensation: the smoothing changes brightness and level (measured in
    demos/objects_radius_attack_report.py).
No sum over the number of figures, no AGC.  From zero state without events the
output is exactly 0.

Slots: N_ACTIVE (24) banks for sounding figures, N_TAILS (96) for tails, N_FADING
(24) for tails being faded out.  Continuing figures keep their slot; new figures
take free active slots largest area first (ties: first cell); without a free slot
a figure is tracked but silent ("sounding X of Y" in the display) and gets a slot
as soon as one is free (silent start, its next events strike it).  A tail is freed
at a block boundary when every mode is below TAIL_FLOOR; when a new tail is needed
and all tail slots are busy the quietest tail fades out over 20 ms in a fading
slot (counted: faded).  v2 -- when even the fading slots are busy nothing is cut:
a whole figure that leaves fades out IN PLACE in its active slot over 20 ms (the
slot is freed afterwards; counted: in place), vanished modes of a continuing
figure fade in place inside its bank (weight -> 0 over 20 ms, undriven).  The
only hard cut left: such in-place fading modes when the figure grows back within
those 20 ms while every tail and fading slot is still busy (counted: dropped).
Undriven modes below TAIL_FLOOR, and faded modes, are zeroed at block boundaries.

Controls: `detector` (Own / Disk, default Disk; changes the masks of the NEXT
events, no packet), `radius_mul` ("Radius x", any finite value >= 0, default 1.0,
the bench slider covers a user-editable range with the default RADIUS_RANGE
0.25..4 -- registry hint `ranges`, REQ memory/req-objects-radius-attack-2026-09-17.md:
the Disk detector uses R_eff = radius_mul * R -- the geometric R of the figure is
never changed, a single cell keeps R = 0, Own ignores it; once R_eff reaches
figures.full_cover_radius the mask is the whole field (display: `covers_all`);
a change acts on the masks of the NEXT events only, no packet; absent from an
older snapshot = 1.0),
`spectrum` (Figure / Laplace, registry default Laplace; absent from an older
snapshot / parameter set = Figure), `frequency_scale` (55..880 Hz, default 220;
Figure law only: retunes every Figure-tuned slot at once, states kept),
`decay_s` (0.20..1.50 s, default 0.80; r ramp 20 ms), `attack_ms` ("Attack",
0..20 ms, default 0 = the previous pulse exactly; q ramp 20 ms; absent from an
older parameter set / snapshot = 0), `events` ("Events", Both / Births / Deaths,
default Both; absent = Both), `excitation` ("Excitation", Uniform / Birth
position, default Uniform; absent = Uniform), `birth_strength` ("Birth strength",
0..4, default 1; absent = 1 = the v4 Birth position exactly), and the seven
Laplace settings (Laplace law only; absent from an older snapshot = their defaults).
Only SR 44100 / block 352 / stereo.  Own model_version / STATE_VERSION (v5 = the
`birth_strength` parameter, no new arrays; v4 = the per-mode pulse states `zfm` /
`zsm` / `zum`, `npulse`, `last_b`, two more counters; a v4 snapshot of model v4 is
accepted as Birth strength 1, a v3 snapshot of model v3 with the v4 arrays at zero,
a v2 snapshot of model v2 additionally as Attack 0 -- bit for bit the older
sound); the snapshot holds every slot array, the tracker
(ids, slots, cells, centres, radii), the id counter, both fields and both
excitations, the ramps, the DC filters and the gain.

Decay law (v6, 2026-09-18, REQ memory/req-objects-decay-2026-09-18.md) -- `decay_law`
= Fixed / Common age / Modal age (0 / 1 / 2, default 0; the bench buttons say
Fixed / Common / Modal, LAW_NAMES holds the full names):
    Fixed      : the previous path bit for bit -- one global r = 10 ** (-3 / (SR
                 decay_s)) with its 20 ms ramp on a manual change, for every slot.
    Common age / Modal age (defined for Spectrum = Laplace, full = on -- any other
                 combination is REFUSED by the registry / set_params / the scene /
                 the snapshot before the valid state changes; Own and Disk both
                 work: q is always taken on the figure's OWN cells, the circle only
                 selects the excitation):
      history  : every cell position holds its freshness u in 0..1 (`age`): a
                 born cell gets 1, a dead cell 0, a survivor keeps its value; after
                 EVERY rendered block u *= exp(-B / (SR AGE_TAU_S)), AGE_TAU_S =
                 0.5 s, on the audio sample clock (the CA pause freezes the field,
                 not u).  The accepted transition is the same G_prev -> G
                 difference the excitation uses (several edits before one render
                 collapse; a re-set of the same value is no event); Stop / reset
                 start a new history; a v5 snapshot imports u = 1 on the live
                 cells of its previous field.  The age belongs to the POSITION,
                 a moving figure does not carry it.
      per mode : for the selected Laplace modes j of a figure (the same modes,
                 the same node order; eigenvectors of the full torus Laplacian),
                 the degenerate group Q(j) (DEGEN_TOL, the whole spectrum):
                     P[j, i] = sum_{k in Q(j)} phi_k(i)^2 / |Q(j)|   (rows sum to 1)
                     q_j     = sum_i P[j, i] u_i                     (clipped to
                               [0, 1] within Q_TOL; a larger excursion is an error)
                     T_j     = decay_s - (decay_s - T_FRESH_S) q_j    T_FRESH_S 0.08
                     gamma_j = ln(1000) / T_j
                 P is cached per figure (cells + settings) and per canonical shape
                 (the Laplacian of the canonical placement is the same matrix at any
                 position, so a repeating phase or a moving figure costs nothing);
                 it is recomputed only for a shape / mode set not seen yet -- never
                 per sample.  Under an adaptive law the frequency call itself hands
                 its graph over (the same numbers, one eigvalsh less).
      Common   : every driven mode of the bank gets gamma_common = mean_j gamma_j
                 (the mean of the RATES, not of the times); Modal: each its own.
      applied  : targets are computed at every block boundary; per sample, for
                 every mode of an adaptive slot,
                     ga <- k ga + (1 - k) gt        k = exp(-1 / (SR GAMMA_SMOOTH_S))
                     r_j = exp(-ga / SR)            GAMMA_SMOOTH_S = 0.020
                     z_j' = r_j e^{i theta_j} z_j + excitation
                 (the same recurrence as Fixed with r_j instead of r).  A new slot
                 / a new mode starts with ga = gt before its first sample; a
                 continuing mode keeps ga (a new target never resets the sound).
                 The law acts on the WHOLE ringing state of the mode, never
                 creates energy, never changes weights, never strikes.
      slots    : `slaw` per slot = 0 (the global r) / 1 (its own gamma states).
                 A switch Fixed -> adaptive starts every slot in use from the
                 gamma of the CURRENT global r (an active bank gets its real
                 targets at the boundary, a tail / fading slot of the Fixed era
                 holds the Decay of that moment); a switch adaptive -> Fixed sets the fixed
                 gamma as the target, the smoothing finishes and the slot returns
                 to the global r once |ga - gt| <= GAMMA_SETTLE gt and the r ramp
                 is done (no step, no event).  A tail / fading slot / in-place
                 fading mode takes its ga / gt with it and HOLDS its last target
                 (no geometry left for a new q); Decay / law changes drive the
                 active banks only; while the law stays Fixed a slaw = 0 tail
                 keeps following the global r (the Fixed semantics of tails and
                 of the Decay knob).  Bit-exact compatibility is
                 promised for runs that stay Fixed the whole time, not for a
                 state after an adaptive history.
    The snapshot (v6) adds `age`, `gam_a` / `gam_t` / `slaw` and the cached P of
    the active figures (`fig_part*`); a v5 snapshot restores as Fixed with the
    gamma states at zero (the same sound).
"""
import math

import numpy as np

from casynth_core import (ENGINE_BY_ID, PATCH_SIZE, extract, laplacian_modes, map_laplacian,
                          eigh_sym)
from casynth_engine import _crop_like_extract
from .engine_api import SoundEngine
from .registry import EngineSpec, register
from . import event_network as en
from . import figures as fg

try:
    import numba as _numba
    _jit = _numba.njit(cache=True, nogil=True)
    HAVE_NUMBA = True
except ImportError:                    # pragma: no cover
    def _jit(f):
        return f
    HAVE_NUMBA = False

ENGINE_ID = 'ca_object_resonators'
LABEL = 'Objects'
# the seven settings of the old Laplace, metadata from the core registry (not duplicated)
LAPLACE_PARAMS = tuple(tuple(p) for p in ENGINE_BY_ID['laplacian']['params'])
SPECTRUM_KEYS = tuple(p[0] for p in LAPLACE_PARAMS)      # n spread alpha shape harm fullshape dyn
RADIUS_RANGE = (0.25, 4.0)             # default slider range of Radius x (the value itself: 0 <= finite)
PARAMS = [('detector', 'Detector', 0, 1, True, 1),
          ('radius_mul', 'Radius x', 0.0, math.inf, False, 1.0),
          ('spectrum', 'Spectrum', 0, 1, True, 1),
          ('events', 'Events', 0, 2, True, 0),
          ('excitation', 'Excitation', 0, 1, True, 0),
          ('birth_strength', 'Birth strength', 0.0, 4.0, False, 1.0),
          ('frequency_scale', 'Freq scale', 55.0, 880.0, False, 220.0),
          ('decay_s', 'Decay', 0.20, 1.50, False, 0.80),
          ('decay_law', 'Decay law', 0, 2, True, 0),
          ('attack_ms', 'Attack', 0.0, 20.0, False, 0.0)] + list(LAPLACE_PARAMS)
# absent from older snapshots / parameter sets: the bit-exact v1 behaviour (Figure law)
OPTIONAL_PARAMS = {'radius_mul': 1.0, 'spectrum': 0, 'attack_ms': 0.0, 'events': 0, 'excitation': 0,
                   'birth_strength': 1.0, 'decay_law': 0}
BIRTH_STRENGTH_RANGE = (0.0, 4.0)      # Birth strength: finite, inside this closed range
BIRTH_STRENGTH_HINT = '0 = Uniform; 1 = original; >1 = stronger selection'
OPTIONAL_PARAMS.update({p[0]: p[5] for p in LAPLACE_PARAMS})
CHOICES = {'detector': ('Own', 'Disk'), 'spectrum': ('Figure', 'Laplace'),
           'events': ('Both', 'Births', 'Deaths'), 'excitation': ('Uniform', 'Birth position'),
           'decay_law': ('Fixed', 'Common', 'Modal')}      # the bench buttons (60 px); full names: LAW_NAMES
LAW_NAMES = ('Fixed', 'Common age', 'Modal age')
DET_OWN, DET_DISK = 0, 1
SPEC_FIGURE, SPEC_LAPLACE = 0, 1
EV_BOTH, EV_BIRTHS, EV_DEATHS = 0, 1, 2
EXC_UNIFORM, EXC_POSITION = 0, 1
LAW_FIXED, LAW_COMMON, LAW_MODAL = 0, 1, 2
POSITION_CONDITION = 'needs Births / Own / Laplace / full'      # the combination Birth position is defined for
LAW_CONDITION = 'needs Laplace + full'                          # the combination Common / Modal age are defined for
# the age-driven loss model (v6): named fixed parameters of this probe, no knobs
T_FRESH_S = 0.08                       # T60 of a mode whose cells are all fresh (q = 1); decay_s is the stable T60
AGE_TAU_S = 0.5                        # memory constant of a cell's freshness (audio sample clock)
GAMMA_SMOOTH_S = 0.020                 # one-pole smoothing of the applied loss rate, per sample
GAMMA_SETTLE = 1e-9                    # a slot converging to Fixed returns to the global r at |ga - gt| <= this * gt
Q_TOL = 1e-9                           # q outside [0, 1] by at most this is clipped ...
Q_ERR = 1e-6                           # ... by more than this it is a calculation error (raised)
SHAPE_CACHE_MAX = 4096                 # participation matrices kept per canonical shape (cleared when full)
LN1000 = math.log(1000.0)              # gamma = ln(1000) / T60 (amplitude -60 dB)
MODEL_VERSION = 'ca_object_resonators_n4_v6'
STATE_VERSION = 6
# older snapshots this engine restores (state version -> model version): the v2 state
# (no Attack) is the v3 state with zu / qq / the attack ramp counter at zero; the v3
# state (no Events / Excitation) is the v4 state with the per-mode pulse states at
# zero, npulse = ndrive of the active slots and the two new counters at zero; the v4
# state (no Birth strength) is the v5 state with birth_strength = 1 (same arrays); the
# v5 state (no Decay law) is the v6 state with decay_law = Fixed, the gamma states at
# zero, no cached participation and u = 1 on the live cells of its previous field
COMPATIBLE_STATES = {2: 'ca_object_resonators_n4_v2', 3: 'ca_object_resonators_n4_v3',
                     4: 'ca_object_resonators_n4_v4', 5: 'ca_object_resonators_n4_v5'}
DEGEN_TOL = 1e-8                       # Birth position: |lambda_k - lambda_j| <= DEGEN_TOL max(1, |lambda_j|) = one group
ZERO_PART = 1e-12                      # Birth position: sum of participations at or below this -> no packet
ATTACK_LN9 = math.log(9.0)             # 10 -> 90 % of a one-pole step response takes ln 9 time constants
SR_REQUIRED = en.SR_REQUIRED
BLOCK_REQUIRED = en.BLOCK_REQUIRED
N_BANK = 24
N_ACTIVE = 24
N_TAILS = 96
N_FADING = 24
N_SLOTS = N_ACTIVE + N_TAILS + N_FADING
HP_HZ = 20.0
RAMP_MS = 20.0                         # decay r, gain, mode weights, panning
OUT_SCALE = 0.5
LAPLACE_GAIN = 0.7                     # Laplace law: weights = LAPLACE_GAIN * map_laplacian amplitudes
                                       # (measured 2026-09-17: A / B RMS gap of the three scenes at 1.0
                                       # was 9.6 / 6.3 / 10.9 dB with 0.25 -> the mean gap closed here;
                                       # the remaining per-scene part is the scene's side_gain)
TAIL_FLOOR = 1e-7                      # |z| below this: a mode / a tail is silent
N_PALETTE = 12                         # colour index = (id - 1) % N_PALETTE (display only)
OVERLAY_TEXT = "Objects: a colour per figure (cells, circle, centre); changes inside strike it"

ROLE_FREE, ROLE_ACTIVE, ROLE_TAIL, ROLE_FADING = 0, 1, 2, 3
I_K, I_R_LEFT, I_G_LEFT, I_Q_LEFT = range(4)
R_CUR, R_TGT, R_INC = range(3)
P_LCUR, P_LINC, P_LTGT, P_RCUR, P_RINC, P_RTGT = range(6)
H_PREV, H_MIX = range(2)
C_EVICT, C_DROP, C_UNVOICED, C_CHANGES, C_INPLACE, C_ZEROPART, C_UNSUPPORTED = range(7)
N_COUNTERS = 7


def decay_r(t60_s, sr=SR_REQUIRED):
    return 10.0 ** (-3.0 / (float(sr) * float(t60_s)))


def gamma_of_r(r, sr=SR_REQUIRED):
    """The loss rate (1 / s) of a per-sample factor r: -sr ln r (r = exp(-gamma / sr))."""
    return -float(sr) * math.log(float(r))


def gamma_smooth_k(sr=SR_REQUIRED, tau_s=GAMMA_SMOOTH_S):
    """The per-sample coefficient of the applied-gamma smoother: exp(-1 / (sr tau))."""
    return math.exp(-1.0 / (float(sr) * float(tau_s)))


def age_decay_k(block, sr=SR_REQUIRED, tau_s=AGE_TAU_S):
    """The per-block factor of a cell's freshness: exp(-block / (sr tau))."""
    return math.exp(-float(block) / (float(sr) * float(tau_s)))


def mode_participation(L, idx, tol=DEGEN_TOL):
    """float64 (m, N): P[j, i] = sum_{k in Q(j)} phi_k(i)^2 / |Q(j)| for the selected
    modes `idx` (eigen-order indices of the Laplacian L), phi the orthonormal
    eigenvectors in the node order of L and Q(j) the degenerate group of j over
    the WHOLE spectrum (degenerate_groups): invariant to the sign and to a
    rotation of the basis inside a group; every row sums to 1 (m = 0: (0, N))."""
    L = np.asarray(L, np.float64)
    idx = np.asarray(idx, np.int64)
    N = int(L.shape[0])
    m = int(len(idx))
    if m == 0:
        return np.zeros((0, N))
    lam, vecs = eigh_sym(L)                  # serialized: see casynth_core._EIGH_LOCK
    sq = vecs * vecs
    P = np.empty((m, N))
    for t, group in enumerate(degenerate_groups(lam, idx, tol)):
        P[t] = sq[:, group].sum(axis=1) / len(group)
    return P


def age_targets(P, u_nodes, t_stable_s, t_fresh_s=T_FRESH_S):
    """(q, T, gamma) of the module docstring for a participation matrix P (m, N)
    and the freshness u_nodes (N,) in the node order of P: q = P u clipped to
    [0, 1] within Q_TOL (RuntimeError beyond Q_ERR), T = t_stable - (t_stable -
    t_fresh) q, gamma = ln(1000) / T -- in this order (no averaging of local
    1 / T).  Empty P -> three empty arrays."""
    P = np.asarray(P, np.float64)
    if P.shape[0] == 0:
        z = np.zeros(0)
        return z, z.copy(), z.copy()
    q = P @ np.asarray(u_nodes, np.float64)
    lo, hi = float(q.min()), float(q.max())
    if lo < -Q_ERR or hi > 1.0 + Q_ERR:
        raise RuntimeError(f"engine {ENGINE_ID}: mode freshness q outside [0, 1]: {lo:.3e} .. {hi:.3e}")
    q = np.clip(q, 0.0, 1.0)
    T = float(t_stable_s) - (float(t_stable_s) - float(t_fresh_s)) * q
    return q, T, LN1000 / T


def decay_law_supported(params):
    """Common / Modal age are defined for Spectrum = Laplace, full = 1."""
    return (int(params.get('spectrum', SPEC_FIGURE)) == SPEC_LAPLACE
            and int(params.get('fullshape', 1)) == 1)


def validate_params(params):
    """The combination rule of the engine (registry hook `validate`): an adaptive
    Decay law outside Laplace / full raises ValueError; nothing else is checked here."""
    law = int(params.get('decay_law', LAW_FIXED))
    if law != LAW_FIXED and not decay_law_supported(params):
        raise ValueError(f"engine {ENGINE_ID}: Decay law {LAW_NAMES[law]} {LAW_CONDITION} "
                         f"(Spectrum Laplace with full = on); set it Fixed first")


def attack_q(attack_ms, sr=SR_REQUIRED):
    """The one-pole coefficient of the Attack smoothing: 0 for attack_ms <= 0 (the
    exact previous path), else exp(-ln 9 / (sr * attack_ms / 1000))."""
    a = float(attack_ms)
    if a <= 0.0:
        return 0.0
    return math.exp(-ATTACK_LN9 / (float(sr) * a / 1000.0))


def pan_of(cx, cols):
    """(L, R) = cos / sin(pi p / 2), p = cx / (cols - 1) clipped to [0, 1]."""
    p = 0.0 if cols <= 1 else min(max(float(cx) / (cols - 1), 0.0), 1.0)
    return math.cos(math.pi * p / 2.0), math.sin(math.pi * p / 2.0)


def trig_of(freqs, sr):
    c = np.empty(len(freqs))
    s = np.empty(len(freqs))
    for j, f in enumerate(freqs):
        th = 2.0 * math.pi * float(f) / sr
        c[j] = math.cos(th)
        s[j] = math.sin(th)
    return c, s


def laplace_settings(params):
    """The seven Laplace settings of a parameter dict (defaults for absent keys)."""
    return {k: params.get(k, OPTIONAL_PARAMS[k]) for k in SPECTRUM_KEYS}


def laplace_modes_of(cells, rows, cols, f0, settings, exc=None, with_graph=False):
    """(freqs, amps) of the old Laplace law on ONE component: `cells` (N, 2) of the
    field (any placement, across the seam too), `settings` the seven keys, `exc`
    the events field of the transition (rows, cols) or None.  amps are the
    map_laplacian amplitudes (before LAPLACE_GAIN); both arrays hold only the
    selected modes (no zero padding).  with_graph (full = 1 only): also the graph
    of the law -- dict(L, order, idx): the Laplacian of the canonical placement,
    the permutation of `cells` into its node order (node k = cells[order[k]]) and
    the eigen-order indices of the selected modes (casynth_core.laplacian_modes
    return_index) -- None for full = 0 (the 8 x 8 window law has no such graph)."""
    cells = np.asarray(cells, np.int64)
    n = int(settings['n'])
    spread, alpha = float(settings['spread']), float(settings['alpha'])
    shape, harm, dyn = float(settings['shape']), float(settings['harm']), float(settings['dyn'])
    canon, order = fg.canonical_placement(cells, rows, cols)
    e_nodes = None
    if exc is not None and shape > 0.0 and dyn > 0.0:
        e_nodes = np.asarray(exc, dtype=float)[cells[order, 0], cells[order, 1]].astype(float)
    graph = None
    if int(settings['fullshape']):
        L = fg.laplacian_matrix(canon, rows, cols)
        if with_graph:
            freqs, amps, idx = laplacian_modes(L, float(f0), n, spread, alpha, shape, harm, dyn, e_nodes,
                                               return_index=True)
            graph = dict(L=L, order=order, idx=idx)
        else:
            freqs, amps = laplacian_modes(L, float(f0), n, spread, alpha, shape, harm, dyn, e_nodes)
    else:
        h, w = int(canon[:, 0].max()) + 1, int(canon[:, 1].max()) + 1
        sub = np.zeros((h, w), np.uint8)
        sub[canon[:, 0], canon[:, 1]] = 1
        patch = extract(sub, PATCH_SIZE)
        exc_patch = None
        if e_nodes is not None:
            esub = np.zeros((h, w))
            esub[canon[:, 0], canon[:, 1]] = e_nodes
            exc_patch = _crop_like_extract(esub, sub, PATCH_SIZE)
        freqs, amps = map_laplacian(patch, float(f0), n, spread, alpha, shape, harm, False, dyn, exc_patch)
    num = int(np.count_nonzero(freqs))                    # the selected modes are the non-zero prefix
    if with_graph:
        return freqs[:num].copy(), amps[:num].copy(), graph
    return freqs[:num].copy(), amps[:num].copy()


def degenerate_groups(lam, idx, tol=DEGEN_TOL):
    """[int64 array] for every eigen index in `idx`: the indices k of the WHOLE
    spectrum `lam` (ascending) with |lam_k - lam_j| <= tol max(1, |lam_j|)."""
    lam = np.asarray(lam, np.float64)
    return [np.nonzero(np.abs(lam - lam[j]) <= tol * max(1.0, abs(float(lam[j]))))[0] for j in idx]


def birth_position_weights(L, idx, born):
    """(b, total): the Birth position coefficients of the module docstring for the
    selected modes `idx` (eigen-order indices of L) and the births `born` (bool (N,)
    in the node order of L): p_j = the mean over the degenerate group of j of the
    summed squared eigenvector entries at the born nodes (invariant to the sign and
    to a rotation of the basis inside the group), b_j = sqrt(m p_j / sum p); with
    sum p <= ZERO_PART (no born node takes part in any selected mode) b = 0 and
    the caller makes no packet.  total = sum p."""
    idx = np.asarray(idx, np.int64)
    m = int(len(idx))
    if m == 0:
        return np.zeros(0), 0.0
    lam, vecs = eigh_sym(np.asarray(L, np.float64))   # serialized (casynth_core)
    born = np.asarray(born, bool)
    part = (vecs[born, :] ** 2).sum(axis=0) if born.any() else np.zeros(len(lam))
    p = np.empty(m)
    for t, group in enumerate(degenerate_groups(lam, idx)):
        p[t] = float(part[group].sum()) / len(group)
    total = float(p.sum())
    if total <= ZERO_PART:
        return np.zeros(m), total
    return np.sqrt(m * p / total), total


def valid_birth_strength(value):
    """True for a finite number inside BIRTH_STRENGTH_RANGE (bool excluded)."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        return False
    v = float(value)
    return math.isfinite(v) and BIRTH_STRENGTH_RANGE[0] <= v <= BIRTH_STRENGTH_RANGE[1]


def birth_strength_profile(b, strength):
    """The Birth strength law of the module docstring on an EXISTING spatial packet
    b (m,) (sum b^2 = m, no all-zero b here): s = 1 returns b itself (no arithmetic),
    0 < s < 1 blends (1 - s) + s b and 1 < s <= 4 raises b ** s, both renormalised
    to sum c^2 = m; s = 0 is not a case of this function (the caller takes the
    Uniform path).  Raises ValueError outside 0 < s <= 4."""
    s = float(strength)
    if not (0.0 < s <= BIRTH_STRENGTH_RANGE[1]):
        raise ValueError(f"engine {ENGINE_ID}: birth_strength_profile needs 0 < s <= "
                         f"{BIRTH_STRENGTH_RANGE[1]}, got {strength!r}")
    if s == 1.0:
        return b
    b = np.asarray(b, np.float64)
    m = int(len(b))
    if m == 0:
        return b
    v = (1.0 - s) + s * b if s < 1.0 else b ** s
    norm = math.sqrt(float((v * v).sum()))
    if not (norm > 0.0):                                # pragma: no cover  (b has a positive entry)
        raise ValueError(f"engine {ENGINE_ID}: Birth strength on an all-zero packet")
    return math.sqrt(m) * v / norm


@_jit
def _render(n, out, role, ndrive, npulse, nlive, zre, zim, cth, sth, wcur, winc, wtgt, wleft,
            zf, zs, zu, zfm, zsm, zum, pan, pleft, rr, gg, qq, ints, cst, hp_h, hp, out_scale, level,
            slaw, gam_a, gam_t, ksm, sr):
    """`n` samples into out (n, 2) = the formulas of the module docstring INCLUDING
    OUT_SCALE and the ramped gain.  level[s] = max |b_s| over the block (display).
    v6: a slot with slaw = 1 smooths its applied gamma per mode and sample (ksm)
    and uses r_j = exp(-gamma / sr) in place of the global r; slaw = 0 slots run
    the previous arithmetic unchanged."""
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
        g = gg[R_CUR]
        q = qq[R_CUR]
        L = 0.0
        R = 0.0
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
            u = q * zu[s] + (1.0 - q) * p        # Attack: q = 0 gives exactly p
            zu[s] = u
            p = u
            # the fed modes: an active slot drives n_driven, a tail only what its last
            # packet (Deaths) left it -- npulse (0 for an ordinary tail)
            nd = ndrive[s] if role[s] == ROLE_ACTIVE else npulse[s]
            adaptive = slaw[s] == 1
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
                    # the per-mode packet states (Birth position): the same pulse and
                    # smoothing; exactly 0.0 while the states are 0, so a uniform packet
                    # reaches the mode as p + 0.0 = p
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
                acc += wcur[s, j] * nre
            b = acc
            a = b if b >= 0.0 else -b
            if a > level[s]:
                level[s] = a
            L += pan[s, P_LCUR] * b
            R += pan[s, P_RCUR] * b
        yL = hp_h * ((hp[0, H_PREV] + L) - hp[0, H_MIX])
        hp[0, H_PREV] = yL
        hp[0, H_MIX] = L
        yR = hp_h * ((hp[1, H_PREV] + R) - hp[1, H_MIX])
        hp[1, H_PREV] = yR
        hp[1, H_MIX] = R
        out[t, 0] = yL * out_scale * g
        out[t, 1] = yR * out_scale * g
        ints[I_K] = k + 1


class Figure:
    """One tracked figure (a plain record; the snapshot stores its fields)."""
    __slots__ = ('id', 'slot', 'cells', 'centre', 'radius')

    def __init__(self, fid, slot, cells, centre, radius):
        self.id = int(fid)
        self.slot = int(slot)
        self.cells = np.ascontiguousarray(cells, dtype=np.int64)
        self.centre = (float(centre[0]), float(centre[1]))
        self.radius = float(radius)


class ObjectResonatorsEngine(SoundEngine):
    STATE_VERSION = STATE_VERSION

    def __init__(self, ctx, params):
        super().__init__(ctx, params)
        if int(ctx.sr) != SR_REQUIRED or int(ctx.block) != BLOCK_REQUIRED or ctx.channels != 2:
            raise ValueError(f"engine {ENGINE_ID}: only sr {SR_REQUIRED} / block {BLOCK_REQUIRED} / "
                             f"stereo are supported (got {ctx.sr} / {ctx.block} / {ctx.channels})")
        for k, v in OPTIONAL_PARAMS.items():
            self.params.setdefault(k, v)
        validate_params(self.params)                     # v6: an adaptive law needs Laplace / full
        self.engine_id = ENGINE_ID
        self.model_version = MODEL_VERSION
        self.sr = float(ctx.sr)
        self.f0 = float(ctx.f0)
        n = int(ctx.block)
        self._out = np.zeros((n, 2))
        self.hp_h = math.exp(-2.0 * math.pi * HP_HZ / self.sr)
        self.ramp_n = int(round(RAMP_MS / 1000.0 * self.sr))
        self.consts = en.consts(self.sr)
        self.cache = fg.SpectrumCache(N_BANK)
        self.pshape = {}                           # (rows, cols, canonical cells, law key) -> P (v6)
        self._graph_hand = None                    # (cells bytes, law key, graph) of the last frequency call
        self.ksm = gamma_smooth_k(self.sr)
        self.age_k = age_decay_k(n, self.sr)
        self._grid = None
        self._exc = None
        # Per-slot arrays a HOST attached (attach_slot_state): they are moved,
        # split and cleared WITH the slots, so a voicing that lives outside this
        # engine -- the FM carrier of the unified engine, REQ
        # memory/req-unified-laplace-2026-09-21.md section 6 -- cannot fall out
        # of step with the bank it reads.  Empty by default and not part of the
        # snapshot: the owner of an array owns its continuation too.
        self.extra_slot = []
        self._zero()
        self._kernel(0, np.zeros((0, 2)))          # compile / load the cached kernel

    # -- model state ----------------------------------------------------------------
    def _zero(self):
        S, M = N_SLOTS, N_BANK
        self._transpose = 1.0          # the sounding pitch / ctx.f0 (the note)
        self.role = np.zeros(S, np.int64)
        self.slot_id = np.zeros(S, np.int64)
        self.smode = np.zeros(S, np.int64)
        self.ndrive = np.zeros(S, np.int64)
        self.npulse = np.zeros(S, np.int64)
        self.nlive = np.zeros(S, np.int64)
        self.zre = np.zeros((S, M))
        self.zim = np.zeros((S, M))
        self.sqrtlam = np.zeros((S, M))
        self.ffreq = np.zeros((S, M))
        self.cth = np.ones((S, M))
        self.sth = np.zeros((S, M))
        self.wcur = np.zeros((S, M))
        self.winc = np.zeros((S, M))
        self.wtgt = np.zeros((S, M))
        self.wleft = np.zeros(S, np.int64)
        self.zf = np.zeros(S)
        self.zs = np.zeros(S)
        self.zu = np.zeros(S)
        self.zfm = np.zeros((S, M))
        self.zsm = np.zeros((S, M))
        self.zum = np.zeros((S, M))
        self.pan = np.zeros((S, 6))
        self.pleft = np.zeros(S, np.int64)
        r = decay_r(self.params['decay_s'], self.sr)
        self.rr = np.array([r, r, 0.0])
        self.gg = np.zeros(3)
        q = attack_q(self.params.get('attack_ms', OPTIONAL_PARAMS['attack_ms']), self.sr)
        self.qq = np.array([q, q, 0.0])
        self.ints = np.zeros(4, np.int64)
        self.hp = np.zeros((2, 2))
        self.level = np.zeros(S)
        self.last_e = np.zeros(S)
        self.last_a = np.zeros(S)
        self.last_b = np.zeros((S, M))             # coefficients of the last packet (display)
        # evictions (tail -> fading), hard drops, unvoiced blocks, changes, in-place fades,
        # Birth position packets with zero participation, packets refused (unsupported combination)
        self.counters = np.zeros(N_COUNTERS, np.int64)
        # v6: the per-slot loss law and the per-mode applied / target gamma (1 / s)
        self.slaw = np.zeros(S, np.int64)
        self.gam_a = np.zeros((S, M))
        self.gam_t = np.zeros((S, M))
        # v6: the freshness of every cell position (set with the field in init / restore)
        self.age = None if self._grid is None else np.zeros(self._grid.shape)
        self.part = {}                             # id -> dict(key, P, r, c): cached participation
        self.figures = {}                          # id -> Figure
        self.next_id = 1
        self.G_prev = None
        self.E_prev = None
        for a in getattr(self, 'extra_slot', ()):      # an attached voicing restarts too
            a[:] = 0.0

    def attach_slot_state(self, array):
        """Hand this engine a per-slot array of the caller's own (N_SLOTS,) that
        should follow the slots: it is zeroed when a slot is freed or the engine
        restarts, copied when a bank is moved to a tail, and carried over when a
        set of modes splits off into one.  The array stays the caller's; nothing
        here ever reads its values."""
        if array.shape[0] != N_SLOTS:
            raise ValueError(f"engine {ENGINE_ID}: attached slot state of "
                             f"{array.shape[0]} != {N_SLOTS} slots")
        self.extra_slot.append(array)
        return array

    def _kernel(self, n, out):
        _render(n, out, self.role, self.ndrive, self.npulse, self.nlive, self.zre, self.zim, self.cth, self.sth,
                self.wcur, self.winc, self.wtgt, self.wleft, self.zf, self.zs, self.zu, self.zfm, self.zsm,
                self.zum, self.pan, self.pleft, self.rr, self.gg, self.qq, self.ints, self.consts, self.hp_h,
                self.hp, OUT_SCALE, self.level, self.slaw, self.gam_a, self.gam_t, self.ksm, self.sr)

    # -- geometry helpers -------------------------------------------------------------
    def _radius_mul(self):
        return float(self.params.get('radius_mul', OPTIONAL_PARAMS['radius_mul']))

    def _eff_radius(self, fig):
        """The Disk detector radius: radius_mul * the geometric R (0 stays 0)."""
        return self._radius_mul() * fig.radius

    def _mask(self, fig):
        rows, cols = self._grid.shape
        if int(self.params['detector']) == DET_OWN:
            return fg.own_mask(fig.cells, rows, cols)
        return fg.disk_mask(fig.centre, self._eff_radius(fig), rows, cols)

    def _scale(self):
        return float(self.params['frequency_scale'])

    def _spectrum(self):
        return int(self.params.get('spectrum', OPTIONAL_PARAMS['spectrum']))

    def _settings(self):
        return laplace_settings(self.params)

    def _laplace_uses_exc(self):
        s = self._settings()
        return self._spectrum() == SPEC_LAPLACE and float(s['shape']) > 0.0 and float(s['dyn']) > 0.0

    def _exc_array(self, exc):
        if exc is None:
            return np.zeros(self._grid.shape)
        return np.asarray(exc, dtype=np.float64)

    def _modes_of(self, cells, exc):
        """(freqs, weights, sqrtlam, mode) of a figure under the current spectrum law."""
        rows, cols = self._grid.shape
        if self._spectrum() == SPEC_LAPLACE:
            if self._decay_law() != LAW_FIXED:
                # v6: the same call with its graph (the numbers never depend on the flag);
                # the participation of this figure reuses it instead of a second eigvalsh
                freqs, amps, graph = laplace_modes_of(cells, rows, cols, self.f0, self._settings(), exc,
                                                      with_graph=True)
                self._graph_hand = (np.ascontiguousarray(cells, dtype=np.int64).tobytes(), self._law_key(), graph)
            else:
                freqs, amps = laplace_modes_of(cells, rows, cols, self.f0, self._settings(), exc)
            return freqs, LAPLACE_GAIN * amps, np.zeros(len(freqs)), SPEC_LAPLACE
        sq = self.cache.get(cells, rows, cols)
        n = len(sq)
        w = np.full(n, 1.0 / n) if n else np.zeros(0)
        return self._scale() * sq, w, sq, SPEC_FIGURE

    # -- slots ------------------------------------------------------------------------
    def _free_slot(self, lo, hi):
        for s in range(lo, hi):
            if self.role[s] == ROLE_FREE:
                return s
        return -1

    def _clear_slot(self, s):
        self.role[s] = ROLE_FREE
        self.slot_id[s] = 0
        self.smode[s] = 0
        self.ndrive[s] = 0
        self.npulse[s] = 0
        self.nlive[s] = 0
        self.zre[s] = 0.0
        self.zim[s] = 0.0
        self.sqrtlam[s] = 0.0
        self.ffreq[s] = 0.0
        self.cth[s] = 1.0
        self.sth[s] = 0.0
        self.wcur[s] = 0.0
        self.winc[s] = 0.0
        self.wtgt[s] = 0.0
        self.wleft[s] = 0
        self._zero_pulse(s)
        self.pan[s] = 0.0
        self.pleft[s] = 0
        self.level[s] = 0.0
        self.last_e[s] = 0.0
        self.last_a[s] = 0.0
        self.last_b[s] = 0.0
        self.slaw[s] = 0
        self.gam_a[s] = 0.0
        self.gam_t[s] = 0.0
        for a in self.extra_slot:
            a[s] = 0.0

    def _zero_pulse(self, s):
        """No excitation left in slot s: the slot and the per-mode pulse states."""
        self.zf[s] = 0.0
        self.zs[s] = 0.0
        self.zu[s] = 0.0
        self.zfm[s] = 0.0
        self.zsm[s] = 0.0
        self.zum[s] = 0.0
        self.npulse[s] = 0

    def _pulse_quiet(self, s):
        """True when every pulse state of slot s is below TAIL_FLOOR (the last
        packet of a tail has finished)."""
        nd = int(self.npulse[s])
        return (max(abs(float(self.zf[s])), abs(float(self.zs[s])), abs(float(self.zu[s]))) < TAIL_FLOOR
                and (nd == 0 or float(np.max(np.abs(self.zfm[s, :nd]))) < TAIL_FLOOR
                     and float(np.max(np.abs(self.zsm[s, :nd]))) < TAIL_FLOOR
                     and float(np.max(np.abs(self.zum[s, :nd]))) < TAIL_FLOOR))

    _SLOT_FIELDS = ('role', 'slot_id', 'smode', 'ndrive', 'npulse', 'nlive', 'zre', 'zim', 'sqrtlam', 'ffreq',
                    'cth', 'sth', 'wcur', 'winc', 'wtgt', 'wleft', 'zf', 'zs', 'zu', 'zfm', 'zsm', 'zum', 'pan',
                    'pleft', 'level', 'last_e', 'last_a', 'last_b', 'slaw', 'gam_a', 'gam_t')

    def _move_slot(self, src, dst):
        for name in self._SLOT_FIELDS:
            a = getattr(self, name)
            a[dst] = a[src]
        for a in self.extra_slot:
            a[dst] = a[src]
        self._clear_slot(src)

    def _energy(self, s):
        nl = int(self.nlive[s])
        if nl == 0:
            return 0.0
        return float(np.max(np.hypot(self.zre[s, :nl], self.zim[s, :nl])))

    def _zero_modes(self, s, idx):
        self.zre[s, idx] = 0.0
        self.zim[s, idx] = 0.0
        self.wcur[s, idx] = 0.0
        self.winc[s, idx] = 0.0
        self.wtgt[s, idx] = 0.0
        self.zfm[s, idx] = 0.0
        self.zsm[s, idx] = 0.0
        self.zum[s, idx] = 0.0
        self.last_b[s, idx] = 0.0
        # the loss rates of a zeroed mode are kept: a quiet mode inside the live range of
        # an adaptive slot still runs the kernel (on a zero state) and a rate stays a rate

    def _release_quiet(self):
        """Block boundary: silent tails freed, finished fades freed, quiet / faded
        undriven modes zeroed, n_live recomputed; a tail whose last packet has
        decayed stops feeding its modes (npulse -> 0)."""
        for s in range(N_SLOTS):
            role = self.role[s]
            if role == ROLE_FREE:
                continue
            nd = int(self.ndrive[s])
            nl = int(self.nlive[s])
            if role != ROLE_ACTIVE and self.npulse[s] > 0 and self._pulse_quiet(s):
                self._zero_pulse(s)
            if nl > nd:
                mag = np.hypot(self.zre[s, nd:nl], self.zim[s, nd:nl])
                quiet = mag < TAIL_FLOOR
                if self.wleft[s] == 0:
                    quiet |= (self.wcur[s, nd:nl] == 0.0) & (self.wtgt[s, nd:nl] == 0.0)   # faded in place
                if quiet.any():
                    self._zero_modes(s, np.nonzero(quiet)[0] + nd)
                    alive = np.nonzero((self.zre[s, :nl] != 0.0) | (self.zim[s, :nl] != 0.0))[0]
                    nl = max(nd, int(alive.max()) + 1 if alive.size else 0)
                    self.nlive[s] = nl
            if role == ROLE_TAIL and nl == 0:
                self._clear_slot(s)
            elif role == ROLE_FADING and self.wleft[s] == 0:
                self._clear_slot(s)

    def _fade_slot(self, s):
        """Weights of slot s -> 0 over 20 ms (the slot is freed when the ramp ends)."""
        self.role[s] = ROLE_FADING
        self.ndrive[s] = 0
        self.slot_id[s] = 0
        self.wtgt[s, :] = 0.0
        if self.ramp_n > 0:
            self.winc[s, :] = (self.wtgt[s] - self.wcur[s]) / self.ramp_n
            self.wleft[s] = self.ramp_n
        else:                                                    # pragma: no cover
            self.wcur[s, :] = 0.0
            self.winc[s, :] = 0.0
            self.wleft[s] = 0

    def _tail_slot(self):
        """A tail slot for a new tail: a free one, else the quietest tail moves to a
        free fading slot (20 ms fade, counted) and its slot is returned; -1 when
        every tail and fading slot is busy (the caller fades in place)."""
        dst = self._free_slot(N_ACTIVE, N_ACTIVE + N_TAILS)
        if dst >= 0:
            return dst
        fdst = self._free_slot(N_ACTIVE + N_TAILS, N_SLOTS)
        if fdst < 0:
            return -1
        q = min(range(N_ACTIVE, N_ACTIVE + N_TAILS), key=lambda t: (self._energy(t), t))
        self._move_slot(q, fdst)
        self._fade_slot(fdst)
        self.counters[C_EVICT] += 1
        return q

    def _to_tail(self, s, last_e=0.0):
        """Active slot s -> a tail slot (the figure left / lost its identity); with
        every tail / fading slot busy the bank fades out in place (counted).
        `last_e` > 0 (Deaths): the bank takes ONE last uniform packet of the
        deaths inside its previous mask with it -- its pulse states and npulse
        stay, the strike is added now, and the tail never gets another packet."""
        if self.nlive[s] == 0:
            self._clear_slot(s)
            return
        self.slot_id[s] = 0
        if last_e > 0.0:
            self._strike(s, last_e)                               # while still active: every driven mode
            self.npulse[s] = self.ndrive[s]                       # ... and the tail keeps feeding them
            self.ndrive[s] = 0
        else:
            self.ndrive[s] = 0
            self._zero_pulse(s)                                   # a tail gets no excitation
            self.last_e[s] = 0.0
            self.last_a[s] = 0.0
            self.last_b[s] = 0.0
        dst = self._tail_slot()
        if dst < 0:
            self._fade_slot(s)
            self.counters[C_INPLACE] += 1
            return
        self._move_slot(s, dst)
        self.role[dst] = ROLE_TAIL

    def _split_tail(self, s, lo, hi):
        """Modes lo..hi-1 of active slot s leave its bank: audible ones ring on in a
        tail slot of their own (states, frequencies, weights, panning copied, no
        pulse); without a tail slot they fade in place (undriven, weight -> 0)."""
        idx = np.arange(lo, hi)
        mag = np.hypot(self.zre[s, idx], self.zim[s, idx])
        keep = idx[mag >= TAIL_FLOOR]
        if keep.size == 0:
            self._zero_modes(s, idx)
            return False
        dst = self._tail_slot()
        if dst < 0:
            return True                                          # the caller fades them in place
        m = keep.size
        self.role[dst] = ROLE_TAIL
        self.smode[dst] = self.smode[s]
        self.ndrive[dst] = 0
        self.nlive[dst] = m
        for name in ('zre', 'zim', 'sqrtlam', 'ffreq', 'cth', 'sth', 'wcur', 'winc', 'wtgt', 'gam_a', 'gam_t'):
            a = getattr(self, name)
            a[dst, :m] = a[s, keep]
        self.slaw[dst] = self.slaw[s]                            # the tail keeps its law and holds its targets
        for a in self.extra_slot:
            a[dst] = a[s]                                        # ... and an attached voicing its phase
        self.wleft[dst] = self.wleft[s]
        self.pan[dst] = self.pan[s]
        self.pleft[dst] = self.pleft[s]
        self._zero_modes(s, idx)
        return False

    # -- tuning -----------------------------------------------------------------------
    def _trim_live(self, s, n):
        """n_live of slot s = the extent of its non-zero states, never below n."""
        nl = int(self.nlive[s])
        alive = np.nonzero((self.zre[s, n:nl] != 0.0) | (self.zim[s, n:nl] != 0.0))[0]
        self.nlive[s] = n + int(alive.max()) + 1 if alive.size else n

    def _tune_slot(self, s, freqs, weights, sq, mode, ramp):
        """Mode j -> mode j: the driven modes get `freqs` / `weights` (weights ramped
        when the slot already sounds; new modes start from zero), vanished modes
        leave the bank (_split_tail: a tail of their own, or fading in place) -- an
        active slot never drives a mode's old state again."""
        n = int(len(freqs))
        n_old = int(self.ndrive[s])
        nl = int(self.nlive[s])
        if n > n_old and nl > n_old:
            # modes still fading in place from an earlier shrink (every pool was busy)
            # stand where new modes must start: out to a tail now, or cut (counted)
            if self._split_tail(s, n_old, nl):
                self._zero_modes(s, np.arange(n_old, nl))
                self.counters[C_DROP] += 1
        if n < n_old:
            self._split_tail(s, n, n_old)
        self.smode[s] = int(mode)
        self.ndrive[s] = n
        self.zfm[s, n:] = 0.0                                     # only fed modes hold packet states
        self.zsm[s, n:] = 0.0
        self.zum[s, n:] = 0.0
        self.sqrtlam[s, :n] = sq
        self.ffreq[s, :n] = freqs              # the ANALYSED frequency of the mode
        c, sn = trig_of(freqs * self._transpose, self.sr)   # what it sounds at
        self.cth[s, :n] = c
        self.sth[s, :n] = sn
        if n > n_old:
            self.zre[s, n_old:n] = 0.0                            # new modes start from zero
            self.zim[s, n_old:n] = 0.0
        self.nlive[s] = max(n, nl)
        self._trim_live(s, n)
        w = np.zeros(N_BANK)
        w[:n] = weights                                           # undriven modes left here fade
        if ramp and self.ramp_n > 0:
            self.wtgt[s] = w
            self.winc[s] = (w - self.wcur[s]) / self.ramp_n
            self.wleft[s] = self.ramp_n
        else:
            self.wcur[s] = w
            self.wtgt[s] = w
            self.winc[s] = 0.0
            self.wleft[s] = 0

    def _tune_figure(self, f, exc, ramp):
        freqs, w, sq, mode = self._modes_of(f.cells, exc)
        s = f.slot
        n_old = int(self.ndrive[s])
        self._tune_slot(s, freqs, w, sq, mode, ramp)
        if self._decay_law() != LAW_FIXED:
            # an adaptive law: the bank's targets from the current history; a NEW slot
            # (slaw 0 -> 1) or the NEW modes of a continuing bank start at their target
            self._target_figure(f, fresh_from=(n_old if self.slaw[s] == 1 else 0))
        elif self.slaw[s] == 1 and int(self.ndrive[s]) > n_old:
            # a bank still converging to Fixed grew: its new modes are fixed from the start
            g = gamma_of_r(self.rr[R_TGT], self.sr)
            self.gam_a[s, n_old:int(self.ndrive[s])] = g
            self.gam_t[s, n_old:int(self.ndrive[s])] = g

    # -- the age-driven losses (v6) ------------------------------------------------------
    def _decay_law(self):
        return int(self.params.get('decay_law', OPTIONAL_PARAMS['decay_law']))

    def _law_key(self):
        st = self._settings()
        return (int(st['n']), float(st['spread']), float(st['alpha']), float(st['shape']), float(st['harm']),
                int(st['fullshape']), float(st['dyn']), self.f0)

    def _participation(self, f):
        """The cached participation matrix of figure f under the current Laplace
        settings -- dict(key, P (m, N), r, c) with the node cells (r, c) in the node
        order of P; recomputed only when the cells or the settings changed."""
        rows, cols = self._grid.shape
        law_key = self._law_key()
        key = (f.cells.tobytes(), law_key)
        cur = self.part.get(f.id)
        if cur is not None and cur['key'] == key:
            return cur
        canon, order = fg.canonical_placement(f.cells, rows, cols)
        skey = (int(rows), int(cols), canon.tobytes(), law_key)
        P = self.pshape.get(skey)
        if P is None:
            hand = self._graph_hand
            if hand is not None and hand[0] == key[0] and hand[1] == law_key:
                graph = hand[2]
            else:
                _freqs, _amps, graph = laplace_modes_of(f.cells, rows, cols, self.f0, self._settings(), self.E_prev,
                                                        with_graph=True)
            P = mode_participation(graph['L'], graph['idx'])
            if len(self.pshape) >= SHAPE_CACHE_MAX:
                self.pshape.clear()                  # a cache: clearing never changes a result
            self.pshape[skey] = P
        cur = dict(key=key, P=P, r=f.cells[order, 0].copy(), c=f.cells[order, 1].copy())
        self.part[f.id] = cur
        return cur

    def _target_figure(self, f, fresh_from=None):
        """The loss targets of the bank of figure f from its history (q -> T -> gamma,
        Common: the mean rate); `fresh_from` = the first mode whose applied gamma
        starts AT the target (None: every applied value is kept)."""
        s = f.slot
        if s < 0 or self.smode[s] != SPEC_LAPLACE:
            return
        part = self._participation(f)
        m = int(part['P'].shape[0])
        if m != int(self.ndrive[s]):                              # pragma: no cover  (same law, same graph)
            raise RuntimeError(f"engine {ENGINE_ID}: participation modes {m} != bank {int(self.ndrive[s])}")
        if m == 0:
            self.slaw[s] = 1
            return
        _q, _T, gamma = age_targets(part['P'], self.age[part['r'], part['c']], float(self.params['decay_s']))
        if self._decay_law() == LAW_COMMON:
            gamma = np.full(m, float(gamma.mean()))
        if self.slaw[s] == 0:
            # the slot enters the adaptive mechanism now (a new bank): every mode of it
            # starts at its target
            self.slaw[s] = 1
            fresh_from = 0
        self.gam_t[s, :m] = gamma
        if fresh_from is not None and fresh_from < m:
            self.gam_a[s, fresh_from:m] = gamma[fresh_from:]

    def _refresh_targets(self):
        """Block boundary under an adaptive law: the targets of every sounding bank
        from the current history (the applied values are kept)."""
        for f in self.figures.values():
            if f.slot >= 0:
                self._target_figure(f)

    def _settle_fixed(self):
        """Block boundary under Fixed: banks still converging from an adaptive history
        follow the fixed target (Decay changes included) and return to the global r
        once settled and the r ramp is done; tails hold what they carry."""
        g = gamma_of_r(self.rr[R_TGT], self.sr)
        for s in range(N_ACTIVE):
            if self.role[s] != ROLE_ACTIVE or self.slaw[s] != 1:
                continue
            nl = int(self.nlive[s])
            self.gam_t[s, :nl] = g
            if int(self.ints[I_R_LEFT]) == 0 and bool(np.all(np.abs(self.gam_a[s, :nl] - g) <= GAMMA_SETTLE * g)):
                self.slaw[s] = 0
                self.gam_a[s] = 0.0
                self.gam_t[s] = 0.0

    def _enter_adaptive(self):
        """The law left Fixed: every slot in use takes gamma states of its own, starting
        from the gamma of the CURRENT global r.  An active bank gets its real targets at
        the boundary before its next sample; a tail / fading slot of the Fixed era holds
        the Decay of the moment of the switch (the target of a running r ramp) -- a
        separated tail ends by the law it left with, later Decay moves drive the active
        banks only."""
        g_cur = gamma_of_r(self.rr[R_CUR], self.sr)
        g_tgt = gamma_of_r(self.rr[R_TGT], self.sr)
        for s in range(N_SLOTS):
            if self.role[s] == ROLE_FREE or self.slaw[s] != 0:
                continue
            nl = int(self.nlive[s])
            self.slaw[s] = 1
            self.gam_a[s, :nl] = g_cur
            self.gam_t[s, :nl] = g_cur if self.role[s] == ROLE_ACTIVE else g_tgt

    def _leave_adaptive(self):
        """The law became Fixed: every active bank aims at the fixed gamma (the
        smoothing finishes, _settle_fixed hands the bank back to the global r)."""
        g = gamma_of_r(self.rr[R_TGT], self.sr)
        for s in range(N_ACTIVE):
            if self.role[s] == ROLE_ACTIVE and self.slaw[s] == 1:
                self.gam_t[s, :int(self.nlive[s])] = g

    def _retune_all(self, exc):
        """Spectrum law / Laplace settings / excitation changed on the same field:
        every sounding figure follows (frequencies at once, weights over 20 ms,
        states kept, no packet)."""
        for f in self.figures.values():
            if f.slot >= 0:
                self._tune_figure(f, exc, ramp=True)

    def _rescale_figure_slots(self):
        """frequency_scale changed: every Figure-tuned slot in use follows (states kept)."""
        scale = self._scale()
        for s in range(N_SLOTS):
            nl = int(self.nlive[s])
            if self.role[s] == ROLE_FREE or nl == 0 or self.smode[s] != SPEC_FIGURE:
                continue
            self.ffreq[s, :nl] = scale * self.sqrtlam[s, :nl]
            c, sn = trig_of(self.ffreq[s, :nl], self.sr)
            self.cth[s, :nl] = c
            self.sth[s, :nl] = sn

    def _set_pan(self, s, cx, ramp):
        cols = self._grid.shape[1]
        L, R = pan_of(cx, cols)
        if ramp and self.ramp_n > 0:
            self.pan[s, P_LTGT] = L
            self.pan[s, P_RTGT] = R
            self.pan[s, P_LINC] = (L - self.pan[s, P_LCUR]) / self.ramp_n
            self.pan[s, P_RINC] = (R - self.pan[s, P_RCUR]) / self.ramp_n
            self.pleft[s] = self.ramp_n
        else:
            self.pan[s] = (L, 0.0, L, R, 0.0, R)
            self.pleft[s] = 0

    def _fed(self, s):
        """The number of modes the pulse of slot s reaches (kernel rule)."""
        return int(self.ndrive[s]) if self.role[s] == ROLE_ACTIVE else int(self.npulse[s])

    def _strike(self, s, e, b=None):
        """The packet of e events into slot s: uniform (b None: a into the slot's
        pulse states, every fed mode gets it) or distributed (b (m,): a b_j into the
        pulse states of mode j; an all-zero b is a packet nobody takes)."""
        a = e / (e + en.EVENT_SAT)
        self.last_e[s] = e
        self.last_a[s] = a
        self.last_b[s] = 0.0
        if b is None:
            self.zf[s] += a
            self.zs[s] += a
            self.last_b[s, :self._fed(s)] = 1.0
        else:
            m = len(b)
            self.zfm[s, :m] += a * b
            self.zsm[s, :m] += a * b
            self.last_b[s, :m] = b

    def _events(self):
        return int(self.params.get('events', OPTIONAL_PARAMS['events']))

    def _excitation(self):
        return int(self.params.get('excitation', OPTIONAL_PARAMS['excitation']))

    def _position_supported(self):
        """Birth position is defined for Events = Births, Own, Laplace, full = 1."""
        return position_supported(self.params)

    def _birth_strength(self):
        return float(self.params.get('birth_strength', OPTIONAL_PARAMS['birth_strength']))

    def _packet(self, f, born, e, exc):
        """The packet of e > 0 events into the slot of figure f: uniform, or (Birth
        position) distributed by the births `born` (bool field) inside the figure,
        reshaped by Birth strength (s = 0: the uniform path itself, s = 1: b as is)."""
        s = f.slot
        if self._excitation() != EXC_POSITION:
            self._strike(s, e)
            return
        if not self._position_supported():
            self.counters[C_UNSUPPORTED] += 1               # never a uniform strike instead
            self.last_e[s] = 0.0
            self.last_a[s] = 0.0
            self.last_b[s] = 0.0
            return
        strength = self._birth_strength()
        if strength == 0.0:                                 # v5: the Uniform path, no participation needed
            self._strike(s, e)
            return
        rows, cols = self._grid.shape
        _freqs, _amps, graph = laplace_modes_of(f.cells, rows, cols, self.f0, self._settings(),
                                                exc, with_graph=True)
        cells = f.cells
        born_nodes = np.asarray(born, bool)[cells[graph['order'], 0], cells[graph['order'], 1]]
        b, total = birth_position_weights(graph['L'], graph['idx'], born_nodes)
        if len(b) != int(self.ndrive[s]):                     # pragma: no cover  (same law, same L)
            raise RuntimeError(f"engine {ENGINE_ID}: Birth position modes {len(b)} != bank {int(self.ndrive[s])}")
        if total <= ZERO_PART:
            self.counters[C_ZEROPART] += 1                  # no packet at any s > 0 (b = 0, never blended)
        else:
            b = birth_strength_profile(b, strength)         # s = 1: b itself
        self._strike(s, e, b)

    # -- the block boundary -------------------------------------------------------------
    def _track(self, G_prev, G, exc):
        rows, cols = G.shape
        births = (G != 0) & (G_prev == 0)
        deaths = (G_prev != 0) & (G == 0)
        self.age[births] = 1.0                                    # v6: a born position is fresh ...
        self.age[deaths] = 0.0                                    # ... a dead one has no history
        events = self._events()
        comps = fg.components(G)
        old_ids = sorted(self.figures)
        old = [self.figures[i] for i in old_ids]
        continued, tails, new = fg.match([f.cells for f in old], [f.centre for f in old],
                                         comps, rows, cols)
        figures = {}
        # figures that lost their identity: their banks become tails (Deaths: with
        # one last packet of the deaths inside their previous mask)
        for p in tails:
            f = old[p]
            if f.slot >= 0:
                last = 0.0
                if events == EV_DEATHS:
                    last = float(np.count_nonzero(deaths & self._mask(f)))
                self._to_tail(f.slot, last)
        # continuing figures: geometry, packet, retune
        for c in sorted(continued):
            f = old[continued[c]]
            cells = comps[c]
            K_prev = self._mask(f)
            centre = fg.centre_of(cells, rows, cols, f.centre)
            radius = fg.radius_of(cells, centre, rows, cols)
            nf = Figure(f.id, f.slot, cells, centre, radius)
            K_cur = self._mask(nf)
            e_b = int(np.count_nonzero(births & K_cur))
            e_d = int(np.count_nonzero(deaths & K_prev))
            e = float(e_b + e_d if events == EV_BOTH else e_b if events == EV_BIRTHS else e_d)
            if nf.slot >= 0:
                self._tune_figure(nf, exc, ramp=True)
                self._set_pan(nf.slot, centre[1], ramp=True)
                if e > 0.0:
                    self._packet(nf, births & K_cur, e, exc)
                else:
                    self.last_e[nf.slot] = 0.0
                    self.last_a[nf.slot] = 0.0
                    self.last_b[nf.slot] = 0.0
            figures[nf.id] = nf
        # new figures: identities in a deterministic order, slots largest first
        new_figs = []
        for c in new:
            cells = comps[c]
            centre = fg.centre_of(cells, rows, cols, None)
            radius = fg.radius_of(cells, centre, rows, cols)
            new_figs.append(Figure(0, -1, cells, centre, radius))
        new_figs.sort(key=lambda f: (int(f.cells[0, 0]), int(f.cells[0, 1])))
        for f in new_figs:
            f.id = self.next_id
            self.next_id += 1
            figures[f.id] = f
        self.figures = figures
        for fid in [k for k in self.part if k not in figures]:
            del self.part[fid]
        self._allocate(exc)
        if events != EV_DEATHS:                        # Deaths: the appearance of a bank is no event
            for f in new_figs:
                if f.slot >= 0:
                    K_cur = self._mask(f)
                    e = float(np.count_nonzero(births & K_cur))
                    if e > 0.0:
                        self._packet(f, births & K_cur, e, exc)
        self.counters[C_CHANGES] += 1

    def _waiting(self):
        return any(f.slot < 0 and len(f.cells) >= 2 for f in self.figures.values())

    def _allocate(self, exc):
        """Free active slots to tracked figures without one: largest area first
        (ties: first cell); figures of one cell never sound."""
        waiting = [f for f in self.figures.values() if f.slot < 0 and len(f.cells) >= 2]
        waiting.sort(key=lambda f: (-len(f.cells), int(f.cells[0, 0]), int(f.cells[0, 1])))
        for f in waiting:
            s = self._free_slot(0, N_ACTIVE)
            if s < 0:
                break
            self.role[s] = ROLE_ACTIVE
            self.slot_id[s] = f.id
            f.slot = s
            self._tune_figure(f, exc, ramp=False)
            self._set_pan(s, f.centre[1], ramp=False)

    def _boundary(self):
        """Everything of the block boundary: releases, then the field / excitation change."""
        self._release_quiet()
        if self.G_prev is None or self.G_prev.shape != self._grid.shape:
            self.G_prev = np.zeros_like(self._grid)
        if self.E_prev is None or self.E_prev.shape != self._grid.shape:
            self.E_prev = np.zeros(self._grid.shape)
        E = self._exc_array(self._exc)
        if not np.array_equal(self.G_prev, self._grid):
            self._track(self.G_prev, self._grid, E)
            self.G_prev = self._grid.copy()
            self.E_prev = E.copy()
        else:
            if not np.array_equal(self.E_prev, E):
                if self._laplace_uses_exc():
                    self._retune_all(E)                  # new events on the same field: weights only
                self.E_prev = E.copy()
            if self.figures and self._waiting():
                self._allocate(self.E_prev)              # a slot may have been freed
        if self._waiting():
            self.counters[C_UNVOICED] += 1
        if self._decay_law() != LAW_FIXED:
            self._refresh_targets()                      # v6: targets from the current history, every block
        else:
            self._settle_fixed()

    def _age_block(self):
        """After every rendered block: every position forgets a little (audio clock)."""
        self.age *= self.age_k

    # -- SoundEngine ----------------------------------------------------------------
    def init(self, grid, exc, gain=0.0):
        self._zero()
        self._grid = np.array(grid, np.uint8, copy=True)
        self._exc = None if exc is None else np.array(exc, np.float64, copy=True)
        self.age = np.zeros(self._grid.shape)            # the first boundary births every live cell
        g = float(gain)
        self.gg[:] = (g, g, 0.0)

    def prime_silent(self):
        """The current field becomes this engine's OWN previous field, so the
        first boundary finds no change and makes no packet.

        A fresh init compares the field with zeros and every live cell is a birth
        -- which is right when a note starts, and wrong when a host only switches
        an articulation ON over a field that has been running (REQ
        memory/req-unified-laplace-2026-09-21.md section 6: switching an axis is
        never an impulse).  The ages of the live positions are set as that first
        boundary would have set them, so an adaptive law continues too."""
        if self._grid is None:
            return
        self.G_prev = self._grid.copy()
        self.E_prev = self._exc_array(self._exc).copy()
        self.age[self._grid != 0] = 1.0

    def update_field(self, grid, exc):
        self._grid = np.array(grid, np.uint8, copy=True)
        self._exc = None if exc is None else np.array(exc, np.float64, copy=True)

    def set_params(self, params):
        old = dict(self.params)
        super().set_params(params)
        for k, v in OPTIONAL_PARAMS.items():
            self.params.setdefault(k, v)
        if not valid_birth_strength(self.params['birth_strength']):
            self.params = old
            raise ValueError(f"engine {ENGINE_ID}: birth_strength must be a finite number in "
                             f"{list(BIRTH_STRENGTH_RANGE)}, got {params.get('birth_strength')!r}")
        try:
            validate_params(self.params)                 # v6: an adaptive law needs Laplace / full
        except ValueError:
            self.params = old
            raise
        if self._grid is None:
            return
        if float(self.params['frequency_scale']) != float(old['frequency_scale']):
            self._rescale_figure_slots()
        law_old = int(old.get('decay_law', OPTIONAL_PARAMS['decay_law']))
        law = self._decay_law()
        if law != law_old:
            # a law change: no packet, no reset -- a continuing bank starts its transition
            # from its actual applied gamma (Common <-> Modal: the boundary retargets)
            if law_old == LAW_FIXED:
                self._enter_adaptive()
            elif law == LAW_FIXED:
                self._leave_adaptive()
        spectral = any(self.params[k] != old.get(k, OPTIONAL_PARAMS[k]) for k in ('spectrum',) + SPECTRUM_KEYS)
        if spectral and self.figures:
            self._retune_all(self.E_prev)
        if float(self.params['decay_s']) != float(old['decay_s']):
            tgt = decay_r(self.params['decay_s'], self.sr)
            self.rr[R_TGT] = tgt
            if self.ramp_n > 0:
                self.rr[R_INC] = (tgt - self.rr[R_CUR]) / self.ramp_n
                self.ints[I_R_LEFT] = self.ramp_n
            else:
                self.rr[R_CUR] = tgt
                self.rr[R_INC] = 0.0
                self.ints[I_R_LEFT] = 0
        if float(self.params['attack_ms']) != float(old.get('attack_ms', OPTIONAL_PARAMS['attack_ms'])):
            tgt = attack_q(self.params['attack_ms'], self.sr)
            self.qq[R_TGT] = tgt
            if self.ramp_n > 0:
                self.qq[R_INC] = (tgt - self.qq[R_CUR]) / self.ramp_n
                self.ints[I_Q_LEFT] = self.ramp_n
            else:
                self.qq[R_CUR] = tgt
                self.qq[R_INC] = 0.0
                self.ints[I_Q_LEFT] = 0

    def _set_gain(self, gain):
        g = float(gain)
        if g != self.gg[R_TGT]:
            self.gg[R_TGT] = g
            if self.ramp_n > 0:
                self.gg[R_INC] = (g - self.gg[R_CUR]) / self.ramp_n
                self.ints[I_G_LEFT] = self.ramp_n
            else:
                self.gg[R_CUR] = g
                self.gg[R_INC] = 0.0
                self.ints[I_G_LEFT] = 0

    SUPPORTS_TRANSPOSE = True      # the note is a scalar on every mode

    def _retune(self, transpose):
        """A new note: every mode's resonator is re-derived at the pitch that now
        sounds.  The complex state z of each mode keeps ringing untouched, so the
        pitch moves without a click and nothing restarts -- exactly what the
        engine already does when the field retunes a figure."""
        self._transpose = float(transpose)
        for s in range(N_SLOTS):
            n = int(self.nlive[s])
            if n <= 0:
                continue
            c, sn = trig_of(self.ffreq[s, :n] * self._transpose, self.sr)
            self.cth[s, :n] = c
            self.sth[s, :n] = sn

    def render(self, gain, t_samples, *, gain_prev=None, transpose=1.0):
        if transpose != self._transpose:
            self._retune(transpose)
        if gain_prev is not None:
            g0 = float(gain_prev)                  # the host overrides the glide start
            self.gg[R_CUR] = self.gg[R_TGT] = g0
            self.gg[R_INC] = 0.0
            self.ints[I_G_LEFT] = 0
        y, peak, n_clip = self.render_float(gain)
        pcm = (np.clip(y, -1.0, 1.0) * 32767).astype(np.int16)
        return np.ascontiguousarray(pcm), peak, n_clip

    def begin_block(self, gain, transpose=1.0, gain_prev=None):
        """Everything render() does BEFORE the samples -- the retune of a new
        note, the host's override of the gain-glide start, the block boundary and
        the gain target -- for a host that then runs a sample kernel of its own
        on this state (the voicings of the unified engine, REQ
        memory/req-unified-laplace-2026-09-21.md section 6).  The call order is
        render()'s own, so a cell that ends with the kernel of _kernel() is this
        engine bit for bit."""
        if transpose != self._transpose:
            self._retune(transpose)
        if gain_prev is not None:
            g0 = float(gain_prev)
            self.gg[R_CUR] = self.gg[R_TGT] = g0
            self.gg[R_INC] = 0.0
            self.ints[I_G_LEFT] = 0
        self._boundary()
        self._set_gain(gain)

    def end_block(self):
        """... and what it does AFTER them: every position forgets a little."""
        self._age_block()

    def render_float(self, gain):
        """One block (block, 2) after OUT_SCALE and the ramped bench gain, before the
        int16 clip, + pre-clip peak / clipped sample count."""
        self._boundary()
        self._set_gain(gain)
        out = self._out
        self._kernel(out.shape[0], out)
        self._age_block()
        a = np.abs(out)
        peak = float(a.max()) if a.size else 0.0
        return out.copy(), peak, int(np.count_nonzero(a > 1.0))

    def raw_block(self, events=True):
        """One block of the formulas (verification helper; the gain ramp is left
        as it is -- set gain through render_float / test hooks)."""
        if events:
            self._boundary()
        out = np.zeros_like(self._out)
        self._kernel(out.shape[0], out)
        self._age_block()
        return out

    def inject(self, slot, a):
        """Test hook: add a packet value to the pulse states of a slot (no field)."""
        self.zf[slot] += float(a)
        self.zs[slot] += float(a)

    def inject_modes(self, slot, deltas):
        """Test hook: add per-mode packet values delta_j to the pulse states of the
        first len(deltas) modes of a slot (no field)."""
        d = np.asarray(deltas, np.float64)
        self.zfm[slot, :len(d)] += d
        self.zsm[slot, :len(d)] += d

    def reset(self, gain=0.0):
        self.init(self._grid, self._exc, gain)

    def display(self):
        rows, cols = (self._grid.shape if self._grid is not None else (0, 0))
        det = int(self.params['detector'])
        spec = self._spectrum()
        ev, ex = self._events(), self._excitation()
        law = self._decay_law()
        figs = []
        for fid in sorted(self.figures):
            f = self.figures[fid]
            s = f.slot
            nd = int(self.ndrive[s]) if s >= 0 else 0
            r_eff = self._eff_radius(f)
            if s >= 0 and nd > 0 and self.slaw[s] == 1:
                gt = self.gam_t[s, :nd]
                ga = self.gam_a[s, :nd]
                t_lo, t_hi = float(LN1000 / gt.max()), float(LN1000 / gt.min())
                t_applied = float(LN1000 / ga.mean())
            else:
                t_lo = t_hi = t_applied = (float(LN1000 / gamma_of_r(self.rr[R_CUR], self.sr)) if s >= 0 else 0.0)
            figs.append(dict(id=fid, color=(fid - 1) % N_PALETTE, slot=s, n=int(len(f.cells)),
                             t_lo=t_lo, t_hi=t_hi, t_applied=t_applied,
                             adaptive=bool(s >= 0 and self.slaw[s] == 1),
                             cells=f.cells.tolist(), centre=[f.centre[0], f.centre[1]],
                             radius=r_eff, radius_geom=f.radius,
                             covers_all=bool(det == DET_DISK and fg.covers_field(r_eff, rows, cols)),
                             modes=nd,
                             f_low=(float(self.ffreq[s, 0]) if s >= 0 and nd > 0 else 0.0),
                             w_max=(float(self.wtgt[s, :nd].max()) if s >= 0 and nd > 0 else 0.0),
                             e=(float(self.last_e[s]) if s >= 0 else 0.0),
                             a=(float(self.last_a[s]) if s >= 0 else 0.0),
                             b=(self.last_b[s, :nd].tolist() if s >= 0 else []),
                             level=(float(self.level[s]) if s >= 0 else 0.0)))
        n_tail = int(np.count_nonzero(self.role == ROLE_TAIL))
        n_fade = int(np.count_nonzero(self.role == ROLE_FADING))
        n_tail_fed = int(np.count_nonzero((self.role != ROLE_ACTIVE) & (self.role != ROLE_FREE) & (self.npulse > 0)))
        return dict(figures=figs, n_figures=len(figs),
                    n_sounding=sum(1 for f in figs if f['slot'] >= 0),
                    n_single=sum(1 for f in figs if f['n'] < 2),
                    n_tails=n_tail, n_fading=n_fade, n_tails_fed=n_tail_fed,
                    evictions=int(self.counters[C_EVICT]),
                    drops=int(self.counters[C_DROP]), unvoiced_blocks=int(self.counters[C_UNVOICED]),
                    changes=int(self.counters[C_CHANGES]), inplace_fades=int(self.counters[C_INPLACE]),
                    zero_participation=int(self.counters[C_ZEROPART]),
                    unsupported_packets=int(self.counters[C_UNSUPPORTED]),
                    detector=det, detector_name=CHOICES['detector'][det], rows=rows, cols=cols,
                    radius_mul=self._radius_mul(),
                    events=ev, events_name=CHOICES['events'][ev],
                    excitation=ex, excitation_name=CHOICES['excitation'][ex],
                    position_supported=bool(self._position_supported()),
                    position_condition=POSITION_CONDITION,
                    birth_strength=self._birth_strength(), birth_strength_hint=BIRTH_STRENGTH_HINT,
                    decay_law=law, decay_law_name=LAW_NAMES[law], law_supported=bool(decay_law_supported(self.params)),
                    law_condition=LAW_CONDITION, t_fresh_s=T_FRESH_S, age_tau_s=AGE_TAU_S,
                    gamma_smooth_s=GAMMA_SMOOTH_S,
                    n_adaptive=int(np.count_nonzero((self.role != ROLE_FREE) & (self.slaw == 1))),
                    spectrum=spec, spectrum_name=CHOICES['spectrum'][spec], f0=self.f0,
                    laplace=self._settings(),
                    frequency_scale=self._scale(), decay_s=float(self.params['decay_s']),
                    r=float(self.rr[R_CUR]), r_target=float(self.rr[R_TGT]),
                    ramp_left=int(self.ints[I_R_LEFT]), gain=float(self.gg[R_CUR]),
                    attack_ms=float(self.params['attack_ms']), q=float(self.qq[R_CUR]),
                    q_target=float(self.qq[R_TGT]), attack_ramp_left=int(self.ints[I_Q_LEFT]),
                    model=self.model_version)

    # -- snapshot ---------------------------------------------------------------------
    _ARRAYS = ('role', 'slot_id', 'smode', 'ndrive', 'npulse', 'nlive', 'zre', 'zim', 'sqrtlam', 'ffreq', 'wcur',
               'winc', 'wtgt', 'wleft', 'zf', 'zs', 'zu', 'zfm', 'zsm', 'zum', 'pan', 'pleft', 'rr', 'gg', 'qq',
               'ints', 'hp', 'level', 'last_e', 'last_a', 'last_b', 'counters', 'slaw', 'gam_a', 'gam_t')

    def _fresh_arrays(self):
        S, M = N_SLOTS, N_BANK
        return dict(role=np.zeros(S, np.int64), slot_id=np.zeros(S, np.int64),
                    smode=np.zeros(S, np.int64),
                    ndrive=np.zeros(S, np.int64), npulse=np.zeros(S, np.int64), nlive=np.zeros(S, np.int64),
                    zre=np.zeros((S, M)), zim=np.zeros((S, M)), sqrtlam=np.zeros((S, M)),
                    ffreq=np.zeros((S, M)), wcur=np.zeros((S, M)), winc=np.zeros((S, M)),
                    wtgt=np.zeros((S, M)), wleft=np.zeros(S, np.int64), zf=np.zeros(S),
                    zs=np.zeros(S), zu=np.zeros(S), zfm=np.zeros((S, M)), zsm=np.zeros((S, M)),
                    zum=np.zeros((S, M)), pan=np.zeros((S, 6)), pleft=np.zeros(S, np.int64),
                    rr=np.zeros(3), gg=np.zeros(3), qq=np.zeros(3), ints=np.zeros(4, np.int64),
                    hp=np.zeros((2, 2)),
                    level=np.zeros(S), last_e=np.zeros(S), last_a=np.zeros(S), last_b=np.zeros((S, M)),
                    counters=np.zeros(N_COUNTERS, np.int64),
                    slaw=np.zeros(S, np.int64), gam_a=np.zeros((S, M)), gam_t=np.zeros((S, M)))

    def export_state(self):
        if self._grid is None:
            raise ValueError(f"engine {ENGINE_ID}: no field yet (init first)")
        ids = sorted(self.figures)
        figs = [self.figures[i] for i in ids]
        offsets = np.zeros(len(figs) + 1, np.int64)
        for k, f in enumerate(figs):
            offsets[k + 1] = offsets[k] + len(f.cells)
        cells = (np.concatenate([f.cells for f in figs], axis=0) if figs
                 else np.zeros((0, 2), np.int64))
        # v6: the cached participation of the figures whose cache is current (row-major
        # (m, N) blocks; 0 modes = nothing cached for that figure)
        parts = []
        modes = np.zeros(len(figs), np.int64)
        poff = np.zeros(len(figs) + 1, np.int64)
        for k, f in enumerate(figs):
            cur = self.part.get(f.id)
            if cur is not None and cur['key'] == (f.cells.tobytes(), self._law_key()) and f.slot >= 0:
                parts.append(cur['P'].reshape(-1))
                modes[k] = cur['P'].shape[0]
            poff[k + 1] = poff[k] + (modes[k] * len(f.cells))
        st = dict(version=self.STATE_VERSION, engine_id=self.engine_id,
                  model_version=self.model_version, params=dict(self.params),
                  sr=int(self.sr), block=int(self._out.shape[0]), out_scale=OUT_SCALE,
                  laplace_gain=LAPLACE_GAIN, f0=self.f0,
                  slots=dict(active=N_ACTIVE, tails=N_TAILS, fading=N_FADING, bank=N_BANK),
                  next_id=int(self.next_id),
                  fig_ids=np.array(ids, np.int64),
                  fig_slots=np.array([f.slot for f in figs], np.int64),
                  fig_centres=np.array([f.centre for f in figs], np.float64).reshape(-1, 2),
                  fig_radii=np.array([f.radius for f in figs], np.float64),
                  fig_cells=np.ascontiguousarray(cells, dtype=np.int64),
                  fig_offsets=offsets,
                  fig_part=(np.concatenate(parts) if parts else np.zeros(0)),
                  fig_part_modes=modes, fig_part_offsets=poff,
                  age=self.age.copy(),
                  law_key=list(self._law_key()),
                  grid_pending=self._grid.copy(),
                  grid_prev=(np.zeros_like(self._grid) if self.G_prev is None
                             else self.G_prev.copy()),
                  exc_pending=(None if self._exc is None else self._exc.copy()),
                  exc_prev=(np.zeros(self._grid.shape) if self.E_prev is None
                            else self.E_prev.copy()))
        for name in self._ARRAYS:
            st[name] = np.array(getattr(self, name), copy=True)
        return st

    def restore_state(self, grid, exc, state):
        version = state.get('version') if isinstance(state, dict) else None
        if version != self.STATE_VERSION and version not in COMPATIBLE_STATES:
            raise ValueError(f"engine {ENGINE_ID}: state version "
                             f"{state.get('version') if isinstance(state, dict) else state!r}"
                             f" != {self.STATE_VERSION}")
        if state.get('engine_id') != self.engine_id:
            raise ValueError(f"engine state is for {state.get('engine_id')!r}, not {self.engine_id!r}")
        want_model = self.model_version if version == self.STATE_VERSION else COMPATIBLE_STATES[version]
        if state.get('model_version') != want_model:
            raise ValueError(f"engine {ENGINE_ID}: model version {state.get('model_version')!r}"
                             f" != {want_model!r}")
        if version != self.STATE_VERSION:
            state = dict(state)
            S, M = N_SLOTS, N_BANK
            if version == 2:
                # an older state without Attack: the same sound with the attack state at zero
                state.setdefault('zu', np.zeros(S))
                state.setdefault('qq', np.zeros(3))
                ints = state.get('ints')
                if isinstance(ints, np.ndarray) and ints.shape == (3,):
                    state['ints'] = np.concatenate([ints, np.zeros(1, ints.dtype)])
            # a state without Events / Excitation (v2, v3): the same sound with the
            # per-mode packet states at zero and no fed tails
            for name in ('zfm', 'zsm', 'zum', 'last_b'):
                state.setdefault(name, np.zeros((S, M)))
            state.setdefault('npulse', np.zeros(S, np.int64))
            cnt = state.get('counters')
            if isinstance(cnt, np.ndarray) and cnt.shape == (5,):
                state['counters'] = np.concatenate([cnt, np.zeros(N_COUNTERS - 5, cnt.dtype)])
            # a state without the Decay law (v2 .. v5): Fixed, no gamma states, no cached
            # participation, and the past age unknown -- u = 1 on the live cells of its
            # previous field (not a recovery of the hidden history; the Fixed sound is the same)
            state.setdefault('slaw', np.zeros(S, np.int64))
            state.setdefault('gam_a', np.zeros((S, M)))
            state.setdefault('gam_t', np.zeros((S, M)))
            gv0 = state.get('grid_prev')
            if isinstance(gv0, np.ndarray):
                state.setdefault('age', (np.asarray(gv0) != 0).astype(np.float64))
            state.setdefault('fig_part', np.zeros(0))
            state.setdefault('fig_part_modes', None)
            state.setdefault('fig_part_offsets', None)
        if int(state.get('sr', -1)) != int(self.sr) or int(state.get('block', -1)) != self._out.shape[0]:
            raise ValueError(f"engine {ENGINE_ID}: state sr / block do not match the context")
        params = state.get('params')
        if isinstance(params, dict):
            params = dict(params)
            for k, v in OPTIONAL_PARAMS.items():       # older snapshots: the bit-exact default
                params.setdefault(k, v)
        if not isinstance(params, dict) or sorted(params) != sorted(p[0] for p in PARAMS):
            raise ValueError(f"engine {ENGINE_ID}: state params do not match the registry")
        if not valid_birth_strength(params['birth_strength']):
            raise ValueError(f"engine {ENGINE_ID}: birth_strength of the state must be a finite number in "
                             f"{list(BIRTH_STRENGTH_RANGE)}, got {params['birth_strength']!r}")
        if int(params['decay_law']) not in (LAW_FIXED, LAW_COMMON, LAW_MODAL):
            raise ValueError(f"engine {ENGINE_ID}: decay_law of the state is not 0 / 1 / 2")
        validate_params(params)                          # an adaptive law needs Laplace / full
        if float(state.get('out_scale', OUT_SCALE)) != OUT_SCALE:
            raise ValueError(f"engine {ENGINE_ID}: out_scale of the state is not {OUT_SCALE}")
        if float(state.get('laplace_gain', LAPLACE_GAIN)) != LAPLACE_GAIN:
            raise ValueError(f"engine {ENGINE_ID}: laplace_gain of the state is not {LAPLACE_GAIN}")
        if float(state.get('f0', self.f0)) != self.f0:
            raise ValueError(f"engine {ENGINE_ID}: f0 of the state {state.get('f0')} != {self.f0}")
        if state.get('slots') != dict(active=N_ACTIVE, tails=N_TAILS, fading=N_FADING, bank=N_BANK):
            raise ValueError(f"engine {ENGINE_ID}: slot layout of the state differs")
        fresh = self._fresh_arrays()
        arrays = {}
        for name, proto in fresh.items():
            a = state.get(name)
            if not isinstance(a, np.ndarray) or a.shape != proto.shape:
                raise ValueError(f"engine {ENGINE_ID}: state array {name!r} missing or "
                                 f"shape {getattr(a, 'shape', None)} != {proto.shape}")
            if not np.all(np.isfinite(a)):
                raise ValueError(f"engine {ENGINE_ID}: state array {name!r} not finite")
            arrays[name] = np.array(a, proto.dtype, copy=True)
        if (arrays['ints'] < 0).any() or (arrays['wleft'] < 0).any() or (arrays['pleft'] < 0).any():
            raise ValueError(f"engine {ENGINE_ID}: negative counters in the state")
        if (arrays['qq'][:2] < 0.0).any() or (arrays['qq'][:2] >= 1.0).any():
            raise ValueError(f"engine {ENGINE_ID}: attack coefficient of the state outside [0, 1)")
        if not np.all(np.isin(arrays['role'], (ROLE_FREE, ROLE_ACTIVE, ROLE_TAIL, ROLE_FADING))):
            raise ValueError(f"engine {ENGINE_ID}: unknown slot role in the state")
        if not np.all(np.isin(arrays['smode'], (SPEC_FIGURE, SPEC_LAPLACE))):
            raise ValueError(f"engine {ENGINE_ID}: unknown spectrum law of a slot in the state")
        if not np.all(np.isin(arrays['slaw'], (0, 1))):
            raise ValueError(f"engine {ENGINE_ID}: unknown loss law of a slot in the state")
        if (arrays['gam_a'] < 0.0).any() or (arrays['gam_t'] < 0.0).any():
            raise ValueError(f"engine {ENGINE_ID}: negative loss rates in the state")
        adaptive = arrays['slaw'] == 1
        if ((arrays['role'] == ROLE_FREE) & adaptive).any():
            raise ValueError(f"engine {ENGINE_ID}: a free slot of the state carries a loss law")
        for s in np.nonzero(adaptive)[0]:
            nl = int(arrays['nlive'][s])
            if nl and (not (arrays['gam_t'][s, :nl] > 0.0).all() or not (arrays['gam_a'][s, :nl] > 0.0).all()):
                raise ValueError(f"engine {ENGINE_ID}: loss rates of adaptive slot {int(s)} are not positive")
        if ((arrays['ndrive'] < 0).any() or (arrays['ndrive'] > N_BANK).any()
                or (arrays['nlive'] < arrays['ndrive']).any() or (arrays['nlive'] > N_BANK).any()):
            raise ValueError(f"engine {ENGINE_ID}: mode counts of the state out of range")
        tails = (arrays['role'] == ROLE_TAIL) | (arrays['role'] == ROLE_FADING)
        if ((arrays['npulse'] < 0).any() or (arrays['npulse'] > arrays['nlive']).any()
                or (arrays['npulse'][~tails] != 0).any()):
            raise ValueError(f"engine {ENGINE_ID}: fed mode counts of the tails in the state are inconsistent")
        g = np.asarray(grid, np.uint8)
        gp = state.get('grid_pending')
        gv = state.get('grid_prev')
        for name, arr in (('grid_pending', gp), ('grid_prev', gv)):
            if not isinstance(arr, np.ndarray) or arr.shape != g.shape:
                raise ValueError(f"engine {ENGINE_ID}: {name} missing or shape "
                                 f"{getattr(arr, 'shape', None)} != {g.shape}")
        if not np.array_equal(np.asarray(gp, np.uint8), g):
            raise ValueError(f"engine {ENGINE_ID}: the pending field of the state differs "
                             f"from the bench field")
        age = state.get('age')
        if not isinstance(age, np.ndarray) or age.shape != g.shape or not np.all(np.isfinite(age)):
            raise ValueError(f"engine {ENGINE_ID}: age missing / wrong shape / not finite")
        if (age < 0.0).any() or (age > 1.0).any():
            raise ValueError(f"engine {ENGINE_ID}: age of the state outside [0, 1]")
        if (age[np.asarray(gv, np.uint8) == 0] != 0.0).any():
            raise ValueError(f"engine {ENGINE_ID}: age of the state is non-zero on a dead cell")
        ep = state.get('exc_pending')
        ev = state.get('exc_prev')
        if ep is not None and (not isinstance(ep, np.ndarray) or ep.shape != g.shape):
            raise ValueError(f"engine {ENGINE_ID}: exc_pending of the state has the wrong shape")
        if not isinstance(ev, np.ndarray) or ev.shape != g.shape or not np.all(np.isfinite(ev)):
            raise ValueError(f"engine {ENGINE_ID}: exc_prev missing / wrong shape / not finite")
        if (exc is None) != (ep is None) or (exc is not None and not np.array_equal(
                np.asarray(exc, np.float64), np.asarray(ep, np.float64))):
            raise ValueError(f"engine {ENGINE_ID}: the pending excitation of the state differs "
                             f"from the bench excitation")
        rows, cols = g.shape
        # the tracker
        ids = state.get('fig_ids')
        slots = state.get('fig_slots')
        centres = state.get('fig_centres')
        radii = state.get('fig_radii')
        cells = state.get('fig_cells')
        offsets = state.get('fig_offsets')
        for name, arr in (('fig_ids', ids), ('fig_slots', slots), ('fig_centres', centres),
                          ('fig_radii', radii), ('fig_cells', cells), ('fig_offsets', offsets)):
            if not isinstance(arr, np.ndarray):
                raise ValueError(f"engine {ENGINE_ID}: {name} missing from the state")
        F = len(ids)
        if (slots.shape != (F,) or centres.shape != (F, 2) or radii.shape != (F,)
                or offsets.shape != (F + 1,) or cells.ndim != 2 or cells.shape[1] != 2
                or offsets[0] != 0 or offsets[-1] != len(cells) or (np.diff(offsets) < 1).any()):
            raise ValueError(f"engine {ENGINE_ID}: tracker arrays of the state are inconsistent")
        if F and (len(np.unique(ids)) != F or (ids < 1).any() or ids.max() >= int(state.get('next_id', 0))):
            raise ValueError(f"engine {ENGINE_ID}: figure ids of the state are inconsistent")
        Gp = np.asarray(gv, np.uint8)
        comps = fg.components(Gp)
        comp_keys = {c.tobytes(): k for k, c in enumerate(comps)}
        figures = {}
        seen = set()
        n_lap = int(params['n'])
        for k in range(F):
            cc = np.ascontiguousarray(cells[offsets[k]:offsets[k + 1]], dtype=np.int64)
            key = cc.tobytes()
            if key not in comp_keys or key in seen:
                raise ValueError(f"engine {ENGINE_ID}: figure {int(ids[k])} of the state is not a "
                                 f"component of its previous field")
            seen.add(key)
            s = int(slots[k])
            if s >= 0:
                if not (0 <= s < N_ACTIVE) or arrays['role'][s] != ROLE_ACTIVE or arrays['slot_id'][s] != ids[k]:
                    raise ValueError(f"engine {ENGINE_ID}: slot of figure {int(ids[k])} is inconsistent")
                nd = int(arrays['ndrive'][s])
                if arrays['smode'][s] == SPEC_FIGURE:
                    sq = fg.spectrum(cc, rows, cols, N_BANK)
                    if nd != len(sq) or not np.allclose(arrays['sqrtlam'][s, :nd], sq, rtol=1e-9, atol=1e-12):
                        raise ValueError(f"engine {ENGINE_ID}: the spectrum of figure {int(ids[k])} does "
                                         f"not follow from its cells")
                elif nd > n_lap or (nd > 0 and abs(float(arrays['ffreq'][s, 0]) - self.f0) > 1e-6 * self.f0):
                    raise ValueError(f"engine {ENGINE_ID}: the Laplace modes of figure {int(ids[k])} do "
                                     f"not follow from the settings (count / lowest mode f0)")
            figures[int(ids[k])] = Figure(int(ids[k]), s, cc, (float(centres[k, 0]), float(centres[k, 1])),
                                          float(radii[k]))
        if len(seen) != len(comps):
            raise ValueError(f"engine {ENGINE_ID}: the previous field has components the state "
                             f"does not track")
        # v6: the cached participation matrices (optional per figure; absent = recomputed
        # at the next boundary from the same cells / settings)
        pmodes = state.get('fig_part_modes')
        poff = state.get('fig_part_offsets')
        pdata = state.get('fig_part')
        part = {}
        if pmodes is not None or poff is not None:
            if (not isinstance(pmodes, np.ndarray) or pmodes.shape != (F,) or not isinstance(poff, np.ndarray)
                    or poff.shape != (F + 1,) or not isinstance(pdata, np.ndarray) or pdata.ndim != 1
                    or poff[0] != 0 or poff[-1] != len(pdata) or not np.all(np.isfinite(pdata))):
                raise ValueError(f"engine {ENGINE_ID}: cached participation of the state is inconsistent")
            law_key = self._law_key_of(params)
            for k in range(F):
                m = int(pmodes[k])
                if m == 0:
                    continue
                fig = figures[int(ids[k])]
                N = len(fig.cells)
                if fig.slot < 0 or m != int(arrays['ndrive'][fig.slot]) or poff[k + 1] - poff[k] != m * N:
                    raise ValueError(f"engine {ENGINE_ID}: cached participation of figure {int(ids[k])} does "
                                     f"not match its bank")
                P = np.array(pdata[poff[k]:poff[k + 1]], np.float64).reshape(m, N)
                if (P < -Q_TOL).any() or (P > 1.0 + Q_TOL).any() or not np.allclose(P.sum(axis=1), 1.0, atol=1e-9):
                    raise ValueError(f"engine {ENGINE_ID}: cached participation of figure {int(ids[k])} is not "
                                     f"a weight matrix")
                _canon, order = fg.canonical_placement(fig.cells, rows, cols)
                part[int(ids[k])] = dict(key=(fig.cells.tobytes(), law_key), P=P,
                                         r=fig.cells[order, 0].copy(), c=fig.cells[order, 1].copy())
        active_ids = set(int(v) for v in arrays['slot_id'][arrays['role'] == ROLE_ACTIVE])
        if active_ids != set(f.id for f in figures.values() if f.slot >= 0):
            raise ValueError(f"engine {ENGINE_ID}: active slots and figures of the state disagree")
        scale = float(params['frequency_scale'])
        for s in range(N_SLOTS):
            nl = int(arrays['nlive'][s])
            if arrays['role'][s] != ROLE_FREE and nl:
                if arrays['smode'][s] == SPEC_FIGURE:
                    if not np.array_equal(arrays['ffreq'][s, :nl], scale * arrays['sqrtlam'][s, :nl]):
                        raise ValueError(f"engine {ENGINE_ID}: frequencies of slot {s} do not follow "
                                         f"from its spectrum and the scale")
                elif not np.all(arrays['ffreq'][s, :nl] > 0.0):
                    raise ValueError(f"engine {ENGINE_ID}: Laplace frequencies of slot {s} are not positive")
        self.params = dict(params)
        self._zero()
        for name, a in arrays.items():
            setattr(self, name, a)
        self.cth = np.ones((N_SLOTS, N_BANK))
        self.sth = np.zeros((N_SLOTS, N_BANK))
        for s in range(N_SLOTS):
            nl = int(self.nlive[s])
            if self.role[s] != ROLE_FREE and nl:
                c, sn = trig_of(self.ffreq[s, :nl], self.sr)
                self.cth[s, :nl] = c
                self.sth[s, :nl] = sn
        self.figures = figures
        self.part = part
        self.age = np.array(age, np.float64, copy=True)
        self.next_id = int(state.get('next_id', 1))
        self._grid = np.array(g, np.uint8, copy=True)
        self._exc = None if exc is None else np.array(exc, np.float64, copy=True)
        self.G_prev = np.array(Gp, np.uint8, copy=True)
        self.E_prev = np.array(ev, np.float64, copy=True)

    def _law_key_of(self, params):
        st = laplace_settings(params)
        return (int(st['n']), float(st['spread']), float(st['alpha']), float(st['shape']), float(st['harm']),
                int(st['fullshape']), float(st['dyn']), self.f0)


def position_supported(params):
    """Birth position is defined for Events = Births, Own, Laplace, full = 1 (the
    engine makes no packet outside this combination; the bench shows the condition)."""
    return (int(params.get('events', EV_BOTH)) == EV_BIRTHS
            and int(params.get('detector', DET_DISK)) == DET_OWN
            and int(params.get('spectrum', SPEC_FIGURE)) == SPEC_LAPLACE
            and int(params.get('fullshape', 1)) == 1)


def inactive(params):
    """Settings that do not act for the current mode (bench: shown as text).  v6: the
    Decay law buttons are locked while the combination cannot take an adaptive law
    (the reason is the text); an adaptive law locks Spectrum and full in return, so
    the combination can never be left unnoticed -- set Fixed first."""
    out = {}
    if int(params.get('excitation', EXC_UNIFORM)) != EXC_POSITION:
        # stored, shown with its value, acts only with Birth position (v5)
        out['birth_strength'] = f"{float(params.get('birth_strength', 1.0)):.2f}  Birth position only"
    law = int(params.get('decay_law', LAW_FIXED))
    if law != LAW_FIXED:
        out['spectrum'] = "Laplace  held by Decay law"
        out['fullshape'] = "on  held by Decay law"
    elif not decay_law_supported(params):
        out['decay_law'] = f"Fixed  ({LAW_CONDITION})"
    if int(params.get('spectrum', SPEC_FIGURE)) == SPEC_LAPLACE:
        out['frequency_scale'] = 'Figure law only (f0 = scene)'
        if float(params.get('shape', 0.0)) <= 0.0:
            out['dyn'] = 'acts only with shape > 0'
    else:
        for k in SPECTRUM_KEYS:
            out.setdefault(k, 'Laplace only')
    return out


def overlay(params, rows, cols):
    """Static part of the overlay: the caption only (figures come from display())."""
    det = CHOICES['detector'][int(params.get('detector', DET_DISK))]
    spec = CHOICES['spectrum'][int(params.get('spectrum', SPEC_FIGURE))]
    ev = CHOICES['events'][int(params.get('events', EV_BOTH))]
    ex = CHOICES['excitation'][int(params.get('excitation', EXC_UNIFORM))]
    return dict(text=f"{OVERLAY_TEXT} [{det}, {spec}] [{ev}, {ex}]")


register(EngineSpec(ENGINE_ID, LABEL, PARAMS, lambda ctx, params: ObjectResonatorsEngine(ctx, params),
                    choices=CHOICES, inactive=inactive, overlay=overlay,
                    ranges={'radius_mul': RADIUS_RANGE}, validate=validate_params,
                    plays_notes=True))
