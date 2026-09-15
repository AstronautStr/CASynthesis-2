"""Java-compatible scalar/vector math helpers for the gutterOsc port.

* ``atan``: a port of fdlibm ``s_atan.c`` (what ``java.lang.Math.atan`` / ``StrictMath.atan``
  execute; HotSpot has no atan intrinsic).  The C runtime's ``math.atan`` differs from it in
  the last ulp for some inputs, which is enough to desynchronise a chaotic system.  Pure
  IEEE-754 double arithmetic -> bit-identical on any platform.
* ``sin``/``exp``: Java returns NaN/Infinity where Python raises; wrap them.
* ``f32``: Java ``(float)`` cast including overflow to +-Infinity and NaN pass-through.

Copyright notice for the fdlibm algorithm:
  Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
  Developed at SunSoft, a Sun Microsystems, Inc. business.  Permission to use, copy, modify,
  and distribute this software is freely granted, provided that this notice is preserved.
"""
import math
import struct

import numpy as np

_ATANHI = (4.63647609000806093515e-01, 7.85398163397448278999e-01, 9.82793723247329054082e-01, 1.57079632679489655800e+00)
_ATANLO = (2.26987774529616870924e-17, 3.06161699786838301793e-17, 1.39033110312309984516e-17, 6.12323399573676603587e-17)
_AT = (3.33333333333329318027e-01, -1.99999999998764832476e-01, 1.42857142725034663711e-01,
       -1.11111104054623557880e-01, 9.09088713343650656196e-02, -7.69187620504482999495e-02,
       6.66107313738753120669e-02, -5.83357013379057348645e-02, 4.97687799461593236017e-02,
       -3.65315727442169155270e-02, 1.62858201153657823623e-02)
_HUGE = 1.0e300


def _hi_lo(x):
    b = struct.pack(">d", x)
    return struct.unpack(">i", b[:4])[0], struct.unpack(">I", b[4:])[0]


def atan(x):
    """fdlibm s_atan.c, scalar."""
    hx, lx = _hi_lo(x)
    ix = hx & 0x7FFFFFFF
    if ix >= 0x44100000:                       # |x| >= 2^66
        if ix > 0x7FF00000 or (ix == 0x7FF00000 and lx != 0):
            return x + x                       # NaN
        return _ATANHI[3] + _ATANLO[3] if hx > 0 else -_ATANHI[3] - _ATANLO[3]
    if ix < 0x3FDC0000:                        # |x| < 0.4375
        if ix < 0x3E200000:                    # |x| < 2^-29
            if _HUGE + x > 1.0:
                return x
        idx = -1
    else:
        x = abs(x)
        if ix < 0x3FF30000:                    # |x| < 1.1875
            if ix < 0x3FE60000:                # 7/16 <= |x| < 11/16
                idx = 0
                x = (2.0 * x - 1.0) / (2.0 + x)
            else:                              # 11/16 <= |x| < 19/16
                idx = 1
                x = (x - 1.0) / (x + 1.0)
        else:
            if ix < 0x40038000:                # |x| < 2.4375
                idx = 2
                x = (x - 1.5) / (1.0 + 1.5 * x)
            else:
                idx = 3
                x = -1.0 / x
    z = x * x
    w = z * z
    s1 = z * (_AT[0] + w * (_AT[2] + w * (_AT[4] + w * (_AT[6] + w * (_AT[8] + w * _AT[10])))))
    s2 = w * (_AT[1] + w * (_AT[3] + w * (_AT[5] + w * (_AT[7] + w * _AT[9]))))
    if idx < 0:
        return x - x * (s1 + s2)
    z = _ATANHI[idx] - ((x * (s1 + s2) - _ATANLO[idx]) - x)
    return -z if hx < 0 else z


def atan_fast(x):
    """``atan`` with the high-word tests replaced by the equivalent float comparisons (every
    threshold has a zero low word, so ``ix < T`` <=> ``|x| < T`` exactly); no struct packing."""
    if x != x:
        return x + x
    ax = -x if x < 0.0 else x
    if ax >= 7.378697629483821e19:             # 2^66 (also +-inf)
        return _ATANHI[3] + _ATANLO[3] if x > 0.0 else -_ATANHI[3] - _ATANLO[3]
    if ax < 0.4375:
        if ax < 1.862645149230957e-09:         # 2^-29
            if _HUGE + x > 1.0:
                return x
        idx = -1
    else:
        if ax < 1.1875:
            if ax < 0.6875:
                idx = 0
                ax = (2.0 * ax - 1.0) / (2.0 + ax)
            else:
                idx = 1
                ax = (ax - 1.0) / (ax + 1.0)
        else:
            if ax < 2.4375:
                idx = 2
                ax = (ax - 1.5) / (1.0 + 1.5 * ax)
            else:
                idx = 3
                ax = -1.0 / ax
    if idx < 0:
        z = x * x
        w = z * z
        s1 = z * (_AT[0] + w * (_AT[2] + w * (_AT[4] + w * (_AT[6] + w * (_AT[8] + w * _AT[10])))))
        s2 = w * (_AT[1] + w * (_AT[3] + w * (_AT[5] + w * (_AT[7] + w * _AT[9]))))
        return x - x * (s1 + s2)
    z = ax * ax
    w = z * z
    s1 = z * (_AT[0] + w * (_AT[2] + w * (_AT[4] + w * (_AT[6] + w * (_AT[8] + w * _AT[10])))))
    s2 = w * (_AT[1] + w * (_AT[3] + w * (_AT[5] + w * (_AT[7] + w * _AT[9]))))
    z = _ATANHI[idx] - ((ax * (s1 + s2) - _ATANLO[idx]) - ax)
    return -z if x < 0.0 else z


def atan_vec(x):
    """Element-wise ``atan_fast`` on a small float64 array (8 nodes: cheaper than masking)."""
    return np.fromiter((atan_fast(float(v)) for v in x), dtype=np.float64, count=len(x))


def atan_np(x):
    """fdlibm s_atan.c on a float64 numpy array (same arithmetic per element as ``atan``)."""
    x = np.asarray(x, dtype=np.float64)
    bits = x.view(np.uint64)
    hx = (bits >> 32).astype(np.int64)
    hx = np.where(hx >= 2**31, hx - 2**32, hx)          # signed high word
    lx = (bits & 0xFFFFFFFF)
    ix = hx & 0x7FFFFFFF
    out = np.empty_like(x)
    neg = hx < 0
    ax = np.abs(x)
    # region selection
    huge = ix >= 0x44100000
    nan = huge & ((ix > 0x7FF00000) | ((ix == 0x7FF00000) & (lx != 0)))
    small = ix < 0x3FDC0000
    tiny = ix < 0x3E200000
    r0 = (~small) & (ix < 0x3FE60000)
    r1 = (~small) & (ix >= 0x3FE60000) & (ix < 0x3FF30000)
    r2 = (ix >= 0x3FF30000) & (ix < 0x40038000)
    r3 = (ix >= 0x40038000) & (~huge)
    xr = np.where(small, x, ax)
    with np.errstate(divide="ignore", invalid="ignore"):
        xr = np.where(r0, (2.0 * ax - 1.0) / (2.0 + ax), xr)
        xr = np.where(r1, (ax - 1.0) / (ax + 1.0), xr)
        xr = np.where(r2, (ax - 1.5) / (1.0 + 1.5 * ax), xr)
        xr = np.where(r3, -1.0 / ax, xr)
        z = xr * xr
        w = z * z
        s1 = z * (_AT[0] + w * (_AT[2] + w * (_AT[4] + w * (_AT[6] + w * (_AT[8] + w * _AT[10])))))
        s2 = w * (_AT[1] + w * (_AT[3] + w * (_AT[5] + w * (_AT[7] + w * _AT[9]))))
        idx = np.where(r0, 0, np.where(r1, 1, np.where(r2, 2, 3)))
        hi = np.asarray(_ATANHI)[idx]
        lo = np.asarray(_ATANLO)[idx]
        zz = hi - ((xr * (s1 + s2) - lo) - xr)
        res_big = np.where(neg, -zz, zz)
        res_small = xr - xr * (s1 + s2)
    out = np.where(small, res_small, res_big)
    out = np.where(tiny, x, out)
    out = np.where(huge & ~nan, np.where(hx > 0, _ATANHI[3] + _ATANLO[3], -_ATANHI[3] - _ATANLO[3]), out)
    out = np.where(nan, x + x, out)
    return out


def sin(x):
    """Java Math.sin: NaN for inf/nan input instead of raising."""
    if math.isinf(x) or math.isnan(x):
        return math.nan
    return math.sin(x)


def exp(x):
    """Java Math.exp: +inf on overflow instead of raising."""
    try:
        return math.exp(x)
    except OverflowError:
        return math.inf


def sqrt(x):
    if x != x or x < 0.0:
        return math.nan
    return math.sqrt(x)


def div(a, b):
    """Java double division: x/0 -> +-inf, 0/0 -> NaN (Python raises)."""
    try:
        return a / b
    except ZeroDivisionError:
        if a != a or a == 0.0:
            return math.nan
        return math.copysign(math.inf, a) * math.copysign(1.0, b)


def f32(x):
    """Java ``(float)`` cast: round to nearest float32 (overflow -> +-inf, NaN kept)."""
    return float(np.float32(x))
