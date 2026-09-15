"""Scalar (per-sample) port of the shipped ``gutterOsc.class`` from Tom Mudd's Gutter Synthesis.

Source: https://github.com/tommmmudd/guttersynthesis (GPL-3.0, (c) Tom Mudd).
IMPORTANT: the executable node used by ``Gutter Synth.maxpat`` is the top-level
``gutterOsc.class`` (6 signal inlets, ONE bank of up to 24 band-pass biquads, one-pole
lowpass + one-pole highpass).  It is NOT compiled from the ``gutterOsc.java`` in the same
repository (that file describes an older two-bank, 8-inlet variant; its compiled form is
``gutterOsc for Java1.6/gutterOsc.class``).  This port follows the decompiled bytecode of the
top-level class (CFR 0.152) and is verified against the original class file executed on a JVM
(see ``verify_node.py``).

Arithmetic is kept statement-for-statement identical to the Java (evaluation order, float32
signal I/O, double state).  Nothing is "improved".

Inlets (signal, float32): gamma, omega, c, dt, singleGain, audioInput.
Outlets: out0 = finalY*0.125 (filters on) | clip(duffX*singleGain) (filters off); out1 = duffX.
"""
import math

from . import jmath
from .jmath import f32

M_PI = math.pi          # Java: double M_PI = Math.PI


class BPFilter:
    """gutterOsc$BPFilter: band-pass biquad (musicdsp.org biquad.c form)."""

    def __init__(self, node):
        self.node = node
        self.a0 = 0.0
        self.a2 = 0.0
        self.b1 = 0.0
        self.b2 = 0.0
        self.a1 = 0.0
        self.filterFreq = 100.0
        self.Q = 30.0
        self.V = math.pow(10.0, 0.05)      # unused by process(); kept for fidelity
        self.sampleRate = 44100.0
        self.prevX2 = 0.0
        self.prevX1 = 0.0
        self.prevY1 = 0.0
        self.prevY2 = 0.0

    def setFreq(self, d):
        self.filterFreq = d

    def setQ(self, d):
        self.Q = d

    def setSampleRate(self, d):
        self.sampleRate = d

    def calcCoeffs(self):
        d = math.tan(M_PI * self.filterFreq / self.sampleRate)
        d2 = 1.0 / (1.0 + d / self.Q + d * d)
        self.a0 = d / self.Q * d2
        self.a2 = -self.a0
        self.b1 = 2.0 * (d * d - 1.0) * d2
        self.b2 = (1.0 - d / self.Q + d * d) * d2

    def reset(self):
        self.prevX2 = 0.0
        self.prevX1 = 0.0
        self.prevY2 = 0.0
        self.prevY1 = 0.0

    def process(self, d):
        d2 = (self.a0 * d + self.a1 * self.prevX1 + self.a2 * self.prevX2
              - self.b1 * self.prevY1 - self.b2 * self.prevY2)
        self.prevX2 = self.prevX1
        self.prevX1 = self.node.duffX        # Java: gutterOsc.this.duffX (== d, the argument)
        self.prevY2 = self.prevY1
        self.prevY1 = d2
        return d2 * self.node.singleGain


class Highpass:
    """gutterOsc$Highpass: one-pole (leaky integrator subtracted). Bypassed by the node if cutoff < 10."""

    def __init__(self):
        self.sampleRate = 44100.0
        self.cutoff = 20.0
        self.cutoffRatio = 1.0
        self.prevVal = 0.0
        self.state = 0.0
        self.b1 = 1.0

    def process(self, d):
        d2 = d - self.state
        self.state += d2 * self.cutoffRatio
        return d2

    def setFreq(self, d):
        self.cutoff = d
        self.cutoffRatio = self.cutoff / (6.283188 * self.sampleRate)

    def setSampleRate(self, d):
        self.sampleRate = d

    def getFreq(self):
        return self.cutoff


class Lowpass:
    """gutterOsc$Lowpass: one-pole y = a0*x + b1*y1 with a0 = sin(2*pi*fc/sr), b1 = a0 - 1.

    NOTE: until setFreq() is called a0 = 0 and b1 = 1, i.e. the output is stuck at 0 -> the
    node is silent.  The patch always sends ``setLowpass`` on load (soften slider restore).
    """

    def __init__(self):
        self.sampleRate = 44100.0
        self.cutoff = 20000.0
        self.cutoffRatio = 1.0
        self.prevVal = 0.0
        self.a0 = 0.0
        self.b1 = 1.0

    def process(self, d):
        d2 = self.a0 * d + self.b1 * self.prevVal
        self.prevVal = d2
        return d2

    def setFreq(self, d):
        self.cutoff = d
        self.cutoffRatio = self.cutoff / self.sampleRate
        self.a0 = math.sin(6.283188 * self.cutoffRatio)
        self.b1 = self.a0 - 1.0

    def setSampleRate(self, d):
        self.sampleRate = d

    def getFreq(self):
        return self.cutoff


class GutterOsc:
    """Port of ``gutterOsc`` (top-level class).  Method names follow the Java 1:1."""

    MAX_FILTERS = 24

    def __init__(self):
        self.sampleRate = 44100.0
        self.M_PI = M_PI
        self.singleGain = 0.0
        self.audioInput = 0.0
        self.enableAudioInput = False
        self.smoothing = 1.0
        self.duffX = 0.0
        self.duffY = 0.0
        self.dx = 0.0
        self.dy = 0.0
        self.gamma = 0.0
        self.omega = 0.0
        self.c = 0.0
        self.t = 0.0
        self.dt = 0.0
        self.finalY = 0.0
        self.distMethod = 2
        self.maxFilters = self.MAX_FILTERS
        self.filterCount = 24
        self.filtersOn = True
        self.resets = 0          # port-only counter: how often resetDuff() fired (NaN / |duffX|>99)
        self.highpass = Highpass()
        self.lowpass = Lowpass()
        self.filterArray = []
        for i in range(self.maxFilters):
            f = BPFilter(self)
            f.setFreq(i / 2.0 * 20.0 * 1.2 + 80.0)
            f.setQ(30.0)
            f.calcCoeffs()
            self.filterArray.append(f)
        self.gamma = 0.1
        self.omega = 1.25
        self.c = 0.3
        self.t = 0.0
        self.dt = 1.0

    # ---- messages (Max "msg" interface); float args are float32 in Java -----------------
    def setFreqN(self, n, f):
        if n < self.filterCount:
            self.filterArray[n].setFreq(f32(f))
            self.filterArray[n].calcCoeffs()

    def setQN(self, n, f):
        if n < self.filterCount:
            self.filterArray[n].setQ(f32(f))
            self.filterArray[n].calcCoeffs()

    def setLowpass(self, f):
        self.lowpass.setFreq(f32(f))

    def setHighpass(self, f):
        self.highpass.setFreq(f32(f))

    def setSingleGain(self, f):
        self.singleGain = f32(f)

    def toggleFilters(self, n):
        self.filtersOn = n > 0

    def setDistortionMethod(self, n):
        self.distMethod = n

    def toggleAudioInput(self, n):
        self.enableAudioInput = n == 1

    def filters(self, n):
        self.filterCount = n

    def resetDuff(self):
        self.duffX = 0.0
        self.duffY = 0.0
        self.dx = 0.0
        self.dy = 0.0
        self.t = 0.0
        self.resets += 1

    def reset(self):
        self.duffX = 0.0
        self.duffY = 0.0
        self.dx = 0.0
        self.dy = 0.0
        self.t = 0.0
        self.dt = 0.0
        for i in range(self.maxFilters):
            self.filterArray[i].reset()

    def distortion(self, d, n):
        d2 = 0.0
        if n == 0:
            d2 = max(min(d, 1.0), -1.0)
        elif n == 1:
            if self.finalY <= -1.0:
                d2 = -0.666666667
            elif d <= 1.0:
                d2 = d - d * d * d / 3.0
            else:
                d2 = 0.666666667
        elif n == 2:
            d2 = jmath.atan(d)          # fdlibm atan == java.lang.Math.atan
        elif n == 3:
            d2 = jmath.div(0.75 * (jmath.sqrt(d * 1.3 * (d * 1.3) + 1.0) * 1.65 - 1.65), d)
        elif n == 4:
            d2 = jmath.div(0.1076 * d * d * d + 3.029 * d, d * d + 3.124)
        elif n == 5:
            d2 = jmath.div(2.0, 1.0 + jmath.exp(-1.0 * d))
        return d2

    def dspsetup(self, sr):
        """Called when DSP is switched on: takes the real sample rate, recalcs, resets filters."""
        self.sampleRate = float(sr)
        self.lowpass.setSampleRate(self.sampleRate)
        self.highpass.setSampleRate(self.sampleRate)
        for i in range(self.maxFilters):
            self.filterArray[i].setSampleRate(self.sampleRate)
            self.filterArray[i].calcCoeffs()
            self.filterArray[i].reset()

    def perform_sample(self, gamma, omega, c, dt, singleGain, audioInput):
        """One sample of ``perform``.  Inputs are float32 signal values; returns (out0, out1) as float32."""
        self.gamma = f32(gamma)
        self.omega = f32(omega)
        self.c = f32(c)
        self.dt = f32(dt)
        self.singleGain = f32(singleGain)
        self.audioInput = f32(audioInput)
        self.finalY = 0.0
        if self.filtersOn:
            for j in range(self.filterCount):
                self.finalY += self.filterArray[j].process(self.duffX)
        else:
            self.finalY = self.duffX
        fy = self.finalY
        if self.enableAudioInput:
            self.dy = fy - fy * fy * fy - self.c * self.duffY + self.gamma * self.audioInput
        else:
            self.dy = fy - fy * fy * fy - self.c * self.duffY + self.gamma * jmath.sin(self.omega * self.t)
        self.duffY += self.dy
        self.dx = self.duffY
        d2 = self.lowpass.process(self.finalY + self.dx)
        d3 = self.highpass.process(d2)
        self.duffX = d2 if self.highpass.getFreq() < 10.0 else d3
        if self.filtersOn:
            self.duffX = self.distortion(self.duffX, self.distMethod)
            out0 = f32(self.finalY * 0.125)
        else:
            self.duffX = max(min(self.duffX, 100.0), -100.0)
            if abs(self.duffX) > 99.0:
                self.resetDuff()
            out0 = f32(max(min(self.duffX * self.singleGain, 1.0), -1.0))
        out1 = f32(self.duffX)
        self.t += self.dt
        if math.isnan(self.duffX):
            self.resetDuff()
        return out0, out1

    def perform(self, ins):
        """``ins``: 6 sequences of equal length (float32 values). Returns (out0 list, out1 list)."""
        n = len(ins[0])
        o0 = [0.0] * n
        o1 = [0.0] * n
        for i in range(n):
            o0[i], o1[i] = self.perform_sample(ins[0][i], ins[1][i], ins[2][i], ins[3][i], ins[4][i], ins[5][i])
        return o0, o1
