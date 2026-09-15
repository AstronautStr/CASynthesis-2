package com.cycling74.msp;
/** Minimal stand-in for the Max/MSP Java API class: only the members gutterOsc.class touches. */
public class MSPSignal {
    public float[] vec;
    public double sr;
    public int n;
    public MSPSignal(int n, double sr) { this.n = n; this.sr = sr; this.vec = new float[n]; }
}
