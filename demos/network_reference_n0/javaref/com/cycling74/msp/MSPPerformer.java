package com.cycling74.msp;
/** Minimal stand-in for the Max/MSP Java API base class: records inlet/outlet declarations only. */
public class MSPPerformer {
    public int[] declaredInlets, declaredOutlets;
    protected void declareInlets(int[] a) { declaredInlets = a; }
    protected void declareOutlets(int[] a) { declaredOutlets = a; }
    public void dspsetup(MSPSignal[] ins, MSPSignal[] outs) {}
    public void perform(MSPSignal[] ins, MSPSignal[] outs) {}
}
