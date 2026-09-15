import com.cycling74.msp.*;
import java.io.*;
import java.lang.reflect.*;
import java.nio.*;
import java.nio.file.*;
import java.util.*;

/**
 * Drives the ORIGINAL gutterOsc.class (shipped binary, unmodified) on given input signals.
 * Spec file lines:
 *   sr 44100 | block 64 | nsamples N | inputs path.f32 (6*N float32 LE, rows gamma,omega,c,dt,gain,audio)
 *   msg name [args...]           -> method call before dspsetup (int/float args by method signature)
 *   at block name [args...]      -> method call before that block index (during DSP)
 *   nodspsetup                   -> skip dspsetup (constructor state only)
 *   trace N                      -> per-sample doubles duffX,duffY,finalY,t,dx,dy for the first N samples
 *   out prefix                   -> writes prefix_out0.f32, prefix_out1.f32, prefix_trace.f64, prefix_state.txt
 */
public class RefRunner {
    static Object call(Object o, String name, String[] args) throws Exception {
        for (Method m : o.getClass().getMethods()) {
            if (!m.getName().equals(name) || m.getParameterCount() != args.length) continue;
            Class<?>[] pt = m.getParameterTypes();
            Object[] av = new Object[args.length];
            for (int i = 0; i < args.length; i++) {
                if (pt[i] == int.class) av[i] = Integer.parseInt(args[i]);
                else if (pt[i] == float.class) av[i] = Float.parseFloat(args[i]);
                else if (pt[i] == double.class) av[i] = Double.parseDouble(args[i]);
                else throw new RuntimeException("unsupported param type " + pt[i]);
            }
            return m.invoke(o, av);
        }
        throw new RuntimeException("no method " + name + "/" + args.length);
    }
    static double field(Object o, String name) throws Exception {
        Field f = o.getClass().getDeclaredField(name); f.setAccessible(true); return f.getDouble(o);
    }
    static Method mPerf;
    static void performOne(Object osc, float[][] in, int i, double sr, float[] o0, float[] o1) throws Exception {
        MSPSignal[] i1 = new MSPSignal[6], o1s = new MSPSignal[3];
        for (int r = 0; r < 6; r++) { i1[r] = new MSPSignal(1, sr); i1[r].vec[0] = in[r][i]; }
        for (int r = 0; r < 3; r++) o1s[r] = new MSPSignal(1, sr);
        mPerf.invoke(osc, (Object) i1, (Object) o1s);
        o0[i] = o1s[0].vec[0]; o1[i] = o1s[1].vec[0];
    }
    public static void main(String[] a) throws Exception {
        List<String> lines = Files.readAllLines(Paths.get(a[0]));
        double sr = 44100; int block = 64; int n = 0; String inputs = null; String out = "ref"; int trace = 0; boolean dsp = true;
        List<String[]> msgs = new ArrayList<>();
        Map<Integer, List<String[]>> at = new TreeMap<>();
        for (String l : lines) {
            l = l.trim(); if (l.isEmpty() || l.startsWith("#")) continue;
            String[] t = l.split("\\s+");
            switch (t[0]) {
                case "sr": sr = Double.parseDouble(t[1]); break;
                case "block": block = Integer.parseInt(t[1]); break;
                case "nsamples": n = Integer.parseInt(t[1]); break;
                case "inputs": inputs = t[1]; break;
                case "out": out = t[1]; break;
                case "trace": trace = Integer.parseInt(t[1]); break;
                case "nodspsetup": dsp = false; break;
                case "msg": msgs.add(Arrays.copyOfRange(t, 1, t.length)); break;
                case "at": at.computeIfAbsent(Integer.parseInt(t[1]), k -> new ArrayList<>()).add(Arrays.copyOfRange(t, 2, t.length)); break;
                default: throw new RuntimeException("bad line: " + l);
            }
        }
        Class<?> cls = Class.forName("gutterOsc");
        Object osc = cls.getConstructor().newInstance();
        for (String[] m : msgs) call(osc, m[0], Arrays.copyOfRange(m, 1, m.length));
        float[][] in = new float[6][n];
        ByteBuffer bb = ByteBuffer.wrap(Files.readAllBytes(Paths.get(inputs))).order(ByteOrder.LITTLE_ENDIAN);
        for (int r = 0; r < 6; r++) for (int i = 0; i < n; i++) in[r][i] = bb.getFloat();
        MSPSignal[] ins = new MSPSignal[6], outs = new MSPSignal[3];
        for (int r = 0; r < 6; r++) ins[r] = new MSPSignal(block, sr);
        for (int r = 0; r < 3; r++) outs[r] = new MSPSignal(block, sr);
        Method mDsp = cls.getMethod("dspsetup", MSPSignal[].class, MSPSignal[].class);
        mPerf = cls.getMethod("perform", MSPSignal[].class, MSPSignal[].class);
        if (dsp) mDsp.invoke(osc, (Object) ins, (Object) outs);
        float[] o0 = new float[n], o1 = new float[n];
        double[] tr = new double[trace * 6];
        int nb = (n + block - 1) / block;
        for (int b = 0; b < nb; b++) {
            List<String[]> ms = at.get(b);
            if (ms != null) for (String[] m : ms) call(osc, m[0], Arrays.copyOfRange(m, 1, m.length));
            int base = b * block, len = Math.min(block, n - base);
            if (trace > base || len < block) {
                // sample-wise perform: the class loops per sample, so the block split does not touch the state
                for (int i = 0; i < len; i++) {
                    performOne(osc, in, base + i, sr, o0, o1);
                    if (base + i < trace) {
                        int k = base + i;
                        tr[k * 6] = field(osc, "duffX"); tr[k * 6 + 1] = field(osc, "duffY"); tr[k * 6 + 2] = field(osc, "finalY");
                        tr[k * 6 + 3] = field(osc, "t"); tr[k * 6 + 4] = field(osc, "dx"); tr[k * 6 + 5] = field(osc, "dy");
                    }
                }
            } else {
                for (int r = 0; r < 6; r++) System.arraycopy(in[r], base, ins[r].vec, 0, len);
                mPerf.invoke(osc, (Object) ins, (Object) outs);
                System.arraycopy(outs[0].vec, 0, o0, base, len); System.arraycopy(outs[1].vec, 0, o1, base, len);
            }
        }
        writeF32(out + "_out0.f32", o0); writeF32(out + "_out1.f32", o1);
        ByteBuffer tb = ByteBuffer.allocate(tr.length * 8).order(ByteOrder.LITTLE_ENDIAN);
        for (double d : tr) tb.putDouble(d);
        Files.write(Paths.get(out + "_trace.f64"), tb.array());
        try (PrintWriter pw = new PrintWriter(out + "_state.txt")) {
            for (String f : new String[]{"duffX", "duffY", "dx", "dy", "finalY", "t", "dt", "gamma", "omega", "c", "singleGain", "sampleRate"})
                pw.println(f + " " + Double.toString(field(osc, f)));
        }
        System.out.println("ok " + n + " samples, block " + block);
    }
    static void writeF32(String p, float[] v) throws IOException {
        ByteBuffer b = ByteBuffer.allocate(v.length * 4).order(ByteOrder.LITTLE_ENDIAN);
        for (float f : v) b.putFloat(f);
        Files.write(Paths.get(p), b.array());
    }
}
