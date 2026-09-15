"""Source provenance for N0: the exact Gutter Synthesis commit, file hashes, and the parameter
table extracted from the executable class + the Max patches (initial values after load).

Authorship / licence: Gutter Synthesis (c) Tom Mudd, https://github.com/tommmmudd/guttersynthesis,
GNU GPL v3.  The Python port in this package is a derivative for research verification; the
original class files are NOT vendored -- point ``verify_node.py --source`` at a clone.
"""
import hashlib
import os

SOURCE_URL = "https://github.com/tommmmudd/guttersynthesis"
SOURCE_COMMIT = "efa4737af31febf09bd746a45bd9cd57c88f74b1"   # 2023-08-14 "Update readme.md"
SOURCE_LICENSE = "GPL-3.0"

# sha256 of the files the reconstruction is based on (at SOURCE_COMMIT)
SOURCE_HASHES = {
    "gutterOsc.class": "b328361bac815537369b4888083773ef4a3d2a5425dd384d1eb8fbb82353ccf9",       # EXECUTABLE node (used by the patch)
    "gutterOsc$BPFilter.class": "2fdc05a6b00327b9069b977121f0fdec7d726cfc2298243bdf47fd9c4d156558",
    "gutterOsc$Highpass.class": "f620d83a9722a582fdce585aa332f04c54da91c0c8c614a3736330370b33f214",
    "gutterOsc$Lowpass.class": "b01e0ea04698be5357407359472e88b3ecfd46a619cb5f25eb26807873462d54",
    "gutterOsc.java": "b0be129479b3b79c8585d0692ced18b1b7717e686dc3b4ef31faea22d9c6c2fd",         # OLDER 2-bank variant, NOT the executable
    "gutterOsc for Java1.6/gutterOsc.class": "ea9a959b9f2be6896bd070eec52b4a32ff98c01d4d43337cff885a0e665abcd9",
    "Gutter Synth.maxpat": "db531601b030f0655c914175a5080cda4fedb6e0ef56df6f861d009af2d36a0f",
    "mxjGutterCompact.maxpat": "6f7fd29ad8f3bcab90daa919b2d21a4f96f5b0bbe6e12e75bb2cce1621d24ca1",
    "filters.txt": "4d989129c432243c6694a0740c58daaf34173fbfe4ffecca8e85aeb348e92c0f",
    "LICENSE": "230184f60bae2feaf244f10a8bac053c8ff33a183bcc365b4d8b876d2b7f4809",
}


def hash_source(root):
    out = {}
    for rel in SOURCE_HASHES:
        p = os.path.join(root, rel)
        if os.path.exists(p):
            with open(p, "rb") as fh:
                out[rel] = hashlib.sha256(fh.read()).hexdigest()
        else:
            out[rel] = None
    return out


# ---------------------------------------------------------------------------------------
# Parameter table: name, unit, initial value after patch load, range, where it comes from.
# "restore" = values stored by the [autopattr] objects in the patches (UI state saved with
# the patch and restored on load).  Slider raw values are 0..256.
# ---------------------------------------------------------------------------------------
PARAMS = [
    # per node (mxjGutterCompact.maxpat, autopattr 'restore' + mapping subpatches)
    dict(name="gain (singleGain)", unit="linear", raw=162, value=162 / 256 * 3.5, range="0..3.5",
         source="restore gain=162; p mapGain: scale 0 256 0. 3.5 -> line~ 30 ms -> mxj inlet 4"),
    dict(name="damp (c base)", unit="1/sample", raw=138, value=(138 / 256) ** 2, range="0..1 (then +link, clip 0.0001..1)",
         source="restore damp=138; p mapDamp: (v/256)^2 -> line~ 15 ms -> +~ link -> clip~ 0.0001 1. -> mxj inlet 2"),
    dict(name="mod (gamma)", unit="forcing amplitude", raw=46, value=(46 / 256) ** 2 * 10, range="0..10",
         source="restore mod=46; p mapMod: (v/256)^2*10 -> line~ 30 ms -> mxj inlet 0"),
    dict(name="rate (dt)", unit="t increment/sample", raw=39, value=(39 / 256) ** 4 * 5, range="0..5",
         source="restore rate=39; p mapRate: (v/256)^4*5 -> line~ 30 ms -> mxj inlet 3"),
    dict(name="omega", unit="rad per t unit", raw=None, value=0.002, range="fixed",
         source="p duff: loadmess 0.002 -> p line (line~ 50 ms) -> mxj inlet 1 (forcing = gamma*sin(omega*t))"),
    dict(name="Q (all 24 filters)", unit="", raw=149, value=0.5 + 399.5 * (149 / 256) ** 2, range="0.5..400",
         source="restore Q=149; p mapQ: (v/256)^2 -> scale 0. 1. 0.5 400. -> multislider(24) -> setQN i q for all 24"),
    dict(name="soften (setLowpass)", unit="Hz", raw=103, value=500 + 15500 * (1 - 103 / 256) ** 2, range="500..16000",
         source="restore soften=103; scale 0 256 1. 0. -> squared -> scale 0. 1. 500. 16000. -> setLowpass"),
    dict(name="setHighpass", unit="Hz", raw=None, value=5.0, range="fixed (<10 => bypassed in class)",
         source="loadmess 5 -> setHighpass $1"),
    dict(name="filter count", unit="", raw=23, value=24, range="1..24",
         source="restore filter_count=23 (menu index) -> +1 -> filters 24"),
    dict(name="filter freqs", unit="Hz", raw=None, value="random per node at load (see FILTER_RANDOM_MIXED_ROOT)", range="50..5000 (x pitch_shift 0.1..2, min 19000)",
         source="p randomise_filters_mixed_root: loadbang -> del 1000 -> uzi 24; multislider 0..1 -> squared -> scale 0. 1. 50. 5000. -> list_mult pitch_shift -> setFreqN"),
    dict(name="pitch_shift", unit="ratio", raw=121.263161, value=0.1 + 1.9 * 121.263161 / 256, range="0.1..2",
         source="loadmess 121.263161 -> pictslider -> scale 0. 256 0.1 2. (= 1.0000)"),
    dict(name="distortion method", unit="", raw=None, value=2, range="0..5", source="class default (atan); no message in patch"),
    dict(name="pan per node", unit="0..1", raw=None, value=[0, 1, 0.15, 0.85, 0.3, 0.7, 0.46, 0.54], range="0..1",
         source="#1 -> pipe 2000 -> -1 -> zl lookup 0 1 0.15 0.85 0.3 0.7 0.46 0.54 -> pan slider; p pan: v=0.25*pan, L=cos(2 pi v), R=cos(2 pi (v+0.75)) via cycle~"),
    # node output chain (Max objects)
    dict(name="node post chain", unit="", raw=None, value="mxj out0 -> clip~ -5. 5. -> svf~ 20 (HP outlet) -> svf~ 30 (HP outlet) -> tanh~ -> p pan", range="",
         source="mxjGutterCompact.maxpat"),
    # network (Gutter Synth.maxpat, p matrix)
    dict(name="matrix input i", unit="", raw=None, value="0.5*(L_i + R_i)", range="", source="p matrix: receive~ iL/iR -> *~ 0.5 -> matrix~ inlet i"),
    dict(name="matrix cells", unit="gain", raw=None, value="preset 1: all 1 except diagonal 0", range="0/1 per cell (matrix~ 8 8 @ramp 2000)",
         source="loadmess 1 -> preset -> matrixctrl -> p matrix (preset_data 1)"),
    dict(name="matrix delay", unit="samples", raw=None, value="2000 + random(0..1999) at load (NOT reproducible); N0 fixes 2000", range="0..8000",
         source="loadbang -> random 2000 -> + 2000 -> s matrix_delay -> delay~ 8000 2000 (all 8 lines)"),
    dict(name="interaction", unit="gain on link", raw=127, value=(127 / 256) ** 2 * 5, range="0..5",
         source="loadmess 127 / restore interaction=127 -> scale 0. 256 0. 1. -> squared -> * 5. -> p matrix inlet 1 -> line~ 50 ms -> *~"),
    dict(name="link -> node", unit="", raw=None, value="send~ NmatrixOut -> receive~ #1matrixOut -> +~ damp -> clip~ 0.0001 1. -> c", range="",
         source="feedback through send~/receive~: +1 signal vector (64 samples assumed, see gutter_network.EXTRA_FEEDBACK_SAMPLES)"),
    dict(name="master out", unit="", raw=None, value="sum(L_n) * 1.0 (soften[1]=256) * 0.4 -> dac", range="",
         source="p leftOuts/p rightOuts -> *~ line~ -> *~ 0.4 -> ezdac~; N0 uses its own fixed MASTER_GAIN with headroom"),
    # master (all-node) sliders of the main patch, sent to every node as 'msg <name> <raw>' on restore
    dict(name="master restore (alternative init)", unit="raw", raw=dict(gain=150, damp=159, mod=46, rate=23, Q=149, soften=172), value="", range="",
         source="Gutter Synth.maxpat autopattr restore -> p prepends -> to_gutters; load order vs. node restore not determinable offline"),
    dict(name="movement preset", unit="", raw=2, value="SLOW: per-node random mod in (0.1..0.17)*256, rate in (0.2..0.8)*256", range="STATIC..RAPID",
         source="restore live.menu[1]=2 -> p movement_presets (random per node at load; N0 uses the deterministic node restore values instead)"),
]

# source formula for the random filter set (p randomise_filters_mixed_root), per filter:
#   r1 = (random(0..999) * 0.00098) ^ 4 ; r2 = (random(0..499) * 0.001) ^ 2 ; v = r1 + r2 (0..1 slider)
#   freq = 50 + 4950 * v^2   (p list_squared -> scale 0. 1. 50. 5000.) ; then * pitch_shift ; min(.., 19000)
FILTER_RANDOM_MIXED_ROOT = "v = (rand999*0.00098)^4 + (rand499*0.001)^2 ; f = 50 + 4950*v^2"


def random_filter_bank(rng, count=24, pitch_shift=1.0):
    """Draw one node's filter set the way the source does at load (seeded here for reproducibility)."""
    r1 = (rng.integers(0, 1000, count) * 0.00098) ** 4
    r2 = (rng.integers(0, 500, count) * 0.001) ** 2
    v = r1 + r2
    f = (50.0 + 4950.0 * v * v) * pitch_shift
    return [float(min(x, 19000.0)) for x in f]


def preset_filter_banks(filters_txt_path=None):
    """The 20 fixed banks of ``filters.txt`` (coll filters): index -> list of Hz."""
    banks = {}
    text = FILTERS_TXT if filters_txt_path is None else open(filters_txt_path).read()
    for line in text.strip().splitlines():
        idx, rest = line.split(",", 1)
        banks[int(idx)] = [float(x) for x in rest.strip().rstrip(";").split()]
    return banks


FILTERS_TXT = """0, 97 156 186 200 243 318 333 383 435 453 495 500 598 678 687 702 720 883 1522 1747 1859 1957 1965 2065;
1, 136 185 312 402 676 765 872 1016 1397 1811 2252 2708 3180;
2, 242 375 513 515 859 940 1040 1300 1775 2830;
3, 205 364 455 455 455 770 770 770 876 1158 1310 1590 2062 2549;
4, 141 222 298 298 298 298 552 758 1041 1345 1578 1671 2000 2341;
5, 68 97 170 248 391 449 531 589 658 711 879 771 807 1053 1200 1255 1460 1478 1521 1685 1666 1784 1921 1954;
6, 140 150 276 285 449 693 730 854 932 979 1417;
7, 30 60 90 122 166 201 270 293 308 490 502 953 1337 1664 4515 4686 5767;
8, 64 98 130 163 196 227 294 327 360 393 426 459 497 526 560 594 628 664 696 732 801 908 980 1052;
9, 338 875 894 1050 1150 1620 1662 1666 1841 1852 2052 2243 2311 2584 2842 2893 2958 2980 3731 4251 4516 4852;
10, 386.52 389.92 387.6 392.95 396.54 399.91 408.22 411.38 612.85 619.66 624.63 625.6 633.9 633.58 636.53 640.57 645.58 648.5 653.81 656.25 657.11 662.2 663.31 665.58;
11, 1001.79 1008.32 1010.39 1020.54 1024.12 1029.16 1033.89 257.87 260.92 263.89 290.34 290.98 312.52 328.6 332. 332.6 337.91 492.14 498.17 498.38 512.71 515.93 517.73 522.3;
12, 267.42 270.95 273.68 277.1 282.19 281.95 288.21 289.19 292.58 297.27 341.06 410.82 413.23 418.27 420.77 423.27 426.64 426.59 432.89 431.39 438.81 440.5 439.26 444.22;
13, 170.96 176.6 176.51 214.11 219.5 222.93 222.38 254.64 253.16 255.84 263.34 353.19 355.7 410.39 413.21 416.35 419.67 423.48 428.06 429.54 432.98 434.75 435.96 441.76;
14, 127.6 132.89 136.54 261.66 267.77 270.69 311.32 314.69 319.62 397.77 398.23 531.74 630.96 632.78 800.53 942.26 944.72;
15, 262.76 267.94 268.35 313.42 313.68 316.51 332.93 398.09 400.57 399.27 527.43 535.01 534.75 625.64 631.93 631.57 799.6 798.52 944.25 946.75 946.91 952.97 1057.93 1060.44;
16, 133.71 135.24 233.29 237.57 240.5 265.28 264.22 270.11 396.8 403.8 402.28 469.33 471.85 478.18 477.63 592.91 599.42 603.17 893.37 952.37 954.94 1192.62 1192.51;
17, 221.21 221.64 223.71 230.55 369.35 373.04 378.99 382.3 381.67 445.69 444.51 448.88 450.86 599.47 750.24 755.82 755.19 893.39 893.03 898.85 897.46 1124.42 1125.9;
18, 132.32 133.33 260.98 265.14 269.87 313.93 315.38 315.4 529.38 530.73 533.4 630.93 631.14 796.83 940.4 944.4 949.23 1064.89 1255.3 1254.54 1258.47 1262.46 2117.52;
19, 309.36 315.01 315.25 318.68 400.84 399.22 599.95 600.82 632.04 631.61 635.78 792.57 797.03 801.87 946.98 944.77 952.91 952.92 1194.99 1262.33 1259.48 1262.87 1266.38 1891.06;
"""
PRESET_NAMES = ["Bell1", "Bell2", "Bell3", "Bell4", "Bell5", "Porcelain Disc", "Porcelain Bowl", "Porcelain Sheet",
                "Piano C", "Choir1", "Choir2", "Choir3", "Choir4", "Organ1", "Organ2", "Organ3", "Organ4", "Organ5", "Organ6", "Organ7"]
