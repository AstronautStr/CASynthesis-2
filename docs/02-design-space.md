# 02 — Design Space: options explored, paths chosen

The accumulated reasoning. Each sub-problem lists what we considered, what we chose,
and why. **Decided** = committed; **Open / roadmap** = mapped but not yet built.

## A. General CA → sound strategies (from oscillators)
A Game-of-Life oscillator is *already a periodic signal*: its period is a natural
fundamental / loop length, and its per-generation configuration is a natural timbre
source. Approaches considered:
- **Grid-as-spectrum (additive):** rows = partials, lit cells = active harmonics →
  a snapshot is a spectrum. *(This became the basis of the current synth.)*
- **Wavetable-from-cycle:** the period → loop length; a per-generation spatial
  profile → the looped waveform.
- **Cell-as-voice:** each lit cell = a tiny oscillator. *(Rejected as too granular — see D.)*
- **Feature-vector → synth patch:** characteristic features → preset parameters.
- **Cycle-as-sequencer:** period = loop, generations = steps. *(The "sequencer" pole
  we deliberately moved away from — see E.)*

**Gotcha (important):** symmetric oscillators (blinker, beacon) have *phase-invariant
scalar aggregates* — population and centroid don't change between phases.
Distinguishing phases requires **spatial** features (row/column projections,
bounding-box aspect, orientation). This is why a bounding-box "horizontalness"
feature was used to make a blinker audibly move.

## B. Motion (gliders / spaceships)
A spaceship is *periodic in shape but translating in space*. Motion is a new axis:
- **Glide / Doppler:** position drift → pitch glide + stereo pan (+ amplitude).
- **Frame of reference (key choice):** *co-moving* (re-centre the object → it looks
  like an oscillator; map movement separately) vs *fixed* (the object sweeps the
  grid). With **fixed frame + grid-as-spectrum, motion automatically becomes a
  frequency sweep** — elegant and general.
- **Velocity & direction as identity:** different ship speeds (c/4, c/2, …) and 8
  directions → a taxonomy of sounds.
- **Wake / trail:** delay / reverb / spectral-freeze turns a sparse mover into a
  sustained texture.

### Ladder of forms → sound classes (a unifying map)
- Still life → static drone / pad
- Oscillator → steady looping timbre
- Spaceship → a *gesture* (loop + glide / sweep)
- Puffer → moving source shedding spectral debris (evolving texture)
- Gun (e.g. Gosper, period 30) → rhythmic *event generator* (each emitted glider = a triggered voice)
- Methuselah (R-pentomino ~1103 gens, acorn ~5206) → long, evolving one-shot that resolves
- Collisions → interaction *events* (voices merging / annihilating)

## C. Capturing the "vibe": perceptual mapping vs learned embedding
We rejected naive **cell → pitch** (coordinate-based; treats a pattern as a bag of
cells). Reframe: **sonify the gestalt.**

**Cross-modal correspondence is the science of the mapping** (bouba/kiki and
relatives — robust, cross-cultural shape↔sound associations):
- boundary jaggedness / angularity → brightness + inharmonicity / harshness
- size / extent → pitch (big = low) [size-pitch symbolism]
- vertical position → pitch height [elevation-pitch]
- density (fill) → spectral fullness / number of partials
- symmetry / order → consonance vs dissonance
- activity (cells changing per generation) → roughness / tremolo / energy
- entropy / compressibility → noise vs tone (ordered = tonal, chaotic = noisy)
- compactness vs diffuseness → focused tone vs spread texture

**ML embedding route (explored, then demoted):**
- An embedding is a short vector capturing a pattern's essence, where similar
  patterns sit nearby. Ways to obtain one: **PCA** (easiest, no training);
  **convolutional autoencoder** (self-supervised compression — note CNN multi-scale
  comes from *depth + downsampling*, not big kernels; Inception / dilated convs are
  parallel-scale options); **pretrained vision-model embeddings** (free, no training).
- **Convolutions are weak at relational / statistical properties.** *Symmetry* is a
  long-range correspondence → compute it directly (flip-and-compare) or via a
  Siamese / attention comparison. *Order/chaos* is a global statistic → compute via
  compressibility (gzip the bitmap), Fourier / autocorrelation peaks, or local-patch
  variety. Lesson: don't make a net discover what you can just compute.

**Decision (LOAD-BEARING): for the perceptual-correspondence goal, hand-crafted
interpretable features are the backbone — not a learned embedding.**
- To wire a feature to the *perceptually correct* synth parameter you must know the
  feature's meaning. A learned embedding gives *internal consistency* (similar →
  similar) but **not perceptual correspondence**, and entangled axes risk *active
  mismatch* (e.g. size driving brightness instead of pitch). Recovering
  correspondence from a black box = the expensive problem of fitting to human
  similarity judgments.
- The cross-modal literature *is* pre-collected human-association data, so
  hand-wiring from it is principled, not a shortcut.
- **An embedding is at most a secondary layer:** (1) a uniqueness / tie-breaker so
  distinct patterns with identical hand-features don't collapse to the same sound,
  and (2) an "unnameable residual texture" channel. Deferred.

## D. Objects as voices (auditory scene)
Global / aggregate features **mis-measure multi-object scenes**: "many small
objects" by total mass reads as "one big object" → wrong pitch / brightness. And the
deeper truth: **separate objects are perceived as separate voices** (Bregman's
Auditory Scene Analysis; Gestalt grouping).

**Architecture (decided):** segment the field into objects → per-object feature
vector (now "size" means *this object's* size) → **each object = one voice / partial**
→ the mix = a polyphonic auditory scene. The right granularity is the **object**
(between the cell and the whole field; a single oscillator is the N = 1 case).

**Hard sub-problems (open):**
- **Segmentation is itself perceptual.** Connected components are a crude first cut;
  human grouping uses **proximity + common fate** (things moving together = one
  object). In Life, *motion coherence* is exactly what makes a glider's 5 cells one
  thing; two diverging gliders are two things → segment by adjacency *and* shared
  motion over time.
- **Voice management over time.** Objects are born, die, merge, split (collisions).
  Voices must be allocated / released: birth → note-on, death → note-off, merge →
  two voices fuse, split → spawn a child. Collisions become *events*.
- **Regime switch.** Countable objects → polyphony; an uncountable churning mass → a
  single noise texture. Perception switches between "N things" and "a texture"; the
  synth should too. This ties directly to the order↔chaos axis (sparse / ordered →
  polyphony, dense chaos → noise).

## E. Synthesizer vs sequencer — the additive reframe (decided)
Mapping size → *different notes* makes a chord per frame = a **sequencer**. Instead:
pick a **carrier note H**; size maps to **which harmonic of H** an object sings (big
= low harmonic near the fundamental, small = high). All objects are harmonics of the
same fundamental → they fuse into **one evolving timbre** = additive synthesis. The
**field = the spectrum of the played note**; an on-screen piano sets H. This is the
synthesizer framing and is what the current prototype implements.

## F. Audio engine (decided → implemented)
- Started with **per-generation windowed grains** (overlap-add via channels) → an
  audible **tremolo / pulsing** at the generation rate (the windowing wasn't
  constant-overlap-add and phases reset each grain).
- Moved to **continuous, phase-coherent synthesis**: each harmonic keeps a running
  phase across audio chunks (no clicks); amplitudes **glide** toward targets
  (constant loudness, smooth spectral morph); chunks are streamed **gaplessly** via
  the channel queue. This is the current engine.

## G. Roadmap (mapped, not yet built)
- **Wire the remaining perceptual axes** so they are audible and *distinct* (each its
  own parameter): jaggedness → distortion / odd-harmonic content; activity →
  tremolo; symmetry → consonance / inharmonicity; order↔chaos → noise blend;
  elongation / aspect → detune or vibrato. Widen feature ranges to the real observed
  data; add the not-yet-computed features (symmetry via flip-and-compare, chaos via
  gzip / Fourier, activity via cell-change count).
- **Wrap-aware segmentation** for the toroidal field: union-find merge of labels
  across the seam + coordinate "unwrapping" for correct per-object bbox / centroid.
- **Object identity tracking across generations** → real attacks / decays on birth /
  death, and collisions as audible events (the auditory-streaming model from D).
- **Common-fate segmentation** (group by shared motion, not just adjacency).
- **Regime switch** to a single noise voice when objects become uncountable.
- **Optional always-on fundamental** for instrument-like playability (so an empty
  field still sounds the pressed note); currently omitted to keep "field = whole
  spectrum" pure.
- **Learned-embedding secondary layer** for uniqueness / texture (deferred; see C).
- **Real-time audio backend** (sounddevice OutputStream callback, or SuperCollider
  driven from Python) if the pygame queue proves limiting for low-latency continuous
  synthesis.
