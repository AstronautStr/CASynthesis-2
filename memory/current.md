# Current Implementation State

_Last updated: 2026-06-15_

## Implemented
- Continuous phase-coherent additive synthesis (pygame-ce + numpy)
- GoL on toroidal field (`scipy.ndimage.convolve`, mode='wrap', 8-connectivity)
- Connected-component segmentation (`scipy.ndimage.label`)
- Perceptual axes wired:
  - **size (area) → harmonic number k** — big object = low harmonic near fundamental (K_MAX = 20)
  - **density (bbox fill) → amplitude** — with 1/k^0.7 rolloff to tame upper partials
  - **centroid X → stereo pan**
- Voice cap: MAX_VOICES = 24 (largest objects prioritised when over limit)
- Carrier note set via on-screen piano (C3–C5, latched)
- Interactive UI: draw/erase cells, play/pause [Space], step [S], random [R], clear [C], speed [↑↓]
- Visualisation: harmonic hue (red=low → blue=high), brightness = amplitude; legend shown

- Pattern library sidebar (collapsible, "Lib ▶ / ◀ Lib" button in toolbar):
  - 29 patterns in 9 categories (separate file: `patterns.py`):
    still lifes ×7, oscillators p2 ×4, p3 ×3 (Pulsar, Jam, Caterer),
    p4 ×2 (Mold, Mazing), p5 ×1 (Octagon II), p8+ ×4 (Figure-8, Kok's galaxy, Tumbler, Pentadecathlon),
    spaceships ×4, guns ×1, methuselahs ×3
  - All patterns verified (explicit cells or simulation-confirmed RLE period)
  - Drag & drop: grid-snapped ghost preview (SRCALPHA), OR-insert on drop
  - Mouse-wheel scrolling in sidebar; scroll thumb indicator
  - Sidebar toggled via `state['sidebar_open']`; window width changes with `set_mode`

## Known issues
- Segmentation not wrap-aware: objects crossing the toroidal seam momentarily split into two voices
- No object identity tracking across generations (no true birth / death events, no collision handling)
- Only 3 of ~8 designed perceptual axes are wired; timbre range is correspondingly narrow

## In-flight
_Nothing currently in-flight._

## Candidate next steps
See `memory/decisions.md` for prioritisation by Researcher.

- Remaining perceptual axes: jaggedness → distortion/inharmonicity, activity → tremolo,
  symmetry → consonance, order/chaos → noise blend, elongation → detune/vibrato
- Wrap-aware segmentation (union-find across toroidal seam)
- Object identity tracking (voice allocation with birth / death / merge / split events)
