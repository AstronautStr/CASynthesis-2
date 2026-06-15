"""
patterns.py — Game of Life pattern library for CASynth.

Each pattern is a list of (row, col) live-cell offsets from anchor (0,0) —
the top-left corner of the bounding box.  Dead cells in a pattern never erase
existing live cells when dropped onto the field.

ALL patterns are verified by running the GoL automaton and confirming the
expected period.  Cell coordinates extracted from playgameoflife.com lexicon
visual grids and cross-checked with LifeWiki descriptions.
"""


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _rle(s):
    """Decode compact RLE body → list of (row, col) live cells."""
    cells, row, col, num = [], 0, 0, 0
    for ch in s:
        if ch.isdigit():
            num = num * 10 + int(ch)
        elif ch == 'o':
            for _ in range(max(1, num)):
                cells.append((row, col)); col += 1
            num = 0
        elif ch == 'b':
            col += max(1, num); num = 0
        elif ch == '$':
            row += max(1, num); col = 0; num = 0
        elif ch == '!':
            break
    return cells


def _dot(rows):
    """Decode list of dot/O strings → list of (row, col) live cells."""
    cells = []
    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            if ch == 'O':
                cells.append((r, c))
    return cells


def _pulsar():
    cells = []
    for dc in [2, 3, 4, 8, 9, 10]:
        for dr in [0, 5, 7, 12]:
            cells.append((dr, dc))
    for dr in [2, 3, 4, 8, 9, 10]:
        for dc in [0, 5, 7, 12]:
            cells.append((dr, dc))
    return cells


# ---------------------------------------------------------------------------
# PATTERNS  — list of (category_name, [(pattern_name, cells), …])
# ---------------------------------------------------------------------------
PATTERNS = [

    # -----------------------------------------------------------------------
    # STILL LIFES — configurations that never change
    # -----------------------------------------------------------------------
    ("Still lifes", [
        ("Block",    [(0,0),(0,1),(1,0),(1,1)]),
        ("Beehive",  [(0,1),(0,2),(1,0),(1,3),(2,1),(2,2)]),
        ("Loaf",     [(0,1),(0,2),(1,0),(1,3),(2,1),(2,3),(3,2)]),
        ("Boat",     [(0,0),(0,1),(1,0),(1,2),(2,1)]),
        ("Ship",     [(0,0),(0,1),(1,0),(1,2),(2,1),(2,2)]),
        ("Tub",      [(0,1),(1,0),(1,2),(2,1)]),
        ("Pond",     [(0,1),(0,2),(1,0),(1,3),(2,0),(2,3),(3,1),(3,2)]),
    ]),

    # -----------------------------------------------------------------------
    # OSCILLATORS — period 2  (verified period = 2)
    # -----------------------------------------------------------------------
    ("Oscillators p2", [
        # Blinker — simplest, 3 cells (John Conway, 1970).
        ("Blinker",
         [(0,0),(0,1),(0,2)]),

        # Toad — 6 cells (Simon Norton, 1970).
        ("Toad",
         [(0,1),(0,2),(0,3),(1,0),(1,1),(1,2)]),

        # Beacon — two diagonal 2×2 blocks, 8 cells (John Conway, 1971).
        ("Beacon",
         [(0,0),(0,1),(1,0),(1,1),(2,2),(2,3),(3,2),(3,3)]),

        # Traffic light — 4 blinkers in symmetric cross, gap=2 between each
        # pair so they do not interfere.  Period 2, 4 components, 12 cells.
        # Bounding box 7x7; centre of cross at (3,3).
        # Verified: gen0==gen2, gen0!=gen1, components=4 each generation.
        ("Traffic light",
         [(0,2),(0,3),(0,4),
          (2,0),(3,0),(4,0),
          (2,6),(3,6),(4,6),
          (6,2),(6,3),(6,4)]),
    ]),

    # -----------------------------------------------------------------------
    # OSCILLATORS — period 3  (verified period = 3)
    # -----------------------------------------------------------------------
    ("Oscillators p3", [
        # Pulsar — largest "natural" p3 oscillator, 48 cells (J. Conway, 1970).
        ("Pulsar", _pulsar()),

        # Jam — smallest p3 by bounding box, 13 cells
        # (Achim Flammenkamp 1988, named by Dean Hickerson 1989).
        ("Jam",
         _dot(["...OO.", "..O..O", "O..O.O", "O...O.", "O.....", "...O..", ".OO..."])),

        # Caterer — p3 oscillator, 12 cells (Dean Hickerson, 1989).
        ("Caterer",
         _dot(["..O.....", "O...OOOO", "O...O...", "O.......", "...O....", ".OO....."]))
    ]),

    # -----------------------------------------------------------------------
    # OSCILLATORS — period 4  (verified period = 4)
    # -----------------------------------------------------------------------
    ("Oscillators p4", [
        # Mold — smallest p4 by bounding box, 12 cells, 6×6
        # (Achim Flammenkamp 1988, named Dean Hickerson 1989).
        ("Mold",
         _dot(["...OO.", "..O..O", "O..O.O", "....O.", "O.OO..", ".O...."])),

        # Mazing — smallest p4 by population (ties Mold), 12 cells
        # (Dave Buckingham, 1973).
        ("Mazing",
         _dot(["...OO..", ".O.O...", "O.....O", ".O...OO", ".......", "...O.O.", "....O.."])),
    ]),

    # -----------------------------------------------------------------------
    # OSCILLATORS — period 5  (verified period = 5)
    # -----------------------------------------------------------------------
    ("Oscillators p5", [
        # Octagon II — first known p5 oscillator, 16 cells, 8×8
        # (Sol Goodman & Arthur Taber, 1971).
        ("Octagon II",
         _dot(["...OO...", "..O..O..", ".O....O.", "O......O",
               "O......O", ".O....O.", "..O..O..", "...OO..."])),
    ]),

    # -----------------------------------------------------------------------
    # OSCILLATORS — period 8 and above  (verified periods)
    # -----------------------------------------------------------------------
    ("Oscillators p8+", [
        # Figure-8 — p8, 18 cells, two 3×3 blocks (Simon Norton, 1970).
        # RLE verified: period = 8.
        ("Figure-8",
         _rle("3o$3o$3o$3b3o$3b3o$3b3o!")),

        # Kok's galaxy — p8, 48 cells, 9×9 (Jan Kok, 1971).
        # Golly's mascot pattern; rotates as two interlocking arcs.
        ("Kok's galaxy",
         _dot(["OOOOOO.OO", "OOOOOO.OO", ".......OO",
               "OO.....OO", "OO.....OO", "OO.....OO",
               "OO.......", "OO.OOOOOO", "OO.OOOOOO"])),

        # Tumbler — p14, 22 cells (George Collins, 1970).
        # RLE verified: period = 14.
        ("Tumbler",
         _rle("2o3b2o$obobobo$obobobo$2bobo$b2ob2o$b2ob2o!")),

        # Pentadecathlon — p15, 12 cells (John Conway, 1970).
        ("Pentadecathlon",
         [(0,1),(1,1),(2,0),(2,2),(3,1),(4,1),(5,1),(6,1),
          (7,0),(7,2),(8,1),(9,1)]),
    ]),

    # -----------------------------------------------------------------------
    # SPACESHIPS — patterns that translate across the field
    # -----------------------------------------------------------------------
    ("Spaceships", [
        # Glider — c/4 diagonal (Richard Guy, 1969). 5 cells.
        ("Glider",
         [(0,1),(1,2),(2,0),(2,1),(2,2)]),

        # LWSS — c/2 orthogonal (John Conway, 1970). 9 cells.
        ("LWSS",
         [(0,1),(0,4),(1,0),(2,0),(2,4),(3,0),(3,1),(3,2),(3,3)]),

        # MWSS — c/2 orthogonal (John Conway, 1970). 11 cells.
        ("MWSS",
         [(0,2),(1,0),(1,4),(2,5),(3,0),(3,5),(4,1),(4,2),(4,3),(4,4),(4,5)]),

        # HWSS — c/2 orthogonal (John Conway, 1970). 13 cells.
        ("HWSS",
         [(0,2),(0,3),(1,0),(1,5),(2,6),(3,0),(3,6),
          (4,1),(4,2),(4,3),(4,4),(4,5),(4,6)]),
    ]),

    # -----------------------------------------------------------------------
    # GUNS — emit gliders or spaceships indefinitely
    # -----------------------------------------------------------------------
    ("Guns", [
        # Gosper Glider Gun — p30, 36 cells (Bill Gosper, 1970).
        # First pattern with unbounded growth.
        ("Gosper Gun", [
            (0,24),(1,22),(1,24),
            (2,12),(2,13),(2,20),(2,21),(2,34),(2,35),
            (3,11),(3,15),(3,20),(3,21),(3,34),(3,35),
            (4,0),(4,1),(4,10),(4,16),(4,20),(4,21),
            (5,0),(5,1),(5,10),(5,14),(5,16),(5,17),(5,22),(5,24),
            (6,10),(6,16),(6,24),(7,11),(7,15),(8,12),(8,13),
        ]),
    ]),

    # -----------------------------------------------------------------------
    # METHUSELAHS — small seeds, long chaotic evolution
    # Great for the synth: evolving patterns = evolving timbral texture
    # -----------------------------------------------------------------------
    ("Methuselahs", [
        # R-pentomino — stabilises after 1103 generations. 5 cells.
        ("R-pentomino",
         [(0,1),(0,2),(1,0),(1,1),(2,1)]),

        # Acorn — stabilises after 5206 generations. 7 cells.
        ("Acorn",
         [(0,1),(1,3),(2,0),(2,1),(2,4),(2,5),(2,6)]),

        # Diehard — disappears after 130 generations. 7 cells.
        ("Diehard",
         [(0,6),(1,0),(1,1),(2,1),(2,5),(2,6),(2,7)]),
    ]),
]
