"""Text-field model shared by EVERY text input of the demo bench (save form
Title / Note, listening Notes).  pygame-free: the view supplies `measure`
(text -> pixel width) and draws; this class owns text, caret, selection,
word rules, wrapping and pixel <-> caret mapping.  One model = one behaviour
for all fields (2026-09-16: the hand-rolled fields had no caret at all).

Conventions
- `cursor`: caret index in [0, len(text)]; `anchor`: selection anchor or
  None; the selection is text[min:max] when they differ.
- Word rule (Ctrl+Backspace / Ctrl+Left): skip trailing separators, then
  the word -- the bench's rule since S4 ("заметка  два " <- "заметка  два три ").
- Layout: visual lines as (start, end) spans of the text; `end` is
  exclusive; a hard '\\n' belongs to no span; a soft break keeps the
  trailing space on the upper line.  The caret at a soft `end` shows at the
  start of the next line (it IS that index).
- `scroll` / `follow`: view state kept here so that the view can bring the
  caret into view after an edit and leave a wheel-scrolled view alone.
"""

SEPARATORS = ' \n\t'


class TextEdit:
    def __init__(self, text='', multiline=False):
        self.multiline = bool(multiline)
        self.text = text
        self.cursor = len(text)
        self.anchor = None
        self.scroll = 0            # first visible line (multiline) / px offset (single-line)
        self.follow = True         # the view should bring the caret into view
        self.goal_x = None         # pixel column kept across Up / Down moves
        self._layout_cache = None

    # -- selection / caret -----------------------------------------------------
    @property
    def selection(self):
        """(lo, hi) of the selected span or None."""
        if self.anchor is None or self.anchor == self.cursor:
            return None
        return (min(self.anchor, self.cursor), max(self.anchor, self.cursor))

    def selected_text(self):
        sel = self.selection
        return self.text[sel[0]:sel[1]] if sel else ''

    def set_text(self, text):
        """Reload from outside (a file): caret at the end, no selection."""
        self.text = text
        self.cursor = len(text)
        self.anchor = None
        self.follow = True
        self.goal_x = None

    def set_cursor(self, i, shift=False):
        """Move the caret; with `shift` the selection grows from where it was."""
        i = max(0, min(int(i), len(self.text)))
        if shift:
            if self.anchor is None:
                self.anchor = self.cursor
        else:
            self.anchor = None
        self.cursor = i
        self.follow = True
        self.goal_x = None

    def select_all(self):
        self.anchor = 0
        self.cursor = len(self.text)
        self.follow = True
        self.goal_x = None

    # -- editing -----------------------------------------------------------------
    def _replace(self, lo, hi, s):
        self.text = self.text[:lo] + s + self.text[hi:]
        self.cursor = lo + len(s)
        self.anchor = None
        self.follow = True
        self.goal_x = None

    def insert(self, s):
        """Typed / pasted text replaces the selection.  A single-line field
        turns line breaks into spaces."""
        s = s.replace('\r', '')
        if not self.multiline:
            s = s.replace('\n', ' ')
        sel = self.selection
        lo, hi = sel if sel else (self.cursor, self.cursor)
        self._replace(lo, hi, s)

    def backspace(self, ctrl=False):
        sel = self.selection
        if sel:
            self._replace(sel[0], sel[1], '')
            return
        if self.cursor == 0:
            return
        lo = self.word_left(self.cursor) if ctrl else self.cursor - 1
        self._replace(lo, self.cursor, '')

    def delete(self, ctrl=False):
        sel = self.selection
        if sel:
            self._replace(sel[0], sel[1], '')
            return
        if self.cursor >= len(self.text):
            return
        hi = self.word_right(self.cursor) if ctrl else self.cursor + 1
        self._replace(self.cursor, hi, '')

    def cut(self):
        s = self.selected_text()
        if s:
            self.insert('')
        return s

    def copy(self):
        return self.selected_text()

    # -- words -------------------------------------------------------------------
    def word_left(self, i):
        """Index where the word-erase span ending at i begins: trailing
        separators first, then the word."""
        t = self.text
        while i > 0 and t[i - 1] in SEPARATORS:
            i -= 1
        while i > 0 and t[i - 1] not in SEPARATORS:
            i -= 1
        return i

    def word_right(self, i):
        t, n = self.text, len(self.text)
        while i < n and t[i] not in SEPARATORS:
            i += 1
        while i < n and t[i] in SEPARATORS:
            i += 1
        return i

    # -- caret movement ----------------------------------------------------------
    def move(self, where, shift=False, ctrl=False, spans=None):
        """'left' | 'right' (Ctrl = by word; a plain move collapses a
        selection to its edge), 'home' | 'end' (Ctrl = whole text; with
        `spans` the visual line, else the hard line)."""
        sel = self.selection
        if where == 'left':
            if sel and not shift:
                i = sel[0]
            else:
                i = self.word_left(self.cursor) if ctrl else self.cursor - 1
        elif where == 'right':
            if sel and not shift:
                i = sel[1]
            else:
                i = self.word_right(self.cursor) if ctrl else self.cursor + 1
        elif where == 'home':
            i = 0 if ctrl else self._line_bounds(spans)[0]
        elif where == 'end':
            i = len(self.text) if ctrl else self._line_bounds(spans)[1]
        else:
            raise ValueError(f"move: unknown direction {where!r}")
        self.set_cursor(i, shift)

    def move_lines(self, delta, shift, spans, measure):
        """Up / down by visual lines keeping the pixel column (the column of
        the last horizontal move, as editors do); past the first / last line
        the caret goes to the text's start / end."""
        k, x = self.caret_pos(spans, measure)
        if self.goal_x is not None:
            x = self.goal_x
        k2 = k + delta
        if k2 < 0:
            i = 0
        elif k2 >= len(spans):
            i = len(self.text)
        else:
            i = self.index_at(spans, measure, x, k2)
        self.set_cursor(i, shift)
        self.goal_x = x

    def _line_bounds(self, spans):
        """(start, end) caret indices of the caret's line."""
        if spans is None:
            s = self.text.rfind('\n', 0, self.cursor) + 1
            e = self.text.find('\n', self.cursor)
            return s, (len(self.text) if e < 0 else e)
        k = self.line_of(self.cursor, spans)
        s, e = spans[k]
        return s, self._line_end(k, spans)

    def _line_end(self, k, spans):
        """The last caret index that still SHOWS on visual line k (a soft
        break's trailing space stays on this line, so the caret goes before
        it -- otherwise it would land on the next line)."""
        s, e = spans[k]
        soft = k + 1 < len(spans) and e < len(self.text) and self.text[e] != '\n'
        if soft and e > s and self.text[e - 1] == ' ':
            return e - 1
        return e

    # -- layout ------------------------------------------------------------------
    def layout(self, measure, width):
        """Visual lines as (start, end) spans (see the module doc).  A
        single-line field is one span.  Cached on (text, width)."""
        key = (self.text, width, self.multiline, id(measure))
        if self._layout_cache is not None and self._layout_cache[0] == key:
            return self._layout_cache[1]
        if not self.multiline:
            spans = [(0, len(self.text))]
        else:
            spans, pos, text = [], 0, self.text
            while True:
                nl = text.find('\n', pos)
                end = len(text) if nl < 0 else nl
                spans.extend(_wrap_span(text, pos, end, measure, width))
                if nl < 0:
                    break
                pos = nl + 1
        self._layout_cache = (key, spans)
        return spans

    def line_of(self, i, spans):
        """Visual line index that holds caret index i."""
        n = len(self.text)
        for k, (s, e) in enumerate(spans):
            if i < e or (i == e and (e == n or self.text[e] == '\n')):
                return k
        return len(spans) - 1

    def caret_pos(self, spans, measure):
        """(visual line, pixel x on that line) of the caret."""
        k = self.line_of(self.cursor, spans)
        s, _e = spans[k]
        return k, measure(self.text[s:self.cursor])

    def index_at(self, spans, measure, x, k):
        """Caret index nearest to pixel x on visual line k (clamped)."""
        k = max(0, min(int(k), len(spans) - 1))
        s, e = spans[k]
        t = self.text
        lo, hi = s, e
        while lo < hi:                       # largest i with measure(t[s:i]) <= x
            mid = (lo + hi + 1) // 2
            if measure(t[s:mid]) <= x:
                lo = mid
            else:
                hi = mid - 1
        i = lo
        if i < e:
            w0, w1 = measure(t[s:i]), measure(t[s:i + 1])
            if x - w0 > w1 - x:
                i += 1
        return min(i, self._line_end(k, spans))


def _wrap_span(text, s, e, measure, width):
    """Greedy wrap of text[s:e] (no '\\n' inside) into (start, end) spans of
    at most `width` pixels: break after the last space that fits, else
    inside the word; at least one character per line."""
    out, start = [], s
    while True:
        if start >= e or measure(text[start:e]) <= width:
            out.append((start, e))
            return out
        lo, hi = start + 1, e - 1
        while lo < hi:                       # largest fit with measure(text[start:fit]) <= width
            mid = (lo + hi + 1) // 2
            if measure(text[start:mid]) <= width:
                lo = mid
            else:
                hi = mid - 1
        fit = lo
        p = text.rfind(' ', start + 1, fit)
        cut = p + 1 if p > start else fit
        out.append((start, cut))
        start = cut
        if start >= e:
            return out
