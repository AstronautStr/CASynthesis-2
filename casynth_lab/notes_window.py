"""Listening notes in a window of their own (2026-09-17, user's request).

The bench's Notes used to be a form drawn INSIDE the pygame window over the field,
so writing blocked the field.  Now `NotesWindow` is a separate operating-system
window (tkinter, pumped from the bench's main loop -- no second thread): drag it
next to the bench, type, click back into the bench to play, click back to write.
Every edit is written at once through `on_change(text)` (the bench writes
<record>/notes.md, small file, atomic replace); closing the window or the
experiment it belongs to (`close()`) never loses text.

The window belongs to ONE record (`rid`); the bench closes it when the record
the notes would go to changes (another record selected, Back to the catalog,
Continue / Save of another record) and when the bench quits.  tkinter must be
driven from the thread that created it: the bench calls `pump()` once per frame
(60 Hz) from its event loop.  Without a display / tkinter (headless tests) the
bench keeps its old in-window form (`available()` is False).
"""
import os

_FONT = ('Segoe UI', 11)
_TITLE_MAX = 60


def available():
    """tkinter importable and a real display: the bench may open the window."""
    if os.environ.get('SDL_VIDEODRIVER', '').lower() == 'dummy':
        return False
    try:
        import tkinter  # noqa: F401
    except Exception:                                   # noqa: BLE001
        return False
    return True


class NotesWindow:
    """One record's notes in a Tk window; `alive` while it is open."""

    def __init__(self, rid, title, text, on_change, on_close=None, position=None, size=(520, 420)):
        import tkinter as tk
        from tkinter import ttk
        self.rid = rid
        self.title = title
        self.text_value = text
        self._on_change = on_change
        self._on_close = on_close
        self._alive = True
        self.root = tk.Tk()
        self.root.title(f"Notes - {title[:_TITLE_MAX]}")
        w, h = size
        if position is not None:
            self.root.geometry(f"{w}x{h}+{int(position[0])}+{int(position[1])}")
        else:
            self.root.geometry(f"{w}x{h}")
        self.root.minsize(320, 200)
        self.root.protocol('WM_DELETE_WINDOW', self._closed_by_user)
        frame = ttk.Frame(self.root, padding=6)
        frame.pack(fill='both', expand=True)
        self.label = ttk.Label(frame, text=f"{title[:_TITLE_MAX]}  -  saved as you type", foreground='#888')
        self.label.pack(anchor='w', pady=(0, 4))
        box = ttk.Frame(frame)
        box.pack(fill='both', expand=True)
        self.widget = tk.Text(box, wrap='word', undo=True, font=_FONT, padx=6, pady=6)
        bar = ttk.Scrollbar(box, orient='vertical', command=self.widget.yview)
        self.widget.configure(yscrollcommand=bar.set)
        bar.pack(side='right', fill='y')
        self.widget.pack(side='left', fill='both', expand=True)
        self.widget.insert('1.0', text)
        self.widget.edit_reset()
        self.widget.edit_modified(False)
        self.widget.bind('<<Modified>>', self._modified)
        self.widget.bind('<Control-a>', self._select_all)
        self.widget.bind('<Control-A>', self._select_all)
        self.widget.focus_set()
        self.root.lift()
        self.root.update()

    # -- events (all on the bench's thread, from pump()) ------------------------------
    def _select_all(self, _ev=None):
        self.widget.tag_add('sel', '1.0', 'end-1c')
        return 'break'

    def _modified(self, _ev=None):
        if not self._alive:
            return
        if not self.widget.edit_modified():
            return
        self.widget.edit_modified(False)
        text = self.widget.get('1.0', 'end-1c')
        if text != self.text_value:
            self.text_value = text
            try:
                self._on_change(text)
            except Exception as e:                      # noqa: BLE001 -- shown, never raised into Tk
                self.label.configure(text=f"NOT saved: {e}"[:90], foreground='#c04040')
                return
            self.label.configure(text=f"{self.title[:_TITLE_MAX]}  -  saved", foreground='#888')

    def _closed_by_user(self):
        self.close()
        if self._on_close is not None:
            self._on_close()

    # -- bench side -----------------------------------------------------------------
    @property
    def alive(self):
        return self._alive

    def pump(self):
        """Process the window's pending events; False once it is gone."""
        if not self._alive:
            return False
        try:
            self.root.update()
        except Exception:                               # noqa: BLE001 -- TclError: destroyed
            self._alive = False
        return self._alive

    def focus(self):
        if self._alive:
            try:
                self.root.lift()
                self.root.focus_force()
                self.widget.focus_set()
            except Exception:                           # noqa: BLE001
                pass

    def close(self):
        """Destroy the window (the text is already written on every edit)."""
        if not self._alive:
            return
        self._alive = False
        try:
            self.root.destroy()
        except Exception:                               # noqa: BLE001
            pass
