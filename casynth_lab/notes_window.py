"""The listening notes in a window of their own (2026-09-17, user's request).

The bench's Notes form used to be drawn INSIDE the bench window over the field,
so writing blocked the field.  Now the SAME form (same drawing, same text model,
same keys) lives in a second pygame window: a real window of the operating
system with its title bar (minimise / close), moved next to the bench, taking
the keyboard focus when clicked and giving it back when the bench is clicked.
The bench draws into `surface` every frame and routes to the form the events
that carry this window (`owns(event)`); the OS close button arrives as
WINDOWCLOSE.  Nothing here touches audio; this module is outside the sound set.
"""

DEFAULT_SIZE = (560, 440)
MIN_SIZE = (320, 200)


class NotesWindow:
    """One record's notes form in its own pygame.Window."""

    def __init__(self, rid, title, size=DEFAULT_SIZE, position=None):
        import pygame
        self.rid = rid
        self.title = title
        kw = dict(size=size, resizable=True)
        if position is not None:
            kw['position'] = (int(position[0]), int(position[1]))
        self.window = pygame.Window(f"Notes: {title}"[:80], **kw)
        try:
            self.window.minimum_size = MIN_SIZE
        except Exception:                               # noqa: BLE001 -- older pygame
            pass
        self._alive = True

    @property
    def alive(self):
        return self._alive

    @property
    def size(self):
        return tuple(self.window.size) if self._alive else (0, 0)

    @property
    def surface(self):
        """The window's drawing surface (fetched each frame: a resize replaces it)."""
        return self.window.get_surface()

    def flip(self):
        if self._alive:
            self.window.flip()

    def owns(self, event):
        """Does this pygame event belong to this window?"""
        return self._alive and getattr(event, 'window', None) is self.window

    def focus(self):
        if self._alive:
            try:
                self.window.focus()
            except Exception:                           # noqa: BLE001
                pass

    def close(self):
        if not self._alive:
            return
        self._alive = False
        try:
            self.window.destroy()
        except Exception:                               # noqa: BLE001
            pass
