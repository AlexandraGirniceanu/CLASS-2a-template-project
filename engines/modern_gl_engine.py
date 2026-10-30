import moderngl_window as mglw

from engines.base_engine import BACKGROUND_COLOR, BaseEngine

_BG = tuple(c / 255 for c in BACKGROUND_COLOR)


class ModernGLEngine(BaseEngine):
    """Uses moderngl-window for windowing, via its lower-level manual-loop API
    (create_window_from_settings) rather than the WindowConfig/run_window_config
    pattern shown in moderngl-window's own docs — loop ownership stays with
    BaseEngine.run() so all three engine tiers share one control-flow shape.
    """

    def setup(self) -> None:
        mglw.settings.WINDOW["class"] = "moderngl_window.context.pyglet.Window"
        mglw.settings.WINDOW["size"] = (self.width, self.height)
        mglw.settings.WINDOW["title"] = self.title
        mglw.settings.WINDOW["resizable"] = True

        self.window = mglw.create_window_from_settings()
        self.ctx = self.window.ctx

        self._should_close = False
        self.window.key_event_func = self._on_key_event

    def _on_key_event(self, key, action, modifiers) -> None:
        if key == self.window.keys.ESCAPE and action == self.window.keys.ACTION_PRESS:
            self._should_close = True

    def handle_events(self) -> bool:
        # moderngl-window's pyglet backend has no standalone event-poll
        # method — new input is pulled in as a side effect of swap_buffers()
        # (called from render()), so this reflects state as of the previous
        # frame's render(), not this instant. That one-frame lag is inherent
        # to the backend, not something to work around here.
        return not (self.window.is_closing or self._should_close)

    def update(self, audio_features: dict, delta_time: float) -> None:
        pass

    def render(self) -> None:
        self.window.clear(*_BG)
        self.window.swap_buffers()

    def teardown(self) -> None:
        self.window.close()
