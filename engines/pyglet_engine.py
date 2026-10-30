import pyglet
from pyglet.gl import glClearColor
from pyglet.window import key
MOVEMENT_PER_SECOND = 120.0

from engines.base_engine import BACKGROUND_COLOR, BaseEngine

_BG = tuple(c / 255 for c in BACKGROUND_COLOR) + (1.0,)



class PygletEngine(BaseEngine):
    def setup(self) -> None:
        self.window = pyglet.window.Window(self.width, self.height, caption=self.title)
        self._should_close = False
        glClearColor(*_BG)

        @self.window.event
        def on_close():
            self._should_close = True
            return pyglet.event.EVENT_HANDLED

        @self.window.event
        def on_key_press(symbol, modifiers):
            if symbol == key.ESCAPE:
                self._should_close = True
                return pyglet.event.EVENT_HANDLED

    def handle_events(self) -> bool:
        # BaseEngine.run() owns the loop, so events are dispatched manually
        # here each frame rather than handing control to pyglet.app.run().
        self.window.dispatch_events()
        return not self._should_close

    def update(self, audio_features: dict, delta_time: float) -> None:
        movement = MOVEMENT_PER_SECOND * delta_time

        if self.growing:
            self.circle_radius += movement
        else:
            self.circle_radius -= movement

        if self.circle_radius >= 150:
            self.circle_radius = 150
            self.growing = False
        elif self.circle_radius <= 50:
            self.circle_radius = 50
            self.growing = True

    def render(self) -> None:
        self.window.clear()

        circle = pyglet.shapes.Circle(
            x=self.width // 2,
            y=self.height // 2,
            radius=self.circle_radius,
            color=(255, 105, 180)
        )
        circle.draw()

        self.window.flip()

    def teardown(self) -> None:
        self.window.close()

    def __init__(self, width: int, height: int, title: str, target_fps: int = 60) -> None:
        super().__init__(width, height, title, target_fps)
        self.circle_radius = 50.0
        self.growing = True
