
import math

import pyglet
from pyglet.gl import glClearColor
from pyglet.window import key

from engines.base_engine import BaseEngine


class PygletEngine(BaseEngine):
    PALETTES = [
        [(100, 20, 61), (169, 30, 98), (233, 79, 155), (255, 181, 213)],
        [(16, 45, 89), (24, 95, 168), (72, 165, 232), (168, 221, 255)],
        [(118, 44, 25), (198, 83, 32), (242, 139, 48), (255, 209, 138)],
        [(57, 32, 94), (112, 68, 163), (177, 122, 224), (228, 201, 255)],
        [(75, 7, 30), (148, 19, 51), (213, 45, 80), (255, 130, 152)],
        [(7, 60, 72), (8, 126, 139), (54, 197, 201), (176, 243, 239)],
    ]

    def __init__(
        self,
        width: int,
        height: int,
        title: str,
        target_fps: int = 60,
    ) -> None:
        super().__init__(width, height, title, target_fps)

        self.palette_index = 0
        self.current_palette = [
            tuple(float(channel) for channel in colour)
            for colour in self.PALETTES[0]
        ]

        self.bar_count = 160
        self.base_radius = min(width, height) * 0.23
        self.bar_heights = [12.0] * self.bar_count
        self.target_heights = [12.0] * self.bar_count

        self.amplitude = 0.0
        self.smoothed_amplitude = 0.0
        self.animation_time = 0.0

        self.background_colour = (12, 5, 15)
        self._should_close = False

    def setup(self) -> None:
        self.window = pyglet.window.Window(
            self.width,
            self.height,
            caption=self.title,
            resizable=True,
            vsync=True,
        )

        glClearColor(0.047, 0.02, 0.059, 1.0)

        @self.window.event
        def on_close():
            self._should_close = True
            return pyglet.event.EVENT_HANDLED

        @self.window.event
        def on_key_press(symbol, modifiers):
            if symbol == key.ESCAPE:
                self._should_close = True
                return pyglet.event.EVENT_HANDLED

            if symbol == key.SPACE:
                self.palette_index = (
                    self.palette_index + 1
                ) % len(self.PALETTES)
                return pyglet.event.EVENT_HANDLED

        @self.window.event
        def on_resize(width, height):
            self.width = width
            self.height = height
            self.base_radius = min(width, height) * 0.23

    def handle_events(self) -> bool:
        self.window.dispatch_events()
        return not self._should_close

    def interpolate_colour(self, palette, position):
        position = max(0.0, min(1.0, position))

        scaled = position * (len(palette) - 1)
        first = min(int(scaled), len(palette) - 2)
        second = first + 1
        fraction = scaled - first

        return tuple(
            int(
                palette[first][channel] * (1.0 - fraction)
                + palette[second][channel] * fraction
            )
            for channel in range(3)
        )

    def update(self, audio_features: dict, delta_time: float) -> None:
        delta_time = max(0.0, min(delta_time, 0.1))
        self.animation_time += delta_time

        raw_amplitude = audio_features.get("amplitude", 0.0)

        try:
            raw_amplitude = float(raw_amplitude)
        except (TypeError, ValueError):
            raw_amplitude = 0.0

        if not math.isfinite(raw_amplitude):
            raw_amplitude = 0.0

        self.amplitude = max(0.0, raw_amplitude)

        amplitude_fraction = 1.0 - math.exp(-8.0 * delta_time)

        self.smoothed_amplitude += (
            self.amplitude - self.smoothed_amplitude
        ) * amplitude_fraction

        target_palette = self.PALETTES[self.palette_index]
        colour_fraction = 1.0 - math.exp(-4.0 * delta_time)

        for index in range(len(self.current_palette)):
            old_colour = self.current_palette[index]
            new_colour = target_palette[index]

            self.current_palette[index] = tuple(
                old_colour[channel]
                + (
                    new_colour[channel] - old_colour[channel]
                ) * colour_fraction
                for channel in range(3)
            )

        frequency_bands = audio_features.get("frequency_bands")

        if frequency_bands is None:
            frequency_bands = audio_features.get("spectrum")

        valid_bands = []

        if frequency_bands is not None:
            try:
                for value in frequency_bands:
                    number = float(value)
                    if math.isfinite(number):
                        valid_bands.append(max(0.0, number))
                    else:
                        valid_bands.append(0.0)
            except (TypeError, ValueError):
                valid_bands = []

        for index in range(self.bar_count):
            angle = (index / self.bar_count) * math.tau

            if valid_bands:
                band_position = (
                    index / self.bar_count
                ) * len(valid_bands)

                band_index = min(
                    int(band_position),
                    len(valid_bands) - 1,
                )

                band_value = valid_bands[band_index]

                energy = min(
                    1.0,
                    math.sqrt(band_value) * 2.5,
                )
            else:
                wave_one = (
                    math.sin(angle * 5.0 + self.animation_time * 1.4)
                    + 1.0
                ) * 0.5

                wave_two = (
                    math.sin(angle * 11.0 - self.animation_time * 0.8)
                    + 1.0
                ) * 0.5

                energy = (
                    wave_one * 0.65 + wave_two * 0.35
                ) * min(1.0, self.smoothed_amplitude * 5.0)

            self.target_heights[index] = (
                8.0 + energy * 125.0
            )

            smoothing = 1.0 - math.exp(-12.0 * delta_time)

            self.bar_heights[index] += (
                self.target_heights[index]
                - self.bar_heights[index]
            ) * smoothing

    def render(self) -> None:
        self.window.clear()

        centre_x = self.width / 2
        centre_y = self.height / 2

        scale = min(self.width / 1280, self.height / 720)
        scale = max(0.3, scale)

        pulse = min(1.0, self.smoothed_amplitude * 3.0)
        radius = self.base_radius + pulse * 12.0 * scale

        batch = pyglet.graphics.Batch()

        outer_circle = pyglet.shapes.Arc(
            x=centre_x,
            y=centre_y,
            radius=radius - 10.0 * scale,
            color=(255, 225, 240),
            thickness=max(1.0, 2.0 * scale),
            segments=180,
            batch=batch,
        )

        inner_circle = pyglet.shapes.Arc(
            x=centre_x,
            y=centre_y,
            radius=radius - 17.0 * scale,
            color=self.interpolate_colour(
                self.current_palette, 0.7
            ),
            thickness=max(1.0, 1.0 * scale),
            segments=180,
            batch=batch,
        )

        bars = []

        for index in range(self.bar_count):
            angle = (
                index / self.bar_count
            ) * math.tau - math.pi / 2

            bar_height = self.bar_heights[index] * scale

            inner_radius = radius + 5.0 * scale
            outer_radius = inner_radius + bar_height

            x1 = centre_x + math.cos(angle) * inner_radius
            y1 = centre_y + math.sin(angle) * inner_radius

            x2 = centre_x + math.cos(angle) * outer_radius
            y2 = centre_y + math.sin(angle) * outer_radius

            colour_position = (
                math.sin(angle + math.pi / 2) + 1.0
            ) * 0.5

            colour = self.interpolate_colour(
                self.current_palette,
                colour_position,
            )

            
            bar = pyglet.shapes.Line(
                x=x1,
                y=y1,
                x2=x2,
                y2=y2,
                thickness=max(1.0, 3.0 * scale),
                color=colour,
                batch=batch,
            )


            bars.append(bar)

        batch.draw()

        self.window.flip()

    def teardown(self) -> None:
        self.window.close()
