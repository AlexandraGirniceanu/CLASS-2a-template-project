
import math

import numpy as np
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
        self.palette = np.array(self.PALETTES[0], dtype=float)
        self.time = 0.0

        self.amplitude = 0.0
        self.bass = 0.0
        self.mid = 0.0
        self.treble = 0.0
        self.onset = 0.0

        self.bass_level = 0.0
        self.mid_level = 0.0
        self.treble_level = 0.0
        self.onset_level = 0.0

        self.rings = 28
        self.segments = 64

        self.vertices = None
        self.edges = None
        self.mesh_lines = []
        self.glow_lines = []
        self.grid_lines = []
        self.background_shapes = []
        self._should_close = False

    def setup(self) -> None:
        self.window = pyglet.window.Window(
            self.width,
            self.height,
            caption=self.title,
            resizable=True,
            vsync=True,
        )

        glClearColor(0.035, 0.012, 0.045, 1.0)

        self.batch = pyglet.graphics.Batch()
        self.background_batch = pyglet.graphics.Batch()

        self.create_mesh()
        self.create_background()

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
            self.update_background_geometry()

    def create_mesh(self) -> None:
        theta = np.linspace(
            0.0,
            math.pi,
            self.rings + 1,
        )

        phi = np.linspace(
            0.0,
            math.tau,
            self.segments,
            endpoint=False,
        )

        theta_grid, phi_grid = np.meshgrid(
            theta,
            phi,
            indexing="ij",
        )

        self.theta = theta_grid.ravel()
        self.phi = phi_grid.ravel()

        self.vertex_count = len(self.theta)

        edges = []

        for ring in range(self.rings + 1):
            for segment in range(self.segments):
                index = ring * self.segments + segment

                next_segment = (
                    ring * self.segments
                    + (segment + 1) % self.segments
                )

                edges.append((index, next_segment))

                if ring < self.rings:
                    next_ring = (
                        (ring + 1) * self.segments
                        + segment
                    )

                    diagonal = (
                        (ring + 1) * self.segments
                        + (segment + 1) % self.segments
                    )

                    edges.append((index, next_ring))
                    edges.append((index, diagonal))

        self.edges = np.array(edges, dtype=np.int32)

        self.mesh_lines = []
        self.glow_lines = []

        for index in range(len(self.edges)):
            glow = pyglet.shapes.Line(
                0, 0, 0, 0,
                thickness=3.0,
                color=(100, 30, 80),
                batch=self.batch,
            )

            line = pyglet.shapes.Line(
                0, 0, 0, 0,
                thickness=1.0,
                color=(255, 140, 200),
                batch=self.batch,
            )

            glow.opacity = 25
            line.opacity = 105

            self.glow_lines.append(glow)
            self.mesh_lines.append(line)

    def create_background(self) -> None:
        self.background_shapes = []
        self.grid_lines = []

        for index in range(24):
            rectangle = pyglet.shapes.Rectangle(
                0, 0,
                self.width,
                self.height,
                color=(20, 5, 25),
                batch=self.background_batch,
            )

            self.background_shapes.append(rectangle)

        for index in range(40):
            line = pyglet.shapes.Line(
                0, 0, 0, 0,
                thickness=1.0,
                color=(120, 70, 110),
                batch=self.background_batch,
            )

            line.opacity = 12
            self.grid_lines.append(line)

        self.update_background_geometry()

    def update_background_geometry(self) -> None:
        if not self.background_shapes:
            return

        for index, rectangle in enumerate(self.background_shapes):
            rectangle.x = 0
            rectangle.y = (
                index * self.height
                / len(self.background_shapes)
            )

            rectangle.width = self.width
            rectangle.height = (
                self.height / len(self.background_shapes) + 2
            )

        spacing = max(35, int(min(self.width, self.height) / 14))

        for index, line in enumerate(self.grid_lines):
            if index < 20:
                x = (
                    index - 10
                ) * spacing + self.width / 2

                line.x = x
                line.y = 0
                line.x2 = x
                line.y2 = self.height

            else:
                y = (
                    index - 30
                ) * spacing + self.height / 2

                line.x = 0
                line.y = y
                line.x2 = self.width
                line.y2 = y

    def handle_events(self) -> bool:
        self.window.dispatch_events()
        return not self._should_close

    def update(self, audio_features: dict, delta_time: float) -> None:
        delta_time = min(max(delta_time, 0.0), 0.1)
        self.time += delta_time

        target_palette = np.array(
            self.PALETTES[self.palette_index],
            dtype=float,
        )

        colour_fraction = 1.0 - math.exp(
            -3.5 * delta_time
        )

        self.palette += (
            target_palette - self.palette
        ) * colour_fraction

        self.amplitude = max(
            0.0,
            float(audio_features.get("amplitude", 0.0)),
        )

        self.bass = max(
            0.0,
            float(audio_features.get("bass", 0.0)),
        )

        self.mid = max(
            0.0,
            float(audio_features.get("mid", 0.0)),
        )

        self.treble = max(
            0.0,
            float(audio_features.get("treble", 0.0)),
        )

        self.onset = max(
            0.0,
            float(audio_features.get("onset", 0.0)),
        )

        bass_target = min(
            1.0,
            math.log1p(self.bass) / 8.0,
        )

        mid_target = min(
            1.0,
            math.log1p(self.mid) / 8.0,
        )

        treble_target = min(
            1.0,
            math.log1p(self.treble) / 8.0,
        )

        onset_target = min(
            1.0,
            math.log1p(self.onset) / 7.0,
        )

        smoothing = 1.0 - math.exp(
            -7.0 * delta_time
        )

        self.bass_level += (
            bass_target - self.bass_level
        ) * smoothing

        self.mid_level += (
            mid_target - self.mid_level
        ) * smoothing

        self.treble_level += (
            treble_target - self.treble_level
        ) * smoothing

        self.onset_level += (
            onset_target - self.onset_level
        ) * smoothing

        self.update_mesh()
        self.update_background_colours()

    def update_mesh(self) -> None:
        t = self.time

        theta = self.theta
        phi = self.phi

        bass_strength = self.bass_level
        mid_strength = self.mid_level
        treble_strength = self.treble_level

        wave_one = np.sin(
            theta * 4.0
            + phi * 3.0
            + t * 0.7
        )

        wave_two = np.sin(
            theta * 7.0
            - phi * 5.0
            - t * 0.9
        )

        wave_three = np.sin(
            theta * 12.0
            + phi * 9.0
            + t * 1.5
        )

        deformation = (
            wave_one * (0.06 + mid_strength * 0.16)
            + wave_two * (0.04 + bass_strength * 0.13)
            + wave_three * (0.01 + treble_strength * 0.05)
        )

        pulse = (
            1.0
            + bass_strength * 0.18
            + self.onset_level * 0.035
        )

        radius = (
            1.0 + deformation
        ) * pulse

        x = (
            radius
            * np.sin(theta)
            * np.cos(phi)
        )

        y = (
            radius
            * np.cos(theta)
        )

        z = (
            radius
            * np.sin(theta)
            * np.sin(phi)
        )

        rotation = t * 0.12

        rotated_x = (
            x * math.cos(rotation)
            + z * math.sin(rotation)
        )

        rotated_z = (
            -x * math.sin(rotation)
            + z * math.cos(rotation)
        )

        tilt = 0.23

        rotated_y = (
            y * math.cos(tilt)
            - rotated_z * math.sin(tilt)
        )

        depth = (
            y * math.sin(tilt)
            + rotated_z * math.cos(tilt)
        )

        scale = min(
            self.width,
            self.height,
        ) * 0.32

        perspective = 3.5 / (
            3.5 - depth * 0.22
        )

        screen_x = (
            self.width / 2
            + rotated_x * scale * perspective
        )

        screen_y = (
            self.height / 2
            + rotated_y * scale * perspective
        )

        vertices = np.column_stack(
            (screen_x, screen_y)
        )

        start_points = vertices[self.edges[:, 0]]
        end_points = vertices[self.edges[:, 1]]

        average_depth = (
            depth[self.edges[:, 0]]
            + depth[self.edges[:, 1]]
        ) * 0.5

        depth_normalised = np.clip(
            (average_depth + 1.5) / 3.0,
            0.0,
            1.0,
        )

        palette_indices = (
            depth_normalised
            * (len(self.palette) - 1)
        )

        first_indices = np.minimum(
            palette_indices.astype(int),
            len(self.palette) - 2,
        )

        fractions = (
            palette_indices - first_indices
        )[:, None]

        colours = (
            self.palette[first_indices]
            * (1.0 - fractions)
            + self.palette[first_indices + 1]
            * fractions
        )

        for index, line in enumerate(self.mesh_lines):
            x1, y1 = start_points[index]
            x2, y2 = end_points[index]

            line.x = float(x1)
            line.y = float(y1)
            line.x2 = float(x2)
            line.y2 = float(y2)

            colour = tuple(
                int(value)
                for value in colours[index]
            )

            line.color = colour

            line.opacity = int(
                45 + depth_normalised[index] * 125
            )

            glow = self.glow_lines[index]

            glow.x = float(x1)
            glow.y = float(y1)
            glow.x2 = float(x2)
            glow.y2 = float(y2)

            glow.color = colour
            glow.opacity = int(
                8 + depth_normalised[index] * 20
            )

    def update_background_colours(self) -> None:
        base = self.palette[0]
        highlight = self.palette[2]

        for index, rectangle in enumerate(self.background_shapes):
            position = (
                index / len(self.background_shapes)
            )

            glow = math.exp(
                -((position - 0.55) ** 2) / 0.085
            )

            colour = (
                base * 0.10
                + highlight * glow * 0.11
                + np.array([6, 4, 12])
            )

            rectangle.color = tuple(
                int(min(255, value))
                for value in colour
            )

        grid_colour = tuple(
            int(value)
            for value in self.palette[1]
        )

        for line in self.grid_lines:
            line.color = grid_colour

    def render(self) -> None:
        self.window.clear()
        self.background_batch.draw()
        self.batch.draw()
        self.window.flip()

    def teardown(self) -> None:
        self.window.close()
