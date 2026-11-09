
import math

import moderngl
import moderngl_window as mglw
import numpy as np

from engines.base_engine import BaseEngine


VERTEX_SHADER = """
#version 330

in vec2 in_position;

void main() {
    gl_Position = vec4(in_position, 0.0, 1.0);
}
"""


FRAGMENT_SHADER = """
#version 330

uniform vec2 resolution;
uniform float time;
uniform float bass;
uniform float mid;
uniform float treble;
uniform float onset;
uniform vec3 palette[4];

out vec4 fragColor;

vec3 gradient(float value) {
    float p = clamp(value, 0.0, 0.99999) * 3.0;
    int i = int(floor(p));

    return mix(
        palette[i],
        palette[i + 1],
        fract(p)
    );
}

float line_pattern(
    float coordinate,
    float frequency,
    float width
) {
    float phase = abs(sin(coordinate * frequency));
    float aa = max(fwidth(phase) * 1.4, 0.004);

    return 1.0 - smoothstep(
        width,
        width + aa,
        phase
    );
}

void main() {
    vec2 uv = (
        gl_FragCoord.xy - 0.5 * resolution
    ) / min(resolution.x, resolution.y);

    float r = length(uv);

    vec3 background = mix(
        vec3(0.008, 0.007, 0.016)
            + palette[0] * 0.075,
        vec3(0.018, 0.012, 0.027)
            + palette[1] * 0.13,
        smoothstep(-0.6, 0.65, uv.y)
    );

    float halo = exp(
        -dot(
            uv * vec2(1.05, 1.25),
            uv * vec2(1.05, 1.25)
        ) * 6.0
    );

    background += (
        palette[2]
        * halo
        * (0.105 + bass * 0.07)
    );

    vec2 light_position = uv - vec2(-0.32, 0.18);

    background += (
        palette[1]
        * exp(-dot(light_position, light_position) * 7.0)
        * 0.035
    );

    vec2 grid_pos = uv * 18.0;
    vec2 grid_distance = abs(fract(grid_pos) - 0.5);
    vec2 grid_aa = max(fwidth(grid_pos), vec2(0.01));

    float grid = max(
        1.0 - smoothstep(
            0.48 - grid_aa.x,
            0.5,
            grid_distance.x
        ),
        1.0 - smoothstep(
            0.48 - grid_aa.y,
            0.5,
            grid_distance.y
        )
    );

    background += (
        palette[2]
        * grid
        * 0.022
        * (0.4 + 0.6 * halo)
    );

    float rotation = time * 0.11;

    float ca = cos(rotation);
    float sa = sin(rotation);

    vec2 p = mat2(
        ca, -sa,
        sa, ca
    ) * uv;

    float angle = atan(p.y, p.x);
    float distance_to_centre = length(p);

    float slow_wave = sin(
        angle * 3.0
        - time * 0.55
        + 0.5 * sin(time * 0.27)
    );

    float medium_wave = sin(
        angle * 5.0
        + time * 0.68
        + 0.45 * sin(angle * 2.0 - time * 0.4)
    );

    float fine_wave = sin(
        angle * 9.0 - time * 0.9
    );

    float deformation = (
        slow_wave * (0.013 + bass * 0.024)
        + medium_wave * (0.010 + mid * 0.025)
        + fine_wave * (0.003 + treble * 0.008)
    );

    float radius = (
        0.307
        + deformation
        + bass * 0.032
        + onset * 0.008
    );

    float signed_distance = (
        distance_to_centre - radius
    );

    float mask = 1.0 - smoothstep(
        -0.005,
        0.012,
        signed_distance
    );

    vec2 sphere_uv = p / max(radius, 0.1);

    float radial_sq = dot(
        sphere_uv,
        sphere_uv
    );

    float z = sqrt(
        max(0.0, 1.0 - radial_sq)
    );

    float latitude = acos(
        clamp(sphere_uv.y, -1.0, 1.0)
    );

    float longitude = atan(
        z,
        sphere_uv.x
    );

    float twist = (
        0.13 * sin(
            time * 0.45 + latitude * 2.0
        )
    );

    float lat_lines = line_pattern(
        latitude
            + 0.015 * sin(
                longitude * 3.0 + time * 0.35
            ),
        23.0,
        0.047
    );

    float lon_lines = line_pattern(
        longitude + twist,
        35.0,
        0.048
    );

    float diagonal_lines = line_pattern(
        latitude * 0.8
            + longitude * 0.55
            + twist,
        28.0,
        0.035
    );

    float wire = max(
        lat_lines,
        max(
            lon_lines,
            diagonal_lines * 0.55
        )
    );

    float surface_light = (
        0.35 + 0.65 * z
    );

    float shimmer = (
        0.88
        + 0.12 * sin(
            time * 1.1 + longitude * 7.0
        )
    );

    vec3 mesh_colour = gradient(
        clamp(
            0.42
                + 0.42 * sphere_uv.y
                + 0.16 * z,
            0.0,
            1.0
        )
    );

    float glow = exp(
        -abs(signed_distance) * 26.0
    );

    float rim = exp(
        -abs(signed_distance) * 95.0
    );

    float inner_haze = (
        exp(-radial_sq * 2.4) * mask
    );

    vec3 colour = background;

    colour += (
        palette[1]
        * inner_haze
        * 0.075
    );

    colour += (
        mesh_colour
        * wire
        * mask
        * surface_light
        * shimmer
        * (0.8 + 0.25 * mid)
    );

    colour += (
        palette[2]
        * glow
        * (0.12 + 0.09 * bass)
    );

    colour += (
        palette[3]
        * rim
        * 0.31
    );

    float vignette = (
        1.0
        - 0.32 * smoothstep(
            0.25,
            0.95,
            r
        )
    );

    colour *= vignette;

    colour = 1.0 - exp(-colour);

    fragColor = vec4(
        colour,
        1.0
    );
}
"""


class ModernGLEngine(BaseEngine):
    PALETTES = [
        [
            (100, 20, 61),
            (169, 30, 98),
            (233, 79, 155),
            (255, 181, 213),
        ],
        [
            (16, 45, 89),
            (24, 95, 168),
            (72, 165, 232),
            (168, 221, 255),
        ],
        [
            (118, 44, 25),
            (198, 83, 32),
            (242, 139, 48),
            (255, 209, 138),
        ],
        [
            (57, 32, 94),
            (112, 68, 163),
            (177, 122, 224),
            (228, 201, 255),
        ],
        [
            (75, 7, 30),
            (148, 19, 51),
            (213, 45, 80),
            (255, 130, 152),
        ],
        [
            (7, 60, 72),
            (8, 126, 139),
            (54, 197, 201),
            (176, 243, 239),
        ],
    ]

    def __init__(
        self,
        width: int,
        height: int,
        title: str,
        target_fps: int = 60,
    ) -> None:
        super().__init__(
            width,
            height,
            title,
            target_fps,
        )

        self.palette_index = 0

        self.palette_data = (
            np.asarray(
                self.PALETTES,
                dtype=np.float32,
            ) / 255.0
        )

        self.current_palette = (
            self.palette_data[0].copy()
        )

        self.animation_time = 0.0

        self.bass_level = 0.0
        self.mid_level = 0.0
        self.treble_level = 0.0
        self.onset_level = 0.0

        self._should_close = False

    def setup(self) -> None:
        mglw.settings.WINDOW["class"] = (
            "moderngl_window.context.pyglet.Window"
        )

        mglw.settings.WINDOW["size"] = (
            self.width,
            self.height,
        )

        mglw.settings.WINDOW["title"] = self.title
        mglw.settings.WINDOW["resizable"] = True

        self.window = (
            mglw.create_window_from_settings()
        )

        self.ctx = self.window.ctx

        self.window.key_event_func = (
            self._on_key_event
        )

        self.program = self.ctx.program(
            vertex_shader=VERTEX_SHADER,
            fragment_shader=FRAGMENT_SHADER,
        )

        vertices = np.array(
            [
                -1.0, -1.0,
                 1.0, -1.0,
                -1.0,  1.0,
                 1.0,  1.0,
            ],
            dtype="f4",
        )

        self.vbo = self.ctx.buffer(
            vertices.tobytes()
        )

        self.vao = self.ctx.vertex_array(
            self.program,
            [
                (
                    self.vbo,
                    "2f",
                    "in_position",
                )
            ],
        )

        self.update_uniforms()

    def _on_key_event(
        self,
        key,
        action,
        modifiers,
    ) -> None:
        if action != self.window.keys.ACTION_PRESS:
            return

        if key == self.window.keys.ESCAPE:
            self._should_close = True

        elif key == self.window.keys.SPACE:
            self.palette_index = (
                self.palette_index + 1
            ) % len(self.PALETTES)

    def handle_events(self) -> bool:
        return not (
            self.window.is_closing
            or self._should_close
        )

    def update(
        self,
        audio_features: dict,
        delta_time: float,
    ) -> None:
        delta_time = min(
            max(delta_time, 0.0),
            0.1,
        )

        self.animation_time += delta_time

        colour_fraction = (
            1.0 - math.exp(-3.5 * delta_time)
        )

        self.current_palette += (
            self.palette_data[self.palette_index]
            - self.current_palette
        ) * colour_fraction

        def normalise_feature(
            name: str,
            divisor: float,
        ) -> float:
            value = float(
                audio_features.get(name, 0.0)
            )

            if not math.isfinite(value):
                return 0.0

            return min(
                1.0,
                math.log1p(max(0.0, value)) / divisor,
            )

        targets = (
            normalise_feature("bass", 8.0),
            normalise_feature("mid", 8.0),
            normalise_feature("treble", 8.0),
            normalise_feature("onset", 7.0),
        )

        smoothing = (
            1.0 - math.exp(-7.0 * delta_time)
        )

        self.bass_level += (
            targets[0] - self.bass_level
        ) * smoothing

        self.mid_level += (
            targets[1] - self.mid_level
        ) * smoothing

        self.treble_level += (
            targets[2] - self.treble_level
        ) * smoothing

        self.onset_level += (
            targets[3] - self.onset_level
        ) * smoothing

        self.update_uniforms()

    def update_uniforms(self) -> None:
        self.program["time"].value = (
            self.animation_time
        )

        self.program["bass"].value = (
            self.bass_level
        )

        self.program["mid"].value = (
            self.mid_level
        )

        self.program["treble"].value = (
            self.treble_level
        )

        self.program["onset"].value = (
            self.onset_level
        )

        self.program["resolution"].value = (
            float(self.width),
            float(self.height),
        )

        self.program["palette"].write(
            self.current_palette.astype(
                "f4"
            ).tobytes()
        )

    def render(self) -> None:
        self.ctx.clear(
            0.02,
            0.01,
            0.03,
            1.0,
        )

        self.vao.render(
            mode=moderngl.TRIANGLE_STRIP
        )

        self.window.swap_buffers()

        size = self.window.size

        if tuple(size) != (
            self.width,
            self.height,
        ):
            self.width, self.height = size

            self.ctx.viewport = (
                0,
                0,
                self.width,
                self.height,
            )

    def teardown(self) -> None:
        self.vao.release()
        self.vbo.release()
        self.program.release()
        self.window.close()
