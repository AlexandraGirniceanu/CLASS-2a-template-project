
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
uniform float amplitude;
uniform vec3 palette[4];

out vec4 fragColor;

#define PI 3.14159265359

vec3 gradient(float t) {
    float p = clamp(t, 0.0, 0.9999) * 3.0;
    int i = int(floor(p));
    return mix(palette[i], palette[i + 1], fract(p));
}

float hash21(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

float meshLine(float v, float frequency, float thickness) {
    float s = abs(sin(v * frequency));
    float aa = max(fwidth(s) * 1.1, 0.003);
    return 1.0 - smoothstep(thickness, thickness + aa, s);
}

float waveShape(float a) {
    return (
        sin(a * 3.0 - time * 0.30)
        * (0.012 + bass * 0.055)
        +
        sin(a * 5.0 + time * 0.42)
        * (0.009 + mid * 0.045)
        +
        sin(a * 8.0 - time * 0.60)
        * (0.003 + treble * 0.017)
    );
}

float backgroundMesh(vec2 uv) {
    float result = 0.0;

    for (int i = 0; i < 12; i++) {
        float f = float(i);

        float curve =
            0.13 * sin(uv.x * 3.0 + time * 0.12)
            +
            0.065 * sin(
                uv.x * 6.0 - time * 0.16 + f * 0.09
            )
            +
            (f - 5.5) * 0.017;

        float d = abs(uv.y - curve);

        result += exp(-d * 115.0);
    }

    return result;
}

void main() {
    vec2 uv = (
        gl_FragCoord.xy - resolution * 0.5
    ) / min(resolution.x, resolution.y);

    float r = length(uv);

    float energy = clamp(
        amplitude * 0.6 + bass * 0.4,
        0.0,
        1.0
    );

    vec3 darkPlum = vec3(0.009, 0.003, 0.015);
    vec3 burgundy = palette[0] * 0.18;
    vec3 magenta = palette[1] * 0.25;

    vec3 colour = mix(
        darkPlum + burgundy,
        darkPlum + magenta,
        smoothstep(-0.65, 0.75, uv.y)
    );

    float backgroundLight = exp(
        -dot(
            uv * vec2(0.85, 1.0),
            uv * vec2(0.85, 1.0)
        ) * 2.0
    );

    colour += (
        palette[1]
        * backgroundLight
        * (0.24 + energy * 0.18)
    );

    float upperGlow = exp(
        -length(uv - vec2(0.0, 0.17)) * 4.0
    );

    colour += palette[2] * upperGlow * 0.075;

    float waves = backgroundMesh(uv);

    float sideFade = smoothstep(
        0.18,
        0.55,
        abs(uv.x)
    );

    colour += (
        palette[2]
        * waves
        * sideFade
        * (0.075 + amplitude * 0.05)
    );

    vec2 starGrid = uv * 38.0;
    vec2 starCell = floor(starGrid);
    vec2 starLocal = fract(starGrid) - 0.5;

    float randomStar = hash21(starCell);

    float star = step(0.978, randomStar);

    float starGlow = exp(
        -length(starLocal) * 18.0
    );

    float twinkle = (
        0.65
        + 0.35 * sin(
            time * 0.7 + randomStar * 20.0
        )
    );

    colour += (
        palette[3]
        * star
        * starGlow
        * twinkle
        * 0.65
    );

    float floorGlow = exp(
        -pow(uv.x / 0.24, 2.0) * 2.0
        -pow((uv.y + 0.46) / 0.055, 2.0) * 2.0
    );

    float floorCore = exp(
        -pow(uv.x / 0.16, 2.0) * 2.0
        -pow((uv.y + 0.46) / 0.012, 2.0) * 2.0
    );

    colour += palette[2] * floorGlow * 0.55;
    colour += palette[3] * floorCore * 0.55;

    float rotation = time * 0.035;

    float ca = cos(rotation);
    float sa = sin(rotation);

    vec2 p = mat2(
        ca, -sa,
        sa, ca
    ) * uv;

    float angle = atan(p.y, p.x);
    float distanceToCentre = length(p);

    float radius =
        0.335
        + amplitude * 0.06
        + bass * 0.075
        + onset * 0.018
        + waveShape(angle);

    float sd = distanceToCentre - radius;

    float mask = 1.0 - smoothstep(
        -0.005,
        0.009,
        sd
    );

    vec2 sphereUV = p / max(radius, 0.1);

    float radialSq = dot(sphereUV, sphereUV);

    float z = sqrt(
        max(0.0, 1.0 - radialSq)
    );

    float latitude = acos(
        clamp(sphereUV.y, -1.0, 1.0)
    );

    float longitude = atan(
        z,
        sphereUV.x
    );

    float twist =
        0.13 * sin(
            latitude * 3.0 + time * 0.24
        )
        +
        mid * 0.34 * sin(
            latitude * 4.0 + time * 0.55
        );

    float ripple = treble * 0.055 * sin(
        latitude * 15.0 - time * 1.4
    );

    float latLines = meshLine(
        latitude + ripple,
        16.0,
        0.045
    );

    float lonLines = meshLine(
        longitude + twist,
        23.0,
        0.045
    );

    float diagonalLines = meshLine(
        latitude * 0.8
        + longitude * 0.55
        + twist,
        18.0,
        0.025
    );

    float wire = max(
        latLines,
        max(lonLines, diagonalLines * 0.42)
    );

    float innerWire = max(
        meshLine(
            latitude + time * 0.018,
            12.0,
            0.035
        ),
        meshLine(
            longitude - time * 0.016,
            17.0,
            0.035
        )
    );

    float innerMask = (
        1.0 - smoothstep(
            0.48,
            0.82,
            sqrt(radialSq)
        )
    );

    vec3 meshColour = gradient(
        clamp(
            0.38
            + 0.40 * sphereUV.y
            + 0.22 * z,
            0.0,
            1.0
        )
    );

    float surfaceLight = 0.55 + 0.45 * z;

    float innerHaze = exp(
        -radialSq * 2.0
    ) * mask;

    colour += (
        palette[2]
        * innerHaze
        * (0.16 + amplitude * 0.20)
    );

    colour += (
        palette[3]
        * innerWire
        * innerMask
        * mask
        * 0.34
    );

    colour += (
        meshColour
        * wire
        * mask
        * surfaceLight
        * (1.75 + amplitude * 0.85)
    );

    float meshHighlight = (
        pow(wire, 2.0)
        * mask
        * (0.3 + 0.7 * z)
    );

    colour += (
        palette[3]
        * meshHighlight
        * (0.35 + energy * 0.25)
    );

    float wideBloom = exp(
        -abs(sd) * 9.0
    );

    float mediumBloom = exp(
        -abs(sd) * 23.0
    );

    float closeBloom = exp(
        -abs(sd) * 65.0
    );

    float sharpRim = exp(
        -abs(sd) * 190.0
    );

    colour += (
        palette[1]
        * wideBloom
        * (0.25 + energy * 0.25)
    );

    colour += (
        palette[2]
        * mediumBloom
        * (0.38 + energy * 0.40)
    );

    colour += (
        palette[3]
        * closeBloom
        * (0.50 + amplitude * 0.35)
    );

    colour += (
        palette[3]
        * sharpRim
        * (0.65 + onset * 0.30)
    );

    float topHighlight = smoothstep(
        -0.1,
        0.8,
        sphereUV.y
    );

    colour += (
        palette[3]
        * sharpRim
        * topHighlight
        * 0.50
    );

    float vignette = 1.0 - 0.28 * smoothstep(
        0.30,
        1.0,
        r
    );

    colour *= vignette;

    colour = 1.0 - exp(-colour);

    fragColor = vec4(colour, 1.0);
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
        self.amplitude_level = 0.0

        self.feature_scales = {
            "bass": 1.0,
            "mid": 1.0,
            "treble": 1.0,
            "onset": 1.0,
            "amplitude": 0.05,
        }

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

        def normalise_feature(name: str) -> float:
            value = float(
                audio_features.get(name, 0.0)
            )

            if not math.isfinite(value):
                return 0.0

            value = max(0.0, value)

            scale = self.feature_scales[name]

            if value > scale:
                scale = value
            else:
                scale *= math.exp(
                    -0.15 * delta_time
                )

            scale = max(scale, 0.000001)
            self.feature_scales[name] = scale

            return min(1.0, value / scale)

        targets = {
            "bass": normalise_feature("bass"),
            "mid": normalise_feature("mid"),
            "treble": normalise_feature("treble"),
            "onset": normalise_feature("onset"),
            "amplitude": normalise_feature("amplitude"),
        }

        def smooth(
            current: float,
            target: float,
        ) -> float:
            rate = (
                14.0 if target > current else 5.0
            )

            fraction = (
                1.0 - math.exp(
                    -rate * delta_time
                )
            )

            return current + (
                target - current
            ) * fraction

        self.bass_level = smooth(
            self.bass_level,
            targets["bass"],
        )

        self.mid_level = smooth(
            self.mid_level,
            targets["mid"],
        )

        self.treble_level = smooth(
            self.treble_level,
            targets["treble"],
        )

        self.onset_level = smooth(
            self.onset_level,
            targets["onset"],
        )

        self.amplitude_level = smooth(
            self.amplitude_level,
            targets["amplitude"],
        )

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

        self.program["amplitude"].value = (
            self.amplitude_level
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
