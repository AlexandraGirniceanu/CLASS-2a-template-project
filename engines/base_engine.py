
"""Shared lifecycle every rendering engine implements, plus the run loop itself.

The loop/timing/frame-budget mechanism is identical across all three engine
tiers on purpose. The specific drawing API belongs to each engine.
"""

import time
from abc import ABC, abstractmethod
from collections.abc import Callable

BACKGROUND_COLOR = (18, 18, 18)


class BaseEngine(ABC):
    def __init__(
        self,
        width: int,
        height: int,
        title: str,
        target_fps: int = 60
    ) -> None:
        self.width = width
        self.height = height
        self.title = title
        self.target_fps = target_fps

    @abstractmethod
    def setup(self) -> None:
        """Create the window and rendering context."""

    @abstractmethod
    def handle_events(self) -> bool:
        """Return False when the application should close."""

    @abstractmethod
    def update(self, audio_features: dict, delta_time: float) -> None:
        """Advance the application state."""

    @abstractmethod
    def render(self) -> None:
        """Draw the current frame."""

    @abstractmethod
    def teardown(self) -> None:
        """Release engine resources."""

    def run(self, feature_source: Callable[[], dict]) -> None:
        """Run the shared engine lifecycle and frame loop."""

        frame_duration = 1.0 / self.target_fps

        self.setup()

        try:
            running = True
            previous_frame_time = time.perf_counter()

            # FPS measurement variables must be outside the loop.
            fps_elapsed = 0.0
            fps_frames = 0
            delta_time_printed = False

            while running:
                frame_start = time.perf_counter()

                delta_time = frame_start - previous_frame_time
                previous_frame_time = frame_start

                # Print delta_time once, in seconds.
                if not delta_time_printed and fps_frames >= 10:
                    print(f"delta_time: {delta_time:.6f} seconds")
                    delta_time_printed = True

                running = self.handle_events()

                if not running:
                    break

                audio_features = feature_source()

                MAX_DELTA_TIME = 0.1
                capped_delta_time = min(delta_time, MAX_DELTA_TIME)

                self.update(audio_features, capped_delta_time)
                self.render()

                # Measure average FPS over approximately one second.
                fps_elapsed += delta_time
                fps_frames += 1

                if fps_elapsed >= 1.0:
                    average_fps = fps_frames / fps_elapsed
                    print(f"Average FPS: {average_fps:.2f}")

                    fps_elapsed = 0.0
                    fps_frames = 0

                # Maintain the target frame rate.
                elapsed = time.perf_counter() - frame_start
                remaining = frame_duration - elapsed

                if remaining > 0:
                    time.sleep(remaining)

        finally:
            self.teardown()
