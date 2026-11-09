
import time
from abc import ABC, abstractmethod
from collections.abc import Callable


BACKGROUND_COLOR = (12, 5, 15)


class BaseEngine(ABC):
    def __init__(
        self,
        width: int,
        height: int,
        title: str,
        target_fps: int = 60,
    ) -> None:
        self.width = width
        self.height = height
        self.title = title
        self.target_fps = target_fps

    @abstractmethod
    def setup(self) -> None:
        pass

    @abstractmethod
    def handle_events(self) -> bool:
        pass

    @abstractmethod
    def update(self, audio_features: dict, delta_time: float) -> None:
        pass

    @abstractmethod
    def render(self) -> None:
        pass

    @abstractmethod
    def teardown(self) -> None:
        pass

    def run(self, feature_source: Callable[[], dict]) -> None:
        frame_duration = 1.0 / self.target_fps
        max_delta_time = 0.1

        self.setup()

        try:
            running = True
            previous_frame_time = time.perf_counter()

            fps_elapsed = 0.0
            fps_frames = 0

            while running:
                frame_start = time.perf_counter()

                delta_time = frame_start - previous_frame_time
                previous_frame_time = frame_start

                running = self.handle_events()

                if not running:
                    break

                audio_features = feature_source()

                capped_delta_time = min(
                    max(delta_time, 0.0),
                    max_delta_time,
                )

                self.update(
                    audio_features,
                    capped_delta_time,
                )

                self.render()

                fps_elapsed += delta_time
                fps_frames += 1

                if fps_elapsed >= 1.0:
                    average_fps = fps_frames / fps_elapsed

                    print(
                        f"Average FPS: {average_fps:.2f}"
                    )

                    fps_elapsed = 0.0
                    fps_frames = 0

                elapsed = time.perf_counter() - frame_start
                remaining = frame_duration - elapsed

                if remaining > 0:
                    time.sleep(remaining)

        finally:
            self.teardown()
