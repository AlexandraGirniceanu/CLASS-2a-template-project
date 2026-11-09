
from pathlib import Path

from audio.pipeline import AudioPipeline
from engines.pyglet_engine import PygletEngine


AUDIO_FILE = (
    Path(__file__).resolve().parent
    / "audio"
    / "grand_project-wonders-of-the-earth-550792.mp3"
)

WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
WINDOW_TITLE = "Audio Spectrum | Interactive Installation"
TARGET_FPS = 60


def run_full_app() -> None:
    if not AUDIO_FILE.is_file():
        raise FileNotFoundError(
            f"Audio file not found: {AUDIO_FILE}"
        )

    pipeline = AudioPipeline(
        file_path=str(AUDIO_FILE),
        sample_rate=22050,
    )

    pipeline.load()

    duration = len(pipeline.signal) / pipeline.sample_rate

    print(f"Audio loaded: {AUDIO_FILE.name}")
    print(f"Sample rate: {pipeline.sample_rate} Hz")
    print(f"Duration: {duration:.2f} seconds")
    print("Press SPACE to change colour palette.")
    print("Press ESC to exit.")

    engine = PygletEngine(
        width=WINDOW_WIDTH,
        height=WINDOW_HEIGHT,
        title=WINDOW_TITLE,
        target_fps=TARGET_FPS,
    )

    engine.run(
        feature_source=pipeline.get_features
    )


if __name__ == "__main__":
    run_full_app()
