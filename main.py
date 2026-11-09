
from pathlib import Path

import pyglet

from audio.pipeline import AudioPipeline
from engines.modern_gl_engine import ModernGLEngine


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

    print(f"Audio loaded: {AUDIO_FILE.name}")
    print(f"Sample rate: {pipeline.sample_rate} Hz")
    print(f"Duration: {pipeline.duration:.2f} seconds")

    audio_source = pyglet.media.load(
        str(AUDIO_FILE),
        streaming=True,
    )

    player = pyglet.media.Player()
    player.loop = True
    player.queue(audio_source)
    player.volume = 0.7

    engine = ModernGLEngine(
        width=WINDOW_WIDTH,
        height=WINDOW_HEIGHT,
        title=WINDOW_TITLE,
        target_fps=TARGET_FPS,
    )

    def get_audio_features() -> dict:
        playback_time = player.time

        if pipeline.duration > 0:
            playback_time %= pipeline.duration

        return pipeline.get_features(
            playback_time=playback_time
        )

    print("Press SPACE to change colour palette.")
    print("Press ESC to exit.")
    print("Starting synchronised audio playback...")

    try:
        player.play()

        engine.run(
            feature_source=get_audio_features
        )

    finally:
        player.pause()
        player.delete()


if __name__ == "__main__":
    run_full_app()
