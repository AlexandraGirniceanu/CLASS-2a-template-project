
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

    audio_source = pyglet.media.load(
        str(AUDIO_FILE),
        streaming=True,
    )

    player = pyglet.media.Player()
    player.queue(audio_source)
    player.volume = 0.7

    engine = ModernGLEngine(
        width=WINDOW_WIDTH,
        height=WINDOW_HEIGHT,
        title=WINDOW_TITLE,
        target_fps=TARGET_FPS,
    )

    duration = float(audio_source.duration or pipeline.duration)
    restart_threshold = max(0.0, duration - 0.15)

    print(f"Audio loaded: {AUDIO_FILE.name}")
    print(f"Sample rate: {pipeline.sample_rate} Hz")
    print(f"Duration: {duration:.2f} seconds")
    print("Press SPACE to change colour palette.")
    print("Press ESC to exit.")
    print("Starting synchronised audio playback...")

    def get_audio_features() -> dict:
        playback_time = max(0.0, player.time)

        if playback_time >= restart_threshold:
            player.pause()
            player.seek(0.0)
            player.play()

            pipeline.previous_position = None
            pipeline.previous_magnitudes = None
            pipeline.smoothed_bands.fill(0.0)

            playback_time = 0.0
            print("Audio restarted.")

        return pipeline.get_features(
            playback_time=playback_time
        )

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
