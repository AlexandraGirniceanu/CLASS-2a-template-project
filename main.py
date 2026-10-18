"""Wires the audio pipeline and rendering engine together and runs the app.

ENGINE and AUDIO_FILE are fixed, version-controlled choices, not runtime
options — committing to one engine tier and one audio file is a project
decision, not something to toggle per run.
"""

from collections.abc import Callable

from audio.pipeline import AudioPipeline
from engines.base_engine import BaseEngine
from engines.modern_gl_engine import ModernGLEngine
from engines.pygame_engine import PygameEngine
from engines.pyglet_engine import PygletEngine

# Set this to whichever engine tier your project uses. See the README for
# how/why this is a one-time project decision, not a runtime toggle.
ENGINE: type[BaseEngine] = ModernGLEngine

AUDIO_FILE = "audio/grand_project-wonders-of-the-earth-550792.mp3"
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
WINDOW_TITLE = "CLASS-2a Project"
TARGET_FPS = 60


def run_engine_only() -> None:
    """Runs the chosen engine with no audio pipeline involved.

    Optional helper for checking your engine in isolation — run_full_app()
    already falls back to a blank feature source until AudioPipeline is
    implemented, so you don't need this to get started.
    """
    engine = ENGINE(width=WINDOW_WIDTH, height=WINDOW_HEIGHT, title=WINDOW_TITLE, target_fps=TARGET_FPS)
    engine.run(feature_source=lambda: {})


NOT_IMPLEMENTED_MESSAGE = "AudioPipeline not implemented yet -- running with a blank feature source."


def tolerate_not_implemented(get_features: Callable[[], dict]) -> Callable[[], dict]:
    """Wraps get_features so a NotImplementedError yields {} for that frame.

    The message is printed once, not every frame. Only NotImplementedError is
    caught — any other exception (e.g. running off the end of the audio)
    still propagates so you can see it.
    """
    warned = False

    def feature_source() -> dict:
        nonlocal warned
        try:
            return get_features()
        except NotImplementedError:
            if not warned:
                print(NOT_IMPLEMENTED_MESSAGE)
                warned = True
            return {}

    return feature_source


def run_full_app():
    engine = ENGINE(width=WINDOW_WIDTH, height=WINDOW_HEIGHT, title=WINDOW_TITLE, target_fps=TARGET_FPS)

    pipeline = AudioPipeline(file_path=AUDIO_FILE)
    try:
        pipeline.load()
        feature_source = tolerate_not_implemented(pipeline.get_features)
    except NotImplementedError:
        print(NOT_IMPLEMENTED_MESSAGE)
        feature_source = lambda: {}

    engine.run(feature_source=feature_source)


if __name__ == "__main__":
    run_full_app()
