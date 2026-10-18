
import librosa
import numpy as np


class AudioPipeline:
    """Loads an audio file and extracts features from it, frame by frame."""

    def __init__(self, file_path: str, sample_rate: int = 22050) -> None:
        self.file_path = file_path
        self.sample_rate = sample_rate
        self.signal = None
        self.position = 0
        self.window_size = 1024

    def load(self) -> None:
        """Load the audio file and store the signal and sample rate."""
        self.signal, self.sample_rate = librosa.load(
            self.file_path,
            sr=self.sample_rate,
            mono=True,
        )
        self.position = 0

    

    def get_features(self) -> dict[str, float]:
        """Return the RMS amplitude of the current audio window."""
        if self.position + self.window_size > len(self.signal):
            self.position = 0

        start = self.position
        end = start + self.window_size
        window = self.signal[start:end]

        rms = float(np.sqrt(np.mean(window ** 2)))

        self.position += self.window_size

        return {"amplitude": rms}

