
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

        # Frequency band edges in Hz
        self.bass_edge_hz = 250
        self.treble_edge_hz = 2000

        # Previous spectrum will be used for onset detection later
        self.previous_magnitudes = None

    def load(self) -> None:
        """Load the audio file and store the signal and sample rate."""
        self.signal, self.sample_rate = librosa.load(
            self.file_path,
            sr=self.sample_rate,
            mono=True,
        )
        self.position = 0
        self.previous_magnitudes = None

    def get_features(self) -> dict[str, float]:
        """Return amplitude and spectral features for the current window."""
        if self.position + self.window_size > len(self.signal):
            self.position = 0

        start = self.position
        end = start + self.window_size
        window = self.signal[start:end]

        # Keep the original RMS amplitude calculation unchanged
        rms = float(np.sqrt(np.mean(window ** 2)))

        # Apply a Hann window before calculating the spectrum
        hann_window = np.hanning(self.window_size)
        windowed_audio = window * hann_window
        magnitudes = np.abs(np.fft.rfft(windowed_audio))

        # Convert the band edges from Hz into FFT bin indices
        frequencies = np.fft.rfftfreq(
            self.window_size,
            d=1 / self.sample_rate,
        )
        bass_edge_bin = round(
            self.bass_edge_hz * self.window_size / self.sample_rate
        )
        treble_edge_bin = round(
            self.treble_edge_hz * self.window_size / self.sample_rate
        )

        # Calculate the energy in each frequency band
        bass = float(np.sum(magnitudes[:bass_edge_bin] ** 2))
        mid = float(
            np.sum(magnitudes[bass_edge_bin:treble_edge_bin] ** 2)
        )
        treble = float(np.sum(magnitudes[treble_edge_bin:] ** 2))

        # Calculate the spectral centroid safely
        magnitude_sum = np.sum(magnitudes)
        if magnitude_sum > 0:
            centroid = float(np.sum(frequencies * magnitudes) / magnitude_sum)
        else:
            centroid = 0.0
        # Calculate spectral flux for onset detection
        if self.previous_magnitudes is None:
            onset = 0.0
        else:
            positive_difference = np.maximum(
                magnitudes - self.previous_magnitudes,
                0,
            )
            onset = float(np.sum(positive_difference))

        # Store the current spectrum for the next call
        self.previous_magnitudes = magnitudes.copy()

        self.position += self.window_size
        return {
            "amplitude": rms,
            "bass": bass,
            "mid": mid,
            "treble": treble,
            "centroid": centroid,
            "onset": onset,
        }
