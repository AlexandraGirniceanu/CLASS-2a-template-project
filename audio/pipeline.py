
import librosa
import numpy as np


class AudioPipeline:
    def __init__(
        self,
        file_path: str,
        sample_rate: int = 22050,
    ) -> None:
        self.file_path = file_path
        self.sample_rate = sample_rate

        self.signal = None
        self.position = 0
        self.window_size = 1024

        self.bass_edge_hz = 250
        self.treble_edge_hz = 2000
        self.band_count = 160

        self.previous_magnitudes = None
        self.previous_position = None

        self.hann_window = np.hanning(
            self.window_size
        )

        self.frequencies = np.fft.rfftfreq(
            self.window_size,
            d=1 / self.sample_rate,
        )

        self.band_edges = np.geomspace(
            30,
            self.sample_rate / 2,
            self.band_count + 1,
        )

        self.smoothed_bands = np.zeros(
            self.band_count,
            dtype=np.float64,
        )

    def load(self) -> None:
        self.signal, self.sample_rate = librosa.load(
            self.file_path,
            sr=self.sample_rate,
            mono=True,
        )

        self.position = 0
        self.previous_position = None
        self.previous_magnitudes = None

        self.hann_window = np.hanning(
            self.window_size
        )

        self.frequencies = np.fft.rfftfreq(
            self.window_size,
            d=1 / self.sample_rate,
        )

        self.band_edges = np.geomspace(
            30,
            self.sample_rate / 2,
            self.band_count + 1,
        )

        self.smoothed_bands = np.zeros(
            self.band_count,
            dtype=np.float64,
        )

    @property
    def duration(self) -> float:
        if self.signal is None:
            return 0.0

        return len(self.signal) / self.sample_rate

    def get_features(
        self,
        playback_time: float | None = None,
    ) -> dict:
        if self.signal is None or len(self.signal) == 0:
            return self._empty_features()

        signal_length = len(self.signal)

        if playback_time is not None:
            playback_time = max(
                0.0,
                float(playback_time),
            )

            self.position = int(
                playback_time * self.sample_rate
            ) % signal_length

        start = self.position

        if (
            self.previous_position is not None
            and start < self.previous_position
        ):
            self.previous_magnitudes = None
            self.smoothed_bands.fill(0.0)

        indices = (
            start + np.arange(self.window_size)
        ) % signal_length

        window = self.signal[indices]

        rms = float(
            np.sqrt(
                np.mean(
                    window.astype(np.float64) ** 2
                )
            )
        )

        peak = float(
            np.max(np.abs(window))
        )

        windowed_audio = (
            window * self.hann_window
        )

        magnitudes = np.abs(
            np.fft.rfft(windowed_audio)
        )

        power = magnitudes ** 2

        bass_mask = (
            self.frequencies < self.bass_edge_hz
        )

        mid_mask = (
            (self.frequencies >= self.bass_edge_hz)
            & (self.frequencies < self.treble_edge_hz)
        )

        treble_mask = (
            self.frequencies >= self.treble_edge_hz
        )

        bass = float(
            np.sum(power[bass_mask])
        )

        mid = float(
            np.sum(power[mid_mask])
        )

        treble = float(
            np.sum(power[treble_mask])
        )

        magnitude_sum = float(
            np.sum(magnitudes)
        )

        if magnitude_sum > 0.0:
            centroid = float(
                np.sum(
                    self.frequencies * magnitudes
                ) / magnitude_sum
            )
        else:
            centroid = 0.0

        if self.previous_magnitudes is None:
            onset = 0.0
        else:
            positive_difference = np.maximum(
                magnitudes - self.previous_magnitudes,
                0.0,
            )

            onset = float(
                np.sum(positive_difference)
            )

        self.previous_magnitudes = (
            magnitudes.copy()
        )

        bin_indices = np.searchsorted(
            self.frequencies,
            self.band_edges,
        )

        frequency_bands = np.zeros(
            self.band_count,
            dtype=np.float64,
        )

        for index in range(self.band_count):
            first = min(
                int(bin_indices[index]),
                len(magnitudes) - 1,
            )

            last = min(
                int(bin_indices[index + 1]),
                len(magnitudes),
            )

            if last <= first:
                band_energy = magnitudes[first]
            else:
                band_energy = np.mean(
                    magnitudes[first:last]
                )

            frequency_bands[index] = (
                band_energy
            )

        frequency_bands = np.log1p(
            frequency_bands
        )

        maximum = float(
            np.max(frequency_bands)
        )

        if maximum > 0.0:
            frequency_bands /= maximum

        self.smoothed_bands += (
            frequency_bands
            - self.smoothed_bands
        ) * 0.35

        self.previous_position = start

        if playback_time is None:
            self.position = (
                start + self.window_size
            ) % signal_length

        return {
            "amplitude": rms,
            "peak": peak,
            "bass": bass,
            "mid": mid,
            "treble": treble,
            "centroid": centroid,
            "onset": onset,
            "frequency_bands": (
                self.smoothed_bands.tolist()
            ),
        }

    def _empty_features(self) -> dict:
        return {
            "amplitude": 0.0,
            "peak": 0.0,
            "bass": 0.0,
            "mid": 0.0,
            "treble": 0.0,
            "centroid": 0.0,
            "onset": 0.0,
            "frequency_bands": (
                [0.0] * self.band_count
            ),
        }
