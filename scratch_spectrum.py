
import numpy as np

from audio.pipeline import AudioPipeline

pipeline = AudioPipeline(
    "audio/grand_project-wonders-of-the-earth-550792.mp3"
)
pipeline.load()

window_size = 1024
start = 1837056  
window = pipeline.signal[start:start + window_size]

hann_window = np.hanning(window_size)
windowed_audio = window * hann_window
magnitudes = np.abs(np.fft.rfft(windowed_audio))


print("Window size:", len(window))
print("Number of FFT bins:", len(magnitudes))
print("Bin 1 frequency (Hz):", pipeline.sample_rate / window_size)
print(
    "Last bin frequency (Hz):",
    (len(magnitudes) - 1) * pipeline.sample_rate / window_size
)


strongest_bin = np.argmax(magnitudes[1:]) + 1

strongest_frequency = strongest_bin * pipeline.sample_rate / window_size

print("Strongest bin:", strongest_bin)
print("Strongest frequency (Hz):", strongest_frequency)
print("Strongest magnitude:", magnitudes[strongest_bin])
print("Neighbouring magnitudes (bins 0 to 8):")
print(magnitudes[0:9])



BASS_EDGE_HZ = 250
TREBLE_EDGE_HZ = 2000


BASS_EDGE_BIN = round(BASS_EDGE_HZ * window_size / pipeline.sample_rate)
TREBLE_EDGE_BIN = round(TREBLE_EDGE_HZ * window_size / pipeline.sample_rate)

print("Bass edge (Hz):", BASS_EDGE_HZ)
print("Bass edge (bin):", BASS_EDGE_BIN)
print("Treble edge (Hz):", TREBLE_EDGE_HZ)
print("Treble edge (bin):", TREBLE_EDGE_BIN)



window = pipeline.signal[start:start + window_size]
hann_window = np.hanning(window_size)
windowed_audio = window * hann_window
magnitudes = np.abs(np.fft.rfft(windowed_audio))

bass_magnitudes = magnitudes[:BASS_EDGE_BIN]
mid_magnitudes = magnitudes[BASS_EDGE_BIN:TREBLE_EDGE_BIN]
treble_magnitudes = magnitudes[TREBLE_EDGE_BIN:]

bass_energy = np.sum(bass_magnitudes ** 2)
mid_energy = np.sum(mid_magnitudes ** 2)
treble_energy = np.sum(treble_magnitudes ** 2)

print("Bass energy:", bass_energy)
print("Mid energy:", mid_energy)
print("Treble energy:", treble_energy)


frequencies = np.fft.rfftfreq(window_size, d=1 / pipeline.sample_rate)
loud_centroid = np.sum(frequencies * magnitudes) / np.sum(magnitudes)

quiet_start = 3297280
quiet_window = pipeline.signal[quiet_start:quiet_start + window_size]
quiet_windowed = quiet_window * np.hanning(window_size)
quiet_magnitudes = np.abs(np.fft.rfft(quiet_windowed))

quiet_centroid = np.sum(frequencies * quiet_magnitudes) / np.sum(quiet_magnitudes)

print("Loud window centroid (Hz):", loud_centroid)
print("Quiet window centroid (Hz):", quiet_centroid)

test_pipeline = AudioPipeline(
    "audio/grand_project-wonders-of-the-earth-550792.mp3"
)
test_pipeline.load()

for i in range(10):
    features = test_pipeline.get_features()
    print(f"Frame {i + 1}:", features)


# Check the original amplitude values for the first 10 windows
amplitude_check = AudioPipeline(
    "audio/grand_project-wonders-of-the-earth-550792.mp3"
)
amplitude_check.load()

print("\nAmplitude check:")
for i in range(10):
    features = amplitude_check.get_features()
    print(f"Frame {i + 1}: {features['amplitude']}")


# Calculate feature ranges across the whole track
range_pipeline = AudioPipeline(
    "audio/grand_project-wonders-of-the-earth-550792.mp3"
)
range_pipeline.load()

all_features = {
    "amplitude": [],
    "bass": [],
    "mid": [],
    "treble": [],
    "centroid": [],
    "onset": [],
}

while range_pipeline.position + range_pipeline.window_size <= len(range_pipeline.signal):
    features = range_pipeline.get_features()

    for name in all_features:
        all_features[name].append(features[name])

for name, values in all_features.items():
    print(
        f"{name}: min={min(values):.6f}, "
        f"max={max(values):.6f}"
    )

# Compare our onset feature with librosa onset strength
import librosa

comparison_pipeline = AudioPipeline(
    "audio/grand_project-wonders-of-the-earth-550792.mp3"
)
comparison_pipeline.load()

our_onsets = []

while (
    comparison_pipeline.position + comparison_pipeline.window_size
    <= len(comparison_pipeline.signal)
):
    features = comparison_pipeline.get_features()
    our_onsets.append(features["onset"])

librosa_onsets = librosa.onset.onset_strength(
    y=comparison_pipeline.signal,
    sr=comparison_pipeline.sample_rate,
    hop_length=comparison_pipeline.window_size,
)

print("\nOnset comparison:")
print("Our onset frames:", len(our_onsets))
print("Librosa onset frames:", len(librosa_onsets))
print("Our strongest onset frame:", int(np.argmax(our_onsets)))
print("Librosa strongest onset frame:", int(np.argmax(librosa_onsets)))

our_strongest_time = (
    2455 * comparison_pipeline.window_size
    / comparison_pipeline.sample_rate
)

librosa_strongest_time = (
    1349 * comparison_pipeline.window_size
    / comparison_pipeline.sample_rate
)

print(f"Our strongest onset time: {our_strongest_time:.2f} seconds")
print(f"Librosa strongest onset time: {librosa_strongest_time:.2f} seconds")

# Part 3: Find the five strongest onset hits
onset_pipeline = AudioPipeline(
    "audio/grand_project-wonders-of-the-earth-550792.mp3"
)
onset_pipeline.load()

onset_results = []

while (
    onset_pipeline.position + onset_pipeline.window_size
    <= len(onset_pipeline.signal)
):
    start_position = onset_pipeline.position
    features = onset_pipeline.get_features()

    onset_results.append({
        "position": start_position,
        "time": start_position / onset_pipeline.sample_rate,
        "onset": features["onset"],
    })

# Sort by onset strength, strongest first
onset_results.sort(key=lambda item: item["onset"], reverse=True)

# Keep hits at least 0.5 seconds apart
selected_hits = []

for hit in onset_results:
    if all(
        abs(hit["time"] - selected["time"]) >= 0.5
        for selected in selected_hits
    ):
        selected_hits.append(hit)

    if len(selected_hits) == 5:
        break

print("\nFive strongest onset hits:")
for number, hit in enumerate(selected_hits, start=1):
    print(
        f"{number}. Time: {hit['time']:.2f}s, "
        f"Onset: {hit['onset']:.4f}"
    )

    
# Task 14: Compare custom onset detection with librosa

# Task 14: Compare custom onset detection with librosa

librosa_onset_times = librosa.onset.onset_detect(
    y=comparison_pipeline.signal,
    sr=comparison_pipeline.sample_rate,
    hop_length=comparison_pipeline.window_size,
    units="time",
)

print("\n--- Task 14: Onset Detection Comparison ---")
print("Librosa onset times:")
print([round(float(t), 2) for t in librosa_onset_times])

# Use the five strongest custom onset times from Task 13
custom_onsets = [101.56, 108.44, 111.87, 114.01, 117.45]
tolerance = 0.2
shared = []

for custom_time in custom_onsets:
    matches = [
        float(t) for t in librosa_onset_times
        if abs(float(t) - custom_time) <= tolerance
    ]

    if matches:
        closest = min(
            matches,
            key=lambda t: abs(t - custom_time)
        )
        shared.append((custom_time, closest))

print("\nMatching detections (within 0.2 seconds):")
for custom_time, librosa_time in shared:
    print(
        f"Custom: {custom_time:.2f}s | "
        f"librosa: {librosa_time:.2f}s"
    )

print(
    f"\nShared detections: {len(shared)} "
    f"out of {len(custom_onsets)}"
)
print(
    "Custom detections without a close match:",
    len(custom_onsets) - len(shared)
)

# Task 15: Compare frequency-band energy with sample energy

energy_window = pipeline.signal[start:start + window_size]

band_energy_total = bass_energy + mid_energy + treble_energy
sample_energy_total = np.sum(energy_window ** 2)

print("\nEnergy Comparison:")
print("Total band energy:", band_energy_total)
print("Sum of squared samples:", sample_energy_total)
print("Ratio (band energy / sample energy):",
      band_energy_total / sample_energy_total)
