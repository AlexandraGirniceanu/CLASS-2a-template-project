
from audio.pipeline import AudioPipeline

pipeline = AudioPipeline(
    "audio/grand_project-wonders-of-the-earth-550792.mp3"
)
pipeline.load()


number_of_windows = len(pipeline.signal) // pipeline.window_size

smallest_amplitude = float("inf")
largest_amplitude = float("-inf")
smallest_call = 0
largest_call = 0

for call in range(1, number_of_windows + 1):
    amplitude = pipeline.get_features()["amplitude"]

    if amplitude < smallest_amplitude:
        smallest_amplitude = amplitude
        smallest_call = call

    if amplitude > largest_amplitude:
        largest_amplitude = amplitude
        largest_call = call

print("Complete windows:", number_of_windows)
print(f"Smallest amplitude: {smallest_amplitude:.8f}, call {smallest_call}")
print(f"Largest amplitude:  {largest_amplitude:.8f}, call {largest_call}")
