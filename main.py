import numpy as np
from transformers import pipeline
import click
import pydub
import torch


def pydub_to_np(audio: pydub.AudioSegment) -> np.ndarray:
    """
    Converts pydub audio segment into np.float32 of shape [duration_in_seconds*sample_rate, channels],
    where each value is in range [-1.0, 1.0].
    Returns tuple (audio_np_array, sample_rate).
    """

    # Get the raw audio data as an array of bytes
    samples = audio.get_array_of_samples()

    # Convert to numpy array
    samples_np = np.array(samples).astype(np.int16)

    # Normalize to range [-1, 1]
    samples_np = samples_np / np.iinfo(samples_np.dtype).max

    return samples_np


@click.command()
@click.argument("input_file", type=click.Path(exists=True))
def main(input_file):
    device = "cuda:0"
    pipe = pipeline(
        "automatic-speech-recognition", model="openai/whisper-large-v3", device=device
    )
    sound = pydub.AudioSegment.from_mp3(input_file)
    sound = sound.set_channels(1).set_frame_rate(16_000)
    sound_array = pydub_to_np(sound)
    result = pipe(
        sound_array,
        max_new_tokens=256,
        generate_kwargs={"task": "transcribe"},
        chunk_length_s=30,
        batch_size=8,
    )
    print(result)


if __name__ == "__main__":
    main()
