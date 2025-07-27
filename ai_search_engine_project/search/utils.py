import numpy as np
import librosa


def chunk_audio(file_path:str, chunk_duration_s:int, target_sr:int) -> tuple[np.ndarray, int, int] :
    """
        Loads an audio file and yields it in chunks of a specified duration.

        Args:
            file_path (str): The path to the audio file.
            chunk_duration_s (int): The desired duration of each chunk in seconds.
            target_sr (int): The target sample rate to load the audio with.

        Yields:
            A tuple containing (audio_chunk, start_time_s, end_time_s)
    """

    waveform, sr = librosa.load(file_path, sr=target_sr, mono=True)

    chunk_length_samples = chunk_duration_s * sr

    start_sample = 0
    while start_sample < len(waveform):
        end_sample = start_sample + chunk_length_samples

        chunk = waveform[start_sample:end_sample]

        if len(chunk) < chunk_length_samples:
            padding = np.zeros(chunk_length_samples - len(chunk))

            chunk = np.concatenate([chunk, padding])

        start_time_s = start_sample // sr
        end_time_s = start_time_s + chunk_duration_s

        yield chunk, start_time_s, end_time_s

        start_sample = end_sample


