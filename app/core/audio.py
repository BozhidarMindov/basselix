import librosa
import numpy as np
import soundfile as sf


def load_audio(path: str) -> tuple[np.ndarray, int]:
    """Load audio from a file, preserving stereo when possible.

    Args:
        path (str): File path to the audio file.

    Returns:
        tuple[np.ndarray, int]:
            - wav (np.ndarray): Audio waveform array with shape `(2, T)` for stereo or `(1, T)` if mono (after stacking),
              dtype `float32`, values roughly in `[-1, 1]`.
            - sr (int): Sample rate in Hz of the loaded audio.

    Notes:
        * Uses `librosa.load` with `sr=None` to preserve the native sample rate.
        * Uses `mono=False` to preserve multichannel audio. If input is mono or single-channel,
          it returns a 1D waveform, which is then stacked to become 2 channels for consistency.
    """
    wav, sr = librosa.load(path, sr=None, mono=False)
    if wav.ndim == 1:
        wav = np.stack([wav, wav], axis=0)  # Convert mono to stereo format (2, T)
    return wav.astype(np.float32), sr


def save_audio(path: str, wav: np.ndarray, sr: int) -> None:
    """Save a waveform array as an audio file.

    Args:
        path (str): File path where to write the audio file.
        wav (np.ndarray): Audio waveform array with shape `(2, T)` or `(1, T)` or `(T,)`.
            Uses channels-first format.
        sr (int): Sample rate in Hz to use when writing the file.

    Returns:
        None

    Notes:
        * Uses the `soundfile` library to write WAV (or other supported) files.
        * `soundfile` expects shape `(T, channels)`, so this function transposes `wav` if necessary.
        * If a mono or single-channel `wav` is passed as 1D, it will be stacked to two channels
          for uniformity before transposing.
    """
    if wav.ndim == 1:
        wav = np.stack([wav, wav], axis=0)
    sf.write(path, wav.T, sr)
