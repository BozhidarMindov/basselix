import os
from typing import Dict, List, Optional

import librosa
import numpy as np

from app.backends.demucs_backend import DemucsBackend
from app.core.audio import load_audio, save_audio
from app.core.registry import REGISTRY

_BACKEND_CLS = {
    "demucs": DemucsBackend,
}


def list_models() -> List[str]:
    """List all available model names.

    Returns:
        List[str]: A list of model names registered in the global `REGISTRY`.
    """
    return list(REGISTRY.keys())


class Separator:
    """High-level interface for music source separation.

    Wraps backend models (like Demucs) to separate audio into stems and optionally save results to disk.
    """

    def __init__(
        self,
        model: str = "demucs:htdemucs",
        device: str = "cuda",
        stems: Optional[List[str]] = None,
        precision: str = "fp16",
        overlap: float = 0.25,
        target_sr: Optional[int] = None,
    ):
        """Initialize the separator.

        Args:
            model (str): Identifier of the separation model to use. Must exist in the global `REGISTRY`.
                Defaults to `demucs:htdemucs`.
            device (str): Target device for inference, e.g., `cuda` or `cpu`.
                Defaults to `cuda`.
            stems (Optional[List[str]]): List of stems to separate. If `None`, all supported stems are returned.
                Defaults to `None`.
            precision (str): Precision setting, either `fp16` or `fp32`.
                Defaults to `fp16`.
            overlap (float): Overlap fraction used by the backend during separation.
                Defaults to `0.25`.
            target_sr (Optional[int]): If set, resample output stems to this sample rate.
                Defaults to `None`.

        Raises:
            ValueError: If the specified model is not found in the registry.
        """
        if model not in REGISTRY:
            raise ValueError(f"Unknown model: {model}. Known: {list_models()}")
        spec = REGISTRY[model]
        backend_name = spec["backend"]
        params = dict(spec.get("params", {}))
        cls = _BACKEND_CLS[backend_name]
        self.backend = cls(
            device=device, precision=precision, overlap=overlap, **params
        )
        self.requested_stems = stems
        self.target_sr = target_sr

    def separate(self, path: str) -> Dict[str, np.ndarray]:
        """Separate an audio file into stems.

        Args:
            path (str): Path to the input audio file.

        Returns:
            Dict[str, np.ndarray]: Mapping of stem name to waveform arrays, each shaped
                `(channels, samples)`, `dtype` `float32`.

        Notes:
            If `target_sr` was specified, outputs are resampled accordingly.
        """
        wav, sr = load_audio(path)
        out = self.backend.separate(wav, sr, stems=self.requested_stems)
        if self.target_sr and self.target_sr != sr:
            for i, j in out.items():
                out[i] = librosa.resample(
                    j, orig_sr=sr, target_sr=self.target_sr, axis=1
                )
            sr = self.target_sr
        self._last_sr = sr
        return out

    def separate_to_dir(self, path: str, out_dir: str) -> Dict[str, str]:
        """Separate an audio file and save its stems to a directory.

        Args:
            path (str): Path to the input audio file.
            out_dir (str): Directory where the output stem files will be saved.

        Returns:
            Dict[str, str]: Mapping of stem name to the saved file path.

        Notes:
            - Each stem is saved as a `.wav` file named after the stem.
            - If vocals are present, an `instrumental.wav` file is also written, containing
              the mix of all non-vocal stems.
        """
        os.makedirs(out_dir, exist_ok=True)
        outs = self.separate(path)
        sr = getattr(self, "_last_sr", None) or librosa.get_samplerate(path)
        paths = {}
        for stem, y in outs.items():
            output = os.path.join(out_dir, f"{stem}.wav")
            save_audio(output, y, sr)
            paths[stem] = output

        # Write instrumental -> mix of non-vocal stems
        if "vocals" in outs and len(outs) >= 2:
            instrumental = sum([v for k, v in outs.items() if k != "vocals"])
            output = os.path.join(out_dir, "instrumental.wav")
            save_audio(output, instrumental, sr)
            paths["instrumental"] = output
        return paths
