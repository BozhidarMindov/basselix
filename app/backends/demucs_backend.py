from typing import Dict, List, Optional

import numpy as np
import torch
from demucs.apply import apply_model
from demucs.pretrained import get_model as demucs_get_model

from app.backends.base import BaseBackend


class DemucsBackend(BaseBackend):
    """Demucs-based audio separation backend."""

    SUPPORTED_STEMS: List[str] = ["drums", "bass", "other", "vocals"]

    def __init__(
        self,
        model_name: str = "htdemucs",
        device: str = "cuda",
        precision: str = "fp16",
        overlap: float = 0.25,
    ) -> None:
        """Initialize the Demucs separation backend.

        Args:
            model_name (str): Name of the pretrained Demucs model to load.
                Defaults to `htdemucs`.
            device (str): Target device for inference, for example, `cuda` or `cpu`.
                Defaults to `cuda`.
            precision (str): Precision setting, either `fp16` or `fp32`.
                Defaults to `fp16`. Note: Only effective when running on CUDA.
            overlap (float): Amount of overlap between input segments during separation.
                Higher values improve quality but slow down inference.
                Defaults to `0.25`.
        """
        dev = (
            "cuda"
            if (device.startswith("cuda") and torch.cuda.is_available())
            else "cpu"
        )
        self.device = torch.device(dev)
        self.use_fp16 = precision == "fp16" and self.device.type == "cuda"
        self.overlap = float(overlap)

        self.model = demucs_get_model(model_name)
        self.model.to(self.device).eval()

    @torch.inference_mode()
    def separate(
        self, wav: np.ndarray, sr: int, stems: Optional[List[str]] = None, **kwargs
    ) -> Dict[str, np.ndarray]:
        """Separate an audio waveform into stems using Demucs.

        Args:
            wav (np.ndarray): Input audio waveform of shape `(channels, samples)`
                or `(samples,)`, dtype `float32`, values in `[-1, 1]`.
            sr (int): Sample rate (Hz) of the input audio.
            stems (Optional[List[str]]): Subset of stems to return. If `None`,
                all supported stems are returned. Defaults to `None`.
            **kwargs: Additional arguments (not used by default).

        Returns:
            Dict[str, np.ndarray]: Mapping from stem name to separated waveform
            of shape `(channels, samples)`, dtype `float32`.

        Raises:
            RuntimeError: If Demucs model inference fails.
        """
        if wav.ndim == 1:
            wav = np.stack([wav, wav], axis=0)
        x = torch.from_numpy(wav).to(self.device).float()

        if x.ndim == 2:
            x = x.unsqueeze(0)

        if self.use_fp16:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                sources = apply_model(
                    self.model, x, shifts=1, split=True, overlap=self.overlap
                )[0]
        else:
            sources = apply_model(
                self.model, x, shifts=1, split=True, overlap=self.overlap
            )[0]

        names = getattr(self.model, "sources", self.SUPPORTED_STEMS)
        out: Dict[str, np.ndarray] = {}
        for i, name in enumerate(names):
            if stems is None or name in stems:
                y = sources[i].detach().float().cpu().numpy()
                out[name] = y
        return out
