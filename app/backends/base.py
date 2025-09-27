from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import numpy as np


class BaseBackend(ABC):
    """Abstract interface for all separation backends."""

    SUPPORTED_STEMS: List[str] = []

    @abstractmethod
    def separate(
        self,
        wav: np.ndarray,
        sr: int,
        stems: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        """Separate audio into specified stems.

        Args:
            wav (np.ndarray): Audio waveform array, shape (channels, samples), dtype `float32`, values in [-1, 1].
            sr (int): The sample rate (in Hz) of the audio data.
            stems (Optional[List[str]]): List of names of stems to separate.
                If None, default stems defined by the backend should be used.
            **kwargs: Additional, backend‐specific keyword arguments.

        Returns:
            Dict[str, np.ndarray]: A mapping from stem name to its separated waveform
            (`np.ndarray`) with the same format as the input (channels, samples), typically `float32` in [-1, 1].
        """
        pass
