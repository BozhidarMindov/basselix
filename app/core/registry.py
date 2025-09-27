from typing import Dict, Any

REGISTRY: Dict[str, Dict[str, Any]] = {
    # Backend: demucs
    "demucs:htdemucs": {
        "backend": "demucs",
        "params": {"model_name": "htdemucs"},
        "stems": ["vocals", "drums", "bass", "other"],
    },
}
