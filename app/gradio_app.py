import os
import tempfile
from pathlib import Path
from typing import Optional, List
import time
import shutil

import gradio as gr
import torch

from app.core.api import Separator, list_models
from app.core.validation import validate_audio_bytes, ALLOWED_EXTENSIONS

# ---------- constants ----------
TEMP_PREFIX = "basselix_"
JANITOR_MAX_AGE_S = 3600

STEM_NAMES: tuple[str, ...] = ("vocals", "drums", "bass", "other", "instrumental")
STEM_LABELS: tuple[str, ...] = tuple(n.capitalize() for n in STEM_NAMES)

USE_GPU = torch.cuda.is_available()
DEVICE = "cuda" if USE_GPU else "cpu"
PRECISION = "fp16" if USE_GPU else "fp32"
DEVICE_LABEL = "CUDA (GPU)" if USE_GPU else "CPU"
MODEL_ID = list_models()[0]


def validate_and_reset(audio_path: Optional[str]):
    """
    Validate an uploaded audio file.

    Args:
        audio_path: The path to the uploaded file.

    Returns:
        A tuple of (audio_path, *([None] * len(STEM_NAMES))) if the file is valid, otherwise raises a gr.Error.

    Raises:
        gr.Error: If the file is invalid.
    """
    # Runs on upload/change of the gr.Audio input
    if not audio_path:
        return None, *([None] * len(STEM_NAMES))
    ext = Path(audio_path).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        # Clear the input and previews and show an error
        raise gr.Error("Unsupported file type. Allowed: WAV, MP3, FLAC.")

    # Keep the file in the input and clear old previews
    return audio_path, *([None] * len(STEM_NAMES))


def cleanup_old_workdirs(
    prefix: str = TEMP_PREFIX, max_age_s: int = JANITOR_MAX_AGE_S
) -> None:
    """Delete leftover temporary directories older than a cutoff.

    Args:
        prefix (str): Directory name prefix to match in the system temp directory.
        max_age_s (int): Age threshold in seconds; directories with a modification time older than this value are removed.

    Returns:
        None
    """
    tmp = Path(tempfile.gettempdir())
    now = time.time()
    for p in tmp.glob(f"{prefix}*"):
        try:
            if p.is_dir() and (now - p.stat().st_mtime) > max_age_s:
                shutil.rmtree(p, ignore_errors=True)
        except Exception:
            pass


def good_file(path: Optional[str]) -> Optional[str]:
    """Check whether a path points to a non-empty file.

    Args:
        path (Optional[str]): File path to validate.

    Returns:
        Optional[str]: `path` if it exists and has size > 0; otherwise `None`.
    """
    return path if path and os.path.exists(path) and os.path.getsize(path) > 0 else None


def separate(
    audio_path: Optional[str],
    overlap: float,
    progress=gr.Progress(track_tqdm=False),
):
    """Run stem separation and return output stem file paths for the UI.

    Returns five paths (or `None` placeholders) in this order: `[vocals, drums, bass, other, instrumental]`.

    Args:
        audio_path (Optional[str]): Path to the uploaded audio file on disk.
        overlap (float): Overlap fraction used by the backend during separation.
            Higher values may reduce artifacts but increase runtime.
        progress: Gradio progress object used to report UI progress updates.

    Returns:
        List[Optional[str]]: List of file paths (or `None`) for each stem in `STEM_NAMES`,
            suitable for feeding to Gradio `Audio` components.
    """
    cleanup_old_workdirs()

    if not audio_path:
        return [None] * len(STEM_NAMES)

    base_name = os.path.basename(audio_path)
    with open(audio_path, "rb") as f:
        raw = f.read()

    ok, reason = validate_audio_bytes(base_name, raw)
    if not ok:
        raise gr.Error(f"Invalid file: {reason}")

    # Fresh temp workspace per request (no caching)
    workdir = Path(tempfile.mkdtemp(prefix=TEMP_PREFIX))
    inpath = workdir / base_name
    inpath.write_bytes(raw)

    progress(0.05, desc="Loading model…")
    separator = Separator(
        model=MODEL_ID,
        device=DEVICE,
        stems=None,  # all stems
        precision=PRECISION,
        overlap=overlap,
    )

    progress(0.35, desc="Separating…")
    result_paths = separator.separate_to_dir(str(inpath), out_dir=str(workdir))
    progress(0.95, desc="Preparing output…")

    outputs: List[Optional[str]] = [good_file(result_paths.get(n)) for n in STEM_NAMES]

    progress(1.0)
    return outputs


with gr.Blocks(
    title="Basselix",
    theme=gr.themes.Default(
        primary_hue=gr.themes.colors.red,
        secondary_hue=gr.themes.colors.pink,
        text_size=gr.themes.sizes.text_lg,
    ),
) as demo:
    gr.Markdown("# 🎛️ Basselix – Stem Separator")
    gr.Markdown(
        "Upload an audio file, choose overlap (higher = fewer artifacts, slower), then click **Separate**."
    )

    with gr.Row():
        audio_in = gr.Audio(
            sources=["upload"],
            type="filepath",
            label="Upload audio (wav/mp3/flac)",
        )
        with gr.Column():
            overlap = gr.Slider(
                0.0,
                0.9,
                value=0.25,
                step=0.05,
                label="Overlap (chunking)",
                info="Overlap lets audio chunks share data to reduce artifacts. "
                "Use low values for faster speed, higher values if you hear glitches.",
            )
            gr.Markdown(f"**Using device:** `{DEVICE_LABEL}`")

    run = gr.Button("Separate", variant="primary")

    gr.Markdown("### Preview")
    with gr.Row():
        row1 = [
            gr.Audio(label=label, interactive=False, loop=True)
            for label in STEM_LABELS[:3]
        ]
    with gr.Row():
        row2 = [
            gr.Audio(label=label, interactive=False, loop=True)
            for label in STEM_LABELS[3:]
        ]
    out_players = row1 + row2

    audio_in.change(
        fn=validate_and_reset,
        inputs=[audio_in],
        outputs=[audio_in, *out_players],
    )

    run.click(
        fn=separate,
        inputs=[audio_in, overlap],
        outputs=out_players,
        api_name="separate",
    )

demo.launch(server_port=7860)
