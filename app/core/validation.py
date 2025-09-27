from io import BytesIO
from pathlib import Path
from typing import Tuple, Optional, List

from mutagen import File as MutagenFile, MutagenError

ALLOWED_EXTENSIONS: List[str] = [".wav", ".mp3", ".flac"]
MAX_MB: int = 250


def validate_audio_bytes(
    name: str,
    data: bytes,
    allowed_ext: Optional[List[str]] = ALLOWED_EXTENSIONS,
    max_mb: int = MAX_MB,
) -> Tuple[bool, Optional[str]]:
    """Validate audio data by filename and raw bytes.

    Checks if the audio data:

    - Is non-empty
    - Does not exceed the specified maximum size
    - Has an allowed file extension
    - Can be parsed by Mutagen (ensuring it’s a supported format and not corrupt)

    Args:
        name (str): The filename (including extension) of the audio file.
        data (bytes): The raw audio file bytes.
        allowed_ext (Optional[List[str]]): List of allowed file extensions (lowercase,
            including the dot). If `None`, all extensions are accepted.
            Defaults to `ALLOWED_EXTENSIONS`.
        max_mb (int): Maximum allowed file size in megabytes. Defaults to `MAX_MB`.

    Returns:
        Tuple[bool, Optional[str]]: A tuple where:
            - First element (`bool`): `True` if validation passed, `False` otherwise
            - Second element (`Optional[str]`): Error message if validation failed,
              `None` if validation passed

    Notes:
        - Internal exceptions from Mutagen (like `MutagenError`) are caught and
          translated into a failure tuple.
        - The function is case-insensitive for file extensions.
    """
    if not data:
        return False, "Empty file!"
    if len(data) > max_mb * 1024 * 1024:
        return False, f"File too large (> {max_mb} MB)!"

    ext = Path(name).suffix.lower()
    if allowed_ext is not None and ext not in allowed_ext:
        allowed_list = ", ".join(sorted(allowed_ext))
        return False, f"Unsupported extension {ext}. Allowed: {allowed_list}"

    try:
        audio = MutagenFile(BytesIO(data))
        if audio is None:
            return False, "Unsupported or corrupt audio file!"
        return True, None
    except MutagenError:
        return False, "Invalid or unreadable audio file!"
    except Exception:
        return False, "Invalid or unreadable audio file!"
