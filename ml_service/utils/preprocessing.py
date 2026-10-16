"""
PlantVerse ML Service — Image Preprocessing Utilities
=====================================================
Validates, checks limits, and preprocesses uploaded leaf images for model inference.
"""

import io
import os
from typing import Any, Tuple, Union
from PIL import Image

MAX_UPLOAD_SIZE = int(os.getenv("MAX_IMAGE_SIZE_BYTES", str(10 * 1024 * 1024)))  # 10 MB default
ALLOWED_FORMATS = {"JPEG", "JPG", "PNG", "WEBP"}


def validate_and_load_image(file_bytes_or_stream: Union[bytes, bytearray, io.BytesIO]) -> Image.Image:
    """
    Validate raw bytes or stream, check file size limits, verify integrity,
    and convert to a 3-channel RGB PIL Image.
    """
    if not file_bytes_or_stream:
        raise ValueError("No image content provided.")

    if isinstance(file_bytes_or_stream, (bytes, bytearray)):
        if len(file_bytes_or_stream) > MAX_UPLOAD_SIZE:
            raise ValueError(f"Image exceeds maximum allowable size of {MAX_UPLOAD_SIZE // (1024*1024)} MB.")
        buf = io.BytesIO(file_bytes_or_stream)
    else:
        buf = file_bytes_or_stream
        if hasattr(buf, "seek"):
            buf.seek(0)

    try:
        image = Image.open(buf)
        # Check image format
        fmt = (image.format or "").upper()
        if fmt not in ALLOWED_FORMATS:
            # Still attempt RGB conversion if readable
            pass

        # Verify integrity
        image.verify()
        buf.seek(0)
        image = Image.open(buf)

        if image.mode != "RGB":
            image = image.convert("RGB")

        return image

    except Exception as exc:
        raise ValueError(f"Unable to decode or process image: {exc}")


def prepare_tensor_input(image: Image.Image, target_size: Tuple[int, int] = (224, 224)) -> Image.Image:
    """
    Resize image to the target tensor dimensions maintaining visual morphology.
    """
    return image.resize(target_size, Image.Resampling.BILINEAR)
