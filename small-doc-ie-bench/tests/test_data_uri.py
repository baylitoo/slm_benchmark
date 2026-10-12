"""One decoder for the inline documents that reach a serving node."""

import base64

import pytest

from docie_bench.serving.solutions import SolutionError
from docie_bench.serving.solutions import _decode_data_uri as ocr_decode
from docie_bench.transformers_server.server import _decode_data_uri as transformers_decode
from docie_bench.vision import DocumentImage, decode_data_uri

PNG = b"\x89PNG\r\n\x1a\n" + b"payload"


def _uri(data: bytes = PNG, media_type: str = "image/png") -> str:
    return f"data:{media_type};base64,{base64.b64encode(data).decode()}"


def test_a_document_image_round_trips_through_its_own_data_uri() -> None:
    image = DocumentImage(page=1, media_type="image/png", data=PNG)
    assert decode_data_uri(image.data_url()) == (PNG, "image/png")


def test_a_remote_url_is_refused_rather_than_fetched() -> None:
    with pytest.raises(ValueError, match="never fetches a remote URL"):
        decode_data_uri("https://example.invalid/scan.png")


def test_a_data_uri_that_is_not_base64_is_refused() -> None:
    with pytest.raises(ValueError, match="must be base64-encoded"):
        decode_data_uri("data:image/png,%89PNG")


def test_an_empty_payload_is_refused_rather_than_decoded_to_nothing() -> None:
    # The OCR adapter used to accept this, write a zero-byte temp file and run
    # OCR over it, reporting an empty document instead of an error.
    with pytest.raises(ValueError, match="no base64 payload"):
        decode_data_uri("data:image/png;base64,")


def test_invalid_base64_is_refused() -> None:
    with pytest.raises(ValueError, match="not valid base64"):
        decode_data_uri("data:image/png;base64,not!base64!")


def test_a_missing_media_type_falls_back_rather_than_failing() -> None:
    assert decode_data_uri(f"data:;base64,{base64.b64encode(PNG).decode()}")[1] == (
        "application/octet-stream"
    )


# ── the two callers keep their own error surfaces ────────────────────────────


@pytest.mark.parametrize(
    ("media_type", "suffix"),
    [
        ("image/png", ".png"),
        ("image/jpeg", ".jpg"),
        ("application/pdf", ".pdf"),
        ("image/unknown-format", ".png"),
    ],
)
def test_the_ocr_adapter_maps_the_media_type_to_a_file_suffix(media_type: str, suffix: str) -> None:
    assert ocr_decode(_uri(media_type=media_type)) == (PNG, suffix)


def test_the_ocr_adapter_reports_its_own_error_type() -> None:
    with pytest.raises(SolutionError, match="image_url"):
        ocr_decode("data:image/png;base64,")


def test_the_transformers_server_still_returns_bare_bytes() -> None:
    assert transformers_decode(_uri()) == PNG


def test_the_transformers_server_still_raises_value_error() -> None:
    with pytest.raises(ValueError, match="remote URL"):
        transformers_decode("https://example.invalid/scan.png")
