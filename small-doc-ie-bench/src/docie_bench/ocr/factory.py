from __future__ import annotations

from docie_bench.ocr.base import OCRBackend
from docie_bench.ocr.minicpm_backend import MiniCPMOCRBackend
from docie_bench.ocr.paddle_backend import PaddleOCRBackend
from docie_bench.ocr.pdf_text import PdfTextBackend
from docie_bench.ocr.tesseract_backend import TesseractBackend
from docie_bench.ocr.xiaomi_backend import XiaomiOCRBackend

# The backends :func:`get_ocr_backend` accepts, with the import each needs. The
# optional ones raise only when constructed, so availability is reported from
# the module being importable rather than by building one on a GET.
OCR_BACKENDS: tuple[tuple[str, tuple[str, ...], str | None], ...] = (
    ("liteparse", ("pdf_text",), None),
    ("tesseract", (), "pytesseract"),
    ("paddleocr", (), "paddleocr"),
    # No module: this one is a MODEL, not a library. Its availability is
    # whether a deployment resolves, which `available_ocr_backends` reports
    # separately -- an import check would call it available on a fleet that
    # has never deployed it.
    ("xiaomi", (), None),
    ("minicpm", (), None),
)


def available_ocr_backends() -> list[dict[str, object]]:
    """Each backend this deployment accepts, and whether its import is present."""
    import importlib.util

    return [
        {
            "name": name,
            "aliases": list(aliases),
            "available": (
                _served_backend_available(name)
                if name in _SERVED_BACKENDS
                else module is None or importlib.util.find_spec(module) is not None
            ),
        }
        for name, aliases, module in OCR_BACKENDS
    ]


#: Backends that are a served model rather than an importable library.
_SERVED_BACKENDS = frozenset({"xiaomi", "minicpm"})


def _served_backend_available(name: str) -> bool:
    """Whether a served backend has a deployment to talk to.

    Best-effort and never raises: the Studio asks this to decide what to offer
    in a dropdown, and a catalog hiccup should hide the option rather than fail
    the request that was listing them.

    Resolution is per call rather than cached, for the reason the api rebuilds
    its control plane per request: a backend that became deployable a moment
    ago should appear, and one whose node went away should stop appearing.
    """
    backends = {"xiaomi": XiaomiOCRBackend, "minicpm": MiniCPMOCRBackend}
    factory = backends.get(name)
    if factory is None:
        return False
    try:
        factory()._profile()
    except Exception:  # noqa: BLE001 - unavailable is the answer, not an error
        return False
    return True


def get_ocr_backend(name: str, *, language: str | None = None) -> OCRBackend:
    normalized = name.lower().strip()
    # "liteparse" is the canonical name; "pdf_text" is the legacy alias for the
    # same backend (PdfTextBackend IS liteparse — PDFium spatial text + an OCR
    # fallback). Lightweight and the sensible default for PDFs.
    if normalized in ("liteparse", "pdf_text"):
        return PdfTextBackend(language=language)
    if normalized == "tesseract":
        return TesseractBackend()
    if normalized == "paddleocr":
        return PaddleOCRBackend(lang=language or "en")
    if normalized == "xiaomi":
        return XiaomiOCRBackend()
    if normalized == "minicpm":
        return MiniCPMOCRBackend()
    raise ValueError(
        f"Unknown OCR backend {name!r}. "
        "Expected liteparse, tesseract, paddleocr, xiaomi, or minicpm."
    )
