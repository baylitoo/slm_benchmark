"""Reversible encryption for Studio-issued API key secrets.

Separate from ``api_key_store`` so the cipher is testable without a database,
and so the one place that handles plaintext secrets is small enough to read in
full.

Why reversible at all: the product requirement is that a key is shown once in
the UI, masked afterwards, and still recoverable by an operator server-side.
A one-way hash cannot do the third part. The cost is explicit -- a database
dump combined with ``API_KEY_ENCRYPTION_KEY`` reveals live credentials -- so
the encryption key must come from the environment and never live in the
database, the image, or the repository. The authentication path does not use
this module at all; it compares SHA-256 digests (see ``ApiKeyRecord``), so a
request never decrypts anything.
"""

from __future__ import annotations

import base64
import hashlib
import os

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from docie_bench.settings import get_settings

# AES-GCM standard nonce width. 96 bits is the size the mode is specified and
# optimized for; anything else forces an extra GHASH derivation step.
_NONCE_BYTES = 12


class KeyEncryptionUnavailableError(RuntimeError):
    """``API_KEY_ENCRYPTION_KEY`` is not configured.

    Raised instead of falling back to storing the secret in cleartext: a
    silent downgrade would leave the database holding live credentials while
    every surface still reported success.
    """


def _cipher() -> AESGCM:
    """AES-GCM under a 32-byte key derived from the configured secret.

    The env value is passed through SHA-256 rather than required to be exactly
    32 bytes, so an operator can use any passphrase or generated token without
    a length rule. SHA-256 (not a slow KDF) is appropriate here because the
    input is expected to be high-entropy generated material, not a human
    password -- and a KDF would run on every operator reveal.
    """
    configured = get_settings().api_key_encryption_key.get_secret_value().strip()
    if not configured:
        raise KeyEncryptionUnavailableError(
            "API_KEY_ENCRYPTION_KEY is not set; Studio-issued API keys cannot be "
            "created or revealed until it is configured"
        )
    return AESGCM(hashlib.sha256(configured.encode("utf-8")).digest())


def encryption_configured() -> bool:
    """Whether key creation/reveal can work, for a surface that wants to say
    so up front instead of failing at the moment of use."""
    return bool(get_settings().api_key_encryption_key.get_secret_value().strip())


def encrypt_secret(secret: str) -> str:
    """``base64(nonce || ciphertext)`` for storage in ``secret_ciphertext``.

    A fresh random nonce per call: reusing one under the same key would break
    AES-GCM's confidentiality AND its authentication, and keys are written
    rarely enough that there is no reason to be clever here.
    """
    nonce = os.urandom(_NONCE_BYTES)
    sealed = _cipher().encrypt(nonce, secret.encode("utf-8"), None)
    return base64.b64encode(nonce + sealed).decode("ascii")


def decrypt_secret(stored: str) -> str:
    """Inverse of :func:`encrypt_secret`.

    A wrong or rotated ``API_KEY_ENCRYPTION_KEY`` surfaces as
    ``KeyEncryptionUnavailableError`` rather than as a generic crypto error, so
    the operator is told the cause instead of seeing a 500: GCM's
    authentication tag makes "cannot be decrypted" and "was tampered with"
    indistinguishable, and for this surface the actionable reading is always
    the former.
    """
    try:
        raw = base64.b64decode(stored.encode("ascii"), validate=True)
    except (ValueError, TypeError) as exc:
        raise KeyEncryptionUnavailableError("stored key secret is not valid base64") from exc
    if len(raw) <= _NONCE_BYTES:
        raise KeyEncryptionUnavailableError("stored key secret is truncated")
    try:
        opened = _cipher().decrypt(raw[:_NONCE_BYTES], raw[_NONCE_BYTES:], None)
    except InvalidTag as exc:
        raise KeyEncryptionUnavailableError(
            "stored key secret could not be decrypted; API_KEY_ENCRYPTION_KEY "
            "does not match the one it was written with"
        ) from exc
    return opened.decode("utf-8")


__all__ = [
    "KeyEncryptionUnavailableError",
    "decrypt_secret",
    "encrypt_secret",
    "encryption_configured",
]
