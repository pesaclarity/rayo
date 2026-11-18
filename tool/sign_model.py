"""Signs a model file for rayo_verify / RayoIntegrity (docs/SECURITY.md), and makes keys.

    python3 tool/sign_model.py keygen KEY.pem          a new Ed25519 key (keep it secret),
                                                       and its public key in hex
    python3 tool/sign_model.py sign KEY.pem MODEL      MODEL.sig: 64 bytes, "pure" Ed25519
                                                       (RFC 8032) over the whole file; and
                                                       the file's SHA-256 and the signature
                                                       in hex
    python3 tool/sign_model.py verify PUBHEX MODEL [SIG]

Needs the `cryptography` package. The same signature as
`openssl pkeyutl -sign -rawin -inkey KEY.pem -in MODEL -out MODEL.sig` (OpenSSL 3).
"""

from __future__ import annotations

import hashlib
import sys

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)


def _public_hex(key: Ed25519PrivateKey) -> str:
    return key.public_key().public_bytes(serialization.Encoding.Raw,
                                         serialization.PublicFormat.Raw).hex()


def main(argv: list[str]) -> int:
    if len(argv) == 3 and argv[1] == "keygen":
        key = Ed25519PrivateKey.generate()
        with open(argv[2], "xb") as f:
            f.write(key.private_bytes(serialization.Encoding.PEM,
                                      serialization.PrivateFormat.PKCS8,
                                      serialization.NoEncryption()))
        print(f"public key: {_public_hex(key)}")
        return 0
    if len(argv) == 4 and argv[1] == "sign":
        with open(argv[2], "rb") as f:
            key = serialization.load_pem_private_key(f.read(), password=None)
        if not isinstance(key, Ed25519PrivateKey):
            raise SystemExit("not an Ed25519 key")
        data = open(argv[3], "rb").read()
        sig = key.sign(data)
        with open(argv[3] + ".sig", "wb") as f:
            f.write(sig)
        print(f"public key: {_public_hex(key)}")
        print(f"sha256:     {hashlib.sha256(data).hexdigest()}")
        print(f"signature:  {sig.hex()}  ({argv[3]}.sig)")
        return 0
    if len(argv) in (4, 5) and argv[1] == "verify":
        data = open(argv[3], "rb").read()
        sig = open(argv[4] if len(argv) == 5 else argv[3] + ".sig", "rb").read()
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(argv[2])).verify(sig, data)
        print("ok")
        return 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
