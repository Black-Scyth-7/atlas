"""Encrypted credential vault for website logins.

Stores per-site {username, password} so Atlas can log the user in on request,
WITHOUT ever keeping passwords in plaintext. Two composable layers of at-rest
protection:

  * Windows DPAPI (default): each secret is sealed with CryptProtectData, tying
    it to the current Windows account. No prompt — Atlas unlocks hands-free —
    and the file is useless if copied to another machine or user account.
  * Master password (optional): if enabled, secrets are ALSO AES-sealed (Fernet)
    under a key derived (PBKDF2-HMAC-SHA256) from a master password you type once
    per session. The vault then stays encrypted even against other programs
    running as you, until you unlock it.

Never logs or prints secrets. On a platform without DPAPI a master password is
required — there's no OS keystore to fall back to.

Standalone self-test (round-trips a fake credential, no real secrets):
    python vault.py
"""

from __future__ import annotations

import base64
import getpass
import json
import os
import re
import secrets
from typing import Callable, Optional


# ---- Windows DPAPI (ctypes; no dependency) ---------------------------------
def _dpapi_available() -> bool:
    return os.name == "nt"


def _dpapi(data: bytes, unprotect: bool) -> bytes:
    import ctypes
    from ctypes import wintypes

    class _BLOB(ctypes.Structure):
        _fields_ = [("cbData", wintypes.DWORD),
                    ("pbData", ctypes.POINTER(ctypes.c_char))]

    crypt32, kernel32 = ctypes.windll.crypt32, ctypes.windll.kernel32
    buf = ctypes.create_string_buffer(data, len(data))
    blob_in = _BLOB(len(data), ctypes.cast(buf, ctypes.POINTER(ctypes.c_char)))
    blob_out = _BLOB()
    fn = crypt32.CryptUnprotectData if unprotect else crypt32.CryptProtectData
    # (data, desc, entropy, reserved, prompt, flags, out)
    ok = fn(ctypes.byref(blob_in), None, None, None, None, 0,
            ctypes.byref(blob_out))
    if not ok:
        raise OSError("DPAPI operation failed (wrong Windows account or "
                      "corrupt vault?)")
    out = ctypes.string_at(blob_out.pbData, blob_out.cbData)
    kernel32.LocalFree(blob_out.pbData)
    return out


def _site_key(raw: str) -> str:
    """Canonicalize a site label so 'Facebook', 'facebook.com', and
    'https://www.facebook.com/login' all map to the same key ('facebook')."""
    s = (raw or "").strip().lower()
    s = re.sub(r"^[a-z]+://", "", s)          # drop scheme
    s = re.sub(r"^www\.", "", s)              # drop www.
    s = s.split("/")[0]                        # drop path
    if "." in s:                               # domain -> registrable label
        parts = [p for p in s.split(".") if p]
        # take the second-to-last label (facebook.com -> facebook,
        # mail.google.com -> google), best-effort.
        s = parts[-2] if len(parts) >= 2 else parts[0]
    # Collapse aliases so 'gmail'/'googlemail' share one Google record.
    return {"gmail": "google", "googlemail": "google"}.get(s, s)


class Vault:
    """Encrypted per-site credential store. See module docstring."""

    _ITER = 200_000
    _CHECK = b"atlas-vault-ok"

    def __init__(self, path: str = "vault.dat", use_dpapi: bool = True,
                 master_password_required: bool = False):
        self.path = path
        self.use_dpapi = use_dpapi and _dpapi_available()
        self.master_required = bool(master_password_required)
        self._fernet = None                    # set after a successful unlock
        self._data = self._load()
        if not self.use_dpapi and not self.master_required:
            # No OS keystore and no master password: refuse to store plaintext.
            raise RuntimeError(
                "Vault has no encryption available: DPAPI is Windows-only and "
                "no master password is enabled. Enable VaultConfig."
                "master_password to use the vault on this platform.")

    # ---- file --------------------------------------------------------------
    def _load(self) -> dict:
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"salt": None, "verifier": None, "sites": {}}

    def _save(self) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2)

    # ---- master-password unlock -------------------------------------------
    def _derive(self, password: str, salt: bytes):
        from cryptography.fernet import Fernet
        import hashlib

        key = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt,
                                  self._ITER, dklen=32)
        return Fernet(base64.urlsafe_b64encode(key))

    def unlock(self, prompt: Callable[[str], str] = getpass.getpass) -> bool:
        """Ensure the master-password layer is unlocked (no-op if disabled).

        First run sets a new master password; later runs verify it against the
        stored check-token. Returns False if the user can't authenticate.
        """
        if not self.master_required or self._fernet is not None:
            return True
        if not self._data.get("salt"):        # first-time setup
            print("Set a master password for the credential vault "
                  "(typed here, hidden). You'll enter it once per session.")
            pw = prompt("  New master password: ")
            if not pw:
                print("  Empty password — vault not enabled.")
                return False
            if pw != prompt("  Confirm: "):
                print("  Passwords didn't match.")
                return False
            salt = secrets.token_bytes(16)
            f = self._derive(pw, salt)
            self._data["salt"] = salt.hex()
            self._data["verifier"] = f.encrypt(self._CHECK).decode("ascii")
            self._fernet = f
            self._save()
            return True
        salt = bytes.fromhex(self._data["salt"])
        for _ in range(3):
            f = self._derive(prompt("Vault master password: "), salt)
            try:
                if f.decrypt(self._data["verifier"].encode("ascii")) == self._CHECK:
                    self._fernet = f
                    return True
            except Exception:
                pass
            print("  Wrong master password.")
        return False

    # ---- seal / open a single secret --------------------------------------
    def _seal(self, plaintext: str) -> str:
        data = plaintext.encode("utf-8")
        layers = []
        if self.master_required:
            data = self._fernet.encrypt(data)      # type: ignore[union-attr]
            layers.append("fernet")
        if self.use_dpapi:
            data = _dpapi(data, unprotect=False)
            layers.append("dpapi")
        return "+".join(layers) + ":" + base64.b64encode(data).decode("ascii")

    def _open(self, token: str) -> str:
        scheme, b64 = token.split(":", 1)
        data = base64.b64decode(b64)
        for layer in reversed(scheme.split("+")):  # dpapi is the outer layer
            if layer == "dpapi":
                data = _dpapi(data, unprotect=True)
            elif layer == "fernet":
                data = self._fernet.decrypt(data)   # type: ignore[union-attr]
        return data.decode("utf-8")

    # ---- public API --------------------------------------------------------
    def set_credential(self, site: str, username: str, password: str) -> str:
        if not self.unlock():
            raise PermissionError("Vault is locked.")
        key = _site_key(site)
        self._data["sites"][key] = {
            "label": site.strip(),
            "username": self._seal(username),
            "password": self._seal(password),
        }
        self._save()
        return key

    def get_credential(self, site: str) -> Optional[tuple[str, str]]:
        rec = self._data["sites"].get(_site_key(site))
        if not rec:
            return None
        if not self.unlock():
            return None
        return self._open(rec["username"]), self._open(rec["password"])

    def has(self, site: str) -> bool:
        return _site_key(site) in self._data["sites"]

    def list_sites(self) -> list[str]:
        return sorted(v.get("label") or k
                      for k, v in self._data["sites"].items())

    def delete_credential(self, site: str) -> bool:
        key = _site_key(site)
        if key in self._data["sites"]:
            del self._data["sites"][key]
            self._save()
            return True
        return False


def _open_configured_vault() -> "Vault":
    """Open the real vault using the app's VaultConfig (same file/settings)."""
    try:
        from config import VaultConfig
        c = VaultConfig()
        return Vault(c.vault_path, use_dpapi=c.use_dpapi,
                     master_password_required=c.master_password)
    except Exception:
        return Vault()  # fall back to defaults (vault.dat, DPAPI)


def _cli_set(site: str = "", username: str = "") -> int:
    """Interactively add/update a saved login in a real terminal.

    This is the reliable way to store a credential: it runs OUTSIDE the voice
    loop, so getpass/input work normally and the password is never spoken,
    transcribed, or logged. Site/username may be passed inline; the PASSWORD is
    ALWAYS prompted (hidden) — never accept it on the command line, where it
    would land in shell history.

    Usage:  python vault.py --set
            python vault.py --set --site gmail --username you@gmail.com
    """
    v = _open_configured_vault()
    print("Add a saved login (password is hidden; stored encrypted).")
    site = (site or input("  Site (e.g. gmail, facebook): ")).strip()
    if not site:
        print("  No site given — nothing saved.")
        return 1
    username = (username or input("  Username / email: ")).strip()
    pw = getpass.getpass("  Password: ")
    if not username or not pw:
        print("  Need both a username and a password — nothing saved.")
        return 1
    if pw != getpass.getpass("  Confirm password: "):
        print("  Passwords didn't match — nothing saved.")
        return 1
    key = v.set_credential(site, username, pw)
    print(f"  Saved (as '{key}'). Now say: \"log into {site}\".")
    print(f"  Vault file: {v.path}  |  sites: {', '.join(v.list_sites())}")
    return 0


def _cli_selftest() -> int:
    import tempfile

    tmp = os.path.join(tempfile.gettempdir(), "atlas_vault_selftest.dat")
    if os.path.exists(tmp):
        os.remove(tmp)
    v = Vault(tmp, use_dpapi=_dpapi_available(), master_password_required=False)
    if not v.use_dpapi:
        print("DPAPI unavailable and no master password — skipping (expected "
              "off Windows).")
        return 0
    v.set_credential("https://www.facebook.com/login", "me@example.com",
                     "S3cret-Passw0rd!")
    got = v.get_credential("facebook")
    print("round-trip:", got == ("me@example.com", "S3cret-Passw0rd!"))
    raw = open(tmp, "r", encoding="utf-8").read()
    print("no plaintext password in file:", "S3cret-Passw0rd!" not in raw)
    print("sites:", v.list_sites())
    os.remove(tmp)
    return 0


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Atlas credential vault.")
    ap.add_argument("--set", action="store_true",
                    help="add/update a saved login (prompts for password)")
    ap.add_argument("--site", default="",
                    help="site for --set (optional; else prompted)")
    ap.add_argument("--username", default="",
                    help="username/email for --set (optional; else prompted)")
    ap.add_argument("--list", action="store_true", help="list saved sites")
    ap.add_argument("--delete", metavar="SITE", help="delete a saved login")
    ap.add_argument("--selftest", action="store_true",
                    help="round-trip a fake credential (no real secrets)")
    a = ap.parse_args()

    if a.set:
        raise SystemExit(_cli_set(a.site, a.username))
    if a.list:
        v = _open_configured_vault()
        sites = v.list_sites()
        print("Saved logins:", ", ".join(sites) if sites else "(none)")
        raise SystemExit(0)
    if a.delete:
        v = _open_configured_vault()
        print("Deleted." if v.delete_credential(a.delete)
              else f"No saved login for {a.delete}.")
        raise SystemExit(0)
    raise SystemExit(_cli_selftest())
