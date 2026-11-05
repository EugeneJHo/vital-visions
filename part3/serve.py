#!/usr/bin/env python3
"""Serve the transmitter (this Mac) and the receiver (the iPhone).

  .venv/bin/python part3/serve.py

Leave it running. Open the transmitter URL on the Mac. Open the receiver
URL on the iPhone, on the same Wi-Fi, with the flash off.
"""

from __future__ import annotations

import json
import socket
import ssl
import subprocess
import threading
from datetime import datetime
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SESSIONS = ROOT / "sessions"
CERTS = ROOT / "certs"
HTTP_PORT = 8080
HTTPS_PORT = 8443


def lan_ip() -> str:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(2.0)
    try:
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        sock.close()


def ensure_cert(ip: str) -> tuple[Path, Path]:
    CERTS.mkdir(parents=True, exist_ok=True)
    cert, key, stamp = CERTS / "cert.pem", CERTS / "key.pem", CERTS / "ip.txt"
    if cert.exists() and key.exists() and stamp.exists() and stamp.read_text().strip() == ip:
        return cert, key
    subprocess.check_call(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-keyout",
            str(key),
            "-out",
            str(cert),
            "-days",
            "30",
            "-nodes",
            "-subj",
            "/CN=vital-visions",
            "-addext",
            f"subjectAltName=IP:127.0.0.1,IP:{ip},DNS:localhost",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    stamp.write_text(ip)
    return cert, key


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def log_message(self, fmt: str, *args) -> None:
        if self.path.startswith("/api/session/latest"):
            return
        super().log_message(fmt, *args)

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path == "/api/session/latest":
            latest = SESSIONS / "latest.json"
            if not latest.is_file():
                self._send_json({"error": "none"}, 404)
                return
            self._send_json(json.loads(latest.read_text()))
            return
        self.path = path
        super().do_GET()

    def do_POST(self) -> None:
        if self.path.split("?", 1)[0] != "/api/session":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        try:
            session = json.loads(self.rfile.read(length))
            bits = session["bits"]
            session["bit_duration"] = float(session["bit_duration"])
            session["delta_alpha"] = float(session["delta_alpha"])
            session["distance_cm"] = int(session["distance_cm"])
            if not isinstance(bits, str) or any(ch not in "01" for ch in bits):
                raise ValueError("bits must be a string of 0s and 1s")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self._send_json({"error": str(exc)}, 400)
            return
        SESSIONS.mkdir(parents=True, exist_ok=True)
        session.setdefault("id", datetime.now().strftime("%Y%m%d-%H%M%S"))
        target = SESSIONS / f"{session['id']}.json"
        text = json.dumps(session, indent=2)
        target.write_text(text)
        (SESSIONS / "latest.json").write_text(text)
        self._send_json({"ok": True, "id": session["id"]})
        print(
            f"Saved {session['id']}: \"{session.get('message', '')}\"  "
            f"alpha {session['delta_alpha']}  {session['distance_cm']} cm  "
            f"{session['bit_duration']} s/bit",
            flush=True,
        )

    def _send_json(self, obj: dict, code: int = 200) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def serve_https(ip: str) -> None:
    cert, key = ensure_cert(ip)
    server = ThreadingHTTPServer(("0.0.0.0", HTTPS_PORT), Handler)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert, key)
    server.socket = context.wrap_socket(server.socket, server_side=True)
    server.serve_forever()


def main() -> None:
    ip = lan_ip()
    https_thread = threading.Thread(target=serve_https, args=(ip,), daemon=True)
    https_thread.start()
    http = ThreadingHTTPServer(("0.0.0.0", HTTP_PORT), Handler)
    print(flush=True)
    print("Transmitter, on this Mac:", flush=True)
    print(f"  http://127.0.0.1:{HTTP_PORT}/transmitter.html", flush=True)
    print(flush=True)
    print("Receiver, on the iPhone (same Wi-Fi, flash off):", flush=True)
    print(f"  https://{ip}:{HTTPS_PORT}/receiver.html", flush=True)
    print(flush=True)
    print("Safari will warn about the certificate. Tap Advanced, then continue.", flush=True)
    print("Leave this window open. Ctrl-C stops the server.", flush=True)
    print(flush=True)
    try:
        http.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
