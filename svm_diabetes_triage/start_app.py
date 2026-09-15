"""Start the SVM teaching website with only Python's standard library."""

from __future__ import annotations

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
DIST_DIR = PROJECT_DIR / "dist"


class QuietHandler(SimpleHTTPRequestHandler):
    """Serve files normally but keep the terminal output easy to read."""

    def log_message(self, format: str, *args: object) -> None:
        return


def make_server(first_port: int) -> tuple[ThreadingHTTPServer, int]:
    handler = partial(QuietHandler, directory=str(DIST_DIR))
    for port in range(first_port, first_port + 20):
        try:
            return ThreadingHTTPServer(("127.0.0.1", port), handler), port
        except OSError:
            continue
    raise OSError(f"Could not find a free port from {first_port} to {first_port + 19}.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the SVM diabetes screening demo.")
    parser.add_argument("--port", type=int, default=8001, help="First local port to try.")
    args = parser.parse_args()

    if not (DIST_DIR / "model.json").exists():
        raise SystemExit("Model missing. Run: python train_model.py")

    server, port = make_server(args.port)
    print("SVM Diabetes Triage is ready.")
    print(f"Open this exact address: http://127.0.0.1:{port}")
    print("Keep this window open. Press Ctrl+C here to stop the server.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
