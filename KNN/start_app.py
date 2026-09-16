"""Start the BMI website on http://127.0.0.1:8000.

Run this file from any folder:
    python C:/path/to/machine_learning/start_app.py

Keep the terminal open while using the website. Press Ctrl+C to stop it.
"""

from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


HOST = "127.0.0.1"
PORT = 8000
WEBSITE_FOLDER = Path(__file__).parent / "dist"


class BmiWebsiteHandler(SimpleHTTPRequestHandler):
    """Serve the dist folder and keep working if a terminal is disconnected."""

    def log_message(self, message_format: str, *values: object) -> None:
        try:
            super().log_message(message_format, *values)
        except OSError:
            # A closed terminal should not stop the browser from receiving files.
            pass


def main() -> None:
    handler = partial(BmiWebsiteHandler, directory=str(WEBSITE_FOLDER))

    try:
        server = ThreadingHTTPServer((HOST, PORT), handler)
    except OSError as error:
        print(f"Could not start the website on port {PORT}.")
        print("Another program may already be using that port.")
        print("Close the old server with Ctrl+C, then try again.")
        raise SystemExit(1) from error

    print("BMI Neighbor is running!")
    print(f"Open: http://{HOST}:{PORT}")
    print("Keep this terminal open. Press Ctrl+C to stop the website.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nBMI Neighbor stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
