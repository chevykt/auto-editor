"""python -m web — start the server and open a browser."""
from __future__ import annotations

import threading
import webbrowser

import uvicorn

HOST = "127.0.0.1"
PORT = 8000


def main() -> None:
    url = f"http://{HOST}:{PORT}"
    print(f"Auto-Editor running at {url}")
    print("Everything stays on this machine. Ctrl-C to stop.")
    # give uvicorn a moment to bind, so the browser does not race it
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    uvicorn.run("web.server:app", host=HOST, port=PORT, log_level="warning")


if __name__ == "__main__":
    main()
