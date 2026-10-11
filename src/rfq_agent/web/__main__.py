"""python -m rfq_agent.web: serve the console on 127.0.0.1:$PORT (default 4103)."""

from __future__ import annotations

import os

import uvicorn


def main() -> None:
    uvicorn.run(
        "rfq_agent.web.app:app",
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "4103")),
        log_level=os.environ.get("LOG_LEVEL", "info"),
    )


if __name__ == "__main__":
    main()
