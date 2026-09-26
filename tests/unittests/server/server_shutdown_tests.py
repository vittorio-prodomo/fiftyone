"""
FiftyOne Server shutdown tests.

| Copyright 2017-2026, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

import json
import os
import signal
import socket
import subprocess
import sys
import time

import pytest

import fiftyone.server.main as fosm


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_listening(port, proc, timeout=60):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            pytest.fail("server exited during startup")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.25)
    pytest.fail("server never started listening")


@pytest.mark.skipif(sys.platform.startswith("win"), reason="POSIX signals")
def test_sigterm_exits_while_an_events_stream_is_open():
    # On Python >= 3.12 hypercorn's shutdown awaits Server.wait_closed(), which
    # waits for EVERY open connection (hypercorn#308). sse_starlette only ends
    # its streams on uvicorn's exit signal, so an open App tab used to keep the
    # server alive forever and session.close() hung in os.waitpid().
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, fosm.__file__, "--port", str(port), "--address", "127.0.0.1"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env={**os.environ, "FIFTYONE_DISABLE_SERVICES": "1"},
    )
    stream = None
    try:
        _wait_listening(port, proc)
        body = json.dumps(
            {"subscription": "shutdown-test", "events": ["state_update"], "initializer": None}
        ).encode()
        stream = socket.create_connection(("127.0.0.1", port))
        stream.sendall(
            b"POST /events HTTP/1.1\r\nHost: 127.0.0.1\r\n"
            b"Content-Type: application/json\r\n"
            + f"Content-Length: {len(body)}\r\n\r\n".encode()
            + body
        )
        stream.settimeout(10)
        assert stream.recv(64).startswith(b"HTTP/1.1 200")

        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            pytest.fail("server still alive 15 s after SIGTERM with an open events stream")
    finally:
        if stream is not None:
            stream.close()
        if proc.poll() is None:
            proc.kill()
            proc.wait()
