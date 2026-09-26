"""
FiftyOne Server media route unit tests.

| Copyright 2017-2026, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from fiftyone.server.routes.media import Media


def test_webp_is_served_as_an_image(tmp_path):
    # Python < 3.13 has no built-in .webp type and many systems' mime.types
    # lack it too; without the route's registration the file goes out as
    # text/plain and only content-sniffing browsers still draw it.
    path = tmp_path / "thumb.webp"
    path.write_bytes(b"RIFF\x1a\x00\x00\x00WEBPVP8L\x0d\x00\x00\x00/\x00\x00\x00\x10\x07\x10\x11\x11\x88\x88\xfe\x07\x00")

    client = TestClient(Starlette(routes=[Route("/media", Media)]))
    response = client.get("/media", params={"filepath": str(path)})

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/webp"
