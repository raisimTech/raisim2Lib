"""Prevent runtime-joint capture from recording an unrelated running server."""

import socket
import sys
import unittest
from pathlib import Path
from unittest import mock

GENERATORS = Path(__file__).resolve().parents[1] / "image_generators"
sys.path.insert(0, str(GENERATORS))
import runtime_joint_media as media
sys.path.pop(0)


class RuntimeJointMediaTest(unittest.TestCase):
    def test_occupied_endpoint_is_rejected_before_starting_capture(self):
        with socket.socket() as existing_server:
            existing_server.bind(("127.0.0.1", 0))
            existing_server.listen(1)
            port = existing_server.getsockname()[1]
            check = media.require_available_endpoint
            with mock.patch.object(media, "require_available_endpoint", lambda: check(port)):
                with mock.patch.object(media.clips, "build_hook") as build_hook:
                    with self.assertRaisesRegex(RuntimeError, "occupied"):
                        media.record(Path("unused-build"), Path("unused-output"), Path("unused-work"))
                    build_hook.assert_not_called()


if __name__ == "__main__":
    unittest.main()
