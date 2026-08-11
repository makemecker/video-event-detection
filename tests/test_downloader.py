import os
import tempfile
import unittest
from unittest.mock import patch

import requests
from urllib3.exceptions import InsecureRequestWarning

from downloader import (
    _authenticate,
    _download_stream_with_progress,
    _format_bytes,
    _format_duration,
    _get_camera_domain_id,
    _get_webclient_base,
    _request,
    download_fragment,
)


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def json(self):
        return self.payload

    def raise_for_status(self):
        return None


class FakeSession:
    def __init__(self, get_responses=None, post_responses=None):
        self.get_responses = list(get_responses or [])
        self.post_responses = list(post_responses or [])
        self.get_calls = []
        self.post_calls = []

    def get(self, url, **kwargs):
        self.get_calls.append((url, kwargs))
        return self.get_responses.pop(0)

    def post(self, url, **kwargs):
        self.post_calls.append((url, kwargs))
        return self.post_responses.pop(0)


class AuthenticationTests(unittest.TestCase):
    def test_returns_access_token(self):
        session = FakeSession(
            post_responses=[FakeResponse({"accessToken": "test-token"})]
        )

        token = _authenticate(
            session,
            "https://cloud.example",
            "user@example.com",
            "password",
            "client-id",
            False
        )

        self.assertEqual(token, "test-token")


class NetworkDiagnosticsTests(unittest.TestCase):
    def test_connection_error_mentions_corporate_vpn(self):
        class OfflineSession:
            def get(self, url, **kwargs):
                raise requests.exceptions.ConnectionError("offline")

        with self.assertRaisesRegex(RuntimeError, "corporate VPN"):
            _request(
                OfflineSession(),
                "get",
                "https://cloud.example/api/v3/test"
            )

    @patch("downloader.requests.Session")
    @patch("downloader.disable_warnings")
    def test_disables_insecure_request_warning_when_ssl_is_not_verified(
            self,
            disable_warnings_mock,
            session_mock,
    ):
        session_mock.side_effect = RuntimeError("stop after warning setup")

        with self.assertRaisesRegex(RuntimeError, "stop after warning setup"):
            download_fragment(
                camera_id="camera",
                archive="archive",
                start="start",
                end="end",
                verify_ssl=False,
            )

        disable_warnings_mock.assert_called_once_with(InsecureRequestWarning)


class DownloadProgressTests(unittest.TestCase):
    def test_formats_sizes_and_durations(self):
        self.assertEqual(_format_bytes(1536), "1.50 KiB")
        self.assertEqual(_format_bytes(2 * 1024 ** 3), "2.00 GiB")
        self.assertEqual(_format_duration(3661), "01:01:01")

    def test_logs_progress_and_writes_the_stream(self):
        class DownloadResponse:
            headers = {"Content-Length": "4"}

            def iter_content(self, chunk_size):
                self.chunk_size = chunk_size
                return iter((b"ab", b"cd"))

        response = DownloadResponse()

        with tempfile.TemporaryDirectory() as directory:
            filepath = os.path.join(directory, "video.mkv")

            with patch(
                    "downloader.time.monotonic",
                    side_effect=(0.0, 5.0, 10.0, 10.0)
            ):
                with self.assertLogs("downloader", level="INFO") as logs:
                    downloaded = _download_stream_with_progress(
                        response=response,
                        filepath=filepath,
                        filename="video.mkv",
                        progress_interval=5,
                        chunk_size=2,
                    )

            with open(filepath, "rb") as file:
                self.assertEqual(file.read(), b"abcd")

        self.assertEqual(downloaded, 4)
        self.assertEqual(response.chunk_size, 2)
        self.assertTrue(
            any("50.0%" in message for message in logs.output)
        )
        self.assertTrue(
            any("ETA 00:00:05" in message for message in logs.output)
        )
        self.assertTrue(
            any("Download finished" in message for message in logs.output)
        )
        self.assertTrue(
            all("[video.mkv]" in message for message in logs.output)
        )


class WebclientRouteTests(unittest.TestCase):
    camera_id = "server/DeviceIpint.1/SourceEndpoint.video:0:0"

    def test_resolves_route_for_camera_domain(self):
        session = FakeSession(get_responses=[
            FakeResponse([
                {"accessPoint": "another-camera", "domainId": 10},
                {"accessPoint": f"hosts/{self.camera_id}", "domainId": 42},
            ]),
            FakeResponse({"publicURL": "/arpserver/987_0/webclient"}),
        ])

        result = _get_webclient_base(
            session,
            "https://cloud.example",
            "test-token",
            self.camera_id,
            False
        )

        self.assertEqual(
            result,
            "https://cloud.example/arpserver/987_0/webclient"
        )
        self.assertEqual(
            session.get_calls[1][0],
            "https://cloud.example/api/v3/ac-backend/public/domains/"
            "42/webclienturl"
        )
        self.assertEqual(
            session.get_calls[0][1]["headers"],
            {"Authorization": "Bearer test-token"}
        )
        self.assertNotIn("params", session.get_calls[0][1])

    def test_rejects_unknown_camera(self):
        session = FakeSession(get_responses=[FakeResponse([])])

        with self.assertRaisesRegex(RuntimeError, "could not be determined"):
            _get_webclient_base(
                session,
                "https://cloud.example",
                "test-token",
                self.camera_id,
                False
            )

    def test_resolves_domain_from_same_device(self):
        cameras = [{
            "accessPoint": (
                "server/DeviceIpint.1/SourceEndpoint.video:0:1"
            ),
            "domainId": 42,
        }]

        self.assertEqual(
            _get_camera_domain_id(cameras, self.camera_id),
            42
        )

    def test_resolves_domain_from_same_server_when_unique(self):
        cameras = [
            {
                "accessPoint": (
                    "server/DeviceIpint.2/SourceEndpoint.video:0:0"
                ),
                "domainId": 42,
            },
            {
                "accessPoint": (
                    "server/DeviceIpint.3/SourceEndpoint.video:0:0"
                ),
                "domainId": 42,
            },
        ]

        self.assertEqual(
            _get_camera_domain_id(cameras, self.camera_id),
            42
        )

    def test_rejects_ambiguous_server_domains(self):
        cameras = [
            {
                "accessPoint": (
                    "server/DeviceIpint.2/SourceEndpoint.video:0:0"
                ),
                "domainId": 42,
            },
            {
                "accessPoint": (
                    "server/DeviceIpint.3/SourceEndpoint.video:0:0"
                ),
                "domainId": 99,
            },
        ]

        with self.assertRaisesRegex(RuntimeError, "could not be determined"):
            _get_camera_domain_id(cameras, self.camera_id)

    def test_rejects_unexpected_public_url(self):
        session = FakeSession(get_responses=[
            FakeResponse([
                {"accessPoint": self.camera_id, "domainId": 42},
            ]),
            FakeResponse({"publicURL": "https://cloud.example/not-a-proxy"}),
        ])

        with self.assertRaisesRegex(RuntimeError, "invalid web client URL"):
            _get_webclient_base(
                session,
                "https://cloud.example",
                "test-token",
                self.camera_id,
                False
            )


if __name__ == "__main__":
    unittest.main()
