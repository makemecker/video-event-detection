import unittest

import requests

from downloader import (
    _authenticate,
    _get_camera_domain_id,
    _get_webclient_base,
    _request,
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
