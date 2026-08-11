import unittest

from main import (
    DEFAULT_CAMERA_IDS,
    all_cameras_succeeded,
    resolve_camera_ids,
)


class MainCameraSelectionTests(unittest.TestCase):
    def test_all_resolves_to_configured_camera_list(self):
        expected = [
            "2", "4", "5_1", "5_2", "5_3", "5_dv", "5_k", "7", "7_st"
        ]
        self.assertEqual(list(DEFAULT_CAMERA_IDS), expected)
        self.assertEqual(resolve_camera_ids(["all"]), expected)

    def test_explicit_camera_ids_are_deduplicated_in_order(self):
        self.assertEqual(
            resolve_camera_ids(["4", "2", "4"]),
            ["4", "2"],
        )

    def test_all_cannot_be_combined_with_ids(self):
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            resolve_camera_ids(["all", "2"])

    def test_success_requires_every_camera(self):
        self.assertTrue(all_cameras_succeeded({
            "2": {"status": "success"},
            "4": {"status": "success"},
        }))
        self.assertFalse(all_cameras_succeeded({
            "2": {"status": "success"},
            "4": {"status": "download_failed"},
        }))


if __name__ == "__main__":
    unittest.main()
