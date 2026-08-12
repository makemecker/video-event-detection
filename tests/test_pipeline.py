import threading
import unittest
from unittest.mock import patch

from pipeline import download_camera_video, run_cameras


def camera_config(camera_id):
    return {
        "camera_id": camera_id,
        "archive": f"archive-{camera_id}",
        "roi": {},
    }


class MultiCameraPipelineTests(unittest.TestCase):
    def test_download_mode_selects_fragmented_downloader_only_with_flag(self):
        config = camera_config("camera-4")

        with (
            patch("pipeline.download_fragment", return_value="full.mkv") as full,
            patch(
                "pipeline.download_fragmented_video",
                return_value="merged.mkv",
            ) as fragmented,
        ):
            self.assertEqual(
                download_camera_video(config, "start", "end"),
                "full.mkv",
            )
            self.assertEqual(
                download_camera_video(
                    config,
                    "start",
                    "end",
                    split_download=True,
                ),
                "merged.mkv",
            )

        full.assert_called_once()
        fragmented.assert_called_once()

    def test_downloads_in_parallel_and_processes_in_order_with_one_model(self):
        configs = {
            "2": camera_config("camera-2"),
            "4": camera_config("camera-4"),
        }
        download_barrier = threading.Barrier(2)
        downloaded = []
        processed = []
        shared_model = object()

        def fake_download(config, start, end, split_download=False):
            self.assertEqual((start, end), ("start", "end"))
            self.assertFalse(split_download)
            download_barrier.wait(timeout=2)
            downloaded.append(config["camera_id"])
            return f"{config['camera_id']}.mkv"

        def fake_process(config, video_path, model=None):
            self.assertEqual(len(downloaded), 2)
            self.assertIs(model, shared_model)
            processed.append((config["camera_id"], video_path))

        with (
            patch(
                "pipeline.get_yesterday_interval_utc",
                return_value=("start", "end"),
            ),
            patch("pipeline.download_camera_video", side_effect=fake_download),
            patch("pipeline.load_yolo_model", return_value=shared_model),
            patch("pipeline.process_camera_video", side_effect=fake_process),
        ):
            results = run_cameras(configs, max_download_workers=2)

        self.assertEqual(
            processed,
            [
                ("camera-2", "camera-2.mkv"),
                ("camera-4", "camera-4.mkv"),
            ],
        )
        self.assertEqual(results["2"]["status"], "success")
        self.assertEqual(results["4"]["status"], "success")

    def test_download_failure_does_not_stop_other_cameras(self):
        configs = {
            "2": camera_config("camera-2"),
            "4": camera_config("camera-4"),
        }
        processed = []

        def fake_download(config, start, end, split_download=False):
            del start, end
            self.assertFalse(split_download)
            if config["camera_id"] == "camera-2":
                raise RuntimeError("download unavailable")
            return "camera-4.mkv"

        def fake_process(config, video_path, model=None):
            del model
            processed.append((config["camera_id"], video_path))

        with (
            patch(
                "pipeline.get_yesterday_interval_utc",
                return_value=("start", "end"),
            ),
            patch("pipeline.download_camera_video", side_effect=fake_download),
            patch("pipeline.load_yolo_model", return_value=object()),
            patch("pipeline.process_camera_video", side_effect=fake_process),
        ):
            results = run_cameras(configs, max_download_workers=2)

        self.assertEqual(results["2"]["status"], "download_failed")
        self.assertIn("download unavailable", results["2"]["error"])
        self.assertEqual(results["4"]["status"], "success")
        self.assertEqual(processed, [("camera-4", "camera-4.mkv")])

    def test_split_download_flag_is_forwarded_to_every_camera(self):
        configs = {
            "2": camera_config("camera-2"),
            "4": camera_config("camera-4"),
        }
        split_flags = []

        def fake_download(config, start, end, split_download=False):
            del start, end
            split_flags.append((config["camera_id"], split_download))
            return f"{config['camera_id']}.mkv"

        with (
            patch(
                "pipeline.get_yesterday_interval_utc",
                return_value=("start", "end"),
            ),
            patch("pipeline.download_camera_video", side_effect=fake_download),
            patch("pipeline.load_yolo_model", return_value=object()),
            patch("pipeline.process_camera_video"),
        ):
            run_cameras(
                configs,
                max_download_workers=2,
                split_download=True,
            )

        self.assertCountEqual(
            split_flags,
            [("camera-2", True), ("camera-4", True)],
        )


if __name__ == "__main__":
    unittest.main()
