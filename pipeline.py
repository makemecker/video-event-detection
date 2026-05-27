import logging

logger = logging.getLogger(__name__)

from downloader import (
    get_yesterday_interval_utc,
    download_fragment
)
from processors.expanded_roi_detection import process_video

def pipeline(camera_config):
    logger.info(f"Running pipeline for camera: {camera_config['camera_id']}")

    try:
        start, end = (
            get_yesterday_interval_utc()
        )

        video_path = download_fragment(
            camera_id=
                camera_config["camera_id"],

            archive=
                camera_config["archive"],

            start=start,
            end=end
        )

        process_video(
            video_path=video_path,
            roi=camera_config["roi"],
            expanded_roi=camera_config.get("expanded_roi"),
            limit_roi=camera_config.get("limit_roi"),
            mode=camera_config.get("detection_mode")
        )

    except Exception:
        logger.exception("Pipeline failed")