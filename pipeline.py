import logging

logger = logging.getLogger(__name__)

from downloader import (
    get_yesterday_interval_utc,
    download_fragment
)

def pipeline(
    camera_config,
    process_video
):
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
            video_path,
            camera_config["roi"],
            camera_config.get("expanded_roi")
        )

    except Exception:
        logger.exception("Pipeline failed")