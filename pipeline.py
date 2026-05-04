import logging

logger = logging.getLogger(__name__)

from downloader import (
    get_yesterday_interval_utc,
    download_fragment
)

from credentials import (
    EMAIL,
    PASSWORD,
    CLIENT_ID
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
            email=EMAIL,
            password=PASSWORD,
            client_id=CLIENT_ID,

            proxy_key=
                camera_config["proxy_key"],

            camera_id=
                camera_config["camera_id"],

            archive=
                camera_config["archive"],

            start=start,
            end=end
        )

        process_video(
            video_path
        )

    except Exception:
        logger.exception("Pipeline failed")