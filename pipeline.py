from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

from downloader import (
    get_yesterday_interval_utc,
    download_fragment,
)
from engine import process_video
from video_utils import load_yolo_model


logger = logging.getLogger(__name__)


def download_camera_video(camera_config, start, end):
    return download_fragment(
        camera_id=camera_config["camera_id"],
        archive=camera_config["archive"],
        start=start,
        end=end,
    )


def process_camera_video(camera_config, video_path, model=None):
    process_video(
        video_path=video_path,
        roi=camera_config["roi"],
        expanded_roi=camera_config.get("expanded_roi"),
        limit_roi=camera_config.get("limit_roi"),
        mode=camera_config.get("detection_mode"),
        model=model,
    )


def pipeline(camera_config):
    logger.info("Running pipeline for camera: %s", camera_config["camera_id"])

    try:
        start, end = get_yesterday_interval_utc()
        video_path = download_camera_video(camera_config, start, end)
        process_camera_video(camera_config, video_path)
        return True
    except Exception:
        logger.exception("Pipeline failed")
        return False


def _error_text(error):
    return f"{type(error).__name__}: {error}"


def _log_camera_summary(results):
    logger.info("=" * 60)
    logger.info("CAMERA PROCESSING SUMMARY")

    for camera_id, result in results.items():
        status = result["status"]
        if status == "success":
            logger.info("Camera %-6s SUCCESS", camera_id)
        else:
            logger.error(
                "Camera %-6s %-17s %s",
                camera_id,
                status.upper(),
                result.get("error", ""),
            )

    successful = sum(
        result["status"] == "success"
        for result in results.values()
    )
    logger.info(
        "Processed successfully: %d/%d cameras",
        successful,
        len(results),
    )
    logger.info("=" * 60)


def run_cameras(camera_configs, max_download_workers=None):
    """Download cameras concurrently, then process them on one GPU in order."""
    if not camera_configs:
        return {}

    results = {
        camera_id: {
            "status": "pending",
            "video_path": None,
            "error": None,
        }
        for camera_id in camera_configs
    }

    start, end = get_yesterday_interval_utc()
    worker_count = min(
        max_download_workers or len(camera_configs),
        len(camera_configs),
    )
    if worker_count < 1:
        raise ValueError("max_download_workers must be at least 1")

    logger.info(
        "Starting parallel download for %d cameras with %d workers",
        len(camera_configs),
        worker_count,
    )

    with ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="camera-download"
    ) as executor:
        futures = {
            executor.submit(
                download_camera_video,
                camera_config,
                start,
                end,
            ): camera_id
            for camera_id, camera_config in camera_configs.items()
        }

        for future in as_completed(futures):
            camera_id = futures[future]
            try:
                video_path = future.result()
                results[camera_id]["video_path"] = video_path
                results[camera_id]["status"] = "downloaded"
                logger.info(
                    "Camera %s download completed: %s",
                    camera_id,
                    video_path,
                )
            except Exception as error:
                results[camera_id]["status"] = "download_failed"
                results[camera_id]["error"] = _error_text(error)
                logger.exception("Camera %s download failed", camera_id)

    downloaded_camera_ids = [
        camera_id
        for camera_id in camera_configs
        if results[camera_id]["status"] == "downloaded"
    ]

    if downloaded_camera_ids:
        try:
            model = load_yolo_model()
        except Exception as error:
            logger.exception("Cannot load YOLO model")
            for camera_id in downloaded_camera_ids:
                results[camera_id]["status"] = "processing_failed"
                results[camera_id]["error"] = _error_text(error)
        else:
            for position, camera_id in enumerate(
                    downloaded_camera_ids,
                    start=1
            ):
                logger.info(
                    "Processing camera %s on GPU (%d/%d)",
                    camera_id,
                    position,
                    len(downloaded_camera_ids),
                )
                try:
                    process_camera_video(
                        camera_configs[camera_id],
                        results[camera_id]["video_path"],
                        model=model,
                    )
                    results[camera_id]["status"] = "success"
                except Exception as error:
                    results[camera_id]["status"] = "processing_failed"
                    results[camera_id]["error"] = _error_text(error)
                    logger.exception(
                        "Camera %s processing failed",
                        camera_id,
                    )

    _log_camera_summary(results)
    return results
