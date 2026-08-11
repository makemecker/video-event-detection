import argparse
import importlib
import logging
import time

from pipeline import pipeline, run_cameras


logger = logging.getLogger(__name__)

DEFAULT_CAMERA_IDS = (
    "2",
    "4",
    "5_1",
    "5_2",
    "5_3",
    "5_dv",
    "5_k",
    "7",
    "7_st",
)


def setup_logging():
    log_format = "%(asctime)s [%(levelname)s] [%(name)s] %(message)s"

    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                "video_detector.log",
                encoding="utf-8",
            ),
        ],
    )


def get_camera(camera_id):
    import_path = f"configs.camera_{camera_id}"

    try:
        module = importlib.import_module(import_path)
    except ModuleNotFoundError as error:
        if error.name != import_path:
            raise
        raise ValueError(f"Unknown camera: {camera_id}") from error

    return module.CAMERA_CONFIG


def positive_integer(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def parse_args():
    parser = argparse.ArgumentParser(description="Vision Event Detector")

    parser.add_argument(
        "camera",
        nargs="+",
        help="Camera ID(s), or 'all'",
    )
    parser.add_argument(
        "--download-workers",
        type=positive_integer,
        default=None,
        help="Maximum parallel downloads (default: all selected cameras)",
    )

    return parser.parse_args()


def resolve_camera_ids(values):
    if "all" in values:
        if len(values) != 1:
            raise ValueError("'all' cannot be combined with camera IDs")
        return list(DEFAULT_CAMERA_IDS)

    return list(dict.fromkeys(values))


def all_cameras_succeeded(results):
    return bool(results) and all(
        result["status"] == "success"
        for result in results.values()
    )


def main():
    setup_logging()

    start_time = time.perf_counter()
    successful = False

    try:
        args = parse_args()
        camera_ids = resolve_camera_ids(args.camera)
        camera_configs = {
            camera_id: get_camera(camera_id)
            for camera_id in camera_ids
        }

        if len(camera_configs) == 1:
            successful = pipeline(
                camera_config=next(iter(camera_configs.values()))
            )
        else:
            results = run_cameras(
                camera_configs,
                max_download_workers=args.download_workers,
            )
            successful = all_cameras_succeeded(results)
    finally:
        elapsed = time.perf_counter() - start_time

        minutes = int(elapsed // 60)
        seconds = elapsed % 60

        logger.info("=" * 60)
        logger.info(
            "Pipeline finished in %d min %.1f sec (%.2f min)",
            minutes,
            seconds,
            elapsed / 60,
        )
        logger.info("=" * 60)

    if not successful:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
