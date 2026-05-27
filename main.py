import argparse
import logging
import importlib
from pipeline import pipeline

logger = logging.getLogger(__name__)

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
    )

def get_camera(camera_id):
    import_path = f"configs.camera_{camera_id}"

    try:
        module = importlib.import_module(import_path)
    except ModuleNotFoundError:
        raise ValueError(f"Unknown camera: {camera_id}")

    return module.CAMERA_CONFIG


def parse_args():
    parser = argparse.ArgumentParser(description="Vision Event Detector")

    parser.add_argument(
        "camera",
        help="Camera ID"
    )

    return parser.parse_args()


def main():
    setup_logging()

    args = parse_args()
    config = get_camera(args.camera)

    pipeline(camera_config=config)


if __name__ == "__main__":
    main()