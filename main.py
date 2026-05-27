import argparse
import logging
from pipeline import pipeline
import importlib

logger = logging.getLogger(__name__)

MAPPING = {
        "2": ("configs.camera_2", "processors.process_camera_2"),
        "4": ("configs.camera_4", "processors.process_camera_4"),
        "5_1": ("configs.camera_5_1", "processors.process_camera_5_1"),
        "5_2": ("configs.camera_5_2", "processors.process_camera_5_2"),
        "5_3": ("configs.camera_5_3", "processors.process_camera_5_3"),
        "7": ("configs.camera_7", "processors.process_camera_7"),
        "dv": ("configs.camera_dv", "processors.process_camera_dv"),
    }

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
    )

def get_camera(camera_id):

    if camera_id not in MAPPING:
        raise ValueError(f"Unknown camera: {camera_id}")

    config_module, processor_module = MAPPING[camera_id]

    logger.info(f"Loading camera {camera_id}")

    config = importlib.import_module(config_module).CAMERA_CONFIG
    processor = importlib.import_module(processor_module).process_video

    return config, processor

def parse_args():
    parser = argparse.ArgumentParser(
        description="Vision Event Detector"
    )

    parser.add_argument(
        "camera",
        choices=MAPPING.keys(),
        help="Camera floor"
    )

    return parser.parse_args()


def main():
    setup_logging()

    args = parse_args()

    config, processor = get_camera(args.camera)

    pipeline(
        camera_config=config,
        process_video=processor
    )


if __name__ == "__main__":
    main()