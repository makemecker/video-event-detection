import os
import cv2
import logging
from datetime import datetime, timedelta
from ultralytics import YOLO
import zipfile
import shutil
from subtitle_provider import SubtitleProvider
import re

logger = logging.getLogger(__name__)

DEFAULT_SETTINGS = {
    "seconds_before": 4,
    "seconds_after": 4,
    "conf_threshold": 0.6,
    "process_every_n_frame": 3,
    "cooldown_seconds": 1,
}

MONTHS = {
    "янв": "01",
    "фев": "02",
    "мар": "03",
    "апр": "04",
    "май": "05",
    "июн": "06",
    "июл": "07",
    "авг": "08",
    "сен": "09",
    "окт": "10",
    "ноя": "11",
    "дек": "12",
}

def parse_video_metadata(video_path):
    filename = os.path.basename(video_path)

    time_part = filename.split("[")[1].split("]")[0]
    start_part = time_part.split("-")[0]

    video_start_datetime = datetime.strptime(start_part, "%Y%m%dT%H%M%S")
    video_start_datetime += timedelta(hours=3)

    cam_date_part = filename.split("[")[0]
    date_str = video_start_datetime.strftime("%d.%m.%Y")

    return {
        "filename": filename,
        "video_start_datetime": video_start_datetime,
        "cam_date_part": cam_date_part,
        "date_str": date_str,
    }

def build_output_paths(cam_date_part, date_str):
    return {
        "output_path": f"{cam_date_part}_{date_str}_output.mp4",
        "event_frames_dir": f"{cam_date_part}_{date_str}_frames",
        "archive_name": f"{cam_date_part}_{date_str}_output.zip",
    }

def init_video_context(video_path, event_frames_dir,
                       seconds_before, seconds_after, cooldown_seconds):

    os.makedirs(event_frames_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Видео не открылось")

    fps = cap.get(cv2.CAP_PROP_FPS)
    subtitle_provider = SubtitleProvider(video_path, fps)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("Video FPS:", fps)
    print("Total frames:", total_frames)
    print("Duration:", total_frames / fps)

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames_before = int(fps * seconds_before)
    frames_after = int(fps * seconds_after)
    cooldown_frames = int(fps * cooldown_seconds)

    logger.info(f"FPS: {fps}")
    logger.info(f"Frames: {total_frames}")
    logger.info(f"Resolution: {frame_w}x{frame_h}")

    logger.info("Loading YOLO...")
    model = YOLO("yolov8x.pt")
    model.to("cuda")

    return {
        "cap": cap,
        "FPS": fps,
        "TOTAL_FRAMES": total_frames,
        "frame_w": frame_w,
        "frame_h": frame_h,

        "frames_before": frames_before,
        "frames_after": frames_after,
        "cooldown_frames": cooldown_frames,

        "model": model,

        "time_format_display": "%Y-%m-%d %H:%M:%S",
        "time_format_filename": "%Y-%m-%d_%H-%M-%S",

        "frame_idx": 0,
        "last_event_frame": -cooldown_frames,
        "event_intervals": [],
        "event_id": 0,
        "person_present": False,
        "subtitle_provider": subtitle_provider,
    }

def normalize_datetime(text):
    for ru, num in MONTHS.items():
        text = text.replace(f"-{ru}-", f"-{num}-")
    return text

def handle_event(
    frame,
    frame_idx,
    frame_h,
    event_frames_dir,
    event_id,
    frames_before,
    frames_after,
    total_frames,
    event_intervals,
    inner_logger,
    subtitle_provider,
):
    print(
        f"EVENT frame={frame_idx}, "
        f"video_sec={frame_idx / subtitle_provider.fps:.3f}, "
        f"subtitle='{subtitle_provider.get(frame_idx)}'"
    )

    raw_time = subtitle_provider.get(frame_idx)
    display_time = normalize_datetime(raw_time or "")
    filename_time = re.sub(r"[^0-9a-zA-Z_-]", "_", raw_time) if raw_time else f"frame_{frame_idx}"

    frame_save = frame.copy()

    cv2.putText(
        frame_save,
        display_time,
        (30, frame_h - 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )

    frame_filename = os.path.join(
        event_frames_dir,
        f"event_{event_id}_{filename_time}.jpg"
    )

    cv2.imwrite(frame_filename, frame_save)

    start = max(0, frame_idx - frames_before)
    end = min(total_frames - 1, frame_idx + frames_after)

    event_intervals.append((start, end))

    inner_logger.info(f"Event {event_id} at frame {frame_idx}")

    return event_id + 1

def merge_intervals(event_intervals):
    event_intervals.sort()
    merged = []

    for interval in event_intervals:
        if not merged:
            merged.append(interval)
        else:
            prev_start, prev_end = merged[-1]
            curr_start, curr_end = interval

            if curr_start <= prev_end:
                merged[-1] = (prev_start, max(prev_end, curr_end))
            else:
                merged.append(interval)

    return merged

def write_output_video(
    video_path,
    output_path,
    merged_intervals,
    fps,
    frame_w,
    frame_h,
    total_frames,
    time_format_display,
    subtitle_provider
):
    from tqdm import tqdm

    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_w, frame_h))

    frame_idx = 0
    interval_idx = 0

    with tqdm(total=total_frames, desc="Writing output") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret or interval_idx >= len(merged_intervals):
                break

            start, end = merged_intervals[interval_idx]

            if frame_idx > end:
                interval_idx += 1

            elif start <= frame_idx <= end:
                display_time = normalize_datetime(subtitle_provider.get(frame_idx) or "")

                cv2.putText(
                    frame,
                    display_time,
                    (30, frame_h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 255),
                    2
                )

                out.write(frame)

            frame_idx += 1
            pbar.update(1)

    cap.release()
    out.release()

def create_archive(
    archive_name,
    output_path,
    event_frames_dir,
):
    with zipfile.ZipFile(archive_name, "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(output_path, os.path.basename(output_path))

        for root, dirs, files in os.walk(event_frames_dir):
            for file in files:
                full_path = os.path.join(root, file)
                relative_path = os.path.relpath(full_path, start=event_frames_dir)
                zipf.write(
                    full_path,
                    os.path.join("event_frames", relative_path)
                )

def cleanup(output_path, event_frames_dir, video_path, archive_name):

    os.remove(output_path)
    shutil.rmtree(event_frames_dir)

    if os.path.exists(archive_name):
        os.remove(video_path)