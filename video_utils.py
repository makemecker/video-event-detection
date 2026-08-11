import os
import cv2
import logging
from datetime import datetime, timedelta
import zipfile
import shutil
from subtitle_provider import SubtitleProvider
import re
from tqdm import tqdm
import subprocess
import json
from pathlib import Path
import tempfile
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

logger = logging.getLogger(__name__)

DEFAULT_SETTINGS = {
    "seconds_before": 4,
    "seconds_after": 4,
    "conf_threshold": 0.6,
    "process_every_n_frame": 3,
    "cooldown_seconds": 1,
}


def load_yolo_model():
    logger.info("Loading YOLO...")
    from ultralytics import YOLO

    model_path = Path(
        os.getenv(
            "YOLO_MODEL_PATH",
            str(PROJECT_ROOT / "yolov8x.pt"),
        )
    )
    if not model_path.is_file():
        raise FileNotFoundError(f"YOLO model not found: {model_path}")

    model = YOLO(str(model_path))
    model.to("cuda")
    return model

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
                       seconds_before, seconds_after, cooldown_seconds,
                       model=None):

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

    if model is None:
        model = load_yolo_model()
    else:
        logger.info("Using preloaded YOLO model")

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
    video_time_sec,
    frame_h,
    event_frames_dir,
    event_id,
    seconds_before,
    seconds_after,
    video_duration_sec,
    event_intervals,
    inner_logger,
    subtitle_provider,
):
    raw_time = subtitle_provider.get_by_time(
        video_time_sec
    )

    display_time = normalize_datetime(
        raw_time or ""
    )

    filename_time = (
        display_time
        .replace(":", "-")
        .replace(" ", "_")
        if display_time
        else f"frame_{frame_idx}"
    )

    frame_save = frame.copy()

    # На скриншот накладывается точное время по PTS.
    if display_time:
        cv2.putText(
            frame_save,
            display_time,
            (30, frame_h - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2,
        )

    frame_filename = os.path.join(
        event_frames_dir,
        f"event_{event_id}_{filename_time}.jpg",
    )

    if not cv2.imwrite(
        frame_filename,
        frame_save,
    ):
        inner_logger.warning(
            "Не удалось сохранить кадр события: %s",
            frame_filename,
        )

    start_sec = max(
        0.0,
        video_time_sec - seconds_before,
    )

    end_sec = min(
        video_duration_sec,
        video_time_sec + seconds_after,
    )

    # Интервалы теперь хранятся в секундах PTS.
    event_intervals.append(
        (start_sec, end_sec)
    )

    inner_logger.info(
        "Event %d: frame=%d, pts=%.3f, "
        "interval=%.3f-%.3f, time=%r",
        event_id,
        frame_idx,
        video_time_sec,
        start_sec,
        end_sec,
        raw_time,
    )

    return event_id + 1

def merge_intervals(
    event_intervals,
    tolerance_sec=0.001,
):
    if not event_intervals:
        return []

    sorted_intervals = sorted(
        event_intervals
    )

    merged = [
        sorted_intervals[0]
    ]

    for current_start, current_end in sorted_intervals[1:]:
        previous_start, previous_end = merged[-1]

        if (
            current_start
            <= previous_end + tolerance_sec
        ):
            merged[-1] = (
                previous_start,
                max(previous_end, current_end),
            )
        else:
            merged.append(
                (current_start, current_end)
            )

    return merged

def write_output_video(
    video_path,
    output_path,
    merged_intervals,
    subtitle_provider,
):
    """
    Создаёт итоговое видео из PTS-интервалов.

    Каждый короткий фрагмент:
    - вырезается из оригинального видео;
    - получает локальный SRT;
    - перекодируется через libx264;
    - затем все фрагменты объединяются без повторного кодирования.
    """

    if not merged_intervals:
        raise ValueError(
            "Нет интервалов для записи"
        )

    video_path = Path(video_path).resolve()
    output_path = Path(output_path).resolve()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    work_dir = Path(
        tempfile.mkdtemp(
            prefix="video_event_clips_"
        )
    )

    clip_paths = []

    try:
        total_intervals = len(
            merged_intervals
        )

        for index, (
            start_sec,
            end_sec,
        ) in enumerate(merged_intervals):
            duration_sec = (
                end_sec - start_sec
            )

            if duration_sec <= 0:
                logger.warning(
                    "Некорректный интервал пропущен: "
                    "%.3f-%.3f",
                    start_sec,
                    end_sec,
                )
                continue

            clip_path = (
                work_dir
                / f"clip_{index:05d}.mp4"
            )

            srt_path = (
                work_dir
                / f"clip_{index:05d}.srt"
            )

            subtitle_provider.write_interval_srt(
                start_sec=start_sec,
                end_sec=end_sec,
                output_path=srt_path,
            )

            subtitle_filter = (
                make_subtitle_filter(
                    srt_path
                )
            )

            command = [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-v", "error",

                # Быстрое позиционирование до нужного интервала.
                "-ss", f"{start_sec:.6f}",
                "-i", str(video_path),

                # Длительность фрагмента.
                "-t", f"{duration_sec:.6f}",

                "-map", "0:v:0",
                "-an",

                # Сбрасываем PTS в ноль, поскольку локальный
                # SRT также начинается с 00:00:00.
                "-vf",
                (
                    "setpts=PTS-STARTPTS,"
                    + subtitle_filter
                ),

                "-c:v", "libx264",
                "-preset", "veryfast",
                "-crf", "20",

                # Используется для совместимости со старым FFmpeg.
                "-vsync", "0",

                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",

                str(clip_path),
            ]

            run_ffmpeg(
                command,
                (
                    f"Рендер фрагмента "
                    f"{index + 1}/{total_intervals}: "
                    f"{start_sec:.3f}-"
                    f"{end_sec:.3f}"
                ),
            )

            if (
                not clip_path.exists()
                or clip_path.stat().st_size == 0
            ):
                raise RuntimeError(
                    f"Не создан фрагмент: {clip_path}"
                )

            clip_paths.append(
                clip_path
            )

        if not clip_paths:
            raise RuntimeError(
                "Не создано ни одного фрагмента"
            )

        # Если клип всего один, concat всё равно можно использовать,
        # чтобы логика оставалась единой.
        concat_file_path = (
            work_dir / "concat.txt"
        )

        with concat_file_path.open(
            "w",
            encoding="utf-8",
        ) as concat_file:
            for clip_path in clip_paths:
                escaped_path = escape_concat_path(
                    clip_path
                )

                concat_file.write(
                    f"file '{escaped_path}'\n"
                )

        concat_command = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-v", "error",

            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_file_path),

            # Повторное кодирование не выполняется.
            "-c", "copy",

            "-movflags", "+faststart",
            str(output_path),
        ]

        run_ffmpeg(
            concat_command,
            "Объединение видеофрагментов",
        )

        if (
            not output_path.exists()
            or output_path.stat().st_size == 0
        ):
            raise RuntimeError(
                f"Итоговое видео не создано: {output_path}"
            )

    finally:
        shutil.rmtree(
            work_dir,
            ignore_errors=True,
        )

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

def get_video_duration(video_path: str) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "json",
            video_path,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "Не удалось определить длительность видео:\n"
            + result.stderr.strip()
        )

    data = json.loads(result.stdout)
    duration = data.get("format", {}).get("duration")

    if duration is None:
        raise RuntimeError(
            f"FFprobe не вернул длительность видео: {video_path}"
        )

    return float(duration)

def run_ffmpeg(
    command,
    description,
):
    logger.info(description)

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"{description} завершилось с ошибкой:\n"
            f"{result.stderr.strip()}"
        )


def escape_subtitle_filter_path(path):
    value = str(
        Path(path).resolve()
    ).replace("\\", "/")

    value = value.replace(":", r"\:")
    value = value.replace("'", r"\'")
    value = value.replace("[", r"\[")
    value = value.replace("]", r"\]")

    return value


def escape_concat_path(path):
    return str(
        Path(path).resolve()
    ).replace("'", r"'\''")


def make_subtitle_filter(srt_path):
    escaped_path = escape_subtitle_filter_path(
        srt_path
    )

    # В тесте FontSize=42 оказался слишком крупным.
    # 18 — примерно в 2,3 раза меньше.
    return (
        f"subtitles='{escaped_path}'"
        ":force_style="
        "'FontSize=18,"
        "PrimaryColour=&H0000FFFF,"
        "OutlineColour=&H00000000,"
        "Outline=2,"
        "MarginV=30,"
        "MarginL=25,"
        "Alignment=1'"
    )
