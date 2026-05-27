import logging
from tqdm import tqdm
from video_utils import (
    parse_video_metadata,
    build_output_paths,
    DEFAULT_SETTINGS,
    init_video_context,
    handle_event,
    merge_intervals,
    write_output_video,
    create_archive,
    cleanup
)

logger = logging.getLogger(__name__)


def boxes_intersect(a, b):
    return not (
            a["x2"] < b["x1"] or
            a["x1"] > b["x2"] or
            a["y2"] < b["y1"] or
            a["y1"] > b["y2"]
    )


def box_inside_with_margin(a, b, margin=15):
    return (
            a["x1"] >= b["x1"] - margin and
            a["y1"] >= b["y1"] - margin and
            a["x2"] <= b["x2"] + margin and
            a["y2"] <= b["y2"] + margin
    )


def process_video(video_path, roi, expanded_roi, limit_roi=None, mode="center_point"):
    if mode not in ("center_point", "crossing"):
        raise ValueError(f"Unknown detection_mode: {mode}")
    try:
        logger.info(f"Видео: {video_path}")

        meta = parse_video_metadata(video_path)
        paths = build_output_paths(meta["cam_date_part"], meta["date_str"])
        settings = DEFAULT_SETTINGS

        video_start_datetime = meta["video_start_datetime"]

        output_path = paths["output_path"]
        event_frames_dir = paths["event_frames_dir"]
        archive_name = paths["archive_name"]

        seconds_before = settings["seconds_before"]
        seconds_after = settings["seconds_after"]
        conf_threshold = settings["conf_threshold"]
        process_every_n_frame = settings["process_every_n_frame"]
        cooldown_seconds = settings["cooldown_seconds"]

        ctx = init_video_context(
            video_path,
            event_frames_dir,
            seconds_before,
            seconds_after,
            cooldown_seconds
        )

        cap = ctx["cap"]
        fps = ctx["FPS"]
        total_frames = ctx["TOTAL_FRAMES"]
        frame_w = ctx["frame_w"]
        frame_h = ctx["frame_h"]
        frames_before = ctx["frames_before"]
        frames_after = ctx["frames_after"]
        cooldown_frames = ctx["cooldown_frames"]
        model = ctx["model"]
        time_format_display = ctx["time_format_display"]
        time_format_filename = ctx["time_format_filename"]
        frame_idx = ctx["frame_idx"]
        last_event_frame = ctx["last_event_frame"]
        event_intervals = ctx["event_intervals"]
        event_id = ctx["event_id"]
        person_present = ctx["person_present"]

        if expanded_roi["x2"] <= expanded_roi["x1"] or \
                expanded_roi["y2"] <= expanded_roi["y1"]:
            raise ValueError("Expanded ROI invalid")

        with tqdm(total=total_frames, desc="Detecting events", mininterval=0.5) as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % process_every_n_frame == 0:

                    roi_frame = frame[
                                expanded_roi["y1"]:expanded_roi["y2"],
                                expanded_roi["x1"]:expanded_roi["x2"]
                                ]

                    results = model(roi_frame, verbose=False)[0]

                    person_detected = False

                    for box in results.boxes:
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])

                        if cls_id == 0 and conf >= conf_threshold:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                            # возвращаем координаты в систему всего кадра
                            x1 += expanded_roi["x1"]
                            x2 += expanded_roi["x1"]
                            y1 += expanded_roi["y1"]
                            y2 += expanded_roi["y1"]

                            if mode == "center_point":
                                # контрольная точка
                                cx = (x1 + x2) / 2
                                cy = y1 + 0.35 * (y2 - y1)

                                # проверка попадания точки в roi
                                if roi["x1"] <= cx <= roi["x2"] and roi["y1"] <= cy <= roi["y2"]:
                                    person_detected = True
                                    break

                            elif mode == "crossing":
                                if limit_roi is None:
                                    raise ValueError("limit_roi is required for crossing mode")

                                person_box = {
                                    "x1": x1,
                                    "y1": y1,
                                    "x2": x2,
                                    "y2": y2
                                }

                                cy = (y1 + y2) / 2

                                if (
                                        boxes_intersect(person_box, roi) and
                                        y1 < roi["y1"] <= cy <= roi["y2"] and
                                        box_inside_with_margin(person_box, limit_roi, margin=0)
                                ):
                                    person_detected = True
                                    break

                    if person_detected and not person_present and \
                            frame_idx - last_event_frame > cooldown_frames:
                        last_event_frame = frame_idx
                        person_present = True

                        event_id = handle_event(
                            frame=frame,
                            frame_idx=frame_idx,
                            video_start_datetime=video_start_datetime,
                            fps=fps,
                            time_format_display=time_format_display,
                            time_format_filename=time_format_filename,
                            frame_h=frame_h,
                            event_frames_dir=event_frames_dir,
                            event_id=event_id,
                            frames_before=frames_before,
                            frames_after=frames_after,
                            total_frames=total_frames,
                            event_intervals=event_intervals,
                            inner_logger=logger
                        )

                    if not person_detected:
                        person_present = False

                frame_idx += 1
                pbar.update(1)

        cap.release()

        logger.info("Merging intervals...")

        merged = merge_intervals(event_intervals)

        logger.info(f"Events: {event_id}")
        logger.info(f"Merged intervals: {len(merged)}")

        logger.info("=== WRITING VIDEO ===")

        write_output_video(
            video_path=video_path,
            output_path=output_path,
            merged_intervals=merged,
            fps=fps,
            frame_w=frame_w,
            frame_h=frame_h,
            total_frames=total_frames,
            video_start_datetime=video_start_datetime,
            time_format_display=time_format_display,
        )

        logger.info(f"Видео сохранено: {output_path}")

        logger.info("Creating archive...")

        create_archive(
            archive_name=archive_name,
            output_path=output_path,
            event_frames_dir=event_frames_dir,
        )

        cleanup(
            output_path=output_path,
            event_frames_dir=event_frames_dir,
            video_path=video_path,
            archive_name=archive_name,
        )

        logger.info(f"Done: {archive_name}")
        logger.info("=== FINISHED ===")

    except Exception:
        logger.exception("CRASH")
        raise
