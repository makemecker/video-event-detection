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

def process_video(video_path):
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

        # roi = {
        #     "x1": 2,
        #     "y1": 341,
        #     "x2": 102,
        #     "y2": 462
        # }

        roi = {
            "x1": 0,
            "y1": 114,
            "x2": 36,
            "y2": 154
        }

        roi_width = roi["x2"] - roi["x1"]
        roi_height = roi["y2"] - roi["y1"]

        margin_factor = 2.5
        margin = int(max(roi_width, roi_height) * margin_factor)

        expanded_roi = {
            "x1": max(0, roi["x1"] - margin),
            "y1": max(0, roi["y1"] - margin),
            "x2": min(frame_w, roi["x2"] + margin),
            "y2": min(frame_h, roi["y2"] + margin),
        }

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

                    person_detected_this_frame = False

                    for box in results.boxes:
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])

                        if cls_id == 0 and conf >= conf_threshold:

                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                            x1 += expanded_roi["x1"]
                            x2 += expanded_roi["x1"]
                            y1 += expanded_roi["y1"]
                            y2 += expanded_roi["y1"]

                            cx = (x1 + x2) / 2
                            cy = y1 + 0.35 * (y2 - y1)

                            if (roi["x1"] <= cx <= roi["x2"] and
                                    roi["y1"] <= cy <= roi["y2"]):
                                person_detected_this_frame = True
                                break

                    if person_detected_this_frame and not person_present and \
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

                    if not person_detected_this_frame:
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