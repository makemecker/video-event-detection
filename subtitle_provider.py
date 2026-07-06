import subprocess
import pysubs2
from bisect import bisect_right
import tempfile
import os

class SubtitleProvider:
    """
    Streams subtitles directly from MKV via ffmpeg and builds fast lookup.
    """

    def __init__(self, video_path, fps):
        self.fps = fps
        self.subs = self._load_from_ffmpeg(video_path)
        self.index = self._build_index()

        # sorted frame timestamps for fast lookup
        self.frames = [x[0] for x in self.index]# seconds
        self.texts = [x[1] for x in self.index]

        # ==== Диагностика ====
        print(f"FPS = {self.fps}")

        if self.subs:
            print("First subtitle:")
            print("  start =", self.subs[0].start)
            print("  text  =", repr(self.subs[0].text))

            print("Last subtitle:")
            print("  start =", self.subs[-1].start)
            print("  text  =", repr(self.subs[-1].text))

        print("First index:", self.index[:5])
        print("Last index:", self.index[-5:])

    def _load_from_ffmpeg(self, video_path):
        tmp = tempfile.NamedTemporaryFile(suffix=".srt", delete=False)
        tmp_path = tmp.name
        tmp.close()

        subprocess.run([
            "ffmpeg",
            "-y",
            "-i", video_path,
            "-map", "0:s:0",
            "-c", "copy",
            tmp_path
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        subs = pysubs2.load(tmp_path, encoding="utf-8")

        os.remove(tmp_path)

        return subs

    def _build_index(self):
        index = []

        for line in self.subs:
            start_sec = line.start / 1000.0
            text = line.text.strip()

            if text:
                index.append((start_sec, text))

        return index

    def get(self, frame_idx):
        t = frame_idx / self.fps  # секунды

        pos = bisect_right(self.frames, t) - 1

        if frame_idx % 1000 == 0:  # чтобы не засорять лог
            print(
                f"frame={frame_idx}, "
                f"time={t:.3f}, "
                f"pos={pos}, "
                f"text={self.texts[pos] if pos >= 0 else ''}"
            )

        if pos < 0:
            return ""

        return self.texts[pos]

    def get_filename_time(self, frame_idx):
        t = self.get(frame_idx)
        return t.replace(":", "-").replace(" ", "_") if t else f"frame_{frame_idx}"