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

    def _load_from_ffmpeg(self, video_path):
        cmd = [
            "ffmpeg",
            "-i", video_path,
            "-map", "0:s?",
            "-f", "srt",
            "pipe:1"
        ]

        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        srt_text = result.stdout.decode("utf-8", errors="ignore")

        f = tempfile.NamedTemporaryFile(mode="w+", suffix=".srt", delete=False)
        f.write(srt_text)
        f.close()

        subs = pysubs2.load(f.name, format="srt")
        os.unlink(f.name)
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
        if pos < 0:
            return ""

        return self.texts[pos]

    def get_filename_time(self, frame_idx):
        t = self.get(frame_idx)
        return t.replace(":", "-").replace(" ", "_") if t else f"frame_{frame_idx}"