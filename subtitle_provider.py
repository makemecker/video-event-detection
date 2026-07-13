import subprocess
import pysubs2
from bisect import bisect_right
import tempfile
import os

class SubtitleProvider:
    """
    Извлекает встроенные SRT-субтитры и выполняет поиск по PTS.
    """

    def __init__(self, video_path, fps=None):
        # fps оставлен для совместимости со старым вызовом,
        # но для определения времени больше не используется.
        self.fps = fps

        self.subs = self._load_from_ffmpeg(video_path)
        self.index = self._build_index()

        # Здесь находятся секунды медиатаймлайна, а не номера кадров.
        self.timestamps = [
            timestamp
            for timestamp, _ in self.index
        ]

        self.texts = [
            text
            for _, text in self.index
        ]

        if not self.timestamps:
            raise ValueError("Индекс субтитров пуст")

    def _load_from_ffmpeg(self, video_path):
        tmp = tempfile.NamedTemporaryFile(
            suffix=".srt",
            delete=False,
        )
        tmp_path = tmp.name
        tmp.close()

        try:
            result = subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-v", "error",
                    "-i", video_path,
                    "-map", "0:s:0",
                    "-c", "copy",
                    tmp_path,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            if result.returncode != 0:
                raise RuntimeError(
                    "Не удалось извлечь субтитры:\n"
                    + result.stderr.strip()
                )

            subs = pysubs2.load(
                tmp_path,
                encoding="utf-8",
            )

            if not subs:
                raise ValueError(
                    f"В видео нет субтитров: {video_path}"
                )

            return subs

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def _build_index(self):
        index = []

        for line in self.subs:
            text = line.text.strip()

            if not text:
                continue

            start_sec = line.start / 1000.0
            index.append((start_sec, text))

        return sorted(
            index,
            key=lambda item: item[0],
        )

    def get_by_time(self, time_sec):
        """
        Получить субтитр по-точному PTS кадра.
        """

        pos = bisect_right(
            self.timestamps,
            time_sec,
        ) - 1

        if pos < 0:
            return ""

        return self.texts[pos]

    def write_interval_srt(
        self,
        start_sec,
        end_sec,
        output_path,
    ):
        """
        Создаёт SRT для отдельного видеофрагмента.

        Временная шкала смещается так, чтобы начало клипа
        соответствовало 00:00:00.
        """

        start_ms = round(start_sec * 1000)
        end_ms = round(end_sec * 1000)

        target = pysubs2.SSAFile()

        for line in self.subs:
            text = line.text.strip()

            if not text:
                continue

            # Субтитр полностью находится до фрагмента.
            if line.end <= start_ms:
                continue

            # Последующие субтитры уже находятся после фрагмента.
            if line.start >= end_ms:
                break

            shifted_start = (
                max(line.start, start_ms)
                - start_ms
            )

            shifted_end = (
                min(line.end, end_ms)
                - start_ms
            )

            if shifted_end <= shifted_start:
                continue

            target.append(
                pysubs2.SSAEvent(
                    start=shifted_start,
                    end=shifted_end,
                    text=text,
                )
            )

        if not target:
            raise RuntimeError(
                "В интервале отсутствуют субтитры: "
                f"{start_sec:.3f}–{end_sec:.3f}"
            )

        target.save(
            str(output_path),
            format_="srt",
            encoding="utf-8",
        )