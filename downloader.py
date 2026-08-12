from datetime import datetime, timedelta, timezone
from requests.adapters import HTTPAdapter
from urllib3 import disable_warnings
from urllib3.exceptions import InsecureRequestWarning
from urllib3.util.retry import Retry
import requests
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from urllib.parse import urljoin, urlsplit
from credentials import (
    EMAIL,
    PASSWORD,
    CLIENT_ID,
    BASE_API,
)


logger = logging.getLogger(__name__)


def _format_bytes(byte_count):
    value = float(byte_count)
    units = ("B", "KiB", "MiB", "GiB", "TiB")

    for unit in units:
        if abs(value) < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024


def _format_duration(seconds):
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _download_stream_with_progress(
        response,
        filepath,
        filename,
        progress_interval=5,
        chunk_size=1024 * 1024
):
    content_length = response.headers.get("Content-Length")
    try:
        total_size = int(content_length) if content_length else None
    except (TypeError, ValueError):
        total_size = None

    if total_size is not None and total_size <= 0:
        total_size = None

    started_at = time.monotonic()
    last_report_at = started_at
    downloaded = 0

    size_label = _format_bytes(total_size) if total_size else "unknown size"
    logger.info("Download started [%s]: %s", filename, size_label)

    with open(filepath, "wb") as file:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if not chunk:
                continue

            file.write(chunk)
            downloaded += len(chunk)

            now = time.monotonic()
            if now - last_report_at < progress_interval:
                continue

            elapsed = max(now - started_at, 0.001)
            bytes_per_second = downloaded / elapsed

            if total_size:
                percent = min(downloaded / total_size * 100, 100)
                remaining = max(total_size - downloaded, 0)
                eta = (
                    remaining / bytes_per_second
                    if bytes_per_second > 0
                    else 0
                )
                logger.info(
                    "Download progress [%s]: %.1f%% (%s/%s), %s/s, ETA %s",
                    filename,
                    percent,
                    _format_bytes(downloaded),
                    _format_bytes(total_size),
                    _format_bytes(bytes_per_second),
                    _format_duration(eta),
                )
            else:
                logger.info(
                    "Download progress [%s]: %s, %s/s, elapsed %s",
                    filename,
                    _format_bytes(downloaded),
                    _format_bytes(bytes_per_second),
                    _format_duration(elapsed),
                )

            last_report_at = now

    elapsed = max(time.monotonic() - started_at, 0.001)
    logger.info(
        "Download finished [%s]: %s in %s, average %s/s",
        filename,
        _format_bytes(downloaded),
        _format_duration(elapsed),
        _format_bytes(downloaded / elapsed),
    )

    return downloaded


def _request(session, method, url, **kwargs):
    try:
        return getattr(session, method)(url, **kwargs)
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as error:
        host = urlsplit(url).netloc or url
        raise RuntimeError(
            f"Cannot connect to {host}. Check that the corporate VPN is "
            "enabled and BASE_API is reachable."
        ) from error


def _authenticate(session, base_api, email, password, client_id, verify_ssl):
    login_url = f"{base_api}/api/v3/ac-backend/users/login"
    auth = _request(
        session,
        "post",
        login_url,
        json={
            "email": email,
            "password": password,
            "clientId": client_id
        },
        verify=verify_ssl,
        timeout=30
    )

    logger.info("Authenticating...")
    auth.raise_for_status()

    access_token = auth.json().get("accessToken")
    if not access_token:
        raise RuntimeError("Authentication response does not contain accessToken")

    logger.info("Login OK")
    return access_token


def _get_camera_domain_id(cameras, camera_id):
    def normalise_access_point(value):
        if not isinstance(value, str):
            return None
        access_point = value.strip().strip("/").casefold()
        if access_point.startswith("hosts/"):
            access_point = access_point[len("hosts/"):]
        return access_point

    target = normalise_access_point(camera_id)
    camera_entries = [
        item for item in cameras
        if isinstance(item, dict)
        and normalise_access_point(item.get("accessPoint"))
    ]

    exact_camera = next(
        (
            item for item in camera_entries
            if normalise_access_point(item["accessPoint"]) == target
        ),
        None
    )
    if exact_camera is not None:
        domain_id = exact_camera.get("domainId")
        if domain_id is None:
            raise RuntimeError(
                f"Camera {camera_id!r} does not contain domainId"
            )
        return domain_id

    def unique_domain_id(predicate):
        domain_ids = {
            item.get("domainId") for item in camera_entries
            if predicate(normalise_access_point(item["accessPoint"]))
            and item.get("domainId") is not None
        }
        return next(iter(domain_ids)) if len(domain_ids) == 1 else None

    target_device = target.rsplit("/", 1)[0] if "/" in target else target
    domain_id = unique_domain_id(
        lambda access_point: access_point.rsplit("/", 1)[0] == target_device
    )
    if domain_id is not None:
        logger.warning(
            "Exact camera is absent in configsync; domain resolved by device"
        )
        return domain_id

    target_server = target.split("/", 1)[0]
    domain_id = unique_domain_id(
        lambda access_point: access_point.split("/", 1)[0] == target_server
    )
    if domain_id is not None:
        logger.warning(
            "Exact camera is absent in configsync; domain resolved by server"
        )
        return domain_id

    raise RuntimeError(
        f"Camera {camera_id!r} was not found and its domain could not be "
        f"determined from {len(camera_entries)} configsync cameras"
    )


def _get_webclient_base(
        session,
        base_api,
        access_token,
        camera_id,
        verify_ssl
):
    headers = {"Authorization": f"Bearer {access_token}"}

    cameras_response = _request(
        session,
        "get",
        f"{base_api}/api/v1/configsync/cameras",
        headers=headers,
        verify=verify_ssl,
        timeout=30
    )
    cameras_response.raise_for_status()
    cameras = cameras_response.json()

    if not isinstance(cameras, list):
        raise RuntimeError("Unexpected response from configsync/cameras")

    domain_id = _get_camera_domain_id(cameras, camera_id)

    webclient_response = _request(
        session,
        "get",
        f"{base_api}/api/v3/ac-backend/public/domains/"
        f"{domain_id}/webclienturl",
        headers=headers,
        verify=verify_ssl,
        timeout=30
    )
    webclient_response.raise_for_status()
    public_url = webclient_response.json().get("publicURL")

    if not isinstance(public_url, str) or not public_url.strip():
        raise RuntimeError("webclienturl response does not contain publicURL")

    webclient_base = urljoin(f"{base_api}/", public_url.strip())
    parsed_url = urlsplit(webclient_base)
    path_parts = [part for part in parsed_url.path.split("/") if part]

    if (
            parsed_url.scheme not in ("http", "https")
            or len(path_parts) < 3
            or path_parts[-3] != "arpserver"
            or not path_parts[-2]
            or path_parts[-1] != "webclient"
            or parsed_url.query
            or parsed_url.fragment
    ):
        raise RuntimeError("AxxonNet returned an invalid web client URL")

    logger.info("AxxonNet web client route resolved")
    return webclient_base.rstrip("/")


def get_yesterday_interval_utc(start_hour=5, start_minute=30, end_hour=19, end_minute=30, cross_day=True):
    now = datetime.now(timezone.utc)

    yesterday_date = (now - timedelta(days=0)).date()

    start_dt = datetime(
        yesterday_date.year,
        yesterday_date.month,
        yesterday_date.day,
        start_hour,
        start_minute,
        tzinfo=timezone.utc
    )

    if cross_day:
        end_dt = start_dt + timedelta(days=1)
    else:
        end_dt = datetime(
            yesterday_date.year,
            yesterday_date.month,
            yesterday_date.day,
            end_hour,
            end_minute,
            tzinfo=timezone.utc
        )

    fmt = "%Y%m%dT%H%M%S.%f"
    return start_dt.strftime(fmt)[:-3], end_dt.strftime(fmt)[:-3]


def download_fragment(
        camera_id,
        archive,
        start,
        end,
        out_dir=".",
        export_format="mkv",
        waittimeout=30000,
        poll_interval=2,
        delete_after_download=True,
        verify_ssl=False,
        download_progress_interval=5
):
    if not verify_ssl:
        disable_warnings(InsecureRequestWarning)

    email = EMAIL
    password = PASSWORD
    client_id = CLIENT_ID
    base_api = BASE_API.rstrip("/")

    session = requests.Session()

    retry = Retry(total=3,
                  backoff_factor=1,
                  status_forcelist=[500, 502, 503, 504],
                  allowed_methods=["GET", "POST"]
                  )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    logger.info("[%s] === EXPORT STEP STARTED ===", camera_id)

    access_token = _authenticate(
        session,
        base_api,
        email,
        password,
        client_id,
        verify_ssl
    )
    base = _get_webclient_base(
        session,
        base_api,
        access_token,
        camera_id,
        verify_ssl
    )

    def start_export(webclient_base):
        return _request(
            session,
            "post",
            f"{webclient_base}/export/archive/{camera_id}/{start}/{end}",
            params={
                "archive": archive,
                "waittimeout": waittimeout,
                "authToken": access_token
            },
            json={
                "format": export_format,
                "comment": "",
                "tocloud": False
            },
            verify=verify_ssl,
            timeout=60
        )

    r = start_export(base)
    if r.status_code in (404, 502):
        r.close()
        logger.warning("[%s] AxxonNet route changed; resolving it again", camera_id)
        base = _get_webclient_base(
            session,
            base_api,
            access_token,
            camera_id,
            verify_ssl
        )
        r = start_export(base)

    if r.status_code != 202:
        logger.error(
            "[%s] Export failed %s: %s", camera_id, r.status_code, r.text
        )
        raise RuntimeError(f"Export start failed:\n{r.status_code}\n{r.text}")

    export_id = r.headers["location"].split("/")[-1]
    logger.info("[%s] Export started id=%s", camera_id, export_id)

    logger.info("[%s] Starting export job...", camera_id)

    consecutive_route_failures = 0
    while True:
        s = _request(
            session,
            "get",
            f"{base}/export/{export_id}/status",
            params={"authToken": access_token},
            verify=verify_ssl,
            timeout=30
        )

        if s.status_code in (404, 502):
            s.close()
            consecutive_route_failures += 1
            if consecutive_route_failures > 3:
                raise RuntimeError(
                    "AxxonNet route remains unavailable after 3 refreshes"
                )
            logger.warning("[%s] AxxonNet route changed; resolving it again", camera_id)
            base = _get_webclient_base(
                session,
                base_api,
                access_token,
                camera_id,
                verify_ssl
            )
            continue

        consecutive_route_failures = 0
        s.raise_for_status()

        data = s.json()

        state = data["state"]
        progress = data.get("progress", 0)

        logger.info(
            "[%s] Export state=%s progress=%.1f%%",
            camera_id,
            state,
            progress * 100,
        )

        if state == 2:
            break

        if state in (3, 4, 5, 6):
            raise RuntimeError(f"Export failed:\n{data}")

        time.sleep(poll_interval)

    files = data.get("files", [])
    if not files:
        raise RuntimeError(f"No files in export response: {data}")

    filename = files[0]

    os.makedirs(out_dir, exist_ok=True)
    filepath = os.path.join(out_dir, filename)

    dl = _request(
        session,
        "get",
        f"{base}/export/{export_id}/file",
        params={
            "name": filename,
            "authToken": access_token
        },
        stream=True,
        verify=verify_ssl,
        timeout=300
    )

    if dl.status_code in (404, 502):
        dl.close()
        logger.warning("[%s] AxxonNet route changed; resolving it again", camera_id)
        base = _get_webclient_base(
            session,
            base_api,
            access_token,
            camera_id,
            verify_ssl
        )
        dl = _request(
            session,
            "get",
            f"{base}/export/{export_id}/file",
            params={
                "name": filename,
                "authToken": access_token
            },
            stream=True,
            verify=verify_ssl,
            timeout=300
        )

    with dl:

        dl.raise_for_status()

        _download_stream_with_progress(
            response=dl,
            filepath=filepath,
            filename=filename,
            progress_interval=download_progress_interval,
        )

    logger.info("[%s] ✓ downloaded: %s", camera_id, filepath)

    if delete_after_download:
        _request(
            session,
            "delete",
            f"{base}/export/{export_id}",
            params={"authToken": access_token},
            verify=verify_ssl,
            timeout=30
        )
        logger.info("[%s] ✓ export deleted", camera_id)

    return filepath


def _parse_export_timestamp(value):
    for fmt in ("%Y%m%dT%H%M%S.%f", "%Y%m%dT%H%M%S"):
        try:
            return datetime.strptime(value, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise ValueError(f"Invalid export timestamp: {value}")


def _format_export_timestamp(value):
    return value.strftime("%Y%m%dT%H%M%S.%f")[:-3]


def split_export_interval(start, end, fragment_hours=1):
    if fragment_hours <= 0:
        raise ValueError("fragment_hours must be greater than zero")

    start_dt = _parse_export_timestamp(start)
    end_dt = _parse_export_timestamp(end)
    if end_dt <= start_dt:
        raise ValueError("Export end must be later than export start")

    intervals = []
    cursor = start_dt
    step = timedelta(hours=fragment_hours)
    while cursor < end_dt:
        fragment_end = min(cursor + step, end_dt)
        intervals.append((
            _format_export_timestamp(cursor),
            _format_export_timestamp(fragment_end),
        ))
        cursor = fragment_end

    return intervals


def _resolve_ffmpeg_executable():
    configured_path = os.getenv("FFMPEG_PATH")
    if configured_path:
        configured = Path(configured_path)
        if configured.is_file():
            return str(configured)

        resolved = shutil.which(configured_path)
        if resolved:
            return resolved

        raise FileNotFoundError(
            f"FFmpeg from FFMPEG_PATH was not found: {configured_path}"
        )

    resolved = shutil.which("ffmpeg")
    if resolved:
        return resolved

    executable_name = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    beside_python = Path(sys.executable).with_name(executable_name)
    if beside_python.is_file():
        return str(beside_python)

    raise FileNotFoundError(
        "FFmpeg was not found. Add it to PATH or set FFMPEG_PATH."
    )


def _escape_concat_path(path):
    return str(Path(path).resolve()).replace("\\", "/").replace(
        "'", r"'\''"
    )


def merge_video_fragments(fragment_paths, output_path, ffmpeg_path=None):
    if not fragment_paths:
        raise ValueError("No video fragments to merge")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output_path.with_name(
        f"{output_path.stem}.merging{output_path.suffix}"
    )
    concat_path = Path(fragment_paths[0]).parent / "concat.txt"

    with concat_path.open("w", encoding="utf-8") as concat_file:
        for fragment_path in fragment_paths:
            concat_file.write(
                f"file '{_escape_concat_path(fragment_path)}'\n"
            )

    command = [
        ffmpeg_path or _resolve_ffmpeg_executable(),
        "-y",
        "-hide_banner",
        "-v", "error",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_path),
        "-map", "0",
        "-c", "copy",
        str(temporary_output),
    ]

    logger.info(
        "Merging %d hourly fragments into %s",
        len(fragment_paths),
        output_path,
    )
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
        )
        if result.returncode != 0:
            raise RuntimeError(
                "FFmpeg could not merge video fragments:\n"
                + result.stderr.strip()
            )

        if not temporary_output.is_file() or temporary_output.stat().st_size == 0:
            raise RuntimeError(
                f"FFmpeg did not create the merged video: {temporary_output}"
            )

        os.replace(temporary_output, output_path)
    except Exception:
        temporary_output.unlink(missing_ok=True)
        raise

    logger.info("Hourly fragments merged: %s", output_path)
    return str(output_path)


def download_fragmented_video(
        camera_id,
        archive,
        start,
        end,
        out_dir=".",
        fragment_hours=1,
        export_format="mkv",
):
    ffmpeg_path = _resolve_ffmpeg_executable()
    intervals = split_export_interval(start, end, fragment_hours)
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    parts_dir = Path(tempfile.mkdtemp(
        prefix=".video-parts-",
        dir=output_dir,
    ))
    fragment_paths = []

    logger.info(
        "[%s] Split download enabled: %d fragments of up to %s hour(s)",
        camera_id,
        len(intervals),
        fragment_hours,
    )

    try:
        for position, (fragment_start, fragment_end) in enumerate(
                intervals,
                start=1,
        ):
            logger.info(
                "[%s] Downloading fragment %d/%d: %s - %s",
                camera_id,
                position,
                len(intervals),
                fragment_start,
                fragment_end,
            )
            fragment_paths.append(download_fragment(
                camera_id=camera_id,
                archive=archive,
                start=fragment_start,
                end=fragment_end,
                out_dir=str(parts_dir),
                export_format=export_format,
            ))

        first_name = Path(fragment_paths[0]).name
        filename_prefix = first_name.split("[", 1)[0]
        extension = Path(first_name).suffix or f".{export_format}"
        filename_start = _parse_export_timestamp(start).strftime(
            "%Y%m%dT%H%M%S"
        )
        filename_end = _parse_export_timestamp(end).strftime(
            "%Y%m%dT%H%M%S"
        )
        output_path = output_dir / (
            f"{filename_prefix}[{filename_start}-{filename_end}]{extension}"
        )

        merged_path = merge_video_fragments(
            fragment_paths,
            output_path,
            ffmpeg_path=ffmpeg_path,
        )
    except Exception:
        logger.exception(
            "[%s] Split download failed; downloaded parts remain in %s",
            camera_id,
            parts_dir,
        )
        raise

    try:
        shutil.rmtree(parts_dir)
    except OSError as error:
        logger.warning(
            "[%s] Merged video is ready, but temporary fragments could not "
            "be deleted from %s: %s",
            camera_id,
            parts_dir,
            error,
        )
    else:
        logger.info("[%s] Temporary hourly fragments deleted", camera_id)
    return merged_path
