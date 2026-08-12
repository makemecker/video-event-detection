"""Check HTTP byte-range support using an export created in one session.

The script creates an export, waits until it is ready, reads two 1 KiB byte
ranges without saving the video, and deletes the diagnostic export afterward.
"""

import argparse
import importlib
import re
import time

import requests
from requests.adapters import HTTPAdapter
from urllib3 import disable_warnings
from urllib3.exceptions import InsecureRequestWarning
from urllib3.util.retry import Retry

from credentials import BASE_API, CLIENT_ID, EMAIL, PASSWORD
from downloader import (
    _authenticate,
    _get_webclient_base,
    _request,
    get_yesterday_interval_utc,
)


SAMPLE_SIZE = 1024
SECOND_SAMPLE_OFFSET = 1024 ** 3 + 16 * 1024 ** 2
CONTENT_RANGE_PATTERN = re.compile(r"^bytes (\d+)-(\d+)/(\d+|\*)$")


def get_camera_config(camera_id):
    try:
        module = importlib.import_module(f"configs.camera_{camera_id}")
    except ModuleNotFoundError as error:
        raise ValueError(f"Unknown camera: {camera_id}") from error
    return module.CAMERA_CONFIG


def create_session():
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["GET", "POST"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def create_export(
        session,
        base,
        token,
        config,
        start,
        end,
        verify_ssl,
):
    response = _request(
        session,
        "post",
        f"{base}/export/archive/{config['camera_id']}/{start}/{end}",
        params={
            "archive": config["archive"],
            "waittimeout": 30000,
            "authToken": token,
        },
        json={"format": "mkv", "comment": "", "tocloud": False},
        verify=verify_ssl,
        timeout=60,
    )
    if response.status_code != 202:
        raise RuntimeError(
            f"Export start failed: {response.status_code} {response.text}"
        )

    export_id = response.headers["location"].split("/")[-1]
    print(f"Diagnostic export started: {export_id}")
    return export_id


def wait_for_export(session, base, token, export_id, verify_ssl):
    while True:
        response = _request(
            session,
            "get",
            f"{base}/export/{export_id}/status",
            params={"authToken": token},
            verify=verify_ssl,
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        state = data["state"]
        progress = data.get("progress", 0) * 100
        print(f"Export state={state} progress={progress:.1f}%")

        if state == 2:
            files = data.get("files", [])
            if not files:
                raise RuntimeError("Export completed without files")
            return files[0]
        if state in (3, 4, 5, 6):
            raise RuntimeError(f"Export failed: {data}")

        time.sleep(2)


def check_range(session, url, params, start, verify_ssl):
    end = start + SAMPLE_SIZE - 1
    response = session.get(
        url,
        params=params,
        headers={"Range": f"bytes={start}-{end}"},
        stream=True,
        verify=verify_ssl,
        timeout=(30, 30),
    )

    try:
        content_range = response.headers.get("Content-Range")
        print(f"Range requested: bytes={start}-{end}")
        print(f"Status:          {response.status_code}")
        print(f"Accept-Ranges:   {response.headers.get('Accept-Ranges', '<absent>')}")
        print(f"Content-Range:   {content_range or '<absent>'}")
        print(f"Content-Length:  {response.headers.get('Content-Length', '<absent>')}")

        if response.status_code != 206:
            print("Result:          NOT SUPPORTED (expected HTTP 206)")
            return False

        match = CONTENT_RANGE_PATTERN.fullmatch(content_range or "")
        if not match or int(match.group(1)) != start or int(match.group(2)) != end:
            print("Result:          INVALID Content-Range")
            return False

        sample = response.raw.read(SAMPLE_SIZE)
        if len(sample) != SAMPLE_SIZE:
            print(f"Result:          SHORT BODY ({len(sample)} bytes)")
            return False

        print("Result:          RANGE SUPPORTED")
        return True
    finally:
        response.close()


def delete_export(session, base, token, export_id, verify_ssl):
    response = _request(
        session,
        "delete",
        f"{base}/export/{export_id}",
        params={"authToken": token},
        verify=verify_ssl,
        timeout=30,
    )
    if response.status_code >= 400:
        print(
            f"Warning: diagnostic export was not deleted "
            f"(HTTP {response.status_code})"
        )
    else:
        print("Diagnostic export deleted.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check HTTP Range support for the Axxon video endpoint"
    )
    parser.add_argument("camera", help="Short camera ID, for example 4")
    parser.add_argument(
        "--verify-ssl",
        action="store_true",
        help="Enable TLS certificate verification",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = get_camera_config(args.camera)
    verify_ssl = args.verify_ssl

    if not verify_ssl:
        disable_warnings(InsecureRequestWarning)

    session = create_session()
    base_api = BASE_API.rstrip("/")
    token = _authenticate(
        session,
        base_api,
        EMAIL,
        PASSWORD,
        CLIENT_ID,
        verify_ssl,
    )
    base = _get_webclient_base(
        session,
        base_api,
        token,
        config["camera_id"],
        verify_ssl,
    )
    start, end = get_yesterday_interval_utc()
    export_id = None

    try:
        export_id = create_export(
            session,
            base,
            token,
            config,
            start,
            end,
            verify_ssl,
        )
        filename = wait_for_export(
            session,
            base,
            token,
            export_id,
            verify_ssl,
        )
        print(f"Export file: {filename}\n")

        url = f"{base}/export/{export_id}/file"
        params = {"name": filename, "authToken": token}

        print("Checking a fragment at the beginning of the file...")
        first_ok = check_range(session, url, params, 0, verify_ssl)
        print()
        print("Checking a fragment after the observed 1 GiB cutoff...")
        second_ok = check_range(
            session,
            url,
            params,
            SECOND_SAMPLE_OFFSET,
            verify_ssl,
        )

        if first_ok and second_ok:
            print("\nConclusion: byte-range resume can be implemented.")
            return 0

        print("\nConclusion: this endpoint does not support byte-range resume.")
        return 1
    finally:
        if export_id is not None:
            delete_export(session, base, token, export_id, verify_ssl)


if __name__ == "__main__":
    raise SystemExit(main())
