from datetime import datetime, timedelta, timezone
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import requests
import logging
import os
import time

logger = logging.getLogger(__name__)

def get_yesterday_interval_utc(start_hour=5, start_minute=30, end_hour=19, end_minute=30):
    now = datetime.now(timezone.utc)

    yesterday_date = (now - timedelta(days=1)).date()

    start_dt = datetime(
        yesterday_date.year,
        yesterday_date.month,
        yesterday_date.day,
        start_hour,
        start_minute,
        tzinfo=timezone.utc
    )

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
        email,
        password,
        client_id,
        proxy_key,
        camera_id,
        archive,
        start,
        end,
        out_dir=".",
        export_format="mkv",
        waittimeout=30000,
        poll_interval=2,
        delete_after_download=True,
        verify_ssl=False
):
    if not proxy_key:
        raise ValueError("proxy_key is required")

    session = requests.Session()

    retry = Retry(total=3,
                  backoff_factor=1,
                  status_forcelist=[500, 502, 503, 504],
                  allowed_methods=["GET", "POST"]
                  )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    logger.info("=== EXPORT STEP STARTED ===")

    base_api = "https://video-cloud.wb.ru"
    login_url = f"{base_api}/api/v3/ac-backend/users/login"

    auth = session.post(
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
    access_token = auth.json()["accessToken"]

    logger.info("Login OK")

    base = f"{base_api}/arpserver/{proxy_key}/webclient"
    export_url = f"{base}/export/archive/{camera_id}/{start}/{end}"

    r = session.post(
        export_url,
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

    if r.status_code != 202:
        logger.error(
            f"Export failed {r.status_code}: {r.text}"
        )
        raise RuntimeError(f"Export start failed:\n{r.status_code}\n{r.text}")

    export_id = r.headers["location"].split("/")[-1]
    logger.info(f"Export started id={export_id}")

    status_url = f"{base}/export/{export_id}/status"

    logger.info("Starting export job...")

    while True:
        s = session.get(
            status_url,
            params={"authToken": access_token},
            verify=verify_ssl,
            timeout=30
        )

        data = s.json()

        state = data["state"]
        progress = data.get("progress", 0)

        logger.info(
            f"Export state={state} progress={progress * 100:.1f}%"
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

    download_url = f"{base}/export/{export_id}/file"

    os.makedirs(out_dir, exist_ok=True)
    filepath = os.path.join(out_dir, filename)

    with session.get(
            download_url,
            params={
                "name": filename,
                "authToken": access_token
            },
            stream=True,
            verify=verify_ssl,
            timeout=300
    ) as dl:

        dl.raise_for_status()

        with open(filepath, "wb") as f:
            for chunk in dl.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    logger.info(f"✓ downloaded:{filepath}")

    if delete_after_download:
        session.delete(
            f"{base}/export/{export_id}",
            params={"authToken": access_token},
            verify=verify_ssl,
            timeout=30
        )
        logger.info("✓ export deleted")

    return filepath