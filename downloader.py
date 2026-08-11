from datetime import datetime, timedelta, timezone
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import requests
import logging
import os
import time
from urllib.parse import urljoin, urlsplit
from credentials import (
    EMAIL,
    PASSWORD,
    CLIENT_ID,
    BASE_API,
)


logger = logging.getLogger(__name__)


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
        return value.strip().strip("/").casefold()

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


def get_yesterday_interval_utc(start_hour=5, start_minute=30, end_hour=19, end_minute=30, cross_day=False):
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
        verify_ssl=False
):
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

    logger.info("=== EXPORT STEP STARTED ===")

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
        logger.warning("AxxonNet route changed; resolving it again")
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
            f"Export failed {r.status_code}: {r.text}"
        )
        raise RuntimeError(f"Export start failed:\n{r.status_code}\n{r.text}")

    export_id = r.headers["location"].split("/")[-1]
    logger.info(f"Export started id={export_id}")

    logger.info("Starting export job...")

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
            logger.warning("AxxonNet route changed; resolving it again")
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
        logger.warning("AxxonNet route changed; resolving it again")
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

        with open(filepath, "wb") as f:
            for chunk in dl.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    logger.info(f"✓ downloaded:{filepath}")

    if delete_after_download:
        _request(
            session,
            "delete",
            f"{base}/export/{export_id}",
            params={"authToken": access_token},
            verify=verify_ssl,
            timeout=30
        )
        logger.info("✓ export deleted")

    return filepath
