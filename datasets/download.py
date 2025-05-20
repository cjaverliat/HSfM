from tqdm import tqdm
import requests
import os
from typing import Optional
import warnings

import gdown
from gdown.download_folder import GoogleDriveFileToDownload

from file import compute_md5_checksum


def get_file_size(session: requests.Session, file_url: str) -> int:
    head = session.head(file_url, allow_redirects=True)

    if head.status_code != 200:
        raise Exception(
            f"Failed to get headers for {file_url}: {head.status_code} {head.reason}"
        )

    file_size = head.headers.get("Content-Length", None)
    return int(file_size) if file_size is not None else None

def list_files_in_google_drive_folder(folder_id: str) -> list[GoogleDriveFileToDownload]:
    return gdown.download_folder(id=folder_id, skip_download=True)

def download_google_drive_folder(
    folder_id: str,
    output_dir: str,
    resume: bool = True,
    use_cookies: bool = True,
    verbose: bool = False,
    verify: bool = True,
):
    gdown.download_folder(
        id=folder_id,
        use_cookies=use_cookies,
        output=output_dir,
        quiet=not verbose,
        resume=resume,
        verify=verify,
    )


def download_google_drive_file(
    file_id: str,
    output_file: str,
    verbose: bool = False,
):
    gdown.download(
        url=f"https://drive.google.com/file/d/{file_id}/view?usp=drive_link",
        output=output_file,
        quiet=not verbose,
        fuzzy=True,
    )


def download_file(
    session: requests.Session,
    file_url: str,
    output_file: str,
    md5_checksum: Optional[str] = None,
    use_remote_file_size: bool = True,
    verbose: bool = False,
    force_download: bool = False,
):
    remote_file_size = None
    if use_remote_file_size:
        try:
            remote_file_size = get_file_size(session, file_url)
        except Exception as e:
            warnings.warn(f"Unable to get the file size: {e}")

    local_file_exists = os.path.exists(output_file)
    local_file_size = os.path.getsize(output_file) if local_file_exists else None

    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)

    if (
        remote_file_size is not None
        and local_file_size is not None
        and local_file_size == remote_file_size
        and not force_download
    ):
        # Ensure checksum is correct
        if md5_checksum is not None:
            with open(output_file, "rb") as f:
                file_checksum = compute_md5_checksum(f)
            if file_checksum != md5_checksum:
                # File already downloaded but checksum mismatch. Redownloading.
                local_file_size = None
                if verbose:
                    print(
                        f"Checksum mismatch for {output_file}. Expected {md5_checksum}, got {file_checksum}. Redownloading."
                    )
            else:
                # File already downloaded. Skipping.
                if verbose:
                    print(
                        f"Skipping {output_file} because it already exists and checksum is correct."
                    )
                return
        else:
            # File already downloaded. Skipping.
            if verbose:
                print(f"Skipping {output_file} because it is already downloaded.")
            return

    if local_file_size is None:
        local_file_size = 0

    partial_download = remote_file_size is not None and not force_download

    if partial_download:
        download_response = session.get(
            file_url,
            headers={"Range": f"bytes={local_file_size}-"},
            stream=True,
            allow_redirects=True,
        )
    else:
        download_response = session.get(file_url, stream=True, allow_redirects=True)
    if download_response.status_code != 200 and download_response.status_code != 206:
        raise Exception(
            f"Failed to download {file_url}: {download_response.status_code} {download_response.reason}"
        )

    if remote_file_size is not None:
        if download_response.status_code == 206:
            # Partial content response
            total_file_size = remote_file_size + local_file_size
        else:
            total_file_size = remote_file_size
    else:
        total_file_size = -1

    mode = "ab" if partial_download else "wb"
    with open(output_file, mode) as f:
        pbar = tqdm(
            total=total_file_size,
            initial=local_file_size,
            desc=f"Downloading {file_url}",
            unit="B",
            unit_scale=True,
            leave=False,
        )
        for chunk in download_response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))
        pbar.close()

    if remote_file_size is not None:
        local_file_size = (
            os.path.getsize(output_file) if os.path.exists(output_file) else None
        )

        if local_file_size != remote_file_size:
            raise Exception(
                f"File size mismatch for {output_file}. Expected {remote_file_size}B, got {local_file_size}B"
            )

    # Ensure checksum is correct
    if md5_checksum is not None:
        with open(output_file, "rb") as f:
            file_checksum = compute_md5_checksum(f)
        if file_checksum != md5_checksum:
            raise Exception(
                f"Checksum mismatch for {output_file}. Expected {md5_checksum}, got {file_checksum}"
            )
