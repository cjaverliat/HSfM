import tarfile
from zipfile import ZipFile
import hashlib
import os
from typing import BinaryIO, Union
from tqdm import tqdm


def compute_md5_checksum(file: Union[BinaryIO, str]) -> str:

    should_close = False

    if isinstance(file, str):
        file = open(file, "rb")
        should_close = True

    file_size = os.fstat(file.fileno()).st_size

    hash_md5 = hashlib.md5()

    pbar = tqdm(
        desc="Computing MD5 checksum",
        total=file_size,
        unit="B",
        unit_scale=True,
        leave=False,
    )

    for chunk in iter(lambda: file.read(4096), b""):
        pbar.update(len(chunk))
        hash_md5.update(chunk)

    pbar.close()

    if should_close:
        file.close()

    return hash_md5.hexdigest()


def load_checksums(checksums_filepath: str) -> dict[str, str]:
    checksums = {}
    with open(checksums_filepath, "r") as f:
        for line in f.read().splitlines(keepends=False):
            v, k = line.split(" ")
            checksums[k] = v
    return checksums


def extract_file_from_zip(zip_file: str, file: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    with ZipFile(zip_file, "r") as f:
        f.extract(file, output_dir)


def _track_progress(tar: tarfile.TarFile, pbar: tqdm):
    members = tar.getmembers()
    pbar.total = len(members)
    for member in members:
        yield member
        pbar.update(1)
    pbar.close()


def extract_tar_with_progress(
    file: str, output_dir: str, desc: str = "Extracting", leave: bool = True
):
    if not os.path.exists(file):
        raise FileNotFoundError(f"File {file} not found")

    os.makedirs(output_dir, exist_ok=True)
    pbar = tqdm(desc=desc, leave=leave)
    pbar.update(0)
    with tarfile.open(file, "r") as tar:
        tar.extractall(output_dir, members=_track_progress(tar, pbar))
