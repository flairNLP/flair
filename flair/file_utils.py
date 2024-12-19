"""Utilities for working with the local dataset cache. Copied from AllenNLP."""

import base64
import functools
import io
import logging
import mmap
import os
import re
import shutil
import tempfile
import typing
import warnings
import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import Optional, Union, cast
from urllib.parse import urlparse

import boto3
import requests
import torch
from botocore import UNSIGNED
from botocore.config import Config
from requests import HTTPError
from tqdm import tqdm as _tqdm

import flair

logger = logging.getLogger("flair")

url_proxies: Optional[dict[str, str]] = None


def set_proxies(proxies: dict[str, str]) -> None:
    r"""Allows for data downloaded from urls to be forwarded to a proxy.

    see https://requests.readthedocs.io/en/latest/user/advanced/#proxies

    Args:
        proxies: A dictionary of proxies according to the requests documentation.
    """
    global url_proxies
    url_proxies = proxies


def load_big_file(f: str):
    """Workaround for loading a big pickle file.

    Files over 2GB cause pickle errors on certain Mac and Windows distributions.
    """
    with open(f, "rb") as f_in:
        # mmap seems to be much more memory efficient
        bf = mmap.mmap(f_in.fileno(), 0, access=mmap.ACCESS_READ)
        f_in.close()
    return bf


def url_to_filename(url: str, etag: Optional[str] = None) -> str:
    """Converts an url into a filename in a reversible way.

    If `etag` is specified, add it on the end, separated by a period
    (which necessarily won't appear in the base64-encoded filename).
    Get rid of the quotes in the etag, since Windows doesn't like them.
    """
    url_bytes = url.encode("utf-8")
    b64_bytes = base64.b64encode(url_bytes)
    decoded = b64_bytes.decode("utf-8")

    if etag:
        # Remove quotes from etag
        etag = etag.replace('"', "")
        return f"{decoded}.{etag}"
    else:
        return decoded


def filename_to_url(filename: str) -> tuple[str, Optional[str]]:
    """Recovers the the url from the encoded filename.

    Returns it and the ETag (which may be ``None``)
    """
    etag: Optional[str]
    try:
        # If there is an etag, it's everything after the first period
        decoded, etag = filename.split(".", 1)
    except ValueError:
        # Otherwise, use None
        decoded, etag = filename, None

    filename_bytes = decoded.encode("utf-8")
    url_bytes = base64.b64decode(filename_bytes)
    return url_bytes.decode("utf-8"), etag


def cached_path(url_or_filename: str, cache_dir: Union[str, Path]) -> Path:
    """Download the given path and return the local path from the cache.

    Given something that might be a URL (or might be a local path),
    determine which. If it's a URL, download the file and cache it, and
    return the path to the cached file. If it's already a local path,
    make sure the file exists and then return the path.
    """
    cache_dir = Path(cache_dir)

    dataset_cache = flair.cache_root / cache_dir if flair.cache_root not in cache_dir.parents else cache_dir

    parsed = urlparse(url_or_filename)

    if parsed.scheme in ("http", "https"):
        # URL, so get it from the cache (downloading if necessary)
        return get_from_cache(url_or_filename, dataset_cache)
    elif parsed.scheme == "s3":
        return download_s3_to_path(parsed.netloc, dataset_cache)
    elif len(parsed.scheme) < 2 and Path(url_or_filename).exists():
        # File, and it exists.
        return Path(url_or_filename)
    elif len(parsed.scheme) < 2:
        # File, but it doesn't exist.
        raise FileNotFoundError(f"file {url_or_filename} not found")
    else:
        # Something unknown
        raise ValueError(f"unable to parse {url_or_filename} as a URL or as a local path")


def download_s3_to_path(bucket_name: str, cache_path: Path) -> Path:
    out_path = cache_path / bucket_name
    if out_path.exists():
        return out_path
    s3 = boto3.resource("s3", config=Config(signature_version=UNSIGNED))
    bucket = s3.Bucket(bucket_name)
    for obj in bucket.objects.iterator():
        if obj.key[-1] == "/":
            continue
        target = out_path / obj.key
        target.parent.mkdir(exist_ok=True, parents=True)
        bucket.download_file(obj.key, str(target))
    return out_path


def unzip_file(file: Union[str, Path], unzip_to: Union[str, Path]):
    from zipfile import ZipFile

    with ZipFile(Path(file), "r") as zipObj:
        # Extract all the contents of zip file in current directory
        zipObj.extractall(Path(unzip_to))


def hf_download(model_name: str) -> str:
    hf_model_name = "pytorch_model.bin"
    revision = "main"

    if "@" in model_name:
        model_name_split = model_name.split("@")
        revision = model_name_split[-1]
        model_name = model_name_split[0]

    # use model name as subfolder
    model_folder = model_name.split("/", maxsplit=1)[1] if "/" in model_name else model_name

    # Lazy import
    from huggingface_hub.file_download import hf_hub_download

    try:
        return hf_hub_download(
            repo_id=model_name,
            filename=hf_model_name,
            revision=revision,
            library_name="flair",
            library_version=flair.__version__,
            cache_dir=flair.cache_root / "models" / model_folder,
        )
    except HTTPError:
        # output information
        Path(flair.cache_root / "models" / model_folder).rmdir()  # remove folder again if not valid
        raise


def unpack_file(file: Path, unpack_to: Path, mode: Optional[str] = None, keep: bool = True):
    """Unpacks an archive file to the given location.

    Args:
        file: Archive file to unpack
        unpack_to: Destination where to store the output
        mode: Type of the archive (zip, tar, gz, targz, rar)
        keep: Indicates whether to keep the archive after extraction or delete it
    """
    if mode == "zip" or (mode is None and str(file).endswith("zip")):
        from zipfile import ZipFile

        with ZipFile(file, "r") as zipObj:
            # Extract all the contents of zip file in current directory
            zipObj.extractall(unpack_to)

    elif mode == "targz" or (mode is None and str(file).endswith("tar.gz") or str(file).endswith("tgz")):
        import tarfile

        with tarfile.open(file, "r:gz") as tarObj:
            tarObj.extractall(unpack_to)

    elif mode == "tar" or (mode is None and str(file).endswith("tar")):
        import tarfile

        with tarfile.open(file, "r") as tarObj:
            tarObj.extractall(unpack_to)

    elif mode == "gz" or (mode is None and str(file).endswith("gz")):
        import gzip

        with gzip.open(str(file), "rb") as f_in, open(str(unpack_to), "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    elif mode == "rar" or (mode is None and str(file).endswith("rar")):
        import patoolib

        patoolib.extract_archive(str(file), outdir=unpack_to, interactive=False)

    else:
        if mode is None:
            raise AssertionError(f"Can't infer archive type from {file}")
        else:
            raise AssertionError(f"Unsupported mode {mode}")

    if not keep:
        os.remove(str(file))


# TODO(joelgrus): do we want to do checksums or anything like that?
def get_from_cache(url: str, cache_dir: Path) -> Path:
    """Given a URL, look for the corresponding file in the local cache or download it.

    return: the path to the cached file.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    filename = re.sub(r".+/", "", url)
    # get cache path to put the file
    cache_path = cache_dir / filename
    if cache_path.exists():
        return cache_path

    # make HEAD request to check ETag
    response = requests.head(url, headers={"User-Agent": "Flair"}, allow_redirects=True, proxies=url_proxies)
    if response.status_code != 200:
        raise OSError(f"HEAD request failed for url {url} with status code {response.status_code}.")

    # add ETag to filename if it exists
    # etag = response.headers.get("ETag")

    if not cache_path.exists():
        # Download to temporary file, then copy to cache dir once finished.
        # Otherwise you get corrupt cache entries if the download gets interrupted.
        fd, temp_filename = tempfile.mkstemp()
        logger.info("%s not found in cache, downloading to %s", url, temp_filename)

        # GET file object
        req = requests.get(url, stream=True, headers={"User-Agent": "Flair"}, proxies=url_proxies)
        content_length = req.headers.get("Content-Length")
        total = int(content_length) if content_length is not None else None
        progress = Tqdm.tqdm(unit="B", total=total, unit_scale=True, unit_divisor=1024)
        with open(temp_filename, "wb") as temp_file:
            for chunk in req.iter_content(chunk_size=1024):
                if chunk:  # filter out keep-alive new chunks
                    progress.update(len(chunk))
                    temp_file.write(chunk)

        progress.close()

        logger.info("copying %s to cache at %s", temp_filename, cache_path)
        shutil.copyfile(temp_filename, str(cache_path))
        logger.info("removing temp file %s", temp_filename)
        os.close(fd)
        os.remove(temp_filename)

    return cache_path


def open_inside_zip(
    archive_path: str,
    cache_dir: Union[str, Path],
    member_path: Optional[str] = None,
    encoding: str = "utf8",
) -> typing.Iterable:
    cached_archive_path = cached_path(archive_path, cache_dir=Path(cache_dir))
    with zipfile.ZipFile(cached_archive_path, "r") as archive:
        if member_path is None:
            members_list = archive.namelist()
            member_path = get_the_only_file_in_the_archive(members_list, archive_path)
        member_path = cast(str, member_path)
        member_file = archive.open(member_path, "r")
    return io.TextIOWrapper(member_file, encoding=encoding)


def extract_single_zip_file(
    archive_path: str,
    cache_dir: Union[str, Path],
    member_path: Optional[str] = None,
) -> Path:
    cache_dir = Path(cache_dir)
    cached_archive_path = cached_path(archive_path, cache_dir=cache_dir)
    dataset_cache = flair.cache_root / cache_dir if flair.cache_root not in cache_dir.parents else cache_dir
    if member_path is not None:
        output_path = dataset_cache / member_path
        if output_path.exists():
            return output_path
    with zipfile.ZipFile(cached_archive_path, "r") as archive:
        if member_path is None:
            members_list = archive.namelist()
            member_path = get_the_only_file_in_the_archive(members_list, archive_path)
        output_path = dataset_cache / member_path

        if not output_path.exists():
            archive.extract(member_path, dataset_cache)
        return output_path


def get_the_only_file_in_the_archive(members_list: Sequence[str], archive_path: str) -> str:
    if len(members_list) > 1:
        raise ValueError(
            "The archive {} contains multiple files, so you must select "
            "one of the files inside providing a uri of the type: {}".format(
                archive_path,
                format_embeddings_file_uri("path_or_url_to_archive", "path_inside_archive"),
            )
        )
    return members_list[0]


def format_embeddings_file_uri(main_file_path_or_url: str, path_inside_archive: Optional[str] = None) -> str:
    if path_inside_archive:
        return f"({main_file_path_or_url})#{path_inside_archive}"
    return main_file_path_or_url


class Tqdm:
    # These defaults are the same as the argument defaults in tqdm.
    default_mininterval: float = 0.1

    @staticmethod
    def set_default_mininterval(value: float) -> None:
        Tqdm.default_mininterval = value

    @staticmethod
    def set_slower_interval(use_slower_interval: bool) -> None:
        """Slows down the tqdm update interval.

        If ``use_slower_interval`` is ``True``, we will dramatically slow down ``tqdm's`` default
        output rate.  ``tqdm's`` default output rate is great for interactively watching progress,
        but it is not great for log files.  You might want to set this if you are primarily going
        to be looking at output through log files, not the terminal.
        """
        if use_slower_interval:
            Tqdm.default_mininterval = 10.0
        else:
            Tqdm.default_mininterval = 0.1

    @staticmethod
    def tqdm(*args, **kwargs):
        new_kwargs = {"mininterval": Tqdm.default_mininterval, **kwargs}

        return _tqdm(*args, **new_kwargs)


def instance_lru_cache(*cache_args, **cache_kwargs):
    def decorator(func):
        @functools.wraps(func)
        def create_cache(self, *args, **kwargs):
            instance_cache = functools.lru_cache(*cache_args, **cache_kwargs)(func)
            instance_cache = instance_cache.__get__(self, self.__class__)
            setattr(self, func.__name__, instance_cache)
            return instance_cache(*args, **kwargs)

        return create_cache

    return decorator


def load_torch_state(model_file: str) -> dict[str, typing.Any]:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        # load_big_file is a workaround byhttps://github.com/highway11git
        # to load models on some Mac/Windows setups
        # see https://github.com/zalandoresearch/flair/issues/351
        f = load_big_file(model_file)
        return torch.load(f, map_location="cpu")
