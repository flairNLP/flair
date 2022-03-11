"""
Utilities for working with the local dataset cache. Copied from AllenNLP
"""
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
import zipfile
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union, cast
from urllib.parse import urlparse

import requests
from tqdm import tqdm as _tqdm

import flair

logger = logging.getLogger("flair")


def load_big_file(f: str) -> mmap.mmap:
    """
    Workaround for loading a big pickle file. Files over 2GB cause pickle errors on certin Mac and Windows distributions.
    :param f:
    :return:
    """
    logger.info(f"loading file {f}")
    with open(f, "rb") as f_in:
        # mmap seems to be much more memory efficient
        bf = mmap.mmap(f_in.fileno(), 0, access=mmap.ACCESS_READ)
        f_in.close()
    return bf


def url_to_filename(url: str, etag: str = None) -> str:
    """
    Converts a url into a filename in a reversible way.
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


def filename_to_url(filename: str) -> Tuple[str, Optional[str]]:
    """
    Recovers the the url from the encoded filename. Returns it and the ETag
    (which may be ``None``)
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
    """
    Given something that might be a URL (or might be a local path),
    determine which. If it's a URL, download the file and cache it, and
    return the path to the cached file. If it's already a local path,
    make sure the file exists and then return the path.
    """
    if type(cache_dir) is str:
        cache_dir = Path(cache_dir)
    dataset_cache = flair.cache_root / cache_dir

    parsed = urlparse(url_or_filename)

    if parsed.scheme in ("http", "https"):
        # URL, so get it from the cache (downloading if necessary)
        return get_from_cache(url_or_filename, dataset_cache)
    elif parsed.scheme == "" and Path(url_or_filename).exists():
        # File, and it exists.
        return Path(url_or_filename)
    elif parsed.scheme == "":
        # File, but it doesn't exist.
        raise FileNotFoundError("file {} not found".format(url_or_filename))
    else:
        # Something unknown
        raise ValueError("unable to parse {} as a URL or as a local path".format(url_or_filename))


def unzip_file(file: Union[str, Path], unzip_to: Union[str, Path]):
    from zipfile import ZipFile

    with ZipFile(Path(file), "r") as zipObj:
        # Extract all the contents of zip file in current directory
        zipObj.extractall(Path(unzip_to))


def unpack_file(file: Path, unpack_to: Path, mode: str = None, keep: bool = True):
    """
    Unpacks a file to the given location.

    :param file Archive file to unpack
    :param unpack_to Destination where to store the output
    :param mode Type of the archive (zip, tar, gz, targz, rar)
    :param keep Indicates whether to keep the archive after extraction or delete it
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

        with gzip.open(str(file), "rb") as f_in:
            with open(str(unpack_to), "wb") as f_out:
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


def download_file(url: str, cache_dir: Union[str, Path]):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    filename = re.sub(r".+/", "", url)
    # get cache path to put the file
    cache_path = cache_dir / filename

    # Download to temporary file, then copy to cache dir once finished.
    # Otherwise you get corrupt cache entries if the download gets interrupted.
    fd, temp_filename = tempfile.mkstemp()
    logger.info("%s not found in cache, downloading to %s", url, temp_filename)

    # GET file object
    req = requests.get(url, stream=True)
    content_length = req.headers.get("Content-Length")
    total = int(content_length) if content_length is not None else None
    progress = Tqdm.tqdm(unit="B", total=total)
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

    progress.close()


# TODO(joelgrus): do we want to do checksums or anything like that?
def get_from_cache(url: str, cache_dir: Path) -> Path:
    """
    Given a URL, look for the corresponding dataset in the local cache.
    If it's not there, download it. Then return the path to the cached file.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    filename = re.sub(r".+/", "", url)
    # get cache path to put the file
    cache_path = cache_dir / filename
    if cache_path.exists():
        return cache_path

    # make HEAD request to check ETag
    response = requests.head(url, headers={"User-Agent": "Flair"}, allow_redirects=True)
    if response.status_code != 200:
        raise IOError(f"HEAD request failed for url {url} with status code {response.status_code}.")

    # add ETag to filename if it exists
    # etag = response.headers.get("ETag")

    if not cache_path.exists():
        # Download to temporary file, then copy to cache dir once finished.
        # Otherwise you get corrupt cache entries if the download gets interrupted.
        fd, temp_filename = tempfile.mkstemp()
        logger.info("%s not found in cache, downloading to %s", url, temp_filename)

        # GET file object
        req = requests.get(url, stream=True, headers={"User-Agent": "Flair"})
        content_length = req.headers.get("Content-Length")
        total = int(content_length) if content_length is not None else None
        progress = Tqdm.tqdm(unit="B", total=total)
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
    archive = zipfile.ZipFile(cached_archive_path, "r")
    if member_path is None:
        members_list = archive.namelist()
        member_path = get_the_only_file_in_the_archive(members_list, archive_path)
    member_path = cast(str, member_path)
    member_file = archive.open(member_path, "r")
    return io.TextIOWrapper(member_file, encoding=encoding)


def get_the_only_file_in_the_archive(members_list: Sequence[str], archive_path: str) -> str:
    if len(members_list) > 1:
        raise ValueError(
            "The archive %s contains multiple files, so you must select "
            "one of the files inside providing a uri of the type: %s"
            % (
                archive_path,
                format_embeddings_file_uri("path_or_url_to_archive", "path_inside_archive"),
            )
        )
    return members_list[0]


def format_embeddings_file_uri(main_file_path_or_url: str, path_inside_archive: Optional[str] = None) -> str:
    if path_inside_archive:
        return "({})#{}".format(main_file_path_or_url, path_inside_archive)
    return main_file_path_or_url


class Tqdm:
    # These defaults are the same as the argument defaults in tqdm.
    default_mininterval: float = 0.1

    @staticmethod
    def set_default_mininterval(value: float) -> None:
        Tqdm.default_mininterval = value

    @staticmethod
    def set_slower_interval(use_slower_interval: bool) -> None:
        """
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
