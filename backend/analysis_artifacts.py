"""Paths for the files produced by one analysis run."""

from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
import shutil
from tempfile import mkdtemp

_analysis_directory: ContextVar[Path | None] = ContextVar("analysis_directory", default=None)


@contextmanager
def analysis_artifacts():
    root = Path("outputs")
    root.mkdir(exist_ok=True)
    directory = Path(mkdtemp(prefix="analysis-", dir=root))
    token = _analysis_directory.set(directory)
    try:
        yield directory
    except BaseException:
        shutil.rmtree(directory, ignore_errors=True)
        raise
    finally:
        _analysis_directory.reset(token)


def artifact_subdirectory(default: str) -> Path:
    directory = Path(default)
    scoped = _analysis_directory.get()
    if scoped is not None:
        directory = scoped / directory.name
    directory.mkdir(parents=True, exist_ok=True)
    return directory
