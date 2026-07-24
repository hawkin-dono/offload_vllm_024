# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared, CUDA-registered host storage for offloaded expert weights.

When the prefetch worker runs in a separate process, both processes issue async
H2D copies from the same host weight store. `pin_memory()` storage cannot do
that -- it is anonymous, process-private memory -- so this module places the
store in file-backed shared memory instead: the parent creates one file per
tensor (under /dev/shm by default), both processes mmap the same file -- the
same physical pages -- and each calls `cudaHostRegister` on its own mapping,
which is what makes `copy_(non_blocking=True)` a true DMA in that process's own
CUDA context. Torch reports registered memory as pinned, so everything
downstream that checks `is_pinned()` or relies on views staying async keeps
working unchanged.
"""

import atexit
import contextlib
import os
import re
import time
import uuid
from dataclasses import dataclass

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

DEFAULT_SHM_DIR = "/dev/shm"

# Arena files are named vllm-expert-arena-<owner pid>-<uuid>-<counter>.
_ARENA_FILE_RE = re.compile(r"^vllm-expert-arena-(\d+)-")


def reclaim_stale_arenas(directory: str) -> None:
    """Unlink arena files whose owning process is gone.

    An owner that dies between creating its files and unlinking them (worst
    case: SIGBUS from tmpfs running out of pages mid-copy, which skips even
    atexit) leaks tens of GiB of shared memory -- enough to starve the next
    run. The owner pid is in the filename, so orphans are cheap to detect.
    """
    try:
        names = os.listdir(directory)
    except OSError:
        return
    freed = 0
    for name in names:
        m = _ARENA_FILE_RE.match(name)
        if m is None or _pid_alive(int(m.group(1))):
            continue
        path = os.path.join(directory, name)
        try:
            freed += os.path.getsize(path)
            os.unlink(path)
        except OSError:
            continue
    if freed:
        logger.warning(
            "Expert cache arena: reclaimed %.1f GiB of stale arena files in "
            "%s (a previous run died before cleaning up)",
            freed / 1024**3,
            directory,
        )


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@dataclass(frozen=True)
class _ArenaEntry:
    """Where one tensor lives: its backing file and how to view it."""

    filename: str
    shape: tuple[int, ...]
    dtype_name: str  # "bfloat16" -- torch dtypes do not pickle by themselves

    @property
    def dtype(self) -> torch.dtype:
        return getattr(torch, self.dtype_name)

    @property
    def nbytes(self) -> int:
        n = 1
        for s in self.shape:
            n *= s
        return n * torch.empty(0, dtype=self.dtype).element_size()


@dataclass(frozen=True)
class ArenaDescriptor:
    """Picklable handle a child process attaches with (see `attach`)."""

    directory: str
    entries: dict[str, _ArenaEntry]


class SharedHostArena:
    """Named host tensors in shared memory, registrable in any process.

    The parent builds the arena (`add` per tensor, then `register`), ships
    `descriptor()` to the worker process, which `attach`es and `register`s its
    own mapping. `unlink` removes the names once every process has attached;
    the pages live until the last mapping goes away.
    """

    def __init__(self, directory: str = DEFAULT_SHM_DIR):
        self._directory = directory
        self._prefix = f"vllm-expert-arena-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        self._entries: dict[str, _ArenaEntry] = {}
        self._tensors: dict[str, torch.Tensor] = {}
        self._registered: dict[str, int] = {}  # key -> data_ptr
        self._owner = True
        self._counter = 0
        reclaim_stale_arenas(directory)
        atexit.register(self.cleanup)

    def check_capacity(self, needed_bytes: int) -> None:
        """Fail loudly up front if the backing filesystem cannot hold the
        arena.

        Running out mid-copy is much worse than an error: tmpfs overcommits
        (creating the file and sizing it are metadata-only), so the first
        write to a page it cannot back kills the process with SIGBUS -- no
        exception, no atexit, and the partial arena leaks.
        """
        st = os.statvfs(self._directory)
        free = st.f_bavail * st.f_frsize
        if needed_bytes > free:
            raise RuntimeError(
                f"{self._directory} has {free / 1024**3:.1f} GiB free but the "
                f"shared expert store needs ~{needed_bytes / 1024**3:.1f} GiB. "
                "Free space there or point prefetch_shm_dir at a larger "
                "tmpfs."
            )

    @classmethod
    def attach(cls, descriptor: ArenaDescriptor) -> "SharedHostArena":
        """Map an existing arena in this (child) process. Registration is a
        separate, explicit step -- it is the slow part and callers may want to
        order it against other init work."""
        arena = cls.__new__(cls)
        arena._directory = descriptor.directory
        arena._prefix = ""
        arena._entries = dict(descriptor.entries)
        arena._tensors = {}
        arena._registered = {}
        arena._owner = False
        arena._counter = 0
        for key, entry in arena._entries.items():
            arena._tensors[key] = arena._map(entry, must_exist=True)
        atexit.register(arena.cleanup)
        return arena

    def _path(self, entry: _ArenaEntry) -> str:
        return os.path.join(self._directory, entry.filename)

    def _map(self, entry: _ArenaEntry, must_exist: bool) -> torch.Tensor:
        path = self._path(entry)
        # from_file(shared=True) would silently create a zero-filled file;
        # attaching to weights that are not there must be loud instead.
        if must_exist and (
            not os.path.exists(path) or os.path.getsize(path) != entry.nbytes
        ):
            raise FileNotFoundError(
                f"Shared arena file {path} is missing or has the wrong "
                f"size; the owning process is gone or unlinked it early."
            )
        storage = torch.UntypedStorage.from_file(path, shared=True, nbytes=entry.nbytes)
        return torch.empty(0, dtype=entry.dtype).set_(storage, 0, entry.shape)

    def add(self, key: str, src: torch.Tensor) -> torch.Tensor:
        """Copy `src` (a CPU tensor) into a new shared entry; return the view.

        The returned tensor aliases the mapped file from storage offset 0, so
        `blob_aliases` and per-expert row views work on it exactly as they do
        on `pin_memory()` storage.
        """
        if not self._owner:
            raise RuntimeError("Only the owning process may add arena entries.")
        if key in self._entries:
            raise ValueError(f"Arena entry {key!r} already exists.")
        if src.device.type != "cpu":
            raise ValueError(f"Arena entries are host tensors; got {src.device}.")
        entry = _ArenaEntry(
            filename=f"{self._prefix}-{self._counter:04d}",
            shape=tuple(src.shape),
            dtype_name=str(src.dtype).removeprefix("torch."),
        )
        self._counter += 1
        dst = self._map(entry, must_exist=False)
        dst.copy_(src)
        self._entries[key] = entry
        self._tensors[key] = dst
        return dst

    def tensor(self, key: str) -> torch.Tensor:
        return self._tensors[key]

    def keys(self) -> list[str]:
        return list(self._entries)

    def nbytes(self) -> int:
        return sum(e.nbytes for e in self._entries.values())

    def descriptor(self) -> ArenaDescriptor:
        return ArenaDescriptor(directory=self._directory, entries=dict(self._entries))

    def register(self) -> None:
        """`cudaHostRegister` every mapping in the calling process's context.

        This is what pins the pages here: without it the copies fall back to
        pageable (synchronous) semantics. Idempotent. Slow for large stores
        (roughly pin_memory() speed), hence separate from `add`/`attach`.
        """
        cudart = torch.cuda.cudart()
        total = 0
        t0 = time.perf_counter()
        for key, tensor in self._tensors.items():
            if key in self._registered:
                continue
            ptr = tensor.data_ptr()
            nbytes = tensor.numel() * tensor.element_size()
            r = cudart.cudaHostRegister(ptr, nbytes, 0)
            if r != 0:
                raise RuntimeError(
                    f"cudaHostRegister failed for arena entry {key!r} "
                    f"({nbytes} bytes): error {int(r)}"
                )
            self._registered[key] = ptr
            total += nbytes
        if total:
            dt = time.perf_counter() - t0
            logger.info(
                "Expert cache arena: registered %.2f GiB of shared host "
                "memory in %.1fs",
                total / 1024**3,
                dt,
            )

    def unregister(self) -> None:
        if not self._registered:
            return
        try:
            cudart = torch.cuda.cudart()
            for ptr in self._registered.values():
                cudart.cudaHostUnregister(ptr)
        except Exception:
            # Interpreter/CUDA teardown order is not ours to control; the OS
            # reclaims everything at process exit anyway.
            pass
        self._registered.clear()

    def unlink(self) -> None:
        """Remove the backing filenames. Safe once every process has attached:
        existing mappings keep the pages alive; only the names go away."""
        if not self._owner:
            return
        for entry in self._entries.values():
            with contextlib.suppress(FileNotFoundError):
                os.unlink(self._path(entry))

    def cleanup(self) -> None:
        self.unregister()
        self.unlink()
