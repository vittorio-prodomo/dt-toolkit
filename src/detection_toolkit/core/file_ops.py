"""File operations with symlink → hardlink → copy fallback and rollback."""

import logging
import os
import platform
import shutil
from pathlib import Path
from typing import List, Literal, Tuple

logger = logging.getLogger(__name__)

LinkType = Literal["auto", "symlink", "hardlink", "copy", "deref_symlink"]


class FileOperations:
    """Unified file operations with fallback strategy and atomic rollback.

    Default strategy ("auto"): tries symlink first, falls back to hardlink,
    then to copy. Every operation is logged so it can be undone with
    ``rollback_operations()``.
    """

    def __init__(self, verbose: bool = True) -> None:
        self.verbose = verbose
        self.operations_log: List[Tuple[str, Path, Path]] = []
        self._symlinks_supported = self._check_symlink_support()

        if self.verbose:
            if self._symlinks_supported:
                logger.info("Platform supports symbolic links")
            else:
                logger.info(
                    "Platform does not support symbolic links; "
                    "will use hardlinks/copies"
                )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_file_link(
        self,
        src_path: Path,
        dst_path: Path,
        link_type: LinkType = "auto",
        use_relative: bool = True,
    ) -> None:
        """Link *src_path* to *dst_path* using the chosen strategy.

        Args:
            src_path: Existing source file.
            dst_path: Destination path (will be created).
            link_type: One of ``"auto"``, ``"symlink"``, ``"hardlink"``,
                ``"copy"``, ``"deref_symlink"``.
            use_relative: For symlinks, prefer relative target paths.
        """
        if not src_path.exists():
            raise FileNotFoundError(f"Source file does not exist: {src_path}")

        if src_path.resolve() == dst_path.resolve():
            if self.verbose:
                logger.debug("Source and destination are identical, skipping: %s", src_path)
            return

        dst_path.parent.mkdir(parents=True, exist_ok=True)

        if link_type == "auto":
            self._auto_link(src_path, dst_path, use_relative)
        elif link_type == "symlink":
            self._create_symlink(src_path, dst_path, use_relative)
            self._log("symlink", src_path, dst_path)
        elif link_type == "hardlink":
            self._create_hardlink(src_path, dst_path)
            self._log("hardlink", src_path, dst_path)
        elif link_type == "copy":
            self._create_copy(src_path, dst_path)
            self._log("copy", src_path, dst_path)
        elif link_type == "deref_symlink":
            self._create_deref_symlink(src_path, dst_path, use_relative)
            self._log("deref_symlink", src_path, dst_path)
        else:
            raise ValueError(f"Unsupported link type: {link_type}")

    def rollback_operations(self) -> None:
        """Undo every logged operation in reverse order."""
        if self.verbose:
            logger.info("Rolling back %d operations ...", len(self.operations_log))

        for operation, _src, dst in reversed(self.operations_log):
            try:
                if dst.is_symlink() or dst.is_file():
                    dst.unlink()
                elif dst.is_dir():
                    shutil.rmtree(dst)
                if self.verbose:
                    logger.debug("Rolled back %s: %s", operation, dst)
            except Exception as exc:
                logger.warning("Failed to rollback %s %s: %s", operation, dst, exc)

        self.operations_log.clear()

    def clear_log(self) -> None:
        """Discard the operation log without undoing anything."""
        self.operations_log.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _auto_link(
        self, src: Path, dst: Path, use_relative: bool
    ) -> None:
        if self._symlinks_supported:
            try:
                self._create_symlink(src, dst, use_relative)
                self._log("symlink", src, dst)
                return
            except (OSError, ValueError) as exc:
                if self.verbose:
                    logger.debug("Symlink failed (%s), trying hardlink ...", exc)

        try:
            self._create_hardlink(src, dst)
            self._log("hardlink", src, dst)
            return
        except OSError as exc:
            if self.verbose:
                logger.debug("Hardlink failed (%s), falling back to copy ...", exc)

        self._create_copy(src, dst)
        self._log("copy", src, dst)

    def _create_symlink(
        self, src: Path, dst: Path, use_relative: bool
    ) -> None:
        if not self._symlinks_supported:
            raise OSError("Symbolic links not supported on this platform")
        if use_relative:
            try:
                rel = os.path.relpath(src, dst.parent)
                os.symlink(rel, dst)
                return
            except (OSError, ValueError):
                pass
        os.symlink(src.absolute(), dst)

    def _create_hardlink(self, src: Path, dst: Path) -> None:
        os.link(src.resolve(), dst)

    def _create_copy(self, src: Path, dst: Path) -> None:
        shutil.copy2(src, dst)

    def _create_deref_symlink(
        self, src: Path, dst: Path, use_relative: bool
    ) -> None:
        """Create a symlink that points to the *resolved* target of src."""
        target = src.resolve()
        if use_relative:
            try:
                rel = os.path.relpath(target, dst.parent)
                os.symlink(rel, dst)
                return
            except (OSError, ValueError):
                pass
        os.symlink(target, dst)

    def _log(self, op: str, src: Path, dst: Path) -> None:
        self.operations_log.append((op, src, dst))
        if self.verbose:
            logger.debug("Created %s: %s -> %s", op, src.name, dst)

    @staticmethod
    def _check_symlink_support() -> bool:
        if platform.system() != "Windows":
            return True
        try:
            import tempfile

            with tempfile.TemporaryDirectory() as tmp:
                src = Path(tmp) / "src.txt"
                lnk = Path(tmp) / "lnk.txt"
                src.write_text("test")
                os.symlink(src, lnk)
                lnk.unlink()
            return True
        except (OSError, NotImplementedError):
            return False
