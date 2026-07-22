"""Offline validation for EmiyaAgent wheel and source distributions.

The verifier intentionally does not extract archives.  It inspects member names,
types, and a small set of bounded metadata files before a release artifact is
published.
"""

from __future__ import annotations

import argparse
import ast
import base64
import configparser
import csv
import email.policy
import gzip
import hashlib
import io
import os
import re
import stat
import sys
import tarfile
import tomllib
import unicodedata
import zipfile
from dataclasses import dataclass
from email.parser import BytesParser
from pathlib import Path, PurePosixPath
from typing import BinaryIO, NoReturn

EXIT_OK = 0
EXIT_INVALID = 1
EXIT_USAGE = 2
EXIT_INTERNAL_ERROR = 3

MAX_ARCHIVE_BYTES = 128 * 1024 * 1024
MAX_MEMBER_BYTES = 32 * 1024 * 1024
MAX_TOTAL_MEMBER_BYTES = 256 * 1024 * 1024
MAX_MEMBERS = 20_000
MAX_METADATA_BYTES = 1024 * 1024
MAX_TAR_STREAM_BYTES = 320 * 1024 * 1024
MAX_MEMBER_PATH_UTF16 = 240
MAX_COMPONENT_UTF16 = 255
LICENSE_EXPRESSION = "MIT"
LICENSE_FILENAME = "LICENSE"

_FORBIDDEN_COMPONENTS = frozenset(
    {
        ".emiyacode",
        ".git",
        ".hg",
        ".mypy_cache",
        ".nox",
        ".pytest_cache",
        ".ruff_cache",
        ".svn",
        ".tox",
        ".uv-cache",
        ".venv",
        "__pycache__",
        "logs",
        "sessions",
    }
)
_FORBIDDEN_EXACT_FILES = frozenset(
    {
        ".env",
        "credentials.json",
        "id_dsa",
        "id_ecdsa",
        "id_ed25519",
        "id_rsa",
        "secrets.json",
    }
)
_PRIVATE_KEY_SUFFIXES = (
    ".jks",
    ".key",
    ".keystore",
    ".p12",
    ".pfx",
    ".pkcs12",
    ".pem",
)
_WINDOWS_RESERVED_NAMES = frozenset(
    {"aux", "con", "nul", "prn"}
    | {f"com{number}" for number in range(1, 10)}
    | {f"lpt{number}" for number in range(1, 10)}
)


class ArtifactValidationError(Exception):
    """An artifact is unsafe or inconsistent with project metadata."""


@dataclass(frozen=True, slots=True)
class ProjectMetadata:
    name: str
    version: str
    scripts: dict[str, str]
    requires_python: str | None
    dependencies: tuple[str, ...]
    readme: str | None
    license_expression: str

    @property
    def wheel_name(self) -> str:
        return re.sub(r"[-_.]+", "_", self.name).lower()

    @property
    def wheel_version(self) -> str:
        return re.sub(r"[^A-Za-z0-9.]+", "_", self.version)

    @property
    def archive_root(self) -> str:
        return f"{self.wheel_name}-{self.version}"

    @property
    def dist_info(self) -> str:
        return f"{self.wheel_name}-{self.wheel_version}.dist-info"


def _invalid(message: str) -> NoReturn:
    raise ArtifactValidationError(message)


def _normalized_component(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold()


def _safe_member_parts(name: str) -> tuple[str, ...]:
    if not name or "\\" in name:
        _invalid("archive contains an unsafe member path")
    if any(unicodedata.category(char) in {"Cc", "Cs"} for char in name):
        _invalid("archive contains an unsafe member path")
    if name.startswith("/") or re.match(r"^[A-Za-z]:", name):
        _invalid("archive contains an unsafe member path")
    if "//" in name:
        _invalid("archive contains an unsafe member path")

    path = PurePosixPath(name)
    parts = path.parts
    if path.is_absolute() or not parts:
        _invalid("archive contains an unsafe member path")
    if any(part in {"", ".", ".."} for part in parts):
        _invalid("archive contains an unsafe member path")
    if any(
        any(char in part for char in '<>:"|?*') or part.endswith((" ", "."))
        for part in parts
    ):
        _invalid("archive contains a non-portable member path")
    for part in parts:
        normalized = _normalized_component(part)
        if normalized in {"", ".", ".."} or normalized.endswith((" ", ".")):
            _invalid("archive contains a non-portable member path")
        if any(char in normalized for char in '/\\<>:"|?*'):
            _invalid("archive contains a non-portable member path")
        if any(unicodedata.category(char) in {"Cc", "Cs"} for char in normalized):
            _invalid("archive contains a non-portable member path")
        try:
            component_units = len(part.encode("utf-16-le")) // 2
        except UnicodeEncodeError as exc:
            raise ArtifactValidationError(
                "archive contains a non-portable member path"
            ) from exc
        if component_units > MAX_COMPONENT_UTF16:
            _invalid("archive member path component is too long")
        device_base = normalized.split(".", 1)[0].rstrip(" ")
        if device_base in _WINDOWS_RESERVED_NAMES:
            _invalid("archive contains a Windows-reserved member path")
    try:
        path_units = len(name.encode("utf-16-le")) // 2
    except UnicodeEncodeError as exc:
        raise ArtifactValidationError(
            "archive contains a non-portable member path"
        ) from exc
    if path_units > MAX_MEMBER_PATH_UTF16:
        _invalid("archive member path is too long")
    return parts


def _canonical_member(parts: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(_normalized_component(part) for part in parts)


def _is_sensitive_member(parts: tuple[str, ...], *, is_dir: bool) -> bool:
    normalized = tuple(_normalized_component(part) for part in parts)
    if any(part in _FORBIDDEN_COMPONENTS for part in normalized):
        return True
    if is_dir:
        return False

    filename = normalized[-1]
    if filename in _FORBIDDEN_EXACT_FILES or filename.startswith(".env."):
        return True
    if filename.endswith(_PRIVATE_KEY_SUFFIXES):
        return True
    if filename.endswith(".jsonl") or filename.endswith(".log"):
        return True
    if ".log." in filename:
        return True
    return False


def _reject_sensitive_member(parts: tuple[str, ...], *, is_dir: bool) -> None:
    if _is_sensitive_member(parts, is_dir=is_dir):
        _invalid("archive contains a forbidden runtime or sensitive file")


def _literal_version(source: str, *, label: str) -> str:
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError) as exc:
        raise ArtifactValidationError(f"{label} is not valid Python") from exc

    values: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__version__"
            for target in node.targets
        ):
            if not isinstance(node.value, ast.Constant) or not isinstance(
                node.value.value, str
            ):
                _invalid(f"{label} has a non-literal __version__")
            values.append(node.value.value)
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "__version__"
        ):
            if not isinstance(node.value, ast.Constant) or not isinstance(
                node.value.value, str
            ):
                _invalid(f"{label} has a non-literal __version__")
            values.append(node.value.value)

    if len(values) != 1 or not values[0]:
        _invalid(f"{label} must define one literal __version__")
    return values[0]


def _metadata_from_pyproject(data: bytes, *, label: str) -> ProjectMetadata:
    if len(data) > MAX_METADATA_BYTES:
        _invalid(f"{label} exceeds the metadata size limit")
    try:
        document = tomllib.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ArtifactValidationError(f"{label} is not valid UTF-8 TOML") from exc

    project = document.get("project")
    if not isinstance(project, dict):
        _invalid(f"{label} has no [project] table")
    name = project.get("name")
    version = project.get("version")
    scripts = project.get("scripts", {})
    requires_python = project.get("requires-python")
    dependencies = project.get("dependencies", [])
    readme = project.get("readme")
    license_expression = project.get("license")
    if not isinstance(name, str) or not name.strip():
        _invalid(f"{label} has no valid project name")
    if not isinstance(version, str) or not version.strip():
        _invalid(f"{label} has no valid project version")
    if not isinstance(scripts, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in scripts.items()
    ):
        _invalid(f"{label} has invalid project scripts")
    if requires_python is not None and (
        not isinstance(requires_python, str) or not requires_python.strip()
    ):
        _invalid(f"{label} has invalid requires-python metadata")
    if not isinstance(dependencies, list) or not all(
        isinstance(value, str) and value.strip() for value in dependencies
    ):
        _invalid(f"{label} has invalid project dependencies")
    if license_expression != LICENSE_EXPRESSION:
        _invalid(
            f"{label} must declare the {LICENSE_EXPRESSION} license expression"
        )
    normalized_dependencies = tuple(value.strip() for value in dependencies)
    if len(set(normalized_dependencies)) != len(normalized_dependencies):
        _invalid(f"{label} has duplicate project dependencies")
    if readme is not None:
        if not isinstance(readme, str) or not readme.strip():
            _invalid(f"{label} has an unsupported project readme declaration")
        readme = readme.strip()
        readme_parts = _safe_member_parts(readme)
        _reject_sensitive_member(readme_parts, is_dir=False)
    return ProjectMetadata(
        name=name,
        version=version,
        scripts=dict(scripts),
        requires_python=requires_python.strip() if requires_python else None,
        dependencies=normalized_dependencies,
        readme=readme,
        license_expression=license_expression,
    )


def _load_project_metadata(project_root: Path) -> ProjectMetadata:
    try:
        pyproject_data = (project_root / "pyproject.toml").read_bytes()
        license_data = (project_root / LICENSE_FILENAME).read_bytes()
        init_data = (project_root / "emiya_agent" / "__init__.py").read_text(
            encoding="utf-8"
        )
    except OSError as exc:
        raise ArtifactValidationError("project metadata cannot be read") from exc

    metadata = _metadata_from_pyproject(pyproject_data, label="project pyproject.toml")
    if not license_data or len(license_data) > MAX_METADATA_BYTES:
        _invalid("project LICENSE has an invalid size")
    source_version = _literal_version(
        init_data, label="project emiya_agent/__init__.py"
    )
    if source_version != metadata.version:
        _invalid("project version declarations do not match")
    if metadata.scripts.get("emiya") != "emiya_agent.cli:main":
        _invalid("project does not declare the expected emiya CLI entry point")
    return metadata


def _is_link_or_reparse(info: os.stat_result) -> bool:
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0) & reparse_flag
    )


def _validate_archive_path(path: Path) -> None:
    try:
        info = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise ArtifactValidationError("release artifact cannot be inspected") from exc
    if _is_link_or_reparse(info) or not stat.S_ISREG(info.st_mode):
        _invalid("dist contains a non-regular artifact")
    if info.st_size <= 0 or info.st_size > MAX_ARCHIVE_BYTES:
        _invalid("release artifact has an invalid size")


def _source_package_files(package_root: Path) -> dict[str, Path]:
    try:
        root_info = package_root.stat(follow_symlinks=False)
    except OSError as exc:
        raise ArtifactValidationError("source package cannot be inspected") from exc
    if _is_link_or_reparse(root_info) or not stat.S_ISDIR(root_info.st_mode):
        _invalid("source package must be a regular directory")

    files: dict[str, Path] = {}
    pending: list[tuple[Path, tuple[str, ...]]] = [(package_root, ("emiya_agent",))]
    visited = 0
    total_size = 0
    while pending:
        directory, prefix = pending.pop()
        try:
            entries = list(os.scandir(directory))
        except OSError as exc:
            raise ArtifactValidationError("source package cannot be read") from exc
        for entry in entries:
            visited += 1
            if visited > MAX_MEMBERS:
                _invalid("source package has too many files")
            try:
                info = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise ArtifactValidationError(
                    "source package member cannot be inspected"
                ) from exc
            if _is_link_or_reparse(info):
                _invalid("source package contains a link or reparse point")

            parts = (*prefix, entry.name)
            _safe_member_parts("/".join(parts))
            if stat.S_ISDIR(info.st_mode):
                if not _is_sensitive_member(parts, is_dir=True):
                    pending.append((Path(entry.path), parts))
                continue
            if not stat.S_ISREG(info.st_mode):
                _invalid("source package contains a non-regular member")
            if _is_sensitive_member(parts, is_dir=False):
                continue
            if info.st_size < 0 or info.st_size > MAX_MEMBER_BYTES:
                _invalid("source package member exceeds the size limit")
            total_size += info.st_size
            if total_size > MAX_TOTAL_MEMBER_BYTES:
                _invalid("source package contents exceed the size limit")
            files["/".join(parts)] = Path(entry.path)
    if not files:
        _invalid("source package contains no distributable files")
    return files


def _source_identity(info: os.stat_result) -> tuple[int, int, int, int]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
    )


def _source_digest(path: Path) -> tuple[int, bytes]:
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        before = path.stat(follow_symlinks=False)
        if _is_link_or_reparse(before) or not stat.S_ISREG(before.st_mode):
            _invalid("source package member is no longer a regular file")
        descriptor = os.open(path, flags)
        try:
            with os.fdopen(descriptor, "rb") as stream:
                descriptor = -1
                opened = os.fstat(stream.fileno())
                if (
                    _is_link_or_reparse(opened)
                    or not stat.S_ISREG(opened.st_mode)
                    or _source_identity(opened) != _source_identity(before)
                ):
                    _invalid("source package changed during verification")
                digest = hashlib.file_digest(stream, "sha256").digest()
        finally:
            if descriptor >= 0:
                os.close(descriptor)
        after = path.stat(follow_symlinks=False)
    except ArtifactValidationError:
        raise
    except OSError as exc:
        raise ArtifactValidationError("source package member cannot be read") from exc
    if (
        _is_link_or_reparse(after)
        or not stat.S_ISREG(after.st_mode)
        or _source_identity(after) != _source_identity(before)
    ):
        _invalid("source package changed during verification")
    return opened.st_size, digest


def _source_file_for_member(project_root: Path, parts: tuple[str, ...]) -> Path:
    """Resolve an sdist member to an existing regular source file, no-follow."""

    current = project_root
    try:
        root_info = current.stat(follow_symlinks=False)
    except OSError as exc:
        raise ArtifactValidationError("source tree cannot be inspected") from exc
    if _is_link_or_reparse(root_info) or not stat.S_ISDIR(root_info.st_mode):
        _invalid("source tree root must be a regular directory")

    for index, part in enumerate(parts):
        current = current / part
        try:
            info = current.stat(follow_symlinks=False)
        except OSError as exc:
            raise ArtifactValidationError(
                "sdist contains a file not present in the source tree"
            ) from exc
        if _is_link_or_reparse(info):
            _invalid("sdist source path contains a link or reparse point")
        if index < len(parts) - 1:
            if not stat.S_ISDIR(info.st_mode):
                _invalid("sdist source path has a non-directory parent")
        elif not stat.S_ISREG(info.st_mode):
            _invalid("sdist member does not map to a regular source file")
    return current


def _stream_digest(stream: BinaryIO, expected_size: int, *, label: str) -> bytes:
    digest = hashlib.sha256()
    remaining = expected_size
    while remaining:
        chunk = stream.read(min(64 * 1024, remaining))
        if not chunk:
            _invalid(f"{label} has an invalid size")
        digest.update(chunk)
        remaining -= len(chunk)
    if stream.read(1):
        _invalid(f"{label} has an invalid size")
    return digest.digest()


def _zip_digest(
    archive: zipfile.ZipFile, info: zipfile.ZipInfo, *, label: str
) -> bytes:
    try:
        with archive.open(info, mode="r") as stream:
            return _stream_digest(stream, info.file_size, label=label)
    except ArtifactValidationError:
        raise
    except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
        raise ArtifactValidationError(f"{label} cannot be read safely") from exc


def _bounded_read(stream: BinaryIO, expected_size: int, *, label: str) -> bytes:
    if expected_size < 0 or expected_size > MAX_METADATA_BYTES:
        _invalid(f"{label} exceeds the metadata size limit")
    data = stream.read(MAX_METADATA_BYTES + 1)
    if len(data) > MAX_METADATA_BYTES or len(data) != expected_size:
        _invalid(f"{label} has an invalid size")
    return data


def _parse_core_metadata(data: bytes, *, label: str) -> tuple[str, str]:
    if len(data) > MAX_METADATA_BYTES:
        _invalid(f"{label} exceeds the metadata size limit")
    try:
        message = BytesParser(policy=email.policy.compat32).parsebytes(data)
    except Exception as exc:  # email parsing can surface malformed custom factories
        raise ArtifactValidationError(f"{label} is malformed") from exc
    metadata_versions = [
        str(value).strip() for value in message.get_all("Metadata-Version", [])
    ]
    names = message.get_all("Name", [])
    versions = message.get_all("Version", [])
    if metadata_versions != ["2.4"]:
        _invalid(f"{label} must contain Metadata-Version 2.4")
    if len(names) != 1 or len(versions) != 1:
        _invalid(f"{label} must contain one Name and Version")
    return str(names[0]), str(versions[0])


def _verify_name_and_version(
    actual_name: str,
    actual_version: str,
    expected: ProjectMetadata,
    *,
    label: str,
) -> None:
    if (
        re.sub(r"[-_.]+", "-", actual_name).casefold()
        != re.sub(r"[-_.]+", "-", expected.name).casefold()
    ):
        _invalid(f"{label} project name does not match pyproject.toml")
    if actual_version != expected.version:
        _invalid(f"{label} version does not match pyproject.toml")


def _verify_distribution_metadata_fields(
    data: bytes, expected: ProjectMetadata, *, label: str
) -> None:
    try:
        message = BytesParser(policy=email.policy.compat32).parsebytes(data)
    except Exception as exc:  # pragma: no cover - defensive parser boundary
        raise ArtifactValidationError(f"{label} is malformed") from exc

    actual_requires_python = [
        str(value).strip() for value in message.get_all("Requires-Python", [])
    ]
    expected_requires_python = (
        [expected.requires_python] if expected.requires_python is not None else []
    )
    if actual_requires_python != expected_requires_python:
        _invalid(f"{label} Requires-Python does not match pyproject.toml")

    actual_dependencies = [
        str(value).strip() for value in message.get_all("Requires-Dist", [])
    ]
    if len(set(actual_dependencies)) != len(actual_dependencies) or sorted(
        actual_dependencies
    ) != sorted(expected.dependencies):
        _invalid(f"{label} dependencies do not match pyproject.toml")

    actual_license_expressions = [
        str(value).strip() for value in message.get_all("License-Expression", [])
    ]
    if actual_license_expressions != [expected.license_expression]:
        _invalid(f"{label} License-Expression does not match pyproject.toml")
    actual_license_files = [
        str(value).strip() for value in message.get_all("License-File", [])
    ]
    if actual_license_files != [LICENSE_FILENAME]:
        _invalid(f"{label} must declare exactly one {LICENSE_FILENAME} license file")


def _verify_wheel_descriptor(data: bytes) -> None:
    try:
        message = BytesParser(policy=email.policy.compat32).parsebytes(data)
    except Exception as exc:  # pragma: no cover - defensive parser boundary
        raise ArtifactValidationError("wheel WHEEL metadata is malformed") from exc
    wheel_versions = [
        str(value).strip() for value in message.get_all("Wheel-Version", [])
    ]
    purelib_values = [
        str(value).strip().casefold()
        for value in message.get_all("Root-Is-Purelib", [])
    ]
    tags = [str(value).strip() for value in message.get_all("Tag", [])]
    if (
        wheel_versions != ["1.0"]
        or purelib_values != ["true"]
        or tags != ["py3-none-any"]
    ):
        _invalid("wheel descriptor does not declare the expected pure-Python tag")


def _record_digest(value: str) -> bytes:
    if not value.startswith("sha256="):
        _invalid("wheel RECORD contains a missing or unsupported hash")
    encoded = value.removeprefix("sha256=")
    if not encoded or "=" in encoded:
        _invalid("wheel RECORD contains an invalid hash")
    try:
        decoded = base64.b64decode(
            encoded + "=" * (-len(encoded) % 4), altchars=b"-_", validate=True
        )
    except (ValueError, base64.binascii.Error) as exc:
        raise ArtifactValidationError("wheel RECORD contains an invalid hash") from exc
    if len(decoded) != hashlib.sha256().digest_size:
        _invalid("wheel RECORD contains an invalid hash")
    return decoded


def _verify_wheel_record(
    archive: zipfile.ZipFile,
    members: dict[str, zipfile.ZipInfo],
    *,
    record_name: str,
) -> None:
    record_info = members[record_name]
    with archive.open(record_info, mode="r") as stream:
        data = _bounded_read(stream, record_info.file_size, label="wheel RECORD")
    try:
        text = data.decode("utf-8")
        rows = list(csv.reader(io.StringIO(text, newline=""), strict=True))
    except (UnicodeDecodeError, csv.Error) as exc:
        raise ArtifactValidationError("wheel RECORD is malformed") from exc

    regular_members = {name for name, info in members.items() if not info.is_dir()}
    seen: set[str] = set()
    for row in rows:
        if len(row) != 3:
            _invalid("wheel RECORD contains an invalid row")
        name, encoded_digest, size_text = row
        _safe_member_parts(name)
        if name in seen or name not in regular_members:
            _invalid("wheel RECORD file set does not match the archive")
        seen.add(name)
        if name == record_name:
            if encoded_digest or size_text:
                _invalid("wheel RECORD must leave its own hash and size empty")
            continue
        if not size_text.isdecimal() or int(size_text) != members[name].file_size:
            _invalid("wheel RECORD contains an invalid member size")
        if _zip_digest(
            archive, members[name], label="wheel RECORD member"
        ) != _record_digest(encoded_digest):
            _invalid("wheel RECORD hash does not match the archive")
    if seen != regular_members:
        _invalid("wheel RECORD file set does not match the archive")


def _verify_wheel(path: Path, expected: ProjectMetadata, *, package_root: Path) -> None:
    expected_filename = (
        f"{expected.wheel_name}-{expected.wheel_version}-py3-none-any.whl"
    )
    if path.name != expected_filename:
        _invalid("wheel filename does not match project metadata")

    metadata_name = f"{expected.dist_info}/METADATA"
    entry_points_name = f"{expected.dist_info}/entry_points.txt"
    wheel_descriptor_name = f"{expected.dist_info}/WHEEL"
    record_name = f"{expected.dist_info}/RECORD"
    license_name = f"{expected.dist_info}/licenses/{LICENSE_FILENAME}"
    init_name = "emiya_agent/__init__.py"
    required = {
        metadata_name,
        entry_points_name,
        wheel_descriptor_name,
        record_name,
        init_name,
        license_name,
    }
    members: dict[str, zipfile.ZipInfo] = {}
    canonical_members: set[tuple[str, ...]] = set()
    total_size = 0

    try:
        with zipfile.ZipFile(path, mode="r") as archive:
            infos = archive.infolist()
            if not infos or len(infos) > MAX_MEMBERS:
                _invalid("wheel has an invalid member count")
            for info in infos:
                parts = _safe_member_parts(info.filename)
                if info.filename in members:
                    _invalid("wheel contains duplicate member names")
                canonical = _canonical_member(parts)
                if canonical in canonical_members:
                    _invalid("wheel contains cross-platform duplicate member names")
                canonical_members.add(canonical)

                unix_mode = (info.external_attr >> 16) & 0xFFFF
                file_type = stat.S_IFMT(unix_mode)
                allowed_types = {0, stat.S_IFDIR if info.is_dir() else stat.S_IFREG}
                if file_type not in allowed_types:
                    _invalid("wheel contains a non-regular member")
                if info.flag_bits & 0x1:
                    _invalid("wheel contains an encrypted member")
                if info.file_size < 0 or info.file_size > MAX_MEMBER_BYTES:
                    _invalid("wheel member exceeds the size limit")
                total_size += info.file_size
                if total_size > MAX_TOTAL_MEMBER_BYTES:
                    _invalid("wheel contents exceed the size limit")

                _reject_sensitive_member(parts, is_dir=info.is_dir())
                if parts[0] not in {"emiya_agent", expected.dist_info}:
                    _invalid("wheel contains an unexpected top-level path")
                members[info.filename] = info

            if not required.issubset(members):
                _invalid("wheel is missing package, metadata, or entry-point files")

            source_files = _source_package_files(package_root)
            wheel_files = {
                name: info
                for name, info in members.items()
                if name.startswith("emiya_agent/") and not info.is_dir()
            }
            if wheel_files.keys() != source_files.keys():
                _invalid("wheel package file set does not match the source tree")
            for name in sorted(source_files):
                info = wheel_files[name]
                source_size, source_digest = _source_digest(source_files[name])
                if (
                    source_size != info.file_size
                    or _zip_digest(archive, info, label="wheel package member")
                    != source_digest
                ):
                    _invalid("wheel package contents do not match the source tree")

            metadata_info = members[metadata_name]
            with archive.open(metadata_info, mode="r") as stream:
                metadata_data = _bounded_read(
                    stream, metadata_info.file_size, label="wheel METADATA"
                )
            name, version = _parse_core_metadata(metadata_data, label="wheel METADATA")
            _verify_name_and_version(name, version, expected, label="wheel METADATA")
            _verify_distribution_metadata_fields(
                metadata_data, expected, label="wheel METADATA"
            )

            license_info = members[license_name]
            source_license = package_root.parent / LICENSE_FILENAME
            source_size, source_digest = _source_digest(source_license)
            if (
                source_size != license_info.file_size
                or _zip_digest(archive, license_info, label="wheel license file")
                != source_digest
            ):
                _invalid("wheel license file does not match the source tree")

            descriptor_info = members[wheel_descriptor_name]
            with archive.open(descriptor_info, mode="r") as stream:
                descriptor_data = _bounded_read(
                    stream, descriptor_info.file_size, label="wheel WHEEL metadata"
                )
            _verify_wheel_descriptor(descriptor_data)

            entry_info = members[entry_points_name]
            with archive.open(entry_info, mode="r") as stream:
                entry_data = _bounded_read(
                    stream, entry_info.file_size, label="wheel entry_points.txt"
                )
            try:
                entry_text = entry_data.decode("utf-8")
                parser = configparser.ConfigParser(interpolation=None, strict=True)
                parser.optionxform = str
                parser.read_string(entry_text)
            except (UnicodeDecodeError, configparser.Error) as exc:
                raise ArtifactValidationError(
                    "wheel entry_points.txt is malformed"
                ) from exc
            if parser.defaults() or set(parser.sections()) != {"console_scripts"}:
                _invalid("wheel contains unexpected entry-point groups")
            actual_scripts = dict(parser.items("console_scripts", raw=True))
            if actual_scripts != expected.scripts:
                _invalid("wheel CLI entry points do not match pyproject.toml")

            init_info = members[init_name]
            with archive.open(init_info, mode="r") as stream:
                init_data = _bounded_read(
                    stream, init_info.file_size, label="wheel emiya_agent/__init__.py"
                )
            try:
                init_text = init_data.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ArtifactValidationError(
                    "wheel emiya_agent/__init__.py is not UTF-8"
                ) from exc
            if (
                _literal_version(init_text, label="wheel emiya_agent/__init__.py")
                != expected.version
            ):
                _invalid("wheel package version does not match pyproject.toml")
            _verify_wheel_record(archive, members, record_name=record_name)
    except ArtifactValidationError:
        raise
    except (OSError, RuntimeError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        raise ArtifactValidationError("wheel cannot be read safely") from exc


def _tar_header_size(header: bytes) -> int:
    raw_size = header[124:136]
    if raw_size[:1] and raw_size[0] & 0x80:
        _invalid("sdist uses an unsupported binary tar size")
    stripped = raw_size.rstrip(b"\0 ").lstrip(b" ")
    if not stripped:
        return 0
    if any(value < ord("0") or value > ord("7") for value in stripped):
        _invalid("sdist contains an invalid tar size")
    return int(stripped, 8)


def _preflight_tar_stream(path: Path) -> None:
    """Bound gzip expansion and extended tar headers before tarfile parses them."""

    total_read = 0

    def read_exact(
        stream: BinaryIO,
        size: int,
        *,
        allow_eof: bool = False,
        collect: bool = True,
    ) -> bytes:
        nonlocal total_read
        chunks: list[bytes] = []
        remaining = size
        while remaining:
            chunk = stream.read(min(64 * 1024, remaining))
            if not chunk:
                if allow_eof and remaining == size:
                    return b""
                _invalid("sdist tar stream is truncated")
            if collect:
                chunks.append(chunk)
            remaining -= len(chunk)
            total_read += len(chunk)
            if total_read > MAX_TAR_STREAM_BYTES:
                _invalid("sdist expanded stream exceeds the size limit")
        return b"".join(chunks)

    try:
        with gzip.open(path, mode="rb") as stream:
            header_count = 0
            while True:
                header = read_exact(stream, 512, allow_eof=True)
                if not header:
                    _invalid("sdist tar stream has no end marker")
                if header == bytes(512):
                    while True:
                        trailing = stream.read(64 * 1024)
                        if not trailing:
                            return
                        total_read += len(trailing)
                        if total_read > MAX_TAR_STREAM_BYTES:
                            _invalid("sdist expanded stream exceeds the size limit")
                        if trailing.strip(b"\0"):
                            _invalid("sdist contains data after its end marker")

                header_count += 1
                if header_count > MAX_MEMBERS * 3:
                    _invalid("sdist has too many raw tar headers")
                size = _tar_header_size(header)
                type_flag = header[156:157]
                if type_flag in {b"x", b"g", b"L", b"K"} and size > MAX_METADATA_BYTES:
                    _invalid("sdist extended tar header exceeds the size limit")
                if type_flag == b"S":
                    _invalid("sdist contains an unsupported sparse tar member")
                if (
                    type_flag not in {b"x", b"g", b"L", b"K"}
                    and size > MAX_MEMBER_BYTES
                ):
                    _invalid("sdist raw tar member exceeds the size limit")
                padded_size = (size + 511) // 512 * 512
                read_exact(stream, padded_size, collect=False)
    except ArtifactValidationError:
        raise
    except (OSError, EOFError, gzip.BadGzipFile) as exc:
        raise ArtifactValidationError("sdist compressed stream is invalid") from exc


def _verify_sdist(path: Path, expected: ProjectMetadata, *, project_root: Path) -> None:
    expected_filename = f"{expected.archive_root}.tar.gz"
    if path.name != expected_filename:
        _invalid("sdist filename does not match project metadata")

    required_suffixes = {
        "pyproject": "pyproject.toml",
        "init": "emiya_agent/__init__.py",
        "metadata": "PKG-INFO",
        "license": LICENSE_FILENAME,
    }
    if expected.readme is not None:
        required_suffixes["readme"] = expected.readme
    members: dict[str, tarfile.TarInfo] = {}
    canonical_members: set[tuple[str, ...]] = set()
    roots: set[str] = set()
    total_size = 0

    try:
        _preflight_tar_stream(path)
        with tarfile.open(path, mode="r:gz") as archive:
            for count, info in enumerate(archive, start=1):
                if count > MAX_MEMBERS:
                    _invalid("sdist has too many members")
                parts = _safe_member_parts(info.name)
                if info.name in members:
                    _invalid("sdist contains duplicate member names")
                canonical = _canonical_member(parts)
                if canonical in canonical_members:
                    _invalid("sdist contains cross-platform duplicate member names")
                canonical_members.add(canonical)
                if not (info.isfile() or info.isdir()) or info.issparse():
                    _invalid("sdist contains a link, device, or special member")
                if info.size < 0 or info.size > MAX_MEMBER_BYTES:
                    _invalid("sdist member exceeds the size limit")
                total_size += info.size
                if total_size > MAX_TOTAL_MEMBER_BYTES:
                    _invalid("sdist contents exceed the size limit")

                _reject_sensitive_member(parts, is_dir=info.isdir())
                roots.add(parts[0])
                if len(roots) > 1:
                    _invalid("sdist does not have a single root directory")
                if len(parts) == 1 and not info.isdir():
                    _invalid("sdist contains a file outside its root directory")
                members[info.name] = info

            if not members or roots != {expected.archive_root}:
                _invalid("sdist root directory does not match project metadata")

            names = {
                key: f"{expected.archive_root}/{suffix}"
                for key, suffix in required_suffixes.items()
            }
            if any(name not in members for name in names.values()):
                _invalid("sdist is missing package or project metadata")

            extracted: dict[str, bytes] = {}
            for key, name in names.items():
                info = members[name]
                if not info.isfile():
                    _invalid("sdist metadata path is not a regular file")
                stream = archive.extractfile(info)
                if stream is None:
                    _invalid("sdist metadata cannot be read")
                with stream:
                    extracted[key] = _bounded_read(
                        stream, info.size, label=f"sdist {required_suffixes[key]}"
                    )

            sdist_project = _metadata_from_pyproject(
                extracted["pyproject"], label="sdist pyproject.toml"
            )
            if sdist_project != expected:
                _invalid("sdist pyproject.toml does not match the source tree")
            try:
                init_text = extracted["init"].decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ArtifactValidationError(
                    "sdist emiya_agent/__init__.py is not UTF-8"
                ) from exc
            if (
                _literal_version(init_text, label="sdist emiya_agent/__init__.py")
                != expected.version
            ):
                _invalid("sdist package version does not match pyproject.toml")
            name, version = _parse_core_metadata(
                extracted["metadata"], label="sdist PKG-INFO"
            )
            _verify_name_and_version(name, version, expected, label="sdist PKG-INFO")
            _verify_distribution_metadata_fields(
                extracted["metadata"], expected, label="sdist PKG-INFO"
            )

            source_package_files = _source_package_files(project_root / "emiya_agent")
            expected_package_members = {
                f"{expected.archive_root}/{name}": source_path
                for name, source_path in source_package_files.items()
            }
            actual_package_members = {
                name: info
                for name, info in members.items()
                if name.startswith(f"{expected.archive_root}/emiya_agent/")
                and info.isfile()
            }
            if actual_package_members.keys() != expected_package_members.keys():
                _invalid("sdist package file set does not match the source tree")

            generated_metadata_name = f"{expected.archive_root}/PKG-INFO"
            for member_name, info in sorted(
                members.items(), key=lambda item: item[1].offset_data
            ):
                if not info.isfile() or member_name == generated_metadata_name:
                    continue
                parts = _safe_member_parts(member_name)[1:]
                source_path = _source_file_for_member(project_root, parts)
                source_size, source_digest = _source_digest(source_path)
                stream = archive.extractfile(info)
                if stream is None:
                    _invalid("sdist member cannot be read")
                with stream:
                    artifact_digest = _stream_digest(
                        stream, info.size, label="sdist member"
                    )
                if source_size != info.size or artifact_digest != source_digest:
                    _invalid("sdist contents do not match the source tree")
    except ArtifactValidationError:
        raise
    except (OSError, EOFError, tarfile.TarError) as exc:
        raise ArtifactValidationError("sdist cannot be read safely") from exc


def verify_release_artifacts(dist_dir: Path, *, project_root: Path) -> ProjectMetadata:
    """Validate the only wheel and sdist in *dist_dir* without extracting them."""

    expected = _load_project_metadata(project_root)
    try:
        dist_info = dist_dir.stat(follow_symlinks=False)
    except OSError as exc:
        raise ArtifactValidationError("dist directory cannot be inspected") from exc
    if _is_link_or_reparse(dist_info) or not stat.S_ISDIR(dist_info.st_mode):
        _invalid("dist directory must not be a link or reparse point")
    try:
        entries = list(os.scandir(dist_dir))
    except OSError as exc:
        raise ArtifactValidationError("dist directory cannot be read") from exc
    placeholders = [entry for entry in entries if entry.name == ".gitignore"]
    if len(placeholders) > 1:
        _invalid("dist contains duplicate placeholder files")
    if placeholders:
        placeholder = Path(placeholders[0].path)
        _validate_archive_path(placeholder)
        try:
            placeholder_data = placeholder.read_bytes()
        except OSError as exc:
            raise ArtifactValidationError("dist placeholder cannot be read") from exc
        if placeholder_data not in {b"*", b"*\n", b"*\r\n"}:
            _invalid("dist .gitignore is not the expected build placeholder")

    paths = [Path(entry.path) for entry in entries if entry.name != ".gitignore"]
    if len(paths) != 2:
        _invalid("dist must contain exactly one wheel and one sdist")
    for path in paths:
        _validate_archive_path(path)
    wheels = [path for path in paths if path.name.endswith(".whl")]
    sdists = [path for path in paths if path.name.endswith(".tar.gz")]
    if len(wheels) != 1 or len(sdists) != 1:
        _invalid("dist must contain exactly one wheel and one .tar.gz sdist")

    _verify_wheel(wheels[0], expected, package_root=project_root / "emiya_agent")
    _verify_sdist(sdists[0], expected, project_root=project_root)
    return expected


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline safety and metadata checks for release artifacts."
    )
    parser.add_argument(
        "--dist-dir",
        type=Path,
        default=None,
        help="directory containing exactly one wheel and one .tar.gz sdist",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    project_root = Path(__file__).resolve().parents[1]
    args = _parser().parse_args(argv)
    dist_dir = args.dist_dir if args.dist_dir is not None else project_root / "dist"
    try:
        metadata = verify_release_artifacts(dist_dir, project_root=project_root)
    except ArtifactValidationError as exc:
        print(f"release artifact verification failed: {exc}", file=sys.stderr)
        return EXIT_INVALID
    except Exception:
        # Do not expose archive contents or exception details in release logs.
        print("release artifact verification failed: internal error", file=sys.stderr)
        return EXIT_INTERNAL_ERROR

    print(f"release artifacts verified: {metadata.name} {metadata.version}")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
