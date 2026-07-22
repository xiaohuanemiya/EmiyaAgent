from __future__ import annotations

import base64
import hashlib
import io
import tarfile
import zipfile
from pathlib import Path

import pytest

import scripts.verify_release_artifacts as release_verifier
from scripts.verify_release_artifacts import (
    ArtifactValidationError,
    ProjectMetadata,
    _safe_member_parts,
    verify_release_artifacts,
)

NAME = "emiya-agent"
VERSION = "0.1.0"
WHEEL_NAME = "emiya_agent"
ROOT_NAME = f"{WHEEL_NAME}-{VERSION}"
DIST_INFO = f"{WHEEL_NAME}-{VERSION}.dist-info"
LICENSE_TEXT = b"MIT License\n\nCopyright (c) test fixture\n"


def _write_project(root: Path) -> None:
    package = root / "emiya_agent"
    package.mkdir(parents=True)
    (root / "README.md").write_bytes(b"# fixture\n")
    (root / "LICENSE").write_bytes(LICENSE_TEXT)
    (package / "__init__.py").write_bytes(f'__version__ = "{VERSION}"\n'.encode())
    (package / "cli.py").write_bytes(b"def main():\n    return 0\n")
    (root / "pyproject.toml").write_bytes(_sdist_pyproject())


def _core_metadata(
    version: str = VERSION, *, extra_headers: tuple[str, ...] = ()
) -> bytes:
    return (
        "Metadata-Version: 2.4\n"
        f"Name: {NAME}\n"
        f"Version: {version}\n"
        "License-Expression: MIT\n"
        "License-File: LICENSE\n"
        + "".join(f"{header}\n" for header in extra_headers)
        + "\n"
    ).encode()


def _sdist_pyproject(version: str = VERSION) -> bytes:
    return (
        "[project]\n"
        f'name = "{NAME}"\n'
        f'version = "{version}"\n'
        'readme = "README.md"\n'
        'license = "MIT"\n'
        "\n"
        "[project.scripts]\n"
        'emiya = "emiya_agent.cli:main"\n'
    ).encode()


def _write_wheel(
    dist: Path,
    *,
    metadata_version: str = VERSION,
    extra_metadata_headers: tuple[str, ...] = (),
    extra_members: dict[str, bytes] | None = None,
    record_override: bytes | None = None,
) -> Path:
    path = dist / f"{WHEEL_NAME}-{VERSION}-py3-none-any.whl"
    members = {
        "emiya_agent/__init__.py": f'__version__ = "{VERSION}"\n'.encode(),
        "emiya_agent/cli.py": b"def main():\n    return 0\n",
        f"{DIST_INFO}/METADATA": _core_metadata(
            metadata_version, extra_headers=extra_metadata_headers
        ),
        f"{DIST_INFO}/entry_points.txt": (
            b"[console_scripts]\nemiya = emiya_agent.cli:main\n"
        ),
        f"{DIST_INFO}/WHEEL": (
            b"Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
        ),
        f"{DIST_INFO}/licenses/LICENSE": LICENSE_TEXT,
    }
    members.update(extra_members or {})
    record_name = f"{DIST_INFO}/RECORD"
    record_rows = [
        (
            f"{name},sha256="
            f"{base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b'=').decode()}"
            f",{len(data)}\n"
        )
        for name, data in sorted(members.items())
    ]
    members[record_name] = (
        record_override
        if record_override is not None
        else "".join([*record_rows, f"{record_name},,\n"]).encode()
    )
    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return path


def _add_tar_bytes(archive: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.mode = 0o644
    info.size = len(data)
    archive.addfile(info, io.BytesIO(data))


def _write_sdist(
    dist: Path,
    *,
    extra_members: dict[str, bytes] | None = None,
    omit_members: set[str] | None = None,
    special_member: tarfile.TarInfo | None = None,
) -> Path:
    path = dist / f"{ROOT_NAME}.tar.gz"
    members = {
        f"{ROOT_NAME}/pyproject.toml": _sdist_pyproject(),
        f"{ROOT_NAME}/emiya_agent/__init__.py": (
            f'__version__ = "{VERSION}"\n'.encode()
        ),
        f"{ROOT_NAME}/emiya_agent/cli.py": b"def main():\n    return 0\n",
        f"{ROOT_NAME}/PKG-INFO": _core_metadata(),
        f"{ROOT_NAME}/README.md": b"# fixture\n",
        f"{ROOT_NAME}/LICENSE": LICENSE_TEXT,
    }
    members.update(extra_members or {})
    for member_name in omit_members or set():
        members.pop(member_name, None)
    with tarfile.open(path, mode="w:gz") as archive:
        root = tarfile.TarInfo(ROOT_NAME)
        root.type = tarfile.DIRTYPE
        root.mode = 0o755
        archive.addfile(root)
        for name, data in members.items():
            _add_tar_bytes(archive, name, data)
        if special_member is not None:
            archive.addfile(special_member)
    return path


def _build_valid_release(tmp_path: Path) -> tuple[Path, Path]:
    project = tmp_path / "project"
    dist = project / "dist"
    dist.mkdir(parents=True)
    _write_project(project)
    _write_wheel(dist)
    _write_sdist(dist)
    return project, dist


def test_valid_release_with_dist_placeholder_passes(tmp_path: Path) -> None:
    project, dist = _build_valid_release(tmp_path)
    (dist / ".gitignore").write_bytes(b"*")

    metadata = verify_release_artifacts(dist, project_root=project)

    assert metadata.name == NAME
    assert metadata.version == VERSION
    assert metadata.scripts == {"emiya": "emiya_agent.cli:main"}
    assert metadata.license_expression == "MIT"


def test_project_requires_mit_license_metadata(tmp_path: Path) -> None:
    project, dist = _build_valid_release(tmp_path)
    pyproject = project / "pyproject.toml"
    pyproject.write_bytes(pyproject.read_bytes().replace(b'license = "MIT"\n', b""))

    with pytest.raises(ArtifactValidationError, match="MIT license expression"):
        verify_release_artifacts(dist, project_root=project)


def test_wheel_rejects_license_content_drift(tmp_path: Path) -> None:
    project, dist = _build_valid_release(tmp_path)
    _write_wheel(
        dist,
        extra_members={f"{DIST_INFO}/licenses/LICENSE": b"not the source license\n"},
    )

    with pytest.raises(ArtifactValidationError, match="license file does not match"):
        verify_release_artifacts(dist, project_root=project)


def test_wheel_rejects_incorrect_license_expression(tmp_path: Path) -> None:
    project, dist = _build_valid_release(tmp_path)
    _write_wheel(
        dist,
        extra_metadata_headers=("License-Expression: Apache-2.0",),
    )

    with pytest.raises(ArtifactValidationError, match="License-Expression"):
        verify_release_artifacts(dist, project_root=project)


def test_sdist_requires_license_file(tmp_path: Path) -> None:
    project, dist = _build_valid_release(tmp_path)
    _write_sdist(dist, omit_members={f"{ROOT_NAME}/LICENSE"})

    with pytest.raises(ArtifactValidationError, match="missing package or project"):
        verify_release_artifacts(dist, project_root=project)


def test_wheel_rejects_malicious_member_paths(tmp_path: Path) -> None:
    project, dist = _build_valid_release(tmp_path)
    _write_wheel(dist, extra_members={"../outside.txt": b"unsafe"})

    with pytest.raises(ArtifactValidationError, match="unsafe member path"):
        verify_release_artifacts(dist, project_root=project)


@pytest.mark.parametrize("member_name", ["bad\\path.txt", "bad\npath.txt"])
def test_member_path_parser_rejects_backslashes_and_controls(
    member_name: str,
) -> None:
    with pytest.raises(ArtifactValidationError, match="unsafe member path"):
        _safe_member_parts(member_name)


@pytest.mark.parametrize(
    "member_name",
    ["bad?.py", "bad*.py", 'bad".py', "bad<.py", "bad|.py", "bad\ud800.py"],
)
def test_member_path_parser_rejects_nonportable_windows_names(
    member_name: str,
) -> None:
    with pytest.raises(ArtifactValidationError, match="unsafe|non-portable"):
        _safe_member_parts(f"emiya_agent/{member_name}")


def test_member_path_parser_rejects_long_paths() -> None:
    with pytest.raises(ArtifactValidationError, match="too long"):
        _safe_member_parts("emiya_agent/" + "/".join(["segment"] * 40) + ".py")


@pytest.mark.parametrize("member_name", ["safe/．．", "safe/name．"])
def test_member_path_parser_rejects_unsafe_nfkc_forms(member_name: str) -> None:
    with pytest.raises(ArtifactValidationError, match="non-portable member path"):
        _safe_member_parts(member_name)


@pytest.mark.parametrize("member_name", ["CON", "con.txt", "COM1.py", "LPT9"])
def test_member_path_parser_rejects_windows_device_names(
    member_name: str,
) -> None:
    with pytest.raises(ArtifactValidationError, match="Windows-reserved"):
        _safe_member_parts(f"emiya_agent/{member_name}")


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("emiya_agent/Readme.txt", "emiya_agent/README.TXT"),
        ("emiya_agent/Kelvin-K.py", "emiya_agent/Kelvin-K.py"),
    ],
)
def test_wheel_rejects_cross_platform_duplicate_names(
    tmp_path: Path, first: str, second: str
) -> None:
    project, dist = _build_valid_release(tmp_path)
    _write_wheel(dist, extra_members={first: b"one", second: b"two"})

    with pytest.raises(ArtifactValidationError, match="cross-platform duplicate"):
        verify_release_artifacts(dist, project_root=project)


def test_sdist_rejects_runtime_or_sensitive_files(tmp_path: Path) -> None:
    project, dist = _build_valid_release(tmp_path)
    _write_sdist(
        dist,
        extra_members={
            f"{ROOT_NAME}/.emiyacode/sessions/turns.jsonl": b"private runtime data"
        },
    )

    with pytest.raises(ArtifactValidationError, match="runtime or sensitive"):
        verify_release_artifacts(dist, project_root=project)


def test_wheel_rejects_metadata_version_mismatch(tmp_path: Path) -> None:
    project, dist = _build_valid_release(tmp_path)
    _write_wheel(dist, metadata_version="9.9.9")

    with pytest.raises(ArtifactValidationError, match="version does not match"):
        verify_release_artifacts(dist, project_root=project)


def test_wheel_requires_core_metadata_version(tmp_path: Path) -> None:
    project, dist = _build_valid_release(tmp_path)
    _write_wheel(
        dist,
        extra_members={
            f"{DIST_INFO}/METADATA": (
                f"Name: {NAME}\nVersion: {VERSION}\n\n"
            ).encode()
        },
    )

    with pytest.raises(ArtifactValidationError, match="Metadata-Version 2.4"):
        verify_release_artifacts(dist, project_root=project)


def test_wheel_rejects_injected_dependency_metadata(tmp_path: Path) -> None:
    project, dist = _build_valid_release(tmp_path)
    _write_wheel(
        dist,
        extra_metadata_headers=(
            "Requires-Dist: attacker @ https://example.invalid/payload.whl",
        ),
    )

    with pytest.raises(ArtifactValidationError, match="dependencies do not match"):
        verify_release_artifacts(dist, project_root=project)


def test_wheel_rejects_extra_entry_point(tmp_path: Path) -> None:
    project, dist = _build_valid_release(tmp_path)
    _write_wheel(
        dist,
        extra_members={
            f"{DIST_INFO}/entry_points.txt": (
                b"[console_scripts]\n"
                b"emiya = emiya_agent.cli:main\n"
                b"unexpected = emiya_agent.cli:main\n"
            )
        },
    )

    with pytest.raises(ArtifactValidationError, match="entry points"):
        verify_release_artifacts(dist, project_root=project)


def test_wheel_rejects_incorrect_record(tmp_path: Path) -> None:
    project, dist = _build_valid_release(tmp_path)
    _write_wheel(
        dist,
        record_override=(
            f"emiya_agent/__init__.py,sha256={'A' * 43},22\n{DIST_INFO}/RECORD,,\n"
        ).encode(),
    )

    with pytest.raises(ArtifactValidationError, match="RECORD"):
        verify_release_artifacts(dist, project_root=project)


def test_wheel_rejects_an_extra_package_file(tmp_path: Path) -> None:
    project, dist = _build_valid_release(tmp_path)
    _write_wheel(dist, extra_members={"emiya_agent/backdoor.py": b"injected = True\n"})

    with pytest.raises(ArtifactValidationError, match="file set"):
        verify_release_artifacts(dist, project_root=project)


def test_wheel_rejects_package_content_drift(tmp_path: Path) -> None:
    project, dist = _build_valid_release(tmp_path)
    _write_wheel(
        dist,
        extra_members={"emiya_agent/cli.py": b"def main():\n    return 99\n"},
    )

    with pytest.raises(ArtifactValidationError, match="contents do not match"):
        verify_release_artifacts(dist, project_root=project)


def test_sdist_rejects_package_content_drift(tmp_path: Path) -> None:
    project, dist = _build_valid_release(tmp_path)
    _write_sdist(
        dist,
        extra_members={
            f"{ROOT_NAME}/emiya_agent/cli.py": b"def main():\n    return 99\n"
        },
    )

    with pytest.raises(ArtifactValidationError, match="contents do not match"):
        verify_release_artifacts(dist, project_root=project)


def test_sdist_rejects_changed_build_configuration(tmp_path: Path) -> None:
    project, dist = _build_valid_release(tmp_path)
    changed_pyproject = _sdist_pyproject() + (
        b"\n[build-system]\n"
        b'requires = ["malicious-backend"]\n'
        b'build-backend = "malicious_backend"\n'
    )
    _write_sdist(
        dist,
        extra_members={f"{ROOT_NAME}/pyproject.toml": changed_pyproject},
    )

    with pytest.raises(ArtifactValidationError, match="contents do not match"):
        verify_release_artifacts(dist, project_root=project)


def test_sdist_rejects_injected_dependency_metadata(tmp_path: Path) -> None:
    project, dist = _build_valid_release(tmp_path)
    _write_sdist(
        dist,
        extra_members={
            f"{ROOT_NAME}/PKG-INFO": _core_metadata(
                extra_headers=("Requires-Dist: attacker>=1",)
            )
        },
    )

    with pytest.raises(ArtifactValidationError, match="dependencies do not match"):
        verify_release_artifacts(dist, project_root=project)


def test_sdist_rejects_file_absent_from_source_tree(tmp_path: Path) -> None:
    project, dist = _build_valid_release(tmp_path)
    _write_sdist(
        dist,
        extra_members={f"{ROOT_NAME}/unexpected.txt": b"not in source\n"},
    )

    with pytest.raises(ArtifactValidationError, match="not present in the source"):
        verify_release_artifacts(dist, project_root=project)


def test_sdist_requires_declared_readme(tmp_path: Path) -> None:
    project, dist = _build_valid_release(tmp_path)
    _write_sdist(
        dist,
        omit_members={f"{ROOT_NAME}/README.md"},
    )

    with pytest.raises(ArtifactValidationError, match="missing package or project"):
        verify_release_artifacts(dist, project_root=project)


def test_sdist_rejects_links(tmp_path: Path) -> None:
    project, dist = _build_valid_release(tmp_path)
    link = tarfile.TarInfo(f"{ROOT_NAME}/linked-secret")
    link.type = tarfile.SYMTYPE
    link.linkname = "../../outside"
    _write_sdist(dist, special_member=link)

    with pytest.raises(ArtifactValidationError, match="link, device, or special"):
        verify_release_artifacts(dist, project_root=project)


def test_sdist_preflight_rejects_oversized_pax_header(tmp_path: Path) -> None:
    path = tmp_path / "oversized-pax.tar.gz"
    info = tarfile.TarInfo("safe.txt")
    info.pax_headers = {"comment": "x" * (release_verifier.MAX_METADATA_BYTES + 1)}
    with tarfile.open(path, mode="w:gz", format=tarfile.PAX_FORMAT) as archive:
        archive.addfile(info)

    with pytest.raises(ArtifactValidationError, match="extended tar header"):
        release_verifier._preflight_tar_stream(path)


def test_unknown_dist_file_is_rejected(tmp_path: Path) -> None:
    project, dist = _build_valid_release(tmp_path)
    (dist / "build-notes.txt").write_text("unexpected", encoding="utf-8")

    with pytest.raises(
        ArtifactValidationError, match="exactly one wheel and one sdist"
    ):
        verify_release_artifacts(dist, project_root=project)


def test_dist_placeholder_must_be_the_expected_regular_file(tmp_path: Path) -> None:
    project, dist = _build_valid_release(tmp_path)
    (dist / ".gitignore").write_text("secrets.txt\n", encoding="utf-8")

    with pytest.raises(ArtifactValidationError, match="expected build placeholder"):
        verify_release_artifacts(dist, project_root=project)


def test_cli_returns_stable_validation_exit_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail_validation(*_args: object, **_kwargs: object) -> ProjectMetadata:
        raise ArtifactValidationError("invalid archive")

    monkeypatch.setattr(release_verifier, "verify_release_artifacts", fail_validation)

    assert release_verifier.main(["--dist-dir", str(tmp_path)]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ("release artifact verification failed: invalid archive\n")


def test_cli_internal_error_does_not_print_exception_details(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail_internally(*_args: object, **_kwargs: object) -> ProjectMetadata:
        raise RuntimeError("do-not-print-this-value")

    monkeypatch.setattr(release_verifier, "verify_release_artifacts", fail_internally)

    assert release_verifier.main(["--dist-dir", str(tmp_path)]) == 3
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "release artifact verification failed: internal error\n"
