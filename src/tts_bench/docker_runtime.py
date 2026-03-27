from __future__ import annotations

from pathlib import Path
from pathlib import PurePosixPath
import os
import subprocess

_REPO_ROOT = Path.cwd().resolve()
_CONTAINER_ROOT = PurePosixPath("/workspace")


def ensure_image(image_tag: str, docker_dir: Path) -> None:
    _check_docker_ready()
    inspect = subprocess.run(
        ["docker", "image", "inspect", image_tag],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if inspect.returncode == 0:
        return

    result = subprocess.run(
        ["docker", "build", "-t", image_tag, str(docker_dir)],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Docker build failed for {image_tag}: {_tail(result.stderr or result.stdout)}"
        )


def run_container(
    image_tag: str,
    env: dict[str, str],
    wants_gpu: bool,
    mount_target: str = "/workspace",
    workdir: str | None = "/workspace",
    entrypoint: str | None = None,
    command_args: list[str] | None = None,
) -> None:
    _check_docker_ready()
    volume = f"{_REPO_ROOT}:{mount_target}"
    command = [
        "docker",
        "run",
        "--rm",
        "-v",
        volume,
        "--shm-size",
        "8g",
        "-e",
        f"HF_HOME={mount_target}/.cache/huggingface",
        "-e",
        f"TRANSFORMERS_CACHE={mount_target}/.cache/huggingface/hub",
        "-e",
        f"TORCH_HOME={mount_target}/.cache/torch",
        "-e",
        f"XDG_CACHE_HOME={mount_target}/.cache",
        "-e",
        "HF_HUB_DISABLE_SYMLINKS_WARNING=1",
    ]
    if workdir:
        command.extend(["-w", workdir])
    if wants_gpu:
        command.extend(["--gpus", "all"])
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = os.environ.get(key, "").strip()
        if value:
            command.extend(["-e", f"{key}={value}"])
    for key, value in env.items():
        command.extend(["-e", f"{key}={value}"])
    if entrypoint:
        command.extend(["--entrypoint", entrypoint])
    command.append(image_tag)
    if command_args:
        command.extend(command_args)

    result = subprocess.run(
        command,
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Docker run failed for {image_tag}: {_tail(result.stderr or result.stdout)}"
        )


def container_path(path: Path, mount_target: str = "/workspace") -> str:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(_REPO_ROOT)
    except ValueError as exc:
        raise RuntimeError(f"Path is outside the repository workspace: {resolved}") from exc
    container_root = PurePosixPath(mount_target)
    return str(container_root.joinpath(*relative.parts))


def _check_docker_ready() -> None:
    result = subprocess.run(
        ["docker", "info"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Docker is unavailable: {_tail(result.stderr)}"
        )


def _tail(value: str, limit: int = 400) -> str:
    cleaned = " ".join(value.strip().split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[-limit:]
