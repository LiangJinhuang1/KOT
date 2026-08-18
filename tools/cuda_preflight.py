#!/usr/bin/env python3
"""Small CUDA/container sanity check for Slurm GPU jobs."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from ctypes import byref, c_char_p, c_int, CDLL
from pathlib import Path

import torch


def run_nvidia_smi(args: list[str]) -> str:
    proc = subprocess.run(
        ["nvidia-smi", *args],
        check=False,
        capture_output=True,
        text=True,
    )
    return (proc.stdout or proc.stderr).strip()


def parse_cuda_version(version: str | None) -> tuple[int, int]:
    if not version:
        return (0, 0)
    match = re.match(r"^(\d+)\.(\d+)", str(version))
    return tuple(map(int, match.groups())) if match else (0, 0)


def cuda_error_text(libcuda, code: int) -> str:
    name = c_char_p()
    message = c_char_p()
    libcuda.cuGetErrorName(c_int(code), byref(name))
    libcuda.cuGetErrorString(c_int(code), byref(message))
    parts = []
    if name.value:
        parts.append(name.value.decode("utf-8", errors="replace"))
    if message.value:
        parts.append(message.value.decode("utf-8", errors="replace"))
    return ": ".join(parts) if parts else "<unknown CUDA error>"


def cuda_driver_init_status() -> tuple[int, str]:
    libcuda = CDLL("libcuda.so.1")
    code = int(libcuda.cuInit(0))
    if code == 0:
        return code, "CUDA driver cuInit(0) succeeded"
    return code, cuda_error_text(libcuda, code)


def main() -> int:
    print("python:", sys.executable)
    print("cwd:", os.getcwd())
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"))
    print("NVIDIA_VISIBLE_DEVICES:", os.environ.get("NVIDIA_VISIBLE_DEVICES", "<unset>"))
    print("SINGULARITYENV_CUDA_VISIBLE_DEVICES:", os.environ.get("SINGULARITYENV_CUDA_VISIBLE_DEVICES", "<unset>"))
    print("APPTAINERENV_CUDA_VISIBLE_DEVICES:", os.environ.get("APPTAINERENV_CUDA_VISIBLE_DEVICES", "<unset>"))
    print("SLURM_JOB_GPUS:", os.environ.get("SLURM_JOB_GPUS", "<unset>"))
    print("SLURM_STEP_GPUS:", os.environ.get("SLURM_STEP_GPUS", "<unset>"))
    print("PYTORCH_ALLOC_CONF:", os.environ.get("PYTORCH_ALLOC_CONF", "<unset>"))
    print("/dev/nvidia*:")
    for path in sorted(Path("/dev").glob("nvidia*")):
        print(path)

    print("nvidia-smi -L:")
    print(run_nvidia_smi(["-L"]) or "<empty>")
    gpu_query = run_nvidia_smi(
        [
            "--query-gpu=name,compute_cap,memory.total,driver_version",
            "--format=csv,noheader",
        ]
    )
    print("nvidia-smi query:")
    print(gpu_query or "<empty>")
    print("nvidia-smi fabric detail:")
    print(run_nvidia_smi(["-q", "-d", "FABRIC"]) or "<empty>")
    print("nvidia-smi topology:")
    print(run_nvidia_smi(["topo", "-m"]) or "<empty>")

    cuinit_code, cuinit_message = cuda_driver_init_status()
    print("CUDA driver cuInit(0):", cuinit_code, cuinit_message)

    print("torch:", torch.__version__)
    print("torch.version.cuda:", torch.version.cuda)

    if "B200" in gpu_query and parse_cuda_version(torch.version.cuda) < (12, 8):
        print(
            "ERROR: This job is on B200/Blackwell, but PyTorch was built with "
            f"CUDA {torch.version.cuda}. Use a CUDA 12.8+ PyTorch container.",
            file=sys.stderr,
        )
        return 42

    cuda_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count()
    print("torch.cuda.is_available:", cuda_available)
    print("torch.cuda.device_count:", device_count)
    if device_count > 0:
        print("torch.cuda.get_device_name(0):", torch.cuda.get_device_name(0))

    if gpu_query and (not cuda_available or device_count < 1):
        print(
            "ERROR: nvidia-smi sees a GPU, but PyTorch sees no usable CUDA device. "
            "This is a Slurm/Singularity GPU visibility, /dev/nvidia device, or "
            "B200 NVSwitch/Fabric Manager initialization issue.",
            file=sys.stderr,
        )
        return 44

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
