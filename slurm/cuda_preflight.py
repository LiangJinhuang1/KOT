"""Refuse to start on a broken CUDA allocation.

Exit 42 is the container (do not requeue); 44/45 are this node.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

import torch

EXIT_CONTAINER_MISMATCH = 42
EXIT_NO_CUDA = 44
EXIT_CUDA_RAISED = 45

MIN_BLACKWELL_CUDA = (12, 8)


def gpu_report() -> str:
    """nvidia-smi listing, even when CUDA does not work.

    A missing binary is not a verdict and must not take the preflight down with an unrecognized exit code.
    """
    try:
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,compute_cap,memory.total,driver_version",
             "--format=csv,noheader"],
            check=False, capture_output=True, text=True,
        )
    except OSError as exc:
        print("nvidia-smi query failed:", repr(exc))
        return ""
    return (smi.stdout.strip() or smi.stderr.strip() or "<empty>")


def parse_cuda_version(version) -> tuple:
    match = re.match(r"^(\d+)\.(\d+)", str(version or ""))
    return tuple(map(int, match.groups())) if match else ()


def usable_devices(count: int) -> list[int]:
    """The devices that survive a real allocation, reported one line each."""
    usable = []
    for device in range(count):
        try:
            torch.empty(1, device=f"cuda:{device}")
        except Exception as exc:
            print(f"cuda:{device} UNUSABLE: {exc!r}", file=sys.stderr)
            continue
        print(f"cuda:{device} smoke allocation: ok")
        usable.append(device)
    return usable


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emit-usable", type=Path,
                        help="Write the comma-separated usable device indices here, "
                             "for a launcher that schedules across them.")
    args = parser.parse_args()

    print("torch:", torch.__version__, "cuda build:", torch.version.cuda)
    print("cuda env:", os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"))
    report = gpu_report()
    print("nvidia-smi query:", report)

    if "B200" in report and parse_cuda_version(torch.version.cuda) < MIN_BLACKWELL_CUDA:
        print(
            "ERROR: B200/Blackwell requires a CUDA 12.8+ PyTorch runtime; this "
            f"container has torch.version.cuda={torch.version.cuda!r}. Use a newer "
            "container -- requeueing would land on the same one.",
            file=sys.stderr,
        )
        return EXIT_CONTAINER_MISMATCH

    # try/except so a driver failure becomes the requeue exit code.
    try:
        available = torch.cuda.is_available()
        count = torch.cuda.device_count()
        print("cuda available:", available)
        print("cuda device_count:", count)
        if not available or count < 1:
            print(
                "ERROR: a GPU allocation is present but PyTorch cannot initialize "
                "CUDA. nvidia-smi listing the cards does not contradict this -- it "
                "reads NVML, which needs no CUDA context. Requeue for another "
                "allocation.",
                file=sys.stderr,
            )
            return EXIT_NO_CUDA
        usable = usable_devices(count)
    except Exception as exc:
        print("CUDA preflight failed:", repr(exc))
        print("ERROR: refusing to start work with broken CUDA.", file=sys.stderr)
        return EXIT_CUDA_RAISED

    if not usable:
        print(
            f"ERROR: all {count} visible device(s) failed a smoke allocation. "
            "Requeue for another allocation.",
            file=sys.stderr,
        )
        return EXIT_NO_CUDA
    if len(usable) < count:
        print(f"WARNING: proceeding on {len(usable)}/{count} GPUs; "
              f"unusable: {sorted(set(range(count)) - set(usable))}", file=sys.stderr)
    if args.emit_usable:
        args.emit_usable.write_text(",".join(str(d) for d in usable))
    print("usable devices:", ",".join(str(d) for d in usable))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
