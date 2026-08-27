"""
Refuse to start work on an allocation whose CUDA is broken.

The failure this exists for looks healthy from the outside: nvidia-smi lists every
B200 with full free memory (NVML needs no CUDA context), while every process that
tries to initialize CUDA fails. Job 4206192 lost all 16 arms to it, four on each of
the four GPUs. A launcher that discovers this per-job discovers it 16 times and
still burns the allocation, so the check belongs before the first job starts.

Exit codes are the contract with the calling SLURM script:

  0   usable
  42  container/driver mismatch (B200 needs a CUDA 12.8+ torch) -- NOT requeueable,
      another allocation has the same container and would fail identically
  44  CUDA reports unavailable, or fewer devices than the allocation should have
  45  CUDA raised while initializing or allocating

44 and 45 are the requeueable ones: they describe this allocation, not this code.

Every visible device is smoke-allocated, not just device 0: slurm/parallel_train.sh
pins jobs round-robin across the GPUs, so a single wedged card would otherwise be
found only by the jobs unlucky enough to land on it.

A partly-broken allocation is USED, not thrown back. Only the devices that fail their
smoke allocation are dropped, and --emit-usable writes the survivors for the launcher
to round-robin over; the requeue verdict is reserved for an allocation with nothing
usable at all. Error 802 is node-level and takes every GPU with it, but a card lost to
ECC, an exclusive-mode holder, or a bus fall-off takes only itself, and giving back
three good B200s over one bad one is not a trade worth making.
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
    """What nvidia-smi says about the cards, which works even when CUDA does not.

    Its absence is not a verdict on the allocation -- a container without the binary
    on PATH still has to be judged by the CUDA probe below -- so a missing nvidia-smi
    degrades to an empty report rather than taking the preflight down with an exit
    code the requeue branch does not recognise.
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

    # The one place try/except earns its keep in this repo: turning a driver-level
    # failure into the exit code the SLURM script requeues on. Without it a raising
    # CUDA init would exit 1, code 45 could never be produced, and the requeue branch
    # that tests for it would be dead.
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
