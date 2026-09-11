#!/usr/bin/env python3
"""Pick two screen survivors on validation, then write one-factor kinetic controls.

Pareto objectives, all from the development split:
  alignment   lower val FOSCTTM
  spread      prediction_spread_ratio (collapse filtered)
  jvp_margin  centred JVP-vs-scVelo cosine minus its permutation null
  rates       1 - mean(kappa_at_floor, alpha_at_floor)

noDyn and collapsed / null-failing runs are infeasible. The two winners keep
every other screen flag; each of the four axes then moves one knob:

  phi-gate              scalar-zero (baseline) | per-gene | none
  kappa                 weak prior 0.01 (baseline) | strong 0.1 | fixed log(2)
  auxiliary u weight    1 (baseline) | 0 | 0.1
  gamma-anchor weight   1 (baseline) | 0.1 | 10

Baselines are the winner itself, so this is 8 new trains per winner (16 total),
not 24. The screen already opens the r2lsi kappa box; strong kappa only raises
the prior to 0.1, and fixed kappa removes the scale head.

Usage, after the screen has val evaluations:
  PYTHONPATH=. python tools/chromatin_kinetic_controls.py select \\
      --runs-root cache/chromatin/runs --prefix r2scr_
  PYTHONPATH=. python tools/chromatin_kinetic_controls.py jobs \\
      --winners cache/results/chromatin/kinetic_winners.json \\
      --out jobs/jobs_chromatin_kinetic_controls.txt
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

SPREAD_FLOOR = 0.05
RATE_FLOOR_LIMIT = 0.1
KAPPA_FIXED = math.log(2)
KAPPA_BOX = (1.0e-3, 1.5)
MAXIMIZE = ("alignment", "spread_score", "jvp_margin", "rate_health")

# Winner identity copied into every control. Axis knobs are applied on top.
PRESERVE_FLAGS = (
    ("dataset", "--dataset"),
    ("law", "--law"),
    ("condition", "--condition"),
    ("seed", "--seed"),
    ("split_seed", "--split-seed"),
    ("lambda_dyn", "--lambda-dyn"),
    ("align_block", "--align-block"),
    ("lambda_held_block", "--lambda-held-block"),
    ("gamma_anchor_csv", "--gamma-anchor-csv"),
    ("lambda_gamma_anchor", "--lambda-gamma-anchor"),
    ("chromatin_transform", "--chromatin-transform"),
    ("regulatory_transform", "--regulatory-transform"),
    ("velocity_tag", "--velocity-tag"),
    ("dyn_residual_weight", "--dyn-residual-weight"),
    ("dyn_direction_weight", "--dyn-direction-weight"),
    ("kappa_min", "--kappa-min"),
    ("kappa_max", "--kappa-max"),
    ("checkpoint_monitor", "--checkpoint-monitor"),
    ("stabilization", "--stabilization"),
    ("phi_gate", "--phi-gate"),
    ("lambda_kappa_prior", "--lambda-kappa-prior"),
    ("kappa_prior_target", "--kappa-prior-target"),
    ("lambda_reg", "--lambda-reg"),
    ("grad_clip", "--grad-clip"),
    ("phi_lag_tau", "--phi-lag-tau"),
    ("alpha_input", "--alpha-input"),
    ("gene_map_mode", "--gene-map-mode"),
    ("rna_kinetic_coords", "--rna-kinetic-coords"),
)


def evaluation_path(run: Path, split: str = "val") -> Path | None:
    for checkpoint in ("best_align", "final"):
        path = run / f"evaluation_{checkpoint}_{split}.json"
        if path.exists():
            return path
    return None


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def score_run(run: Path, split: str = "val", *, allow_missing_jvp: bool = False
              ) -> dict | None:
    """One row of validation objectives, or None if the run cannot be scored."""
    config_path, gate_path = run / "run_config.json", run / "preflight.json"
    if not (config_path.exists() and gate_path.exists()):
        return None
    config, gate = load_json(config_path), load_json(gate_path)
    evaluation = {}
    eval_path = evaluation_path(run, split)
    if eval_path is not None:
        evaluation = load_json(eval_path)
    task_b = evaluation.get("task_b") or {}
    if "knn_foscttm" in task_b:
        foscttm = task_b["knn_foscttm"]
        alignment_source = "evaluation_task_b"
    else:
        foscttm = gate.get("foscttm")
        alignment_source = "preflight"
    floor = task_b.get("knn_foscttm_permuted_floor", gate.get("foscttm_permuted_floor"))
    spread = gate.get("prediction_spread_ratio")
    centred = gate.get("jvp_vs_reference_cosine_centred_median")
    null = gate.get("jvp_vs_reference_cosine_centred_null_p95")
    lambda_dyn = config.get("lambda_dyn")
    if config.get("condition") == "noDyn":
        lambda_dyn = 0.0
    jvp_margin = None
    if centred is not None and null is not None:
        jvp_margin = float(centred) - float(null)
    infeasible = []
    if config.get("condition") != "full" or not lambda_dyn:
        infeasible.append("no dynamics")
    if spread is None or not math.isfinite(float(spread)) or float(spread) < SPREAD_FLOOR:
        infeasible.append("collapsed spread")
    pearson = gate.get("state_pearson_median")
    if pearson is None or not math.isfinite(float(pearson)) or float(pearson) <= 0:
        infeasible.append("state Pearson is not positive")
    if foscttm is None or not math.isfinite(float(foscttm)):
        infeasible.append("missing FOSCTTM")
    elif floor is None or not math.isfinite(float(floor)) or float(foscttm) >= float(floor):
        infeasible.append("FOSCTTM not below the permuted floor")
    if jvp_margin is None or not math.isfinite(jvp_margin):
        if allow_missing_jvp:
            jvp_margin = 0.0
        else:
            infeasible.append("missing JVP-vs-scVelo margin")
    elif jvp_margin <= 0:
        infeasible.append("JVP-vs-scVelo does not beat the permutation null")
    if "kappa_at_floor" not in gate or "alpha_at_floor" not in gate:
        infeasible.append("missing rate-floor statistics")
        kappa_floor = alpha_floor = float("nan")
    else:
        kappa_floor = float(gate["kappa_at_floor"])
        alpha_floor = float(gate["alpha_at_floor"])
        if not math.isfinite(kappa_floor) or not math.isfinite(alpha_floor):
            infeasible.append("missing rate-floor statistics")
        if kappa_floor > RATE_FLOOR_LIMIT:
            infeasible.append("kappa at floor")
        if alpha_floor > RATE_FLOOR_LIMIT:
            infeasible.append("alpha at floor")
    foscttm = float(foscttm) if foscttm is not None and math.isfinite(float(foscttm)) else math.nan
    spread = float(spread) if spread is not None and math.isfinite(float(spread)) else math.nan
    return {
        "run": run.name,
        "run_dir": str(run),
        "config": config,
        "feasible": not infeasible,
        "infeasible": infeasible,
        "foscttm": foscttm,
        "foscttm_floor": None if floor is None else float(floor),
        "alignment_source": alignment_source,
        "spread": spread,
        "jvp_margin": jvp_margin,
        "kappa_at_floor": kappa_floor,
        "alpha_at_floor": alpha_floor,
        "alignment": -foscttm if math.isfinite(foscttm) else math.nan,
        "spread_score": min(spread, 1.0) if math.isfinite(spread) else math.nan,
        "rate_health": (1.0 - 0.5 * (kappa_floor + alpha_floor)
                        if math.isfinite(kappa_floor) and math.isfinite(alpha_floor)
                        else math.nan),
        "lambda_dyn": lambda_dyn,
        "transform": config.get("chromatin_transform"),
        "regulatory_transform": config.get("regulatory_transform", "same"),
        "stabilization": config.get("stabilization", "legacy"),
    }


def dominates(left: dict, right: dict, keys: tuple[str, ...] = MAXIMIZE) -> bool:
    """left dominates right on the higher-is-better scores."""
    no_worse = all(left[key] >= right[key] for key in keys)
    better = any(left[key] > right[key] for key in keys)
    return no_worse and better


def pareto_front(rows: list[dict], keys: tuple[str, ...] = MAXIMIZE) -> list[dict]:
    return [row for row in rows
            if not any(dominates(other, row, keys) for other in rows if other is not row)]


def pick_winners(rows: list[dict], n: int = 2) -> list[dict]:
    """Two Pareto survivors: best val alignment, then the remaining best JVP margin.

    Dominated runs are not used as a second winner. If the front has fewer than
    two points, the caller should refuse to launch the kinetic controls.
    """
    feasible = [row for row in rows if row["feasible"]]
    front = list(pareto_front(feasible))
    chosen: list[dict] = []
    if front:
        best_align = min(front, key=lambda row: (row["foscttm"], -row["jvp_margin"]))
        best_align = dict(best_align)
        best_align["picked_for"] = "alignment"
        chosen.append(best_align)
    if len(chosen) < n:
        rest = [row for row in front if row["run"] not in {item["run"] for item in chosen}]
        if rest:
            best_kinetics = max(rest, key=lambda row: (
                row["jvp_margin"], row["rate_health"], -row["foscttm"]))
            best_kinetics = dict(best_kinetics)
            best_kinetics["picked_for"] = "jvp_margin"
            chosen.append(best_kinetics)
    return chosen[:n]


def control_overrides() -> list[dict]:
    """Eight new settings; the four baselines are the winner run itself."""
    return [
        {"tag": "gate_pergene", "phi_gate": "per-gene"},
        {"tag": "gate_none", "phi_gate": "none"},
        {"tag": "kstrong", "lambda_kappa_prior": 0.1,
         "kappa_min": KAPPA_BOX[0], "kappa_max": KAPPA_BOX[1], "fixed_kappa": None},
        {"tag": "kfixed", "fixed_kappa": KAPPA_FIXED, "lambda_kappa_prior": 0.0,
         "kappa_min": None, "kappa_max": None},
        {"tag": "u0", "lambda_held_block": 0.0},
        {"tag": "u0p1", "lambda_held_block": 0.1},
        {"tag": "g0p1", "lambda_gamma_anchor": 0.1},
        {"tag": "g10", "lambda_gamma_anchor": 10.0},
    ]


def stem_from_winner(name: str) -> str:
    text = name.removeprefix("r2scr_").removesuffix("_seed42")
    return text or name


def flag_value(value) -> list[str]:
    if isinstance(value, float) and value == int(value) and abs(value) < 1e6:
        return [str(int(value))]
    return [str(value)]


def train_tokens(config: dict, run_dir: str) -> list[str]:
    tokens = ["train"]
    skip_min_max = config.get("fixed_kappa") is not None
    for key, flag in PRESERVE_FLAGS:
        if skip_min_max and key in ("kappa_min", "kappa_max"):
            continue
        if key == "phi_lag_tau" and not config.get(key):
            continue
        if key == "regulatory_transform" and config.get(key, "same") == "same":
            continue
        if key == "rna_kinetic_coords" and config.get(key, "shared_log1p") == "shared_log1p":
            continue
        value = config.get(key)
        if value is None:
            continue
        tokens.extend([flag, *flag_value(value)])
    if config.get("fixed_kappa") is not None:
        tokens.extend(["--fixed-kappa", f"{float(config['fixed_kappa']):.12g}"])
    tokens.extend(["--run-dir", run_dir])
    return tokens


def apply_override(config: dict, override: dict) -> tuple[str, dict]:
    updated = dict(config)
    spec = dict(override)
    tag = spec.pop("tag")
    updated.update(spec)
    return tag, updated


def control_jobs(winners: list[dict]) -> list[str]:
    lines = []
    for winner in winners:
        stem = stem_from_winner(winner["run"])
        for override in control_overrides():
            tag, config = apply_override(winner["config"], override)
            run_dir = f"cache/chromatin/runs/r2kin_{stem}_{tag}_seed42"
            lines.append(" ".join(train_tokens(config, run_dir)))
    return lines


def job_file_text(winners: list[dict]) -> str:
    names = ", ".join(item["run"] for item in winners)
    header = f"""# Kinetic interrogation of the two Pareto survivors from the input screen.
# Winners (validation): {names}
#
# Generated by tools/chromatin_kinetic_controls.py. Do not invent a third
# configuration. Development evaluate: --eval-split val. Do not score BMMC test.
#
# Eight new trains per winner. Baselines (scalar-zero, λ_κ=0.01, L_u=1, L_γ=1)
# are the winner runs themselves.
#
# Submit from the repo root after the screen velocity caches exist:
#   while IFS= read -r line; do
#     [[ "$line" =~ ^train ]] || continue
#     name=${{line##*--run-dir cache/chromatin/runs/}}
#     sbatch --job-name="$name" \\
#       --export=ALL,RUN_CMD="PYTHONPATH=. python -u run_kot_chromatin.py $line" \\
#       slurm/train_slurm.sh
#   done < jobs/jobs_chromatin_kinetic_controls.txt
"""
    return header + "\n" + "\n".join(control_jobs(winners)) + "\n"


def discover_runs(root: Path, prefix: str) -> list[Path]:
    return sorted(path for path in root.glob(prefix + "*")
                  if path.is_dir() and (path / "run_config.json").exists())


def select_main(args: argparse.Namespace) -> int:
    rows = [score_run(path, args.eval_split, allow_missing_jvp=args.allow_missing_jvp)
            for path in discover_runs(args.runs_root, args.prefix)]
    rows = [row for row in rows if row is not None]
    winners = pick_winners(rows, args.n)
    payload = {
        "eval_split": args.eval_split,
        "allow_missing_jvp": args.allow_missing_jvp,
        "n_scored": len(rows),
        "n_feasible": sum(row["feasible"] for row in rows),
        "winners": [{key: value for key, value in row.items() if key != "config"}
                    | {"config": row["config"]} for row in winners],
        "infeasible": [{"run": row["run"], "reasons": row["infeasible"]}
                       for row in rows if not row["feasible"]],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, default=str))
    print(f"[kinetic-controls] scored {payload['n_scored']} runs, "
          f"{payload['n_feasible']} feasible; wrote {args.out}")
    for row in winners:
        print(f"  winner {row['run']} ({row.get('picked_for', '?')})  "
              f"FOSCTTM {row['foscttm']:.4f}  spread {row['spread']:.3f}  "
              f"JVP margin {row['jvp_margin']:+.4f}  rates {row['rate_health']:.3f}")
    if len(winners) < args.n:
        print(f"[kinetic-controls] Pareto front has {len(winners)} point(s); "
              "need two non-dominated survivors before launching controls")
        return 1
    return 0


def jobs_main(args: argparse.Namespace) -> int:
    payload = load_json(args.winners)
    winners = payload["winners"] if "winners" in payload else payload
    if len(winners) != 2:
        raise ValueError(f"need exactly two winners, got {len(winners)}")
    text = job_file_text(winners)
    args.out.write_text(text)
    trains = [line for line in text.splitlines() if line.startswith("train ")]
    print(f"[kinetic-controls] wrote {len(trains)} trains to {args.out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    select = sub.add_parser("select", help="Pareto-pick two feasible screen runs")
    select.add_argument("--runs-root", type=Path, default=Path("cache/chromatin/runs"))
    select.add_argument("--prefix", default="r2scr_")
    select.add_argument("--eval-split", default="val")
    select.add_argument("--n", type=int, default=2)
    select.add_argument(
        "--allow-missing-jvp", action="store_true",
        help="keep a run eligible when scVelo cosine is not yet on preflight; "
             "a present cosine that loses to its permutation null still fails")
    select.add_argument("--out", type=Path,
                        default=Path("cache/results/chromatin/kinetic_winners.json"))
    select.set_defaults(func=select_main)
    jobs = sub.add_parser("jobs", help="write the 16 one-factor trains")
    jobs.add_argument("--winners", type=Path, required=True)
    jobs.add_argument("--out", type=Path,
                      default=Path("jobs/jobs_chromatin_kinetic_controls.txt"))
    jobs.set_defaults(func=jobs_main)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
