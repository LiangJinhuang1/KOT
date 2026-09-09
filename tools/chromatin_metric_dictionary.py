#!/usr/bin/env python3
"""Write the metric dictionary that goes with the R2 diagnostic CSVs.

A table is unreadable without direction, reference, and which columns are gates. FOSCTTM is lower-is-better and must be compared to a floor that is not zero; a low dynamics loss means nothing on its own because kappa can drive the RHS to zero instead of satisfying it.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

OUT = Path("cache/results/chromatin/r2_metric_dictionary.csv")

ENTRIES = [
    ("run", "identity", "Run directory under cache/chromatin/runs.", ""),
    ("transform", "setting", "Normalisation of the chromatin input c: as_is | cp10k_log1p | tfidf_lsi.", ""),
    ("phi_gate", "setting", "How much of phi's neural path reaches the output. scalar-zero starts the gate at tanh(0)=0, which also zeroes the network's gradient; none removes the gate; per-gene gives each gene its own.", ""),
    ("lambda_dyn", "setting", "Weight on the ODE (dynamics) loss. 0 is the noDyn condition.", ""),
    ("gate", "verdict", "Whether the run passed the launch gate (preflight_passed.json exists).", "PASS is required before the corruption arms are read."),
    ("gate_floor_convention", "provenance", "Which FOSCTTM floor the run's verdict was recorded against.", "constant_map_0.25_SUPERSEDED runs were judged against a floor that is 0.25 by arithmetic, not by data; re-read them against 0.5."),
    ("foscttm_s / foscttm_u", "metric, LOWER is better", "Fraction Of Samples Closer Than the True Match, on the spliced and unspliced blocks. 0 = perfect retrieval of each cell's partner.", "Must be read against permuted_floor, NOT against 0."),
    ("permuted_floor", "reference", "FOSCTTM of the same predictions under a shuffled correspondence: the no-per-cell-information baseline.", "~0.5. The older constant-map floor is 0.25 for ANY data because a collapsed reference ties every distance in one of the two averaged directions."),
    ("gene_pearson", "metric, higher is better", "Median across genes of the correlation between predicted and observed level across cells (spliced block).", "Compare with the identity and ridge rows of r2_supervised_*.csv, not with 0."),
    ("spread", "diagnostic", "Prediction standard deviation over cells divided by the target's.", "Below 0.05 fails the gate as a collapsed phi; a falling value means phi is contracting toward the target mean."),
    ("val_align", "loss", "Sinkhorn alignment loss on validation cells.", "Can improve while FOSCTTM worsens: a contracted phi matches the target distribution in aggregate without matching cells."),
    ("loss_held_block", "loss", "Supervision on the block the transport plan does not see (u).", ""),
    ("loss_dyn", "loss", "Residual between the pushforward J_phi(c)v and the law's right-hand side.", "MEANINGLESS without kappa_at_floor: kappa multiplies the entire RHS, so driving it to its floor sends the loss to ~0 without the law explaining anything."),
    ("kappa_at_floor", "diagnostic, LOWER is better", "Fraction of cells whose per-cell time-scale kappa sits at its minimum (1e-3).", "High values mean the dynamics term switched ITSELF off."),
    ("alpha_at_floor", "diagnostic, LOWER is better", "Fraction of (cell, gene) transcription rates at their minimum (1e-5).", "High values mean production is off, so u decays with no source."),
    ("kappa_median", "diagnostic", "Median per-cell kinetic time-scale.", ""),
    ("anchor_error_percent", "diagnostic", "Median relative error of gamma against the TimeLapse-seq RNA half-life anchors.", "Near 0 means the prior is acting as a hard pin, not a prior: no residual is left for the data to inform."),
    ("n_gamma_anchors", "provenance", "Genes matched to the external RNA-decay table (Schofield 2018, K562).", "Covers ~16% of output genes; the rest of gamma and all of beta are unanchored."),
    ("jvp_vs_scvelo_centred", "metric, higher is better", "Per-cell cosine between the model's pushforward and scVelo's RNA velocity, after removing the shared mean direction.", "The RAW cosine is not usable: every field of this kind shares a mean direction, so a shuffled fit scores +0.39 on it."),
    ("jvp_null_p95", "reference", "95th percentile of the same statistic under a permutation of the cell correspondence.", "The chance level. jvp_ratio_to_null below 1.0 means worse than chance."),
    ("jvp_margin / jvp_ratio_to_null", "metric, higher is better", "The centred cosine minus, and divided by, its own permutation null.", "Compare ratios, not margins: both the statistic and its null scale together across transforms."),
    ("jvp_rhs_cosine", "diagnostic", "Cosine between the pushforward and the fitted right-hand side.", "Near 0 means the residual was reduced by matching magnitudes rather than directions."),
    ("residual_norm", "diagnostic", "Median norm of the dynamics residual.", ""),
    ("grad_mag_ratio", "diagnostic", "Weighted dynamics gradient norm divided by the alignment gradient norm.", "Says which objective actually drove training."),
    ("network_over_prediction", "diagnostic", "Share of phi's centred output contributed by the neural path (r2_phi.csv).", "Near 0 means phi is its linear affine warm start and the network is inert."),
    ("mean_only / identity / ridge", "reference rows", "In r2_supervised_*.csv: the training mean (floor), Gc with no model, and a PAIRED supervised ridge (ceiling).", "The ridge sees the cell correspondence KOT must discover, so it is a ceiling and never a fair competitor."),
    ("ridge on c / ridge on v_c", "reference rows", "In r2_scvelo_*.csv: paired ridges predicting scVelo velocity from the chromatin state and from the chromatin displacement.", "ridge on v_c is the linear analogue of the model's J_phi(c)v: if phi were linear, the pushforward WOULD be W v_c."),
]


def main() -> int:
    frame = pd.DataFrame(ENTRIES, columns=["column", "kind", "meaning", "how_to_read"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT, index=False)
    print(f"[dictionary] wrote {len(frame)} entries -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
