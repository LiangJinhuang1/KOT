from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad

PARAMS = {
    "T_BIF": 5.0, "T_MAX": 10.0, "TAU": 1.25,
    "MU_CRIT": 0.0, "BRANCH_BIAS": 0.05, "BRANCH_INIT_Z": 0.01,
    "ALPHA_BASE": 2.0, "ALPHA_MU_GAIN": 0.5, "ALPHA_FATE_GAIN": 2.5,
    "BETA_R": 1.4, "GAMMA_R": 1.0, "GAMMA_P": 0.5, "K_S": 1.2,
    "RNA_COUNT_SCALE": 80.0, "PROTEIN_COUNT_SCALE": 40.0,
    "SEND_DATE": "2026-04-28",
    "SYSTEM_FLAG": "PITCHFORK_RNA_PROTEIN_BIFURCATION_ACTIVE",
}

DEFAULT_SYNTHETIC_SEED = 42


def get_mu(t):
    return (t - PARAMS["T_BIF"]) / 2.0


def get_dynamics(t, y, fate_sign, history):
    """Derivatives for [latent fate z, unspliced RNA, spliced RNA, protein]."""
    z, u, s, p = y
    mu = get_mu(t)

    # Latent fate: supercritical pitchfork.
    bias = PARAMS["BRANCH_BIAS"] * fate_sign * max(mu - PARAMS["MU_CRIT"], 0)
    dz = mu * z - z**3 + bias

    # RNA transcription, splicing, mature RNA degradation.
    alpha = max(0.05, PARAMS["ALPHA_BASE"]
                + PARAMS["ALPHA_MU_GAIN"] * max(mu - PARAMS["MU_CRIT"], 0)
                + PARAMS["ALPHA_FATE_GAIN"] * z)
    du = alpha - PARAMS["BETA_R"] * u
    ds = PARAMS["BETA_R"] * u - PARAMS["GAMMA_R"] * s

    # Protein: delayed translation.
    s_delayed = np.interp(t - PARAMS["TAU"], history["t"], history["s"])
    dp = PARAMS["K_S"] * s_delayed - PARAMS["GAMMA_P"] * p

    return np.array([dz, du, ds, dp])


def solve_system(t_span, y0, fate_sign, shared_history=None):
    """RK4 integration with built-in history tracking for the delayed term."""
    history = {"t": [t_span[0]], "s": [y0[2]]}
    if shared_history:
        history["t"] = shared_history["t"] + history["t"]
        history["s"] = shared_history["s"] + history["s"]

    results = [y0]
    for i in range(len(t_span) - 1):
        t, t_next = t_span[i], t_span[i + 1]
        h = t_next - t
        y = results[-1]

        k1 = get_dynamics(t, y, fate_sign, history)
        k2 = get_dynamics(t + h / 2, y + h * k1 / 2, fate_sign, history)
        k3 = get_dynamics(t + h / 2, y + h * k2 / 2, fate_sign, history)
        k4 = get_dynamics(t_next, y + h * k3, fate_sign, history)

        y_next = np.maximum(y + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4), [-np.inf, 0, 0, 0])
        results.append(y_next)
        history["t"].append(t_next)
        history["s"].append(y_next[2])

    return np.array(results), history


def noisy_matrix(values, noise, rng):
    values = np.asarray(values)
    return np.clip(values + rng.normal(0, noise, values.shape), 0, None).astype("float32")


def count_matrix(values, scale, rng):
    rates = np.clip(np.asarray(values) * scale, 0, None)
    return rng.poisson(rates).astype("float32")


def feature_modulation(z, mu, n_features, rng):
    """Smooth per-feature changes along fate (z) and pseudotime (mu)."""
    scales = np.ones(n_features)
    fate_loadings = np.zeros(n_features)
    mu_loadings = np.zeros(n_features)
    if n_features > 1:
        scales[1:] = rng.lognormal(mean=0.0, sigma=0.25, size=n_features - 1)
        fate_loadings[1:] = rng.normal(0.0, 0.45, size=n_features - 1)
        mu_loadings[1:] = rng.normal(0.0, 0.15, size=n_features - 1)

    return scales * np.exp(
        z[:, None] * fate_loadings[None, :] + mu[:, None] * mu_loadings[None, :]
    )


def build_protein_adata(adata, cell_ids, protein, protein_mean, protein_names, seed):
    protein_adata = ad.AnnData(X=protein.astype("float32"))
    protein_adata.obs_names = cell_ids
    protein_adata.var_names = protein_names
    protein_adata.var["feature_types"] = "Protein"
    protein_adata.obs = adata.obs.copy()
    # Mean-scale protein layer — needed when KOT trains in ODE-native coordinates.
    protein_adata.layers["protein_mean"] = protein_mean.astype("float32")
    protein_adata.uns["rna_protein_feature_links"] = adata.uns["rna_protein_feature_links"]
    protein_adata.uns["synthetic_seed"] = seed
    return protein_adata


def generate_data(n_cells=5000, noise=0.03, seed=DEFAULT_SYNTHETIC_SEED, n_rna_genes=30, n_proteins=10):
    rng = np.random.default_rng(seed)
    n_p = n_cells // 3
    n_a = (n_cells - n_p) // 2
    n_b = n_cells - n_p - n_a
    t_p = np.linspace(0, PARAMS["T_BIF"], n_p)
    t_a = np.linspace(PARAMS["T_BIF"], PARAMS["T_MAX"], n_a)
    t_b = np.linspace(PARAMS["T_BIF"], PARAMS["T_MAX"], n_b)

    u0 = PARAMS["ALPHA_BASE"] / PARAMS["BETA_R"]
    s0 = PARAMS["BETA_R"] * u0 / PARAMS["GAMMA_R"]
    p0 = (PARAMS["K_S"] / PARAMS["GAMMA_P"]) * s0
    y0 = [0, u0, s0, p0]

    res_p, hist_p = solve_system(t_p, y0, 0)
    branch_start = [res_p[-1, 1], res_p[-1, 2], res_p[-1, 3]]
    res_a, _ = solve_system(t_a, [PARAMS["BRANCH_INIT_Z"], *branch_start], 1, hist_p)
    res_b, _ = solve_system(t_b, [-PARAMS["BRANCH_INIT_Z"], *branch_start], -1, hist_p)

    full_res = np.vstack([res_p, res_a, res_b])
    t_all = np.concatenate([t_p, t_a, t_b])
    mu_all = get_mu(t_all)
    cell_ids = [f"cell_{i:05d}" for i in range(len(t_all))]

    rna_modulation = feature_modulation(full_res[:, 0], mu_all, n_rna_genes, rng)
    protein_modulation = feature_modulation(full_res[:, 0], mu_all, n_proteins, rng)
    beta_genes = PARAMS["BETA_R"] * rng.lognormal(mean=0.0, sigma=0.18, size=n_rna_genes)
    gamma_genes = PARAMS["GAMMA_R"] * rng.lognormal(mean=0.0, sigma=0.18, size=n_rna_genes)

    unspliced_mean = full_res[:, 1, None] * rna_modulation
    spliced_mean = full_res[:, 2, None] * rna_modulation * beta_genes[None, :] / gamma_genes[None, :]
    protein_mean = full_res[:, 3, None] * protein_modulation
    base_velocity = PARAMS["BETA_R"] * full_res[:, 1] - PARAMS["GAMMA_R"] * full_res[:, 2]
    true_velocity = (
        beta_genes[None, :] * unspliced_mean - gamma_genes[None, :] * spliced_mean
    ).astype("float32")

    spliced = count_matrix(noisy_matrix(spliced_mean, noise, rng), PARAMS["RNA_COUNT_SCALE"], rng)
    unspliced = count_matrix(noisy_matrix(unspliced_mean, noise, rng), PARAMS["RNA_COUNT_SCALE"], rng)
    protein = count_matrix(noisy_matrix(protein_mean, noise, rng), PARAMS["PROTEIN_COUNT_SCALE"], rng)

    rna_names = [f"Synthetic_RNA_{i:02d}" for i in range(n_rna_genes)]
    protein_names = [f"Synthetic_Protein_{i:02d}" for i in range(n_proteins)]
    n_links = min(n_rna_genes, n_proteins)
    feature_links = {
        "rna": rna_names[:n_links],
        "protein": protein_names[:n_links],
        "weight": [1.0] * n_links,
        "sign": [1] * n_links,
    }

    adata = ad.AnnData(X=spliced)
    adata.obs_names = cell_ids
    adata.var_names = rna_names
    adata.var["feature_type"] = "rna"
    adata.var["feature_types"] = "GEX"
    adata.layers["spliced"] = spliced
    adata.layers["unspliced"] = unspliced
    adata.layers["true_velocity"] = true_velocity
    # Mean-scale (ODE-native) layers — the simulator's own coordinate system,
    # where the linear ODE dp/dt = K_S·s − GAMMA_P·p is correctly specified.
    adata.layers["spliced_mean"] = spliced_mean.astype("float32")
    adata.layers["unspliced_mean"] = unspliced_mean.astype("float32")
    adata.layers["true_velocity_mean"] = true_velocity.astype("float32")
    adata.obsm["protein_expression"] = pd.DataFrame(protein, index=cell_ids, columns=protein_names)
    adata.obsm["protein_mean"] = pd.DataFrame(
        protein_mean.astype("float32"), index=cell_ids, columns=protein_names,
    )
    adata.obs = pd.DataFrame({
        "time": t_all, "mu": mu_all, "z": full_res[:, 0],
        "unspliced_rna": full_res[:, 1], "spliced_rna": full_res[:, 2],
        "protein": full_res[:, 3], "true_velocity": base_velocity,
        "state": ["Progenitor"] * n_p + ["Branch_A"] * n_a + ["Branch_B"] * n_b,
    }, index=cell_ids)
    adata.uns["protein_names"] = protein_names
    adata.uns["rna_protein_feature_links"] = feature_links
    adata.uns["synthetic_seed"] = seed
    adata.uns["velocity_model"] = {
        "unspliced_equation": "du/dt = alpha - beta*u",
        "spliced_equation": "ds/dt = beta*u - gamma*s",
        "beta": beta_genes,
        "gamma": gamma_genes,
        "scvelo_mode": "deterministic",
    }

    u_x = np.where(
        t_all <= PARAMS["T_BIF"],
        3 * t_all / PARAMS["T_BIF"],
        3 + 2.8 * (t_all - PARAMS["T_BIF"]) / (PARAMS["T_MAX"] - PARAMS["T_BIF"]),
    )
    u_y = np.zeros_like(u_x)
    branch_span = PARAMS["T_MAX"] - PARAMS["T_BIF"]
    u_y[(adata.obs.state == "Branch_A").to_numpy()] = 2.2 * (t_a - PARAMS["T_BIF"]) / branch_span
    u_y[(adata.obs.state == "Branch_B").to_numpy()] = -2.2 * (t_b - PARAMS["T_BIF"]) / branch_span
    adata.obsm["X_umap"] = np.vstack([u_x, u_y]).T + rng.normal(0, 0.05, (len(t_all), 2))

    adata.uns.update({"send_date": PARAMS["SEND_DATE"], "flag": PARAMS["SYSTEM_FLAG"]})
    return adata


def save_synthetic(
    out_rna="cache/velocity/synthetic/synthetic_scvelo_results.h5ad",
    out_protein="cache/synthetic/protein.h5ad",
    seed=DEFAULT_SYNTHETIC_SEED,
    **kwargs,
):
    adata = generate_data(seed=seed, **kwargs)
    protein_df = adata.obsm["protein_expression"]
    protein_adata = build_protein_adata(
        adata,
        list(protein_df.index),
        protein_df.values,
        adata.obsm["protein_mean"].values,
        list(protein_df.columns),
        seed,
    )

    out_rna = Path(out_rna)
    out_protein = Path(out_protein)
    out_rna.parent.mkdir(parents=True, exist_ok=True)
    out_protein.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(out_rna)
    protein_adata.write_h5ad(out_protein)

    print(f"Saved RNA    → {out_rna}")
    print(f"Saved Protein → {out_protein}")
    print(f"Synthetic seed: {seed}")
    return adata, protein_adata
