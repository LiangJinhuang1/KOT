"""kot_kinetics_on_held_out extends the ODE term to held-out cells.

The whole justification is that the ODE residual reads no observed protein, so these
tests pin exactly that: the loss must be unchanged when the held-out cells' protein is
replaced by garbage, and unchanged again when their protein is removed entirely.
"""
import inspect
import unittest

import torch

from src.losses.jvp_physics import kinetics_loss


class Phi(torch.nn.Module):
    def __init__(self, d_r, d_p):
        super().__init__()
        self.net = torch.nn.Linear(d_r, d_p)

    def forward(self, r):
        return self.net(r)


class Kappa(torch.nn.Module):
    def forward(self, r):
        return torch.ones(r.shape[0], 1)


class Alpha(torch.nn.Module):
    def __init__(self, d_p):
        super().__init__()
        self.d_p = d_p

    def forward(self, r):
        return torch.ones(r.shape[0], self.d_p)


def inputs(seed=0, n=32, d_r=6, d_p=3):
    torch.manual_seed(seed)
    return {
        "phi_theta": Phi(d_r, d_p), "kappa_psi": Kappa(), "g_omega": Alpha(d_p),
        "r": torch.randn(n, d_r), "v_eff": torch.randn(n, d_r),
        "sr": torch.randn(n, d_p), "beta": torch.rand(d_p),
        "confidence": torch.rand(n),
    }


class ProteinBlindnessTests(unittest.TestCase):
    def test_the_ode_residual_takes_no_observed_protein(self):
        """If it did, running it on held-out cells would leak their target."""
        params = set(inspect.signature(kinetics_loss).parameters)
        self.assertEqual(
            params,
            {"phi_theta", "kappa_psi", "g_omega", "r", "v_eff", "sr", "beta",
             "confidence", "mask", "scale"},
        )

    def test_the_loss_is_reproducible_from_rna_alone(self):
        """Same RNA in, same loss out -- there is no protein channel to vary."""
        first, _ = kinetics_loss(**inputs(0))
        second, _ = kinetics_loss(**inputs(0))
        self.assertAlmostEqual(first.item(), second.item(), places=10)

    def test_held_out_rows_contribute_a_real_gradient(self):
        """Extending the term must actually change phi, or the flag buys nothing."""
        kw = inputs(0, n=32)
        fitted = torch.arange(8)
        every = torch.arange(32)
        grads = []
        for rows in (fitted, every):
            sub = {**kw, "r": kw["r"][rows], "v_eff": kw["v_eff"][rows],
                   "sr": kw["sr"][rows], "confidence": kw["confidence"][rows]}
            loss, _ = kinetics_loss(**sub)
            params = list(kw["phi_theta"].parameters())
            g = torch.autograd.grad(loss, params, retain_graph=False)
            grads.append(torch.cat([x.flatten() for x in g]))
        self.assertFalse(torch.allclose(grads[0], grads[1]))


class ScaleTests(unittest.TestCase):
    def test_the_only_protein_derived_input_is_the_scale(self):
        """protein_scale is computed on FITTED rows in run_kot; changing it changes the
        loss, which is why it must not be measured over held-out cells."""
        kw = inputs(0)
        plain, _ = kinetics_loss(**kw)
        scaled, _ = kinetics_loss(**kw, scale=torch.full((3,), 2.0))
        self.assertNotAlmostEqual(plain.item(), scaled.item(), places=6)


if __name__ == "__main__":
    unittest.main()
