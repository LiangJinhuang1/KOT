"""J_phi(r)v is the real derivative, checked against central finite differences.

The kinetics term is built on one forward-mode JVP. If it drifts from the true directional
derivative -- a changed activation, a spectral-norm wrapper, a torch.func regression -- every
physics diagnostic in the project keeps reporting numbers, just about the wrong operator.

The strong check is not the size of the disagreement but how it SCALES. Central differences
are O(eps^2), so halving eps by 10 must cut the error by ~100 as long as the analytic value
is exact. If the JVP were subtly wrong the error would instead flatten out at the JVP's own
bias, which no single-tolerance assertion would catch. Was an exploratory CLI that only ran
when someone remembered to run it.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F

from src.models.KOT import KOTModel

SEED = 42
BATCH = 128
D_RNA = 30
D_PROTEIN = 10


def build_model(dtype, activation="silu", spectral_norm=False):
    torch.manual_seed(SEED)
    model = KOTModel(D_RNA, D_PROTEIN, phi_dims=[128, 64], kappa_dims=[16], g_dims=[16],
                     phi_init_gain=0.5, activation=activation, init_method="xavier",
                     phi_spectral_norm=spectral_norm)
    return model.to(dtype=dtype).eval()


def inputs(dtype):
    """A batch and unit-norm directions, so the relative error has a stable denominator."""
    torch.manual_seed(SEED + 1)
    r = torch.randn(BATCH, D_RNA, dtype=dtype)
    v = torch.randn(BATCH, D_RNA, dtype=dtype)
    return r, v / v.norm(dim=1, keepdim=True).clamp_min(1e-12)


def relative_error(model, r, v, eps):
    """Median |central difference - jvp| / |jvp| over the batch, and their cosine."""
    _phi_r, jvp = torch.func.jvp(model.phi, (r,), (v,))
    fd = (model.phi(r + eps * v) - model.phi(r - eps * v)) / (2.0 * eps)
    rel = (fd - jvp).norm(dim=1) / jvp.norm(dim=1).clamp_min(1e-12)
    cos = F.cosine_similarity(fd, jvp, dim=1, eps=1e-12)
    return float(rel.median().detach()), float(cos.min().detach())


class JvpFiniteDifferenceTests(unittest.TestCase):

    def test_error_scales_as_eps_squared(self):
        """The signature of an exact derivative: 10x smaller eps, ~100x smaller error.

        A JVP carrying its own bias would flatten instead, and still pass a loose tolerance.
        """
        model = build_model(torch.float64)
        r, v = inputs(torch.float64)
        coarse, _ = relative_error(model, r, v, 1e-2)
        fine, _ = relative_error(model, r, v, 1e-3)
        self.assertGreater(coarse / fine, 50.0, f"error flattened: {coarse:.2e} -> {fine:.2e}")
        self.assertLess(coarse / fine, 200.0, f"error fell faster than eps^2: {coarse:.2e} -> {fine:.2e}")

    def test_float64_agreement(self):
        model = build_model(torch.float64)
        r, v = inputs(torch.float64)
        rel, cos = relative_error(model, r, v, 1e-3)
        self.assertLess(rel, 1e-8)          # measured 1.1e-9
        self.assertGreater(cos, 1 - 1e-9)

    def test_float32_agreement(self):
        """float32 is what training runs in. Its optimum sits at a LARGER eps than
        float64's -- past that, roundoff in phi dominates the truncation term."""
        model = build_model(torch.float32)
        r, v = inputs(torch.float32)
        rel, cos = relative_error(model, r, v, 1e-2)
        self.assertLess(rel, 1e-3)          # measured 9.5e-5
        self.assertGreater(cos, 1 - 1e-4)

    def test_holds_under_the_configured_activations(self):
        """gelu is the project default (kot_activation); silu is the sweep alternative.
        A non-smooth activation would break the finite-difference comparison itself."""
        for activation in ("gelu", "silu"):
            with self.subTest(activation=activation):
                model = build_model(torch.float64, activation=activation)
                r, v = inputs(torch.float64)
                rel, cos = relative_error(model, r, v, 1e-3)
                self.assertLess(rel, 1e-8)
                self.assertGreater(cos, 1 - 1e-9)

    def test_holds_under_spectral_norm(self):
        """The kot_unbounded_sn arm trains with phi_spectral_norm=True.

        Its finite-difference error at a given eps is ~600x the plain model's, which is
        curvature rather than a defect: spectral norm enlarges the third derivative of phi,
        and that is exactly the term the O(eps^2) truncation error of a central difference
        carries. The scaling settles it -- still exactly eps^2, so the JVP is the true
        derivative here too, and only the eps where the comparison is informative moves.
        """
        model = build_model(torch.float64, spectral_norm=True)
        r, v = inputs(torch.float64)
        coarse, _ = relative_error(model, r, v, 1e-3)
        fine, cos = relative_error(model, r, v, 1e-4)
        self.assertGreater(coarse / fine, 50.0)
        self.assertLess(fine, 1e-8)          # measured 6.9e-9
        self.assertGreater(cos, 1 - 1e-9)


if __name__ == "__main__":
    unittest.main()
