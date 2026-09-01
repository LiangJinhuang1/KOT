import tempfile
import unittest
from pathlib import Path

import anndata as ad
import numpy as np

from src.evaluation.protocol import (
    HOLDOUT_SECOND,
    PROTOCOL_KEY,
    NATIVE,
    PAIRED,
    plan_fit_restriction,
    predictor_kind,
    protocol_stamp,
    read_protocol,
    resolve_oos_mode,
    restrict_second_modality,
    scatter_rows,
)


def context(n=10, d_rna=3, d_prot=4):
    rna = ad.AnnData(X=np.zeros((n, d_rna), dtype=np.float32))
    protein = ad.AnnData(X=np.arange(n * d_prot, dtype=np.float32).reshape(n, d_prot))
    return {
        "x": np.zeros((n, d_rna), dtype=np.float32),
        "y": np.asarray(protein.X),
        "rna_adata": rna,
        "second_adata": protein,
        "second_label": "protein",
        "modality_pair": "rna_protein",
    }


class CapabilityTests(unittest.TestCase):
    def test_every_kot_variant_resolves_without_being_listed(self):
        for name in ("kot", "kot_nodyn", "kot_noanchor", "kot_unbounded_sn"):
            self.assertEqual(resolve_oos_mode(name), NATIVE)

    def test_an_undeclared_model_is_an_error_not_a_default(self):
        with self.assertRaisesRegex(KeyError, "no declared out-of-sample capability"):
            resolve_oos_mode("some_new_baseline")

    def test_totalvi_is_paired_and_the_ot_baselines_are_not(self):
        self.assertEqual(resolve_oos_mode("totalvi"), PAIRED)
        for name in ("scot", "moscot", "glue", "uniport", "linear_ode"):
            self.assertEqual(resolve_oos_mode(name), HOLDOUT_SECOND)


class PredictorTests(unittest.TestCase):
    def test_feature_space_runs_are_direct_whatever_the_model_is_called(self):
        self.assertEqual(predictor_kind({"kot_use_feature_space": True}), "direct")
        self.assertEqual(predictor_kind({}), "latent")


class RestrictionPlanTests(unittest.TestCase):
    """A paired oracle must never be credited with a holdout it cannot honour."""

    def test_paired_model_is_refused_under_a_fit_restriction(self):
        with self.assertRaisesRegex(ValueError, "paired oracle"):
            plan_fit_restriction({"fit_obs_key": "gene", "fit_obs_values": ["NT"]},
                                 "totalvi", np.arange(5), "protein")

    def test_opted_in_paired_model_is_recorded_as_having_fitted_everything(self):
        rows, mode = plan_fit_restriction(
            {"fit_obs_key": "gene", "fit_obs_values": ["NT"], "allow_paired_oracle": True},
            "totalvi", np.arange(5), "protein")
        self.assertIsNone(rows)
        self.assertEqual(mode, PAIRED)

    def test_unrestricted_runs_are_untouched(self):
        rows, mode = plan_fit_restriction({}, "totalvi", None, "protein")
        self.assertIsNone(rows)
        self.assertEqual(mode, PAIRED)

    def test_a_capable_model_keeps_its_rows(self):
        fit_rows = np.array([0, 2, 4])
        rows, mode = plan_fit_restriction({"fit_obs_key": "gene"}, "glue", fit_rows, "protein")
        np.testing.assert_array_equal(rows, fit_rows)
        self.assertEqual(mode, HOLDOUT_SECOND)


class RestrictSecondModalityTests(unittest.TestCase):
    def test_rna_side_is_untouched_and_protein_side_is_cut(self):
        ctx = context(n=10)
        fit_rows = np.array([1, 3, 5])
        out = restrict_second_modality(ctx, fit_rows)
        self.assertEqual(out["x"].shape[0], 10)
        self.assertEqual(out["y"].shape[0], 3)
        self.assertEqual(out["second_adata"].n_obs, 3)
        np.testing.assert_array_equal(out["y"], ctx["y"][fit_rows])
        np.testing.assert_array_equal(out["fit_rows"], fit_rows)

    def test_the_caller_can_recover_the_pairing(self):
        ctx = context(n=10)
        fit_rows = np.array([1, 3, 5])
        out = restrict_second_modality(ctx, fit_rows)
        np.testing.assert_array_equal(out["x"][out["fit_rows"]], ctx["x"][fit_rows])


class ScatterTests(unittest.TestCase):
    def test_held_out_rows_are_nan_not_zero(self):
        """Zero would read as a coordinate at the origin to anything that forgot to mask."""
        values = np.ones((3, 2), dtype=np.float32)
        out = scatter_rows(values, np.array([0, 2, 4]), 5)
        self.assertEqual(out.shape, (5, 2))
        np.testing.assert_array_equal(out[[0, 2, 4]], values)
        self.assertTrue(np.isnan(out[[1, 3]]).all())


class StampTests(unittest.TestCase):
    def test_restricted_run_is_stamped_out_of_distribution(self):
        stamp = protocol_stamp(
            {"fit_obs_key": "gene", "fit_obs_values": ["NT"], "kot_use_feature_space": True},
            "kot", n_cells=100, fit_rows=np.arange(30))
        self.assertTrue(stamp["out_of_distribution"])
        self.assertFalse(stamp["paired_oracle"])
        self.assertEqual((stamp["n_fit_cells"], stamp["n_cells"]), (30, 100))
        self.assertEqual(stamp["predictor"], "direct")

    def test_unrestricted_run_reports_every_cell_as_fitted(self):
        stamp = protocol_stamp({}, "glue", n_cells=100, fit_rows=None)
        self.assertFalse(stamp["out_of_distribution"])
        self.assertEqual(stamp["n_fit_cells"], 100)

    def test_stamp_survives_an_h5ad_round_trip(self):
        """uns cannot hold None, so the stamp must be plain scalars and lists."""
        stamp = protocol_stamp({"fit_obs_key": "gene", "fit_obs_values": ["NT"]},
                               "glue", n_cells=6, fit_rows=np.arange(2))
        adata = ad.AnnData(X=np.zeros((6, 2), dtype=np.float32))
        adata.uns[PROTOCOL_KEY] = stamp
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "p.h5ad"
            adata.write_h5ad(path)
            self.assertEqual(read_protocol(ad.read_h5ad(path).uns), stamp)

    def test_an_unstamped_file_reads_as_in_distribution(self):
        protocol = read_protocol({})
        self.assertEqual(protocol["oos_mode"], "unstamped")
        self.assertFalse(protocol["out_of_distribution"])


if __name__ == "__main__":
    unittest.main()
