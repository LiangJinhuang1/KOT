import unittest

import numpy as np

from src.data.splits import (
    fitted_row_indices,
    resolve_split_seed,
    split_digest,
    validation_mask,
)

BARCODES = [f"cell_{i:05d}" for i in range(4000)]


class ResolveSplitSeedTests(unittest.TestCase):
    def test_off_ignores_the_model_seed(self):
        self.assertEqual(resolve_split_seed(20260825, 42, False), 20260825)
        self.assertEqual(resolve_split_seed(20260825, 123, False), 20260825)

    def test_on_gives_each_model_seed_its_own(self):
        seeds = {resolve_split_seed(20260825, s, True) for s in (42, 123, 2026, 6)}
        self.assertEqual(len(seeds), 4)

    def test_base_and_model_seed_do_not_commute(self):
        # base + seed would map both of these to 3 and silently share a split.
        self.assertNotEqual(resolve_split_seed(1, 2, True), resolve_split_seed(2, 1, True))


class UnstratifiedTests(unittest.TestCase):
    def test_fraction_is_approximately_honoured(self):
        mask = validation_mask(BARCODES, 0.2, 20260825)
        self.assertAlmostEqual(mask.mean(), 0.2, delta=0.02)

    def test_zero_fraction_holds_nothing_out(self):
        self.assertFalse(validation_mask(BARCODES, 0.0, 20260825).any())

    def test_fraction_outside_the_unit_interval_is_rejected(self):
        for bad in (-0.1, 1.0, 1.5):
            with self.assertRaisesRegex(ValueError, "val_fraction"):
                validation_mask(BARCODES, bad, 20260825)

    def test_membership_survives_cells_being_dropped(self):
        # The property the plain threshold exists for: a cell's side is its own.
        full = validation_mask(BARCODES, 0.2, 20260825)
        kept = [b for i, b in enumerate(BARCODES) if i % 7]
        subset = validation_mask(kept, 0.2, 20260825)
        expected = np.array([full[BARCODES.index(b)] for b in kept])
        np.testing.assert_array_equal(subset, expected)


class StratifiedTests(unittest.TestCase):
    def setUp(self):
        # 3 common strata plus one deliberately rare enough to be lost by a threshold.
        self.strata = np.array(
            ["a"] * 2000 + ["b"] * 1200 + ["c"] * 782 + ["rare"] * 18
        )

    def test_every_stratum_gets_its_share(self):
        mask = validation_mask(BARCODES, 0.1, 20260825, strata=self.strata)
        for level in np.unique(self.strata):
            sel = self.strata == level
            got = int((sel & mask).sum())
            want = int(np.floor(0.1 * sel.sum() + 0.5))
            self.assertEqual(got, want, f"stratum {level}")

    def test_rare_stratum_is_represented(self):
        # 18 cells at 10%: the threshold leaves this empty ~15% of the time.
        mask = validation_mask(BARCODES, 0.1, 20260825, strata=self.strata)
        self.assertEqual(int((mask & (self.strata == "rare")).sum()), 2)

    def test_total_is_close_to_the_requested_fraction(self):
        mask = validation_mask(BARCODES, 0.1, 20260825, strata=self.strata)
        self.assertAlmostEqual(mask.mean(), 0.1, delta=0.005)

    def test_mismatched_strata_length_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "strata has"):
            validation_mask(BARCODES, 0.1, 20260825, strata=self.strata[:-1])


class PairingTests(unittest.TestCase):
    """What the sweep is actually read on: same seed -> same split, across configs."""

    def _mask(self, model_seed, strata=None):
        return validation_mask(
            BARCODES, 0.1,
            resolve_split_seed(20260825, model_seed, True),
            strata=strata,
        )

    def test_same_seed_gives_the_same_split(self):
        # The config never enters the key, so two configs at one seed must agree.
        np.testing.assert_array_equal(self._mask(42), self._mask(42))

    def test_different_seeds_give_different_splits(self):
        for other in (123, 2026, 6):
            self.assertNotEqual(
                split_digest(BARCODES, self._mask(42)),
                split_digest(BARCODES, self._mask(other)),
            )

    def test_seeds_differ_under_stratification_too(self):
        strata = np.array(["a"] * 2000 + ["b"] * 2000)
        self.assertNotEqual(
            split_digest(BARCODES, self._mask(42, strata)),
            split_digest(BARCODES, self._mask(123, strata)),
        )




class FittedRowIndicesTests(unittest.TestCase):
    """The shared rule for 'which cells did this run fit'. kot.py restricts the loss
    and diagnostics with it; runner.py restricts the benchmark FOSCTTM with it. If the
    two ever disagreed, diagnostics.json and comparison_*.csv would describe different
    cell sets while both calling the number mean_foscttm."""

    def obs(self, n=4000, strata=False):
        import pandas as pd
        frame = pd.DataFrame(index=[f"cell_{i:05d}" for i in range(n)])
        if strata:
            frame["cell_type"] = [f"t{i % 7}" for i in range(n)]
        return frame

    def test_no_split_means_every_cell(self):
        self.assertIsNone(fitted_row_indices(self.obs(), {"val_fraction": 0.0}, 42))

    def test_holdout_off_means_every_cell(self):
        cfg = {"val_fraction": 0.2, "val_holdout_from_training": False}
        self.assertIsNone(fitted_row_indices(self.obs(), cfg, 42))

    def test_fitting_the_tuning_subset_inverts_the_side(self):
        obs = self.obs()
        big = fitted_row_indices(obs, {"val_fraction": 0.2}, 42)
        small = fitted_row_indices(
            obs, {"val_fraction": 0.2, "fit_tuning_subset": True}, 42)
        self.assertLess(small.size, big.size)
        self.assertEqual(small.size + big.size, len(obs))
        self.assertEqual(set(small) & set(big), set())

    def test_fixed_split_ignores_the_model_seed(self):
        obs = self.obs()
        cfg = {"val_fraction": 0.2, "fit_tuning_subset": True, "val_split_per_seed": False}
        first = fitted_row_indices(obs, cfg, 42)
        self.assertTrue(np.array_equal(first, fitted_row_indices(obs, cfg, 999)))

    def test_per_seed_split_does_not(self):
        obs = self.obs()
        cfg = {"val_fraction": 0.2, "fit_tuning_subset": True, "val_split_per_seed": True}
        self.assertFalse(np.array_equal(fitted_row_indices(obs, cfg, 42),
                                        fitted_row_indices(obs, cfg, 999)))

    def test_stratification_is_honoured_not_silently_dropped(self):
        # bmmc_cite_retained stratifies by cell_type. A helper that ignored the column
        # would return the unstratified mask and land on different cells than training.
        obs = self.obs(strata=True)
        plain = {"val_fraction": 0.2, "fit_tuning_subset": True}
        strat = {**plain, "val_stratify_by": "cell_type"}
        self.assertFalse(np.array_equal(fitted_row_indices(obs, plain, 42),
                                        fitted_row_indices(obs, strat, 42)))


class FitObsTests(unittest.TestCase):
    """Papalexi CRISPR: train on NT only, score knockout transcriptomes out of sample."""

    def obs(self, n=200):
        import pandas as pd
        labels = ["NT"] * 80 + ["CMTM6"] * 60 + ["STAT1"] * 60
        return pd.DataFrame({"gene": labels},
                            index=[f"cell_{i:05d}" for i in range(n)])

    def test_only_allowed_labels_are_fitted(self):
        obs = self.obs()
        cfg = {"val_fraction": 0.0, "fit_obs_key": "gene", "fit_obs_values": ["NT"]}
        fitted = fitted_row_indices(obs, cfg, 42)
        self.assertEqual(fitted.tolist(), list(range(80)))

    def test_group_holdout_survives_fit_tuning_subset(self):
        # Inverting the random slice must not put knockout cells back into the loss.
        obs = self.obs()
        cfg = {
            "val_fraction": 0.2,
            "fit_tuning_subset": True,
            "fit_obs_key": "gene",
            "fit_obs_values": ["NT"],
        }
        fitted = fitted_row_indices(obs, cfg, 42)
        self.assertTrue((obs["gene"].to_numpy()[fitted] == "NT").all())
        self.assertLess(fitted.size, 80)

    def test_empty_allow_list_is_rejected(self):
        obs = self.obs()
        with self.assertRaisesRegex(ValueError, "fit_obs_values is empty"):
            fitted_row_indices(obs, {"fit_obs_key": "gene", "fit_obs_values": []}, 42)

    def test_missing_column_is_rejected(self):
        obs = self.obs()
        with self.assertRaisesRegex(ValueError, "fit_obs_key"):
            fitted_row_indices(
                obs, {"fit_obs_key": "perturbation", "fit_obs_values": ["NT"]}, 42)

    def test_random_holdout_without_group_is_unchanged(self):
        obs = self.obs()
        cfg = {"val_fraction": 0.2}
        fitted = fitted_row_indices(obs, cfg, 42)
        self.assertEqual(fitted.size, int((~validation_mask(obs.index, 0.2, 20260825)).sum()))


if __name__ == "__main__":
    unittest.main()
