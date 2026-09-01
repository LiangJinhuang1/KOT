import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.kot import (
    apply_velocity_ablation,
    branch_accuracy_scores,
    corrupt_velocity,
    permute_s_rows,
    resolve_velocity_ablation,
    velocity_row_permutation,
)


class ResolveVelocityAblationTests(unittest.TestCase):
    """The enum and the boolean it replaced have to agree on one answer.

    Runs from the permutation control carry kot_velocity_shuffle in their saved config;
    if the enum silently won, re-reading those runs would report them as un-ablated.
    """

    def test_default_is_the_real_arm(self):
        self.assertEqual(resolve_velocity_ablation({}), "none")

    def test_each_arm_resolves_to_itself(self):
        for arm in ("none", "shuffle", "reverse", "zero"):
            self.assertEqual(resolve_velocity_ablation({"kot_velocity_ablation": arm}), arm)

    def test_legacy_boolean_still_selects_the_shuffle(self):
        self.assertEqual(resolve_velocity_ablation({"kot_velocity_shuffle": True}), "shuffle")

    def test_legacy_boolean_agreeing_with_the_enum_is_accepted(self):
        cfg = {"kot_velocity_shuffle": True, "kot_velocity_ablation": "shuffle"}
        self.assertEqual(resolve_velocity_ablation(cfg), "shuffle")

    def test_two_different_controls_at_once_is_an_error(self):
        cfg = {"kot_velocity_shuffle": True, "kot_velocity_ablation": "zero"}
        with self.assertRaises(ValueError):
            resolve_velocity_ablation(cfg)

    def test_unknown_arm_is_an_error_not_a_silent_real_run(self):
        with self.assertRaises(ValueError):
            resolve_velocity_ablation({"kot_velocity_ablation": "reversed"})


class VelocityAblationTests(unittest.TestCase):
    """Each arm must break exactly the one thing it names and leave the rest alone."""

    def setUp(self):
        rng = np.random.default_rng(0)
        self.velocity = rng.normal(size=(12, 5)).astype(np.float32)
        self.confidence = np.linspace(0.1, 1.0, 12, dtype=np.float32)

    def ablate(self, arm):
        return apply_velocity_ablation(
            self.velocity, self.confidence, {"kot_velocity_ablation": arm, "seed": 42},
        )

    def test_none_returns_the_real_field_untouched(self):
        velocity, confidence = self.ablate("none")
        self.assertTrue(np.array_equal(velocity, self.velocity))
        self.assertTrue(np.array_equal(confidence, self.confidence))

    def test_reverse_flips_direction_and_keeps_every_norm(self):
        """The gauge divides by the median per-cell norm, so a reversed arm that changed
        norms would differ from the real arm in scale as well as in time direction."""
        velocity, confidence = self.ablate("reverse")
        self.assertTrue(np.allclose(velocity, -self.velocity))
        self.assertTrue(np.allclose(np.linalg.norm(velocity, axis=1),
                                    np.linalg.norm(self.velocity, axis=1)))
        self.assertTrue(np.array_equal(confidence, self.confidence))

    def test_reverse_keeps_each_velocity_on_its_own_cell(self):
        velocity, _ = self.ablate("reverse")
        cosines = (velocity * self.velocity).sum(1) / (
            np.linalg.norm(velocity, axis=1) * np.linalg.norm(self.velocity, axis=1)
        )
        self.assertTrue(np.allclose(cosines, -1.0, atol=1e-5))

    def test_zero_makes_the_jvp_vanish_exactly(self):
        """J_phi.v is linear in v, so an exactly-zero v is what makes the LHS vanish and
        leaves the residual equal to the state term alone."""
        velocity, confidence = self.ablate("zero")
        self.assertTrue(np.array_equal(velocity, np.zeros_like(self.velocity)))
        self.assertEqual(velocity.dtype, self.velocity.dtype)
        self.assertTrue(np.array_equal(confidence, self.confidence))

    def test_shuffle_moves_confidence_with_its_own_velocity(self):
        velocity, confidence = self.ablate("shuffle")
        pairs_before = {(tuple(v), float(c)) for v, c in zip(self.velocity, self.confidence)}
        pairs_after = {(tuple(v), float(c)) for v, c in zip(velocity, confidence)}
        self.assertEqual(pairs_before, pairs_after)
        self.assertFalse(np.array_equal(velocity, self.velocity))

    def test_confidence_may_be_absent(self):
        """Checkpoint re-evaluation rebuilds the velocity but reports physics only, which
        never weights by confidence."""
        velocity, confidence = apply_velocity_ablation(
            self.velocity, None, {"kot_velocity_ablation": "reverse"},
        )
        self.assertIsNone(confidence)
        self.assertTrue(np.allclose(velocity, -self.velocity))


class CorruptVelocityTests(unittest.TestCase):
    """The dose-response arm must land exactly on the two arms already run at its ends.

    If x=0 and x=1 were merely close to the real and shuffled arms, the corruption sweep
    would be five unrelated runs rather than one axis with two known anchor points.
    """

    def setUp(self):
        rng = np.random.default_rng(0)
        self.velocity = rng.normal(size=(16, 4)).astype(np.float32)
        self.confidence = np.linspace(0.1, 1.0, 16, dtype=np.float32)

    def test_x_zero_is_the_real_field(self):
        velocity, confidence = corrupt_velocity(self.velocity, self.confidence, 0.0, seed=42)
        self.assertTrue(np.allclose(velocity, self.velocity))
        self.assertTrue(np.allclose(confidence, self.confidence))

    def test_x_one_reproduces_the_shuffle_arm_exactly(self):
        """Same permutation and same seed as apply_velocity_ablation('shuffle'), so the
        endpoint IS that arm rather than another draw of the same kind."""
        mixed, mixed_conf = corrupt_velocity(self.velocity, self.confidence, 1.0, seed=42)
        shuffled, shuffled_conf = apply_velocity_ablation(
            self.velocity, self.confidence, {"kot_velocity_ablation": "shuffle", "seed": 42},
        )
        self.assertTrue(np.allclose(mixed, shuffled))
        self.assertTrue(np.allclose(mixed_conf, shuffled_conf))

    def test_confidence_rides_the_same_x_as_the_velocity(self):
        x = 0.25
        perm = velocity_row_permutation(16, seed=42)
        velocity, confidence = corrupt_velocity(self.velocity, self.confidence, x, seed=42)
        # Confidence is a weight in [0, 1], not a direction, so it is mixed but never
        # rescaled -- scaling it would change how much each cell's residual counts.
        self.assertTrue(np.allclose(confidence,
                                    (1 - x) * self.confidence + x * self.confidence[perm]))
        # The velocity is that same mixture up to the one norm-restoring scalar, so it
        # stays parallel to it row by row.
        raw = (1 - x) * self.velocity + x * self.velocity[perm]
        scale = np.linalg.norm(velocity, axis=1) / np.linalg.norm(raw, axis=1)
        self.assertTrue(np.allclose(scale, scale[0]))
        self.assertTrue(np.allclose(velocity, raw * scale[0], atol=1e-5))

    def test_interior_keeps_the_median_norm_of_the_real_field(self):
        """Without this the mixture is ~0.71x as long at x=0.5, so a FOSCTTM dip in the
        middle of the sweep could be a magnitude effect rather than a direction one."""
        reference = float(np.median(np.linalg.norm(self.velocity, axis=1)))
        for x in (0.25, 0.5, 0.75):
            mixed, _ = corrupt_velocity(self.velocity, self.confidence, x, seed=42)
            self.assertAlmostEqual(
                float(np.median(np.linalg.norm(mixed, axis=1))), reference, places=4)

    def test_the_rescale_is_exactly_one_at_both_endpoints(self):
        """The correction must not perturb x=0 or x=1, or the sweep would no longer have
        the real and shuffle arms as its endpoints."""
        for x in (0.0, 1.0):
            mixed, _ = corrupt_velocity(self.velocity, self.confidence, x, seed=42)
            expected = self.velocity if x == 0.0 else self.velocity[
                velocity_row_permutation(16, seed=42)]
            self.assertTrue(np.allclose(mixed, expected, atol=1e-6))

    def test_intermediate_x_moves_monotonically_away_from_the_real_field(self):
        previous = 0.0
        for x in (0.25, 0.5, 0.75, 1.0):
            mixed, _ = corrupt_velocity(self.velocity, self.confidence, x, seed=42)
            distance = float(np.linalg.norm(mixed - self.velocity))
            self.assertGreater(distance, previous)
            previous = distance

    def test_dispatcher_rejects_a_corruption_stacked_on_another_arm(self):
        with self.assertRaises(ValueError):
            apply_velocity_ablation(self.velocity, self.confidence,
                                    {"kot_velocity_corrupt": 0.5,
                                     "kot_velocity_ablation": "reverse"})

    def test_dispatcher_rejects_x_outside_the_unit_interval(self):
        for bad in (-0.1, 1.5):
            with self.assertRaises(ValueError):
                apply_velocity_ablation(self.velocity, self.confidence,
                                        {"kot_velocity_corrupt": bad})

    def test_dispatcher_routes_x_and_leaves_the_real_field_at_zero(self):
        velocity, _ = apply_velocity_ablation(
            self.velocity, self.confidence, {"kot_velocity_corrupt": 0.0, "seed": 42})
        self.assertTrue(np.array_equal(velocity, self.velocity))
        velocity, _ = apply_velocity_ablation(
            self.velocity, self.confidence, {"kot_velocity_corrupt": 0.5, "seed": 42})
        self.assertFalse(np.allclose(velocity, self.velocity))


class BranchAccuracyTests(unittest.TestCase):
    """Progenitor cells sit before the bifurcation and were never ambiguous, so they must
    not be allowed to inflate the metric the branch stage exists to measure."""

    def test_perfect_matching_scores_one(self):
        labels = np.array(["Progenitor", "Branch_A", "Branch_B", "Branch_A"])
        out = branch_accuracy_scores(labels, labels)
        self.assertAlmostEqual(out["branch_accuracy"], 1.0)
        self.assertAlmostEqual(out["branch_accuracy_branched"], 1.0)
        self.assertEqual(out["branch_n_branched"], 3)

    def test_progenitors_are_excluded_from_the_branched_score(self):
        true = np.array(["Progenitor", "Progenitor", "Branch_A", "Branch_B"])
        pred = np.array(["Progenitor", "Progenitor", "Branch_B", "Branch_A"])
        out = branch_accuracy_scores(true, pred)
        self.assertAlmostEqual(out["branch_accuracy"], 0.5)      # the two progenitors
        self.assertAlmostEqual(out["branch_accuracy_branched"], 0.0)  # both branches swapped
        self.assertEqual(out["branch_n_branched"], 2)

    def test_a_trunk_only_stage_reports_nan_rather_than_a_fake_one(self):
        labels = np.array(["Trunk"] * 5)
        out = branch_accuracy_scores(labels, labels)
        self.assertAlmostEqual(out["branch_accuracy"], 1.0)
        self.assertTrue(np.isnan(out["branch_accuracy_branched"]))
        self.assertEqual(out["branch_n_branched"], 0)

    def test_no_labels_is_nan_not_an_error(self):
        out = branch_accuracy_scores(np.array([]), np.array([]))
        self.assertTrue(np.isnan(out["branch_accuracy"]))
        self.assertTrue(np.isnan(out["branch_accuracy_branched"]))


class PermuteSRowsTests(unittest.TestCase):
    """The mapping control must move only WHICH protein owns a gene row.

    S also decides how many proteins enter the kinetic residual and which genes' velocity
    reaches the Jacobian. If the permutation changed either, a gap against the real arm
    would confound the mapping with the size of the kinetics term.
    """

    def setUp(self):
        # 6 proteins x 8 genes: four mapped rows with different weights and sparsity,
        # two alignment-only proteins whose rows are all zero.
        self.S = np.zeros((6, 8), dtype=np.float32)
        self.S[0, 1] = 1.0
        self.S[1, 3] = 0.5
        self.S[2, 4] = 2.0
        self.S[2, 6] = 1.5
        self.S[4, 0] = 0.25
        self.permuted = permute_s_rows(self.S, seed=42)

    def test_the_rows_themselves_are_unchanged(self):
        before = sorted(tuple(row) for row in self.S)
        after = sorted(tuple(row) for row in self.permuted)
        self.assertEqual(before, after)

    def test_the_mapped_protein_set_is_unchanged(self):
        """A non-zero S row and kinetic_mask=True are the same condition
        (assert_no_orphan_s_rows), so preserving the mapped rows preserves the mask and
        the number of proteins in the residual."""
        mapped_before = np.nonzero((self.S != 0).any(axis=1))[0]
        mapped_after = np.nonzero((self.permuted != 0).any(axis=1))[0]
        self.assertTrue(np.array_equal(mapped_before, mapped_after))

    def test_the_gene_support_is_unchanged(self):
        """build_velocity_weight reads the COLUMN support of S to decide which genes'
        velocity may enter J_phi.v; a row permutation must leave it identical."""
        self.assertTrue(np.array_equal((self.S != 0).any(axis=0),
                                       (self.permuted != 0).any(axis=0)))

    def test_the_velocity_gene_filter_keeps_the_same_number_of_proteins(self):
        """kot_kinetics_require_velocity_gene drops proteins whose linked genes have no
        stable fit. It is a per-row test, so a permutation inside the mapped rows can
        change WHICH proteins survive but never HOW MANY."""
        velocity_genes = np.array([1, 1, 0, 0, 1, 0, 0, 0], dtype=bool)
        survivors = lambda S: int((S.astype(bool) & velocity_genes[None, :]).any(axis=1).sum())
        self.assertEqual(survivors(self.S), survivors(self.permuted))

    def test_the_assignment_actually_changes(self):
        self.assertFalse(np.array_equal(self.permuted, self.S))

    def test_unmapped_rows_stay_empty(self):
        """An alignment-only protein has no mapping to destroy; handing it one would add
        proteins to the residual instead of only rewiring the ones already in it."""
        for row in (3, 5):
            self.assertTrue(np.array_equal(self.permuted[row], np.zeros(8, dtype=np.float32)))

    def test_same_seed_reproduces_and_different_seeds_differ(self):
        self.assertTrue(np.array_equal(permute_s_rows(self.S, seed=7),
                                       permute_s_rows(self.S, seed=7)))
        differ = [not np.array_equal(permute_s_rows(self.S, seed=7),
                                     permute_s_rows(self.S, seed=s)) for s in range(8, 20)]
        self.assertTrue(any(differ))

    def test_shape_and_dtype_survive(self):
        self.assertEqual(self.permuted.shape, self.S.shape)
        self.assertEqual(self.permuted.dtype, self.S.dtype)


if __name__ == "__main__":
    unittest.main()
