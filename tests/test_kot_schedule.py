import unittest

import torch

from src.training.kot import (
    dyn_weight_factor,
    lr_schedule_factor,
    shuffled_batches,
    subsample_rows,
)


class DynWeightFactorTests(unittest.TestCase):
    """The kinetics curriculum: a linear ramp to full lambda_dyn over W epochs."""

    def test_zero_warmup_is_full_strength_from_epoch_one(self):
        self.assertEqual(dyn_weight_factor(1, 0), 1.0)
        self.assertEqual(dyn_weight_factor(500, 0), 1.0)

    def test_ramp_reaches_full_strength_at_epoch_w_not_w_plus_one(self):
        self.assertEqual(dyn_weight_factor(1, 50), 0.02)
        self.assertEqual(dyn_weight_factor(25, 50), 0.5)
        self.assertEqual(dyn_weight_factor(50, 50), 1.0)
        self.assertEqual(dyn_weight_factor(51, 50), 1.0)

    def test_no_monitored_epoch_sees_a_partial_lambda_dyn(self):
        # run_kot gates checkpointing and early stopping on `epoch > W`. Every monitored
        # epoch must therefore be at full strength: while the weight is still ramping the
        # align loss is the lowest it will ever be, precisely because nothing is pulling
        # against it, so a best_align recorded there would select a model carrying far
        # less dynamics than it trained with -- and it would rank WELL on FOSCTTM.
        for warmup in (0, 50, 100, 200):
            for epoch in range(warmup + 1, warmup + 200):
                self.assertEqual(dyn_weight_factor(epoch, warmup), 1.0)
            if warmup:
                self.assertLess(dyn_weight_factor(warmup - 1, warmup), 1.0)


class LrScheduleFactorTests(unittest.TestCase):
    def test_defaults_reproduce_plain_cosine_annealing(self):
        # warmup=0, start=1, min=0 is the CosineAnnealingLR(T_max=n_epochs) this replaced.
        self.assertAlmostEqual(lr_schedule_factor(0, 0, 500, 1.0, 0.0), 1.0)
        self.assertAlmostEqual(lr_schedule_factor(250, 0, 500, 1.0, 0.0), 0.5)
        self.assertAlmostEqual(lr_schedule_factor(500, 0, 500, 1.0, 0.0), 0.0)

    def test_locked_schedule_warms_up_then_decays_to_the_floor(self):
        # The shape every sweep line now carries: 10-epoch warmup from 0.1, cosine to 0.01.
        self.assertAlmostEqual(lr_schedule_factor(0, 10, 500, 0.1, 0.01), 0.1)
        self.assertAlmostEqual(lr_schedule_factor(5, 10, 500, 0.1, 0.01), 0.55)
        self.assertAlmostEqual(lr_schedule_factor(10, 10, 500, 0.1, 0.01), 1.0)
        self.assertAlmostEqual(lr_schedule_factor(500, 10, 500, 0.1, 0.01), 0.01)


class TrainingRowTests(unittest.TestCase):
    """Held-out cells must not reach either loss term.

    Both losses draw their cells through these two helpers, so a validation cell
    entering training would have to come through one of them -- which is what makes
    these two tests the actual guarantee behind the split.
    """

    def setUp(self):
        self.device = torch.device("cpu")
        self.n = 1000
        self.val_idx = torch.arange(0, self.n, 10)                    # every 10th cell
        keep = torch.ones(self.n, dtype=torch.bool)
        keep[self.val_idx] = False
        self.train_idx = torch.nonzero(keep, as_tuple=False).flatten()

    def test_batches_cover_every_training_cell_exactly_once(self):
        batches = shuffled_batches(self.train_idx.numel(), 128, self.device, self.train_idx)
        seen = torch.cat(batches)
        self.assertEqual(seen.numel(), self.train_idx.numel())
        torch.testing.assert_close(torch.sort(seen).values, self.train_idx)

    def test_batches_and_subsamples_never_contain_a_held_out_cell(self):
        val = set(self.val_idx.tolist())
        for batch in shuffled_batches(self.train_idx.numel(), 128, self.device, self.train_idx):
            self.assertFalse(val & set(batch.tolist()))
        for _ in range(20):
            picks = subsample_rows(self.train_idx.numel(), 64, self.device, self.train_idx)
            self.assertEqual(picks.numel(), 64)
            self.assertFalse(val & set(picks.tolist()))

    def test_no_split_means_every_cell(self):
        # train_idx=None is the no-holdout path, and it must be the identity on rows:
        # the full-batch alignment step relies on it to skip gathering (n, D_r).
        batches = shuffled_batches(self.n, 128, self.device, None)
        torch.testing.assert_close(
            torch.sort(torch.cat(batches)).values, torch.arange(self.n)
        )


if __name__ == "__main__":
    unittest.main()
