import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch

from terrain_higan.projects.zn_map_upscaler.config import ZnUpscalerConfig
from terrain_higan.projects.zn_map_upscaler.profiling import run_training_loop
from terrain_higan.projects.zn_map_upscaler.stages import (
    ZnUpscalerLossTrendStage,
    ZnUpscalerStepData,
)
from terrain_higan.training import TrainingLoop


class LossStage:
    def __init__(self, losses):
        self._losses = losses

    def run(self, context):
        critic_loss, generator_loss = self._losses[context.step_number - 1]
        context.step_data.critic_loss = torch.tensor(critic_loss, requires_grad=True)
        context.step_data.generator_adversarial_loss = torch.tensor(
            generator_loss, requires_grad=True,
        )


class ZnUpscalerLossTrendTest(unittest.TestCase):
    def run_losses(self, losses, lookback_steps=4, minimum_steps=0):
        trend = ZnUpscalerLossTrendStage(lookback_steps, minimum_steps=minimum_steps)
        loop = TrainingLoop([LossStage(losses), trend], ZnUpscalerStepData, len(losses))
        config = ZnUpscalerConfig(
            num_steps=len(losses), loss_lookback_steps=lookback_steps,
            minimum_steps=minimum_steps,
        )
        with patch("builtins.print"):
            run_training_loop(loop, config, should_continue=trend.should_continue)
        return loop, trend

    def test_either_improvement_keeps_training(self):
        for second_window in ((2.0, 2.0), (0.0, 0.0), (2.0, 0.0)):
            with self.subTest(second_window=second_window):
                losses = [(1.0, 1.0)] * 4 + [second_window] * 4 + [(1.0, 1.0)]
                loop, trend = self.run_losses(losses)
                self.assertEqual(loop.context.step_number, 9)
                self.assertTrue(trend.should_continue(loop.context))

    def test_neither_improvement_stops_at_window_boundary(self):
        for second_window in ((1.0, 1.0), (0.0, 2.0), (1.0, 2.0), (0.0, 1.0)):
            with self.subTest(second_window=second_window):
                loop, trend = self.run_losses(
                    [(1.0, 1.0)] * 4 + [second_window] * 4 + [(2.0, 0.0)] * 4,
                )
                self.assertEqual(loop.context.step_number, 8)
                self.assertFalse(trend.should_continue(loop.context))

    def test_noisy_window_means_override_last_step_direction(self):
        loop, trend = self.run_losses(
            [(0.0, 0.0)] * 4
            + [(10.0, 10.0)] * 3 + [(-20.0, 20.0)]
            + [(0.0, 0.0)],
        )
        self.assertEqual(loop.context.step_number, 9)
        self.assertTrue(trend.should_continue(loop.context))
        self.assertEqual(trend._previous_means, (2.5, 12.5))

    def test_each_window_compares_to_immediately_previous_window(self):
        losses = [(1.0, 1.0)] * 4 + [(2.0, 0.0)] * 4 + [(1.5, 0.5)] * 4
        loop, trend = self.run_losses(losses + [(3.0, -1.0)] * 4)
        self.assertEqual(loop.context.step_number, 12)
        self.assertFalse(trend.should_continue(loop.context))
        self.assertFalse(trend._critic_sum.requires_grad)
        self.assertFalse(trend._generator_sum.requires_grad)

    def test_warmup_partial_windows_and_hard_cap(self):
        for step_count in (1, 4, 7):
            with self.subTest(step_count=step_count):
                loop, trend = self.run_losses([(1.0, 1.0)] * step_count)
                self.assertEqual(loop.context.step_number, step_count)
                self.assertTrue(trend.should_continue(loop.context))
        loop, trend = self.run_losses([(1.0, 1.0), (2.0, 0.0)], lookback_steps=1)
        self.assertEqual(loop.context.step_number, 2)
        self.assertTrue(trend.should_continue(loop.context))

    def test_invalid_lookback_is_rejected(self):
        for lookback in (0, -1):
            with self.subTest(lookback=lookback):
                with self.assertRaises(ValueError):
                    ZnUpscalerConfig(loss_lookback_steps=lookback)
                with self.assertRaises(ValueError):
                    ZnUpscalerLossTrendStage(lookback)

    def test_no_stopping_before_minimum(self):
        loop, trend = self.run_losses([(1.0, 1.0)] * 16, minimum_steps=12)
        self.assertEqual(loop.context.step_number, 12)
        self.assertFalse(trend.should_continue(loop.context))

    def test_minimum_between_boundaries_waits_for_complete_window(self):
        loop, trend = self.run_losses([(1.0, 1.0)] * 16, minimum_steps=10)
        self.assertEqual(loop.context.step_number, 12)
        self.assertFalse(trend.should_continue(loop.context))

    def test_improvement_at_minimum_overrides_earlier_worsening(self):
        losses = [(1.0, 1.0)] * 4 + [(0.0, 2.0)] * 4 + [(2.0, 0.0)] * 4
        loop, trend = self.run_losses(losses + [(1.0, 1.0)] * 4, minimum_steps=12)
        self.assertEqual(loop.context.step_number, 16)
        self.assertFalse(trend.should_continue(loop.context))

    def test_minimum_does_not_replace_two_window_warmup(self):
        loop, trend = self.run_losses([(1.0, 1.0)] * 12, minimum_steps=3)
        self.assertEqual(loop.context.step_number, 8)
        self.assertFalse(trend.should_continue(loop.context))

    def test_invalid_minimum(self):
        with self.assertRaises(ValueError):
            ZnUpscalerLossTrendStage(minimum_steps=-1)
        with self.assertRaises(ValueError):
            ZnUpscalerConfig(minimum_steps=-1)
        with self.assertRaises(ValueError):
            ZnUpscalerConfig(num_steps=40_000)
        ZnUpscalerConfig(profile=True, profile_only=True, num_steps=100)

    def test_profile_capture_completes_then_applies_stop_decision(self):
        with TemporaryDirectory() as directory:
            config = ZnUpscalerConfig(
                device="cpu", profile=True, profile_wait_steps=1,
                profile_warmup_steps=1, profile_active_steps=2, num_steps=8,
                loss_lookback_steps=1,
                minimum_steps=0,
            )
            object.__setattr__(config, "run_dir", Path(directory))
            trend = ZnUpscalerLossTrendStage(1, minimum_steps=0)
            loop = TrainingLoop(
                [LossStage([(1.0, 1.0)] * 8), trend], ZnUpscalerStepData, 8,
            )
            with patch("builtins.print"), patch(
                "terrain_higan.projects.zn_map_upscaler.profiling._save_capture",
            ) as save_capture:
                run_training_loop(loop, config, should_continue=trend.should_continue)
            self.assertEqual(loop.context.step_number, 4)
            save_capture.assert_called_once()


if __name__ == "__main__":
    unittest.main()
