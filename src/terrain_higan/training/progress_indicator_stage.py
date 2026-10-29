from __future__ import annotations

import sys
import time
from datetime import timedelta
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from terrain_higan.training.training_loop import TrainingContext


class ProgressIndicatorStage:
    """Displays an animated progress indicator for training steps."""

    def __init__(self, num_steps: int):
        """Initialize the progress indicator with spinner animation."""
        self._spinner_chars = "|/-\\"
        self._spinner_idx = 0
        self._start_time = time.perf_counter()
        self._last_update_time = time.perf_counter()
        self._num_steps = num_steps

    def run(self, context: TrainingContext[Any]) -> None:
        """Update and display the progress indicator for the current step."""
        current_time = time.perf_counter()
        step_number = context.step_number
        if current_time - self._last_update_time < 1.0 and step_number < self._num_steps:
            return

        self._last_update_time = current_time

        spinner_char = self._spinner_chars[self._spinner_idx % len(self._spinner_chars)]
        self._spinner_idx += 1

        progress = step_number / self._num_steps * 100

        elapsed_time = current_time - self._start_time
        elapsed_timedelta = timedelta(seconds=int(elapsed_time))
        elapsed_str = str(elapsed_timedelta)
        step_rate = step_number / elapsed_time

        sys.stdout.write(
            f"\r[{spinner_char}] Step {step_number}/{self._num_steps} "
            f"({progress:.1f}%) | Time: {elapsed_str} | "
            f"Step Rate: {step_rate:.1f} steps/s"
        )
        if step_number >= self._num_steps:
            sys.stdout.write("\n")
        sys.stdout.flush()
