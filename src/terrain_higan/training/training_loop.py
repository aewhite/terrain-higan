from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar


StepDataT = TypeVar("StepDataT")


@dataclass
class TrainingContext(Generic[StepDataT]):
    step_number: int
    step_data: StepDataT


class TrainingLoopStage(Protocol[StepDataT]):
    def run(self, context: TrainingContext[StepDataT]) -> None:
        ...


class TrainingLoop(Generic[StepDataT]):
    def __init__(
        self,
        stages: Sequence[TrainingLoopStage[StepDataT]],
        step_data_factory: Callable[[], StepDataT],
        num_steps: int,
    ) -> None:
        from terrain_higan.training.progress_indicator_stage import ProgressIndicatorStage

        self._num_steps = num_steps
        self._stages = list(stages)
        self._stages.append(ProgressIndicatorStage(num_steps))
        self._step_data_factory = step_data_factory
        self._context = TrainingContext(
            step_number=0,
            step_data=self._step_data_factory(),
        )

    @property
    def context(self) -> TrainingContext[StepDataT]:
        return self._context

    def run(
        self,
        should_continue: Callable[[TrainingContext[StepDataT]], bool] | None = None,
    ) -> None:
        if should_continue is None:
            should_continue = lambda context: True

        while self._context.step_number < self._num_steps and should_continue(self._context):
            self._context.step_number += 1
            self._context.step_data = self._step_data_factory()

            for stage in self._stages:
                stage.run(self._context)
