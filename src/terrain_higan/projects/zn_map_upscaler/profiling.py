"""Bounded training traces and optional named ranges for this project."""

import json
import time
from contextlib import nullcontext
from contextvars import ContextVar
from dataclasses import asdict

import torch

from terrain_higan.projects.zn_map_upscaler.config import ZnUpscalerConfig
from terrain_higan.training import TrainingLoop


_RANGES_ENABLED = ContextVar("zn_upscaler_profiling_ranges", default=False)
_RANGE_PREFIX = "zn/"


def profile_range(name: str):
    if _RANGES_ENABLED.get():
        return torch.profiler.record_function(_RANGE_PREFIX + name)
    return nullcontext()


class ProfiledStage:
    def __init__(self, stage) -> None:
        self._stage = stage

    def run(self, context) -> None:
        with profile_range(f"stage/{type(self._stage).__name__}"):
            self._stage.run(context)


def validate_profile_device(config: ZnUpscalerConfig) -> None:
    if config.profile and torch.device(config.device).type == "cuda":
        if not torch.cuda.is_available() or torch.profiler.ProfilerActivity.CUDA not in torch.profiler.supported_activities():
            raise RuntimeError("CUDA profiling is unavailable in this PyTorch build; a CPU-only trace cannot diagnose GPU training")


def run_training_loop(loop: TrainingLoop, config: ZnUpscalerConfig, should_continue=None) -> None:
    if not config.profile:
        loop.run(should_continue=should_continue)
        return

    validate_profile_device(config)
    device = torch.device(config.device)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    completed_steps = 0
    capture_saved = False
    baseline_start = None
    active_start = None
    timings = {}

    def timestamp() -> float:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return time.perf_counter()

    def save_capture(profiler) -> None:
        nonlocal capture_saved
        _save_capture(profiler, config, completed_steps, timings)
        capture_saved = True

    with torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(
            wait=config.profile_wait_steps,
            warmup=config.profile_warmup_steps,
            active=config.profile_active_steps,
            repeat=1,
        ),
        on_trace_ready=save_capture,
        record_shapes=config.profile_shapes,
        profile_memory=config.profile_memory,
        with_stack=config.profile_stacks,
    ) as profiler:
        token = _RANGES_ENABLED.set(False)
        try:
            def continue_capture(context) -> bool:
                nonlocal completed_steps, baseline_start, active_start
                # The loop checks this before the next step, after all previous stages.
                if context.step_number > completed_steps:
                    completed_steps = context.step_number
                    if completed_steps == config.profile_wait_steps and baseline_start is not None:
                        timings["baseline_seconds"] = timestamp() - baseline_start
                    if completed_steps == config.profile_total_steps and active_start is not None:
                        timings["active_seconds"] = timestamp() - active_start
                    profiler.step()
                # Use the last active-count steps of the unrecorded wait as a baseline.
                if completed_steps == config.profile_wait_steps - config.profile_active_steps and baseline_start is None:
                    baseline_start = timestamp()
                    timings["baseline_first_step"] = completed_steps + 1
                    timings["baseline_last_step"] = config.profile_wait_steps
                if completed_steps == config.profile_wait_steps + config.profile_warmup_steps and active_start is None:
                    active_start = timestamp()
                _RANGES_ENABLED.set(completed_steps >= config.profile_wait_steps)
                return completed_steps < config.profile_total_steps

            loop.run(should_continue=continue_capture)
            # At num_steps the shared loop exits without calling should_continue again.
            continue_capture(loop.context)
        finally:
            _RANGES_ENABLED.reset(token)

    if not capture_saved:
        raise RuntimeError("Training ended before the complete profiling window was saved")
    if not config.profile_only:
        loop.run(should_continue=should_continue)


def _save_capture(profiler, config: ZnUpscalerConfig, completed_steps: int, timings: dict) -> None:
    output_dir = config.run_dir / "profiling"
    output_dir.mkdir(parents=True, exist_ok=True)
    first_step = config.profile_wait_steps + config.profile_warmup_steps + 1
    stem = f"steps_{first_step:06d}_{completed_steps:06d}"
    trace_path = output_dir / f"{stem}.trace.json"
    profiler.export_chrome_trace(str(trace_path))
    averages = profiler.key_averages(group_by_input_shape=config.profile_shapes)
    # Recent PyTorch uses device time; older builds expose CUDA-specific names.
    device_time = "self_device_time_total" if hasattr(averages[0], "self_device_time_total") else "self_cuda_time_total"
    ranges = [
        {
            "name": event.key,
            "calls": event.count,
            "cpu_total_us": event.cpu_time_total,
            "self_cpu_total_us": event.self_cpu_time_total,
            "device_total_us": getattr(event, "device_time_total", getattr(event, "cuda_time_total", 0.0)),
            "self_device_total_us": getattr(event, device_time),
        }
        for event in averages
        if event.key.startswith(_RANGE_PREFIX) and event.device_type == torch.autograd.DeviceType.CPU
    ]
    # CUDA annotation spans include idle/wait time and are not kernel timings.
    # Keep their CPU counterparts for attribution; rank only operators/kernels.
    averages[:] = [event for event in averages if not event.is_user_annotation]
    cpu_table = averages.table(sort_by="self_cpu_time_total", row_limit=50)
    device_table = averages.table(sort_by=device_time, row_limit=50)
    (output_dir / f"{stem}.cpu.txt").write_text(cpu_table + "\n")
    (output_dir / f"{stem}.device.txt").write_text(device_table + "\n")
    (output_dir / f"{stem}.ranges.json").write_text(json.dumps({
        "note": "Ranges overlap/nest. Do not sum inclusive times or equate CPU durations with GPU elapsed time.",
        "ranges": ranges,
    }, indent=2) + "\n")
    device = torch.device(config.device)
    with trace_path.open() as trace_file:
        trace = json.load(trace_file)
    kernel_count = sum(event.get("cat") == "kernel" for event in trace["traceEvents"])
    metadata = {
        "config": asdict(config),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "patch_size": config.patch_size,
        "patches_per_batch": config.patches_per_batch,
        "candidate_patches_per_batch": config.input_maps_per_batch,
        "model_layer_channels": config.model_layer_channels,
        "critic_layer_specs": config.critic_layer_specs,
        "active_first_step": first_step,
        "active_last_step": completed_steps,
        "cuda_kernel_events": kernel_count,
        "complete": completed_steps == config.profile_total_steps,
        "profiler_step_offset": "ProfilerStep#0 corresponds to training step 1",
        "window_timings": timings,
    }
    if "baseline_seconds" in timings and "active_seconds" in timings:
        metadata["profile_to_baseline_time_ratio"] = timings["active_seconds"] / timings["baseline_seconds"]
    (output_dir / f"{stem}.metadata.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n")
    print(f"Profiling artifacts: {output_dir}")
    if "profile_to_baseline_time_ratio" in metadata:
        print(f"Profile/baseline wall-time ratio: {metadata['profile_to_baseline_time_ratio']:.2f}x "
              "(equal step counts; check logging events and startup effects)")
    print(averages.table(sort_by=device_time if device.type == "cuda" else "self_cpu_time_total", row_limit=10))
    if device.type == "cuda" and not kernel_count:
        raise RuntimeError(f"No CUDA kernels in {trace_path}; check CUPTI support/permissions. CPU trace and summaries were saved.")
