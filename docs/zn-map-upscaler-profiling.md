# Map Upscaler Profiling

On the CUDA training host, capture ordinary training with:

```bash
python -m terrain_higan.projects.zn_map_upscaler.train --z-level 3 --profile --profile-only
```

Replace `3` with the affected level. Defaults run 20 normal steps, discard 3
profiler warmup steps, and record steps 24-33. `--profile-only` stops after the
capture without saving trained checkpoints, updating `latest`, or rebuilding
output maps. Startup input-PDF analysis and the temporary export probe still run.
Without `--profile-only`, training continues after the one capture and finishes
with the usual exports and map generation. Profiling is off unless requested.

| Option | Default | Effect |
| --- | --- | --- |
| `--profile` | Off | Enable a CPU/CUDA trace |
| `--profile-wait-steps` | 20 | Normal steps before warmup |
| `--profile-warmup-steps` | 3 | Profiler steps whose measurements are discarded |
| `--profile-active-steps` | 10 | Training steps retained in the capture |
| `--profile-shapes` | Off | Record tensor shapes and group operator summaries by shape |
| `--profile-memory` | Off | Track tensor allocation/deallocation events |
| `--profile-stacks` | Off | Record operator source locations/call stacks |
| `--profile-only` | Off | Stop after capture; requires `--profile` |

Wait/warmup counts can be zero; active steps must be positive. The configured
training length must cover the whole window. Shapes, memory, and stacks add
overhead; enable them for a targeted second run.

To include default scalar/histogram logging at step 100:

```bash
python -m terrain_higan.projects.zn_map_upscaler.train --z-level 3 --profile --profile-only --profile-wait-steps 90 --profile-warmup-steps 3 --profile-active-steps 12
```

To include default image logging and the large-map preview at step 2500:

```bash
python -m terrain_higan.projects.zn_map_upscaler.train --z-level 3 --profile --profile-only --profile-wait-steps 2490 --profile-warmup-steps 3 --profile-active-steps 12
```

## Artifacts

The command prints the run's `profiling/` directory and its top operators. Each
capture writes files named `steps_<first>_<last>`:

- `.trace.json`: open in [Perfetto](https://ui.perfetto.dev/) to inspect CPU work,
  CUDA kernels, copies, synchronization, and stream overlap.
- `.device.txt` and `.cpu.txt`: top 50 entries ranked by self device/CPU time,
  with inclusive times and call counts.
- `.ranges.json`: named stage/operation timings, including separate generator
  and critic gradients, optimizer steps, and logging/preview work.
- `.metadata.json`: capture steps, configuration, versions, GPU, and CUDA kernel
  count. A trace saved during an exception is marked incomplete. It also records
  active-window wall time and, when the wait is at least as long as the active
  window, the last equally sized unrecorded wait window and their time ratio.

`ProfilerStep#0` corresponds to training step 1. Named ranges start with `zn/`.
The real-patch loader and input-map loader each use a separate CUDA stream.
Input maps use one coordinator with a sequential file/sample cursor and
`input_map_loading_workers` CPU staging workers (default 2). Distinct file buffers
are filled in parallel and consumed in order through two reusable pinned buffer
sets, with one batch prefetched ahead. Look for pinned HtoD
copies on the input-map stream overlapping training kernels. The single reader
advances sequentially through samples and files across batches, wrapping only
at the pool boundary; the two buffer sets share that cursor. Partially consumed
files remain on the device until exhausted without another read/upload. Main-thread ranges
do not include background worker internals. The real-patch loader's range includes
its next CPU-batch fetch; DataLoader worker internals are outside the main-process
trace. Startup PDF analysis, export probing, and final exports/materialization
are outside the capture.

## Interpretation

Start with the device summary, then locate the leading operators under named
ranges in the trace. Inspect convolution forward/backward, PDF selection, losses,
and transfers. Input maps use `input_maps/load_to_device` and `wait_for_prefetch`
for consuming the worker's batch, followed by `sample_metric_crops`,
`metric_means`, `pdf_selection`, and `gather_context` ranges. All crop processing happens on the training device;
selected map indices do not return to the CPU. CPU waits around PDF selection,
`.item()`, and image `.cpu()` transfers may reveal synchronization costs.
PDF selection caches its quantile-probability tensor by central fraction, device,
and dtype, so warmed-up steps do not repeatedly transfer those constants from
the host. Quantile calculation and boolean indexing can still synchronize.

Ranges nest and streams overlap: inclusive times are not additive. CPU range
duration is not GPU elapsed time. Compare throughput against an equivalent
unprofiled, warmed-up window with CUDA synchronization only at its boundaries;
The profiler measures those wall-time windows with synchronization only at their
boundaries and excludes trace export time. Its ratio is an overhead estimate:
startup effects or different logging events can affect the comparison. Keep batch
sizes and logging intervals fixed, and account for cache/training-state differences between early and late
captures. The trace does not measure hardware occupancy or prove whether a
kernel is compute- or bandwidth-limited.

If CUDA profiling is unavailable, or the exported trace contains no CUDA kernels,
the command reports an error instead of presenting the capture as a usable GPU
profile. In the latter case it preserves the CPU trace and summaries for diagnosis.

API details: [PyTorch profiler](https://docs.pytorch.org/docs/stable/profiler.html).
