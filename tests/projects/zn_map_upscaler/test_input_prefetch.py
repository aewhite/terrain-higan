import json
import threading
import unittest
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pytest
import torch

from terrain_higan.projects.zn_map_upscaler.stages import MaterializedInputMapStage


@pytest.mark.cuda
@unittest.skipUnless(torch.cuda.is_available(), "CUDA host required")
class MaterializedInputPrefetchTest(unittest.TestCase):
    def make_stage(self, root, maps_per_batch=7, device="cuda"):
        return MaterializedInputMapStage(
            maps_dir=root, z_level=1, maps_per_batch=maps_per_batch,
            metric_patch_size=3,
            input_map_patch_mean_pdf=torch.tensor(((-1.0, 1.0), (1.0, 1.0)), device=device),
            selected_patch_count=2, inverse_pdf_weight_central_fraction=0.95,
            inverse_pdf_weight_cap_stddevs=5.0, device=torch.device(device),
        )

    def write_maps(self, root, file_sizes=(4, 4, 4)):
        source = root / "z0"
        source.mkdir()
        sample_count = sum(file_sizes)
        maps = np.arange(sample_count * 32 * 32, dtype=np.float32).reshape(sample_count, 1, 32, 32)
        offset = 0
        for file_index, count in enumerate(file_sizes):
            np.save(source / f"maps_{file_index:06d}.npy", maps[offset:offset + count])
            offset += count
        return maps

    def test_prefetch_reuses_pinned_buffers_without_corrupting_batches(self):
        with TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            source_maps = self.write_maps(root)
            reads = []
            load = np.load

            def read_maps(path, **kwargs):
                reads.append(Path(path).name)
                return load(path, **kwargs)

            stack.enter_context(patch(
                "terrain_higan.projects.zn_map_upscaler.stages.np.load", side_effect=read_maps,
            ))
            stage = self.make_stage(root)
            stack.callback(stage.close)
            consumer = torch.cuda.Stream()
            batches = []
            pointers = None
            for step in range(24):
                with torch.cuda.stream(consumer):
                    maps, valid_rows, valid_columns = stage._load_map_batch()
                    self.assertEqual((valid_rows, valid_columns), (20, 20))
                    batches.append(maps.clone())
                if step == 1:
                    stage._prefetched_batch.result()
                    pointers = [[buffer.data_ptr() for buffer in slot] for slot in stage._pinned_buffers]
            stage._prefetched_batch.result()
            self.assertEqual(pointers, [[buffer.data_ptr() for buffer in slot] for slot in stage._pinned_buffers])
            self.assertEqual([len(slot) for slot in stage._pinned_buffers], [2, 2])
            self.assertTrue(all(buffer.is_pinned() for slot in stage._pinned_buffers for buffer in slot))
            consumer.synchronize()
            for step, batch in enumerate(batches):
                indices = (step * 7 + np.arange(7)) % 12
                torch.testing.assert_close(batch, torch.from_numpy(source_maps[indices]).cuda())
            self.assertEqual(reads, [f"maps_{index % 3:06d}.npy" for index in range(44)])

    def test_samples_and_file_reads_advance_linearly_across_batches(self):
        file_sizes = (3, 2, 7)
        for device in ("cpu", "cuda"):
            for batch_size in (2, 5, 7, 17):
                with self.subTest(device=device, batch_size=batch_size), TemporaryDirectory() as directory:
                    root = Path(directory)
                    source_maps = self.write_maps(root, file_sizes)
                    stage = self.make_stage(root, maps_per_batch=batch_size, device=device)
                    reads = []
                    load = np.load

                    def read_maps(path, **kwargs):
                        reads.append(Path(path).name)
                        return load(path, **kwargs)

                    try:
                        with patch(
                            "terrain_higan.projects.zn_map_upscaler.stages.np.load", side_effect=read_maps,
                        ):
                            for step in range(13):
                                maps, _, _ = stage._load_map_batch()
                                indices = (step * batch_size + np.arange(batch_size)) % len(source_maps)
                                torch.testing.assert_close(maps, torch.from_numpy(source_maps[indices]).to(device))
                                batches_read = step + 1
                                if device == "cuda":
                                    stage._prefetched_batch.result()
                                    batches_read += 1
                                passes, tail = divmod(batches_read * batch_size, len(source_maps))
                                files_read = passes * len(file_sizes)
                                if tail:
                                    files_read += int(np.searchsorted(np.cumsum(file_sizes), tail)) + 1
                                self.assertEqual(reads, [
                                    f"maps_{index % len(file_sizes):06d}.npy" for index in range(files_read)
                                ])
                    finally:
                        stage.close()

    def test_uploads_are_pinned_and_use_a_different_stream_from_consumption(self):
        with TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            self.write_maps(root)
            stage = self.make_stage(root)
            stack.callback(stage.close)
            main_thread = threading.get_ident()
            read_threads = []
            load = np.load

            def read_maps(*args, **kwargs):
                read_threads.append(threading.get_ident())
                return load(*args, **kwargs)

            stack.enter_context(patch(
                "terrain_higan.projects.zn_map_upscaler.stages.np.load", side_effect=read_maps,
            ))
            with torch.profiler.profile(activities=[
                torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA,
            ]) as profiler:
                for step in range(4):
                    maps, _, _ = stage._load_map_batch()
                    maps.square().sum()
                stage.close()
                torch.cuda.synchronize()
            self.assertTrue(read_threads)
            self.assertTrue(all(thread != main_thread for thread in read_threads))
            trace_path = root / "prefetch.trace.json"
            profiler.export_chrome_trace(str(trace_path))
            events = json.loads(trace_path.read_text())["traceEvents"]
            copies = [event for event in events if event.get("cat") == "gpu_memcpy"
                      and event["args"].get("bytes") == 4 * 32 * 32 * 4]
            self.assertTrue(copies)
            self.assertTrue(all("Pinned -> Device" in event["name"] for event in copies))
            transfer_streams = {event["args"]["stream"] for event in copies}
            compute_streams = {event["args"]["stream"] for event in events if event.get("cat") == "kernel"}
            self.assertTrue(compute_streams - transfer_streams)

    def test_parallel_staging_preserves_order_when_files_finish_out_of_order(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source_maps = self.write_maps(root)
            stage = self.make_stage(root, maps_per_batch=8)
            first_started = threading.Event()
            second_finished = threading.Event()
            completions = []
            worker_threads = set()
            stage_file = stage._stage_file

            def delayed_stage(cpu_maps, pinned_maps):
                first_value = cpu_maps[0, 0, 0, 0].item()
                worker_threads.add(threading.get_ident())
                if first_value == 0:
                    first_started.set()
                    self.assertTrue(second_finished.wait(timeout=5))
                else:
                    self.assertTrue(first_started.wait(timeout=5))
                stage_file(cpu_maps, pinned_maps)
                completions.append(first_value)
                if first_value != 0:
                    second_finished.set()

            try:
                with patch.object(stage, "_stage_file", side_effect=delayed_stage):
                    maps, _, _ = stage._load_map_batch()
                    torch.testing.assert_close(maps, torch.from_numpy(source_maps[:8]).cuda())
                    stage.close()
            finally:
                stage.close()
            self.assertEqual(completions[:2], [4 * 32 * 32, 0])
            self.assertGreaterEqual(len(worker_threads), 2)

    def test_worker_errors_reach_the_caller_and_shutdown_is_safe(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_maps(root)
            stage = self.make_stage(root)
            try:
                with patch.object(stage, "_read_map_batch", side_effect=ValueError("bad map file")):
                    with self.assertRaisesRegex(ValueError, "bad map file"):
                        stage._load_map_batch()
            finally:
                stage.close()
            stage.close()
            self.assertIsNone(stage._executor)
            self.assertIsNone(stage._staging_executor)
            self.assertIsNone(stage._prefetched_batch)
            self.assertEqual(stage._pinned_buffers, [[], []])


if __name__ == "__main__":
    unittest.main()
