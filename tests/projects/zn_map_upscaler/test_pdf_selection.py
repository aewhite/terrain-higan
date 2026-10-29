import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch

from terrain_higan.etopo.analyze_patches import evaluate_patch_mean_pdf
from terrain_higan.projects.zn_map_upscaler.stages import (
    select_pdf_weighted_patches,
    select_pdf_weighted_values,
)


class PdfSelectionTest(unittest.TestCase):
    def test_zero_and_near_zero_pdf_densities_produce_finite_positive_weights(self):
        for dtype in (torch.float32, torch.float64):
            for densities in ((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1e-38, 1e-38, 1e-38)):
                with self.subTest(dtype=dtype, densities=densities):
                    pdf = torch.tensor(((-1.0, 0.0, 1.0), densities), dtype=dtype)
                    values = torch.linspace(-2, 2, 257, dtype=dtype)
                    with patch("torch.multinomial", wraps=torch.multinomial) as sample:
                        indices = select_pdf_weighted_values(
                            values, pdf, sample_count=16, central_fraction=0.95, cap_stddevs=5.0,
                        )
                    weights = sample.call_args.args[0]
                    self.assertTrue(torch.isfinite(weights).all())
                    self.assertTrue((weights > 0).all())
                    self.assertEqual(indices.unique().numel(), 16)

    def test_cached_probabilities_preserve_weights_and_seeded_selection(self):
        devices = ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",)
        for device in devices:
            for dtype in (torch.float32, torch.float64):
                for central_fraction in (0.5, 0.95, 1.0):
                    with self.subTest(device=device, dtype=dtype, central_fraction=central_fraction):
                        values = torch.linspace(-2, 2, 257, device=device, dtype=dtype)
                        pdf = torch.tensor(((-1.0, 0.0, 1.0), (0.01, 1.0, 0.2)), device=device, dtype=dtype)
                        weights = evaluate_patch_mean_pdf(pdf, values).reciprocal()
                        tail = (1.0 - central_fraction) / 2.0
                        bounds = torch.quantile(weights, torch.tensor((tail, 1.0 - tail), device=device, dtype=dtype))
                        central = weights[(weights >= bounds[0]) & (weights <= bounds[1])]
                        expected_weights = weights.clamp_max(central.mean() + 5 * central.std(correction=0))
                        with patch("torch.multinomial", return_value=torch.arange(16, device=device)) as sample:
                            select_pdf_weighted_values(
                                values, pdf, sample_count=16,
                                central_fraction=central_fraction, cap_stddevs=5.0,
                            )
                        torch.testing.assert_close(sample.call_args.args[0], expected_weights, rtol=0, atol=0)
                        torch.manual_seed(7)
                        expected_indices = torch.multinomial(expected_weights, 16, replacement=False)
                        torch.manual_seed(7)
                        indices = select_pdf_weighted_values(
                            values, pdf, sample_count=16,
                            central_fraction=central_fraction, cap_stddevs=5.0,
                        )
                        torch.testing.assert_close(indices, expected_indices)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA host required")
    def test_warmed_selection_does_not_copy_probabilities_from_host(self):
        values = torch.linspace(-1, 1, 256, device="cuda")
        patches = values.reshape(256, 1, 1, 1)
        pdf = torch.tensor(((-1.0, 0.0, 1.0), (0.1, 1.0, 0.2)), device="cuda")
        options = dict(sample_count=16, central_fraction=0.93, cap_stddevs=5.0)
        select_pdf_weighted_values(values, pdf, **options)
        select_pdf_weighted_patches(patches, pdf, **options)
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[
            torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA,
        ]) as profiler:
            for iteration in range(3):
                select_pdf_weighted_values(values, pdf, **options)
                select_pdf_weighted_patches(patches, pdf, **options)
            torch.cuda.synchronize()
        with TemporaryDirectory() as directory:
            trace_path = Path(directory) / "selection.trace.json"
            profiler.export_chrome_trace(str(trace_path))
            events = json.loads(trace_path.read_text())["traceEvents"]
        self.assertTrue(any(event.get("cat") == "kernel" for event in events))
        host_copies = [event for event in events if event.get("cat") == "gpu_memcpy"
                       and event["name"].startswith("Memcpy HtoD")]
        self.assertEqual(host_copies, [])


if __name__ == "__main__":
    unittest.main()
