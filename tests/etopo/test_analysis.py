import struct
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

import terrain_higan.etopo.analyze_patches as analysis
from terrain_higan.etopo.normalization import (
    ELEVATION_MEAN_M,
    ELEVATION_STD_M,
    UPSCALE_STATS,
    normalize_elevation,
    normalize_residuals,
    scale_patch_deviations,
)
from terrain_higan.projects.zn_map_upscaler.analyze_input_maps import analyze_input_maps


class AnalyzePatchesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def write_level(self, z, size, count=5):
        raw = np.arange(count * size**2, dtype=np.float32).reshape(count, 1, size, size) * 13 - 2000
        raw[0] = -5000
        directory = self.root / "dataset" / f"z{z}" / f"{size}x{size}"
        directory.mkdir(parents=True)
        raw[:3].tofile(directory / "shard-0000")
        raw[3:].tofile(directory / "shard-0001")
        return raw

    def scan(self, **kwargs):
        with patch.object(analysis, "write_analysis_report"), patch.object(analysis, "_print_constants"):
            return analysis.analyze_patches(
                base_dir=self.root / "dataset", output_dir=self.root / "output", device="cpu",
                batch_size=2, sample_size=4, normalization_sample_count=3, **kwargs,
            )

    def test_weighted_totals_include_final_batches_and_exclude_synthetic_copies(self):
        raw0 = self.write_level(0, 3, 4)
        raw1 = self.write_level(1, 4)
        raw10 = self.write_level(10, 10)
        result = self.scan()
        raw = torch.from_numpy(np.concatenate([part.ravel() for part in (raw0, raw1, raw10)]))
        stats = result.physical_total
        self.assertEqual(stats.patch_count, 14)
        self.assertEqual(stats.flat_patch_count, 3)
        self.assertEqual(stats.elevation_m.count, len(raw))
        self.assertAlmostEqual(stats.elevation_m.mean, raw.double().mean().item(), places=10)
        self.assertAlmostEqual(stats.elevation_m.stddev, raw.double().std(correction=0).item(), places=10)
        self.assertEqual(int(stats.elevation.histogram.counts.sum()), len(raw))
        self.assertEqual(int(stats.patch_means.histogram.counts.sum()), 14)
        self.assertEqual(int(stats.nonflat_patch_stds.histogram.counts.sum()), 11)
        self.assertEqual(stats.downsampled_patch_stds.moments.count, 14)
        self.assertEqual(stats.downsampled_to_original_std_ratios.moments.count, 11)
        self.assertEqual(stats.nonflat_downsampled_patch_stds.moments.count, 11)
        self.assertEqual(int(stats.patch_std_relationship.sum()), 11)
        self.assertEqual(result.z10_source.patch_count, 5)
        self.assertEqual(len(result.levels), 13)
        for level in result.levels:
            self.assertEqual(level.sample.sample_count, 4)
        self.assertFalse((self.root / "output/patch_analysis.json").exists())

    def test_synthetic_full_scans_and_fits_use_previous_synthetic_source(self):
        raw = self.write_level(10, 10)
        result = self.scan()
        normalized = normalize_elevation(torch.from_numpy(raw))
        source = normalized
        for index, level in enumerate(result.levels):
            target = normalized if index == 0 else scale_patch_deviations(normalized, analysis.SYNTHETIC_STD_SCALES[index - 1])
            residuals = target - F.interpolate(F.avg_pool2d(source, 2), scale_factor=2, mode="nearest")
            self.assertEqual(level.dataset.training_residuals.moments.minimum, residuals.min().item())
            self.assertEqual(level.dataset.training_residuals.moments.maximum, residuals.max().item())
            downsampled_stds = F.avg_pool2d(target.double(), 2).std(dim=(1, 2, 3), correction=0)
            self.assertAlmostEqual(level.dataset.downsampled_patch_stds.moments.mean, downsampled_stds.mean().item(), places=12)
            original_stds = target.double().std(dim=(1, 2, 3), correction=0)
            expected_ratios = downsampled_stds[1:] / original_stds[1:]
            self.assertAlmostEqual(level.dataset.downsampled_to_original_std_ratios.moments.mean, expected_ratios.mean().item(), places=12)
            expected = analysis._find_divisor(residuals[:4].flatten())
            self.assertEqual(level.sample.calibration.residual_divisor_estimate, expected)
            source = target

    def test_existing_normalization_is_used_while_estimates_are_reported(self):
        raw = self.write_level(10, 10)
        result = self.scan()
        expected = normalize_elevation(torch.from_numpy(raw))
        self.assertEqual(result.z10_source.elevation.moments.minimum, expected.min().item())
        self.assertEqual(result.metadata["existing_elevation_mean_m"], ELEVATION_MEAN_M)
        self.assertEqual(result.metadata["existing_elevation_std_m"], ELEVATION_STD_M)
        values = analysis.sample_elevation_values(self.root / "dataset", sample_count=3)
        self.assertEqual(result.normalization_estimate_m.mean, values.mean(dtype=np.float64))
        self.assertAlmostEqual(result.normalization_estimate_m.stddev, values.std(dtype=np.float64), places=10)
        c = result.levels[0].sample.calibration
        residuals = analysis._upscale_deltas(expected[:4]).flatten()
        existing = normalize_residuals(residuals, 10)
        self.assertEqual(c.existing_residual_divisor, UPSCALE_STATS[10].residual_divisor)
        self.assertAlmostEqual(c.existing_normalized_residuals.moments.stddev, existing.double().std(correction=0).item(), places=10)

    def test_normalization_sampling_is_seeded_and_uses_distinct_global_patch_ids(self):
        raw = self.write_level(10, 10)
        seed = 901
        rng = np.random.default_rng(seed)
        ids = rng.choice(5, size=4, replace=False)
        pixels = rng.integers(0, 100, size=4)
        expected = raw.reshape(5, 100)[ids, pixels]
        values = analysis.sample_elevation_values(self.root / "dataset", sample_count=4, seed=seed)
        np.testing.assert_array_equal(values, expected)
        mean, std = analysis.compute_normalization(self.root / "dataset", sample_count=4, seed=seed)
        self.assertEqual(mean, expected.mean(dtype=np.float64))
        self.assertEqual(std, expected.std(dtype=np.float64))
        with self.assertRaises(ValueError):
            analysis.sample_elevation_values(self.root / "dataset", sample_count=6)

    def test_partial_shard_is_rejected(self):
        path = self.root / "shard-0000"
        path.write_bytes(b"broken")
        with self.assertRaisesRegex(ValueError, "partial patch"):
            list(analysis._iter_raw_patch_batches([path], 4, 2))

    def test_flat_populations_and_zero_variance_downscales_have_no_fitted_divisor(self):
        patches = torch.ones(3, 1, 4, 4)
        sample = analysis._analyze_sample(patches, 1)
        self.assertIsNone(sample.calibration.residual_divisor_estimate)
        self.assertIsNone(sample.calibration.downscale_patch_divisor_estimate)
        stats = analysis.analyze_patch_batches([patches], z_level=1, patch_size=4).dataset
        self.assertEqual(stats.flat_patch_count, 3)
        self.assertEqual(stats.nonflat_patch_stds.moments.count, 0)
        self.assertEqual(stats.reconstruction_residuals.moments.count, 0)
        self.assertEqual(stats.training_residuals.moments.count, 48)
        self.assertEqual(stats.downsampled_patch_stds.moments.count, 3)
        self.assertEqual(stats.downsampled_patch_stds.moments.maximum, 0)
        self.assertEqual(stats.downsampled_to_original_std_ratios.moments.count, 0)
        self.assertEqual(int(stats.patch_std_relationship.sum()), 0)
        checkerboard = torch.tensor([[1., -1., 1., -1.], [-1., 1., -1., 1.]]).repeat(2, 1).reshape(1, 1, 4, 4)
        c = analysis._analyze_sample(checkerboard, 1).calibration
        self.assertIsNotNone(c.residual_divisor_estimate)
        self.assertIsNone(c.downscale_patch_divisor_estimate)
        self.assertAlmostEqual(c.candidate_normalized_residuals.moments.stddev, 1, places=6)

    def test_downsampling_std_retention_distinguishes_coarse_and_fine_variation(self):
        coarse = torch.tensor([[0., 2.], [4., 6.]], dtype=torch.float64).repeat_interleave(2, 0).repeat_interleave(2, 1)
        detail = torch.tensor([[1., -1.], [-1., 1.]], dtype=torch.float64).repeat(2, 2)
        patches = torch.stack((torch.full_like(coarse, 3), coarse, detail, coarse + detail)).unsqueeze(1)
        stats = analysis.analyze_patch_batches([patches], z_level=1, patch_size=4).dataset
        retained = (5 / 6) ** 0.5
        self.assertAlmostEqual(stats.downsampled_patch_stds.moments.mean, 5 ** 0.5 / 2)
        self.assertAlmostEqual(stats.downsampled_patch_stds.moments.stddev, 5 ** 0.5 / 2)
        self.assertEqual(stats.downsampled_to_original_std_ratios.moments.count, 3)
        self.assertEqual(stats.downsampled_to_original_std_ratios.moments.minimum, 0)
        self.assertAlmostEqual(stats.downsampled_to_original_std_ratios.moments.maximum, 1)
        self.assertAlmostEqual(stats.downsampled_to_original_std_ratios.moments.mean, (1 + retained) / 3)
        self.assertEqual(int(stats.downsampled_patch_stds.histogram.counts[0]), 2)
        self.assertEqual(int(stats.patch_std_relationship[:, 0].sum()), 1)
        self.assertEqual(int(stats.patch_std_relationship.sum()), 3)
        self.assertEqual(stats.nonflat_downsampled_patch_stds.moments.count, 3)
        self.assertAlmostEqual(stats.nonflat_downsampled_patch_stds.moments.mean, 2 * 5 ** 0.5 / 3)

        merged = analysis.PatchStats()
        for batch in patches.split(3):
            merged.merge(analysis.analyze_patch_batches([batch], z_level=1, patch_size=4).dataset)
        for name in ("downsampled_patch_stds", "nonflat_downsampled_patch_stds", "downsampled_to_original_std_ratios"):
            actual, expected = getattr(merged, name), getattr(stats, name)
            self.assertEqual(actual.moments.count, expected.moments.count)
            self.assertAlmostEqual(actual.moments.mean, expected.moments.mean)
            self.assertAlmostEqual(actual.moments.stddev, expected.moments.stddev)
            np.testing.assert_array_equal(actual.histogram.counts, expected.histogram.counts)
        np.testing.assert_array_equal(merged.patch_std_relationship, stats.patch_std_relationship)

    def test_downsampling_relationship_is_invariant_to_elevation_offset_and_scale(self):
        patches = torch.arange(32, dtype=torch.float64).reshape(2, 1, 4, 4)
        expected = analysis.analyze_patch_batches([patches], z_level=1, patch_size=4).dataset
        target = patches * 0.25 + 10
        actual = analysis.analyze_patch_batches(
            [target], z_level=11, patch_size=4, source_batches=[patches * 4],
        ).dataset
        self.assertAlmostEqual(actual.downsampled_patch_stds.moments.mean, expected.downsampled_patch_stds.moments.mean / 4)
        self.assertAlmostEqual(actual.downsampled_to_original_std_ratios.moments.mean, expected.downsampled_to_original_std_ratios.moments.mean)
        np.testing.assert_array_equal(actual.downsampled_to_original_std_ratios.histogram.counts, expected.downsampled_to_original_std_ratios.histogram.counts)

    def test_z0_pools_all_nine_pixels_and_preserves_residual_signs(self):
        patches = torch.arange(9, dtype=torch.float32).reshape(1, 1, 3, 3)
        stats = analysis.analyze_patch_batches([patches], z_level=0, patch_size=3).dataset
        residual = patches - patches.mean()
        self.assertEqual(stats.training_residuals.moments.minimum, residual.min().item())
        self.assertEqual(stats.reconstruction_residuals.moments.maximum, -residual.min().item())
        self.assertEqual(stats.reconstruction_residuals.moments.count, 9)
        self.assertEqual(stats.downsampled_patch_stds.moments.mean, 0)
        self.assertEqual(stats.downsampled_to_original_std_ratios.moments.mean, 0)
        self.assertIsNone(analysis._analyze_sample(patches, 0).calibration)

    def test_histogram_edge_counts_include_out_of_range_observations(self):
        histogram = analysis.Histogram.empty(np.array([-1., 0., 1.]))
        with patch.object(analysis, "HISTOGRAM_CHUNK_VALUES", 2):
            histogram.update(torch.tensor([-2., -1., 0., 1., 2.]))
        np.testing.assert_array_equal(histogram.counts, [2, 3])
        self.assertEqual(histogram.below_range, 1)
        self.assertEqual(histogram.above_range, 1)

    def test_residual_chart_uses_bin_percentages_and_trims_empty_tails(self):
        distribution = analysis.Distribution.empty(np.array([-100., -1., -1e-8, 0., 1e-8, 1., 100.]))
        distribution.update(torch.tensor([-0.5, 0., 0., 0.5]))
        plt = analysis._pyplot()
        figure, axis = plt.subplots()
        try:
            analysis._draw_distribution(axis, distribution, label="Residuals", percent=True)
            values, edges, _ = axis.patches[0].get_data()
            np.testing.assert_allclose(values, [25., 0., 50., 25.])
            np.testing.assert_allclose(edges, [-1., -1e-8, 0., 1e-8, 1.])
            self.assertEqual(axis.get_ylabel(), "Observations per bin (%)")
            analysis._set_distribution_scale(axis, distribution, "symlog")
            self.assertEqual(axis.get_xlim(), (-1., 1.))
            self.assertEqual(axis.get_yscale(), "log")
        finally:
            plt.close(figure)

    def test_std_comparison_overlays_matched_nonflat_populations_on_log_axes(self):
        coarse = torch.tensor([[0., 2.], [4., 6.]], dtype=torch.float64).repeat_interleave(2, 0).repeat_interleave(2, 1)
        detail = torch.tensor([[1., -1.], [-1., 1.]], dtype=torch.float64).repeat(2, 2)
        patches = torch.stack((torch.zeros_like(coarse), coarse, detail, coarse + detail)).unsqueeze(1)
        level = analysis.analyze_patch_batches([patches], z_level=1, patch_size=4)
        plt = analysis._pyplot()
        figure, axis = plt.subplots()
        try:
            analysis.write_patch_std_comparison(axis, level.dataset)
            self.assertEqual(axis.get_xscale(), "log")
            self.assertEqual(axis.get_yscale(), "log")
            self.assertEqual(axis.get_legend_handles_labels()[1], ["Original", "Downsampled"])
            original, downsampled = (artist.get_data()[0] for artist in axis.patches)
            self.assertAlmostEqual(original.sum(), 100)
            self.assertAlmostEqual(downsampled.sum(), 200 / 3)
            analysis.write_patch_std_trends(axis, [level], log_scale=True)
            self.assertEqual(axis.lines[-1].get_ydata()[0], level.dataset.nonflat_downsampled_patch_stds.moments.mean)
        finally:
            plt.close(figure)

    def test_chart_limits_follow_each_levels_observed_bins(self):
        plt = analysis._pyplot()
        for edges, scale, values in (
            (analysis.RESIDUAL_BINS, "symlog", [-0.001, 0., 0.001]),
            (analysis.RESIDUAL_BINS, "symlog", [-0.000001, 0., 0.000001]),
            (analysis.PATCH_STD_BINS, "log", [0.001, 0.002, 0.003]),
            (analysis.PATCH_STD_BINS, "log", [0.000001, 0.000002, 0.000003]),
        ):
            with self.subTest(scale=scale, values=values):
                distribution = analysis.Distribution.empty(edges)
                distribution.update(torch.tensor(values))
                figure, axis = plt.subplots()
                try:
                    analysis._draw_distribution(axis, distribution, label="Full scan", percent=True)
                    analysis._set_distribution_scale(axis, distribution, scale)
                    low, high = axis.get_xlim()
                    self.assertLessEqual(low, min(values))
                    self.assertGreaterEqual(high, max(values))
                    self.assertLessEqual(max(abs(low), abs(high)), max(map(abs, values)) * 1.3)
                    if scale == "log":
                        self.assertGreater(low, min(values) / 1.3)
                    else:
                        linthresh = axis.xaxis.get_transform().linthresh
                        self.assertLess(linthresh, max(map(abs, values)))
                        self.assertTrue(all(value == 0 or abs(value) >= linthresh for value in axis.get_xticks()))
                finally:
                    plt.close(figure)

    def test_report_names_distinguish_pooled_physical_and_z10_source(self):
        self.write_level(0, 3)
        self.write_level(10, 10)
        results = self.scan(z_levels=[0, 10])
        results.levels = [level for level in results.levels if level.z_level <= 10]
        obsolete = [self.root / f"dataset_{kind}_{index}.png" for kind in ("histograms", "metrics") for index in (0, 1)]
        for path in obsolete:
            path.write_bytes(b"old numbered report")
        analysis.write_analysis_report(results, self.root)
        for population in ("physical_dataset", "z10_source_dataset"):
            for kind in ("histograms", "metrics", "downsampling"):
                self.assertTrue((self.root / f"{population}_{kind}.png").read_bytes().startswith(b"\x89PNG"))
        for name in ("z0_downsampling", "z10_downsampling", "level_downsampling_0", "patch_std_comparison_log"):
            self.assertTrue((self.root / f"{name}.png").read_bytes().startswith(b"\x89PNG"))
        self.assertFalse(any(path.exists() for path in obsolete))

    def test_source_batches_must_match_targets(self):
        patches = torch.ones(1, 1, 4, 4)
        for sources in ([], [patches, patches], [torch.ones(2, 1, 4, 4)]):
            with self.assertRaises(ValueError):
                analysis.analyze_patch_batches([patches], z_level=1, patch_size=4, source_batches=sources)

    def test_pdf_artifacts_interpolation_tails_and_constant_samples(self):
        means = torch.tensor([-2., -1., 0., 2., 3.])
        result = analysis.analyze_patch_means(means, grid_size=128)
        pdf = result.pdf
        self.assertEqual(tuple(pdf.shape), (2, 128))
        self.assertAlmostEqual((pdf[1].sum() * (pdf[0, 1] - pdf[0, 0])).item(), 1, places=5)
        path = self.root / "pdf.pt"
        analysis.save_patch_mean_pdf(pdf, path)
        torch.testing.assert_close(analysis.load_patch_mean_pdf(path), pdf)
        evaluated = analysis.evaluate_patch_mean_pdf(pdf, torch.tensor([-100., 100.]))
        torch.testing.assert_close(evaluated, pdf[1, [0, -1]])
        midpoint = (pdf[0, 20] + pdf[0, 21]) / 2
        torch.testing.assert_close(analysis.evaluate_patch_mean_pdf(pdf, midpoint), pdf[1, 20:22].mean())
        constant = analysis.analyze_patch_means(torch.ones(3), grid_size=128)
        self.assertTrue(torch.isfinite(constant.pdf).all().item())
        with self.assertRaises(ValueError):
            analysis.analyze_patch_means(torch.empty(0))

    def test_map_analysis_reuses_normalized_means_and_combines_charts(self):
        directory = self.root / "maps/z11s"
        directory.mkdir(parents=True)
        np.save(directory / "maps_0000.npy", np.full((2, 1, 16, 16), 1.25, dtype=np.float32))
        with patch("terrain_higan.projects.zn_map_upscaler.analyze_input_maps.save_patch_mean_chart") as render:
            result = analyze_input_maps(
                z_level=12, maps_dir=self.root / "maps", output_dir=self.root / "output",
                source_level_name="z11s", output_level_name="z12s", patch_size=6,
                patches_per_map=4, seed=9, device="cpu",
            )
        self.assertEqual(result.sample_count, 8)
        self.assertEqual(result.pdf_path.name, "z12s_input_map_patch_mean_pdf.pt")
        self.assertEqual(result.histogram_path.name, "z12s_input_map_patch_mean_histogram.png")
        self.assertEqual(render.call_count, 1)
        self.assertEqual(render.call_args.args[0].distribution.moments.mean, 1.25)
        self.assertFalse((self.root / "output/z12s_input_map_patch_mean_pdf.png").exists())

    def test_training_import_does_not_load_matplotlib(self):
        probe = subprocess.run([
            sys.executable, "-c", "import sys; from terrain_higan.etopo.analyze_patches import evaluate_patch_mean_pdf; "
            "assert 'matplotlib' not in sys.modules",
        ], capture_output=True, text=True, env={
            **os.environ, "PYTHONPATH": str(Path(analysis.__file__).parents[2]),
        })
        self.assertEqual(probe.returncode, 0, probe.stderr)

    def test_selected_levels_and_missing_inputs(self):
        self.write_level(1, 4)
        result = self.scan(z_levels=[1])
        self.assertIsNone(result.normalization_estimate_m)
        self.assertIsNone(result.z10_source)
        with self.assertRaises(FileNotFoundError):
            self.scan(z_levels=[10])
        with self.assertRaises(ValueError):
            self.scan(z_levels=[11])



def test_analyze_dataset_uses_all_raw_patches(tmp_path) -> None:
    shard_path = tmp_path / "shard-0000"
    values = (0.0, 4.0, 8.0, 12.0, -10.0, -6.0, -2.0, 2.0)
    shard_path.write_bytes(struct.pack(f"{len(values)}f", *values))

    stats = analysis.analyze_patch_batches(
        (normalize_elevation(batch) for batch in analysis._iter_raw_patch_batches([shard_path], 2, 1)),
        z_level=1, patch_size=2,
    ).dataset
    normalized_patches = normalize_elevation(
        torch.tensor(values, dtype=torch.float32).reshape(2, 1, 2, 2)
    )
    raw_residuals = normalized_patches - normalized_patches.mean(dim=(2, 3), keepdim=True)

    assert stats.patch_count == 2
    assert stats.elevation.moments.minimum == normalized_patches.min().item()
    assert stats.elevation.moments.maximum == normalized_patches.max().item()
    assert stats.training_residuals.moments.minimum == raw_residuals.min().item()
    assert stats.training_residuals.moments.maximum == raw_residuals.max().item()


def test_report_excludes_flat_patches_and_replaces_old_pdf(tmp_path) -> None:
    base_dir = tmp_path / "dataset"
    for z_level, patch_size in ((0, 3), (1, 4)):
        shard_dir = base_dir / f"z{z_level}" / f"{patch_size}x{patch_size}"
        shard_dir.mkdir(parents=True)
        flat = np.full((patch_size, patch_size), 100.0, dtype=np.float32)
        varied = np.arange(patch_size**2, dtype=np.float32).reshape(patch_size, patch_size)
        np.stack((flat, flat, varied)).tofile(shard_dir / "shard-0000")

    output = tmp_path / "patch_analysis.pdf"
    output.write_bytes(b"old report")
    stats = analysis.analyze_patches(base_dir=base_dir, output_dir=tmp_path, batch_size=2, device="cpu").levels

    assert len(stats) == 2
    for level in stats:
        assert level.dataset.patch_count == 3
        assert level.dataset.flat_patch_count == 2
        assert level.dataset.nonflat_patch_stds.moments.minimum > 0
        assert level.dataset.nonflat_patch_stds.histogram.counts.sum() == 1
        assert level.dataset.reconstruction_residuals.histogram.counts.sum() == level.patch_size**2
    assert output.read_bytes().startswith(b"%PDF-")
    assert stats[0].dataset.downsampled_to_original_std_ratios.moments.mean == 0
    assert stats[1].dataset.downsampled_patch_stds.moments.count == 3

    pngs = list(tmp_path.glob("*.png"))
    assert pngs
    assert all(path.read_bytes().startswith(b"\x89PNG") for path in pngs)
