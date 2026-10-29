from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file

import terrain_higan.projects.zn_map_upscaler.analyze_input_maps as input_analysis
import terrain_higan.projects.zn_map_upscaler.generate_input_maps as input_maps
import terrain_higan.projects.zn_map_upscaler.stages as zn_stages
from terrain_higan.etopo.normalization import (
    SYNTHETIC_STD_SCALES,
    UPSCALE_STATS,
    normalize_residuals,
    scale_patch_deviations,
)
from terrain_higan.models.zn_training_constants import PATCH_SIZES
from terrain_higan.models.upscaling import mean_preserving_bilinear_2x
from terrain_higan.projects.zn_map_upscaler.config import Z_LEVEL_CONFIGS, ZnUpscalerConfig, ZnUpscalerLevelConfig
from terrain_higan.projects.zn_map_upscaler.models import ZnUpscalerCritic, ZnUpscalerModel
from terrain_higan.projects.zn_map_upscaler.geometry import center_crop
from terrain_higan.projects.zn_map_upscaler.save import load_zn_map_upscaler
from terrain_higan.projects.zn_map_upscaler.stages import (
    MaterializedInputMapStage,
    ScaleRealPatchesStage,
    ZnUpscalerStepData,
    ZnUpscalerTrainStage,
)
from terrain_higan.training.losses import residual_aggregate_loss


def test_super_config_keeps_real_data_and_generated_series_separate(tmp_path) -> None:
    paths = {
        "model_dir": tmp_path / "models",
        "maps_dir": tmp_path / "maps",
        "analysis_dir": tmp_path / "analysis",
    }
    normal = ZnUpscalerConfig(z_level=10, **paths)
    first = ZnUpscalerConfig(z_level=11, **paths)
    second = ZnUpscalerConfig(z_level=12, **paths)
    last = ZnUpscalerConfig(z_level=20, **paths)

    assert len(PATCH_SIZES) == 21
    assert PATCH_SIZES[11:] == [10] * 10
    assert [(level.level_name, level.super_scaler, level.real_z_level) for level in Z_LEVEL_CONFIGS[:10]] == [
        (f"z{z_level}", False, z_level) for z_level in range(1, 11)
    ]
    assert [(level.level_name, level.super_scaler, level.real_z_level) for level in Z_LEVEL_CONFIGS[10:]] == [
        (f"z{z_level}s", True, 10) for z_level in range(11, 21)
    ]
    assert ZnUpscalerLevelConfig(3, critic_layer_specs="4/3 2/1").level_name == "z3"
    assert ZnUpscalerLevelConfig(3, critic_layer_specs="4/3 2/1").super_scaler is False
    assert normal.model_run_dir.parent == paths["model_dir"] / "z10"
    assert normal.output_map_dir == paths["maps_dir"] / "z10"
    assert first.real_z_level == 10
    assert first.stats_z_level == 11
    assert second.stats_z_level == 12
    assert last.stats_z_level == 20
    assert first.patch_size == 10
    assert first.patch_mean_pdf_path == paths["analysis_dir"] / "z10_patch_mean_pdf.pt"
    assert first.input_map_patch_mean_pdf_path == paths["analysis_dir"] / "z11s_input_map_patch_mean_pdf.pt"
    assert first.input_map_source_dir == paths["maps_dir"] / "z10"
    assert first.output_map_dir == paths["maps_dir"] / "z11s"
    assert first.model_run_dir.parent == paths["model_dir"] / "z11s"
    assert first.run_dir.name.startswith("zn_map_upscaler_z11s_")
    assert first.predecessor_model_paths[-1] == paths["model_dir"] / "z10/latest/upscaler.safetensors"
    assert second.input_map_source_dir == paths["maps_dir"] / "z11s"
    assert second.predecessor_model_paths[-1] == paths["model_dir"] / "z11s/latest/upscaler.safetensors"
    assert last.predecessor_model_paths[-1] == paths["model_dir"] / "z19s/latest/upscaler.safetensors"
    assert last.residual_limits == (
        UPSCALE_STATS[20].dataset_minimum_raw_residual,
        UPSCALE_STATS[20].dataset_maximum_raw_residual,
    )


def test_super_real_override_accepts_only_z10() -> None:
    config = ZnUpscalerConfig(z_level=12, real_dataset_z_level=10)
    assert config.real_z_level == 10
    assert config.patch_size == 10
    assert config.input_metric_patch_size == 6
    assert config.critic_layer_specs == ZnUpscalerConfig(z_level=10).critic_layer_specs
    assert config.stats_z_level == 12

    for source_level in (0, 6, 11):
        try:
            ZnUpscalerConfig(z_level=11, real_dataset_z_level=source_level)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Accepted unavailable real dataset z{source_level}")


def test_super_checkpoint_loads_with_its_own_residual_statistics(tmp_path) -> None:
    input_map = torch.randn(1, 1, 16, 16)
    noise = torch.randn(1, 4, 16, 16)
    for z_level in (11, 20):
        model = ZnUpscalerModel(z_level=z_level, layer_channels=(8, 8, 4, 2)).eval()
        path = tmp_path / f"z{z_level}s" / "latest/upscaler.safetensors"
        path.parent.mkdir(parents=True)
        save_file(model.state_dict(), path)

        loaded = load_zn_map_upscaler(path)

        assert loaded._z_level == z_level
        assert loaded._residual_clipper.min_x == UPSCALE_STATS[z_level].dataset_minimum_raw_residual
        assert loaded._output_clipper.max_x == UPSCALE_STATS[z_level].dataset_maximum_value
        with torch.inference_mode():
            assert torch.equal(loaded(input_map, noise), model(input_map, noise))


def test_super_real_patch_stage_matches_analyzer_transform() -> None:
    patches = torch.arange(200, dtype=torch.float32).reshape(2, 1, 10, 10) / 100
    for z_level in range(11, 21):
        context = SimpleNamespace(step_data=SimpleNamespace(real_patches=patches.clone()))

        ScaleRealPatchesStage(z_level).run(context)

        scale = SYNTHETIC_STD_SCALES[z_level - 11]
        source_scale = 1.0 if z_level == 11 else SYNTHETIC_STD_SCALES[z_level - 12]
        source = patches if z_level == 11 else scale_patch_deviations(patches, source_scale)
        torch.testing.assert_close(context.step_data.real_source_patches, source)
        torch.testing.assert_close(patches, torch.arange(200).reshape(2, 1, 10, 10) / 100)
        torch.testing.assert_close(
            context.step_data.real_patches,
            scale_patch_deviations(patches, scale),
        )
        torch.testing.assert_close(
            context.step_data.real_patches.mean(dim=(1, 2, 3)),
            patches.mean(dim=(1, 2, 3)),
            rtol=0,
            atol=1e-6,
        )
        torch.testing.assert_close(
            context.step_data.real_patches.std(dim=(1, 2, 3), correction=0),
            patches.std(dim=(1, 2, 3), correction=0) * scale,
            rtol=1e-5,
            atol=1e-6,
        )


def test_statistic_loss_uses_previous_level_source_for_super_scalers(monkeypatch) -> None:
    observed_residuals = []

    def record_residual_loss(generated, real):
        observed_residuals.append((generated.detach(), real.detach()))
        return residual_aggregate_loss(generated, real)

    monkeypatch.setattr(zn_stages, "residual_aggregate_loss", record_residual_loss)
    patches = torch.linspace(-1.0, 1.0, 400).reshape(4, 1, 10, 10)
    for z_level in (1, 10, *range(11, 21)):
        model = ZnUpscalerModel(z_level=z_level, layer_channels=(8, 8, 4, 2))
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
        critic = ZnUpscalerCritic(z_level=z_level, layer_specs="4/3 2/1")
        stats = UPSCALE_STATS[z_level]
        stage = ZnUpscalerTrainStage(
            model=model,
            critic=critic,
            z_level=z_level,
            model_optimizer=torch.optim.SGD(model.parameters(), lr=0.0),
            critic_optimizer=torch.optim.SGD(critic.parameters(), lr=0.0),
            patch_mean_pdf=torch.tensor(((-2.0, 2.0), (1.0, 1.0))),
            patch_size=10,
            patches_per_batch=2,
            residual_aggregate_loss_weight=1.0,
            residual_bounds_loss_weight=1.0,
            residual_block_mean_loss_weight=1.0,
            residual_limits=(stats.dataset_minimum_raw_residual, stats.dataset_maximum_raw_residual),
            inverse_pdf_weight_central_fraction=0.95,
            inverse_pdf_weight_cap_stddevs=5.0,
        )
        step_data = ZnUpscalerStepData()
        step_data.real_patches = patches.clone()
        step_data.input_map = torch.zeros(2, 1, 16, 16)
        step_data.input_map[..., 5, 6] = 0.001
        step_data.input_metric_patches = torch.zeros(2, 1, 6, 6)
        step_data.output_metric_rows = torch.zeros(2, dtype=torch.long)
        step_data.output_metric_columns = torch.zeros(2, dtype=torch.long)
        context = SimpleNamespace(step_data=step_data)
        target = patches
        source = patches
        if z_level > 10:
            ScaleRealPatchesStage(z_level).run(context)
            target = scale_patch_deviations(patches, SYNTHETIC_STD_SCALES[z_level - 11])
            if z_level > 11:
                source = scale_patch_deviations(patches, SYNTHETIC_STD_SCALES[z_level - 12])
        expected = normalize_residuals(
            target - F.interpolate(
                F.avg_pool2d(source, 2),
                scale_factor=2,
                mode="nearest",
            ),
            z_level,
        )

        real_indices = torch.tensor([2, 0])
        with patch("torch.randint", return_value=torch.zeros(2, dtype=torch.long)), \
                patch.object(zn_stages, "select_pdf_weighted_patches", return_value=real_indices):
            stage.run(context)

        generated_residuals, real_residuals = observed_residuals[-1]
        bilinear_parent = mean_preserving_bilinear_2x(step_data.input_map)[..., 8:-8, 8:-8]
        nearest_parent = F.interpolate(
            step_data.input_map, scale_factor=2, mode="nearest",
        )[..., 8:-8, 8:-8]
        expected_generated_residuals = center_crop(bilinear_parent - nearest_parent, 12)[..., :10, :10]
        torch.testing.assert_close(
            generated_residuals, normalize_residuals(expected_generated_residuals, z_level),
        )
        torch.testing.assert_close(
            step_data.generator_residual_block_mean_loss,
            ((bilinear_parent - nearest_parent).mean(dim=(-2, -1)) / stats.residual_divisor).square().mean(),
        )
        torch.testing.assert_close(real_residuals, expected[real_indices])
        torch.testing.assert_close(
            step_data.generator_residual_aggregate_loss,
            residual_aggregate_loss(generated_residuals, expected[real_indices]),
        )
        assert torch.isfinite(step_data.generator_loss)
        assert torch.isfinite(step_data.critic_loss)
        assert step_data.real_patches.shape[0] == 2
        assert real_residuals.shape[0] == 2


def test_materialized_stage_can_read_super_predecessor(tmp_path) -> None:
    source_dir = tmp_path / "z11s"
    source_dir.mkdir()
    np.save(source_dir / "maps_000000.npy", np.zeros((2, 1, 32, 32), dtype=np.float32))
    stage = MaterializedInputMapStage(
        maps_dir=tmp_path,
        z_level=12,
        source_level_name="z11s",
        maps_per_batch=2,
        metric_patch_size=6,
        input_map_patch_mean_pdf=torch.tensor(((-1.0, 1.0), (1.0, 1.0))),
        selected_patch_count=1,
        inverse_pdf_weight_central_fraction=0.95,
        inverse_pdf_weight_cap_stddevs=5.0,
        device=torch.device("cpu"),
    )
    context = SimpleNamespace(step_data=SimpleNamespace())

    stage.run(context)

    assert context.step_data.input_map.shape == (1, 1, 16, 16)
    assert context.step_data.input_metric_patches.shape == (1, 1, 6, 6)


def test_input_map_analysis_uses_super_source_and_output_label(tmp_path, monkeypatch) -> None:
    source_dir = tmp_path / "maps/z11s"
    source_dir.mkdir(parents=True)
    np.save(source_dir / "maps_000000.npy", np.ones((2, 1, 16, 16), dtype=np.float32))
    monkeypatch.setattr(input_analysis, "save_patch_mean_chart", lambda *args, **kwargs: None)

    result = input_analysis.analyze_input_maps(
        z_level=12,
        maps_dir=tmp_path / "maps",
        output_dir=tmp_path / "analysis",
        source_level_name="z11s",
        output_level_name="z12s",
        patch_size=6,
        device="cpu",
    )

    assert result.source_map_count == 2
    assert result.patch_size == 6
    assert result.pdf_path == tmp_path / "analysis/z12s_input_map_patch_mean_pdf.pt"
    assert result.pdf_path.is_file()


def test_generate_maps_uses_super_checkpoint(tmp_path, monkeypatch) -> None:
    class NearestUpscaler(nn.Module):
        def forward(self, input_map: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
            return F.interpolate(input_map, scale_factor=2, mode="nearest")

    source_dir = tmp_path / "maps/z10"
    source_dir.mkdir(parents=True)
    np.save(source_dir / "maps_000000.npy", np.full((2, 1, 32, 32), 7, dtype=np.float32))
    loaded_paths = []

    def load_model(path):
        loaded_paths.append(path)
        return NearestUpscaler()

    monkeypatch.setattr(input_maps, "load_zn_map_upscaler", load_model)
    output_dir = tmp_path / "maps/z11s"
    model_dir = tmp_path / "models"

    count = input_maps.generate_maps(
        z_level=11,
        level_name="z11s",
        total_samples=2,
        batch_size=2,
        file_batch_size=2,
        map_size=32,
        mode="rebuild",
        output_dir=output_dir,
        source_dir=source_dir,
        model_dir=model_dir,
        z0_model_path=tmp_path / "unused.pt2",
        device=torch.device("cpu"),
        progress=False,
    )

    assert count == 2
    assert loaded_paths == [model_dir / "z11s/latest/upscaler.safetensors"]
    assert np.array_equal(
        np.load(output_dir / "maps_000000.npy"),
        np.full((2, 1, 32, 32), 7, dtype=np.float32),
    )


def test_generate_maps_rejects_mismatched_level_name_before_rebuild(tmp_path) -> None:
    output_dir = tmp_path / "existing"
    output_dir.mkdir()
    marker = output_dir / "keep.txt"
    marker.write_text("keep")

    try:
        input_maps.generate_maps(
            z_level=11,
            level_name="z11",
            total_samples=1,
            batch_size=1,
            file_batch_size=1,
            map_size=32,
            mode="rebuild",
            output_dir=output_dir,
            source_dir=tmp_path / "missing",
            model_dir=tmp_path / "models",
            z0_model_path=tmp_path / "unused.pt2",
            device=torch.device("cpu"),
            progress=False,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Accepted a normal-series name for z11s")

    assert marker.read_text() == "keep"
