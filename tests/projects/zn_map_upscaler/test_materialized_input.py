from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from terrain_higan.projects.zn_map_upscaler.stages import (
    MaterializedInputMapStage,
    calculate_large_map_input_size,
    calculate_large_map_z0_noise_size,
    sample_patches_from_regions,
)


def test_large_map_z0_noise_size_targets_2048_output() -> None:
    for z_level in range(1, 11):
        noise_size = calculate_large_map_z0_noise_size(z_level, 2_048)
        map_size = noise_size - 16
        for upscaler_index in range(1, z_level + 1):
            map_size = min(
                map_size,
                calculate_large_map_input_size(z_level - upscaler_index + 1, 2_048),
            )
            map_size = 2 * map_size - 16

        assert noise_size >= 48
        assert map_size >= 2_048


@pytest.mark.parametrize("device", ("cpu", pytest.param("cuda", marks=pytest.mark.cuda)))
def test_materialized_stage_keeps_selected_metric_crop_and_context(tmp_path, device) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA host required")
    maps_dir = tmp_path / "maps" / "z0"
    maps_dir.mkdir(parents=True)
    maps = np.arange(4 * 13 * 13, dtype=np.float32).reshape(4, 1, 13, 13)
    np.save(maps_dir / "maps_000000.npy", maps)
    stage = MaterializedInputMapStage(
        maps_dir=tmp_path / "maps",
        z_level=1,
        maps_per_batch=4,
        metric_patch_size=3,
        input_map_patch_mean_pdf=torch.tensor(((-10.0, 10.0), (1.0, 1.0)), device=device),
        selected_patch_count=2,
        inverse_pdf_weight_central_fraction=0.95,
        inverse_pdf_weight_cap_stddevs=5.0,
        device=torch.device(device),
    )
    context = SimpleNamespace(step_data=SimpleNamespace())

    torch.manual_seed(7)
    stage.run(context)
    stage.close()

    input_maps = context.step_data.input_map
    metric_patches = context.step_data.input_metric_patches
    assert input_maps.shape == (2, 1, 13, 13)
    assert metric_patches.shape == (2, 1, 3, 3)
    assert input_maps.device.type == device
    assert metric_patches.device.type == device
    assert torch.equal(context.step_data.output_metric_rows, torch.zeros(2, dtype=torch.long, device=device))
    assert torch.equal(context.step_data.output_metric_columns, torch.zeros(2, dtype=torch.long, device=device))
    metric_rows = context.step_data.output_metric_rows // 2 + 5
    metric_columns = context.step_data.output_metric_columns // 2 + 5
    for input_map, metric_patch, row, column in zip(
            input_maps,
            metric_patches,
            metric_rows,
            metric_columns,
    ):
        assert torch.equal(input_map[..., row:row + 3, column:column + 3], metric_patch)


def test_output_patch_is_sampled_from_the_metric_crop_doubled_footprint() -> None:
    metric_patches = torch.arange(2 * 3 * 3, dtype=torch.float32).reshape(2, 1, 3, 3)
    output_footprints = F.interpolate(metric_patches, scale_factor=2, mode="nearest")
    output_maps = torch.zeros(2, 1, 48, 48)
    rows = torch.tensor((0, 42))
    columns = torch.tensor((2, 40))
    for output_map, footprint, row, column in zip(output_maps, output_footprints, rows, columns):
        output_map[..., row:row + 6, column:column + 6] = footprint

    patches = sample_patches_from_regions(
        output_maps,
        rows=rows,
        columns=columns,
        region_size=6,
        patch_size=4,
    )

    assert patches.shape == (2, 1, 4, 4)
    for patch, footprint in zip(patches, output_footprints):
        assert any(
            torch.equal(patch, footprint[..., row:row + 4, column:column + 4])
            for row in range(3)
            for column in range(3)
        )


@pytest.mark.parametrize("maps_per_batch", (1, 2, 4, 5, 8, 12))
@pytest.mark.parametrize("device", ("cpu", pytest.param("cuda", marks=pytest.mark.cuda)))
def test_materialized_stage_batches_independently_of_file_sizes(tmp_path, maps_per_batch, device) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA host required")
    source = tmp_path / "z0"
    source.mkdir()
    maps = np.broadcast_to(
        np.arange(5, dtype=np.float32).reshape(5, 1, 1, 1), (5, 1, 13, 13),
    ).copy()
    np.save(source / "maps_000000.npy", maps[:2])
    np.save(source / "maps_000001.npy", maps[2:])
    stage = MaterializedInputMapStage(
        maps_dir=tmp_path,
        z_level=1,
        maps_per_batch=maps_per_batch,
        metric_patch_size=3,
        input_map_patch_mean_pdf=torch.tensor(((-10.0, 10.0), (1.0, 1.0)), device=device),
        selected_patch_count=maps_per_batch,
        inverse_pdf_weight_central_fraction=0.95,
        inverse_pdf_weight_cap_stddevs=5.0,
        device=torch.device(device),
    )
    context = SimpleNamespace(step_data=SimpleNamespace())
    with patch(
        "terrain_higan.projects.zn_map_upscaler.stages.select_pdf_weighted_values",
        return_value=torch.arange(maps_per_batch, device=device),
    ), patch(
        "torch.randint", side_effect=lambda high, size, **kwargs: torch.zeros(
            size, dtype=torch.long, **kwargs,
        ),
    ):
        stage.run(context)
        stage.close()

    input_maps = context.step_data.input_map
    metric_patches = context.step_data.input_metric_patches
    assert input_maps.shape == (maps_per_batch, 1, 13, 13)
    assert metric_patches.shape == (maps_per_batch, 1, 3, 3)
    expected_indices = np.arange(maps_per_batch) % len(maps)
    torch.testing.assert_close(input_maps, torch.from_numpy(maps[expected_indices]).to(device))
    torch.testing.assert_close(metric_patches, input_maps[..., 5:8, 5:8])
    np.testing.assert_array_equal(np.load(source / "maps_000000.npy"), maps[:2])
    np.testing.assert_array_equal(np.load(source / "maps_000001.npy"), maps[2:])


@pytest.mark.parametrize("device", ("cpu", pytest.param("cuda", marks=pytest.mark.cuda)))
def test_materialized_stage_gathers_crops_on_device_across_files(tmp_path, device) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA host required")
    source = tmp_path / "z0"
    source.mkdir()
    maps = np.arange(5 * 19 * 21, dtype=np.float32).reshape(5, 1, 19, 21)
    np.save(source / "maps_000000.npy", maps[:3])
    np.save(source / "maps_000001.npy", maps[3:])
    stage = MaterializedInputMapStage(
        maps_dir=tmp_path,
        z_level=1,
        maps_per_batch=7,
        metric_patch_size=3,
        input_map_patch_mean_pdf=torch.tensor(((-10.0, 10.0), (1.0, 1.0)), device=device),
        selected_patch_count=4,
        inverse_pdf_weight_central_fraction=0.95,
        inverse_pdf_weight_cap_stddevs=5.0,
        device=torch.device(device),
    )
    rows = torch.tensor((5, 6, 7, 8, 9, 10, 11), device=device)
    columns = torch.tensor((13, 12, 11, 10, 9, 8, 7), device=device)
    source_indices = np.array((0, 1, 2, 3, 4, 0, 1))
    selected_indices = torch.tensor((6, 3, 5, 0), device=device)
    context = SimpleNamespace(step_data=SimpleNamespace())
    with patch.object(
        stage, "_sample_metric_coordinates", return_value=(rows, columns),
    ), patch(
        "terrain_higan.projects.zn_map_upscaler.stages.select_pdf_weighted_values",
        return_value=selected_indices,
    ) as selection, patch(
        "terrain_higan.projects.zn_map_upscaler.stages.np.concatenate",
        side_effect=AssertionError("Map batches must be assembled on the training device"),
    ), patch.object(
        torch.Tensor, "cpu", side_effect=AssertionError("Selections must stay on device"),
    ), patch.object(
        torch.Tensor, "numpy", side_effect=AssertionError("Crops must stay in PyTorch"),
    ):
        stage.run(context)
        stage.close()

    expected_metrics = np.stack([
        maps[source_index, :, row:row + 3, column:column + 3]
        for source_index, row, column in zip(source_indices, rows.tolist(), columns.tolist())
    ])
    torch.testing.assert_close(
        selection.call_args.args[0], torch.from_numpy(expected_metrics.mean(axis=(1, 2, 3))).to(device),
    )
    rows_cpu = rows.tolist()
    columns_cpu = columns.tolist()
    expected_contexts = np.stack([
        maps[source_indices[index], :, rows_cpu[index] - 5:rows_cpu[index] + 8,
             columns_cpu[index] - 5:columns_cpu[index] + 8]
        for index in selected_indices.tolist()
    ])
    torch.testing.assert_close(context.step_data.input_map, torch.from_numpy(expected_contexts).to(device))
    torch.testing.assert_close(
        context.step_data.input_metric_patches,
        torch.from_numpy(expected_metrics[selected_indices.tolist()]).to(device),
    )
