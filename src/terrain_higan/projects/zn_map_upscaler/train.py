import argparse
import gc
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch.utils.tensorboard import SummaryWriter

from terrain_higan.etopo.analyze_patches import load_patch_mean_pdf
from terrain_higan.etopo.dataloader import DEFAULT_ETOPO_BASE_DIR
from terrain_higan.projects.zn_map_upscaler.analyze_input_maps import analyze_input_maps
from terrain_higan.projects.zn_map_upscaler.config import ZnUpscalerConfig
from terrain_higan.projects.zn_map_upscaler.generate_input_maps import generate_maps
from terrain_higan.projects.zn_map_upscaler.models import ZnUpscalerCritic, ZnUpscalerModel
from terrain_higan.projects.zn_map_upscaler.profiling import (
    ProfiledStage,
    run_training_loop,
    validate_profile_device,
)
from terrain_higan.projects.zn_map_upscaler.save import load_zn_map_upscaler, save_zn_map_upscaler
from terrain_higan.projects.zn_map_upscaler.stages import (
    MaterializedInputMapStage,
    ScaleRealPatchesStage,
    ZnUpscalerLargeMapPreviewStage,
    ZnUpscalerLossTrendStage,
    ZnUpscalerStepData,
    ZnUpscalerTensorBoardStage,
    ZnUpscalerTrainStage,
)
from terrain_higan.tensorboard_utils.tensorboard_helpers import VALID_RUN_MARKER_FILE
from terrain_higan.training import PatchLoaderStage, TrainingLoop


def train_zn_map_upscaler(config: ZnUpscalerConfig) -> None:
    _train_and_save(config)
    if config.profile_only:
        return
    # Drop training models, optimizers, loop batches, and prefetchers before inference.
    gc.collect()
    device = torch.device(config.device)
    if device.type == "cuda":
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
    materialize_trained_maps(config, device)


def _train_and_save(config: ZnUpscalerConfig) -> None:
    validate_profile_device(config)
    device = torch.device(config.device)
    real_shard_dir = (
            DEFAULT_ETOPO_BASE_DIR / f"z{config.real_z_level}"
            / f"{config.patch_size}x{config.patch_size}"
    )
    if not any(real_shard_dir.glob("shard-*")):
        raise FileNotFoundError(f"No real patch shards found in {real_shard_dir}")
    if not config.patch_mean_pdf_path.is_file():
        raise FileNotFoundError(config.patch_mean_pdf_path)
    analyze_training_input_maps(config, device)
    patch_mean_pdf = load_patch_mean_pdf(config.patch_mean_pdf_path, device=device)
    input_map_patch_mean_pdf = load_patch_mean_pdf(
        config.input_map_patch_mean_pdf_path,
        device=device,
    )
    model = ZnUpscalerModel(
        z_level=config.stats_z_level,
        layer_channels=config.model_layer_channels,
    ).to(device)
    critic = ZnUpscalerCritic(
        z_level=config.stats_z_level,
        layer_specs=config.critic_layer_specs,
    ).to(device)
    _probe_model_save(model, config)
    model_optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=config.adam_betas,
    )
    critic_optimizer = torch.optim.AdamW(
        critic.parameters(),
        lr=config.learning_rate,
        betas=config.adam_betas,
    )
    writer = SummaryWriter(config.run_dir)
    (config.run_dir / VALID_RUN_MARKER_FILE).touch()
    input_map_stage = None
    try:
        loss_trend = ZnUpscalerLossTrendStage(
            config.loss_lookback_steps, minimum_steps=config.minimum_steps,
        )
        input_map_stage = MaterializedInputMapStage(
            maps_dir=config.maps_dir,
            z_level=config.z_level,
            source_level_name=config.predecessor_level_name,
            maps_per_batch=config.input_maps_per_batch,
            loading_workers=config.input_map_loading_workers,
            metric_patch_size=config.input_metric_patch_size,
            input_map_patch_mean_pdf=input_map_patch_mean_pdf,
            selected_patch_count=config.patches_per_batch,
            inverse_pdf_weight_central_fraction=config.inverse_pdf_weight_central_fraction,
            inverse_pdf_weight_cap_stddevs=config.inverse_pdf_weight_cap_stddevs,
            device=device,
        )
        stages = [
            input_map_stage,
            PatchLoaderStage(
                z_level=config.real_z_level,
                patch_size=config.patch_size,
                patches_per_batch=config.input_maps_per_batch,
                device=device,
            ),
        ]
        if config.super_scaler:
            stages.append(ScaleRealPatchesStage(config.z_level))
        stages.extend([
            ZnUpscalerTrainStage(
                model=model,
                critic=critic,
                z_level=config.stats_z_level,
                model_optimizer=model_optimizer,
                critic_optimizer=critic_optimizer,
                patch_mean_pdf=patch_mean_pdf,
                patch_size=config.patch_size,
                patches_per_batch=config.patches_per_batch,
                residual_aggregate_loss_weight=config.generator_residual_aggregate_loss_weight,
                residual_bounds_loss_weight=config.generator_residual_bounds_loss_weight,
                residual_block_mean_loss_weight=(
                    config.generator_residual_block_mean_loss_weight
                    * config.level_config.residual_block_mean_loss_multiplier
                ),
                residual_limits=config.residual_limits,
                inverse_pdf_weight_central_fraction=config.inverse_pdf_weight_central_fraction,
                inverse_pdf_weight_cap_stddevs=config.inverse_pdf_weight_cap_stddevs,
            ),
            loss_trend,
            ZnUpscalerTensorBoardStage(
                writer,
                z_level=config.stats_z_level,
                log_every_steps=config.log_every_steps,
                image_every_steps=config.image_every_steps,
                critic_extreme_patch_count=config.critic_extreme_patch_count,
            ),
            ZnUpscalerLargeMapPreviewStage(
                writer,
                z_level=config.z_level,
                model=model,
                z0_pt2_path=config.z0_pt2_path,
                predecessor_model_paths=config.predecessor_model_paths,
                image_every_steps=config.image_every_steps,
                device=device,
                target_size=config.large_map_preview_target_size,
                z0_minimum_noise_size=config.z0_minimum_noise_size,
            ),
        ])
        loop = TrainingLoop(
            stages=[ProfiledStage(stage) for stage in stages] if config.profile else stages,
            step_data_factory=ZnUpscalerStepData,
            num_steps=config.num_steps,
        )
        run_training_loop(loop, config, should_continue=loss_trend.should_continue)
        if config.profile_only:
            print("Profiling complete; skipping checkpoint export and output-map materialization")
            return
        model_path, pt2_path, onnx_path = save_zn_map_upscaler(model, config)
        print(f"Saved zn upscaler state dict: {model_path}")
        print(f"Exported zn upscaler PT2: {pt2_path}")
        print(f"Exported zn upscaler ONNX: {onnx_path}")
    finally:
        try:
            if input_map_stage is not None:
                input_map_stage.close()
        finally:
            writer.close()


def _probe_model_save(model: ZnUpscalerModel, config: ZnUpscalerConfig) -> None:
    with TemporaryDirectory() as temporary_dir:
        probe_paths = save_zn_map_upscaler(
            model,
            config,
            output_dir=Path(temporary_dir),
            update_latest=False,
        )
        for probe_path in probe_paths:
            if not probe_path.is_file() or probe_path.stat().st_size == 0:
                raise RuntimeError(f"Zn upscaler save probe did not write {probe_path}")
        load_zn_map_upscaler(
            probe_paths[0],
            z_level=config.stats_z_level,
            layer_channels=config.model_layer_channels,
        )


def analyze_training_input_maps(config: ZnUpscalerConfig, device: torch.device) -> None:
    analysis = analyze_input_maps(
        z_level=config.z_level,
        maps_dir=config.maps_dir,
        output_dir=config.analysis_dir,
        source_level_name=config.predecessor_level_name,
        output_level_name=config.level_name,
        patch_size=config.input_metric_patch_size,
        device=device,
    )
    print(f"Saved {config.level_name} input-map PDF: {analysis.pdf_path}")


def materialize_trained_maps(config: ZnUpscalerConfig, device: torch.device) -> None:
    output_dir = config.output_map_dir
    source_dir = config.input_map_source_dir
    generated_sample_count = generate_maps(
        z_level=config.z_level,
        level_name=config.level_name,
        total_samples=config.input_map_total_samples,
        batch_size=config.input_map_generation_batch_size,
        file_batch_size=config.input_maps_per_batch,
        map_size=config.input_map_size,
        mode="rebuild",
        output_dir=output_dir,
        source_dir=source_dir,
        model_dir=config.model_dir,
        z0_model_path=config.z0_pt2_path,
        device=device,
        progress=True,
    )
    print(f"Materialized {generated_sample_count:,} {config.level_name} maps in {output_dir}")


def main(argv: list[str] | None = None) -> None:
    defaults = ZnUpscalerConfig()
    parser = argparse.ArgumentParser(description="Train one map-upscaler level")
    parser.add_argument("--device", default=defaults.device)
    parser.add_argument("--num-steps", type=int, default=defaults.num_steps)
    parser.add_argument(
        "--input-maps-per-batch", type=int, default=defaults.input_maps_per_batch,
        help="Candidate count for predecessor maps and real ETOPO patches",
    )
    parser.add_argument(
        "--patches-per-batch", type=int, default=defaults.patches_per_batch,
        help="Selected patches per real/generated population for generator and critic training",
    )
    parser.add_argument("--input-map-loading-workers", type=int, default=defaults.input_map_loading_workers)
    parser.add_argument(
        "--input-map-generation-batch-size", type=int, default=defaults.input_map_generation_batch_size,
        help="Inference batch size for materializing maps after training",
    )
    parser.add_argument("--z-level", type=int, default=defaults.z_level)
    parser.add_argument("--real-dataset-z-level", type=int)
    parser.add_argument(
        "--critic-extreme-patch-count", type=int, default=16,
        help="Patches per highest/lowest real/fake critic-score image grid",
    )
    parser.add_argument(
        "--minimum-steps", type=int, default=defaults.minimum_steps,
        help="Minimum training steps before loss trends can stop training",
    )
    parser.add_argument(
        "--loss-lookback-steps", type=int, default=10_000,
        help="Steps per loss-mean window for early stopping (requires two complete windows)",
    )
    parser.add_argument("--profile", action="store_true", help="Capture CPU/CUDA training activity")
    parser.add_argument("--profile-wait-steps", type=int, default=20, help="Normal steps before profiler warmup")
    parser.add_argument("--profile-warmup-steps", type=int, default=3, help="Profiler warmup steps, discarded")
    parser.add_argument("--profile-active-steps", type=int, default=10, help="Steps retained in the trace")
    parser.add_argument("--profile-shapes", action="store_true", help="Record operator input shapes")
    parser.add_argument("--profile-memory", action="store_true", help="Record tensor allocations/deallocations")
    parser.add_argument("--profile-stacks", action="store_true", help="Record operator source call stacks")
    parser.add_argument("--profile-only", action="store_true", help="Stop after capture without saving trained models or rebuilding maps")
    args = parser.parse_args(argv)
    if torch.device(args.device).type != "cuda" or not torch.cuda.is_available():
        parser.error("Upscaler training requires CUDA; inference and map generation also support CPU")
    train_zn_map_upscaler(
        ZnUpscalerConfig(
            device=args.device,
            num_steps=args.num_steps,
            input_maps_per_batch=args.input_maps_per_batch,
            patches_per_batch=args.patches_per_batch,
            input_map_loading_workers=args.input_map_loading_workers,
            input_map_generation_batch_size=args.input_map_generation_batch_size,
            z_level=args.z_level,
            real_dataset_z_level=args.real_dataset_z_level,
            critic_extreme_patch_count=args.critic_extreme_patch_count,
            minimum_steps=args.minimum_steps,
            loss_lookback_steps=args.loss_lookback_steps,
            profile=args.profile,
            profile_wait_steps=args.profile_wait_steps,
            profile_warmup_steps=args.profile_warmup_steps,
            profile_active_steps=args.profile_active_steps,
            profile_shapes=args.profile_shapes,
            profile_memory=args.profile_memory,
            profile_stacks=args.profile_stacks,
            profile_only=args.profile_only,
        )
    )


if __name__ == "__main__":
    main()
