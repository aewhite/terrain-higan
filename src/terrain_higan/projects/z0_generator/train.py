import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch.utils.tensorboard import SummaryWriter

from terrain_higan.projects.z0_generator.config import Z0GeneratorConfig
from terrain_higan.projects.z0_generator.export import export_z0_generator
from terrain_higan.projects.z0_generator.models import Z0Discriminator, Z0Generator
from terrain_higan.projects.z0_generator.stages import GANTrainStage, TensorBoardLogStage, Z0GeneratorStepData
from terrain_higan.projects.zn_map_upscaler.config import ZnUpscalerConfig
from terrain_higan.projects.zn_map_upscaler.generate_input_maps import generate_maps
from terrain_higan.tensorboard_utils.tensorboard_helpers import VALID_RUN_MARKER_FILE
from terrain_higan.training import PatchLoaderStage, TrainingLoop


def train_z0_generator(
    input_map_config: ZnUpscalerConfig | None = None,
    *,
    config: Z0GeneratorConfig | None = None,
) -> None:
    config = config or Z0GeneratorConfig()
    device = torch.device(config.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Z0 training requires CUDA; inference and preprocessing also support CPU")
    if input_map_config is None:
        input_map_config = ZnUpscalerConfig(z_level=1)
    if config.generated_size < config.patch_size:
        raise ValueError(
            f"generated_size must be at least patch_size, got {config.generated_size} and {config.patch_size}"
        )
    if config.critic_output_mean_loss_weight < 0.0:
        raise ValueError(
            "critic_output_mean_loss_weight must be non-negative, "
            f"got {config.critic_output_mean_loss_weight}"
        )

    generator = Z0Generator(output_clip_range=config.generator_output_clip_range).to(device)
    discriminator = Z0Discriminator(
        patch_size=config.patch_size,
        output_clip=config.discriminator_output_clip,
    ).to(device)

    _probe_generator_export(generator, config)

    generator_optimizer = torch.optim.AdamW(
        generator.parameters(),
        lr=config.generator_learning_rate,
        betas=config.generator_adam_betas,
    )
    discriminator_optimizer = torch.optim.AdamW(
        discriminator.parameters(),
        lr=config.discriminator_learning_rate,
        betas=config.discriminator_adam_betas,
    )

    writer = SummaryWriter(config.run_dir)
    (config.run_dir / VALID_RUN_MARKER_FILE).touch()
    try:
        stages = [
            PatchLoaderStage(
                z_level=config.z_level,
                patch_size=config.patch_size,
                patches_per_batch=config.discriminator_patches_per_batch,
                device=device,
            ),
            GANTrainStage(
                generator=generator,
                discriminator=discriminator,
                generator_optimizer=generator_optimizer,
                discriminator_optimizer=discriminator_optimizer,
                noise_size=config.noise_size,
                patch_size=config.patch_size,
                generator_adversarial_loss_weight=config.generator_adversarial_loss_weight,
                generator_elevation_loss_weight=config.generator_elevation_loss_weight,
                real_patch_shift_stddev=config.real_patch_shift_stddev,
                critic_output_mean_loss_weight=config.critic_output_mean_loss_weight,
                device=device,
            ),
            TensorBoardLogStage(
                writer=writer,
                patch_size=config.patch_size,
                log_every_steps=config.log_every_steps,
                image_every_steps=config.image_every_steps,
            ),
        ]
        TrainingLoop(
            stages=stages,
            step_data_factory=Z0GeneratorStepData,
            num_steps=config.num_steps,
        ).run()
        state_dict_path, pt2_path, onnx_path, metadata_path = export_z0_generator(generator, config)
        print(f"Exported z0 generator state dict: {state_dict_path}")
        print(f"Exported z0 generator PT2: {pt2_path}")
        print(f"Exported z0 generator ONNX: {onnx_path}")
        print(f"Exported z0 generator metadata: {metadata_path}")
        materialize_trained_z0_maps(config, input_map_config, device)
    finally:
        writer.close()


def _probe_generator_export(generator: Z0Generator, config: Z0GeneratorConfig) -> None:
    with TemporaryDirectory() as temporary_dir:
        probe_paths = export_z0_generator(
            generator,
            config,
            output_dir=Path(temporary_dir),
            update_latest=False,
        )
        for probe_path in probe_paths:
            if not probe_path.is_file() or probe_path.stat().st_size == 0:
                raise RuntimeError(f"Z0 generator export probe did not write {probe_path}")


def materialize_trained_z0_maps(
        config: Z0GeneratorConfig,
        input_map_config: ZnUpscalerConfig,
        device: torch.device,
) -> None:
    output_dir = input_map_config.maps_dir / "z0"
    generated_sample_count = generate_maps(
        z_level=0,
        level_name="z0",
        total_samples=input_map_config.input_map_total_samples,
        batch_size=input_map_config.input_map_generation_batch_size,
        file_batch_size=input_map_config.input_maps_per_batch,
        map_size=input_map_config.input_map_size,
        mode="rebuild",
        output_dir=output_dir,
        source_dir=None,
        model_dir=input_map_config.model_dir,
        z0_model_path=config.pt2_path,
        device=device,
        progress=True,
    )
    print(f"Materialized {generated_sample_count:,} z0 maps in {output_dir}")


def main(argv: list[str] | None = None) -> None:
    defaults = Z0GeneratorConfig()
    parser = argparse.ArgumentParser(description="Train the z0 generator (requires CUDA)")
    parser.add_argument("--device", default=defaults.device)
    parser.add_argument("--generated-size", type=int, default=defaults.generated_size)
    parser.add_argument("--num-steps", type=int, default=defaults.num_steps)
    args = parser.parse_args(argv)
    train_z0_generator(config=Z0GeneratorConfig(
        device=args.device, generated_size=args.generated_size, num_steps=args.num_steps,
    ))


if __name__ == "__main__":
    main()
