from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from terrain_higan.etopo.normalization import normalize_elevation
from terrain_higan.paths import DEFAULT_MODELS_DIR, DEFAULT_RUNS_DIR

GENERATOR_KERNEL_SIZES = (5, 5, 5, 3, 3)
GENERATOR_SHRINK = sum(kernel_size - 1 for kernel_size in GENERATOR_KERNEL_SIZES)
Z0_OUTPUT_RANGE_M = (-6146.4, 5075.4)
GENERATOR_STATE_DICT_FILENAME = "generator.safetensors"
GENERATOR_PT2_FILENAME = "generator.pt2"
GENERATOR_ONNX_FILENAME = "generator.onnx"
GENERATOR_METADATA_FILENAME = "metadata.json"


def default_run_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return DEFAULT_RUNS_DIR / f"z0_{timestamp}"


def default_generator_output_clip_range() -> tuple[float, float]:
    output_min, output_max = Z0_OUTPUT_RANGE_M
    return (
        float(normalize_elevation(output_min)),
        float(normalize_elevation(output_max)),
    )


@dataclass(frozen=True)
class Z0GeneratorConfig:
    device: str = "cuda"
    z_level: int = 0
    patch_size: int = 3
    generated_size: int = 188
    num_steps: int = 100_000
    generator_learning_rate: float = 1e-4
    discriminator_learning_rate: float = 1e-4
    generator_adam_betas: tuple[float, float] = (0.0, 0.9)
    discriminator_adam_betas: tuple[float, float] = (0.0, 0.9)
    generator_adversarial_loss_weight: float = 1.0
    generator_elevation_loss_weight: float = 0.1
    real_patch_shift_stddev: float = 0.1
    critic_output_mean_loss_weight: float = 1.0
    discriminator_output_clip: float = 10.0
    log_every_steps: int = 100
    image_every_steps: int = 2500
    run_dir: Path = field(default_factory=default_run_dir)
    model_dir: Path = DEFAULT_MODELS_DIR / "z0_generator"
    generator_output_clip_range: tuple[float, float] = field(
        default_factory=default_generator_output_clip_range,
    )

    @property
    def noise_size(self) -> int:
        return self.generated_size + GENERATOR_SHRINK

    @property
    def model_run_dir(self) -> Path:
        return self.model_dir / self.run_dir.name

    @property
    def pt2_path(self) -> Path:
        return self.model_run_dir / GENERATOR_PT2_FILENAME

    @property
    def state_dict_path(self) -> Path:
        return self.model_run_dir / GENERATOR_STATE_DICT_FILENAME

    @property
    def onnx_path(self) -> Path:
        return self.model_run_dir / GENERATOR_ONNX_FILENAME

    @property
    def metadata_path(self) -> Path:
        return self.model_run_dir / GENERATOR_METADATA_FILENAME

    @property
    def patches_per_batch(self) -> int:
        patch_grid_size = self.generated_size - self.patch_size + 1
        return patch_grid_size * patch_grid_size

    @property
    def discriminator_patches_per_batch(self) -> int:
        return self.patches_per_batch // (self.patch_size * self.patch_size)
