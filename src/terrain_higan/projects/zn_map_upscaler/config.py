from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from terrain_higan.etopo.normalization import SYNTHETIC_SOURCE_Z_LEVEL, UPSCALE_STATS
from terrain_higan.models.zn_training_constants import PATCH_SIZES
from terrain_higan.paths import DEFAULT_DATA_DIR, DEFAULT_MODELS_DIR, DEFAULT_RUNS_DIR

METRIC_CONTEXT_MARGIN = 4
LARGE_MAP_PREVIEW_TARGET_SIZE = 2_048
Z0_MINIMUM_NOISE_SIZE = 48
MODEL_FILENAME = "upscaler.safetensors"
PT2_FILENAME = "upscaler.pt2"
ONNX_FILENAME = "upscaler.onnx"


def default_run_dir(level_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return DEFAULT_RUNS_DIR / f"zn_map_upscaler_{level_name}_{timestamp}"


@dataclass(frozen=True)
class ZnUpscalerLevelConfig:
    """Settings that identify one z(n-1) to zn transition."""

    z_level: int
    critic_layer_specs: str
    super_scaler: bool = False
    real_dataset_z_level: int | None = None
    model_layer_channels: tuple[int, ...] = (256, 256, 32, 16)
    residual_block_mean_loss_multiplier: float = 1.0

    @property
    def level_name(self) -> str:
        return f"z{self.z_level}{'s' if self.super_scaler else ''}"

    @property
    def real_z_level(self) -> int:
        return self.real_dataset_z_level if self.real_dataset_z_level is not None else self.z_level

    @property
    def patch_size(self) -> int:
        return PATCH_SIZES[self.z_level]


Z_LEVEL_CONFIGS = (
        [
            ZnUpscalerLevelConfig(1, critic_layer_specs="128/3 64/2 16/1", residual_block_mean_loss_multiplier=1.00),
            ZnUpscalerLevelConfig(2, critic_layer_specs="128/3 64/2 16/1", residual_block_mean_loss_multiplier=1.00),
            ZnUpscalerLevelConfig(3, critic_layer_specs="128/3 128/3 32/2 16/1", residual_block_mean_loss_multiplier=1.00),
            ZnUpscalerLevelConfig(4, critic_layer_specs="128/3 128/3 32/2 16/1", residual_block_mean_loss_multiplier=1.0),
            ZnUpscalerLevelConfig(5, critic_layer_specs="128/4 128/3 64/3 16/1"),
            ZnUpscalerLevelConfig(6, critic_layer_specs="128/4 128/3 64/3 16/1"),
            ZnUpscalerLevelConfig(7, critic_layer_specs="128/5 128/4 64/3 16/1"),
            ZnUpscalerLevelConfig(8, critic_layer_specs="128/5 128/4 64/3 16/1"),
            ZnUpscalerLevelConfig(9, critic_layer_specs="128/5 128/4 64/3 16/1"),
            ZnUpscalerLevelConfig(10, critic_layer_specs="128/5 128/4 64/3 16/1"),
        ] + [
            ZnUpscalerLevelConfig(
                z_level,
                super_scaler=True,
                real_dataset_z_level=SYNTHETIC_SOURCE_Z_LEVEL,
                critic_layer_specs="128/5 128/4 64/3 16/1",
            )
            for z_level in range(11, 21)
        ]
)


@dataclass(frozen=True)
class ZnUpscalerConfig:
    z_level: int = 1
    real_dataset_z_level: int | None = None
    device: str = "cuda"
    num_steps: int = 250_000
    minimum_steps: int = 60_000
    loss_lookback_steps: int = 10_000
    profile: bool = False
    profile_wait_steps: int = 20
    profile_warmup_steps: int = 3
    profile_active_steps: int = 10
    profile_shapes: bool = False
    profile_memory: bool = False
    profile_stacks: bool = False
    profile_only: bool = False
    learning_rate: float = 1e-4
    adam_betas: tuple[float, float] = (0.0, 0.9)
    generator_residual_aggregate_loss_weight: float = 1.0
    generator_residual_bounds_loss_weight: float = 1.00
    # Legacy name: this weight now applies to whole-map residual means.
    generator_residual_block_mean_loss_weight: float = 0.1
    inverse_pdf_weight_central_fraction: float = 0.95
    inverse_pdf_weight_cap_stddevs: float = 5.0
    log_every_steps: int = 100
    image_every_steps: int = 2500
    critic_extreme_patch_count: int = 16
    run_dir: Path = field(init=False)
    model_dir: Path = DEFAULT_MODELS_DIR / "zn_map_upscaler"
    maps_dir: Path = DEFAULT_DATA_DIR / "maps"
    analysis_dir: Path = DEFAULT_DATA_DIR / "analysis"
    # Candidate count for both predecessor maps and real ETOPO patches.
    input_maps_per_batch: int = 8_192 * 2
    # Selected patches per population, shared by generator and critic training.
    patches_per_batch: int = 512
    input_map_loading_workers: int = 2
    input_map_total_samples: int = 1_048_576
    # Inference batch size for materializing map pools after training.
    input_map_generation_batch_size: int = 512
    input_map_size: int = 32
    large_map_preview_target_size: int = LARGE_MAP_PREVIEW_TARGET_SIZE
    z0_minimum_noise_size: int = Z0_MINIMUM_NOISE_SIZE

    def __post_init__(self) -> None:
        if self.input_map_loading_workers <= 0:
            raise ValueError("input_map_loading_workers must be positive")
        if self.critic_extreme_patch_count <= 0:
            raise ValueError("critic_extreme_patch_count must be positive")
        if self.minimum_steps < 0:
            raise ValueError("minimum_steps must be non-negative")
        if not self.profile_only and self.minimum_steps > self.num_steps:
            raise ValueError("num_steps must cover minimum_steps")
        if self.loss_lookback_steps <= 0:
            raise ValueError("loss_lookback_steps must be positive")
        if self.profile_only and not self.profile:
            raise ValueError("profile_only requires profile=True")
        if self.profile:
            if self.profile_wait_steps < 0 or self.profile_warmup_steps < 0 or self.profile_active_steps <= 0:
                raise ValueError("profile wait/warmup steps must be non-negative and active steps positive")
            if self.num_steps < self.profile_total_steps:
                raise ValueError("num_steps must cover the complete profiling window")
        if not 1 <= self.z_level <= len(Z_LEVEL_CONFIGS):
            raise ValueError(f"z_level must be in 1..{len(Z_LEVEL_CONFIGS)}, got {self.z_level}")
        if self.real_dataset_z_level is not None:
            if self.real_dataset_z_level not in range(1, SYNTHETIC_SOURCE_Z_LEVEL + 1):
                raise ValueError(
                    f"real_dataset_z_level must be an existing ETOPO level in 1..{SYNTHETIC_SOURCE_Z_LEVEL}"
                )
            if not self.super_scaler and self.real_dataset_z_level != self.z_level:
                raise ValueError("real_dataset_z_level override requires super_scaler")
            if self.super_scaler and self.real_dataset_z_level != SYNTHETIC_SOURCE_Z_LEVEL:
                raise ValueError(f"super scaler real_dataset_z_level must be {SYNTHETIC_SOURCE_Z_LEVEL}")
        if self.patch_size % 2:
            raise ValueError("real dataset patch size must be even")
        if (
                self.input_maps_per_batch <= 0
                or self.input_map_total_samples <= 0
                or self.input_map_generation_batch_size <= 0
                or self.input_map_size <= 0
                or self.large_map_preview_target_size <= 16
                or self.z0_minimum_noise_size <= 0
        ):
            raise ValueError(
                "input map batch sizes, total samples, and map size must be positive"
            )
        if self.input_map_total_samples % self.input_maps_per_batch:
            raise ValueError(
                "input_map_total_samples must be divisible by input_maps_per_batch"
            )
        if not 0 < self.patches_per_batch <= self.input_maps_per_batch:
            raise ValueError("patches_per_batch must be positive and no greater than input_maps_per_batch")
        object.__setattr__(self, "run_dir", default_run_dir(self.level_name))

    @property
    def profile_total_steps(self) -> int:
        return self.profile_wait_steps + self.profile_warmup_steps + self.profile_active_steps

    @property
    def level_config(self) -> ZnUpscalerLevelConfig:
        return Z_LEVEL_CONFIGS[self.z_level - 1]

    @property
    def patch_size(self) -> int:
        return PATCH_SIZES[self.real_z_level]

    @property
    def real_z_level(self) -> int:
        if self.real_dataset_z_level is not None:
            return self.real_dataset_z_level
        return self.level_config.real_z_level

    @property
    def super_scaler(self) -> bool:
        return self.level_config.super_scaler

    @property
    def stats_z_level(self) -> int:
        return self.z_level

    @property
    def level_name(self) -> str:
        return self.level_config.level_name

    @property
    def predecessor_level_name(self) -> str:
        return Z_LEVEL_CONFIGS[self.z_level - 2].level_name if self.z_level > 1 else "z0"

    @property
    def input_map_source_dir(self) -> Path:
        return self.maps_dir / self.predecessor_level_name

    @property
    def output_map_dir(self) -> Path:
        return self.maps_dir / self.level_name

    @property
    def input_metric_patch_size(self) -> int:
        return self.patch_size // 2 + 1

    @property
    def residual_limits(self) -> tuple[float, float]:
        stats = UPSCALE_STATS[self.stats_z_level]
        return stats.dataset_minimum_raw_residual, stats.dataset_maximum_raw_residual

    @property
    def model_layer_channels(self) -> tuple[int, ...]:
        return self.level_config.model_layer_channels

    @property
    def critic_layer_specs(self) -> str:
        if self.super_scaler:
            return Z_LEVEL_CONFIGS[self.real_z_level - 1].critic_layer_specs
        return self.level_config.critic_layer_specs

    @property
    def model_run_dir(self) -> Path:
        return self.model_dir / self.level_name / self.run_dir.name

    @property
    def model_path(self) -> Path:
        return self.model_run_dir / MODEL_FILENAME

    @property
    def pt2_path(self) -> Path:
        return self.model_run_dir / PT2_FILENAME

    @property
    def onnx_path(self) -> Path:
        return self.model_run_dir / ONNX_FILENAME

    @property
    def patch_mean_pdf_path(self) -> Path:
        return self.analysis_dir / f"z{self.real_z_level}_patch_mean_pdf.pt"

    @property
    def input_map_patch_mean_pdf_path(self) -> Path:
        return self.analysis_dir / f"{self.level_name}_input_map_patch_mean_pdf.pt"

    @property
    def predecessor_model_paths(self) -> tuple[Path, ...]:
        return tuple(
            self.model_dir / level.level_name / "latest" / MODEL_FILENAME
            for level in Z_LEVEL_CONFIGS[:self.z_level - 1]
        )

    @property
    def z0_pt2_path(self) -> Path:
        return self.model_dir.parent / "z0_generator" / "latest" / "generator.pt2"
