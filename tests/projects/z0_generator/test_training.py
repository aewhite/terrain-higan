import unittest
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import torch
import pytest
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter

from terrain_higan.projects.z0_generator import stages as z0_stages
from terrain_higan.projects.z0_generator.models import Z0Discriminator
from terrain_higan.projects.z0_generator.stages import GANTrainStage, TensorBoardLogStage, Z0GeneratorStepData
from terrain_higan.training import TrainingContext
from terrain_higan.visualization.grids import make_logging_image_grid


class Z0GeneratorTrainingTest(unittest.TestCase):
    def test_cpu(self) -> None:
        _check_training("cpu", constant_field=False)

    def test_cpu_constant_field(self) -> None:
        _check_training("cpu", constant_field=True)

    @pytest.mark.cuda
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda(self) -> None:
        _check_training("cuda", constant_field=False)

    @pytest.mark.cuda
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_constant_field(self) -> None:
        _check_training("cuda", constant_field=True)

    def test_total_loss_histograms_collect_and_reset_each_interval(self) -> None:
        writer = Mock()
        stage = TensorBoardLogStage(writer=writer, patch_size=3, log_every_steps=2, image_every_steps=100)
        for step in range(1, 5):
            stage.run(TrainingContext(
                step_number=step,
                step_data=Z0GeneratorStepData(
                    generator_loss=torch.tensor(float(step), requires_grad=True),
                    discriminator_loss=torch.tensor(float(-step), requires_grad=True),
                ),
            ))
            if step == 1:
                writer.add_histogram.assert_not_called()

        calls = writer.add_histogram.call_args_list
        assert [call.args[0] for call in calls] == [
            "loss/generator_histogram", "loss/critic_histogram",
            "loss/generator_histogram", "loss/critic_histogram",
        ]
        assert [call.args[2] for call in calls] == [2, 2, 4, 4]
        for call, expected in zip(calls, ([1.0, 2.0], [-1.0, -2.0], [3.0, 4.0], [-3.0, -4.0])):
            torch.testing.assert_close(call.args[1], torch.tensor(expected))
            assert not call.args[1].requires_grad

    def test_event_file_keeps_scalar_and_histogram_tags_distinct(self) -> None:
        data = Z0GeneratorStepData(
            generator_loss=torch.tensor(1.0),
            discriminator_loss=torch.tensor(2.0),
            real_scores=torch.tensor([1.0, 2.0]),
            fake_scores=torch.tensor([-1.0, -2.0]),
            real_patches=torch.tensor([0.1, 0.2]),
            generated_norm_field=torch.tensor([0.3, 0.4]),
        )
        with TemporaryDirectory() as run_dir:
            with SummaryWriter(run_dir) as writer:
                TensorBoardLogStage(writer=writer, patch_size=3, log_every_steps=1, image_every_steps=100).run(
                    TrainingContext(step_number=1, step_data=data)
                )
            tags = EventAccumulator(run_dir).Reload().Tags()
        assert set(tags["histograms"]) == {
            "elevations/real", "elevations/generated", "scores/critic_real", "scores/critic_fake",
            "loss/generator_histogram", "loss/critic_histogram",
        }
        assert set(tags["scalars"]) == {
            "loss/generator", "loss/critic", "scores/critic_real_mean", "scores/critic_fake_mean",
        }
        assert set(tags["scalars"]).isdisjoint(tags["histograms"])

    def test_previews_preserve_patch_order_and_independent_logging_intervals(self) -> None:
        writer = Mock()
        stage = TensorBoardLogStage(writer=writer, patch_size=2, log_every_steps=2, image_every_steps=3)
        field = torch.linspace(-0.5, 0.5, 35).reshape(1, 1, 5, 7)
        real_patches = torch.zeros(2, 1, 2, 2)
        expected_patches = torch.stack([
            field[0, :, row:row + 2, column:column + 2]
            for row in range(0, 4, 2)
            for column in range(0, 6, 2)
        ])
        expected_images = (
            ("images/real_patches", make_logging_image_grid(real_patches, 64, 24)),
            ("images/generated_patches", make_logging_image_grid(expected_patches, 64, 24)),
            ("images/output_maps", make_logging_image_grid(field, 1, 4)),
        )
        rng_state = torch.get_rng_state()
        with patch(
                "terrain_higan.projects.z0_generator.stages._extract_patches",
                wraps=z0_stages._extract_patches,
        ) as extract:
            for step in range(1, 7):
                stage.run(TrainingContext(
                    step_number=step,
                    step_data=Z0GeneratorStepData(
                        real_patches=real_patches,
                        generated_norm_field=field,
                        generator_loss=torch.tensor(float(step)),
                    ),
                ))
                assert extract.call_count == step // 3
        assert writer.flush.call_count == 3
        assert [call.args[2] for call in writer.add_scalar.call_args_list] == [2, 4, 6]
        image_calls = writer.add_image.call_args_list
        assert len(image_calls) == 6
        for index, call in enumerate(image_calls):
            tag, expected = expected_images[index % 3]
            assert call.args[0] == tag
            assert call.args[2] == (3 if index < 3 else 6)
            torch.testing.assert_close(call.args[1], expected, rtol=0, atol=0)
        assert torch.equal(torch.get_rng_state(), rng_state)

    def test_image_logging_accepts_partial_step_data(self) -> None:
        for data, expected_tags in (
                (Z0GeneratorStepData(), []),
                (Z0GeneratorStepData(real_patches=torch.zeros(1, 1, 3, 3)), ["images/real_patches"]),
                (
                        Z0GeneratorStepData(generated_norm_field=torch.zeros(1, 1, 7, 8)),
                        ["images/generated_patches", "images/output_maps"],
                ),
        ):
            with self.subTest(expected_tags=expected_tags):
                writer = Mock()
                TensorBoardLogStage(
                    writer=writer, patch_size=3, log_every_steps=2, image_every_steps=1,
                ).run(TrainingContext(step_number=1, step_data=data))
                assert [call.args[0] for call in writer.add_image.call_args_list] == expected_tags
                writer.add_scalar.assert_not_called()
                writer.add_histogram.assert_not_called()
                writer.flush.assert_not_called()


def _check_training(device_name: str, constant_field: bool) -> None:
    device = torch.device(device_name)
    torch.manual_seed(7)
    generator = torch.nn.Conv2d(1, 1, kernel_size=1).to(device)
    if constant_field:
        with torch.no_grad():
            generator.weight.zero_()
            generator.bias.fill_(0.2)
    discriminator = Z0Discriminator(patch_size=3, output_clip=10.0).to(device)
    initial_generator = [parameter.detach().clone() for parameter in generator.parameters()]
    initial_discriminator = [parameter.detach().clone() for parameter in discriminator.parameters()]
    stage = GANTrainStage(
        generator=generator,
        discriminator=discriminator,
        generator_optimizer=torch.optim.SGD(generator.parameters(), lr=0.01),
        discriminator_optimizer=torch.optim.SGD(discriminator.parameters(), lr=0.01),
        noise_size=5,
        patch_size=3,
        generator_adversarial_loss_weight=1.0,
        generator_elevation_loss_weight=0.1,
        real_patch_shift_stddev=0.1,
        critic_output_mean_loss_weight=1.0,
        device=device,
    )
    real_patches = torch.linspace(-0.5, 0.5, 18, device=device).reshape(2, 1, 3, 3)
    context = TrainingContext(step_number=1, step_data=Z0GeneratorStepData(real_patches=real_patches))
    stage.run(context)
    data = context.step_data

    assert torch.count_nonzero(data.real_patch_shifts).item() > 0
    torch.testing.assert_close(
        data.generator_loss,
        data.generator_adversarial_loss + 0.1 * data.generator_elevation_loss,
    )
    assert data.generated_norm_patches.shape == real_patches.shape
    assert data.real_scores.shape == data.fake_scores.shape == (2,)
    torch.testing.assert_close(
        data.discriminator_loss,
        data.discriminator_adversarial_loss + data.discriminator_output_mean_loss,
    )
    for model, initial_parameters in (
            (generator, initial_generator),
            (discriminator, initial_discriminator),
    ):
        assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
        assert any(
            not torch.equal(initial, parameter)
            for initial, parameter in zip(initial_parameters, model.parameters())
        )

    writer = Mock()
    TensorBoardLogStage(writer=writer, patch_size=3, log_every_steps=1, image_every_steps=1).run(context)
    logged_scalars = {call.args[0]: call.args[1] for call in writer.add_scalar.call_args_list}
    assert set(logged_scalars) == {
        "loss/generator", "loss/generator_adversarial", "loss/generator_elevation",
        "loss/critic", "loss/critic_adversarial", "loss/critic_output_mean",
        "scores/critic_real_mean", "scores/critic_fake_mean",
    }
    assert logged_scalars["scores/critic_real_mean"] == data.real_scores.mean().item()
    assert logged_scalars["scores/critic_fake_mean"] == data.fake_scores.mean().item()
    assert logged_scalars["loss/critic"] == data.discriminator_loss.item()
    histograms = {call.args[0]: call.args[1] for call in writer.add_histogram.call_args_list}
    assert set(histograms) == {
        "elevations/real", "elevations/generated", "scores/critic_real", "scores/critic_fake",
        "loss/generator_histogram", "loss/critic_histogram",
    }
    for tag, expected in (
            ("elevations/real", data.real_patches),
            ("elevations/generated", data.generated_norm_field),
            ("scores/critic_real", data.real_scores),
            ("scores/critic_fake", data.fake_scores),
            ("loss/generator_histogram", data.generator_loss.reshape(1)),
            ("loss/critic_histogram", data.discriminator_loss.reshape(1)),
    ):
        torch.testing.assert_close(histograms[tag], expected)
    image_tags = {call.args[0] for call in writer.add_image.call_args_list}
    assert image_tags == {"images/real_patches", "images/generated_patches", "images/output_maps"}
    logged_tags = list(logged_scalars) + list(histograms) + list(image_tags)
    assert not any(
        "3x3" in tag or "2x2" in tag or "1x1" in tag or "binary" in tag or "discriminator" in tag
        for tag in logged_tags
    )
