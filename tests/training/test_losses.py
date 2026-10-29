import torch

from terrain_higan.training.losses import residual_aggregate_loss


def test_residual_aggregate_loss_matches_batch_wide_mean_and_std() -> None:
    generated = torch.tensor([[[[0.0, 1.0]]], [[[2.0, 3.0]]]], requires_grad=True)
    real = torch.tensor([[[[1.0, 3.0]]], [[[5.0, 7.0]]]])

    actual = residual_aggregate_loss(generated, real)
    expected = (generated.mean() - real.mean()).square() + (
        generated.std(correction=0) - real.std(correction=0)
    ).square()

    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert generated.grad is not None
