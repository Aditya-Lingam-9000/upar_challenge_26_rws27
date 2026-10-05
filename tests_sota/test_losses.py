"""Unit tests for SOTA Losses (Soft SDM, ASL, MLM, Strip SDM)."""
import torch
from sota_pipeline.losses import (
    AsymmetricLoss,
    CrossModalMLMLoss,
    MultiInstanceSDMLoss,
    StripSDMLoss,
    TotalChampionshipLoss,
)


def test_soft_sdm_loss():
    criterion = MultiInstanceSDMLoss(epsilon_hamming=1)
    B, D = 4, 256
    v = torch.randn(B, D, requires_grad=True)
    t = torch.randn(B, D, requires_grad=True)
    # Samples 0 and 1 share identical attributes
    labels = torch.zeros(B, 40)
    labels[0, :5] = 1.0
    labels[1, :5] = 1.0  # Identical to sample 0
    labels[2, 10:15] = 1.0
    labels[3, 20:25] = 1.0

    loss = criterion(v, t, labels)
    assert not torch.isnan(loss)
    assert loss.item() > 0.0
    loss.backward()
    assert v.grad is not None
    assert t.grad is not None
    print(f"[PASS] MultiInstanceSDMLoss computed: {loss.item():.4f}")


def test_asymmetric_loss():
    asl = AsymmetricLoss(gamma_neg=4.0, gamma_pos=0.0, clip=0.05)
    B, K = 4, 40
    logits = torch.randn(B, K, requires_grad=True)
    # Highly imbalanced targets (mostly 0s)
    targets = torch.zeros(B, K)
    targets[0, 1] = 1.0
    targets[2, 35] = 1.0

    loss = asl(logits, targets)
    assert not torch.isnan(loss)
    assert loss.item() > 0.0
    loss.backward()
    assert logits.grad is not None
    print(f"[PASS] AsymmetricLoss computed: {loss.item():.4f}")


def test_total_championship_loss():
    total_criterion = TotalChampionshipLoss()
    B, D, K = 4, 256, 40
    img_metric = torch.randn(B, D, requires_grad=True)
    txt_metric = torch.randn(B, D, requires_grad=True)
    attr_logits = torch.randn(B, K, requires_grad=True)
    attr_labels = torch.randint(0, 2, (B, K)).float()

    loss, metrics = total_criterion(
        img_metric=img_metric,
        txt_metric=txt_metric,
        attr_logits=attr_logits,
        attr_labels=attr_labels,
    )
    assert "loss_sdm" in metrics
    assert "loss_asl" in metrics
    assert "loss_total" in metrics
    loss.backward()
    print(f"[PASS] TotalChampionshipLoss: {loss.item():.4f}, Metrics: {metrics}")


if __name__ == "__main__":
    test_soft_sdm_loss()
    test_asymmetric_loss()
    test_total_championship_loss()
    print("All loss tests passed!")
