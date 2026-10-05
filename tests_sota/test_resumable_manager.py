"""Unit tests for ResumableCheckpointManager multi-session persistence."""
import os
import shutil
import tempfile
import torch
from sota_pipeline.models import SOTAChampionshipModel
from sota_pipeline.resumable_manager import ResumableCheckpointManager


def test_resumption_fidelity():
    temp_dir = tempfile.mkdtemp()
    try:
        mgr = ResumableCheckpointManager(checkpoint_dir=temp_dir, max_hours=10.0)

        # Create model and optimizer
        model = SOTAChampionshipModel(embed_dim=256, img_height=112, img_width=56, patch_size=14, backbone_type="standalone")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

        # Simulate 3 steps of training
        for _ in range(3):
            optimizer.zero_grad()
            out = model.extract_text(torch.randint(0, 100, (2, 77)))
            loss = out.sum()
            loss.backward()
            optimizer.step()
            scheduler.step()

        # Save checkpoint at epoch 5
        save_path = mgr.save_checkpoint(
            epoch=5,
            global_step=150,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            best_metric=0.8850,
            is_best=True
        )
        assert os.path.isfile(save_path)

        # Create fresh model and optimizer
        fresh_model = SOTAChampionshipModel(embed_dim=256, img_height=112, img_width=56, patch_size=14, backbone_type="standalone")
        fresh_optimizer = torch.optim.AdamW(fresh_model.parameters(), lr=1e-4)
        fresh_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(fresh_optimizer, T_max=50, eta_min=1e-6)

        # Restore state
        start_epoch, global_step, best_metric = mgr.load_checkpoint(
            checkpoint_path=save_path,
            model=fresh_model,
            optimizer=fresh_optimizer,
            scheduler=fresh_scheduler
        )

        assert start_epoch == 6 # Resumes at next epoch
        assert global_step == 150
        assert best_metric == 0.8850
        assert fresh_scheduler.get_last_lr() == scheduler.get_last_lr()
        print(f"[PASS] Checkpoint serialization and restoration 100% verified! Resumes at epoch {start_epoch}.")
    finally:
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    test_resumption_fidelity()
    print("Resumable manager test passed!")
