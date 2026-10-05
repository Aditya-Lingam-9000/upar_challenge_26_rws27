"""End-to-End Comprehensive Pipeline Integration Test.

Verifies:
1. SOTAChampionshipModel forward + multi-task loss + backward + optimizer step.
2. Retrieval evaluation with mADM calculation and RRF fusion.
3. Pruning of training-only MLM module and INT8 quantization.
4. Codabench submission zip building and air-gapped verification.
"""
import os
import shutil
import tempfile
import numpy as np
import torch

from sota_pipeline.distill_and_quantize import (
    apply_dynamic_quantization,
    prune_and_strip_training_modules,
)
from sota_pipeline.evaluate import compute_cmc_map, compute_madm
from sota_pipeline.losses import TotalChampionshipLoss
from sota_pipeline.models import SOTAChampionshipModel
from sota_pipeline.package_submission import (
    build_submission_package,
    verify_submission_package,
)
from sota_pipeline.prompt_engine import DynamicPromptEngine
from sota_pipeline.reranking import InformationWeightedL1, ReciprocalRankFusion


def test_complete_championship_lifecycle():
    print("=" * 80)
    print("[INTEGRATION TEST] Starting End-to-End SOTA Championship Lifecycle Test...")
    print("=" * 80)

    temp_dir = tempfile.mkdtemp()
    try:
        B, D, K = 4, 256, 40
        H, W = 112, 56

        # 1. Architecture & Loss Setup
        print("\n--- 1. Architecture & Multi-Task Loss Setup ---")
        model = SOTAChampionshipModel(
            embed_dim=D,
            num_attrs=K,
            img_height=H,
            img_width=W,
            patch_size=14,
            backbone_type="standalone",
            use_mlm=True
        )
        criterion = TotalChampionshipLoss(lambda_sdm=1.0, lambda_mlm=0.5, lambda_asl=1.0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        # 2. Simulated Training Step
        print("\n--- 2. Forward, Loss, Backward, Optimizer Step ---")
        images = torch.randn(B, 3, H, W)
        text_tokens = torch.randint(0, 100, (B, 77))
        masked_tokens = text_tokens.clone()
        masked_tokens[:, 10] = 4 # <MASK>
        masked_labels = torch.full((B, 77), fill_value=-100)
        masked_labels[:, 10] = text_tokens[:, 10]
        attr_labels = torch.randint(0, 2, (B, K)).float()

        optimizer.zero_grad()
        outputs = model(images, text_tokens, masked_tokens=masked_tokens)
        loss, metrics = criterion(
            img_metric=outputs["img_metric"],
            txt_metric=outputs["txt_metric"],
            attr_logits=outputs["attr_logits"],
            attr_labels=attr_labels,
            vocab_logits=outputs.get("vocab_logits"),
            masked_labels=masked_labels
        )
        loss.backward()
        optimizer.step()
        print(f"[PASS] Training step complete! Total Loss: {loss.item():.4f}, Metrics: {metrics}")

        # 3. Retrieval Evaluation & Metric Calculation
        print("\n--- 3. Retrieval Evaluation (mADM & RRF) ---")
        model.eval()
        with torch.no_grad():
            img_emb, attr_logits = model.extract_visual(images)
            txt_emb = model.extract_text(text_tokens)

        # Continuous metric distance
        sim_m = torch.matmul(txt_emb, img_emb.t())
        dist_m = (1.0 - sim_m).numpy()

        # Attribute distance
        info_l1 = InformationWeightedL1()
        dist_a = info_l1.compute_distance(attr_labels.numpy(), torch.sigmoid(attr_logits).numpy()).numpy()

        # RRF Fusion
        rrf = ReciprocalRankFusion(k0=60, alpha=0.65)
        ranked = rrf.fuse_ranks(dist_m, dist_a)

        madm = compute_madm(ranked, attr_labels.numpy(), attr_labels.numpy())
        r1, r5, r10, map_s = compute_cmc_map(ranked, attr_labels.numpy(), attr_labels.numpy())
        print(f"[PASS] Evaluated Retrieval: mADM = {madm:.4f} | Rank-1 = {r1:.2f}% | mAP = {map_s:.2f}%")

        # 4. Checkpoint Serialization, Pruning & Quantization
        print("\n--- 4. Checkpoint Serialization & INT8 Quantization ---")
        raw_pt = os.path.join(temp_dir, "raw_checkpoint.pt")
        pruned_pt = os.path.join(temp_dir, "pruned_model.pt")
        quant_pt = os.path.join(temp_dir, "quantized_model.pt")

        torch.save({"model_state_dict": model.state_dict(), "best_metric": madm, "epoch": 1}, raw_pt)
        prune_and_strip_training_modules(raw_pt, pruned_pt)

        clean_model = SOTAChampionshipModel(embed_dim=D, num_attrs=K, img_height=H, img_width=W, patch_size=14, backbone_type="standalone", use_mlm=False)
        pruned_state = torch.load(pruned_pt, weights_only=False)["model_state_dict"]
        clean_model.load_state_dict(pruned_state, strict=False)

        quant_model = apply_dynamic_quantization(clean_model)
        torch.save(quant_model.state_dict(), quant_pt)
        print(f"[PASS] Pruned & INT8 Quantized model saved to: {quant_pt}")

        # 5. Codabench Submission Packaging & Air-Gapped Verification
        print("\n--- 5. Submission Packaging & Air-Gapped Simulation ---")
        submission_zip = os.path.join(temp_dir, "test_submission.zip")
        build_submission_package(weights_path=quant_pt, output_zip=submission_zip)
        verify_submission_package(submission_zip)
        print(f"[PASS] Codabench submission zip verified!")

        print("\n" + "=" * 80)
        print("[SUCCESS] ALL END-TO-END INTEGRATION TESTS PASSED!")
        print("=" * 80 + "\n")
    finally:
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    test_complete_championship_lifecycle()
