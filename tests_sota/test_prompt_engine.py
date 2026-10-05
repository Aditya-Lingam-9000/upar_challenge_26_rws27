"""Unit tests for DynamicPromptEngine and Universal Tokenization."""
import numpy as np
import torch
from sota_pipeline.prompt_engine import (
    CANONICAL_ATTRIBUTES,
    DynamicPromptEngine,
)


def test_prompt_synthesis():
    engine = DynamicPromptEngine(p_dropout=0.0) # Deterministic for testing
    attr_vec = np.zeros(40, dtype=int)
    attr_vec[0] = 1 # Age-Young
    attr_vec[3] = 1 # Gender-Female
    attr_vec[8] = 1 # UpperBody-Color-Black
    attr_vec[35] = 1 # Accessory-Backpack

    prompt = engine.synthesize_prompt(attr_vec, is_train=False)
    assert isinstance(prompt, str)
    assert len(prompt) > 10
    print("[PASS] Full prompt synthesis:", prompt)

    # Test Anatomical Strip Prompts
    prompt_head = engine.synthesize_prompt(attr_vec, is_train=False, anatomical_strip="head")
    prompt_torso = engine.synthesize_prompt(attr_vec, is_train=False, anatomical_strip="torso")
    prompt_legs = engine.synthesize_prompt(attr_vec, is_train=False, anatomical_strip="legs")

    assert "head" in prompt_head.lower()
    assert "torso" in prompt_torso.lower()
    assert "legs" in prompt_legs.lower()
    print("[PASS] Strip prompt head:", prompt_head)
    print("[PASS] Strip prompt torso:", prompt_torso)
    print("[PASS] Strip prompt legs:", prompt_legs)


def test_tokenization_shapes():
    engine = DynamicPromptEngine()
    prompts = [
        "A surveillance photo of a young female in black shirt.",
        "A photo of a pedestrian carrying a backpack.",
    ]
    tokens = engine.tokenize(prompts)
    assert tokens.ndim == 2
    assert tokens.shape[0] == 2
    assert tokens.shape[1] == 77
    assert tokens.dtype == torch.long
    print("[PASS] Tokenizer output shape:", tokens.shape)


if __name__ == "__main__":
    test_prompt_synthesis()
    test_tokenization_shapes()
    print("All prompt engine tests passed!")
