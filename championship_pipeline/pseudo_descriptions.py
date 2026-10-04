"""Championship Cross-Modal Pseudo-Description & Tokenization Engine.

Transforms discrete 40-dimensional UPAR binary attribute vectors into grammatically
structured natural language descriptions and token sequences using a pure PyTorch
self-contained vocabulary with ZERO external NLP dependencies.
"""
from __future__ import annotations

import re
from typing import Any
import numpy as np
import torch

CANONICAL_ATTRIBUTES = [
    "Age-Young", "Age-Adult", "Age-Old", "Gender-Female",
    "Hair-Length-Short", "Hair-Length-Long", "Hair-Length-Bald",
    "UpperBody-Length-Short", "UpperBody-Color-Black", "UpperBody-Color-Blue",
    "UpperBody-Color-Brown", "UpperBody-Color-Green", "UpperBody-Color-Grey",
    "UpperBody-Color-Orange", "UpperBody-Color-Pink", "UpperBody-Color-Purple",
    "UpperBody-Color-Red", "UpperBody-Color-White", "UpperBody-Color-Yellow",
    "UpperBody-Color-Other", "LowerBody-Length-Short", "LowerBody-Color-Black",
    "LowerBody-Color-Blue", "LowerBody-Color-Brown", "LowerBody-Color-Green",
    "LowerBody-Color-Grey", "LowerBody-Color-Orange", "LowerBody-Color-Pink",
    "LowerBody-Color-Purple", "LowerBody-Color-Red", "LowerBody-Color-White",
    "LowerBody-Color-Yellow", "LowerBody-Color-Other",
    "LowerBody-Type-Trousers&Shorts", "LowerBody-Type-Skirt&Dress",
    "Accessory-Backpack", "Accessory-Bag", "Accessory-Glasses-Normal",
    "Accessory-Glasses-Sun", "Accessory-Hat",
]

COLOR_NAMES = [
    "black", "blue", "brown", "green", "grey",
    "orange", "pink", "purple", "red", "white", "yellow", "other colored"
]

# Fixed domain vocabulary for all possible pedestrian pseudo-descriptions
# Contains exactly 85 words covering all 40 attributes and grammatical connectors
VOCABULARY = [
    "<PAD>", "<UNK>", "<SOS>", "<EOS>",
    "this", "is", "a", "photo", "of", "young", "adult", "elderly", "person",
    "female", "male", "with", "short", "long", "bald", "head", "hair",
    "dressed", "in", "sleeved", "upper", "garment", "and", "lower", "clothing",
    "trousers", "pants", "skirt", "dress", "carrying", "wearing", "no", "prominent",
    "backpack", "bag", "glasses", "sunglasses", "hat", "cap", "full", "length",
    "black", "blue", "brown", "green", "grey", "orange", "pink", "purple",
    "red", "white", "yellow", "other", "colored", "neutral", "dark", "bright"
]

WORD_TO_IDX = {word: idx for idx, word in enumerate(VOCABULARY)}
IDX_TO_WORD = {idx: word for idx, word in enumerate(VOCABULARY)}
VOCAB_SIZE = len(VOCABULARY)
MAX_TOKEN_LENGTH = 40


class PseudoDescriptionEngine:
    """Deterministic converter from 40-dim binary attributes to language and tokens."""

    def __init__(self, max_length: int = MAX_TOKEN_LENGTH):
        self.max_length = max_length
        self.vocab_size = VOCAB_SIZE

    def attributes_to_text(self, q: list[int] | np.ndarray) -> str:
        """Constructs a natural descriptive sentence from 40 binary attributes."""
        q = np.asarray(q, dtype=np.int32)
        if len(q) != 40:
            raise ValueError(f"Expected 40 attributes, got {len(q)}")

        # 1. Age
        if q[0] == 1:
            age = "young"
        elif q[2] == 1:
            age = "elderly"
        else:
            age = "adult"

        # 2. Gender (0 = Male, 1 = Female in UPAR canon)
        gender = "female" if q[3] == 1 else "male"

        # 3. Hair
        if q[6] == 1:
            hair = "bald head"
        elif q[5] == 1:
            hair = "long hair"
        else:
            hair = "short hair"

        # 4. Upper body length and colors
        upper_len = "short sleeved" if q[7] == 1 else "long sleeved"
        upper_colors = [COLOR_NAMES[i] for i in range(12) if q[8 + i] == 1]
        upper_color_str = " ".join(upper_colors) if upper_colors else "neutral colored"

        # 5. Lower body length, type, and colors
        lower_type = "skirt dress" if q[34] == 1 else "trousers pants"
        lower_len = "short" if q[20] == 1 else "full length"
        lower_colors = [COLOR_NAMES[i] for i in range(12) if q[21 + i] == 1]
        lower_color_str = " ".join(lower_colors) if lower_colors else "dark colored"

        # 6. Accessories
        acc_tokens = []
        if q[35] == 1:
            acc_tokens.append("backpack")
        if q[36] == 1:
            acc_tokens.append("bag")
        if q[37] == 1:
            acc_tokens.append("glasses")
        if q[38] == 1:
            acc_tokens.append("sunglasses")
        if q[39] == 1:
            acc_tokens.append("hat")

        if acc_tokens:
            acc_str = "carrying wearing " + " ".join(acc_tokens)
        else:
            acc_str = "with no prominent accessories"

        sentence = (
            f"this is a photo of {age} {gender} with {hair} dressed in "
            f"{upper_color_str} {upper_len} upper garment and "
            f"{lower_color_str} {lower_len} {lower_type} {acc_str}"
        )
        return sentence

    def tokenize_text(self, text: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts sentence text to token indices and attention mask."""
        clean_words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
        tokens = [WORD_TO_IDX.get("<SOS>")]
        for w in clean_words:
            tokens.append(WORD_TO_IDX.get(w, WORD_TO_IDX["<UNK>"]))
            if len(tokens) >= self.max_length - 1:
                break
        tokens.append(WORD_TO_IDX.get("<EOS>"))

        seq_len = len(tokens)
        pad_len = self.max_length - seq_len
        tokens = tokens + [WORD_TO_IDX["<PAD>"]] * pad_len
        # mask is True for valid tokens, False for padding
        mask = [True] * seq_len + [False] * pad_len

        return torch.tensor(tokens, dtype=torch.long), torch.tensor(mask, dtype=torch.bool)

    def process_query(self, q: list[int] | np.ndarray) -> dict[str, Any]:
        """Full processing from binary query to hard vector, text, and tokens."""
        q_arr = np.asarray(q, dtype=np.float32)
        text = self.attributes_to_text(q_arr)
        tokens, mask = self.tokenize_text(text)
        return {
            "hard_vector": torch.tensor(q_arr, dtype=torch.float32),
            "text": text,
            "tokens": tokens,
            "mask": mask,
        }
