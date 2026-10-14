"""Dynamic Lexical Prompt Synthesis & Universal Tokenization Engine.

Converts 40-dimensional binary attribute vectors into fluent, natural language
pedestrian descriptions using synonym jitter, attribute dropout, anatomical
part decomposition, and universal tokenization.
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch

CANONICAL_ATTRIBUTES: List[str] = [
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
    "Accessory-Glasses-Sun", "Accessory-Hat"
]

# Anatomical Partition Index Sets
HEAD_ATTR_INDICES = [0, 1, 2, 3, 4, 5, 6, 37, 38, 39]
TORSO_ATTR_INDICES = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 35, 36]
LEGS_ATTR_INDICES = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]

# Rich Natural Language Synonym Lexicon
SYNONYM_LEXICON: Dict[str, List[str]] = {
    "Age-Young": ["young", "youthful", "adolescent"],
    "Age-Adult": ["adult", "middle-aged"],
    "Age-Old": ["elderly", "senior", "older"],
    "Gender-Female": ["female", "woman", "girl"],
    "Hair-Length-Short": ["with short hair", "having short-cropped hair", "with short hair style"],
    "Hair-Length-Long": ["with long hair", "having flowing hair", "sporting long hair"],
    "Hair-Length-Bald": ["bald", "with a shaved head", "without hair on head"],
    
    "UpperBody-Length-Short": ["wearing short sleeves", "in a short-sleeved top", "with short-sleeved shirt"],
    "UpperBody-Color-Black": ["wearing a black top", "in a black upper garment", "with black shirt"],
    "UpperBody-Color-Blue": ["wearing a blue shirt", "in blue top clothing", "with blue jacket"],
    "UpperBody-Color-Brown": ["wearing a brown shirt", "in brown upper clothing", "with brown top"],
    "UpperBody-Color-Green": ["wearing a green shirt", "in green upper garment", "with green top"],
    "UpperBody-Color-Grey": ["wearing a grey top", "in gray upper garment", "with grey shirt"],
    "UpperBody-Color-Orange": ["wearing an orange shirt", "in bright orange top", "with orange garment"],
    "UpperBody-Color-Pink": ["wearing a pink top", "in pink upper clothing", "with pink shirt"],
    "UpperBody-Color-Purple": ["wearing a purple top", "in violet shirt", "with purple garment"],
    "UpperBody-Color-Red": ["wearing a red shirt", "in crimson top", "with red upper clothing"],
    "UpperBody-Color-White": ["wearing a white shirt", "in clean white top", "with white garment"],
    "UpperBody-Color-Yellow": ["wearing a yellow top", "in yellow shirt", "with bright yellow clothing"],
    "UpperBody-Color-Other": ["wearing a multi-colored top", "with distinct patterned shirt"],
    
    "LowerBody-Length-Short": ["wearing short lower clothing", "with shorts or knee-length wear"],
    "LowerBody-Color-Black": ["wearing black trousers", "in black pants", "with black bottoms"],
    "LowerBody-Color-Blue": ["wearing blue pants", "in blue jeans", "with blue trousers"],
    "LowerBody-Color-Brown": ["wearing brown pants", "in brown trousers", "with brown bottoms"],
    "LowerBody-Color-Green": ["wearing green pants", "in green trousers", "with green bottoms"],
    "LowerBody-Color-Grey": ["wearing grey pants", "in grey trousers", "with gray bottoms"],
    "LowerBody-Color-Orange": ["wearing orange pants", "in orange bottoms"],
    "LowerBody-Color-Pink": ["wearing pink pants", "in pink bottoms"],
    "LowerBody-Color-Purple": ["wearing purple bottoms", "in purple trousers"],
    "LowerBody-Color-Red": ["wearing red pants", "in red trousers"],
    "LowerBody-Color-White": ["wearing white pants", "in white trousers", "with white bottoms"],
    "LowerBody-Color-Yellow": ["wearing yellow pants", "in yellow bottoms"],
    "LowerBody-Color-Other": ["wearing other colored pants", "with patterned bottoms"],
    
    "LowerBody-Type-Trousers&Shorts": ["dressed in trousers or shorts", "wearing pants"],
    "LowerBody-Type-Skirt&Dress": ["wearing a skirt or dress", "in a feminine dress", "sporting a skirt"],
    
    "Accessory-Backpack": ["carrying a backpack", "with a knapsack on back", "wearing a back pack"],
    "Accessory-Bag": ["carrying a bag or purse", "holding a handbag", "with a shoulder bag"],
    "Accessory-Glasses-Normal": ["wearing eyeglasses", "with prescription spectacles", "with clear glasses"],
    "Accessory-Glasses-Sun": ["wearing dark sunglasses", "with shades", "wearing sun glasses"],
    "Accessory-Hat": ["wearing a hat or cap", "with headwear", "sporting a baseball cap"],
}

# Universal Built-in Vocabulary for Zero-Dependency Fallback Tokenizer
UNIVERSAL_VOCAB: List[str] = [
    "<PAD>", "<UNK>", "<SOS>", "<EOS>", "<MASK>",
    "a", "photo", "of", "surveillance", "image", "capture", "pedestrian", "person",
    "young", "youthful", "adolescent", "adult", "middle-aged", "elderly", "senior", "older",
    "female", "woman", "girl", "male", "man", "boy",
    "with", "having", "sporting", "wearing", "carrying", "holding", "in", "dressed",
    "short", "long", "bald", "hair", "head", "cropped", "flowing", "style", "shaved",
    "sleeves", "sleeved", "top", "shirt", "garment", "jacket", "upper", "clothing", "coat",
    "trousers", "pants", "jeans", "bottoms", "shorts", "skirt", "dress", "lower", "knee-length",
    "backpack", "knapsack", "back", "pack", "bag", "handbag", "purse", "shoulder",
    "glasses", "eyeglasses", "spectacles", "clear", "sunglasses", "shades", "sun", "hat", "cap", "headwear",
    "black", "blue", "brown", "green", "grey", "gray", "orange", "pink", "purple", "violet",
    "red", "crimson", "white", "yellow", "bright", "dark", "clean", "patterned", "multi-colored", "other"
]
WORD2IDX = {w: i for i, w in enumerate(UNIVERSAL_VOCAB)}
IDX2WORD = {i: w for i, w in enumerate(UNIVERSAL_VOCAB)}


class DynamicPromptEngine:
    """Generates natural language prompts with attribute dropout and synonym jitter."""

    def __init__(self, p_dropout: float = 0.15, max_length: int = 77):
        self.p_dropout = p_dropout
        self.max_length = max_length
        self._clip_tokenizer = self._initialize_tokenizer()

    def _initialize_tokenizer(self):
        """Loads official authentic open_clip tokenizer. Fallbacks are strictly prohibited."""
        try:
            import open_clip
            return open_clip.tokenize
        except Exception as e:
            raise RuntimeError(
                f"[DynamicPromptEngine] CRITICAL: Failed to load authentic open_clip.tokenize: {e}. "
                "Fallbacks are strictly prohibited per championship specification."
            ) from e

    def synthesize_prompt(
        self,
        attr_vec: Union[np.ndarray, torch.Tensor, List[int]],
        is_train: bool = True,
        anatomical_strip: Optional[str] = None
    ) -> str:
        """Synthesizes a fluent English prompt from a 40-D binary attribute vector."""
        if isinstance(attr_vec, torch.Tensor):
            attr_vec = attr_vec.detach().cpu().numpy()
        attr_vec = np.asarray(attr_vec)

        # Select target attribute indices
        if anatomical_strip == "head":
            target_indices = HEAD_ATTR_INDICES
        elif anatomical_strip == "torso":
            target_indices = TORSO_ATTR_INDICES
        elif anatomical_strip == "legs":
            target_indices = LEGS_ATTR_INDICES
        else:
            target_indices = list(range(len(CANONICAL_ATTRIBUTES)))

        phrases: List[str] = []
        for idx in target_indices:
            if idx >= len(attr_vec):
                continue
            if attr_vec[idx] == 1:
                # Attribute dropout during training forces invariant feature discovery
                if is_train and random.random() < self.p_dropout:
                    continue
                attr_name = CANONICAL_ATTRIBUTES[idx]
                synonyms = SYNONYM_LEXICON.get(attr_name, [attr_name.replace("-", " ").lower()])
                phrases.append(random.choice(synonyms) if is_train else synonyms[0])

        if is_train:
            random.shuffle(phrases)

        if not phrases:
            if anatomical_strip:
                return f"A surveillance crop of pedestrian {anatomical_strip} region."
            return "A surveillance photo of an unidentified pedestrian."

        prefix = "A surveillance photo of a "
        if anatomical_strip:
            prefix = f"A photo of the pedestrian's {anatomical_strip} part, "
        return prefix + ", ".join(phrases) + "."

    def tokenize(self, text_or_list: Union[str, List[str]]) -> torch.Tensor:
        """Tokenizes text string or batch of text into a [B, max_length] tensor using authentic BPE."""
        if isinstance(text_or_list, str):
            text_or_list = [text_or_list]

        tokens = self._clip_tokenizer(text_or_list)
        if tokens.shape[-1] != self.max_length:
            if tokens.shape[-1] < self.max_length:
                pad = torch.zeros((tokens.shape[0], self.max_length - tokens.shape[-1]), dtype=tokens.dtype)
                tokens = torch.cat([tokens, pad], dim=-1)
            else:
                tokens = tokens[:, :self.max_length]
        return tokens
