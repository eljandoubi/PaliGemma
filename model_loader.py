"""Model loader script"""

import os
import json
import glob
from typing import Tuple
from tqdm import tqdm
from transformers import AutoTokenizer
from safetensors import safe_open
from modeling_paligemma import (PaliGemmaForConditionalGeneration,
                                PaliGemmaConfig)


def load_hf_model(model_path: str,
                  device: str
                  ) -> Tuple[PaliGemmaForConditionalGeneration, AutoTokenizer]:
    """Load huggingface model into pytorch module."""

    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
    assert tokenizer.padding_side == "right"

    # Find all the *.safetensors files
    safetensors_files = glob.glob(os.path.join(model_path, "*.safetensors"))

    # ... and load them one by one in the tensors dictionary
    tensors = {}
    for safetensors_file in tqdm(safetensors_files,desc="Load weights"):
        with safe_open(safetensors_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)

    # Load the model's config
    with open(os.path.join(model_path, "config.json"), "r", encoding="utf-8") as f:
        model_config_file = json.load(f)
        config = PaliGemmaConfig.from_dict(model_config_file)

    # Create the model using the configuration
    model = PaliGemmaForConditionalGeneration(config).to(device)

    # Load the state dict of the model
    model.load_state_dict(tensors, strict=False)

    # Tie weights
    model.tie_weights()

    return (model, tokenizer)
