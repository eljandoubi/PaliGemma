"""Model loader script"""

import os
import json
import glob
from typing import Tuple
from tqdm import tqdm
import transformers as hf
from safetensors import safe_open
from src.models.paligemma import (PaliGemmaForConditionalGeneration,
                                  PaliGemmaConfig)


def load_hf_model(model_id: str,
                  model_path: str,
                  device: str
                  ) -> Tuple[PaliGemmaForConditionalGeneration, hf.AutoTokenizer]:
    """Load huggingface model into pytorch module."""

    token = os.environ.get("HF_TOKEN")
    print("Found HF token:", token is not None)
    print(f"Download {model_id} from huggingface")
    hf.PaliGemmaForConditionalGeneration.from_pretrained(
        model_id, cache_dir=os.path.join(model_path, "tmp"),
        token=token).save_pretrained(model_path)

    tokenizer = hf.AutoTokenizer.from_pretrained(
        model_id, padding_side="right")
    assert tokenizer.padding_side == "right"

    # Find all the *.safetensors files
    safetensors_files = glob.glob(os.path.join(model_path, "*.safetensors"))

    # ... and load them one by one in the tensors dictionary
    tensors = {}
    for safetensors_file in tqdm(
            safetensors_files, desc="Load weights to RAM"):
        with safe_open(safetensors_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)

    # Load the model's config
    print("Load the model's config")
    with open(os.path.join(model_path, "config.json"), "r", encoding="utf-8") as f:
        model_config_file = json.load(f)
        config = PaliGemmaConfig.from_dict(model_config_file)

    # Create the model using the configuration
    model = PaliGemmaForConditionalGeneration(config).to(device)

    print("Load tensors to model")
    # Load the state dict of the model
    model.load_state_dict(tensors, strict=False)

    # Tie weights
    model.tie_weights()

    return (model, tokenizer)
