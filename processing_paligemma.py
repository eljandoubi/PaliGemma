"""PaliGemma processing script"""

from typing import Dict, List, Optional, Union, Tuple, Iterable
import numpy as np
from PIL import Image
import torch
from transformers import PreTrainedTokenizerFast

IMAGENET_STANDARD_MEAN = (0.5,) * 3
IMAGENET_STANDARD_STD = (0.5,) * 3
IMAGE_TOKEN = "<image>"
# These tokens are used for object detection (bounding boxes)
# Plus tokens are used for object segmentation
EXTRA_TOKENS = [f"<loc{i:0>4}>" for i in range(1024)] \
    + [f"<seg{i:0>3}>" for i in range(128)]


def rescale(
    image: np.ndarray, scale: float,
    dtype: np.dtype = np.float32
) -> np.ndarray:
    """rescale an image(np.ndarry) by a factor of scale(float)"""
    rescaled_image = image * scale
    rescaled_image = rescaled_image.astype(dtype)
    return rescaled_image


def resize(
        image: Image.Image,
        size: Tuple[int, int],
        resample: Optional[Image.Resampling] = None,
        reducing_gap: Optional[int] = None,
) -> np.ndarray:
    """Resize an image(PIL Image)"""
    height, width = size
    resized_image = image.resize(
        (width, height), resample=resample, reducing_gap=reducing_gap
    )
    return resized_image


def normalize(
        image: np.ndarray,
        mean: Union[float, Iterable[float]],
        std: Union[float, Iterable[float]],
) -> np.ndarray:
    """normalize an image(np.ndarry): (image - mean) / std"""
    mean = np.array(mean, dtype=image.dtype)
    std = np.array(std, dtype=image.dtype)
    image = (image - mean) / std
    return image


def process_images(
        images: List[Image.Image],
        size: Optional[Union[int, Tuple[int]]] = None,
        resample: Optional[Image.Resampling] = None,
        rescale_factor: Optional[float] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
) -> List[np.ndarray]:
    """Process images in order to be passed to PaliGemma modem"""
    if isinstance(size, int):
        height = width = size
    else:
        height, width = size[0], size[1]

    images = [
        resize(image=image, size=(height, width),
               resample=resample)
        for image in images
    ]
    # Convert each image to a numpy array
    images = [np.array(image) for image in images]
    # Rescale the pixel values to be in the range [0, 1]
    images = [rescale(image, scale=rescale_factor)
              for image in images]
    # Normalize the images to have mean 0 and standard deviation 1
    images = [normalize(image, mean=image_mean, std=image_std)
              for image in images]
    # Move the channel dimension to the first dimension.
    # The model expects images in the format [Channel, Height, Width]
    images = [image.transpose(2, 0, 1) for image in images]
    return images


def add_image_tokens_to_prompt(prefix_prompt: str, bos_token: str,
                               image_seq_len: int, image_token: str
                               ) -> str:
    """Make placeholder for images tokens in the prompt"""
    return f"{image_token * image_seq_len}{bos_token}{prefix_prompt}\n"


class PaliGemmaProcessor:
    """Process image and text for PaliGemma model"""

    def __init__(self, tokenizer: PreTrainedTokenizerFast,
                 num_image_tokens: int,
                 image_size: int):

        self.image_seq_length = num_image_tokens
        self.image_size = image_size

        tokenizer.add_special_tokens({"additional_special_tokens":
                                      [IMAGE_TOKEN]})
        tokenizer.add_tokens(EXTRA_TOKENS)

        self.image_token_id = tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)

        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False

        self.tokenizer = tokenizer

    def __call__(self,
                 text: List[str],
                 images: List[Image.Image],
                 padding: str = "longest",
                 truncation: bool = True,
                 ) -> Dict[str, torch.Tensor]:

        nb_txt = len(text)
        nb_imgs = len(images)
        assert nb_txt == nb_imgs == 1, f"Received {nb_imgs} images for {nb_txt} prompts."

        pixel_values = process_images(
            images=images,
            size=self.image_size,
            resample=Image.Resampling.BICUBIC,
            rescale_factor=1.0 / 255,
            image_mean=IMAGENET_STANDARD_MEAN,
            image_std=IMAGENET_STANDARD_STD,
        )
        # Convert the list of numpy arrays to a single numpy array
        # with shape [Batch_Size, Channel, Height, Width]
        pixel_values = np.stack(arrays=pixel_values,
                                axis=0)
        # Convert the numpy array to a PyTorch tensor
        pixel_values = torch.from_numpy(ndarray=pixel_values)
        # Prepend a `self.image_seq_length` number of image tokens 
        # to the prompt
        input_strings = [
            add_image_tokens_to_prompt(prefix_prompt=prompt,
                                       bos_token=self.tokenizer.bos_token,
                                       image_seq_len=self.image_seq_length,
                                       image_token=IMAGE_TOKEN,
                                       )
            for prompt in text]
        # Returns the input_ids and attention_mask as PyTorch tensors
        inputs = self.tokenizer(
            input_strings,
            return_tensors="pt",
            padding=padding,
            truncation=truncation,
        )

        return {"pixel_values": pixel_values, **inputs}
