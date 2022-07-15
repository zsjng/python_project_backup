import json
from datetime import datetime

import cv2
import torch
from einops import rearrange
from loguru import logger
from torchvision.transforms import transforms
from torchvision.utils import save_image


def imagenet_classnames() -> dict:
    with open("./data/imagenet_class_index.json") as f:
        class_names = json.load(f)
        return class_names


def get_timestamp() -> str:
    return datetime.now().strftime("%y%m%d_%H%M%S")


def shap_normalize(image: torch.Tensor) -> torch.Tensor:
    """
    This is only used for normalizing the SHAP imagenet50 distribution dataset.
    """
    image /= 255.0
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    image = (image - mean) / std
    return torch.tensor(rearrange(image, "b w h c -> b c w h")).float()


def load_single_image_from_path(image_path: str) -> torch.Tensor:
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    tfs = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return tfs(image).unsqueeze(dim=0)  # type: ignore


def normalise(image: torch.Tensor) -> torch.Tensor:
    normalise_func = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
    )
    return normalise_func(image)


def inverse_normalise(image: torch.Tensor) -> torch.Tensor:
    inverse_normalise_func = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
    )
    return inverse_normalise_func(image)


def save_img_tensor(image: torch.Tensor, filename: str, denorm: bool = True) -> None:
    image_copy = image.clone()

    if image.shape[0] == 3:
        # image is in shape for dwt input - (c, b, w, h)
        image_copy = rearrange(image_copy, "c b w h -> b c w h")

    if denorm:
        image_copy = inverse_normalise(image_copy)

    # assert that the image is of shape (b, c, w, h)
    assert image_copy.shape[0] == 1 and image_copy.shape[1] == 3

    save_image_path = f"{filename}_{get_timestamp()}.png"
    save_image(image_copy, save_image_path)
    logger.debug(f"Saved image to {save_image_path}.")
