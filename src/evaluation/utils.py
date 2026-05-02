import torch
import torchvision
from torchvision import transforms

import PIL.Image as Image
import pathlib

def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return image

def load_images_tensor(image_paths, transform, device):
    images = [load_image(path) for path in image_paths]
    images_tensor = torch.stack([transform(img) for img in images]).to(torch.device)
    return images_tensor