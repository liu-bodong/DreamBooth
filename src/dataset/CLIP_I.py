import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import CLIPProcessor, CLIPModel


class CLIP(nn.Module):
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        super(CLIP, self).__init__()
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def forward(self, images, texts):
        inputs = self.processor(
            text=texts, images=images, return_tensors="pt", padding=True
        )
        outputs = self.model(**inputs)
        return outputs.text_embeds, outputs.image_embeds


if __name__ == "__main__":
    # Example usage
    model = CLIP()
    images = [
        "path/to/image1.jpg",
        "path/to/image2.jpg",
    ]  # Replace with actual image paths
    texts = ["A description of image 1", "A description of image 2"]

    text_embeddings, image_embeddings = model(images, texts)
    print("Text Embeddings:", text_embeddings)
    print("Image Embeddings:", image_embeddings)
