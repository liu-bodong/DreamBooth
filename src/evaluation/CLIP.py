import torch
from torch.nn import functional as F
import torchvision
from torchvision import transforms


from transformers import CLIPProcessor, CLIPModel

class CLIPEvaluator:
    def __init__(self, device):
        self.device = device
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        self.model.eval()
        
        for param in self.model.parameters():
            param.requires_grad = False
            
    def embed(self, images):
        """
        Return the CLIP embeddings for a batch of images.

        Args:
            images (PIL.Image): a list of PIL images
        """
        with torch.no_grad():
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            vision_outputs = self.model.vision_model(pixel_values=inputs["pixel_values"])
            image_features = self.model.visual_projection(vision_outputs.pooler_output)
            embeddings = F.normalize(image_features, dim=-1)
        return embeddings
        
        
    def clip_i_score(self, generated, real):
        """
        Compute the CLIP score between the generated images and real images

        Args:
            generated (_type_): _description_
            real (_type_): _description_
        """
        generated_embeddings = self.embed(generated)
        real_embeddings = self.embed(real)

        sim_matrix = generated_embeddings @ real_embeddings.T
        score = sim_matrix.mean().item()
        return score
    

    def clip_t_score(self, generated_images, real_texts):
        """
        Compute the CLIP score between the generated image with its corresponding prompt
        
        """
        with torch.no_grad(): 
            generated_embeddings = self.embed(generated_images)

            inputs = self.processor(text=real_texts, return_tensors="pt", padding=True).to(self.device)
            text_outputs = self.model.text_model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
            text_features = self.model.text_projection(text_outputs.pooler_output)
            real_embeddings = F.normalize(text_features, dim=-1)
            
            sim = F.cosine_similarity(generated_embeddings, real_embeddings)
        
        return sim.item()