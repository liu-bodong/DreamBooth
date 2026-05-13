import torch
from torch.nn import functional as F
from torchvision import transforms


class DINOEvaluator:
    def __init__(self, device):
        self.device = device
        self.model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16').to(device)
        self.model.eval()
        
        for param in self.model.parameters():
            param.requires_grad = False
            
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
    def embed(self, images):
        """
        Return the DINO embeddings for a batch of images.

        Args:
            images (PIL.Image): a list of PIL images
        """
        with torch.no_grad():
            images = torch.stack([self.transform(img) for img in images]).to(self.device)
            # images = images.view(-1, 3, 224, 224)
            embeddings = self.model(images)
            # print(f"DINO embeddings shape: {embeddings.shape}")
            embeddings = F.normalize(embeddings, p=2, dim=1, out=embeddings)
        return embeddings
        
        
    def dino_score(self, generated, real):
        """
        Compute the DINO score matrix between generated and real images.

        Args:
            generated (_type_): _description_
            real (_type_): _description_
        """
        generated_embeddings = self.embed(generated)
        real_embeddings = self.embed(real)

        # This is pairsise, wrong?
        # similarity = F.cosine_similarity(generated_embeddings, real_embeddings)
        
        # this computes the avg sim comparing each to all
        sim_matrix = generated_embeddings @ real_embeddings.T
        return sim_matrix
        # score = sim_matrix.mean().item()
        # return score
    