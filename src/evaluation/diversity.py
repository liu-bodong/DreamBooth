import torch
from torch.nn import functional as F
import torchvision
from torchvision import transforms
import PIL.Image as Image
import pathlib
import lpips
from itertools import combinations

class DiversityEvaluator:
    def __init__(self, device):
        self.device = device
        
        # TODO: should we call it model instead?
        # as we might use other methods?
        self.lpips = lpips.LPIPS(net='alex').to(self.device)
        
        self.lpips.eval()
        for param in self.lpips.parameters():
            param.requires_grad = False
            
        self.transform = transforms.Compose([
            # transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(-1, 1),  # LPIPS expects inputs in the range [-1, 1]
        ])

    def diversity_score(self, images):
        images = torch.stack([self.transform(img) for img in images]).to(self.device)
        # images.shape = (N, 3, H, W)
        N = images.shape[0]
        index_pairs = list(combinations(range(N), 2))
        scores = []
        for i, j in index_pairs:
            img1 = images[i].unsqueeze(0)  # Shape: (1, 3, H, W)
            img2 = images[j].unsqueeze(0)  # Shape: (1, 3, H, W)
            score = self.lpips(img1, img2).item()
            scores.append(score)
            
        scores = torch.tensor(scores)
        
        return scores
        