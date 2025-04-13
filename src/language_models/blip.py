from PIL import Image
import requests
import torch
import os
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from models.blip import blip_decoder
from typing import Dict, Any, List, Tuple

class BLIPModel:
    def __init__(self, model_path: str, device: str = 'cpu', image_size: int = 384):
        """
        Initialize the BLIP model.

        Args:
            model_path (str): Path to the BLIP model weights.
            device (str): Device to load the model on ('cpu' or 'cuda').
            image_size (int): Size of the input images. Assumed to be square.
        """
        self.device = device
        self.image_size = image_size
        self.model = self.load_blip_model(model_path)

    def load_blip_model(self, model_path: str) -> Any:
        """
        Load the BLIP model.

        Args:
            model_path (str): Path to the BLIP model weights.
            device (str): Device to load the model on ('cpu' or 'cuda').
            image_size (int): Size of the input images. Assumed to be square.

        Returns:
            Any: Loaded BLIP model.
        """
        blip_model = blip_decoder(
            pretrained=model_path,
            image_size=self.image_size,
            vit='base',
            num_query_token=100
        ).to(self.device)
        
        blip_model.eval()
        return blip_model
    
    def load_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Load and preprocess an image.

        Args:
            image_path (str): Path to the image.
            image_size (Tuple[int, int]): Desired size of the image.

        Returns:
            torch.Tensor: Preprocessed image tensor.
        """
        transform = transforms.Compose([
            transforms.Resize(self.image_size, interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

        return transform(image).unsqueeze(0).to(self.device)
    
    def generate_caption(self, image: torch.Tensor) -> str:
        """
        Generate a caption for the given image.

        Args:
            image (torch.Tensor): Preprocessed image tensor.

        Returns:
            str: Generated caption.
        """
        with torch.no_grad():
            caption = self.model.generate(image, sample=False, num_beams=3, max_length=16, min_length=5)
        
        return caption[0]
    
if __name__ == "__main__":
    # Example usage
    model_path = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base.pth"
    image_path = "datasets/cifake/test/REAL/0040.jpg"
    model = BLIPModel(model_path=model_path)
    image = model.load_image(image_path)
    caption = model.generate_caption(image)
    print(caption)