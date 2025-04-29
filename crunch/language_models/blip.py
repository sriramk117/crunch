from PIL import Image
import requests
import torch
import os
import sys
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from typing import Dict, Any, List, Tuple

class ImageCaptioningModel:
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
        self.model, self.tokenizer = self.load_model(model_path)

    def load_model(self, model_path: str) -> Any:
        """
        Load the image captioning model.

        Args:
            model_path (str): Path to the image captioning model weights.
            device (str): Device to load the model on ('cpu' or 'cuda').
            image_size (int): Size of the input images. Assumed to be square.

        Returns:
            Any: Loaded image captioning model.
        """
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float16)
        model = model.to(device='cpu', dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model.eval()
        return model, tokenizer
    
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
    
    def generate_caption(self, image, temperature: float = 0.7) -> str:
        """
        Generate a caption for the given image.

        Args:
            image: Preprocessed image.
            temperature (float): Sampling temperature for generation.

        Returns:
            str: Generated caption.
        """
        question = 'What is in the image?'
        conversation = [{'role': 'user', 'content': question}]

        caption, context, _ = self.model.chat(
            image=image,
            msgs=conversation,
            context=None,
            tokenizer=self.tokenizer,
            sampling=True,
            temperature=temperature,
        )
    
        return caption
    
if __name__ == "__main__":
    # Example usage
    model_path = "openbmb/MiniCPM-V"
    image_path = "datasets/cifake/test/REAL/0040.jpg"
    model = ImageCaptioningModel(model_path=model_path)
    image = Image.open(image_path).convert("RGB")
    caption = model.generate_caption(image)
    print(caption)