import os
import random
import numpy as np
import torch
import tqdm
import sys

from PIL import Image
from typing import List, Dict, Any

sys.path.append("src")
from embedding_models.clip import CLIPEmbeddingModel

def process_cifake_data(input_dir: str, num_samples: int = 1000) -> None:
    """
    Process CIFake data and save the processed images and metadata.

    Args:
        input_dir (str): Directory containing the CIFake dataset.
        output_dir (str): Directory to save the processed data.
        num_samples (int): Number of samples to process. Default is 1000.
    """

    # Initialize the CLIP model
    clip_model = CLIPEmbeddingModel()

    # Load data
    test_path = os.path.join(input_dir, "test")
    fake_dir = os.path.join(test_path, "FAKE")
    real_dir = os.path.join(test_path, "REAL")
    
    # Check if directories exist
    for dir_path in [fake_dir, real_dir]:
        if not os.path.exists(dir_path):
            raise FileNotFoundError(f"Directory {dir_path} does not exist.")
    
    # Check if the number of samples is less than the available images
    if num_samples > len(os.listdir(fake_dir)) or num_samples > len(os.listdir(real_dir)):
        raise ValueError(f"Number of samples {num_samples} exceeds available images in directories.")
        
    # Randomly sample specified number of images
    fake_image_paths = random.sample(os.listdir(fake_dir), num_samples)
    real_image_paths = random.sample(os.listdir(real_dir), num_samples)

    # Process images 
    real_images = []
    fake_images = []

    for img_path in fake_image_paths:
        img_path = os.path.join(fake_dir, img_path)
        img = Image.open(img_path)
        fake_images.append(img)
    
    for img_path in real_image_paths:
        img_path = os.path.join(real_dir, img_path)
        img = Image.open(img_path)
        real_images.append(img)


    # Store image embeddings for all real and fake images
    real_image_embed = torch.cat([clip_model.embed_images(real_images)])
    fake_image_embed = torch.cat([clip_model.embed_images(fake_images)])

    return real_image_embed, fake_image_embed

if __name__ == "__main__":
    input_dir = "datasets/cifake"
    num_samples = 5

    real_image_embed, fake_image_embed = process_cifake_data(input_dir, num_samples)

    print("Real Image Embeddings Shape:", real_image_embed.shape)
    print("Fake Image Embeddings Shape:", fake_image_embed.shape)
    print("Real Image Embeddings:", real_image_embed)
    print("Fake Image Embeddings:", fake_image_embed)

    