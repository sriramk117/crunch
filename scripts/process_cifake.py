import os
import random
import numpy as np
import torch
import tqdm
import time
import sys

from PIL import Image
from typing import List, Dict, Any

sys.path.append("src")
from embedding_models.clip import CLIPEmbeddingModel
from embedding_models.dim_reduce import apply_umap
from clustering.k_means import KMeans
from data_viz.clustering_plot import plot_clusters

def compute_emb(input_dir: str, num_samples: int = 1000, batch_size: int = 32) -> None:
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
    real_image_embed = clip_model.embed_images(images=real_images, batch_size=batch_size)
    fake_image_embed = clip_model.embed_images(images=fake_images, batch_size=batch_size)

    return real_image_embed, fake_image_embed

def run_clustering(embeddings: torch.Tensor, k: int = 6, title: str = "K-Means Clustering") -> None:
    """
    Run KMeans clustering on the embeddings.

    Args:
        embeddings (torch.Tensor): The embeddings to cluster.
        k (int): Number of clusters. Default is 6.
    """
    kmeans = KMeans(k=k)
    centroids, labels, clusters = kmeans.fit(embeddings)
    
    # Plot clusters
    plot_clusters(embeddings.numpy(), clusters, title=title)

if __name__ == "__main__":
    input_dir = "datasets/cifake"
    num_samples = 100

    start_time = time.time()

    real_image_embed, fake_image_embed = compute_emb(input_dir, num_samples)

    # Elapsed time
    elapsed_time = time.time() - start_time
    print(f"Processing completed in {elapsed_time:.2f} seconds.")

    print("Real Image Embeddings Shape:", real_image_embed.shape)
    print("Fake Image Embeddings Shape:", fake_image_embed.shape)
    print("Real Image Embeddings:", real_image_embed)
    print("Fake Image Embeddings:", fake_image_embed)

    # Apply UMAP for dimensionality reduction
    reduced_real_embed = apply_umap(real_image_embed.numpy(), n_components=2)
    reduced_fake_embed = apply_umap(fake_image_embed.numpy(), n_components=2)

    reduced_real_embed = torch.tensor(reduced_real_embed)
    reduced_fake_embed = torch.tensor(reduced_fake_embed)

    # Run clustering on real and fake image embeddings
    run_clustering(reduced_real_embed, k=6, title="CIFAKE Real Image Dataset Semantic Clustering")
    run_clustering(reduced_fake_embed, k=6, title="CIFAKE Synthetic Image Dataset Semantic Clustering")
    print("Clustering completed.")

    