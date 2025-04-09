import os
import random
import numpy as np
import torch
import tqdm
import time
import sys
import argparse
import json
import hdbscan

from PIL import Image
from typing import List, Dict, Any
from tqdm import tqdm

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

def run_k_means_clustering(embeddings: torch.Tensor, k: int = 6, title: str = "K-Means Clustering", save_path: str = None) -> None:
    """
    Run KMeans clustering on the embeddings.

    Args:
        embeddings (torch.Tensor): The embeddings to cluster.
        k (int): Number of clusters. Default is 6.
        title (str): Title of the plot.
        save_path (str): Path to save the plot. If None, the plot will be shown instead.
    """
    kmeans = KMeans(k=k)
    centroids, labels, clusters = kmeans.fit(embeddings)
    
    # Plot clusters
    plot_clusters(embeddings.numpy(), clusters, title=title, save_path=save_path)

def run_hdbscan_clustering(embeddings: torch.Tensor, min_cluster_size: int = 20, title: str = "HDBSCAN Clustering", save_path: str = None) -> None:
    """
    Run HDBSCAN clustering on the embeddings.

    Args:
        embeddings (torch.Tensor): The embeddings to cluster.
        min_cluster_size (int): Minimum cluster size. Default is 20.
        title (str): Title of the plot.
        save_path (str): Path to save the plot. If None, the plot will be shown instead.
    """
    import hdbscan
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    cluster_labels = clusterer.fit_predict(embeddings.numpy())
    
    # Create clusters dictionary
    clusters = {}
    for i, label in enumerate(cluster_labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(embeddings[i].numpy())

    # Plot clusters
    plot_clusters(embeddings.numpy(), clusters, title=title, save_path=save_path)

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Process CIFake dataset and run clustering.")
    parser.add_argument("--input_dir", type=str, default="datasets/cifake", help="Input directory containing CIFake dataset.")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to process.")
    parser.add_argument("--output_dir", type=str, default="plots", help="Output directory to save processed data.")
    args = parser.parse_args()

    input_dir = args.input_dir
    num_samples = args.num_samples
    output_dir = args.output_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

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

    # Run k-means clustering on real and fake image embeddings
    run_k_means_clustering(reduced_real_embed, k=6, title="CIFAKE Real Image Dataset Semantic Clustering", save_path=os.path.join(output_dir, "real_kmeans_clusters.png"))
    run_k_means_clustering(reduced_fake_embed, k=6, title="CIFAKE Synthetic Image Dataset Semantic Clustering", save_path=os.path.join(output_dir, "fake_kmeans_clusters.png"))
    print("K-Means clustering completed.")

    # Run HDBSCAN clustering
    run_hdbscan_clustering(reduced_real_embed, min_cluster_size=20, title="CIFAKE Real Image Dataset Semantic Clustering", save_path=os.path.join(output_dir, "real_hdbscan_clusters.png"))
    run_hdbscan_clustering(reduced_fake_embed, min_cluster_size=20, title="CIFAKE Synthetic Image Dataset Semantic Clustering", save_path=os.path.join(output_dir, "fake_hdbscan_clusters.png"))
    print("Clustering completed.")

    