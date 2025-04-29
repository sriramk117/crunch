import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import hdbscan 
import altair as alt
alt.renderers.enable("browser") # Enable offline rendering for Altair
plt.style.use('ggplot')

from PIL import Image
from matplotlib import cm
from typing import Dict, List

sys.path.append("src")
from embedding_models.clip import CLIPEmbeddingModel
from embedding_models.dim_reduce import apply_umap
from clustering.k_means import KMeans
from language_models.labeling_agent import LabelingAgent
import pandas as pd

def compute_clip_emb(input_dir: str, num_samples: int = 1000, batch_size: int = 32) -> None:
    """
    Process image data and compute CLIP embeddings.

    Args:
        input_dir (str): Directory containing the images.
        num_samples (int): Number of samples to process. Default is 1000.
        batch_size (int): Number of images to process in a batch. Default is 32.
    """
    # Check if the input directory exists
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Directory {input_dir} does not exist.")
    
    # Check if the number of samples is less than the available images
    image_paths = os.listdir(input_dir)
    if num_samples > len(image_paths):
        raise ValueError(f"Number of samples {num_samples} exceeds available images in directory.")

    # Initialize the CLIP model
    clip_model = CLIPEmbeddingModel()

    # Load and compute embeddings for each of the images
    images = []
    sample_paths = []
    for img_path in image_paths:
        img_path = os.path.join(input_dir, img_path)
        img = Image.open(img_path)
        images.append(img)
        sample_paths.append(img_path)
    images = images[:num_samples]  # Limit to num_samples
    image_embeddings = clip_model.embed_images(images, batch_size=batch_size)

    return image_embeddings, sample_paths

def plot_clusters_matplt(embeddings: np.ndarray, labels_dict: Dict, title: str = "t-SNE", save_path: str = None) -> None:
    """
    Plot clusters in a 2D space using the provided embeddings and labels.

    Args:
        embeddings (np.ndarray): The 2D embeddings to plot.
        labels_dict (Dict): The cluster labels for each embedding.
        title (str): Title of the plot.
        save_path (str): Path to save the plot. If None, the plot will be shown instead.

    Returns:
        None
    """
    plt.figure(figsize=(10, 8))
    
    # Create a colormap
    unique_labels = labels_dict.keys()
    colors = cm.get_cmap('viridis', len(unique_labels))

    for i, label in enumerate(unique_labels):
        x_vals = [emb[0] for emb in labels_dict[label]]
        y_vals = [emb[1] for emb in labels_dict[label]]
        plt.scatter(x_vals, y_vals,
                    color=colors(i), label=f'Cluster {label}', alpha=0.6)

    plt.title(title)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    plt.close()


def plot_clusters(embeddings, labels_dict, title, save_path=None):
    """
    Plot clusters in a 2D space using Altair.

    Args:
        embeddings (np.ndarray): The 2D embeddings to plot.
        labels_dict (Dict): The cluster labels for each embedding.
        title (str): Title of the plot.
        save_path (str): Path to save the plot. If None, the plot will be shown instead.
    """
    data = []
    for label, points in labels_dict.items():
        for point in points:
            data.append({"x": point[0], "y": point[1], "cluster": label})
    df = pd.DataFrame(data)

    chart = alt.Chart(df).mark_circle(size=60).encode(
        x='x:Q',
        y='y:Q',
        color='cluster:N',
        tooltip=['x', 'y', 'cluster']
    ).properties(
        title=title,
        width=800,
        height=800
    ).interactive()

    chart.show()
    if save_path:
        chart.save(save_path)
        print(f"Plot saved to {save_path}")

def plot_clusters_images(embeddings, labels_dict, image_paths_dict, title, save_path=None):
    """
    Plot clusters in a 2D space using Altair with images.

    Args:
        embeddings (np.ndarray): The 2D embeddings to plot.
        labels_dict (Dict): The cluster labels for each embedding.
        image_paths_dict (Dict): Dictionary mapping embeddings to their corresponding image paths.
        title (str): Title of the plot.
        save_path (str): Path to save the plot. If None, the plot will be shown instead.
    """
    data = []
    for label, points in labels_dict.items():
        for point in points:
            img_path = image_paths_dict[tuple(point)]
            data.append({"x": point[0], "y": point[1], "cluster": label, "image": img_path})
    df = pd.DataFrame(data)

    chart = alt.Chart(df).mark_image().encode(
        x='x:Q',
        y='y:Q',
        url='image:N',
        tooltip=['x', 'y', 'cluster']
    ).properties(
        title=title,
        width=800,
        height=800
    ).interactive()

    chart.show()
    if save_path:
        chart.save(save_path)
        print(f"Plot saved to {save_path}")

def plot_clusters_bar_graph(embeddings, labels_dict, title, save_path=None):
    """
    Plot clusters in a bar graph using Altair.

    Args:
        embeddings (np.ndarray): The 2D embeddings to plot.
        labels_dict (Dict): The cluster labels for each embedding.
        title (str): Title of the plot.
        save_path (str): Path to save the plot. If None, the plot will be shown instead.
    """
    data = []
    for label, points in labels_dict.items():
        data.append({"Cluster": label, "Count": len(points)})
    df = pd.DataFrame(data)

    chart = alt.Chart(df).mark_bar().encode(
        x='Cluster:O',
        y='Count:Q',
        color='Cluster:N',
        tooltip=['Cluster', 'Count']
    ).properties(
        title=title,
        width=800,
        height=400
    )

    chart.show()
    if save_path:
        chart.save(save_path)
        print(f"Plot saved to {save_path}")

def cluster_and_plot(embeddings, method="kmeans", k=6, min_cluster_size=20, title="Clustering", save_path=None):
    """
    Perform clustering and plot the results.

    Args:
        embeddings (torch.Tensor): The embeddings to cluster.
        method (str): Clustering method ('kmeans' or 'hdbscan').
        k (int): Number of clusters for K-Means.
        min_cluster_size (int): Minimum cluster size for HDBSCAN.
        title (str): Title of the plot.
        save_path (str): Path to save the plot.
    """
    if method == "kmeans":
        if k is None:
            raise ValueError("k must be specified for KMeans clustering.")
        elif k <= 0:
            raise ValueError("k must be a positive integer for KMeans clustering.")
        elif k > len(embeddings):
            raise ValueError("k must be less than or equal to the number of embeddings for KMeans clustering.")
        
        kmeans = KMeans(k=k)
        centroids, labels, clusters = kmeans.fit(embeddings)
    elif method == "hdbscan":
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
        cluster_labels = clusterer.fit_predict(embeddings.np())
        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(embeddings[i].np())
    else:
        raise ValueError("Invalid clustering algorithm. Choose between following methods: 'kmeans' or 'hdbscan'.")

    plot_clusters(embeddings.np(), clusters, title=title, save_path=save_path)

def cluster_kmeans(
        directory_path, 
        num_samples=1000,
        batch_size=32,
        embedding_method="CLIP", 
        k=6, 
        title="KMeans Clustering", 
        api_key=None,
        save_path=None):
    """
    Perform KMeans clustering and plot the results.

    Args:
        embeddings (torch.Tensor): The embeddings to cluster.
        embedding_method (str): The embedding method used.
        num_samples (int): Number of samples to process.
        batch_size (int): Number of images to process in a batch.
        k (int): Number of clusters.
        title (str): Title of the plot.
        save_path (str): Path to save the plot.
    """
    embeddings = None
    image_paths = None
    if embedding_method == "CLIP":
        embeddings, image_paths = compute_clip_emb(directory_path, num_samples=num_samples, batch_size=batch_size)
    else:
        raise ValueError("Invalid embedding method. Choose 'CLIP'.")
    
    if embeddings is None or len(embeddings) == 0:
        raise ValueError("No embeddings were generated. Please check the input directory and embedding method.")
    
    # Apply dimensionality reduction
    embeddings = apply_umap(embeddings, n_components=2)
    
    if k is None:
        raise ValueError("k must be specified for KMeans clustering.")
    elif k <= 0:
        raise ValueError("k must be a positive integer for KMeans clustering.")
    elif k > len(embeddings):
        raise ValueError("k must be less than or equal to the number of embeddings for KMeans clustering.")
    
    kmeans = KMeans(k=k)
    centroids, labels, clusters = kmeans.fit(embeddings)

    # Check if user wants clusters to be semantically labeled
    # TODO: We assume that the user wants semantic labeling if an API key is provided. This 
    # can be improved by adding some setting or shifting from this Python library based infrastructure
    # to a more user-friendly interface.
    if api_key:
        labeling_agent = LabelingAgent(model="gpt-4o-mini", clusters=clusters, api_key=api_key)
        clusters, labels = labeling_agent.label_clusters()
        print(f"Clusters labeled by OpenAI API. Here are the labels: {labels}")

    plot_clusters(embeddings.numpy(), clusters, title=title, save_path=save_path)

def cluster_hdbscan(
        directory_path, 
        embedding_method="CLIP", 
        num_samples=1000,
        batch_size=32,
        min_cluster_size=20,
        reveal_images=False, 
        title="HDBSCAN Clustering", 
        api_key=None,
        save_path=None):
    """
    Perform HDBSCAN clustering and plot the results.

    Args:
        embeddings (torch.Tensor): The embeddings to cluster.
        embedding_method (str): The embedding method used.
        num_samples (int): Number of samples to process.
        batch_size (int): Number of images to process in a batch.
        min_cluster_size (int): Minimum cluster size for HDBSCAN.
        reveal_images (bool): Whether to reveal images in the plot.
        title (str): Title of the plot.
        api_key (str): OpenAI API key for semantic labeling.
        save_path (str): Path to save the plot.
    """
    embeddings = None
    image_paths = None
    if embedding_method == "CLIP":
        embeddings, image_paths = compute_clip_emb(directory_path, num_samples=num_samples, batch_size=batch_size)
    else:
        raise ValueError("Invalid embedding method. Choose 'CLIP'.")
    
    if embeddings is None or len(embeddings) == 0:
        raise ValueError("No embeddings were generated. Please check the input directory and embedding method.")

    # Apply dimensionality reduction
    embeddings = apply_umap(embeddings, n_components=2)

    # Create a dictionary mapping embeddings to their corresponding image paths
    image_paths_dict = {}
    for embedding, image_path in zip(embeddings, image_paths):
        image_paths_dict[tuple(embedding)] = image_path

    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    cluster_labels = clusterer.fit_predict(embeddings)
    clusters = {}
    for i, label in enumerate(cluster_labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(embeddings[i]) 

    # Check if user wants clusters to be semantically labeled
    # TODO: We assume that the user wants semantic labeling if an API key is provided. This 
    # can be improved by adding some setting or shifting from this Python library based infrastructure
    # to a more user-friendly interface.
    if api_key:
        labeling_agent = LabelingAgent(model="gpt-4o-mini", clusters=clusters, image_paths_dict=image_paths_dict, api_key=api_key)
        clusters, labels = labeling_agent.label_clusters()

    if reveal_images:
        plot_clusters_images(embeddings, clusters, image_paths_dict, title=title, save_path=save_path)
    else:
        plot_clusters(embeddings, clusters, title=title, save_path=save_path)
    

def cluster_bar_graph(
        directory_path, 
        embedding_method="CLIP", 
        num_samples=1000,
        batch_size=32,
        min_cluster_size=20,
        title="HDBSCAN Clustering", 
        api_key=None,
        save_path=None):
    """
    Perform HDBSCAN clustering and plot the results as a bar graph.
    """

    embeddings = None
    image_paths = None
    if embedding_method == "CLIP":
        embeddings, image_paths = compute_clip_emb(directory_path, num_samples=num_samples, batch_size=batch_size)
    else:
        raise ValueError("Invalid embedding method. Choose 'CLIP'.")
    
    if embeddings is None or len(embeddings) == 0:
        raise ValueError("No embeddings were generated. Please check the input directory and embedding method.")
    
    # Apply dimensionality reduction
    embeddings = apply_umap(embeddings, n_components=2)

    # Create a dictionary mapping embeddings to their corresponding image paths
    image_paths_dict = {}
    for embedding, image_path in zip(embeddings, image_paths):
        image_paths_dict[tuple(embedding)] = image_path

    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    cluster_labels = clusterer.fit_predict(embeddings)
    clusters = {}
    for i, label in enumerate(cluster_labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(embeddings[i]) 

    # Check if user wants clusters to be semantically labeled
    # TODO: We assume that the user wants semantic labeling if an API key is provided. This 
    # can be improved by adding some setting or shifting from this Python library based infrastructure
    # to a more user-friendly interface.
    if api_key:
        labeling_agent = LabelingAgent(model="gpt-4o-mini", clusters=clusters, image_paths_dict=image_paths_dict, api_key=api_key)
        clusters, labels = labeling_agent.label_clusters()

    plot_clusters_bar_graph(embeddings, clusters, title=title, save_path=save_path)

if __name__ == "__main__":
    # Example usage
    embeddings = torch.randn(1500, 2)  # Replace with actual 2D embeddings
    clusterer = hdbscan.HDBSCAN(min_cluster_size=50)
    clusterer.fit(embeddings)
    labels = clusterer.labels_
    clusters = {}
    for i in range(len(labels)):
        if labels[i] not in clusters:
            clusters[labels[i]] = []
        clusters[labels[i]].append(embeddings[i].np())
    # clusters = {}
    # for i, label in enumerate(cluster_labels):
    #     if label not in clusters:
    #         clusters[label] = []
    #     clusters[label].append(embeddings[i].np())
    #print("Cluster labels by HDBSCAN:", cluster_labels)
    #print("Clusters:", clusters)

    # Example usage with KMeans
    kmeans = KMeans(k=20, tol=0)
    centroids, labels, clusters = kmeans.fit(embeddings)
    print("Cluster labels by KMeans:", clusters)

    plot_clusters(embeddings.np(), clusters, title="KMeans Clustering")

    #plot_clusters(embeddings, clusters, title="HDBSCAN Clustering")
    #plot_clusters_matplt(embeddings, clusters, title="KMeans Clustering")

