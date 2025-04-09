import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
from typing import Dict
import hdbscan 
plt.style.use('ggplot')

sys.path.append("src")
from clustering.k_means import KMeans

def plot_clusters(embeddings: np.ndarray, labels_dict: Dict, title: str = "Clustering Plot", save_path: str = None) -> None:
    """
    Plot clusters in a 2D space using the provided embeddings and labels.

    Args:
        embeddings (np.ndarray): The 2D embeddings to plot.
        labels (np.ndarray): The cluster labels for each embedding.
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

if __name__ == "__main__":
    # Example usage
    embeddings = torch.randn(100, 2)  # Replace with actual 2D embeddings

    clusterer = hdbscan.HDBSCAN(min_cluster_size=5)
    cluster_labels = clusterer.fit_predict(embeddings)
    clusters = {}
    for i, label in enumerate(cluster_labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(embeddings[i].numpy())
    print("Cluster labels by HDBSCAN:", cluster_labels)
    print("Clusters:", clusters)

    # Example usage with KMeans
    #kmeans = KMeans(k=6, tol=0)
    #centroids, labels, clusters = kmeans.fit(embeddings)

    plot_clusters(embeddings, clusters, title="KMeans Clustering")

