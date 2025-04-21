import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
from typing import Dict
import hdbscan 
import altair as alt
alt.renderers.enable("browser") # Enable offline rendering for Altair
plt.style.use('ggplot')

sys.path.append("src")
from clustering.k_means import KMeans
import pandas as pd

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

def plot_clusters(embeddings: np.ndarray, labels_dict: Dict, title: str = "Semantic Clustering", save_path: str = None) -> None:
    # Prepare data for Altair
    data = []
    for label, points in labels_dict.items():
        for point in points:
            data.append({"x": point[0], "y": point[1], "cluster": label})
    df = pd.DataFrame(data)
    
    # Create Altair plot
    chart = alt.Chart(df).mark_circle(size=60).encode(
        x='x:Q',
        y='y:Q',
        color='cluster:N',
        tooltip=['x', 'y', 'cluster']
    ).properties(
        title=title,
        width=800,
        height=800
    )
    chart = chart.interactive()  # Enable zooming and panning

    chart.show()
    if save_path:
        chart.save(save_path)
        print(f"Plot saved to {save_path}")

if __name__ == "__main__":
    # Example usage
    embeddings = torch.randn(1500, 2)  # Replace with actual 2D embeddings
    print(embeddings.shape)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=50)
    clusterer.fit(embeddings)
    labels = clusterer.labels_
    clusters = {}
    for i in range(len(labels)):
        if labels[i] not in clusters:
            clusters[labels[i]] = []
        clusters[labels[i]].append(embeddings[i].numpy())
    # clusters = {}
    # for i, label in enumerate(cluster_labels):
    #     if label not in clusters:
    #         clusters[label] = []
    #     clusters[label].append(embeddings[i].numpy())
    #print("Cluster labels by HDBSCAN:", cluster_labels)
    #print("Clusters:", clusters)

    # Example usage with KMeans
    kmeans = KMeans(k=20, tol=0)
    centroids, labels, clusters = kmeans.fit(embeddings)
    print("Cluster labels by KMeans:", clusters)

    plot_clusters(embeddings, clusters, title="KMeans Clustering")

    #plot_clusters(embeddings, clusters, title="HDBSCAN Clustering")
    #plot_clusters_matplt(embeddings, clusters, title="KMeans Clustering")

