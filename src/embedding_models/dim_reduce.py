import numpy as np
from sklearn.decomposition import PCA
import umap
import torch
from tqdm import tqdm
import time

def apply_pca(embeddings, n_components=2):
    """
    Apply Principal Component Analysis (PCA) to reduce the dimensionality of embeddings.

    Parameters:
        embeddings (np.ndarray): The input embeddings to reduce.
        n_components (int): The number of dimensions to reduce to.

    Returns:
        np.ndarray: The embeddings reduced to n_components dimensions.
    """
    start_time = time.time()
    pca = PCA(n_components=n_components)
    reduced_embeddings = pca.fit_transform(embeddings)

    # Convert back to torch tensor if the input was a torch tensor
    reduced_embeddings = torch.tensor(reduced_embeddings).numpy()
    end_time = time.time()
    print(f"PCA reduction took {end_time - start_time:.2f} seconds")
    return reduced_embeddings

def apply_umap(embeddings, n_components=2, n_neighbors=15, min_dist=0.1, random_state=None):
    """
    Apply Uniform Manifold Approximation and Projection (UMAP) to reduce the dimensionality of embeddings.

    Parameters:
        embeddings (np.ndarray): The input embeddings to reduce.
        n_components (int): The number of dimensions to reduce to.
        n_neighbors (int): The size of the local neighborhood (in terms of number of neighboring points) used for manifold approximation.
        min_dist (float): The minimum distance between points in the low-dimensional space.
        random_state (int or None): Random seed for reproducibility.

    Returns:
        np.ndarray: The embeddings reduced to n_components dimensions.
    """
    start_time = time.time()
    reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state)
    reduced_embeddings = reducer.fit_transform(embeddings)
    end_time = time.time()
    print(f"UMAP reduction took {end_time - start_time:.2f} seconds")
    return reduced_embeddings

if __name__ == "__main__":
    # Testing the PCA and UMAP functions
    embeddings = np.random.rand(100, 512)  # Replace with actual embeddings
    reduced_embeddings_pca = apply_pca(embeddings, n_components=2)
    reduced_embeddings_umap = apply_umap(embeddings, n_components=2)
    
    print("Original Embeddings Shape:", embeddings.shape)
    print("PCA Reduced Embeddings Shape:", reduced_embeddings_pca.shape)
    print("UMAP Reduced Embeddings Shape:", reduced_embeddings_umap.shape)
    print("PCA Reduced Embeddings:", reduced_embeddings_pca)
    print("UMAP Reduced Embeddings:", reduced_embeddings_umap)