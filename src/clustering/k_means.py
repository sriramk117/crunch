import torch
import numpy as np
from typing import List, Dict, Any
from tqdm import tqdm
import json

class KMeans:
    def __init__(self, k: int, max_iter: int = 50, tol: float = 1e-4):
        """
        Initialize KMeans clustering.

        Args:
            k (int): The number of clusters.
            max_iter (int): Maximum number of iterations.
            tol (float): Tolerance to declare convergence.
        """
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = None
        self.labels_ = None

    def terminate(self, old_centroids: torch.Tensor, iter: int) -> bool:
        """
        Check if the algorithm should terminate.

        Args:
            old_centroids (torch.Tensor): Old centroids.
            new_centroids (torch.Tensor): New centroids.
            iter (int): Current iteration.

        Returns:
            bool: True if the algorithm should terminate, False otherwise.
        """
        if iter >= self.max_iter:
            return True
        return torch.norm(old_centroids - self.centroids) < self.tol
    
    def fit(self, X: torch.Tensor) -> None:
        """
        Fit the K-Means model to the data.
        
        Args:
            X (torch.Tensor): Input data of shape (n_samples, n_features).
        """
        
        self.centroids = X[torch.randperm(X.size(0))[:self.k]]
        old_centroids = torch.zeros_like(self.centroids)

        iter = 0
        pbar = tqdm(total=self.max_iter, desc="KMeans Iterations", unit="iteration")
        pbar.set_postfix({"centroids": self.centroids.tolist()})
        while not self.terminate(old_centroids, iter):
            iter += 1
            old_centroids = self.centroids.clone()

            # Assign labels based on the nearest centroid
            self.labels_ = self.assign_labels(X)

            # Update centroids
            self.centroids = self.update_centroids(X)

            pbar.update(1)
            pbar.set_postfix({"centroids": self.centroids.tolist()})

        pbar.close()
        # Final assignment of labels
        self.labels_ = self.assign_labels(X)

        clusters = {}
        for sample in range(X.size(0)):
            label = self.labels_[sample].item()
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(X[sample].tolist())
        
        cluster_keys = sorted(clusters.keys())
        clusters_sorted = {key: clusters[key] for key in cluster_keys}
        return self.centroids, self.labels_, clusters_sorted

    def assign_labels(self, X: torch.Tensor) -> torch.Tensor:
        """
        Assign labels to each point based on the nearest centroid.

        Args:
            X (torch.Tensor): Input data of shape (n_samples, n_features).

        Returns:
            torch.Tensor: Labels for each point.
        """
        distances = torch.cdist(X, self.centroids)
        return torch.argmin(distances, dim=1)
    
    def update_centroids(self, X: torch.Tensor) -> torch.Tensor:
        """
        Update centroids based on the current labels.

        Args:
            X (torch.Tensor): Input data of shape (n_samples, n_features).

        Returns:
            torch.Tensor: Updated centroids.
        """
        new_centroids = torch.zeros(self.k, X.size(1))
        for i in range(self.k):
            new_centroids[i] = X[self.labels_ == i].mean(dim=0)
        return new_centroids

if __name__ == "__main__":
    # Example usage
    data = torch.randn(100, 2)  # 100 samples, 2 features
    kmeans = KMeans(k=10, tol=0)
    centroids, labels, clusters = kmeans.fit(data)
    
    print("Data:", data)
    print("Centroids:", centroids)
    print("Labels:", labels)
    print("Clusters:", json.dumps(clusters, indent=2))