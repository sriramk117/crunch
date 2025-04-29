import torch
import numpy as np
from typing import List, Dict, Any
from tqdm import tqdm
import json

class GMM:
    def __init__(self, n_components: int, max_iter: int = 100, tol: float = 1e-4):
        """
        Initialize Gaussian Mixture Model (GMM) clustering.

        Args:
            n_components (int): The number of mixture components.
            max_iter (int): Maximum number of iterations.
            tol (float): Tolerance to declare convergence.
        """
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.means_ = None
        self.covariances_ = None
        self.weights_ = None
        self.labels_ = None

    