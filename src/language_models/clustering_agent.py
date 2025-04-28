import openai
import os
import numpy as np
from typing import List, Dict
from PIL import Image
import random
from tqdm import tqdm

class ClusteringAgent:
    def __init__(self, model: str = "gpt-3.5-turbo", api_key: str = None):
        """
        Initialize the ClusteringAgent with the specified OpenAI model.
        
        Args:
            model (str): The OpenAI model to use for clustering.
        """
        self.model = model
        self.clusters = {}
        self.api_key = api_key
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        openai.api_key = self.api_key

        self.role_prompt = f"""You are a labeling agent. Your responses are meant to be one or a few words long (i.e. a phrase).
                            Be direct and concise and get to the point; minimize tokens. Don’t elaborate unless requested. 
                            Don’t be redundant or repetitive. Don’t provide “recaps” that just duplicate what you already said. """


    def find_center(self, embeddings: np.ndarray, image_paths: List[str]) -> np.ndarray:
        """
        Find the centroid of a cluster of embeddings.
        
        Args:
            embeddings (np.ndarray): The embeddings to find the centroid of.
        
        Returns:
            np.ndarray: The centroid of the embeddings.
        """
        center = np.mean(embeddings, axis=0)
        l2_distances = np.linalg.norm(embeddings - center, axis=1)
        closest_index = np.argmin(l2_distances)
        center_embedding = embeddings[closest_index]
        center_image_path = image_paths[closest_index]

        return center_embedding, center_image_path

    def label_clusters(self) -> Dict:
        """
        Label clusters using OpenAI's GPT model.
        
        Args:
            clusters (Dict): Dictionary containing cluster data.
            model (str): The OpenAI model to use for labeling.
        
        Returns:
            Dict: Dictionary with labeled clusters.
        """
        
        labeled_clusters = {}
        
        for cluster_id, embeddings in self.clusters.items():

            prompt = f"""Give a brief label or short phrase (with a maximum of three words) that clearly categorizes/describes 
                        the attached image."""
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[{
                    "role": "system",
                    "content": self.role_prompt
                }, {
                    "role": "user",
                    "content": prompt
                }, {
                    "role": "user",
                    "content": f"Image: {embeddings}"
                }],
            )
            label = response.choices[0].message['content'].strip()  # Added strip() to clean up the response
            labeled_clusters[cluster_id] = label
        
        return labeled_clusters
