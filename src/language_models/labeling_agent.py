import openai
import os
import numpy as np
from typing import List, Dict
from PIL import Image
import random
from tqdm import tqdm
from openai import OpenAI
import base64

class LabelingAgent:
    """
    A class to handle clustering and labeling of image embeddings using OpenAI's family of GPT vision models.
    This class is designed to work with the OpenAI API to generate labels for clusters of images based on their embeddings.
    """
    def __init__(self, model: str = "gpt-4o-mini", clusters: Dict = None, image_paths_dict: Dict = None, api_key: str = None):
        """
        Initialize the LabelingAgent with the specified OpenAI model.
        
        Args:
            model (str): The OpenAI model to use for clustering.
            clusters (Dict): Dictionary that holds clustered embeddings.
            image_paths_dict (Dict): Dictionary mapping embeddings to their corresponding image paths.
            api_key (str): OpenAI API key for accessing the model.
        """
        self.model = model
        self.clusters = clusters if clusters is not None else {}
        self.image_paths_dict = image_paths_dict if image_paths_dict is not None else {}
        self.api_key = api_key
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        
        self.client = OpenAI(api_key=self.api_key)

        self.role_prompt = f"""You are a labeling agent. Your responses are meant to be one or a few words long (i.e. a phrase).
                            Be direct and concise and get to the point; minimize tokens. Don’t elaborate unless requested. 
                            Don’t be redundant or repetitive. Don’t provide “recaps” that just duplicate what you already said. """


    def find_center(self, embeddings: np.ndarray) -> np.ndarray:
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
        #print(f"Center embedding: {center_embedding}")
        #print(f"Closest index: {closest_index}")
        center_image_path = self.image_paths_dict[tuple(center_embedding)]
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
        new_keys = []
        print(len(self.clusters.keys()))
        for cluster_id, embeddings in self.clusters.items():
            # Find the closest image to the centroid of the cluster
            center_embedding, center_image_path = self.find_center(embeddings)
            
            # Convert the image to a format suitable for the OpenAI API
            image = Image.open(center_image_path)
            image = image.convert("RGB")

            # Save the image to a temporary file
            temp_image_path = f"temp_image_{cluster_id}.png"
            base64_image = base64.b64encode(open(temp_image_path, "rb").read()).decode("utf-8")

            prompt = f"""Give a brief label or short phrase (with a maximum of three words) that categorizes/describes 
                        the attached image. Think extremely general categories. For instance, if the image is of a dog,
                        you could say "pet" or "animal". If the image is of a car, you could say "vehicle" or "transportation"."""
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{
                    "role": "system",
                    "content": self.role_prompt
                }, {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url", 
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }, 
                        },
                    ]
                }],
            )
            label = response.choices[0].message.content.strip().lower()
            labeled_clusters[label] = embeddings
            new_keys.append(label)
        
        return labeled_clusters, new_keys
