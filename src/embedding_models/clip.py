import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from typing import List, Dict, Any
from tqdm import tqdm

class CLIPEmbeddingModel:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch16"):
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def embed_text(self, texts: List[str], batch_size: int = 32) -> torch.Tensor:
        txt_embeddings = []

        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding texts"):
            batch = texts[i:i + batch_size]
            inputs = self.processor(text=batch, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                outputs = self.model.get_text_features(**inputs)
            txt_embeddings.append(outputs)

        return torch.cat(txt_embeddings, dim=0)

    def embed_images(self, images: List[Any], batch_size: int = 32) -> torch.Tensor:
        img_embeddings = []
        
        for i in tqdm(range(0, len(images), batch_size), desc="Embedding images"):
            batch = images[i:i + batch_size]
            inputs = self.processor(images=batch, return_tensors="pt", padding=True)
            with torch.no_grad():
                outputs = self.model.get_image_features(**inputs)
            img_embeddings.append(outputs)
            
        return torch.cat(img_embeddings, dim=0)

    
    def embed(self, data: List[Dict[str, Any]], batch_size: int = 32) -> torch.Tensor:
        """
        Embed a list of dictionaries containing either 'text' or 'image' keys.

        Args:
            data (List[Dict[str, Any]]): List of dictionaries with 'text' or 'image' keys.

        Returns:
            torch.Tensor: Embedded representations.
        """
        texts = [item['text'] for item in data if 'text' in item]
        images = [item['image'] for item in data if 'image' in item]

        text_embeddings = self.embed_text(texts, batch_size=batch_size) if texts else None
        image_embeddings = self.embed_images(images, batch_size=batch_size) if images else None

        return text_embeddings, image_embeddings
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embedding space.

        Returns:
            int: Dimension of the embedding space.
        """
        return self.model.config.projection_dim
    
# Example usage
if __name__ == "__main__":
    model = CLIPEmbeddingModel()
    data = []
    
    text_embeddings, image_embeddings = model.embed(data)
    print("Text Embeddings:", text_embeddings)
    print("Image Embeddings:", image_embeddings)
    print("Embedding Dimension:", model.get_embedding_dimension())
