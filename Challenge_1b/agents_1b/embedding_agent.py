# agents_1b/embedding_agent.py

from sentence_transformers import SentenceTransformer

class EmbeddingAgent:
    """
    Handles loading the Sentence Transformer model and creating embeddings.
    """
    def __init__(self, model_name: str):
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        print("Model loaded successfully.")

    def create_embeddings(self, texts: list) -> list:
        """Encodes a list of text chunks."""
        return self.model.encode(texts, convert_to_tensor=True, show_progress_bar=True)

    def create_query_embedding(self, query: str):
        """Encodes a single query string."""
        return self.model.encode(query, convert_to_tensor=True)