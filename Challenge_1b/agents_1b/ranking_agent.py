# agents_1b/ranking_agent.py

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class RankingAgent:
    """
    Calculates relevance scores between a query and document chunks.
    """
    def calculate_relevance_scores(self, query_embedding, chunk_embeddings) -> np.ndarray:
        """
        Calculates cosine similarity scores.
        """
        # Reshape query embedding to work with the function
        query_embedding_reshaped = query_embedding.reshape(1, -1)
        
        # Calculate and return the similarity scores
        return cosine_similarity(
            query_embedding_reshaped.cpu().numpy(),
            chunk_embeddings.cpu().numpy()
        )[0] # We take the first element because we only have one query