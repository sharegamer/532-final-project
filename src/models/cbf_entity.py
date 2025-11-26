"""
Content-Based Filtering using Entity embeddings.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd


class EntityRecommender:
    """Content-based recommender using entity embeddings."""

    def __init__(self, news_entity_features, news_ids):
        """
        Initialize entity-based recommender.

        Args:
            news_entity_features: numpy array of entity feature vectors (n_news, embedding_dim)
            news_ids: List/array of news IDs corresponding to rows
        """
        self.entity_features = np.array(news_entity_features)
        self.news_ids = np.array(news_ids)
        self.news_id_to_idx = {nid: idx for idx, nid in enumerate(news_ids)}
        self.similarity_matrix = None

    def compute_similarity_matrix(self):
        """
        Precompute pairwise similarity matrix for all news articles.
        """
        self.similarity_matrix = cosine_similarity(self.entity_features)
        return self.similarity_matrix

    def get_similar_news(self, news_id, top_k=10):
        """
        Get most similar news articles based on entity embeddings.

        Args:
            news_id: Target news ID
            top_k: Number of similar articles to return

        Returns:
            List of (news_id, similarity_score) tuples
        """
        if news_id not in self.news_id_to_idx:
            return []

        idx = self.news_id_to_idx[news_id]

        # Compute similarity
        if self.similarity_matrix is None:
            news_vector = self.entity_features[idx:idx+1]
            similarities = cosine_similarity(news_vector, self.entity_features)[0]
        else:
            similarities = self.similarity_matrix[idx]

        # Get top-k most similar (excluding itself)
        top_indices = np.argsort(similarities)[::-1][1:top_k+1]

        results = [(self.news_ids[i], similarities[i]) for i in top_indices]

        return results

    def recommend_for_user(self, user_history, candidate_news, top_k=10):
        """
        Recommend news for a user based on entity similarity.

        Args:
            user_history: List of news IDs the user has read
            candidate_news: List of candidate news IDs to rank
            top_k: Number of recommendations to return

        Returns:
            List of (news_id, score) tuples
        """
        if len(user_history) == 0:
            return []

        # Filter valid history items
        valid_history = [nid for nid in user_history if nid in self.news_id_to_idx]

        if len(valid_history) == 0:
            return []

        # Get indices
        history_indices = [self.news_id_to_idx[nid] for nid in valid_history]

        # Compute user profile as average of history entity vectors
        user_profile = self.entity_features[history_indices].mean(axis=0).reshape(1, -1)

        # Get candidate indices
        valid_candidates = [nid for nid in candidate_news if nid in self.news_id_to_idx]
        candidate_indices = [self.news_id_to_idx[nid] for nid in valid_candidates]

        if len(candidate_indices) == 0:
            return []

        # Compute similarity scores
        candidate_vectors = self.entity_features[candidate_indices]
        scores = cosine_similarity(user_profile, candidate_vectors)[0]

        # Rank and return top-k
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = [(valid_candidates[i], scores[i]) for i in top_indices]

        return results

    def batch_recommend(self, user_histories, candidate_news_list, top_k=10):
        """
        Generate recommendations for multiple users.

        Args:
            user_histories: Dictionary mapping user_id to list of news IDs
            candidate_news_list: Dictionary mapping user_id to candidate news IDs
            top_k: Number of recommendations per user

        Returns:
            Dictionary mapping user_id to list of (news_id, score) tuples
        """
        recommendations = {}

        for user_id in user_histories:
            history = user_histories[user_id]
            candidates = candidate_news_list.get(user_id, [])

            if len(candidates) > 0:
                recs = self.recommend_for_user(history, candidates, top_k)
                recommendations[user_id] = recs

        return recommendations

    def save_model(self, path):
        """
        Save model to disk.

        Args:
            path: Directory path to save model
        """
        import pickle
        import os

        os.makedirs(path, exist_ok=True)

        with open(f'{path}/entity_recommender.pkl', 'wb') as f:
            pickle.dump({
                'entity_features': self.entity_features,
                'news_ids': self.news_ids,
                'news_id_to_idx': self.news_id_to_idx,
                'similarity_matrix': self.similarity_matrix
            }, f)

    @staticmethod
    def load_model(path):
        """
        Load model from disk.

        Args:
            path: Directory path containing saved model

        Returns:
            EntityRecommender instance
        """
        import pickle

        with open(f'{path}/entity_recommender.pkl', 'rb') as f:
            data = pickle.load(f)

        recommender = EntityRecommender(data['entity_features'], data['news_ids'])
        recommender.similarity_matrix = data['similarity_matrix']

        return recommender
