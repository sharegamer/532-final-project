"""
Knowledge-Aware recommendation using entity and relation embeddings.
Implements knowledge graph-based scoring mechanism.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import json


class KnowledgeAwareRecommender:
    """Knowledge graph-based recommender using entity and relation embeddings."""

    def __init__(self, news_df, entity_embeddings, relation_embeddings):
        """
        Initialize knowledge-aware recommender.

        Args:
            news_df: DataFrame with news data including entity information
            entity_embeddings: Dictionary mapping entity IDs to embedding vectors
            relation_embeddings: Dictionary mapping relation IDs to embedding vectors
        """
        self.news_df = news_df.copy()
        self.entity_embeddings = entity_embeddings
        self.relation_embeddings = relation_embeddings
        self.news_entity_graph = {}
        self.news_knowledge_vectors = {}

        # Build knowledge graph representation
        self._build_knowledge_graph()

    def _parse_entities(self, entity_str):
        """Parse entity JSON string."""
        if pd.isna(entity_str) or entity_str == '[]':
            return []

        try:
            entities = json.loads(entity_str)
            return [e.get('WikidataId', '') for e in entities if 'WikidataId' in e]
        except:
            return []

    def _build_knowledge_graph(self):
        """
        Build knowledge graph representation for each news article.
        Combines entity embeddings with structural information.
        """
        for idx, row in self.news_df.iterrows():
            news_id = row['news_id']

            # Parse entities from title and abstract
            title_entities = self._parse_entities(row.get('title_entities', '[]'))
            abstract_entities = self._parse_entities(row.get('abstract_entities', '[]'))

            all_entities = list(set(title_entities + abstract_entities))

            # Store entity graph
            self.news_entity_graph[news_id] = all_entities

            # Compute knowledge-aware representation
            if len(all_entities) > 0:
                # Get entity embeddings
                entity_vecs = [self.entity_embeddings[e] for e in all_entities
                              if e in self.entity_embeddings]

                if len(entity_vecs) > 0:
                    # Average entity embeddings as knowledge vector
                    knowledge_vec = np.mean(entity_vecs, axis=0)
                else:
                    # Zero vector if no valid entities
                    embedding_dim = len(next(iter(self.entity_embeddings.values())))
                    knowledge_vec = np.zeros(embedding_dim)
            else:
                embedding_dim = len(next(iter(self.entity_embeddings.values())))
                knowledge_vec = np.zeros(embedding_dim)

            self.news_knowledge_vectors[news_id] = knowledge_vec

    def compute_knowledge_similarity(self, news_id1, news_id2):
        """
        Compute knowledge-aware similarity between two news articles.

        Args:
            news_id1: First news ID
            news_id2: Second news ID

        Returns:
            Similarity score
        """
        if news_id1 not in self.news_knowledge_vectors or news_id2 not in self.news_knowledge_vectors:
            return 0.0

        vec1 = self.news_knowledge_vectors[news_id1].reshape(1, -1)
        vec2 = self.news_knowledge_vectors[news_id2].reshape(1, -1)

        similarity = cosine_similarity(vec1, vec2)[0][0]

        # Boost similarity if entities overlap
        entities1 = set(self.news_entity_graph.get(news_id1, []))
        entities2 = set(self.news_entity_graph.get(news_id2, []))

        if len(entities1) > 0 and len(entities2) > 0:
            overlap = len(entities1 & entities2) / len(entities1 | entities2)
            similarity = similarity * 0.7 + overlap * 0.3

        return similarity

    def recommend_for_user(self, user_history, candidate_news, top_k=10):
        """
        Recommend news using knowledge-aware scoring.

        Args:
            user_history: List of news IDs the user has read
            candidate_news: List of candidate news IDs to rank
            top_k: Number of recommendations

        Returns:
            List of (news_id, score) tuples
        """
        if len(user_history) == 0:
            return []

        # Filter valid history
        valid_history = [nid for nid in user_history if nid in self.news_knowledge_vectors]

        if len(valid_history) == 0:
            return []

        # Compute user knowledge profile
        history_vecs = [self.news_knowledge_vectors[nid] for nid in valid_history]
        user_profile = np.mean(history_vecs, axis=0).reshape(1, -1)

        # Score candidates
        scores = []
        for news_id in candidate_news:
            if news_id in self.news_knowledge_vectors:
                candidate_vec = self.news_knowledge_vectors[news_id].reshape(1, -1)

                # Base similarity
                base_score = cosine_similarity(user_profile, candidate_vec)[0][0]

                # Entity overlap bonus
                candidate_entities = set(self.news_entity_graph.get(news_id, []))
                user_entities = set()
                for hist_id in valid_history:
                    user_entities.update(self.news_entity_graph.get(hist_id, []))

                if len(candidate_entities) > 0 and len(user_entities) > 0:
                    overlap = len(candidate_entities & user_entities) / len(candidate_entities | user_entities)
                    final_score = base_score * 0.7 + overlap * 0.3
                else:
                    final_score = base_score

                scores.append((news_id, final_score))

        # Sort and return top-k
        scores.sort(key=lambda x: x[1], reverse=True)

        return scores[:top_k]

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

    def get_entity_importance(self, news_id):
        """
        Get importance scores for entities in a news article.

        Args:
            news_id: News ID

        Returns:
            Dictionary mapping entity IDs to importance scores
        """
        if news_id not in self.news_entity_graph:
            return {}

        entities = self.news_entity_graph[news_id]
        importance_scores = {}

        # Simple importance based on embedding magnitude
        for entity in entities:
            if entity in self.entity_embeddings:
                magnitude = np.linalg.norm(self.entity_embeddings[entity])
                importance_scores[entity] = magnitude

        return importance_scores

    def save_model(self, path):
        """
        Save model to disk.

        Args:
            path: Directory path to save model
        """
        import pickle
        import os

        os.makedirs(path, exist_ok=True)

        with open(f'{path}/knowledge_aware_recommender.pkl', 'wb') as f:
            pickle.dump({
                'news_entity_graph': self.news_entity_graph,
                'news_knowledge_vectors': self.news_knowledge_vectors,
                'entity_embeddings': self.entity_embeddings,
                'relation_embeddings': self.relation_embeddings
            }, f)

    @staticmethod
    def load_model(path, news_df):
        """
        Load model from disk.

        Args:
            path: Directory path containing saved model
            news_df: News DataFrame (needed for reconstruction)

        Returns:
            KnowledgeAwareRecommender instance
        """
        import pickle

        with open(f'{path}/knowledge_aware_recommender.pkl', 'rb') as f:
            data = pickle.load(f)

        # Create instance with loaded data
        recommender = KnowledgeAwareRecommender.__new__(KnowledgeAwareRecommender)
        recommender.news_df = news_df
        recommender.entity_embeddings = data['entity_embeddings']
        recommender.relation_embeddings = data['relation_embeddings']
        recommender.news_entity_graph = data['news_entity_graph']
        recommender.news_knowledge_vectors = data['news_knowledge_vectors']

        return recommender
