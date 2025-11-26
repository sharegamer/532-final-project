"""
Hybrid recommendation model combining CBF (TF-IDF + Entity), CF (ALS), and Knowledge-Aware approaches.
"""

import numpy as np
from typing import Dict, List, Tuple


class HybridRecommender:
    """Hybrid recommender combining multiple recommendation strategies."""

    def __init__(self, tfidf_recommender, entity_recommender, als_recommender, knowledge_recommender):
        """
        Initialize hybrid recommender.

        Args:
            tfidf_recommender: TFIDFRecommender instance
            entity_recommender: EntityRecommender instance
            als_recommender: ALSRecommender instance
            knowledge_recommender: KnowledgeAwareRecommender instance
        """
        self.tfidf_rec = tfidf_recommender
        self.entity_rec = entity_recommender
        self.als_rec = als_recommender
        self.knowledge_rec = knowledge_recommender

        # Default weights (can be tuned)
        self.weights = {
            'tfidf': 0.25,
            'entity': 0.25,
            'als': 0.3,
            'knowledge': 0.2
        }

    def set_weights(self, tfidf=0.25, entity=0.25, als=0.3, knowledge=0.2):
        """
        Set weights for different recommendation components.

        Args:
            tfidf: Weight for TF-IDF recommender
            entity: Weight for entity recommender
            als: Weight for ALS recommender
            knowledge: Weight for knowledge-aware recommender
        """
        total = tfidf + entity + als + knowledge
        self.weights = {
            'tfidf': tfidf / total,
            'entity': entity / total,
            'als': als / total,
            'knowledge': knowledge / total
        }

    def _normalize_scores(self, recommendations: List[Tuple[str, float]]) -> Dict[str, float]:
        """
        Normalize recommendation scores to [0, 1] range.

        Args:
            recommendations: List of (news_id, score) tuples

        Returns:
            Dictionary mapping news_id to normalized score
        """
        if len(recommendations) == 0:
            return {}

        scores = [score for _, score in recommendations]
        min_score = min(scores)
        max_score = max(scores)

        if max_score == min_score:
            return {news_id: 1.0 for news_id, _ in recommendations}

        normalized = {}
        for news_id, score in recommendations:
            normalized[news_id] = (score - min_score) / (max_score - min_score)

        return normalized

    def recommend_for_user(self, user_id, user_history, candidate_news, top_k=10):
        """
        Generate hybrid recommendations for a user.

        Args:
            user_id: User ID
            user_history: List of news IDs the user has read
            candidate_news: List of candidate news IDs to rank
            top_k: Number of recommendations

        Returns:
            List of (news_id, score, component_scores) tuples
        """
        # Get recommendations from each component
        tfidf_recs = self.tfidf_rec.recommend_for_user(user_history, candidate_news, top_k=len(candidate_news))
        entity_recs = self.entity_rec.recommend_for_user(user_history, candidate_news, top_k=len(candidate_news))
        knowledge_recs = self.knowledge_rec.recommend_for_user(user_history, candidate_news, top_k=len(candidate_news))

        # ALS recommendations (if user exists in model)
        try:
            als_recs = self.als_rec.recommend_for_user(user_id, top_k=len(candidate_news))
            # Filter to only candidates
            als_recs = [(nid, score) for nid, score in als_recs if nid in candidate_news]
        except:
            als_recs = []

        # Normalize scores
        tfidf_scores = self._normalize_scores(tfidf_recs)
        entity_scores = self._normalize_scores(entity_recs)
        als_scores = self._normalize_scores(als_recs)
        knowledge_scores = self._normalize_scores(knowledge_recs)

        # Combine scores
        combined_scores = {}
        component_scores_dict = {}

        for news_id in candidate_news:
            score = 0.0
            components = {}

            if news_id in tfidf_scores:
                components['tfidf'] = tfidf_scores[news_id]
                score += self.weights['tfidf'] * tfidf_scores[news_id]

            if news_id in entity_scores:
                components['entity'] = entity_scores[news_id]
                score += self.weights['entity'] * entity_scores[news_id]

            if news_id in als_scores:
                components['als'] = als_scores[news_id]
                score += self.weights['als'] * als_scores[news_id]

            if news_id in knowledge_scores:
                components['knowledge'] = knowledge_scores[news_id]
                score += self.weights['knowledge'] * knowledge_scores[news_id]

            if len(components) > 0:
                combined_scores[news_id] = score
                component_scores_dict[news_id] = components

        # Sort by combined score
        sorted_items = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)

        # Return top-k with component scores
        results = []
        for news_id, score in sorted_items[:top_k]:
            results.append((news_id, score, component_scores_dict[news_id]))

        return results

    def batch_recommend(self, user_data, top_k=10):
        """
        Generate recommendations for multiple users.

        Args:
            user_data: Dictionary mapping user_id to dict with 'history' and 'candidates'
            top_k: Number of recommendations per user

        Returns:
            Dictionary mapping user_id to list of recommendations
        """
        recommendations = {}

        for user_id, data in user_data.items():
            history = data['history']
            candidates = data['candidates']

            recs = self.recommend_for_user(user_id, history, candidates, top_k)
            recommendations[user_id] = recs

        return recommendations

    def get_component_contributions(self, user_id, user_history, candidate_news, top_k=10):
        """
        Analyze contribution of each component to final recommendations.

        Args:
            user_id: User ID
            user_history: User reading history
            candidate_news: Candidate news items
            top_k: Number of recommendations

        Returns:
            Dictionary with analysis of component contributions
        """
        recs = self.recommend_for_user(user_id, user_history, candidate_news, top_k)

        analysis = {
            'recommendations': recs,
            'avg_contributions': {
                'tfidf': 0.0,
                'entity': 0.0,
                'als': 0.0,
                'knowledge': 0.0
            },
            'component_counts': {
                'tfidf': 0,
                'entity': 0,
                'als': 0,
                'knowledge': 0
            }
        }

        # Calculate average contributions
        for news_id, score, components in recs:
            for comp_name in components:
                analysis['avg_contributions'][comp_name] += components[comp_name]
                analysis['component_counts'][comp_name] += 1

        # Average
        for comp_name in analysis['avg_contributions']:
            if analysis['component_counts'][comp_name] > 0:
                analysis['avg_contributions'][comp_name] /= analysis['component_counts'][comp_name]

        return analysis

    def save_model(self, path):
        """
        Save hybrid model configuration.

        Args:
            path: Directory path to save model
        """
        import pickle
        import os

        os.makedirs(path, exist_ok=True)

        with open(f'{path}/hybrid_weights.pkl', 'wb') as f:
            pickle.dump({'weights': self.weights}, f)

        # Individual models should be saved separately

    def load_weights(self, path):
        """
        Load hybrid model weights.

        Args:
            path: Directory path containing saved weights
        """
        import pickle

        with open(f'{path}/hybrid_weights.pkl', 'rb') as f:
            data = pickle.load(f)
            self.weights = data['weights']

        return self
