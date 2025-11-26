"""
Test cases for recommendation models.
"""

import unittest
import sys
import os
sys.path.append('./src')

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from models.cbf_tfidf import TFIDFRecommender
from models.cbf_entity import EntityRecommender
from models.hybrid import HybridRecommender
from evaluation import RecommendationEvaluator


class TestTFIDFRecommender(unittest.TestCase):
    """Test TF-IDF based recommender."""

    def setUp(self):
        """Set up test data."""
        # Create simple TF-IDF matrix
        self.tfidf_matrix = csr_matrix(np.array([
            [1.0, 0.5, 0.0],
            [0.5, 1.0, 0.2],
            [0.0, 0.2, 1.0],
            [0.8, 0.3, 0.1]
        ]))
        self.news_ids = ['N1', 'N2', 'N3', 'N4']
        self.recommender = TFIDFRecommender(self.tfidf_matrix, self.news_ids)

    def test_initialization(self):
        """Test recommender initialization."""
        self.assertEqual(len(self.recommender.news_ids), 4)
        self.assertEqual(len(self.recommender.news_id_to_idx), 4)
        self.assertIn('N1', self.recommender.news_id_to_idx)

    def test_get_similar_news(self):
        """Test finding similar news."""
        similar = self.recommender.get_similar_news('N1', top_k=2)

        self.assertEqual(len(similar), 2)
        self.assertTrue(all(isinstance(item, tuple) for item in similar))
        self.assertTrue(all(len(item) == 2 for item in similar))

        # Check that news_id is not in results (excluding itself)
        news_ids = [nid for nid, _ in similar]
        self.assertNotIn('N1', news_ids)

    def test_recommend_for_user(self):
        """Test user recommendations."""
        user_history = ['N1', 'N2']
        candidates = ['N3', 'N4']

        recs = self.recommender.recommend_for_user(user_history, candidates, top_k=2)

        self.assertLessEqual(len(recs), 2)
        self.assertTrue(all(isinstance(item, tuple) for item in recs))

        # Check that recommendations are from candidates
        news_ids = [nid for nid, _ in recs]
        for nid in news_ids:
            self.assertIn(nid, candidates)

    def test_empty_history(self):
        """Test with empty user history."""
        recs = self.recommender.recommend_for_user([], ['N1', 'N2'], top_k=5)
        self.assertEqual(len(recs), 0)

    def test_invalid_news_id(self):
        """Test with invalid news ID."""
        similar = self.recommender.get_similar_news('N_INVALID', top_k=2)
        self.assertEqual(len(similar), 0)


class TestEntityRecommender(unittest.TestCase):
    """Test Entity-based recommender."""

    def setUp(self):
        """Set up test data."""
        # Create simple entity feature matrix
        self.entity_features = np.array([
            [1.0, 0.5, 0.0],
            [0.5, 1.0, 0.2],
            [0.0, 0.2, 1.0],
            [0.8, 0.3, 0.1]
        ])
        self.news_ids = ['N1', 'N2', 'N3', 'N4']
        self.recommender = EntityRecommender(self.entity_features, self.news_ids)

    def test_initialization(self):
        """Test recommender initialization."""
        self.assertEqual(len(self.recommender.news_ids), 4)
        self.assertEqual(self.recommender.entity_features.shape, (4, 3))

    def test_recommend_for_user(self):
        """Test user recommendations."""
        user_history = ['N1']
        candidates = ['N2', 'N3', 'N4']

        recs = self.recommender.recommend_for_user(user_history, candidates, top_k=2)

        self.assertLessEqual(len(recs), 2)
        # Check recommendations are valid
        for news_id, score in recs:
            self.assertIn(news_id, candidates)
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)


class TestRecommendationEvaluator(unittest.TestCase):
    """Test evaluation metrics."""

    def setUp(self):
        """Set up test data."""
        self.evaluator = RecommendationEvaluator()

    def test_dcg_at_k(self):
        """Test DCG calculation."""
        relevances = [1, 0, 1, 0, 1]
        dcg = self.evaluator.dcg_at_k(relevances, k=3)

        self.assertGreater(dcg, 0)
        self.assertIsInstance(dcg, float)

    def test_ndcg_at_k(self):
        """Test NDCG calculation."""
        relevances = [1, 0, 1, 0, 1]
        ndcg = self.evaluator.ndcg_at_k(relevances, k=3)

        self.assertGreaterEqual(ndcg, 0.0)
        self.assertLessEqual(ndcg, 1.0)

    def test_perfect_ndcg(self):
        """Test perfect ranking NDCG."""
        relevances = [1, 1, 1, 0, 0]
        ndcg = self.evaluator.ndcg_at_k(relevances, k=3)

        # Perfect ranking should have NDCG = 1.0
        self.assertAlmostEqual(ndcg, 1.0, places=5)

    def test_mrr(self):
        """Test MRR calculation."""
        # First item relevant
        mrr1 = self.evaluator.mrr([1, 0, 0])
        self.assertEqual(mrr1, 1.0)

        # Second item relevant
        mrr2 = self.evaluator.mrr([0, 1, 0])
        self.assertEqual(mrr2, 0.5)

        # No relevant items
        mrr3 = self.evaluator.mrr([0, 0, 0])
        self.assertEqual(mrr3, 0.0)

    def test_precision_at_k(self):
        """Test Precision@k."""
        relevances = [1, 1, 0, 1, 0]

        p_at_3 = self.evaluator.precision_at_k(relevances, k=3)
        self.assertAlmostEqual(p_at_3, 2/3)

        p_at_5 = self.evaluator.precision_at_k(relevances, k=5)
        self.assertAlmostEqual(p_at_5, 3/5)

    def test_recall_at_k(self):
        """Test Recall@k."""
        relevances = [1, 1, 0, 1, 0]
        total_relevant = 4

        r_at_3 = self.evaluator.recall_at_k(relevances, total_relevant, k=3)
        self.assertAlmostEqual(r_at_3, 2/4)

        r_at_5 = self.evaluator.recall_at_k(relevances, total_relevant, k=5)
        self.assertAlmostEqual(r_at_5, 3/4)

    def test_evaluate_ranking(self):
        """Test full ranking evaluation."""
        predictions = [('N1', 0.9), ('N2', 0.7), ('N3', 0.5), ('N4', 0.3)]
        ground_truth = {'N1': 1, 'N2': 0, 'N3': 1, 'N4': 0}

        results = self.evaluator.evaluate_ranking(predictions, ground_truth, k_values=[2, 4])

        # Check all expected metrics are present
        self.assertIn('NDCG@2', results)
        self.assertIn('NDCG@4', results)
        self.assertIn('Precision@2', results)
        self.assertIn('MRR', results)

        # Check values are in valid range
        for metric, value in results.items():
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)

    def test_catalog_coverage(self):
        """Test catalog coverage metric."""
        recommendations = [
            ['N1', 'N2', 'N3'],
            ['N2', 'N4', 'N5'],
            ['N1', 'N3', 'N6']
        ]
        total_items = 10

        coverage = self.evaluator.catalog_coverage(recommendations, total_items)

        # Should cover 6 unique items out of 10
        self.assertAlmostEqual(coverage, 0.6)

    def test_diversity(self):
        """Test diversity metric."""
        # All unique recommendations
        recs1 = [['N1', 'N2', 'N3'], ['N4', 'N5', 'N6']]
        div1 = self.evaluator.diversity(recs1)
        self.assertEqual(div1, 1.0)

        # Some duplicates
        recs2 = [['N1', 'N1', 'N2'], ['N3', 'N3', 'N3']]
        div2 = self.evaluator.diversity(recs2)
        self.assertLess(div2, 1.0)


class TestDataValidation(unittest.TestCase):
    """Test data validation and edge cases."""

    def test_empty_candidates(self):
        """Test with empty candidate list."""
        tfidf_matrix = csr_matrix(np.random.rand(5, 10))
        news_ids = [f'N{i}' for i in range(5)]
        recommender = TFIDFRecommender(tfidf_matrix, news_ids)

        recs = recommender.recommend_for_user(['N0'], [], top_k=5)
        self.assertEqual(len(recs), 0)

    def test_large_top_k(self):
        """Test with top_k larger than available items."""
        tfidf_matrix = csr_matrix(np.random.rand(5, 10))
        news_ids = [f'N{i}' for i in range(5)]
        recommender = TFIDFRecommender(tfidf_matrix, news_ids)

        candidates = ['N1', 'N2']
        recs = recommender.recommend_for_user(['N0'], candidates, top_k=100)

        # Should return at most len(candidates) items
        self.assertLessEqual(len(recs), len(candidates))


def run_tests():
    """Run all tests."""
    unittest.main(argv=[''], verbosity=2, exit=False)


if __name__ == '__main__':
    run_tests()
