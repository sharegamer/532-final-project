"""
Evaluation metrics for recommendation systems.
Includes ranking metrics (NDCG, MRR), classification metrics (AUC, Accuracy), and diversity metrics.
"""

import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from typing import List, Dict, Tuple
import pandas as pd


class RecommendationEvaluator:
    """Comprehensive evaluation for recommendation systems."""

    def __init__(self):
        """Initialize evaluator."""
        pass

    @staticmethod
    def dcg_at_k(relevances: List[float], k: int) -> float:
        """
        Calculate Discounted Cumulative Gain at k.

        Args:
            relevances: List of relevance scores (1 for relevant, 0 for not)
            k: Cutoff position

        Returns:
            DCG@k score
        """
        relevances = np.array(relevances[:k])
        if len(relevances) == 0:
            return 0.0

        # DCG = sum(rel_i / log2(i + 1)) for i in 1..k
        discounts = np.log2(np.arange(2, len(relevances) + 2))
        return np.sum(relevances / discounts)

    @staticmethod
    def ndcg_at_k(relevances: List[float], k: int) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain at k.

        Args:
            relevances: List of relevance scores in predicted order
            k: Cutoff position

        Returns:
            NDCG@k score
        """
        dcg = RecommendationEvaluator.dcg_at_k(relevances, k)

        # Ideal DCG (sort by relevance descending)
        ideal_relevances = sorted(relevances, reverse=True)
        idcg = RecommendationEvaluator.dcg_at_k(ideal_relevances, k)

        if idcg == 0:
            return 0.0

        return dcg / idcg

    @staticmethod
    def mrr(relevances: List[float]) -> float:
        """
        Calculate Mean Reciprocal Rank.

        Args:
            relevances: List of relevance scores (1 for relevant, 0 for not)

        Returns:
            MRR score
        """
        for i, rel in enumerate(relevances):
            if rel > 0:
                return 1.0 / (i + 1)
        return 0.0

    @staticmethod
    def precision_at_k(relevances: List[float], k: int) -> float:
        """
        Calculate Precision@k.

        Args:
            relevances: List of relevance scores
            k: Cutoff position

        Returns:
            Precision@k score
        """
        if k == 0:
            return 0.0

        relevances = relevances[:k]
        return sum(relevances) / k

    @staticmethod
    def recall_at_k(relevances: List[float], total_relevant: int, k: int) -> float:
        """
        Calculate Recall@k.

        Args:
            relevances: List of relevance scores
            total_relevant: Total number of relevant items
            k: Cutoff position

        Returns:
            Recall@k score
        """
        if total_relevant == 0:
            return 0.0

        relevances = relevances[:k]
        return sum(relevances) / total_relevant

    def evaluate_ranking(self, predictions: List[Tuple[str, float]], ground_truth: Dict[str, int], k_values=[5, 10]) -> Dict:
        """
        Evaluate ranking performance.

        Args:
            predictions: List of (item_id, score) tuples in ranked order
            ground_truth: Dictionary mapping item_id to relevance (0 or 1)
            k_values: List of k values to evaluate

        Returns:
            Dictionary with evaluation metrics
        """
        # Extract relevances in predicted order
        relevances = [ground_truth.get(item_id, 0) for item_id, _ in predictions]
        total_relevant = sum(ground_truth.values())

        results = {}

        # Calculate metrics for each k
        for k in k_values:
            results[f'NDCG@{k}'] = self.ndcg_at_k(relevances, k)
            results[f'Precision@{k}'] = self.precision_at_k(relevances, k)
            results[f'Recall@{k}'] = self.recall_at_k(relevances, total_relevant, k)

        # MRR (overall)
        results['MRR'] = self.mrr(relevances)

        return results

    def evaluate_classification(self, predictions: List[float], labels: List[int]) -> Dict:
        """
        Evaluate as binary classification.

        Args:
            predictions: Predicted scores
            labels: Ground truth labels (0 or 1)

        Returns:
            Dictionary with classification metrics
        """
        predictions = np.array(predictions)
        labels = np.array(labels)

        # Binary predictions (threshold at 0.5)
        binary_preds = (predictions >= 0.5).astype(int)

        results = {
            'AUC': roc_auc_score(labels, predictions),
            'Accuracy': accuracy_score(labels, binary_preds),
            'Precision': precision_score(labels, binary_preds, zero_division=0),
            'Recall': recall_score(labels, binary_preds, zero_division=0),
            'F1': f1_score(labels, binary_preds, zero_division=0)
        }

        return results

    def evaluate_batch(self, user_predictions: Dict[str, List[Tuple[str, float]]],
                      user_ground_truth: Dict[str, Dict[str, int]], k_values=[5, 10]) -> Dict:
        """
        Evaluate multiple users' predictions.

        Args:
            user_predictions: Dict mapping user_id to list of (item_id, score) tuples
            user_ground_truth: Dict mapping user_id to dict of item_id -> relevance
            k_values: List of k values to evaluate

        Returns:
            Aggregated evaluation metrics
        """
        all_metrics = []

        for user_id in user_predictions:
            if user_id in user_ground_truth:
                metrics = self.evaluate_ranking(
                    user_predictions[user_id],
                    user_ground_truth[user_id],
                    k_values
                )
                all_metrics.append(metrics)

        # Aggregate metrics (mean)
        aggregated = {}
        if len(all_metrics) > 0:
            for key in all_metrics[0].keys():
                values = [m[key] for m in all_metrics]
                aggregated[key] = np.mean(values)
                aggregated[f'{key}_std'] = np.std(values)

        return aggregated

    @staticmethod
    def catalog_coverage(recommendations: List[List[str]], total_items: int) -> float:
        """
        Calculate catalog coverage.

        Args:
            recommendations: List of recommendation lists
            total_items: Total number of items in catalog

        Returns:
            Coverage ratio
        """
        recommended_items = set()
        for rec_list in recommendations:
            recommended_items.update(rec_list)

        return len(recommended_items) / total_items

    @staticmethod
    def diversity(recommendations: List[List[str]]) -> float:
        """
        Calculate diversity (average uniqueness ratio).

        Args:
            recommendations: List of recommendation lists

        Returns:
            Diversity score
        """
        diversity_scores = []

        for rec_list in recommendations:
            if len(rec_list) > 0:
                unique = len(set(rec_list))
                diversity_scores.append(unique / len(rec_list))

        return np.mean(diversity_scores) if len(diversity_scores) > 0 else 0.0

    def create_evaluation_report(self, model_results: Dict[str, Dict], output_path=None) -> pd.DataFrame:
        """
        Create comprehensive evaluation report comparing different models.

        Args:
            model_results: Dictionary mapping model_name to evaluation metrics
            output_path: Optional path to save report as CSV

        Returns:
            DataFrame with comparison
        """
        report_df = pd.DataFrame(model_results).T

        # Sort columns
        metric_order = ['NDCG@5', 'NDCG@10', 'MRR', 'Precision@5', 'Precision@10',
                       'Recall@5', 'Recall@10', 'AUC', 'Accuracy']
        existing_metrics = [m for m in metric_order if m in report_df.columns]

        report_df = report_df[existing_metrics]

        if output_path:
            report_df.to_csv(output_path)

        return report_df

    def print_evaluation_summary(self, metrics: Dict, model_name='Model'):
        """
        Print formatted evaluation summary.

        Args:
            metrics: Dictionary of evaluation metrics
            model_name: Name of the model being evaluated
        """
        print(f"\n{'='*60}")
        print(f"Evaluation Results: {model_name}")
        print(f"{'='*60}")

        # Ranking metrics
        ranking_metrics = {k: v for k, v in metrics.items() if any(x in k for x in ['NDCG', 'MRR', 'Precision', 'Recall'])}
        if ranking_metrics:
            print("\nRanking Metrics:")
            for metric, value in ranking_metrics.items():
                print(f"  {metric:20s}: {value:.4f}")

        # Classification metrics
        classification_metrics = {k: v for k, v in metrics.items() if any(x in k for x in ['AUC', 'Accuracy', 'F1'])}
        if classification_metrics:
            print("\nClassification Metrics:")
            for metric, value in classification_metrics.items():
                print(f"  {metric:20s}: {value:.4f}")

        print(f"{'='*60}\n")
