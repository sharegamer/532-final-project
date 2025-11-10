"""
Collaborative Filtering using ALS (Alternating Least Squares) with Spark MLlib.
"""

from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col
import numpy as np
import pandas as pd


class ALSRecommender:
    """Collaborative filtering recommender using Spark ALS."""

    def __init__(self, spark_session):
        """
        Initialize ALS recommender.

        Args:
            spark_session: Active Spark session
        """
        self.spark = spark_session
        self.model = None
        self.user_mapping = None
        self.news_mapping = None
        self.reverse_user_mapping = None
        self.reverse_news_mapping = None

    def prepare_data(self, interactions_df, train_ratio=0.8):
        """
        Prepare data for ALS training.

        Args:
            interactions_df: Spark DataFrame with (user_id, news_id, label) columns
            train_ratio: Ratio of data to use for training

        Returns:
            train_df, test_df
        """
        # Convert string IDs to integer indices for ALS
        from pyspark.ml.feature import StringIndexer

        # Index users
        user_indexer = StringIndexer(inputCol='user_id', outputCol='user_idx')
        user_indexer_model = user_indexer.fit(interactions_df)
        interactions_df = user_indexer_model.transform(interactions_df)

        # Index news
        news_indexer = StringIndexer(inputCol='news_id', outputCol='news_idx')
        news_indexer_model = news_indexer.fit(interactions_df)
        interactions_df = news_indexer_model.transform(interactions_df)

        # Store mappings for later use
        self.user_mapping = {row['user_id']: int(row['user_idx'])
                            for row in interactions_df.select('user_id', 'user_idx').distinct().collect()}
        self.news_mapping = {row['news_id']: int(row['news_idx'])
                            for row in interactions_df.select('news_id', 'news_idx').distinct().collect()}

        self.reverse_user_mapping = {v: k for k, v in self.user_mapping.items()}
        self.reverse_news_mapping = {v: k for k, v in self.news_mapping.items()}

        # Convert label to rating (0 -> 0, 1 -> 1, can be scaled)
        interactions_df = interactions_df.withColumn('rating', col('label').cast('float'))

        # Split data
        train_df, test_df = interactions_df.randomSplit([train_ratio, 1 - train_ratio], seed=42)

        return train_df, test_df

    def train(self, train_df, rank=10, max_iter=10, reg_param=0.1, cold_start_strategy='drop'):
        """
        Train ALS model.

        Args:
            train_df: Training data with (user_idx, news_idx, rating)
            rank: Number of latent factors
            max_iter: Maximum iterations
            reg_param: Regularization parameter
            cold_start_strategy: Strategy for cold start ('drop' or 'nan')

        Returns:
            Trained ALS model
        """
        als = ALS(
            rank=rank,
            maxIter=max_iter,
            regParam=reg_param,
            userCol='user_idx',
            itemCol='news_idx',
            ratingCol='rating',
            coldStartStrategy=cold_start_strategy,
            implicitPrefs=False  # We have explicit feedback (0/1 labels)
        )

        self.model = als.fit(train_df)

        return self.model

    def evaluate(self, test_df, metric='rmse'):
        """
        Evaluate model on test data.

        Args:
            test_df: Test data
            metric: Evaluation metric ('rmse', 'mse', 'mae')

        Returns:
            Evaluation score
        """
        predictions = self.model.transform(test_df)

        evaluator = RegressionEvaluator(
            metricName=metric,
            labelCol='rating',
            predictionCol='prediction'
        )

        score = evaluator.evaluate(predictions)

        return score

    def recommend_for_user(self, user_id, top_k=10):
        """
        Generate top-K recommendations for a user.

        Args:
            user_id: User ID (original string ID)
            top_k: Number of recommendations

        Returns:
            List of (news_id, score) tuples
        """
        if user_id not in self.user_mapping:
            return []

        user_idx = self.user_mapping[user_id]

        # Get recommendations from model
        user_df = self.spark.createDataFrame([(user_idx,)], ['user_idx'])
        recommendations = self.model.recommendForUserSubset(user_df, top_k)

        # Extract results
        results = []
        for row in recommendations.collect():
            for rec in row['recommendations']:
                news_idx = rec['news_idx']
                score = rec['rating']
                news_id = self.reverse_news_mapping.get(news_idx, None)
                if news_id:
                    results.append((news_id, float(score)))

        return results

    def recommend_for_users(self, user_ids, top_k=10):
        """
        Generate recommendations for multiple users.

        Args:
            user_ids: List of user IDs
            top_k: Number of recommendations per user

        Returns:
            Dictionary mapping user_id to list of (news_id, score) tuples
        """
        # Filter valid users
        valid_users = [(self.user_mapping[uid],) for uid in user_ids if uid in self.user_mapping]

        if len(valid_users) == 0:
            return {}

        # Create DataFrame
        users_df = self.spark.createDataFrame(valid_users, ['user_idx'])

        # Get recommendations
        recommendations = self.model.recommendForUserSubset(users_df, top_k)

        # Parse results
        results = {}
        for row in recommendations.collect():
            user_idx = row['user_idx']
            user_id = self.reverse_user_mapping[user_idx]
            recs = []

            for rec in row['recommendations']:
                news_idx = rec['news_idx']
                score = rec['rating']
                news_id = self.reverse_news_mapping.get(news_idx, None)
                if news_id:
                    recs.append((news_id, float(score)))

            results[user_id] = recs

        return results

    def get_user_factors(self):
        """
        Get learned user latent factors.

        Returns:
            Spark DataFrame with user factors
        """
        return self.model.userFactors

    def get_item_factors(self):
        """
        Get learned item (news) latent factors.

        Returns:
            Spark DataFrame with item factors
        """
        return self.model.itemFactors

    def save_model(self, path):
        """
        Save ALS model to disk.

        Args:
            path: Directory path to save model
        """
        import pickle
        import os

        os.makedirs(path, exist_ok=True)

        # Save Spark ALS model
        self.model.write().overwrite().save(f'{path}/als_model')

        # Save mappings
        with open(f'{path}/als_mappings.pkl', 'wb') as f:
            pickle.dump({
                'user_mapping': self.user_mapping,
                'news_mapping': self.news_mapping,
                'reverse_user_mapping': self.reverse_user_mapping,
                'reverse_news_mapping': self.reverse_news_mapping
            }, f)

    def load_model(self, path):
        """
        Load ALS model from disk.

        Args:
            path: Directory path containing saved model
        """
        import pickle
        from pyspark.ml.recommendation import ALSModel

        # Load Spark ALS model
        self.model = ALSModel.load(f'{path}/als_model')

        # Load mappings
        with open(f'{path}/als_mappings.pkl', 'rb') as f:
            data = pickle.load(f)
            self.user_mapping = data['user_mapping']
            self.news_mapping = data['news_mapping']
            self.reverse_user_mapping = data['reverse_user_mapping']
            self.reverse_news_mapping = data['reverse_news_mapping']

        return self
