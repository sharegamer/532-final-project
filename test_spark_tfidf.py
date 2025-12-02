"""
Test script to verify Spark TF-IDF implementation works correctly.
"""

import sys
sys.path.append('./src')

from pyspark.sql import SparkSession
from data_loader import MINDDataLoader
from preprocessing import FeatureExtractor
from models.cbf_tfidf import TFIDFRecommender
import numpy as np

print("="*60)
print("Testing Spark TF-IDF Implementation")
print("="*60)

# Initialize Spark
print("\n1. Initializing Spark...")
spark = SparkSession.builder \
    .appName("Test_Spark_TFIDF") \
    .config("spark.driver.memory", "2g") \
    .getOrCreate()
print("✅ Spark initialized")

# Load sample news data
print("\n2. Loading news data...")
data_loader = MINDDataLoader(spark_session=spark, use_spark=True)
news_spark = data_loader.load_news_spark('data/MINDsmall_train/news.tsv')
news_spark.cache()
news_count = news_spark.count()
print(f"✅ Loaded {news_count:,} news articles")

# Extract TF-IDF using Spark
print("\n3. Extracting TF-IDF features with Spark...")
feature_extractor = FeatureExtractor(use_spark=True)
tfidf_matrix, news_ids, tfidf_model = feature_extractor.extract_tfidf_spark(
    news_spark,
    num_features=5000
)
print(f"✅ TF-IDF matrix shape: {tfidf_matrix.shape}")
print(f"✅ Matrix type: {type(tfidf_matrix)}")
print(f"✅ Number of news IDs: {len(news_ids)}")
print(f"✅ Sparsity: {1 - (tfidf_matrix.nnz / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1])):.2%}")

# Verify dimensions match
assert tfidf_matrix.shape[0] == len(news_ids), "Mismatch between matrix rows and news IDs"
assert tfidf_matrix.shape[0] == news_count, "Mismatch between matrix rows and news count"
print("\n✅ Dimension checks passed")

# Test TFIDFRecommender
print("\n4. Testing TFIDFRecommender...")
recommender = TFIDFRecommender(tfidf_matrix, news_ids)

# Test similarity computation
sample_news_id = news_ids[0]
similar_news = recommender.get_similar_news(sample_news_id, top_k=5)
print(f"✅ Similar news to {sample_news_id}:")
for news_id, score in similar_news:
    print(f"   - {news_id}: {score:.4f}")

# Test user recommendations
print("\n5. Testing user recommendations...")
sample_history = news_ids[:5].tolist()
candidate_news = news_ids[10:30].tolist()
recommendations = recommender.recommend_for_user(sample_history, candidate_news, top_k=5)
print(f"✅ Top 5 recommendations for user with history {sample_history[:3]}:")
for news_id, score in recommendations:
    print(f"   - {news_id}: {score:.4f}")

print("\n" + "="*60)
print("✅ All tests passed!")
print("="*60)

# Cleanup
spark.stop()
