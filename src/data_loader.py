"""
Data loading module using PySpark for efficient processing of MIND dataset.
Handles behaviors.tsv and news.tsv files.
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, split, explode, size, when
from pyspark.sql.types import StructType, StructField, StringType, ArrayType
import pandas as pd
import numpy as np


class MINDDataLoader:
    """Loads and processes MIND dataset using Spark for optimization."""

    def __init__(self, spark_session=None, use_spark=True):
        """
        Initialize data loader.

        Args:
            spark_session: Existing Spark session or None to create new one
            use_spark: Whether to use Spark for processing (for benchmarking)
        """
        self.use_spark = use_spark
        if use_spark and spark_session is None:
            self.spark = SparkSession.builder \
                .appName("MIND_Recommendation") \
                .config("spark.driver.memory", "4g") \
                .config("spark.executor.memory", "4g") \
                .getOrCreate()
        else:
            self.spark = spark_session

    def load_behaviors_spark(self, behaviors_path):
        """
        Load behaviors.tsv using Spark.
        Format: ImpressionID, UserID, Time, History, Impressions

        Args:
            behaviors_path: Path to behaviors.tsv file

        Returns:
            Spark DataFrame with parsed behavior data
        """
        # Read TSV without header
        df = self.spark.read.csv(
            behaviors_path,
            sep='\t',
            header=False,
            inferSchema=False
        )

        # Rename columns
        df = df.toDF('impression_id', 'user_id', 'time', 'history', 'impressions')

        # Parse history (space-separated news IDs)
        df = df.withColumn('history_list',
                          when(col('history').isNotNull(), split(col('history'), ' '))
                          .otherwise(None))

        # Parse impressions (space-separated news-label pairs)
        df = df.withColumn('impressions_list', split(col('impressions'), ' '))

        return df

    def load_behaviors_pandas(self, behaviors_path):
        """
        Load behaviors.tsv using pandas (for comparison).

        Args:
            behaviors_path: Path to behaviors.tsv file

        Returns:
            Pandas DataFrame with parsed behavior data
        """
        df = pd.read_csv(
            behaviors_path,
            sep='\t',
            header=None,
            names=['impression_id', 'user_id', 'time', 'history', 'impressions']
        )

        # Parse history and impressions
        df['history_list'] = df['history'].apply(
            lambda x: x.split(' ') if pd.notna(x) else []
        )
        df['impressions_list'] = df['impressions'].apply(
            lambda x: x.split(' ') if pd.notna(x) else []
        )

        return df

    def load_news_spark(self, news_path):
        """
        Load news.tsv using Spark.
        Format: NewsID, Category, SubCategory, Title, Abstract, URL, TitleEntities, AbstractEntities

        Args:
            news_path: Path to news.tsv file

        Returns:
            Spark DataFrame with news data
        """
        df = self.spark.read.csv(
            news_path,
            sep='\t',
            header=False,
            inferSchema=False
        )

        # Rename columns
        df = df.toDF('news_id', 'category', 'subcategory', 'title',
                     'abstract', 'url', 'title_entities', 'abstract_entities')

        return df

    def load_news_pandas(self, news_path):
        """
        Load news.tsv using pandas.

        Args:
            news_path: Path to news.tsv file

        Returns:
            Pandas DataFrame with news data
        """
        df = pd.read_csv(
            news_path,
            sep='\t',
            header=None,
            names=['news_id', 'category', 'subcategory', 'title',
                   'abstract', 'url', 'title_entities', 'abstract_entities']
        )

        return df

    def load_entity_embeddings(self, entity_path):
        """
        Load entity embeddings from .vec file.

        Args:
            entity_path: Path to entity_embedding.vec file

        Returns:
            Dictionary mapping entity IDs to embedding vectors, and embedding dimension
        """
        embeddings = {}
        dim = None

        with open(entity_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) > 1:
                    entity_id = parts[0]
                    vector = np.array([float(x) for x in parts[1:]])
                    embeddings[entity_id] = vector

                    # Get dimension from first vector
                    if dim is None:
                        dim = len(vector)

        return embeddings, dim

    def load_relation_embeddings(self, relation_path):
        """
        Load relation embeddings from .vec file.

        Args:
            relation_path: Path to relation_embedding.vec file

        Returns:
            Dictionary mapping relation IDs to embedding vectors, and embedding dimension
        """
        embeddings = {}
        dim = None

        with open(relation_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) > 1:
                    relation_id = parts[0]
                    vector = np.array([float(x) for x in parts[1:]])
                    embeddings[relation_id] = vector

                    # Get dimension from first vector
                    if dim is None:
                        dim = len(vector)

        return embeddings, dim

    def create_user_item_matrix_spark(self, behaviors_df):
        """
        Create user-item interaction matrix from behaviors using Spark.

        Args:
            behaviors_df: Spark DataFrame from load_behaviors_spark()

        Returns:
            Spark DataFrame with (user_id, news_id, label) format
        """
        # Explode impressions to get individual user-news-label rows
        from pyspark.sql.functions import regexp_extract

        df = behaviors_df.select('user_id', explode('impressions_list').alias('impression'))

        # Split impression into news_id and label
        df = df.withColumn('news_id', regexp_extract('impression', r'([^-]+)-', 1))
        df = df.withColumn('label', regexp_extract('impression', r'-(\d+)', 1).cast('int'))

        df = df.select('user_id', 'news_id', 'label')

        return df

    def create_user_item_matrix_pandas(self, behaviors_df):
        """
        Create user-item interaction matrix from behaviors using pandas.

        Args:
            behaviors_df: Pandas DataFrame from load_behaviors_pandas()

        Returns:
            Pandas DataFrame with (user_id, news_id, label) format
        """
        rows = []

        for _, row in behaviors_df.iterrows():
            user_id = row['user_id']
            for impression in row['impressions_list']:
                if '-' in impression:
                    news_id, label = impression.rsplit('-', 1)
                    rows.append({
                        'user_id': user_id,
                        'news_id': news_id,
                        'label': int(label)
                    })

        return pd.DataFrame(rows)

    def stop_spark(self):
        """Stop Spark session."""
        if self.spark:
            self.spark.stop()
