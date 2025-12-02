"""
Preprocessing module for feature extraction and text processing.
Handles TF-IDF computation and entity extraction.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from pyspark.ml.feature import Tokenizer, HashingTF, IDF, StopWordsRemover
from pyspark.sql.functions import concat_ws, col, udf, lower, regexp_replace
from pyspark.sql.types import StringType
from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd
import json
import re


class FeatureExtractor:
    """Extracts features from news articles for recommendation."""

    def __init__(self, use_spark=True):
        """
        Initialize feature extractor.

        Args:
            use_spark: Whether to use Spark for TF-IDF computation
        """
        self.use_spark = use_spark
        self.tfidf_vectorizer = None
        self.tfidf_model = None

    def extract_tfidf_sklearn(self, news_df, max_features=5000):
        """
        Extract TF-IDF features using scikit-learn.

        Args:
            news_df: Pandas DataFrame with news data
            max_features: Maximum number of features

        Returns:
            TF-IDF matrix, feature names, and fitted vectorizer
        """
        # Combine title and abstract
        news_df['text'] = news_df['title'].fillna('') + ' ' + news_df['abstract'].fillna('')

        # Create TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2
        )

        # Fit and transform
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(news_df['text'])

        return tfidf_matrix, self.tfidf_vectorizer.get_feature_names_out(), self.tfidf_vectorizer

    def extract_tfidf_spark(self, news_spark_df, num_features=5000):
        """
        Extract TF-IDF features using Spark MLlib.

        Args:
            news_spark_df: Spark DataFrame with news data
            num_features: Number of hash features

        Returns:
            tuple: (tfidf_matrix as scipy sparse matrix, news_ids array, tfidf_model)
        """
        # Combine title and abstract
        news_df_text = news_spark_df.withColumn(
            'text',
            concat_ws(' ',
                     col('title').cast('string'),
                     col('abstract').cast('string'))
        )

        # Clean text: lowercase and remove URLs/special chars
        news_df_text = news_df_text.withColumn('text', lower(col('text')))
        news_df_text = news_df_text.withColumn('text', regexp_replace(col('text'), r'http\S+', ''))
        news_df_text = news_df_text.withColumn('text', regexp_replace(col('text'), r'[^a-z0-9\s]', ' '))

        # Tokenize
        tokenizer = Tokenizer(inputCol='text', outputCol='words_raw')
        words_data = tokenizer.transform(news_df_text)

        # Remove stop words
        remover = StopWordsRemover(inputCol='words_raw', outputCol='words')
        words_data = remover.transform(words_data)

        # HashingTF
        hashing_tf = HashingTF(inputCol='words', outputCol='raw_features', numFeatures=num_features)
        featurized_data = hashing_tf.transform(words_data)

        # IDF
        idf = IDF(inputCol='raw_features', outputCol='tfidf_features', minDocFreq=2)
        self.tfidf_model = idf.fit(featurized_data)
        tfidf_data = self.tfidf_model.transform(featurized_data)

        # Select news_id and tfidf_features and order by news_id for consistency
        tfidf_result = tfidf_data.select('news_id', 'tfidf_features').orderBy('news_id')

        # Convert to scipy sparse matrix
        tfidf_matrix, news_ids = self._spark_vectors_to_scipy_sparse(tfidf_result, num_features)

        return tfidf_matrix, news_ids, self.tfidf_model

    def _spark_vectors_to_scipy_sparse(self, spark_df, num_features):
        """
        Convert Spark DataFrame with SparseVector column to scipy sparse matrix.

        Args:
            spark_df: Spark DataFrame with 'news_id' and 'tfidf_features' columns
            num_features: Number of features (dimension)

        Returns:
            tuple: (scipy csr_matrix, numpy array of news_ids)
        """
        # Collect data from Spark to driver
        rows_data = spark_df.collect()

        # Extract news IDs
        news_ids = np.array([row['news_id'] for row in rows_data])

        # Build scipy sparse matrix
        data_list = []
        row_indices = []
        col_indices = []

        for row_idx, row in enumerate(rows_data):
            vector = row['tfidf_features']
            # SparseVector has indices and values attributes
            for idx, value in zip(vector.indices, vector.values):
                row_indices.append(row_idx)
                col_indices.append(int(idx))
                data_list.append(float(value))

        # Create CSR matrix
        tfidf_matrix = csr_matrix(
            (data_list, (row_indices, col_indices)),
            shape=(len(news_ids), num_features)
        )

        return tfidf_matrix, news_ids

    def parse_entities(self, entity_str):
        """
        Parse entity string from TSV.

        Args:
            entity_str: JSON string containing entity information

        Returns:
            List of entity WikidataIds
        """
        if pd.isna(entity_str) or entity_str == '[]':
            return []

        try:
            entities = json.loads(entity_str)
            return [e.get('WikidataId', '') for e in entities if 'WikidataId' in e]
        except:
            return []

    def extract_entity_features(self, news_df, entity_embeddings):
        """
        Extract entity-based features for news articles.

        Args:
            news_df: Pandas DataFrame with news data
            entity_embeddings: Dictionary of entity embeddings

        Returns:
            DataFrame with entity feature vectors
        """
        # Parse entities from both title and abstract
        news_df['title_entities_list'] = news_df['title_entities'].apply(self.parse_entities)
        news_df['abstract_entities_list'] = news_df['abstract_entities'].apply(self.parse_entities)

        # Combine all entities
        news_df['all_entities'] = news_df.apply(
            lambda x: list(set(x['title_entities_list'] + x['abstract_entities_list'])),
            axis=1
        )

        # Get embedding dimension
        embedding_dim = len(next(iter(entity_embeddings.values())))

        # Aggregate entity embeddings for each news article
        entity_features = []
        for entities in news_df['all_entities']:
            if len(entities) > 0:
                # Average embeddings of all entities
                embeddings = [entity_embeddings[e] for e in entities if e in entity_embeddings]
                if len(embeddings) > 0:
                    avg_embedding = np.mean(embeddings, axis=0)
                else:
                    avg_embedding = np.zeros(embedding_dim)
            else:
                avg_embedding = np.zeros(embedding_dim)

            entity_features.append(avg_embedding)

        news_df['entity_features'] = list(entity_features)

        return news_df

    def extract_category_features(self, news_df):
        """
        Extract category-based features.

        Args:
            news_df: Pandas DataFrame with news data

        Returns:
            One-hot encoded category features
        """
        from sklearn.preprocessing import LabelEncoder, OneHotEncoder

        # Encode categories
        le_category = LabelEncoder()
        le_subcategory = LabelEncoder()

        news_df['category_encoded'] = le_category.fit_transform(news_df['category'].fillna('unknown'))
        news_df['subcategory_encoded'] = le_subcategory.fit_transform(news_df['subcategory'].fillna('unknown'))

        return news_df, le_category, le_subcategory

    def build_user_profiles(self, behaviors_df, news_features_df):
        """
        Build user profiles from their reading history.

        Args:
            behaviors_df: DataFrame with user behaviors
            news_features_df: DataFrame with news features

        Returns:
            DataFrame with user profile vectors
        """
        user_profiles = []

        for user_id, group in behaviors_df.groupby('user_id'):
            # Get clicked news
            clicked_news = group[group['label'] == 1]['news_id'].tolist()

            if len(clicked_news) > 0:
                # Average features of clicked news
                clicked_features = news_features_df[
                    news_features_df['news_id'].isin(clicked_news)
                ]

                if len(clicked_features) > 0:
                    # Compute average profile
                    # This would depend on the feature type (TF-IDF, entity, etc.)
                    user_profiles.append({
                        'user_id': user_id,
                        'clicked_news': clicked_news,
                        'num_clicks': len(clicked_news)
                    })

        return pd.DataFrame(user_profiles)


class TextCleaner:
    """Utility class for text cleaning and preprocessing."""

    @staticmethod
    def clean_text(text):
        """
        Clean text data.

        Args:
            text: Input text string

        Returns:
            Cleaned text
        """
        if pd.isna(text):
            return ''

        # Remove URLs
        text = re.sub(r'http\S+', '', text)

        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^a-zA-Z0-9\s\.,!?]', '', text)

        # Remove extra whitespace
        text = ' '.join(text.split())

        return text.lower()

    @staticmethod
    def clean_dataframe(df, text_columns):
        """
        Clean text columns in a dataframe.

        Args:
            df: Input DataFrame
            text_columns: List of column names to clean

        Returns:
            DataFrame with cleaned text
        """
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].apply(TextCleaner.clean_text)

        return df
