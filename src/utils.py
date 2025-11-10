"""
Utility functions for the recommendation system.
"""

import time
import numpy as np
import pandas as pd
from functools import wraps
import matplotlib.pyplot as plt
import seaborn as sns


def timer(func):
    """
    Decorator to measure execution time of functions.

    Args:
        func: Function to time

    Returns:
        Wrapped function that prints execution time
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time

        print(f"[Timer] {func.__name__} executed in {execution_time:.4f} seconds")

        return result

    return wrapper


def benchmark_comparison(spark_func, pandas_func, *args, **kwargs):
    """
    Compare execution time between Spark and Pandas implementations.

    Args:
        spark_func: Spark implementation function
        pandas_func: Pandas implementation function
        *args, **kwargs: Arguments to pass to both functions

    Returns:
        Dictionary with timing results and speedup
    """
    # Time Spark implementation
    start_spark = time.time()
    spark_result = spark_func(*args, **kwargs)
    spark_time = time.time() - start_spark

    # Time Pandas implementation
    start_pandas = time.time()
    pandas_result = pandas_func(*args, **kwargs)
    pandas_time = time.time() - start_pandas

    speedup = pandas_time / spark_time if spark_time > 0 else 0

    results = {
        'spark_time': spark_time,
        'pandas_time': pandas_time,
        'speedup': speedup,
        'spark_result': spark_result,
        'pandas_result': pandas_result
    }

    print(f"\n{'='*60}")
    print(f"Benchmark Comparison")
    print(f"{'='*60}")
    print(f"Spark Time:   {spark_time:.4f} seconds")
    print(f"Pandas Time:  {pandas_time:.4f} seconds")
    print(f"Speedup:      {speedup:.2f}x")
    print(f"{'='*60}\n")

    return results


def plot_metrics_comparison(results_dict, save_path=None):
    """
    Plot comparison of metrics across different models.

    Args:
        results_dict: Dictionary mapping model names to metric dictionaries
        save_path: Optional path to save the plot

    Returns:
        matplotlib figure
    """
    # Convert to DataFrame
    df = pd.DataFrame(results_dict).T

    # Select key metrics to plot
    metrics_to_plot = [col for col in df.columns if not col.endswith('_std')]

    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Plot ranking metrics
    ranking_metrics = [m for m in metrics_to_plot if any(x in m for x in ['NDCG', 'MRR', 'Precision', 'Recall'])]
    if ranking_metrics:
        df[ranking_metrics].plot(kind='bar', ax=axes[0])
        axes[0].set_title('Ranking Metrics Comparison', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Score', fontsize=12)
        axes[0].set_xlabel('Model', fontsize=12)
        axes[0].legend(loc='best', fontsize=10)
        axes[0].grid(axis='y', alpha=0.3)
        axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha='right')

    # Plot classification metrics
    classification_metrics = [m for m in metrics_to_plot if any(x in m for x in ['AUC', 'Accuracy', 'F1'])]
    if classification_metrics:
        df[classification_metrics].plot(kind='bar', ax=axes[1])
        axes[1].set_title('Classification Metrics Comparison', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Score', fontsize=12)
        axes[1].set_xlabel('Model', fontsize=12)
        axes[1].legend(loc='best', fontsize=10)
        axes[1].grid(axis='y', alpha=0.3)
        axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_spark_speedup(benchmark_results, save_path=None):
    """
    Visualize Spark speedup over Pandas.

    Args:
        benchmark_results: Dictionary with benchmark results
        save_path: Optional path to save plot

    Returns:
        matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart comparing times
    operations = list(benchmark_results.keys())
    spark_times = [benchmark_results[op]['spark_time'] for op in operations]
    pandas_times = [benchmark_results[op]['pandas_time'] for op in operations]

    x = np.arange(len(operations))
    width = 0.35

    ax1.bar(x - width/2, spark_times, width, label='Spark', color='#2ecc71')
    ax1.bar(x + width/2, pandas_times, width, label='Pandas', color='#e74c3c')
    ax1.set_ylabel('Time (seconds)', fontsize=12)
    ax1.set_title('Execution Time Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(operations, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # Speedup chart
    speedups = [benchmark_results[op]['speedup'] for op in operations]
    ax2.bar(operations, speedups, color='#3498db')
    ax2.axhline(y=1, color='red', linestyle='--', label='No speedup')
    ax2.set_ylabel('Speedup (x)', fontsize=12)
    ax2.set_title('Spark Speedup over Pandas', fontsize=14, fontweight='bold')
    ax2.set_xticklabels(operations, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def get_sample_users(behaviors_df, n_users=5, min_history=3):
    """
    Get sample users for demo/testing.

    Args:
        behaviors_df: Behaviors DataFrame
        n_users: Number of users to sample
        min_history: Minimum history length

    Returns:
        List of user IDs
    """
    if isinstance(behaviors_df, pd.DataFrame):
        # Pandas
        user_history_counts = behaviors_df.groupby('user_id')['history'].apply(
            lambda x: len(x.iloc[0].split(' ')) if isinstance(x.iloc[0], str) else 0
        )
    else:
        # Spark - convert to pandas for sampling
        user_history_counts = behaviors_df.toPandas().groupby('user_id')['history'].apply(
            lambda x: len(x.iloc[0].split(' ')) if isinstance(x.iloc[0], str) else 0
        )

    # Filter users with sufficient history
    valid_users = user_history_counts[user_history_counts >= min_history].index.tolist()

    # Sample
    if len(valid_users) > n_users:
        sample_users = np.random.choice(valid_users, n_users, replace=False).tolist()
    else:
        sample_users = valid_users

    return sample_users


def format_news_display(news_df, news_id):
    """
    Format news article for display.

    Args:
        news_df: News DataFrame
        news_id: News ID to display

    Returns:
        Formatted string
    """
    news_row = news_df[news_df['news_id'] == news_id]

    if len(news_row) == 0:
        return f"News {news_id} not found"

    news_row = news_row.iloc[0]

    # Safely get text fields
    title = str(news_row.get('title', 'N/A')) if pd.notna(news_row.get('title')) else 'N/A'
    abstract = str(news_row.get('abstract', 'N/A')) if pd.notna(news_row.get('abstract')) else 'N/A'
    category = str(news_row.get('category', 'N/A')) if pd.notna(news_row.get('category')) else 'N/A'
    subcategory = str(news_row.get('subcategory', 'N/A')) if pd.notna(news_row.get('subcategory')) else 'N/A'

    display = f"""
News ID: {news_id}
Category: {category} > {subcategory}
Title: {title}
Abstract: {abstract[:200]}...
"""

    return display


def create_user_candidates_dict(behaviors_df, news_df):
    """
    Create dictionary mapping users to their history and candidate news.

    Args:
        behaviors_df: Behaviors DataFrame (pandas)
        news_df: News DataFrame

    Returns:
        Dictionary with user data
    """
    user_data = {}

    for _, row in behaviors_df.iterrows():
        user_id = row['user_id']

        # Parse history
        history = row.get('history_list', [])
        if isinstance(history, str):
            history = history.split(' ')
        elif history is None:
            history = []

        # Parse candidates from impressions
        impressions = row.get('impressions_list', [])
        if isinstance(impressions, str):
            impressions = impressions.split(' ')
        elif impressions is None:
            impressions = []

        candidates = []
        ground_truth = {}

        for imp in impressions:
            if '-' in imp:
                news_id, label = imp.rsplit('-', 1)
                candidates.append(news_id)
                ground_truth[news_id] = int(label)

        user_data[user_id] = {
            'history': history,
            'candidates': candidates,
            'ground_truth': ground_truth
        }

    return user_data


def save_recommendations(recommendations, output_path):
    """
    Save recommendations to file.

    Args:
        recommendations: Dictionary mapping user_id to recommendations
        output_path: Path to save file
    """
    import json

    with open(output_path, 'w') as f:
        # Convert to serializable format
        output = {}
        for user_id, recs in recommendations.items():
            output[user_id] = [
                {'news_id': news_id, 'score': float(score)}
                for news_id, score in recs
            ]

        json.dump(output, f, indent=2)

    print(f"Recommendations saved to {output_path}")
