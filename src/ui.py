"""
UI components for displaying recommendations and evaluation results.
Uses ipywidgets for interactive Jupyter notebook interface.
"""

import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
import pandas as pd
import matplotlib.pyplot as plt


class RecommendationUI:
    """Interactive UI for recommendation system."""

    def __init__(self, news_df, user_data, recommenders_dict):
        """
        Initialize UI.

        Args:
            news_df: DataFrame with news articles
            user_data: Dictionary with user histories and candidates
            recommenders_dict: Dictionary mapping model names to recommender objects
        """
        self.news_df = news_df
        self.user_data = user_data
        self.recommenders = recommenders_dict
        self.current_recommendations = {}

    def format_news_card(self, news_id, score=None, rank=None):
        """
        Format news article as HTML card.

        Args:
            news_id: News ID
            score: Recommendation score (optional)
            rank: Rank position (optional)

        Returns:
            HTML string
        """
        news_row = self.news_df[self.news_df['news_id'] == news_id]

        if len(news_row) == 0:
            return f"<div class='news-card'>News {news_id} not found</div>"

        news_row = news_row.iloc[0]

        rank_badge = f"<span class='rank-badge'>#{rank}</span>" if rank else ""
        score_badge = f"<span class='score-badge'>Score: {score:.3f}</span>" if score else ""

        # Safely get text fields
        title = str(news_row.get('title', 'N/A')) if pd.notna(news_row.get('title')) else 'N/A'
        abstract = str(news_row.get('abstract', 'N/A')) if pd.notna(news_row.get('abstract')) else 'N/A'
        category = str(news_row.get('category', 'N/A')) if pd.notna(news_row.get('category')) else 'N/A'
        subcategory = str(news_row.get('subcategory', 'N/A')) if pd.notna(news_row.get('subcategory')) else 'N/A'

        html = f"""
        <div style='border: 1px solid #ddd; padding: 15px; margin: 10px 0; border-radius: 8px; background-color: #f9f9f9;'>
            <div style='margin-bottom: 5px;'>
                {rank_badge} {score_badge}
                <span style='background-color: #3498db; color: white; padding: 3px 8px; border-radius: 4px; font-size: 12px; margin-left: 10px;'>
                    {category}
                </span>
            </div>
            <h4 style='margin: 10px 0; color: #2c3e50;'>{title}</h4>
            <p style='color: #7f8c8d; font-size: 14px; margin: 5px 0;'>{abstract[:200]}...</p>
            <div style='font-size: 12px; color: #95a5a6; margin-top: 10px;'>
                <strong>ID:</strong> {news_id} | <strong>Subcategory:</strong> {subcategory}
            </div>
        </div>
        """

        return html

    def display_user_history(self, user_id, max_items=5):
        """
        Display user's reading history.

        Args:
            user_id: User ID
            max_items: Maximum items to show
        """
        if user_id not in self.user_data:
            print(f"User {user_id} not found")
            return

        history = self.user_data[user_id]['history']

        html = f"""
        <div style='background-color: #ecf0f1; padding: 15px; border-radius: 8px; margin: 10px 0;'>
            <h3 style='color: #2c3e50; margin-top: 0;'>Reading History for User: {user_id}</h3>
            <p style='color: #7f8c8d;'>Total articles read: {len(history)}</p>
        </div>
        """

        display(HTML(html))

        # Display history items
        for i, news_id in enumerate(history[:max_items]):
            display(HTML(self.format_news_card(news_id)))

        if len(history) > max_items:
            display(HTML(f"<p style='color: #7f8c8d; font-style: italic;'>... and {len(history) - max_items} more articles</p>"))

    def display_recommendations(self, user_id, model_name='Hybrid', top_k=10):
        """
        Display recommendations for a user.

        Args:
            user_id: User ID
            model_name: Name of the recommender model to use
            top_k: Number of recommendations to show
        """
        if user_id not in self.user_data:
            print(f"User {user_id} not found")
            return

        if model_name not in self.recommenders:
            print(f"Model {model_name} not found")
            return

        recommender = self.recommenders[model_name]
        user_info = self.user_data[user_id]

        # Generate recommendations
        if model_name == 'Hybrid':
            recommendations = recommender.recommend_for_user(
                user_id,
                user_info['history'],
                user_info['candidates'],
                top_k=top_k
            )
            # Extract simple (news_id, score) for display
            recs = [(news_id, score) for news_id, score, _ in recommendations]
        elif model_name == 'ALS':
            recs = recommender.recommend_for_user(user_id, top_k=top_k)
        else:
            recs = recommender.recommend_for_user(
                user_info['history'],
                user_info['candidates'],
                top_k=top_k
            )

        # Display header
        html = f"""
        <div style='background-color: #2ecc71; padding: 15px; border-radius: 8px; margin: 20px 0;'>
            <h2 style='color: white; margin: 0;'>Top {top_k} Recommendations - {model_name} Model</h2>
            <p style='color: white; margin: 5px 0;'>User: {user_id}</p>
        </div>
        """
        display(HTML(html))

        # Display recommendations
        for rank, (news_id, score) in enumerate(recs[:top_k], 1):
            display(HTML(self.format_news_card(news_id, score=score, rank=rank)))

        self.current_recommendations[user_id] = recs

    def display_model_comparison(self, user_id, top_k=5):
        """
        Display recommendations from all models side-by-side.

        Args:
            user_id: User ID
            top_k: Number of recommendations per model
        """
        if user_id not in self.user_data:
            print(f"User {user_id} not found")
            return

        user_info = self.user_data[user_id]

        # Get recommendations from all models
        all_recs = {}

        for model_name, recommender in self.recommenders.items():
            try:
                if model_name == 'Hybrid':
                    recs = recommender.recommend_for_user(
                        user_id,
                        user_info['history'],
                        user_info['candidates'],
                        top_k=top_k
                    )
                    all_recs[model_name] = [(news_id, score) for news_id, score, _ in recs]
                elif model_name == 'ALS':
                    recs = recommender.recommend_for_user(user_id, top_k=top_k)
                    all_recs[model_name] = recs
                else:
                    recs = recommender.recommend_for_user(
                        user_info['history'],
                        user_info['candidates'],
                        top_k=top_k
                    )
                    all_recs[model_name] = recs
            except Exception as e:
                all_recs[model_name] = []
                print(f"Error getting recommendations from {model_name}: {e}")

        # Display comparison
        html = f"""
        <div style='background-color: #3498db; padding: 15px; border-radius: 8px; margin: 20px 0;'>
            <h2 style='color: white; margin: 0;'>Model Comparison - Top {top_k} Recommendations</h2>
            <p style='color: white; margin: 5px 0;'>User: {user_id}</p>
        </div>
        """
        display(HTML(html))

        # Create comparison table
        comparison_data = []
        for i in range(top_k):
            row = {'Rank': i + 1}
            for model_name, recs in all_recs.items():
                if i < len(recs):
                    news_id, score = recs[i]
                    news_title = self.news_df[self.news_df['news_id'] == news_id]['title'].iloc[0] if len(self.news_df[self.news_df['news_id'] == news_id]) > 0 else 'N/A'
                    row[model_name] = f"{news_title[:40]}... ({score:.3f})"
                else:
                    row[model_name] = '-'
            comparison_data.append(row)

        df = pd.DataFrame(comparison_data)
        display(df)

    def create_interactive_widget(self):
        """
        Create interactive widget for exploring recommendations.

        Returns:
            Widget container
        """
        # User selection dropdown
        user_ids = list(self.user_data.keys())[:50]  # Limit to first 50 for performance
        user_dropdown = widgets.Dropdown(
            options=user_ids,
            description='User:',
            style={'description_width': 'initial'}
        )

        # Model selection dropdown
        model_dropdown = widgets.Dropdown(
            options=list(self.recommenders.keys()),
            description='Model:',
            style={'description_width': 'initial'}
        )

        # Top-K slider
        top_k_slider = widgets.IntSlider(
            value=5,
            min=1,
            max=20,
            step=1,
            description='Top-K:',
            style={'description_width': 'initial'}
        )

        # Show history checkbox
        show_history_checkbox = widgets.Checkbox(
            value=True,
            description='Show Reading History',
            style={'description_width': 'initial'}
        )

        # Output area
        output = widgets.Output()

        # Update function
        def update_display(change):
            with output:
                clear_output(wait=True)

                if show_history_checkbox.value:
                    self.display_user_history(user_dropdown.value, max_items=3)

                self.display_recommendations(
                    user_dropdown.value,
                    model_dropdown.value,
                    top_k_slider.value
                )

        # Attach observers
        user_dropdown.observe(update_display, names='value')
        model_dropdown.observe(update_display, names='value')
        top_k_slider.observe(update_display, names='value')
        show_history_checkbox.observe(update_display, names='value')

        # Initial display
        update_display(None)

        # Create layout
        controls = widgets.VBox([
            widgets.HBox([user_dropdown, model_dropdown]),
            widgets.HBox([top_k_slider, show_history_checkbox])
        ])

        widget = widgets.VBox([controls, output])

        return widget

    def display_metrics_dashboard(self, evaluation_results):
        """
        Display evaluation metrics dashboard.

        Args:
            evaluation_results: Dictionary with evaluation metrics
        """
        html = """
        <style>
            .metric-card {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                border-radius: 10px;
                margin: 10px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                text-align: center;
                min-width: 200px;
            }
            .metric-value {
                font-size: 32px;
                font-weight: bold;
                margin: 10px 0;
            }
            .metric-name {
                font-size: 14px;
                opacity: 0.9;
            }
        </style>
        """

        display(HTML(html))

        # Display metrics in cards
        for model_name, metrics in evaluation_results.items():
            html = f"<h3 style='color: #2c3e50; margin-top: 30px;'>{model_name}</h3>"
            html += "<div style='display: flex; flex-wrap: wrap; justify-content: flex-start;'>"

            for metric_name, value in metrics.items():
                if not metric_name.endswith('_std'):
                    html += f"""
                    <div class='metric-card'>
                        <div class='metric-name'>{metric_name}</div>
                        <div class='metric-value'>{value:.4f}</div>
                    </div>
                    """

            html += "</div>"
            display(HTML(html))
