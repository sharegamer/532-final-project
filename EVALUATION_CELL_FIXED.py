"""
Fixed evaluation cell with proper error handling.
Replace Cell 27 in the notebook with this code.
"""

print("\n🔬 Evaluating Models...\n")

evaluator = RecommendationEvaluator()
evaluation_results = {}

# Prepare data for batch evaluation
user_predictions = {}
user_ground_truth = {}

# Filter valid users (with history and candidates)
valid_eval_users = []
for user_id in eval_users:
    if (len(user_data[user_id]['history']) > 0 and
        len(user_data[user_id]['candidates']) > 0 and
        len(user_data[user_id]['ground_truth']) > 0):
        user_ground_truth[user_id] = user_data[user_id]['ground_truth']
        valid_eval_users.append(user_id)

print(f"Filtered to {len(valid_eval_users)} valid users (out of {len(eval_users)})")

# Evaluate each model
models_to_eval = {
    'TF-IDF': tfidf_recommender,
    'Entity': entity_recommender,
    'Knowledge-Aware': knowledge_recommender,
    'Hybrid': hybrid_recommender
}

for model_name, recommender in models_to_eval.items():
    print(f"\nEvaluating {model_name}...")

    user_preds = {}
    errors = 0

    for user_id in valid_eval_users:
        try:
            history = user_data[user_id]['history']
            candidates = user_data[user_id]['candidates']

            if model_name == 'Hybrid':
                recs = recommender.recommend_for_user(user_id, history, candidates, top_k=10)
                if len(recs) > 0:
                    user_preds[user_id] = [(news_id, score) for news_id, score, _ in recs]
            else:
                recs = recommender.recommend_for_user(history, candidates, top_k=10)
                if len(recs) > 0:
                    user_preds[user_id] = recs
        except Exception as e:
            errors += 1
            if errors <= 3:  # Only print first 3 errors
                print(f"  Warning: Error for user {user_id}: {str(e)[:80]}")
            continue

    print(f"  Generated predictions for {len(user_preds)}/{len(valid_eval_users)} users")
    if errors > 0:
        print(f"  Skipped {errors} users due to errors")

    # Evaluate only if we have sufficient predictions
    if len(user_preds) >= 10:  # Need at least 10 users for meaningful metrics
        # Filter ground truth to match predictions
        filtered_gt = {uid: user_ground_truth[uid] for uid in user_preds.keys()}

        try:
            metrics = evaluator.evaluate_batch(user_preds, filtered_gt, k_values=[5, 10])
            evaluation_results[model_name] = metrics
            evaluator.print_evaluation_summary(metrics, model_name)
        except Exception as e:
            print(f"  ⚠️ Evaluation failed: {str(e)[:100]}")
    else:
        print(f"  ⚠️ Insufficient predictions ({len(user_preds)}) for {model_name}, skipping evaluation")

print("\n✅ Evaluation complete!")
print(f"\nSuccessfully evaluated {len(evaluation_results)} models")
