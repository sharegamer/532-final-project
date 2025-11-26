# Evaluation Cell Fix

## Common Issues and Solutions

### Issue 1: Empty Recommendations
Some users might not have valid recommendations from all models, causing evaluation errors.

### Issue 2: ALS User Mapping
ALS might not have all users in its training set, causing lookup errors.

### Issue 3: Empty Candidate Lists
Some users might have no candidates or no history.

## Fixed Evaluation Cell

Replace the evaluation cell with this more robust version:

```python
print("\n🔬 Evaluating Models...\n")

evaluator = RecommendationEvaluator()
evaluation_results = {}

# Prepare data for batch evaluation
user_predictions = {}
user_ground_truth = {}

# Filter users with valid data
valid_eval_users = []
for user_id in eval_users:
    if len(user_data[user_id]['candidates']) > 0 and len(user_data[user_id]['history']) > 0:
        user_ground_truth[user_id] = user_data[user_id]['ground_truth']
        valid_eval_users.append(user_id)

print(f"Evaluating on {len(valid_eval_users)} users with valid data")

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
    success_count = 0

    for user_id in valid_eval_users:
        try:
            history = user_data[user_id]['history']
            candidates = user_data[user_id]['candidates']

            if model_name == 'Hybrid':
                recs = recommender.recommend_for_user(user_id, history, candidates, top_k=10)
                if len(recs) > 0:
                    user_preds[user_id] = [(news_id, score) for news_id, score, _ in recs]
                    success_count += 1
            else:
                recs = recommender.recommend_for_user(history, candidates, top_k=10)
                if len(recs) > 0:
                    user_preds[user_id] = recs
                    success_count += 1
        except Exception as e:
            # Skip users that cause errors
            print(f"  Warning: Skipped user {user_id}: {str(e)[:50]}")
            continue

    print(f"  Successfully generated recommendations for {success_count}/{len(valid_eval_users)} users")

    # Evaluate only if we have predictions
    if len(user_preds) > 0:
        # Filter ground truth to only users with predictions
        filtered_gt = {uid: user_ground_truth[uid] for uid in user_preds.keys()}

        metrics = evaluator.evaluate_batch(user_preds, filtered_gt, k_values=[5, 10])
        evaluation_results[model_name] = metrics

        evaluator.print_evaluation_summary(metrics, model_name)
    else:
        print(f"  ⚠️ No valid predictions for {model_name}")

print("\n✅ Evaluation complete!")
```

## Alternative: Evaluate Without Hybrid First

If the Hybrid model is causing issues, evaluate other models first:

```python
# Evaluate non-hybrid models first
models_to_eval = {
    'TF-IDF': tfidf_recommender,
    'Entity': entity_recommender,
    'Knowledge-Aware': knowledge_recommender
}

# Then add Hybrid separately if needed
```

## Debug: Check Sample Recommendation

Before full evaluation, test with a single user:

```python
# Test with one user
test_user = valid_eval_users[0]
test_history = user_data[test_user]['history']
test_candidates = user_data[test_user]['candidates']

print(f"Test user: {test_user}")
print(f"History length: {len(test_history)}")
print(f"Candidates: {len(test_candidates)}")

# Test each model
for model_name, recommender in models_to_eval.items():
    print(f"\n{model_name}:")
    try:
        if model_name == 'Hybrid':
            recs = recommender.recommend_for_user(test_user, test_history, test_candidates, top_k=5)
            print(f"  Recommendations: {len(recs)}")
            if len(recs) > 0:
                print(f"  Sample: {recs[0]}")
        else:
            recs = recommender.recommend_for_user(test_history, test_candidates, top_k=5)
            print(f"  Recommendations: {len(recs)}")
            if len(recs) > 0:
                print(f"  Sample: {recs[0]}")
    except Exception as e:
        print(f"  Error: {e}")
```
