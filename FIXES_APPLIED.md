# Fixes Applied to MIND Recommendation System

## Issue 1: Entity/Relation Embedding Loading Error ✅ FIXED

**Error:**
```
ValueError: invalid literal for int() with base 10: 'Q41'
```

**Location:** `src/data_loader.py` lines 137-189

**Cause:** Code assumed `.vec` files had header line with count/dimension, but MIND dataset files start directly with data.

**Fix:** Modified `load_entity_embeddings()` and `load_relation_embeddings()` to:
- Remove header parsing
- Infer dimension from first vector
- Process all lines as data

**Status:** ✅ Fixed

---

## Issue 2: TF-IDF Recommender TypeError ✅ FIXED

**Error:**
```
TypeError: np.matrix is not supported. Please convert to a numpy array with np.asarray.
```

**Location:** `src/models/cbf_tfidf.py` lines 90 and 150

**Cause:** `scipy.sparse.matrix.mean(axis=0)` returns `np.matrix` object, which scikit-learn 1.2+ doesn't support.

**Fix:** Wrapped `.mean(axis=0)` with `np.asarray()`:
```python
# Before:
user_profile = self.tfidf_matrix[history_indices].mean(axis=0)

# After:
user_profile = np.asarray(self.tfidf_matrix[history_indices].mean(axis=0))
```

**Status:** ✅ Fixed

---

## Recommended: Enhanced Evaluation Cell

**Current Issue:** Basic evaluation cell lacks error handling for edge cases like:
- Users with empty history or candidates
- Recommendation failures
- Evaluation metric errors

**Solution:** Replace Cell 27 with the robust version from `EVALUATION_CELL_FIXED.py`:

```python
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
```

**Benefits:**
- Filters out users with empty data
- Handles individual user errors gracefully
- Reports error counts
- Validates sufficient data for evaluation
- Try-catch for evaluation failures

---

## How to Apply Fixes

### In Jupyter Notebook:

1. **Restart Kernel:**
   - Kernel → Restart Kernel

2. **Reimport Fixed Modules:**
   - Re-run Cell 2 (imports)

3. **Update Evaluation Cell (Cell 27):**
   - Copy code from `EVALUATION_CELL_FIXED.py`
   - Replace current Cell 27 code

4. **Run from Beginning:**
   - Run All Cells or run from Cell 16 onwards

### Expected Results:

All cells should now run without errors, producing:
- Successful data loading
- All 4 models trained (TF-IDF, Entity, Knowledge-Aware, Hybrid)
- Evaluation metrics for all models
- Saved models in `models/` directory
- Results in `results/` directory

---

## Summary of Changes

| File | Lines | Change |
|------|-------|--------|
| `src/data_loader.py` | 137-162 | Remove header parsing from entity embeddings |
| `src/data_loader.py` | 164-189 | Remove header parsing from relation embeddings |
| `src/models/cbf_tfidf.py` | 90 | Add `np.asarray()` wrapper |
| `src/models/cbf_tfidf.py` | 150 | Add `np.asarray()` wrapper |
| `main.ipynb` Cell 27 | All | Enhanced error handling (recommended) |

---

## Verification

To verify fixes are working:

```python
# Test 1: Entity embedding loading
from src.data_loader import MINDDataLoader
loader = MINDDataLoader(use_spark=False)
entity_emb, dim = loader.load_entity_embeddings('data/MINDsmall_train/entity_embedding.vec')
print(f"✅ Loaded {len(entity_emb)} entities, dim={dim}")

# Test 2: TF-IDF recommender
from scipy.sparse import csr_matrix
from models.cbf_tfidf import TFIDFRecommender
tfidf = csr_matrix(np.random.rand(10, 20))
rec = TFIDFRecommender(tfidf, [f'N{i}' for i in range(10)])
result = rec.recommend_for_user(['N0'], ['N1', 'N2'], top_k=2)
print(f"✅ TF-IDF working: {len(result)} recommendations")
```

Both tests should pass without errors.
