# ALS Evaluation Error Fix

## Problem

When evaluating the ALS model, you encountered the error:
```
AttributeError: 'NoneType' object has no attribute 'sc'
```

## Root Causes

### 1. Spark Session Stopped Prematurely
The Spark session was stopped in cell 42 using `spark.stop()`. When you later tried to re-run cell 28 to evaluate models (including ALS), the ALS recommender couldn't work because it needs an active Spark session to create DataFrames.

**Location**: Cell 42
**Issue**: `spark.stop()` was called before all evaluations were complete

### 2. Incorrect Tuple Unpacking
The ALS evaluation code had a bug where it tried to unpack 3 values from a 2-tuple:

```python
# ❌ Wrong - ALS returns (news_id, score) not 3 values
recs = recommender.recommend_for_user(user_id, top_k=10)
user_preds[user_id] = [(news_id, score) for news_id, score, _ in recs]
```

**Location**: Cell 28
**Issue**: ALS `recommend_for_user()` returns `(news_id, score)` tuples, not 3-element tuples

## Solutions Applied

### Fix 1: Updated Cell 28 (Evaluation Code)
Fixed the ALS evaluation to correctly handle 2-tuples and filter by candidates:

```python
elif model_name == 'ALS':
    # ALS returns (news_id, score) tuples - no candidate filtering
    recs = recommender.recommend_for_user(user_id, top_k=10)
    # Filter to only candidates to match other models
    candidate_set = set(candidates)
    user_preds[user_id] = [(news_id, score) for news_id, score in recs if news_id in candidate_set]
```

### Fix 2: Updated Cell 42 (Spark Cleanup)
Commented out `spark.stop()` to prevent premature session termination:

```python
# Stop Spark session when done
# IMPORTANT: Only run this cell when you're completely done with all evaluations
# Uncomment the line below to stop the Spark session
# spark.stop()
# print("✅ Spark session stopped")
```

### Fix 3: Added Spark Session Restart Cell
Added a new cell (before cell 42) to restart the Spark session if needed:

```python
# Run this cell if you get "NoneType has no attribute 'sc'" error with ALS
try:
    spark.sparkContext.getConf().getAll()
    print("✅ Spark session is active")
except:
    print("⚠️  Spark session was stopped. Restarting...")
    spark = SparkSession.builder \
        .appName("MIND_Recommendation_System") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .config("spark.sql.shuffle.partitions", "10") \
        .getOrCreate()

    als_recommender.spark = spark
    print(f"✅ Spark session restarted")
```

## How to Use

### If Starting Fresh
1. Run all cells from the beginning in order
2. Cell 28 will now correctly evaluate the ALS model
3. Only run cell 42 (spark.stop) when completely done

### If You Already Stopped Spark
1. Run the new "Restart Spark Session" cell
2. Then re-run cell 28 to evaluate models
3. ALS evaluation should now work

## Key Takeaways

1. **Don't stop Spark prematurely**: Keep the Spark session active until all ALS operations are complete
2. **Check return types**: ALS recommender returns 2-tuples `(news_id, score)`, not 3-tuples
3. **Candidate filtering**: ALS recommends from all items, so we filter to candidates for fair comparison
4. **Session recovery**: You can restart a stopped Spark session and reassign it to recommenders

## Testing

To verify the fix works:
```python
# After running the restart cell, test ALS recommendations
sample_user = list(user_data.keys())[0]
recs = als_recommender.recommend_for_user(sample_user, top_k=5)
print(f"ALS recommendations for {sample_user}:")
for news_id, score in recs:
    print(f"  {news_id}: {score:.4f}")
```
