# Spark Cloudpickle Serialization Error Fix

## Error

```
IndexError: tuple index out of range
File "/opt/anaconda3/envs/mindrec/lib/python3.11/site-packages/pyspark/cloudpickle/cloudpickle.py", line 334
```

## Cause

This is a known compatibility issue between:
- Python 3.11
- PySpark 3.3.x
- cloudpickle library

The error occurs when Spark tries to serialize Python functions for distributed execution, particularly when saving ALS models.

## Solutions

### Solution 1: Downgrade cloudpickle (Recommended)

```bash
pip install 'cloudpickle<3.0.0'
# or specifically
pip install cloudpickle==2.2.1
```

### Solution 2: Skip ALS Model Saving

Replace Cell 30 (model saving) with this version that skips ALS saving:

```python
print("\n💾 Saving Models...")

# Create models directory
os.makedirs('models', exist_ok=True)

# Save each model
tfidf_recommender.save_model('models')
print("✅ TF-IDF model saved")

entity_recommender.save_model('models')
print("✅ Entity model saved")

# Skip ALS model saving due to Spark serialization issue
print("⚠️  ALS model saving skipped (Spark serialization issue)")
print("   Note: ALS model is still in memory and usable")

knowledge_recommender.save_model('models')
print("✅ Knowledge-Aware model saved")

hybrid_recommender.save_model('models')
print("✅ Hybrid model saved")

print("\n✅ Models saved to ./models/ (except ALS)")
```

### Solution 3: Use Alternative ALS Saving Method

Modify `src/models/cf_als.py` to save only the model parameters:

```python
def save_model(self, path):
    """Save ALS model parameters only."""
    import pickle
    import os

    os.makedirs(path, exist_ok=True)

    # Save mappings and model info (skip actual Spark model)
    with open(f'{path}/als_mappings.pkl', 'wb') as f:
        pickle.dump({
            'user_mapping': self.user_mapping,
            'news_mapping': self.news_mapping,
            'reverse_user_mapping': self.reverse_user_mapping,
            'reverse_news_mapping': self.reverse_news_mapping,
            'rank': self.model.rank if self.model else None,
            'note': 'Spark ALS model not saved due to serialization issues'
        }, f)

    print("  Note: ALS parameters saved, model weights not persisted")
```

### Solution 4: Upgrade PySpark (if possible)

```bash
pip install pyspark>=3.5.0
```

Note: This may require other dependency updates.

## Recommended Quick Fix

For immediate resolution, use **Solution 2** (skip ALS saving). The ALS model remains in memory and fully functional for the current session - you just won't be able to reload it in a future session.

## Why This Happens

The error occurs because:
1. Python 3.11 changed bytecode format
2. PySpark 3.3.x's cloudpickle hasn't fully adapted
3. Complex Spark ML objects are difficult to serialize

## Workaround in Your Notebook

1. **Option A: Quick Fix** - Replace Cell 30 with the code from Solution 2

2. **Option B: Fix Environment** - Run in terminal:
   ```bash
   pip install 'cloudpickle==2.2.1'
   ```
   Then restart your Jupyter kernel and re-run from Cell 20 (ALS training) onwards

## Status

This is a known PySpark issue, not a bug in your code. The models work fine at runtime; only persistence is affected.
