# Quick Start Guide

## Prerequisites

- Python 3.8+
- Java 8+ (for PySpark)
- 4GB+ RAM recommended

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Verify Dataset

Check that the MIND dataset is present:
```bash
ls data/MINDsmall_train/
# Should show: behaviors.tsv, news.tsv, entity_embedding.vec, relation_embedding.vec

ls data/MINDsmall_dev/
# Should show: behaviors.tsv, news.tsv, entity_embedding.vec, relation_embedding.vec
```

## Running the Project

### Option 1: Jupyter Notebook (Recommended)

```bash
jupyter notebook main.ipynb
```

Then run all cells sequentially.

### Option 2: JupyterLab

```bash
jupyter lab main.ipynb
```

## What the Notebook Does

1. **Initialize Spark** - Sets up distributed computing environment
2. **Load Data** - Loads news and user behavior data with Spark optimization
3. **Extract Features** - Creates TF-IDF and entity embeddings
4. **Train Models** - Builds 5 recommendation models:
   - TF-IDF Content-Based
   - Entity Content-Based
   - ALS Collaborative Filtering
   - Knowledge-Aware
   - Hybrid (combines all)
5. **Evaluate** - Compares models using NDCG, MRR, AUC, etc.
6. **Interactive UI** - Displays recommendations interactively
7. **Save Models** - Persists trained models to disk

## Expected Runtime

- First run: ~10-15 minutes (including Spark initialization and model training)
- Subsequent runs: ~5-8 minutes (with cached data)

## Testing

Run unit tests:
```bash
cd tests
python test_models.py
```

## Output

After running, you'll find:
- `models/` - Saved trained models
- `results/` - Evaluation reports and visualizations
  - `evaluation_report.csv` - Metrics comparison
  - `metrics_comparison.png` - Visualization

## Common Issues

### Java Not Found
Install Java 8 or later:
```bash
# macOS
brew install openjdk@8

# Ubuntu/Debian
sudo apt-get install openjdk-8-jdk
```

### Memory Issues
If you encounter memory errors, reduce the dataset size or increase Spark memory:
```python
spark = SparkSession.builder \
    .config("spark.driver.memory", "8g") \
    .getOrCreate()
```

### Import Errors
Make sure you're running from the project root directory and src/ is in the path:
```python
import sys
sys.path.append('./src')
```

## Key Features to Explore

1. **Spark Optimization** - Compare Spark vs Pandas performance (cells 3-4)
2. **Model Comparison** - See how different approaches perform (cell 7)
3. **Interactive UI** - Explore recommendations for different users (cell 9)
4. **Parameter Testing** - Try different hybrid weights (cell 11)

## Next Steps

After running successfully:
1. Experiment with different ALS parameters (rank, iterations)
2. Adjust hybrid model weights
3. Try different feature extraction parameters
4. Evaluate on the dev set
5. Implement additional recommendation strategies

## Support

Check these files for more details:
- `README.md` - Full documentation
- `IMPLEMENTATION_PLAN.md` - Design and architecture
- `src/` - Source code with docstrings

## Troubleshooting

### Spark UI
Access Spark UI at: http://localhost:4040 (when Spark is running)

### Clear Cached Data
```python
# In notebook
spark.catalog.clearCache()
```

### Restart Spark
```python
# In notebook
spark.stop()
# Then re-run the Spark initialization cell
```
